from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
from sqlalchemy import text
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_TABLE = "match_features_v3_elo"
MODEL_NAME = "production_logistic_elo_v3"
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
RANDOM_STATE = 42
DOMINANT_CLASS_MAX_PROB = 0.50
THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402
from tier3_freeze_audit import BASE_FEATURE_COLUMNS, ELO_FEATURE_COLUMNS  # noqa: E402


EXPECTED_TRAINING_ROWS = 1900
EXPECTED_FEATURE_COUNT = 32
PRODUCTION_SEASON_NOTE = (
    "2025-26 is included because production training happens after final holdout "
    "reporting."
)
PRODUCTION_USE_NOTE = "This artifact is for 2026-27 production predictions."

ARTIFACT_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_ARTIFACT = ARTIFACT_DIR / "production_logistic_elo_v3.pkl"
FEATURE_ARTIFACT = ARTIFACT_DIR / "production_features_v3.json"
DRAW_THRESHOLD_ARTIFACT = ARTIFACT_DIR / "production_draw_threshold_v3.json"
METADATA_ARTIFACT = ARTIFACT_DIR / "production_metadata_v3.json"

WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "match_features_v3_h2h_experiment",
    "match_features_v3_style_experiment",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "elo_current_v3",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]
ALLOWED_COUNT_CHANGES = {"elo_current_v3"}

FORBIDDEN_EXACT_FEATURES = {
    "match_id",
    "season",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    TARGET_COLUMN,
    "home_goals",
    "away_goals",
    "home_win",
    "is_draw",
    "away_win",
    "final_score",
    "created_at",
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
FORBIDDEN_FEATURE_TOKENS = [
    "h2h",
    "style",
    "pressure",
    "poisson",
    "odds",
    "betting",
    "manager",
    "sentiment",
    "injury",
    "rivalry",
    "derby",
    "calibration",
]
DOCUMENTED_FINAL_HOLDOUT = {
    "accuracy": 0.4868,
    "log_loss": 1.0601,
    "brier_score": 0.6372,
}
SANITY_CHECK_TOLERANCE = 0.001


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_training_data(conn) -> pandas.DataFrame:
    query = text(
        f"""
        SELECT *
        FROM {SOURCE_TABLE}
        WHERE season_id = ANY(:train_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pandas.read_sql(query, conn, params={"train_seasons": TRAIN_SEASONS})
    df["match_date"] = pandas.to_datetime(df["match_date"])
    return df


def get_feature_columns(df) -> list[str]:
    feature_columns = [*BASE_FEATURE_COLUMNS, *ELO_FEATURE_COLUMNS]
    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing frozen production feature column(s): {missing}")
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_columns)}"
        )
    validate_no_forbidden_features(feature_columns)
    return feature_columns


def validate_training_data(df, feature_columns) -> None:
    errors: list[str] = []
    if len(df) != EXPECTED_TRAINING_ROWS:
        errors.append(f"expected {EXPECTED_TRAINING_ROWS} rows, found {len(df)}")

    seasons = sorted(df["season_id"].dropna().unique().tolist())
    if seasons != TRAIN_SEASONS:
        errors.append(f"training seasons {seasons} != {TRAIN_SEASONS}")

    season_counts = df.groupby("season_id").size().to_dict()
    bad_season_counts = {
        season: int(count)
        for season, count in season_counts.items()
        if int(count) != 380
    }
    if bad_season_counts:
        errors.append(f"unexpected season row counts: {bad_season_counts}")

    duplicate_match_ids = int(df["match_id"].duplicated().sum())
    if duplicate_match_ids:
        errors.append(f"duplicate match_id count: {duplicate_match_ids}")

    required_columns = {
        "match_id",
        "season_id",
        "match_date",
        "home_team",
        "away_team",
        TARGET_COLUMN,
        *feature_columns,
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        errors.append(f"missing required column(s): {missing_columns}")

    null_targets = int(df[TARGET_COLUMN].isna().sum()) if TARGET_COLUMN in df else -1
    if null_targets:
        errors.append(f"null target count: {null_targets}")

    unknown_labels = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
    if unknown_labels:
        errors.append(f"unknown target label(s): {unknown_labels}")

    null_feature_counts = {
        column: int(df[column].isna().sum())
        for column in feature_columns
        if int(df[column].isna().sum()) > 0
    }
    if null_feature_counts:
        print(f"Feature nulls will be median-imputed: {null_feature_counts}")

    non_numeric_features = [
        column
        for column in feature_columns
        if not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        errors.append(f"non-numeric feature(s): {non_numeric_features}")

    if errors:
        raise ValueError("Production training data validation failed: " + "; ".join(errors))

    print("PASS: production training data validated.")
    print(f"Training seasons: {', '.join(TRAIN_SEASONS)}")
    print(f"Training rows: {len(df)}")
    print(f"Feature count: {len(feature_columns)}")


def validate_no_forbidden_features(feature_columns) -> None:
    feature_set = set(feature_columns)
    forbidden_exact = sorted(feature_set & FORBIDDEN_EXACT_FEATURES)
    forbidden_tokens = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    duplicate_features = sorted(
        column for column in feature_set if feature_columns.count(column) > 1
    )
    errors: list[str] = []
    if forbidden_exact:
        errors.append(f"forbidden exact feature(s): {forbidden_exact}")
    if forbidden_tokens:
        errors.append(f"forbidden token feature(s): {forbidden_tokens}")
    if duplicate_features:
        errors.append(f"duplicate feature(s): {duplicate_features}")
    if errors:
        raise ValueError("Forbidden production feature validation failed: " + "; ".join(errors))
    print("PASS: production feature exclusions validated.")


def build_production_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def select_draw_threshold_from_training(model, X_train, y_train, labels) -> dict:
    probabilities = _predict_probabilities(model, X_train, labels)
    argmax_predictions = _predict_argmax(probabilities, labels)
    argmax_metrics = _evaluate_predictions(
        y_train,
        argmax_predictions,
        probabilities,
        labels,
    )

    candidates: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    for draw_threshold in THRESHOLD_GRID:
        overlay_predictions = _apply_draw_overlay(probabilities, labels, draw_threshold)
        metrics = _evaluate_predictions(y_train, overlay_predictions, probabilities, labels)
        labels_changed_to_draw = _count_labels_changed_to_draw(
            argmax_predictions,
            overlay_predictions,
        )
        candidate = {
            "draw_threshold": draw_threshold,
            "training_accuracy": metrics["accuracy"],
            "training_draw_recall": metrics["draw_recall"],
            "training_draw_precision": metrics["draw_precision"],
            "training_draw_f1": metrics["draw_f1"],
            "labels_changed_to_draw": labels_changed_to_draw,
            "passed_accuracy_gate": metrics["accuracy"] >= argmax_metrics["accuracy"] - 0.01,
        }
        candidates.append(candidate)
        if not candidate["passed_accuracy_gate"]:
            continue

        selection_key = (
            metrics["draw_f1"],
            metrics["accuracy"],
            -labels_changed_to_draw,
            draw_threshold,
        )
        if best_config is None or selection_key > best_config["selection_key"]:
            best_config = {
                "selected_draw_threshold": draw_threshold,
                "selection_key": selection_key,
                "training_argmax_accuracy": argmax_metrics["accuracy"],
                "training_argmax_draw_recall": argmax_metrics["draw_recall"],
                "training_argmax_draw_precision": argmax_metrics["draw_precision"],
                "training_argmax_draw_f1": argmax_metrics["draw_f1"],
                "training_overlay_accuracy": metrics["accuracy"],
                "training_overlay_draw_recall": metrics["draw_recall"],
                "training_overlay_draw_precision": metrics["draw_precision"],
                "training_overlay_draw_f1": metrics["draw_f1"],
                "training_labels_changed_to_draw": labels_changed_to_draw,
                "threshold_candidates": candidates,
            }

    if best_config is None:
        raise RuntimeError("No draw threshold candidate survived training accuracy gate")

    best_config.pop("selection_key", None)
    best_config["rule"] = {
        "start_from": "argmax_prediction",
        "change_to_draw_if": [
            "P(D) >= selected_draw_threshold",
            "draw is second-highest class",
            f"dominant class probability < {DOMINANT_CLASS_MAX_PROB:.2f}",
        ],
        "probabilities_changed": False,
        "selected_from": "1900-row production training set only",
        "not_a_validation_result": True,
    }
    return best_config


def run_preseason_sanity_check(df, feature_columns) -> dict:
    train_df = df.loc[df["season_id"].isin(TRAIN_SEASONS[:-1])].copy()
    holdout_df = df.loc[df["season_id"] == TRAIN_SEASONS[-1]].copy()
    if len(train_df) != 1520 or len(holdout_df) != 380:
        raise ValueError(
            "Preseason sanity check expected 1520/380 rows, found "
            f"{len(train_df)}/{len(holdout_df)}"
        )

    sanity_model = build_production_pipeline()
    sanity_model.fit(
        train_df[feature_columns].copy(),
        _encode_labels(train_df[TARGET_COLUMN], LABELS),
    )
    probabilities = _predict_probabilities(
        sanity_model,
        holdout_df[feature_columns].copy(),
        LABELS,
    )
    predictions = _predict_argmax(probabilities, LABELS)
    metrics = _evaluate_predictions(
        holdout_df[TARGET_COLUMN].tolist(),
        predictions,
        probabilities,
        LABELS,
    )
    material_differences = {
        metric_name: float(metrics[metric_name] - expected_value)
        for metric_name, expected_value in DOCUMENTED_FINAL_HOLDOUT.items()
        if abs(float(metrics[metric_name] - expected_value)) > SANITY_CHECK_TOLERANCE
    }
    matched_documented_holdout = not material_differences
    print("=== Preseason Sanity Check ===")
    print("Informational only; no tuning is allowed from this check.")
    print(
        "2021-22 through 2024-25 -> 2025-26: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"brier={metrics['brier_score']:.4f}, "
        f"draw_f1={metrics['draw_f1']:.4f}"
    )
    if not matched_documented_holdout:
        print(f"SANITY_CHECK_WARNING: material differences {material_differences}")
    else:
        print("PASS: sanity check matched documented final holdout metrics.")

    return {
        "train_seasons": TRAIN_SEASONS[:-1],
        "holdout_season": TRAIN_SEASONS[-1],
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "metrics": metrics,
        "documented_reference": DOCUMENTED_FINAL_HOLDOUT,
        "matched_documented_holdout": matched_documented_holdout,
        "material_differences": material_differences,
        "note": "Informational reproduction only; no tuning performed.",
    }


def save_production_artifacts(model, feature_columns, draw_config, metadata) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _validate_artifact_targets_absent()

    feature_payload = {
        "model_name": MODEL_NAME,
        "source_table": SOURCE_TABLE,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "forbidden_feature_families_excluded": [
            "H2H",
            "style",
            "pressure",
            "Poisson",
            "odds",
            "manager",
            "sentiment",
            "injury",
            "rivalry",
            "derby",
            "calibration",
        ],
    }
    draw_payload = {
        "model_name": MODEL_NAME,
        **draw_config,
    }

    with MODEL_ARTIFACT.open("xb") as handle:
        joblib.dump(model, handle)
    _write_json_new(FEATURE_ARTIFACT, feature_payload)
    _write_json_new(DRAW_THRESHOLD_ARTIFACT, draw_payload)
    _write_json_new(METADATA_ARTIFACT, metadata)

    for artifact_path in [
        MODEL_ARTIFACT,
        FEATURE_ARTIFACT,
        DRAW_THRESHOLD_ARTIFACT,
        METADATA_ARTIFACT,
    ]:
        if not artifact_path.exists():
            raise RuntimeError(f"Expected artifact was not created: {artifact_path}")

    print("PASS: production artifacts created.")
    for artifact_path in [
        MODEL_ARTIFACT,
        FEATURE_ARTIFACT,
        DRAW_THRESHOLD_ARTIFACT,
        METADATA_ARTIFACT,
    ]:
        print(f"- {artifact_path.relative_to(PROJECT_ROOT)}")


def create_or_refresh_elo_current(conn) -> None:
    create_sql = text(
        """
        CREATE TABLE IF NOT EXISTS elo_current_v3 (
            team_name TEXT PRIMARY KEY,
            elo_rating FLOAT NOT NULL,
            source_season TEXT NOT NULL,
            source_match_id INTEGER NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    refresh_sql = text(
        """
        WITH team_elos AS (
            SELECT
                home_team AS team_name,
                home_elo_after AS elo_rating,
                season_id AS source_season,
                match_id AS source_match_id,
                match_date,
                kickoff_time
            FROM elo_ratings_v3
            UNION ALL
            SELECT
                away_team AS team_name,
                away_elo_after AS elo_rating,
                season_id AS source_season,
                match_id AS source_match_id,
                match_date,
                kickoff_time
            FROM elo_ratings_v3
        ),
        ranked AS (
            SELECT
                team_name,
                elo_rating,
                source_season,
                source_match_id,
                ROW_NUMBER() OVER (
                    PARTITION BY team_name
                    ORDER BY match_date DESC, kickoff_time DESC NULLS LAST, source_match_id DESC
                ) AS rank_number
            FROM team_elos
        )
        INSERT INTO elo_current_v3 (
            team_name,
            elo_rating,
            source_season,
            source_match_id,
            updated_at
        )
        SELECT
            team_name,
            elo_rating,
            source_season,
            source_match_id,
            CURRENT_TIMESTAMP
        FROM ranked
        WHERE rank_number = 1
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(create_sql)
        db_conn.execute(text("DELETE FROM elo_current_v3"))
        db_conn.execute(refresh_sql)

    with conn.connect() as db_conn:
        top_teams = db_conn.execute(
            text(
                """
                SELECT team_name, elo_rating, source_season, source_match_id
                FROM elo_current_v3
                ORDER BY elo_rating DESC
                LIMIT 10
                """
            )
        ).mappings().all()

    print("PASS: elo_current_v3 created/refreshed.")
    print("Top 10 Elo teams:")
    for rank, row in enumerate(top_teams, start=1):
        print(
            f"{rank}. {row['team_name']}: {float(row['elo_rating']):.2f} "
            f"({row['source_season']} match_id={row['source_match_id']})"
        )


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            exists = db_conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = CURRENT_SCHEMA()
                            AND table_name = :table_name
                    )
                    """
                ),
                {"table_name": table_name},
            ).scalar_one()
            if exists:
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_counts_unchanged_except_allowed(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    unexpected = {
        table_name: counts
        for table_name, counts in changed.items()
        if table_name not in ALLOWED_COUNT_CHANGES
    }
    if unexpected:
        raise RuntimeError(f"Unexpected watched table count changes: {unexpected}")

    print("PASS: watched source/Tier 2 table counts unchanged.")
    if "elo_current_v3" in changed:
        print(
            "Allowed count change: "
            f"elo_current_v3 {changed['elo_current_v3'][0]} -> {changed['elo_current_v3'][1]}"
        )
    else:
        print("elo_current_v3 count unchanged.")


def main() -> None:
    print("=== Production Phase P1: Train Production Logistic Elo Model ===")
    print("Production fit only; no tuning, no competing models, no validation claims.")
    _validate_artifact_targets_absent()

    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)

    df = load_training_data(conn)
    feature_columns = get_feature_columns(df)
    validate_training_data(df, feature_columns)
    sanity_check = run_preseason_sanity_check(df, feature_columns)

    model = build_production_pipeline()
    model.fit(
        df[feature_columns].copy(),
        _encode_labels(df[TARGET_COLUMN], LABELS),
    )
    draw_config = select_draw_threshold_from_training(
        model,
        df[feature_columns].copy(),
        df[TARGET_COLUMN].tolist(),
        LABELS,
    )
    print(
        "Selected production draw threshold from production training rows only: "
        f"{draw_config['selected_draw_threshold']:.2f}"
    )

    create_or_refresh_elo_current(conn)

    metadata = {
        "model_name": MODEL_NAME,
        "source_table": SOURCE_TABLE,
        "training_seasons": TRAIN_SEASONS,
        "training_row_count": len(df),
        "feature_count": len(feature_columns),
        "feature_artifact_filename": FEATURE_ARTIFACT.name,
        "model_artifact_filename": MODEL_ARTIFACT.name,
        "draw_threshold_artifact_filename": DRAW_THRESHOLD_ARTIFACT.name,
        "metadata_artifact_filename": METADATA_ARTIFACT.name,
        "model_type": "sklearn.pipeline.Pipeline",
        "sklearn_pipeline_steps": [
            "SimpleImputer(strategy='median')",
            "StandardScaler()",
            f"LogisticRegression(max_iter=2000, random_state={RANDOM_STATE})",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "production_training_note": PRODUCTION_SEASON_NOTE,
        "production_use_note": PRODUCTION_USE_NOTE,
        "preseason_sanity_check": sanity_check,
        "draw_threshold": draw_config["selected_draw_threshold"],
        "no_tuning_statement": (
            "No hyperparameters, feature sets, or thresholds were tuned in P1."
        ),
    }
    save_production_artifacts(model, feature_columns, draw_config, metadata)

    after_counts = capture_watched_table_counts(conn)
    assert_counts_unchanged_except_allowed(before_counts, after_counts)
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        print(f"- {table_name}: {before_counts.get(table_name)} -> {after_counts.get(table_name)}")

    print("PASS: production model trained on all 1900 completed historical rows.")
    print("PASS: no existing models/saved file was overwritten.")
    print("No Streamlit, Tier 2 artifact, H2H, style, pressure, Poisson, calibration, odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


def _validate_artifact_targets_absent() -> None:
    existing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in [
            MODEL_ARTIFACT,
            FEATURE_ARTIFACT,
            DRAW_THRESHOLD_ARTIFACT,
            METADATA_ARTIFACT,
        ]
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Production artifact target(s) already exist; refusing to overwrite: "
            + ", ".join(existing)
        )


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _predict_probabilities(model, x_values, labels) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(x_values)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned, labels)


def _predict_argmax(probabilities, labels) -> list[str]:
    probabilities = _normalize_probabilities(probabilities, labels)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    return [labels[int(index)] for index in prediction_indexes]


def _apply_draw_overlay(probabilities, labels, draw_threshold) -> list[str]:
    probabilities = _normalize_probabilities(probabilities, labels)
    predictions = _predict_argmax(probabilities, labels)
    draw_index = labels.index("D")
    overlay_predictions: list[str] = []
    for row_index, probability_row in enumerate(probabilities):
        draw_prob = float(probability_row[draw_index])
        max_prob = float(probability_row.max())
        should_change_to_draw = (
            draw_prob >= draw_threshold
            and _is_draw_second_highest(probability_row, labels)
            and max_prob < DOMINANT_CLASS_MAX_PROB
        )
        overlay_predictions.append("D" if should_change_to_draw else predictions[row_index])
    return overlay_predictions


def _is_draw_second_highest(probability_row, labels) -> bool:
    probabilities = numpy.asarray(probability_row, dtype=float)
    draw_index = labels.index("D")
    descending_indexes = numpy.argsort(-probabilities, kind="mergesort")
    return int(descending_indexes[1]) == draw_index


def _evaluate_predictions(y_true, y_pred, probabilities, labels) -> dict:
    y_true_encoded = _encode_labels(y_true, labels)
    y_pred_encoded = _encode_labels(y_pred, labels)
    probabilities = _normalize_probabilities(probabilities, labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=[labels.index("D")],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(
            log_loss(y_true_encoded, probabilities, labels=list(range(len(labels))))
        ),
        "brier_score": _multiclass_brier_score(y_true_encoded, probabilities, labels),
        "draw_recall": float(recall[0]),
        "draw_precision": float(precision[0]),
        "draw_f1": float(f1[0]),
    }


def _multiclass_brier_score(y_true_encoded, probabilities, labels) -> float:
    probabilities = _normalize_probabilities(probabilities, labels)
    y_one_hot = numpy.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[numpy.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(numpy.mean(numpy.sum((probabilities - y_one_hot) ** 2, axis=1)))


def _normalize_probabilities(probabilities, labels) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    if probabilities.shape[1] != len(labels):
        raise ValueError("Probability column count does not match labels")
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("Non-finite probability value found")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Probability row with non-positive sum found")
    normalized = probabilities / row_sums
    if not numpy.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Probability rows do not sum to 1")
    return normalized


def _encode_labels(values, labels: list[str]) -> numpy.ndarray:
    if isinstance(values, pandas.Series):
        raw_values = values.tolist()
    else:
        raw_values = list(values)

    if all(isinstance(value, (int, numpy.integer)) for value in raw_values):
        encoded = numpy.asarray(raw_values, dtype=int)
        unknown_indexes = sorted(set(encoded.tolist()) - set(range(len(labels))))
        if unknown_indexes:
            raise ValueError(f"Unknown encoded label indexes: {unknown_indexes}")
        return encoded

    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_labels = sorted(set(raw_values) - set(labels))
    if unknown_labels:
        raise ValueError(f"Unknown label values: {unknown_labels}")
    return numpy.asarray([label_to_index[value] for value in raw_values], dtype=int)


def _count_labels_changed_to_draw(
    baseline_predictions: list[str],
    overlay_predictions: list[str],
) -> int:
    return int(
        sum(
            baseline != "D" and overlay == "D"
            for baseline, overlay in zip(baseline_predictions, overlay_predictions)
        )
    )


def _git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (numpy.integer,)):
        return int(value)
    if isinstance(value, (numpy.floating,)):
        return float(value)
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
