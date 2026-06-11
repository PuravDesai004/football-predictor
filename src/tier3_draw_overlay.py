from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy
import pandas
from sqlalchemy import text
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import get_engine
from tier3_validation import validate_historical_match_integrity


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_TEST_SEASON = "2025-26"
FOLDS = [
    {
        "fold": 1,
        "train_seasons": ["2021-22", "2022-23"],
        "validation_seasons": ["2023-24"],
    },
    {
        "fold": 2,
        "train_seasons": ["2021-22", "2022-23", "2023-24"],
        "validation_seasons": ["2024-25"],
    },
]
RANDOM_STATE = 42
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
DOMINANT_CLASS_MAX_PROB = 0.50
THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]

EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_SEASON_ROWS = 380
MAX_CHANGED_TO_DRAW_RATE = 0.15

BASE_ELO_FEATURE_COLUMNS = [
    "home_home_matches_last5",
    "away_away_matches_last5",
    "home_overall_matches_last5",
    "away_overall_matches_last5",
    "home_overall_matches_last10",
    "away_overall_matches_last10",
    "home_goals_scored_home_last5",
    "home_goals_conceded_home_last5",
    "home_clean_sheet_rate_home_last5",
    "away_goals_scored_away_last5",
    "away_goals_conceded_away_last5",
    "away_clean_sheet_rate_away_last5",
    "home_xg_home_last5",
    "home_xga_home_last5",
    "away_xg_away_last5",
    "away_xga_away_last5",
    "home_points_overall_last5",
    "away_points_overall_last5",
    "home_points_overall_last10",
    "away_points_overall_last10",
    "home_goal_diff_overall_last5",
    "away_goal_diff_overall_last5",
    "home_xg_overall_last5",
    "home_xga_overall_last5",
    "away_xg_overall_last5",
    "away_xga_overall_last5",
    "home_elo_before",
    "away_elo_before",
    "elo_diff_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
]

FORBIDDEN_POST_MATCH_ELO_COLUMNS = {
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
EXCLUDED_FEATURE_COLUMNS = {
    "match_id",
    "season",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "result",
    "home_goals",
    "away_goals",
    "home_win",
    "is_draw",
    "away_win",
    "created_at",
    *FORBIDDEN_POST_MATCH_ELO_COLUMNS,
}
FORBIDDEN_FEATURE_TOKENS = [
    "h2h",
    "style",
    "pressure",
    "poisson",
    "odds",
    "manager",
    "sentiment",
    "injury",
    "rivalry",
    "derby",
]
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "match_features_v3_style_experiment",
    "match_features_v3_h2h_experiment",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_elo_feature_data(conn) -> pandas.DataFrame:
    if FINAL_TEST_SEASON in DEV_SEASONS:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be a development season")

    query = text(
        """
        SELECT *
        FROM match_features_v3_elo
        WHERE season_id = ANY(:dev_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pandas.read_sql(query, conn, params={"dev_seasons": DEV_SEASONS})

    errors: list[str] = []
    if len(df) != EXPECTED_DEV_ROW_COUNT:
        errors.append(f"expected {EXPECTED_DEV_ROW_COUNT} rows, found {len(df)}")
    if df.empty:
        errors.append("development Elo feature dataframe is empty")
    if df["match_id"].duplicated().any():
        errors.append(f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}")
    if df[TARGET_COLUMN].isna().any():
        errors.append(f"null target count: {int(df[TARGET_COLUMN].isna().sum())}")

    seasons_present = sorted(df["season_id"].dropna().unique().tolist())
    if seasons_present != DEV_SEASONS:
        errors.append(f"development seasons {seasons_present} != {DEV_SEASONS}")

    season_counts = df.groupby("season_id").size().to_dict()
    bad_counts = {
        season_id: int(count)
        for season_id, count in season_counts.items()
        if int(count) != EXPECTED_SEASON_ROWS
    }
    if bad_counts:
        errors.append(f"bad development season row counts: {bad_counts}")

    unknown_results = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
    if unknown_results:
        errors.append(f"unknown result labels: {unknown_results}")

    forbidden_loaded = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & set(df.columns))
    if forbidden_loaded:
        errors.append(f"forbidden post-match Elo columns loaded: {forbidden_loaded}")

    forbidden_family_columns = [
        column
        for column in df.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_family_columns:
        errors.append(f"forbidden experimental columns loaded: {forbidden_family_columns}")

    if errors:
        raise ValueError("Elo feature data validation failed: " + "; ".join(errors))

    df["match_date"] = pandas.to_datetime(df["match_date"])
    validate_no_final_holdout_loaded(df)

    print("=== Elo Development Feature Data ===")
    print(f"Development dataframe rows: {len(df)}")
    for season_id, count in df.groupby("season_id").size().sort_index().items():
        print(f"- {season_id}: {int(count)} rows")
    print(f"{FINAL_TEST_SEASON} was not loaded into the modeling dataframe.")
    return df


def validate_no_final_holdout_loaded(df) -> None:
    if FINAL_TEST_SEASON in set(df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} was loaded into the modeling dataframe")
    print(f"PASS: {FINAL_TEST_SEASON} not loaded.")


def get_feature_columns(df) -> list[str]:
    missing_columns = sorted(set(BASE_ELO_FEATURE_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing base + Elo feature columns: {missing_columns}")

    forbidden_features = sorted(set(BASE_ELO_FEATURE_COLUMNS) & EXCLUDED_FEATURE_COLUMNS)
    if forbidden_features:
        raise ValueError(f"Forbidden columns selected as features: {forbidden_features}")

    forbidden_family_features = [
        column
        for column in BASE_ELO_FEATURE_COLUMNS
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_family_features:
        raise ValueError(
            "Forbidden feature family selected: "
            f"{forbidden_family_features}"
        )

    non_numeric_columns = [
        column
        for column in BASE_ELO_FEATURE_COLUMNS
        if not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Non-numeric feature columns selected: {non_numeric_columns}")

    print(f"Base + Elo feature count: {len(BASE_ELO_FEATURE_COLUMNS)}")
    return BASE_ELO_FEATURE_COLUMNS.copy()


def build_logistic_elo_model():
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


def predict_probabilities(model, X, labels) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned)


def predict_argmax(probabilities, labels) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    return [labels[int(index)] for index in prediction_indexes]


def is_draw_second_highest(probability_row, labels) -> bool:
    probabilities = numpy.asarray(probability_row, dtype=float)
    if probabilities.shape != (len(labels),):
        raise ValueError("Probability row shape does not match labels")
    draw_index = labels.index("D")
    descending_indexes = numpy.argsort(-probabilities, kind="mergesort")
    return int(descending_indexes[1]) == draw_index


def apply_draw_overlay(probabilities, labels, draw_threshold) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    predictions = predict_argmax(probabilities, labels)
    draw_index = labels.index("D")

    overlay_predictions: list[str] = []
    for row_index, probability_row in enumerate(probabilities):
        draw_prob = float(probability_row[draw_index])
        max_prob = float(probability_row.max())
        should_change_to_draw = (
            draw_prob >= draw_threshold
            and is_draw_second_highest(probability_row, labels)
            and max_prob < DOMINANT_CLASS_MAX_PROB
        )
        overlay_predictions.append("D" if should_change_to_draw else predictions[row_index])
    return overlay_predictions


def select_draw_threshold_from_training(probabilities, y_true, labels) -> dict:
    probabilities = _normalize_probabilities(probabilities)
    argmax_predictions = predict_argmax(probabilities, labels)
    argmax_metrics = evaluate_predictions(y_true, argmax_predictions, probabilities, labels)

    best_config: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for draw_threshold in THRESHOLD_GRID:
        overlay_predictions = apply_draw_overlay(probabilities, labels, draw_threshold)
        metrics = evaluate_predictions(y_true, overlay_predictions, probabilities, labels)
        labels_changed_to_draw = _count_labels_changed_to_draw(
            argmax_predictions,
            overlay_predictions,
        )
        rejected_for_accuracy = (
            metrics["accuracy"] < argmax_metrics["accuracy"] - 0.01
        )
        candidate = {
            "draw_threshold": draw_threshold,
            "training_accuracy": metrics["accuracy"],
            "training_draw_recall": metrics["draw_recall"],
            "training_draw_precision": metrics["draw_precision"],
            "training_draw_f1": metrics["draw_f1"],
            "labels_changed_to_draw": labels_changed_to_draw,
            "rejected_for_accuracy": rejected_for_accuracy,
        }
        candidates.append(candidate)
        if rejected_for_accuracy:
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
    return best_config


def evaluate_predictions(y_true, y_pred, probabilities, labels) -> dict:
    y_true_encoded = _encode_labels(y_true, labels)
    y_pred_encoded = _encode_labels(y_pred, labels)
    probabilities = _normalize_probabilities(probabilities)
    encoded_labels = list(range(len(labels)))
    draw_index = labels.index("D")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=[draw_index],
        average=None,
        zero_division=0,
    )
    predicted_distribution = {
        label: int(numpy.sum(y_pred_encoded == index))
        for index, label in enumerate(labels)
    }

    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(log_loss(y_true_encoded, probabilities, labels=encoded_labels)),
        "brier_score": _multiclass_brier_score(y_true_encoded, probabilities, labels),
        "draw_recall": float(recall[0]),
        "draw_precision": float(precision[0]),
        "draw_f1": float(f1[0]),
        "predicted_class_distribution": predicted_distribution,
        "confusion_counts": _confusion_counts(y_true_encoded, y_pred_encoded, labels),
    }


def run_fold(df, fold_config) -> dict:
    fold = int(fold_config["fold"])
    train_seasons = fold_config["train_seasons"]
    validation_seasons = fold_config["validation_seasons"]
    train_df = df.loc[df["season_id"].isin(train_seasons)].copy()
    validation_df = df.loc[df["season_id"].isin(validation_seasons)].copy()

    _validate_fold_window(fold, train_df, validation_df, train_seasons, validation_seasons)

    feature_columns = get_feature_columns(df)
    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()
    y_train_encoded = _encode_labels(train_df[TARGET_COLUMN], LABELS)
    y_train = train_df[TARGET_COLUMN].tolist()
    y_validation = validation_df[TARGET_COLUMN].tolist()

    model = build_logistic_elo_model()
    model.fit(X_train, y_train_encoded)

    training_probabilities = predict_probabilities(model, X_train, LABELS)
    validation_probabilities = predict_probabilities(model, X_validation, LABELS)
    threshold_config = select_draw_threshold_from_training(
        training_probabilities,
        y_train,
        LABELS,
    )

    normal_predictions = predict_argmax(validation_probabilities, LABELS)
    overlay_predictions = apply_draw_overlay(
        validation_probabilities,
        LABELS,
        threshold_config["selected_draw_threshold"],
    )

    normal_metrics = evaluate_predictions(
        y_validation,
        normal_predictions,
        validation_probabilities,
        LABELS,
    )
    overlay_metrics = evaluate_predictions(
        y_validation,
        overlay_predictions,
        validation_probabilities,
        LABELS,
    )
    labels_changed_to_draw = _count_labels_changed_to_draw(
        normal_predictions,
        overlay_predictions,
    )

    probability_metrics_unchanged = (
        normal_metrics["log_loss"] == overlay_metrics["log_loss"]
        and normal_metrics["brier_score"] == overlay_metrics["brier_score"]
    )
    if not probability_metrics_unchanged:
        raise RuntimeError(
            "Probability metric changed after hard-label overlay: "
            f"log_loss {normal_metrics['log_loss']} vs {overlay_metrics['log_loss']}, "
            f"brier {normal_metrics['brier_score']} vs {overlay_metrics['brier_score']}"
        )

    print(f"=== Fold {fold} ===")
    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Validation seasons: {', '.join(validation_seasons)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(
        f"Selected draw threshold from training only: "
        f"{threshold_config['selected_draw_threshold']:.2f}"
    )
    print(
        "Training selection metrics: "
        f"argmax_draw_f1={threshold_config['training_argmax_draw_f1']:.4f}, "
        f"overlay_draw_f1={threshold_config['training_overlay_draw_f1']:.4f}, "
        f"argmax_accuracy={threshold_config['training_argmax_accuracy']:.4f}, "
        f"overlay_accuracy={threshold_config['training_overlay_accuracy']:.4f}, "
        f"changed_to_draw={threshold_config['training_labels_changed_to_draw']}"
    )
    print("Normal argmax validation metrics:")
    _print_model_result("logistic_elo_argmax", normal_metrics, changed_to_draw=0)
    print("Draw overlay validation metrics:")
    _print_model_result(
        "logistic_elo_draw_overlay",
        overlay_metrics,
        changed_to_draw=labels_changed_to_draw,
    )
    print("PASS: log_loss and Brier unchanged by hard-label overlay.")

    return {
        "fold": fold,
        "train_seasons": train_seasons,
        "validation_seasons": validation_seasons,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "selected_draw_threshold": threshold_config["selected_draw_threshold"],
        "threshold_config": threshold_config,
        "normal_metrics": normal_metrics,
        "overlay_metrics": overlay_metrics,
        "labels_changed_to_draw": labels_changed_to_draw,
        "probability_metrics_unchanged": probability_metrics_unchanged,
    }


def aggregate_results(fold_results) -> pandas.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]

    for result in fold_results:
        for prediction_mode, metrics, changed_to_draw in [
            ("normal_argmax", result["normal_metrics"], 0),
            (
                "draw_overlay",
                result["overlay_metrics"],
                result["labels_changed_to_draw"],
            ),
        ]:
            row = {
                "prediction_mode": prediction_mode,
                "fold": result["fold"],
                "train_rows": result["train_rows"],
                "validation_rows": result["validation_rows"],
                "selected_draw_threshold": result["selected_draw_threshold"],
                "labels_changed_to_draw": changed_to_draw,
                **{metric: metrics[metric] for metric in metric_columns},
                "pred_H": metrics["predicted_class_distribution"]["H"],
                "pred_D": metrics["predicted_class_distribution"]["D"],
                "pred_A": metrics["predicted_class_distribution"]["A"],
            }
            rows.append(row)

    results_df = pandas.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for prediction_mode, mode_df in results_df.groupby("prediction_mode", sort=False):
        aggregate_row = {
            "prediction_mode": prediction_mode,
            "folds": int(mode_df["fold"].nunique()),
            "labels_changed_to_draw": int(mode_df["labels_changed_to_draw"].sum()),
            "pred_H": int(mode_df["pred_H"].sum()),
            "pred_D": int(mode_df["pred_D"].sum()),
            "pred_A": int(mode_df["pred_A"].sum()),
        }
        for metric in metric_columns:
            aggregate_row[metric] = float(mode_df[metric].mean())
        aggregate_rows.append(aggregate_row)

    aggregate_df = pandas.DataFrame(aggregate_rows)
    print("=== Aggregate Metrics ===")
    print(_format_aggregate_table(aggregate_df))
    return aggregate_df


def print_overlay_verdict(fold_results, aggregate_df) -> None:
    del aggregate_df

    fold_deltas: list[dict[str, float]] = []
    thresholds = [float(result["selected_draw_threshold"]) for result in fold_results]
    for result in fold_results:
        normal = result["normal_metrics"]
        overlay = result["overlay_metrics"]
        fold_deltas.append(
            {
                "fold": result["fold"],
                "accuracy": float(overlay["accuracy"] - normal["accuracy"]),
                "log_loss": float(overlay["log_loss"] - normal["log_loss"]),
                "brier_score": float(overlay["brier_score"] - normal["brier_score"]),
                "draw_recall": float(overlay["draw_recall"] - normal["draw_recall"]),
                "draw_precision": float(overlay["draw_precision"] - normal["draw_precision"]),
                "draw_f1": float(overlay["draw_f1"] - normal["draw_f1"]),
                "labels_changed_to_draw": result["labels_changed_to_draw"],
                "validation_rows": result["validation_rows"],
            }
        )

    threshold_spread = max(thresholds) - min(thresholds)
    threshold_stable = threshold_spread <= 0.06
    probability_metrics_unchanged = all(
        result["probability_metrics_unchanged"] for result in fold_results
    )
    changed_not_excessive = all(
        delta["labels_changed_to_draw"] <= MAX_CHANGED_TO_DRAW_RATE * delta["validation_rows"]
        for delta in fold_deltas
    )

    gates = {
        "draw_f1_improves_at_least_0.03_both_folds": all(
            delta["draw_f1"] >= 0.03 for delta in fold_deltas
        ),
        "draw_recall_improves_both_folds": all(
            delta["draw_recall"] > 0 for delta in fold_deltas
        ),
        "accuracy_drop_not_more_than_0.01_each_fold": all(
            delta["accuracy"] >= -0.01 for delta in fold_deltas
        ),
        "thresholds_stable_within_0.06": threshold_stable,
        "probability_metrics_unchanged": probability_metrics_unchanged,
        "labels_changed_to_draw_not_excessive": changed_not_excessive,
    }

    print("=== Overlay vs Argmax Fold Deltas ===")
    for delta in fold_deltas:
        print(f"Fold {delta['fold']}:")
        print(f"- accuracy: {delta['accuracy']:+.4f}")
        print(f"- log_loss: {delta['log_loss']:+.4f}")
        print(f"- brier_score: {delta['brier_score']:+.4f}")
        print(f"- draw_recall: {delta['draw_recall']:+.4f}")
        print(f"- draw_precision: {delta['draw_precision']:+.4f}")
        print(f"- draw_f1: {delta['draw_f1']:+.4f}")
        print(f"- labels_changed_to_draw: {int(delta['labels_changed_to_draw'])}")

    print("=== Threshold Stability ===")
    print(
        "Selected thresholds: "
        + ", ".join(f"{threshold:.2f}" for threshold in thresholds)
    )
    print(f"Threshold spread: {threshold_spread:.2f}")
    if not threshold_stable:
        print("THRESHOLD_UNSTABLE")

    print("=== Acceptance Gate Checks ===")
    for gate_name, passed in gates.items():
        print(f"{gate_name}: {'PASS' if passed else 'FAIL'}")

    verdict = (
        "ACCEPT_DRAW_OVERLAY_EXPERIMENTAL_SERVING_HELPER"
        if all(gates.values())
        else "REJECT_DRAW_OVERLAY"
    )
    print("=== Phase 9A Verdict ===")
    print(f"Verdict: {verdict}")
    if verdict == "ACCEPT_DRAW_OVERLAY_EXPERIMENTAL_SERVING_HELPER":
        print("Reason: overlay passed all hard-label helper gates without changing probabilities.")
    else:
        print("Reason: overlay failed at least one hard-label helper gate.")
    print("Probability model remains logistic_elo_expanding; probabilities were not adjusted.")


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


def assert_watched_counts_unchanged(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    if changed:
        raise RuntimeError(f"Watched table counts changed unexpectedly: {changed}")
    print("Watched table counts unchanged.")


def main() -> None:
    if LABELS != ["H", "D", "A"]:
        raise RuntimeError(f"Label order changed unexpectedly: {LABELS}")

    print("=== Tier 3 Phase 9A Hard-Label Draw Overlay Experiment ===")
    print("Overlay may change hard labels only; it does not adjust probabilities.")
    conn = get_db_connection()
    validate_historical_match_integrity(conn)

    before_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts before experiment", before_counts)

    df = load_elo_feature_data(conn)
    validate_no_final_holdout_loaded(df)
    feature_columns = get_feature_columns(df)
    print(f"Feature columns used: {len(feature_columns)} base + Elo columns only.")

    fold_results = [run_fold(df, fold_config) for fold_config in FOLDS]
    aggregate_df = aggregate_results(fold_results)
    print_overlay_verdict(fold_results, aggregate_df)

    after_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts after experiment", after_counts)
    assert_watched_counts_unchanged(before_counts, after_counts)

    print(f"PASS: {FINAL_TEST_SEASON} was not loaded, tuned, evaluated, or reported for model metrics.")
    print("PASS: selected draw thresholds used training rows only.")
    print("PASS: log_loss and Brier were unchanged by the hard-label overlay.")
    print("No database writes occurred.")
    print("No model artifacts were saved.")
    print("No Streamlit, Tier 2 artifact, H2H, style, pressure, Poisson-feature, betting odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    if probabilities.shape[1] != len(LABELS):
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


def _multiclass_brier_score(
    y_true_encoded: numpy.ndarray,
    probabilities: numpy.ndarray,
    labels: list[str],
) -> float:
    probabilities = _normalize_probabilities(probabilities)
    y_one_hot = numpy.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[numpy.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(numpy.mean(numpy.sum((probabilities - y_one_hot) ** 2, axis=1)))


def _confusion_counts(
    y_true_encoded: numpy.ndarray,
    y_pred_encoded: numpy.ndarray,
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=list(range(len(labels))),
    )
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(labels)
        }
        for true_index, true_label in enumerate(labels)
    }


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


def _validate_fold_window(
    fold: int,
    train_df: pandas.DataFrame,
    validation_df: pandas.DataFrame,
    train_seasons: list[str],
    validation_seasons: list[str],
) -> None:
    loaded_seasons = set(train_df["season_id"]) | set(validation_df["season_id"])
    if FINAL_TEST_SEASON in loaded_seasons:
        raise ValueError(f"{FINAL_TEST_SEASON} entered fold {fold}")
    overlap = sorted(set(train_seasons) & set(validation_seasons))
    if overlap:
        raise ValueError(f"Fold {fold} train/validation overlap: {overlap}")
    if len(train_df) != len(train_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(train_seasons) * EXPECTED_SEASON_ROWS} "
            f"train rows, found {len(train_df)}"
        )
    if len(validation_df) != len(validation_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(validation_seasons) * EXPECTED_SEASON_ROWS} "
            f"validation rows, found {len(validation_df)}"
        )
    if train_df["match_date"].max() >= validation_df["match_date"].min():
        raise ValueError(f"Fold {fold} date leakage")
    print(
        f"Fold {fold} date check: "
        f"train max {train_df['match_date'].max().date()} < "
        f"validation min {validation_df['match_date'].min().date()}"
    )


def _print_model_result(
    model_name: str,
    metrics: dict[str, Any],
    changed_to_draw: int,
) -> None:
    distribution = metrics["predicted_class_distribution"]
    print(
        f"{model_name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"brier_score={metrics['brier_score']:.4f}, "
        f"draw_recall={metrics['draw_recall']:.4f}, "
        f"draw_precision={metrics['draw_precision']:.4f}, "
        f"draw_f1={metrics['draw_f1']:.4f}, "
        f"changed_to_draw={changed_to_draw}"
    )
    print(
        "Predicted distribution: "
        + ", ".join(f"{label}={distribution[label]}" for label in LABELS)
    )
    _print_confusion_counts(metrics["confusion_counts"])


def _print_confusion_counts(confusion_counts: dict[str, dict[str, int]]) -> None:
    print("Confusion counts (rows=true, columns=pred):")
    print("true\\pred    H    D    A")
    for true_label in LABELS:
        row = confusion_counts[true_label]
        print(
            f"{true_label:>4}     "
            f"{row['H']:4d} {row['D']:4d} {row['A']:4d}"
        )


def _format_aggregate_table(aggregate_df: pandas.DataFrame) -> str:
    columns = [
        "prediction_mode",
        "folds",
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
        "labels_changed_to_draw",
        "pred_H",
        "pred_D",
        "pred_A",
    ]
    formatters = {
        "accuracy": "{:.4f}".format,
        "log_loss": "{:.4f}".format,
        "brier_score": "{:.4f}".format,
        "draw_recall": "{:.4f}".format,
        "draw_precision": "{:.4f}".format,
        "draw_f1": "{:.4f}".format,
    }
    return aggregate_df[columns].to_string(index=False, formatters=formatters)


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name in WATCHED_TABLES:
        print(f"{table_name}: {counts.get(table_name)}")


if __name__ == "__main__":
    main()
