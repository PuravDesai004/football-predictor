from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ARTIFACT = PROJECT_ROOT / "models" / "saved" / "production_logistic_elo_v3.pkl"
FEATURE_ARTIFACT = PROJECT_ROOT / "models" / "saved" / "production_features_v3.json"
DRAW_CONFIG_ARTIFACT = PROJECT_ROOT / "models" / "saved" / "production_draw_threshold_v3.json"
METADATA_ARTIFACT = PROJECT_ROOT / "models" / "saved" / "production_metadata_v3.json"
MODEL_NAME = "production_logistic_elo_v3"
LABELS = ["H", "D", "A"]
DEFAULT_TARGET_SEASON = "2026-27"
SOURCE_FEATURE_TABLE = "production_upcoming_match_features_v3"

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402


PREDICTION_TABLES = [
    "production_prediction_runs",
    "production_match_predictions",
]
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "elo_current_v3",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "match_features_v3_style_experiment",
    "match_features_v3_h2h_experiment",
    "production_ingestion_runs",
    "production_fpl_bootstrap_snapshots",
    "production_fpl_fixture_snapshots",
    "production_football_data_match_staging",
    "production_understat_xg_staging",
    "production_data_freshness",
    *PREDICTION_TABLES,
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]
ALLOWED_CHANGED_TABLES = set(PREDICTION_TABLES)
RUN_STATUSES = {"started", "success", "skipped", "failed"}
PREDICTION_MODES = {"upcoming", "historical_replay", "dry_run"}
EXPECTED_FEATURE_COUNT = 32
DOMINANT_CLASS_MAX_PROB = 0.50
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
]
FORBIDDEN_EXACT_FEATURES = {
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
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
SCHEMA_REQUIRED_COLUMNS = {
    "production_prediction_runs": [
        "prediction_run_id",
        "run_started_at",
        "run_finished_at",
        "run_status",
        "target_season",
        "target_gameweek",
        "prediction_mode",
        "model_name",
        "model_artifact_path",
        "feature_artifact_path",
        "draw_threshold",
        "source_feature_table",
        "rows_loaded",
        "rows_predicted",
        "error_message",
        "created_at",
    ],
    "production_match_predictions": [
        "prediction_id",
        "prediction_run_id",
        "target_season",
        "target_gameweek",
        "match_id",
        "fixture_id",
        "match_date",
        "kickoff_time",
        "home_team",
        "away_team",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "argmax_prediction",
        "overlay_prediction",
        "draw_overlay_applied",
        "draw_risk_flag",
        "confidence",
        "model_name",
        "feature_source",
        "prediction_created_at",
        "actual_result",
        "was_correct_argmax",
        "was_correct_overlay",
        "scored_at",
        "created_at",
    ],
}


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def create_prediction_tables(conn) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS production_prediction_runs (
            prediction_run_id SERIAL PRIMARY KEY,
            run_started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            run_finished_at TIMESTAMP NULL,
            run_status TEXT NOT NULL,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NULL,
            prediction_mode TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_artifact_path TEXT NOT NULL,
            feature_artifact_path TEXT NOT NULL,
            draw_threshold FLOAT NOT NULL,
            source_feature_table TEXT NOT NULL,
            rows_loaded INTEGER NOT NULL DEFAULT 0,
            rows_predicted INTEGER NOT NULL DEFAULT 0,
            error_message TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (run_status IN ('started', 'success', 'skipped', 'failed')),
            CHECK (prediction_mode IN ('upcoming', 'historical_replay', 'dry_run')),
            CHECK (target_gameweek IS NULL OR target_gameweek > 0),
            CHECK (draw_threshold >= 0 AND draw_threshold <= 1),
            CHECK (rows_loaded >= 0),
            CHECK (rows_predicted >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_match_predictions (
            prediction_id BIGSERIAL PRIMARY KEY,
            prediction_run_id INTEGER NOT NULL
                REFERENCES production_prediction_runs(prediction_run_id),
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NULL,
            match_id INTEGER NULL,
            fixture_id INTEGER NULL,
            match_date DATE NOT NULL,
            kickoff_time TIMESTAMP NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            prob_home_win FLOAT NOT NULL,
            prob_draw FLOAT NOT NULL,
            prob_away_win FLOAT NOT NULL,
            argmax_prediction TEXT NOT NULL,
            overlay_prediction TEXT NOT NULL,
            draw_overlay_applied BOOLEAN NOT NULL,
            draw_risk_flag BOOLEAN NOT NULL,
            confidence FLOAT NOT NULL,
            model_name TEXT NOT NULL,
            feature_source TEXT NOT NULL,
            prediction_created_at TIMESTAMP NOT NULL,
            actual_result TEXT NULL,
            was_correct_argmax BOOLEAN NULL,
            was_correct_overlay BOOLEAN NULL,
            scored_at TIMESTAMP NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (home_team <> away_team),
            CHECK (target_gameweek IS NULL OR target_gameweek > 0),
            CHECK (prob_home_win >= 0 AND prob_home_win <= 1),
            CHECK (prob_draw >= 0 AND prob_draw <= 1),
            CHECK (prob_away_win >= 0 AND prob_away_win <= 1),
            CHECK (ABS((prob_home_win + prob_draw + prob_away_win) - 1.0) <= 0.000001),
            CHECK (argmax_prediction IN ('H', 'D', 'A')),
            CHECK (overlay_prediction IN ('H', 'D', 'A')),
            CHECK (actual_result IS NULL OR actual_result IN ('H', 'D', 'A')),
            CHECK (confidence >= 0 AND confidence <= 1)
        )
        """,
    ]
    with conn.begin() as db_conn:
        for statement in statements:
            db_conn.execute(text(statement))
    _verify_prediction_tables(conn)
    print("PASS: production prediction tables exist and required columns are present.")


def load_artifacts() -> tuple[object, list[str], dict, dict]:
    for artifact_path in [
        MODEL_ARTIFACT,
        FEATURE_ARTIFACT,
        DRAW_CONFIG_ARTIFACT,
        METADATA_ARTIFACT,
    ]:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing production artifact: {artifact_path}")

    model = joblib.load(MODEL_ARTIFACT)
    with FEATURE_ARTIFACT.open("r", encoding="utf-8") as handle:
        feature_payload = json.load(handle)
    with DRAW_CONFIG_ARTIFACT.open("r", encoding="utf-8") as handle:
        draw_config = json.load(handle)
    with METADATA_ARTIFACT.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_columns = feature_payload.get("features")
    if not isinstance(feature_columns, list):
        raise ValueError("Feature artifact missing list field: features")
    return model, [str(column) for column in feature_columns], draw_config, metadata


def validate_artifacts(model, feature_columns, draw_config, metadata) -> None:
    errors: list[str] = []
    if not hasattr(model, "predict_proba"):
        errors.append("model artifact does not expose predict_proba")
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        errors.append(
            f"feature count expected {EXPECTED_FEATURE_COUNT}, found {len(feature_columns)}"
        )
    if metadata.get("model_name") != MODEL_NAME:
        errors.append(f"metadata model_name {metadata.get('model_name')} != {MODEL_NAME}")
    if metadata.get("feature_count") != EXPECTED_FEATURE_COUNT:
        errors.append(
            f"metadata feature_count {metadata.get('feature_count')} != {EXPECTED_FEATURE_COUNT}"
        )
    draw_threshold = _extract_draw_threshold(draw_config)
    if not isinstance(draw_threshold, (int, float)):
        errors.append("draw threshold is missing or non-numeric")
    elif not 0 <= float(draw_threshold) <= 1:
        errors.append(f"draw threshold outside 0..1: {draw_threshold}")

    duplicate_features = sorted(
        {column for column in feature_columns if feature_columns.count(column) > 1}
    )
    forbidden_exact = sorted(set(feature_columns) & FORBIDDEN_EXACT_FEATURES)
    forbidden_tokens = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if duplicate_features:
        errors.append(f"duplicate feature name(s): {duplicate_features}")
    if forbidden_exact:
        errors.append(f"forbidden exact feature(s): {forbidden_exact}")
    if forbidden_tokens:
        errors.append(f"forbidden feature family token(s): {forbidden_tokens}")

    if errors:
        raise ValueError("Production artifact validation failed: " + "; ".join(errors))

    print("PASS: production artifacts validated.")
    print(f"Model artifact: {MODEL_ARTIFACT.relative_to(PROJECT_ROOT)}")
    print(f"Feature artifact: {FEATURE_ARTIFACT.relative_to(PROJECT_ROOT)}")
    print(f"Draw threshold: {float(draw_threshold):.2f}")
    print(f"Feature count: {len(feature_columns)}")


def start_prediction_run(conn, target_season, target_gameweek, prediction_mode) -> int:
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"Invalid prediction_mode: {prediction_mode}")
    draw_threshold = _read_draw_threshold_from_artifact()
    query = text(
        """
        INSERT INTO production_prediction_runs (
            run_status,
            target_season,
            target_gameweek,
            prediction_mode,
            model_name,
            model_artifact_path,
            feature_artifact_path,
            draw_threshold,
            source_feature_table
        )
        VALUES (
            'started',
            :target_season,
            :target_gameweek,
            :prediction_mode,
            :model_name,
            :model_artifact_path,
            :feature_artifact_path,
            :draw_threshold,
            :source_feature_table
        )
        RETURNING prediction_run_id
        """
    )
    with conn.begin() as db_conn:
        prediction_run_id = int(
            db_conn.execute(
                query,
                {
                    "target_season": target_season,
                    "target_gameweek": target_gameweek,
                    "prediction_mode": prediction_mode,
                    "model_name": MODEL_NAME,
                    "model_artifact_path": str(MODEL_ARTIFACT.relative_to(PROJECT_ROOT)),
                    "feature_artifact_path": str(FEATURE_ARTIFACT.relative_to(PROJECT_ROOT)),
                    "draw_threshold": draw_threshold,
                    "source_feature_table": SOURCE_FEATURE_TABLE,
                },
            ).scalar_one()
        )
    print(f"Started prediction_run_id={prediction_run_id} mode={prediction_mode}")
    return prediction_run_id


def finish_prediction_run(
    conn,
    prediction_run_id,
    status,
    rows_loaded,
    rows_predicted,
    error_message=None,
) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Invalid prediction run status: {status}")
    query = text(
        """
        UPDATE production_prediction_runs
        SET
            run_finished_at = CURRENT_TIMESTAMP,
            run_status = :run_status,
            rows_loaded = :rows_loaded,
            rows_predicted = :rows_predicted,
            error_message = :error_message
        WHERE prediction_run_id = :prediction_run_id
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(
            query,
            {
                "prediction_run_id": prediction_run_id,
                "run_status": status,
                "rows_loaded": int(rows_loaded),
                "rows_predicted": int(rows_predicted),
                "error_message": error_message,
            },
        )
    print(f"Finished prediction_run_id={prediction_run_id} status={status}")


def source_feature_table_exists(conn) -> bool:
    with conn.connect() as db_conn:
        exists = _table_exists(db_conn, SOURCE_FEATURE_TABLE)
    print(f"{SOURCE_FEATURE_TABLE} exists: {exists}")
    return exists


def load_upcoming_feature_rows(conn, target_season, target_gameweek=None) -> pandas.DataFrame:
    if not source_feature_table_exists(conn):
        return pandas.DataFrame()
    columns = _table_columns(conn, SOURCE_FEATURE_TABLE)
    season_column = _first_existing(columns, ["target_season", "season_id", "season"])
    if season_column is None:
        raise ValueError(
            f"{SOURCE_FEATURE_TABLE} has no target_season/season_id/season column"
        )

    where_clauses = [f"{season_column} = :target_season"]
    params: dict[str, Any] = {"target_season": target_season}
    if target_gameweek is not None:
        gameweek_column = _first_existing(columns, ["target_gameweek", "gameweek", "event_id"])
        if gameweek_column is None:
            raise ValueError(
                f"{SOURCE_FEATURE_TABLE} cannot filter target_gameweek={target_gameweek}"
            )
        where_clauses.append(f"{gameweek_column} = :target_gameweek")
        params["target_gameweek"] = target_gameweek

    order_columns = [
        column
        for column in ["match_date", "kickoff_time", "fixture_id", "match_id"]
        if column in columns
    ]
    order_sql = ", ".join(order_columns) if order_columns else season_column
    query = text(
        f"""
        SELECT *
        FROM {SOURCE_FEATURE_TABLE}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY {order_sql}
        """
    )
    df = pandas.read_sql(query, conn, params=params)
    print(
        f"Loaded {len(df)} feature row(s) from {SOURCE_FEATURE_TABLE} "
        f"for {target_season}"
        + (f" gameweek {target_gameweek}" if target_gameweek is not None else "")
    )
    return df


def validate_prediction_features(df, feature_columns) -> None:
    if df.empty:
        print("Prediction feature dataframe has 0 rows.")
        return

    errors: list[str] = []
    missing_features = sorted(set(feature_columns) - set(df.columns))
    if missing_features:
        errors.append(f"missing production feature column(s): {missing_features}")

    required_metadata = ["match_date", "home_team", "away_team"]
    missing_metadata = sorted(set(required_metadata) - set(df.columns))
    if missing_metadata:
        errors.append(f"missing prediction metadata column(s): {missing_metadata}")

    season_column = _df_first_existing(df, ["target_season", "season_id", "season"])
    if season_column is None:
        errors.append("missing target season metadata column")

    non_numeric_features = [
        column
        for column in feature_columns
        if column in df.columns and not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        errors.append(f"non-numeric feature column(s): {non_numeric_features}")

    null_metadata = {
        column: int(df[column].isna().sum())
        for column in required_metadata
        if column in df.columns and int(df[column].isna().sum()) > 0
    }
    if null_metadata:
        errors.append(f"null required metadata count(s): {null_metadata}")

    if "home_team" in df and "away_team" in df:
        same_team = int((df["home_team"] == df["away_team"]).sum())
        if same_team:
            errors.append(f"home_team equals away_team for {same_team} row(s)")

    feature_nulls = {
        column: int(df[column].isna().sum())
        for column in feature_columns
        if column in df.columns and int(df[column].isna().sum()) > 0
    }
    if feature_nulls:
        print(f"Feature nulls will be handled by the production pipeline: {feature_nulls}")

    if errors:
        raise ValueError("Prediction feature validation failed: " + "; ".join(errors))
    print(f"PASS: prediction feature rows validated ({len(df)} rows).")


def predict_probabilities(model, X) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    probabilities = numpy.asarray(raw_probabilities, dtype=float)

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        observed_classes = model.named_steps["model"].classes_
        aligned = numpy.zeros((probabilities.shape[0], len(LABELS)), dtype=float)
        for source_index, class_index in enumerate(observed_classes):
            aligned[:, int(class_index)] = probabilities[:, source_index]
        probabilities = aligned

    probabilities = _normalize_probabilities(probabilities)
    print(f"Predicted probability rows: {probabilities.shape[0]}")
    return probabilities


def predict_argmax(probabilities) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    predictions = [LABELS[int(index)] for index in prediction_indexes]
    _validate_prediction_labels(predictions, "argmax")
    return predictions


def is_draw_second_highest(probability_row) -> bool:
    probabilities = numpy.asarray(probability_row, dtype=float)
    if probabilities.shape != (len(LABELS),):
        raise ValueError("Probability row shape does not match labels")
    draw_index = LABELS.index("D")
    descending_indexes = numpy.argsort(-probabilities, kind="mergesort")
    return int(descending_indexes[1]) == draw_index


def apply_draw_overlay(
    probabilities,
    draw_threshold,
    dominant_class_max_prob,
) -> tuple[list[str], list[bool]]:
    probabilities = _normalize_probabilities(probabilities)
    argmax_predictions = predict_argmax(probabilities)
    draw_index = LABELS.index("D")

    overlay_predictions: list[str] = []
    overlay_applied: list[bool] = []
    for row_index, probability_row in enumerate(probabilities):
        draw_prob = float(probability_row[draw_index])
        max_prob = float(probability_row.max())
        should_change_to_draw = (
            argmax_predictions[row_index] != "D"
            and draw_prob >= float(draw_threshold)
            and is_draw_second_highest(probability_row)
            and max_prob < float(dominant_class_max_prob)
        )
        overlay_predictions.append("D" if should_change_to_draw else argmax_predictions[row_index])
        overlay_applied.append(bool(should_change_to_draw))

    _validate_prediction_labels(overlay_predictions, "overlay")
    return overlay_predictions, overlay_applied


def build_prediction_rows(
    df,
    probabilities,
    argmax_predictions,
    overlay_predictions,
    overlay_applied,
    prediction_run_id,
    metadata,
) -> pandas.DataFrame:
    probabilities = _normalize_probabilities(probabilities)
    if len(df) != len(probabilities):
        raise ValueError("Prediction dataframe and probability row counts differ")

    draw_threshold = float(metadata["draw_threshold"])
    season_column = _df_first_existing(df, ["target_season", "season_id", "season"])
    gameweek_column = _df_first_existing(df, ["target_gameweek", "gameweek", "event_id"])
    prediction_time = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []

    for row_index, source_row in df.reset_index(drop=True).iterrows():
        probability_row = probabilities[row_index]
        prob_home = float(probability_row[LABELS.index("H")])
        prob_draw = float(probability_row[LABELS.index("D")])
        prob_away = float(probability_row[LABELS.index("A")])
        rows.append(
            {
                "prediction_run_id": int(prediction_run_id),
                "target_season": _clean_text(
                    source_row.get(season_column)
                    if season_column
                    else metadata.get("target_season")
                ),
                "target_gameweek": _nullable_int(
                    source_row.get(gameweek_column)
                    if gameweek_column
                    else metadata.get("target_gameweek")
                ),
                "match_id": _nullable_int(source_row.get("match_id")),
                "fixture_id": _nullable_int(source_row.get("fixture_id")),
                "match_date": _to_date(source_row.get("match_date")),
                "kickoff_time": _to_timestamp(source_row.get("kickoff_time")),
                "home_team": _clean_text(source_row.get("home_team")),
                "away_team": _clean_text(source_row.get("away_team")),
                "prob_home_win": prob_home,
                "prob_draw": prob_draw,
                "prob_away_win": prob_away,
                "argmax_prediction": argmax_predictions[row_index],
                "overlay_prediction": overlay_predictions[row_index],
                "draw_overlay_applied": bool(overlay_applied[row_index]),
                "draw_risk_flag": bool(
                    overlay_predictions[row_index] == "D" or prob_draw >= draw_threshold
                ),
                "confidence": float(probability_row.max()),
                "model_name": MODEL_NAME,
                "feature_source": SOURCE_FEATURE_TABLE,
                "prediction_created_at": prediction_time,
                "actual_result": None,
                "was_correct_argmax": None,
                "was_correct_overlay": None,
                "scored_at": None,
            }
        )

    prediction_rows = pandas.DataFrame(rows)
    expected_draw_risk = (
        (prediction_rows["overlay_prediction"] == "D")
        | (prediction_rows["prob_draw"] >= draw_threshold)
    )
    if not (prediction_rows["draw_risk_flag"] == expected_draw_risk).all():
        raise ValueError("draw_risk_flag does not match overlay/threshold rule")
    _validate_prediction_rows(prediction_rows)
    return prediction_rows


def write_predictions(conn, prediction_rows) -> int:
    if prediction_rows.empty:
        print("Wrote 0 production_match_predictions rows.")
        return 0

    _assert_no_existing_prediction_keys(conn, prediction_rows)
    query = text(
        """
        INSERT INTO production_match_predictions (
            prediction_run_id,
            target_season,
            target_gameweek,
            match_id,
            fixture_id,
            match_date,
            kickoff_time,
            home_team,
            away_team,
            prob_home_win,
            prob_draw,
            prob_away_win,
            argmax_prediction,
            overlay_prediction,
            draw_overlay_applied,
            draw_risk_flag,
            confidence,
            model_name,
            feature_source,
            prediction_created_at,
            actual_result,
            was_correct_argmax,
            was_correct_overlay,
            scored_at
        )
        VALUES (
            :prediction_run_id,
            :target_season,
            :target_gameweek,
            :match_id,
            :fixture_id,
            :match_date,
            :kickoff_time,
            :home_team,
            :away_team,
            :prob_home_win,
            :prob_draw,
            :prob_away_win,
            :argmax_prediction,
            :overlay_prediction,
            :draw_overlay_applied,
            :draw_risk_flag,
            :confidence,
            :model_name,
            :feature_source,
            :prediction_created_at,
            :actual_result,
            :was_correct_argmax,
            :was_correct_overlay,
            :scored_at
        )
        """
    )
    records = [_db_safe_record(row) for row in prediction_rows.to_dict(orient="records")]
    with conn.begin() as db_conn:
        db_conn.execute(query, records)
    print(f"Wrote {len(records)} production_match_predictions rows.")
    return len(records)


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            if _table_exists(db_conn, table_name):
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_counts_unchanged_except_predictions(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    unexpected = {
        table_name: counts
        for table_name, counts in changed.items()
        if table_name not in ALLOWED_CHANGED_TABLES
    }
    if unexpected:
        raise RuntimeError(f"Unexpected watched table count changes: {unexpected}")

    print("PASS: watched non-prediction table counts unchanged.")
    if changed:
        print("Allowed prediction table count changes:")
        for table_name in sorted(changed):
            print(f"- {table_name}: {changed[table_name][0]} -> {changed[table_name][1]}")
    else:
        print("No watched table counts changed.")


def main() -> None:
    args = parse_args()
    conn = get_db_connection()
    create_prediction_tables(conn)

    model, feature_columns, draw_config, metadata = load_artifacts()
    validate_artifacts(model, feature_columns, draw_config, metadata)
    draw_threshold = float(_extract_draw_threshold(draw_config))

    if args.init_schema_only:
        print("=== Production P3A schema initialization only ===")
        print_prediction_table_counts(conn)
        return

    prediction_mode = "dry_run" if args.dry_run else "upcoming"
    before_counts = capture_watched_table_counts(conn)
    prediction_run_id = start_prediction_run(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
        prediction_mode=prediction_mode,
    )
    rows_loaded = 0
    rows_predicted = 0
    status = "failed"
    error_message: str | None = None

    try:
        if not source_feature_table_exists(conn):
            error_message = f"SKIPPED_NO_FEATURE_TABLE: {SOURCE_FEATURE_TABLE} does not exist"
            finish_prediction_run(conn, prediction_run_id, "skipped", 0, 0, error_message)
            print(error_message)
            status = "skipped"
        else:
            feature_df = load_upcoming_feature_rows(
                conn,
                target_season=args.target_season,
                target_gameweek=args.target_gameweek,
            )
            rows_loaded = len(feature_df)
            if feature_df.empty:
                error_message = "SKIPPED_NO_FEATURE_ROWS"
                finish_prediction_run(conn, prediction_run_id, "skipped", 0, 0, error_message)
                print(error_message)
                status = "skipped"
            elif args.dry_run:
                validate_prediction_features(feature_df, feature_columns)
                finish_prediction_run(
                    conn,
                    prediction_run_id,
                    "skipped",
                    rows_loaded,
                    0,
                    "DRY_RUN_NO_PREDICTIONS_WRITTEN",
                )
                print("DRY_RUN_NO_PREDICTIONS_WRITTEN")
                status = "skipped"
            else:
                validate_prediction_features(feature_df, feature_columns)
                probabilities = predict_probabilities(
                    model,
                    feature_df[feature_columns].copy(),
                )
                argmax_predictions = predict_argmax(probabilities)
                overlay_predictions, overlay_applied = apply_draw_overlay(
                    probabilities,
                    draw_threshold,
                    DOMINANT_CLASS_MAX_PROB,
                )
                row_metadata = dict(metadata)
                row_metadata.update(
                    {
                        "target_season": args.target_season,
                        "target_gameweek": args.target_gameweek,
                        "draw_threshold": draw_threshold,
                    }
                )
                prediction_rows = build_prediction_rows(
                    feature_df,
                    probabilities,
                    argmax_predictions,
                    overlay_predictions,
                    overlay_applied,
                    prediction_run_id,
                    row_metadata,
                )
                rows_predicted = write_predictions(conn, prediction_rows)
                finish_prediction_run(
                    conn,
                    prediction_run_id,
                    "success",
                    rows_loaded,
                    rows_predicted,
                )
                status = "success"
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        finish_prediction_run(
            conn,
            prediction_run_id,
            "failed",
            rows_loaded,
            rows_predicted,
            error_message,
        )
        print(f"FAILED: {error_message}")
        raise
    finally:
        after_counts = capture_watched_table_counts(conn)
        assert_counts_unchanged_except_predictions(before_counts, after_counts)
        print_watched_count_comparison(before_counts, after_counts)
        print_latest_prediction_run(conn)

    print(
        "Prediction run summary: "
        f"status={status}, rows_loaded={rows_loaded}, rows_predicted={rows_predicted}"
    )
    print("No fake fixtures, features, or predictions were created.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production P3A prediction foundation")
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    return parser.parse_args()


def print_prediction_table_counts(conn) -> None:
    with conn.connect() as db_conn:
        for table_name in PREDICTION_TABLES:
            count = int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
            print(f"{table_name}: {count}")


def print_watched_count_comparison(before_counts, after_counts) -> None:
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        print(f"- {table_name}: {before_counts.get(table_name)} -> {after_counts.get(table_name)}")


def print_latest_prediction_run(conn) -> None:
    with conn.connect() as db_conn:
        latest_run = db_conn.execute(
            text(
                """
                SELECT *
                FROM production_prediction_runs
                ORDER BY prediction_run_id DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        prediction_count_by_run = db_conn.execute(
            text(
                """
                SELECT prediction_run_id, COUNT(*) AS row_count
                FROM production_match_predictions
                GROUP BY prediction_run_id
                ORDER BY prediction_run_id
                """
            )
        ).mappings().all()
    print("Latest production_prediction_runs row:")
    print(_json_dumps_mapping(latest_run))
    print("production_match_predictions row count by prediction_run_id:")
    for row in prediction_count_by_run:
        print(f"- prediction_run_id={row['prediction_run_id']}: {row['row_count']}")


def _verify_prediction_tables(conn) -> None:
    with conn.connect() as db_conn:
        for table_name, required_columns in SCHEMA_REQUIRED_COLUMNS.items():
            if not _table_exists(db_conn, table_name):
                raise RuntimeError(f"{table_name} table does not exist")
            columns = [
                row["column_name"]
                for row in db_conn.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = CURRENT_SCHEMA()
                            AND table_name = :table_name
                        """
                    ),
                    {"table_name": table_name},
                ).mappings().all()
            ]
            missing = sorted(set(required_columns) - set(columns))
            if missing:
                raise RuntimeError(f"{table_name} missing required column(s): {missing}")


def _table_exists(db_conn, table_name: str) -> bool:
    return bool(
        db_conn.execute(
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
    )


def _table_columns(conn, table_name: str) -> list[str]:
    with conn.connect() as db_conn:
        return [
            row["column_name"]
            for row in db_conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = :table_name
                    ORDER BY ordinal_position
                    """
                ),
                {"table_name": table_name},
            ).mappings().all()
        ]


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _df_first_existing(df: pandas.DataFrame, candidates: list[str]) -> str | None:
    return _first_existing(list(df.columns), candidates)


def _extract_draw_threshold(draw_config: dict) -> float:
    for key in ["selected_draw_threshold", "draw_threshold"]:
        value = draw_config.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError("Draw config artifact does not contain a numeric threshold")


def _read_draw_threshold_from_artifact() -> float:
    with DRAW_CONFIG_ARTIFACT.open("r", encoding="utf-8") as handle:
        return _extract_draw_threshold(json.load(handle))


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    if probabilities.shape[1] != len(LABELS):
        raise ValueError("Probability column count does not match labels")
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("Non-finite probability value found")
    if numpy.any(probabilities < 0) or numpy.any(probabilities > 1):
        raise ValueError("Probability outside 0..1 found")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Probability row with non-positive sum found")
    normalized = probabilities / row_sums
    if not numpy.allclose(normalized.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Probability rows do not sum to 1")
    return normalized


def _validate_prediction_labels(predictions: list[str], label_name: str) -> None:
    invalid = sorted(set(predictions) - set(LABELS))
    if invalid:
        raise ValueError(f"Invalid {label_name} prediction label(s): {invalid}")


def _validate_prediction_rows(prediction_rows: pandas.DataFrame) -> None:
    errors: list[str] = []
    if prediction_rows.empty:
        return
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    if prediction_rows[probability_columns].isna().any().any():
        errors.append("null probability value found")
    for column in probability_columns:
        if ((prediction_rows[column] < 0) | (prediction_rows[column] > 1)).any():
            errors.append(f"{column} outside 0..1")
    probability_sums = prediction_rows[probability_columns].sum(axis=1)
    if not numpy.allclose(probability_sums, 1.0, atol=1e-6):
        errors.append("probability rows do not sum to 1")
    for column in ["argmax_prediction", "overlay_prediction"]:
        invalid = sorted(set(prediction_rows[column].dropna()) - set(LABELS))
        if invalid:
            errors.append(f"invalid {column} label(s): {invalid}")
    if not numpy.allclose(
        prediction_rows["confidence"],
        prediction_rows[probability_columns].max(axis=1),
        atol=1e-12,
    ):
        errors.append("confidence does not equal max probability")
    if prediction_rows["actual_result"].notna().any():
        errors.append("actual_result must be null at prediction time")
    if prediction_rows["was_correct_argmax"].notna().any():
        errors.append("was_correct_argmax must be null at prediction time")
    if prediction_rows["was_correct_overlay"].notna().any():
        errors.append("was_correct_overlay must be null at prediction time")
    if prediction_rows["scored_at"].notna().any():
        errors.append("scored_at must be null at prediction time")
    if errors:
        raise ValueError("Prediction row validation failed: " + "; ".join(errors))


def _assert_no_existing_prediction_keys(conn, prediction_rows: pandas.DataFrame) -> None:
    duplicate_descriptions: list[str] = []
    with conn.connect() as db_conn:
        for row in prediction_rows.to_dict(orient="records"):
            params = {
                "target_season": row["target_season"],
                "target_gameweek": row["target_gameweek"],
                "match_id": row["match_id"],
                "fixture_id": row["fixture_id"],
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            }
            duplicate_count = int(
                db_conn.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM production_match_predictions
                        WHERE target_season = :target_season
                            AND (
                                (target_gameweek IS NULL AND :target_gameweek IS NULL)
                                OR target_gameweek = :target_gameweek
                            )
                            AND (
                                (:match_id IS NOT NULL AND match_id = :match_id)
                                OR (:fixture_id IS NOT NULL AND fixture_id = :fixture_id)
                                OR (
                                    match_date = :match_date
                                    AND home_team = :home_team
                                    AND away_team = :away_team
                                )
                            )
                        """
                    ),
                    params,
                ).scalar_one()
            )
            if duplicate_count:
                duplicate_descriptions.append(
                    f"{row['target_season']} gw={row['target_gameweek']} "
                    f"{row['home_team']} vs {row['away_team']} {row['match_date']}"
                )
    if duplicate_descriptions:
        raise RuntimeError(
            "Existing predictions found; --replace is not implemented in P3A. "
            f"Examples: {duplicate_descriptions[:5]}"
        )


def _to_date(value):
    if value is None or pandas.isna(value):
        return None
    parsed = pandas.to_datetime(value, errors="coerce")
    if pandas.isna(parsed):
        raise ValueError(f"Unparseable match_date: {value}")
    return parsed.date()


def _to_timestamp(value):
    if value is None or pandas.isna(value):
        return None
    parsed = pandas.to_datetime(value, utc=True, errors="coerce")
    if pandas.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(tzinfo=None)


def _nullable_int(value):
    if value is None or pandas.isna(value):
        return None
    return int(value)


def _clean_text(value) -> str | None:
    if value is None or pandas.isna(value):
        return None
    text_value = str(value).strip()
    return text_value if text_value else None


def _db_safe_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _db_safe_value(value) for key, value in row.items()}


def _db_safe_value(value):
    if isinstance(value, pandas.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.to_pydatetime()
    if isinstance(value, float) and pandas.isna(value):
        return None
    if pandas.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _json_dumps_mapping(row) -> str:
    if row is None:
        return "{}"
    return json.dumps(
        {key: _json_safe(value) for key, value in dict(row).items()},
        sort_keys=True,
        default=str,
    )


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
