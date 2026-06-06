from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import pandas
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_TABLE = "production_match_predictions"
PREDICTION_RUN_TABLE = "production_prediction_runs"
HEALTH_TABLE = "production_model_health_log"
HISTORICAL_MATCHES_TABLE = "historical_matches"
LABELS = ["H", "D", "A"]
DEFAULT_TARGET_SEASON = "2026-27"

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402


WATCHED_TABLES = [
    HISTORICAL_MATCHES_TABLE,
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
    "production_team_name_mapping",
    "production_upcoming_match_features_v3",
    PREDICTION_RUN_TABLE,
    PREDICTION_TABLE,
    HEALTH_TABLE,
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]
ALLOWED_ROW_COUNT_CHANGE_TABLES = {HEALTH_TABLE}
HEALTH_REQUIRED_COLUMNS = [
    "health_log_id",
    "computed_at",
    "target_season",
    "target_gameweek",
    "model_name",
    "prediction_count",
    "scored_count",
    "accuracy_argmax",
    "accuracy_overlay",
    "log_loss",
    "brier",
    "draw_recall_argmax",
    "draw_precision_argmax",
    "draw_f1_argmax",
    "draw_recall_overlay",
    "draw_precision_overlay",
    "draw_f1_overlay",
    "home_pred_rate_argmax",
    "draw_pred_rate_argmax",
    "away_pred_rate_argmax",
    "home_actual_rate",
    "draw_actual_rate",
    "away_actual_rate",
    "notes",
    "created_at",
]
PREDICTION_REQUIRED_COLUMNS = [
    "prediction_id",
    "prediction_run_id",
    "target_season",
    "target_gameweek",
    "match_id",
    "fixture_id",
    "match_date",
    "home_team",
    "away_team",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "argmax_prediction",
    "overlay_prediction",
    "actual_result",
    "was_correct_argmax",
    "was_correct_overlay",
    "scored_at",
]
RESULT_REQUIRED_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "result",
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def create_health_table(conn) -> None:
    statement = f"""
        CREATE TABLE IF NOT EXISTS {HEALTH_TABLE} (
            health_log_id BIGSERIAL PRIMARY KEY,
            computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NULL,
            model_name TEXT NOT NULL,
            prediction_count INTEGER NOT NULL DEFAULT 0,
            scored_count INTEGER NOT NULL DEFAULT 0,
            accuracy_argmax FLOAT NULL,
            accuracy_overlay FLOAT NULL,
            log_loss FLOAT NULL,
            brier FLOAT NULL,
            draw_recall_argmax FLOAT NULL,
            draw_precision_argmax FLOAT NULL,
            draw_f1_argmax FLOAT NULL,
            draw_recall_overlay FLOAT NULL,
            draw_precision_overlay FLOAT NULL,
            draw_f1_overlay FLOAT NULL,
            home_pred_rate_argmax FLOAT NULL,
            draw_pred_rate_argmax FLOAT NULL,
            away_pred_rate_argmax FLOAT NULL,
            home_actual_rate FLOAT NULL,
            draw_actual_rate FLOAT NULL,
            away_actual_rate FLOAT NULL,
            notes TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (target_gameweek IS NULL OR target_gameweek > 0),
            CHECK (prediction_count >= 0),
            CHECK (scored_count >= 0),
            CHECK (accuracy_argmax IS NULL OR (accuracy_argmax >= 0 AND accuracy_argmax <= 1)),
            CHECK (accuracy_overlay IS NULL OR (accuracy_overlay >= 0 AND accuracy_overlay <= 1)),
            CHECK (log_loss IS NULL OR log_loss >= 0),
            CHECK (brier IS NULL OR brier >= 0),
            CHECK (draw_recall_argmax IS NULL OR (draw_recall_argmax >= 0 AND draw_recall_argmax <= 1)),
            CHECK (draw_precision_argmax IS NULL OR (draw_precision_argmax >= 0 AND draw_precision_argmax <= 1)),
            CHECK (draw_f1_argmax IS NULL OR (draw_f1_argmax >= 0 AND draw_f1_argmax <= 1)),
            CHECK (draw_recall_overlay IS NULL OR (draw_recall_overlay >= 0 AND draw_recall_overlay <= 1)),
            CHECK (draw_precision_overlay IS NULL OR (draw_precision_overlay >= 0 AND draw_precision_overlay <= 1)),
            CHECK (draw_f1_overlay IS NULL OR (draw_f1_overlay >= 0 AND draw_f1_overlay <= 1)),
            CHECK (home_pred_rate_argmax IS NULL OR (home_pred_rate_argmax >= 0 AND home_pred_rate_argmax <= 1)),
            CHECK (draw_pred_rate_argmax IS NULL OR (draw_pred_rate_argmax >= 0 AND draw_pred_rate_argmax <= 1)),
            CHECK (away_pred_rate_argmax IS NULL OR (away_pred_rate_argmax >= 0 AND away_pred_rate_argmax <= 1)),
            CHECK (home_actual_rate IS NULL OR (home_actual_rate >= 0 AND home_actual_rate <= 1)),
            CHECK (draw_actual_rate IS NULL OR (draw_actual_rate >= 0 AND draw_actual_rate <= 1)),
            CHECK (away_actual_rate IS NULL OR (away_actual_rate >= 0 AND away_actual_rate <= 1))
        )
    """
    with conn.begin() as db_conn:
        db_conn.execute(text(statement))
    _verify_health_table(conn)
    print(f"PASS: {HEALTH_TABLE} exists and required columns are present.")


def load_unscored_predictions(conn, target_season=None, target_gameweek=None) -> pandas.DataFrame:
    _verify_table_columns(conn, PREDICTION_TABLE, PREDICTION_REQUIRED_COLUMNS)
    where_clauses = ["scored_at IS NULL"]
    params: dict[str, Any] = {}
    if target_season is not None:
        where_clauses.append("target_season = :target_season")
        params["target_season"] = target_season
    if target_gameweek is not None:
        where_clauses.append("target_gameweek = :target_gameweek")
        params["target_gameweek"] = int(target_gameweek)

    query = text(
        f"""
        SELECT *
        FROM {PREDICTION_TABLE}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY match_date, kickoff_time, prediction_id
        """
    )
    df = pandas.read_sql(query, conn, params=params)
    if not df.empty:
        df["match_date"] = pandas.to_datetime(df["match_date"], errors="coerce").dt.date
    print(
        f"Unscored predictions found: {len(df)}"
        + (f" for {target_season}" if target_season is not None else "")
        + (f" gameweek {target_gameweek}" if target_gameweek is not None else "")
    )
    return df


def load_actual_results(conn, target_season) -> pandas.DataFrame:
    _verify_table_columns(conn, HISTORICAL_MATCHES_TABLE, RESULT_REQUIRED_COLUMNS)
    query = text(
        f"""
        SELECT
            match_id,
            season_id AS target_season,
            match_date,
            home_team,
            away_team,
            result AS actual_result
        FROM {HISTORICAL_MATCHES_TABLE}
        WHERE season_id = :target_season
            AND result IN ('H', 'D', 'A')
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    results_df = pandas.read_sql(query, conn, params={"target_season": target_season})
    if not results_df.empty:
        results_df["match_date"] = pandas.to_datetime(
            results_df["match_date"],
            errors="coerce",
        ).dt.date
    _validate_result_rows(results_df)
    print(f"Actual results found for {target_season}: {len(results_df)}")
    return results_df


def join_predictions_to_results(pred_df, results_df) -> pandas.DataFrame:
    if pred_df.empty or results_df.empty:
        print("Joined completed results for 0 prediction row(s).")
        return _empty_scored_frame(pred_df)

    pred = pred_df.copy()
    results = results_df.copy()
    pred["match_date"] = pandas.to_datetime(pred["match_date"], errors="coerce").dt.date
    results["match_date"] = pandas.to_datetime(results["match_date"], errors="coerce").dt.date

    matched_frames: list[pandas.DataFrame] = []
    matched_prediction_ids: set[int] = set()

    if "match_id" in pred.columns and "match_id" in results.columns:
        id_pred = pred.loc[pred["match_id"].notna()].copy()
        if not id_pred.empty:
            id_result = results[["match_id", "actual_result"]].rename(
                columns={"actual_result": "joined_actual_result"}
            )
            id_joined = id_pred.merge(id_result, on="match_id", how="left")
            id_joined = id_joined.loc[id_joined["joined_actual_result"].notna()].copy()
            if not id_joined.empty:
                id_joined["actual_result"] = id_joined["joined_actual_result"]
                id_joined = id_joined.drop(columns=["joined_actual_result"])
                matched_prediction_ids.update(int(value) for value in id_joined["prediction_id"])
                matched_frames.append(id_joined)

    remaining = pred.loc[~pred["prediction_id"].isin(matched_prediction_ids)].copy()
    key_columns = ["target_season", "match_date", "home_team", "away_team"]
    if not remaining.empty:
        duplicated_result_keys = int(results.duplicated(key_columns).sum())
        if duplicated_result_keys:
            raise RuntimeError(
                "Actual result key is not unique for "
                f"{duplicated_result_keys} row(s); refusing fallback join."
            )
        key_result = results[key_columns + ["actual_result"]].rename(
            columns={"actual_result": "joined_actual_result"}
        )
        key_joined = remaining.merge(key_result, on=key_columns, how="left")
        key_joined = key_joined.loc[key_joined["joined_actual_result"].notna()].copy()
        if not key_joined.empty:
            key_joined["actual_result"] = key_joined["joined_actual_result"]
            key_joined = key_joined.drop(columns=["joined_actual_result"])
            matched_frames.append(key_joined)

    if not matched_frames:
        print(
            f"Joined completed results for 0 of {len(pred_df)} unscored prediction row(s)."
        )
        return _empty_scored_frame(pred_df)

    scored_df = pandas.concat(matched_frames, ignore_index=True)
    scored_df = scored_df.drop_duplicates(subset=["prediction_id"], keep="first")
    scored_df = scored_df.loc[scored_df["actual_result"].isin(LABELS)].copy()
    print(
        f"Joined completed results for {len(scored_df)} "
        f"of {len(pred_df)} unscored prediction row(s)."
    )
    return scored_df


def score_prediction_rows(scored_df) -> pandas.DataFrame:
    if scored_df.empty:
        return _empty_scored_frame(scored_df)

    scored = scored_df.copy()
    _validate_scoring_input(scored)
    scored["was_correct_argmax"] = scored["argmax_prediction"] == scored["actual_result"]
    scored["was_correct_overlay"] = scored["overlay_prediction"] == scored["actual_result"]
    scored["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    print(f"Scored prediction rows: {len(scored)}")
    return scored


def update_prediction_scores(conn, scored_df) -> int:
    if scored_df.empty:
        print(f"Updated 0 {PREDICTION_TABLE} scoring row(s).")
        return 0

    query = text(
        f"""
        UPDATE {PREDICTION_TABLE}
        SET
            actual_result = :actual_result,
            was_correct_argmax = :was_correct_argmax,
            was_correct_overlay = :was_correct_overlay,
            scored_at = :scored_at
        WHERE prediction_id = :prediction_id
            AND scored_at IS NULL
        """
    )
    records = [
        _db_safe_record(
            {
                "prediction_id": int(row["prediction_id"]),
                "actual_result": row["actual_result"],
                "was_correct_argmax": bool(row["was_correct_argmax"]),
                "was_correct_overlay": bool(row["was_correct_overlay"]),
                "scored_at": row["scored_at"],
            }
        )
        for row in scored_df.to_dict(orient="records")
    ]
    updated_count = 0
    with conn.begin() as db_conn:
        for record in records:
            updated_count += int(db_conn.execute(query, record).rowcount or 0)

    if updated_count != len(records):
        raise RuntimeError(
            f"Expected to update {len(records)} prediction row(s), updated {updated_count}."
        )
    print(f"Updated {updated_count} {PREDICTION_TABLE} scoring row(s).")
    return updated_count


def compute_health_metrics(scored_df, target_season=None, target_gameweek=None) -> dict:
    if scored_df.empty:
        raise ValueError("Cannot compute health metrics with 0 scored predictions.")

    scored = scored_df.copy()
    _validate_scoring_input(scored)
    probabilities = _normalized_probability_matrix(scored)
    actual_indexes = numpy.asarray([LABELS.index(label) for label in scored["actual_result"]])
    one_hot = numpy.zeros_like(probabilities)
    one_hot[numpy.arange(len(scored)), actual_indexes] = 1.0

    clipped = numpy.clip(probabilities, 1e-15, 1.0)
    log_loss = float(-numpy.mean(numpy.log(clipped[numpy.arange(len(scored)), actual_indexes])))
    brier = float(numpy.mean(numpy.sum((probabilities - one_hot) ** 2, axis=1)))
    argmax_metrics = _draw_metrics(scored["argmax_prediction"], scored["actual_result"])
    overlay_metrics = _draw_metrics(scored["overlay_prediction"], scored["actual_result"])

    if target_season is None:
        target_season = _single_or_multiple(scored["target_season"])
    model_name = _single_or_multiple(scored["model_name"])
    notes: list[str] = []
    if len(scored) < 5:
        notes.append("LOW_SAMPLE_SIZE")
    if model_name == "multiple":
        notes.append("MULTIPLE_MODELS")

    metrics = {
        "computed_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "target_season": str(target_season),
        "target_gameweek": _nullable_int(target_gameweek),
        "model_name": model_name,
        "prediction_count": int(len(scored)),
        "scored_count": int(len(scored)),
        "accuracy_argmax": float((scored["argmax_prediction"] == scored["actual_result"]).mean()),
        "accuracy_overlay": float(
            (scored["overlay_prediction"] == scored["actual_result"]).mean()
        ),
        "log_loss": log_loss,
        "brier": brier,
        "draw_recall_argmax": argmax_metrics["draw_recall"],
        "draw_precision_argmax": argmax_metrics["draw_precision"],
        "draw_f1_argmax": argmax_metrics["draw_f1"],
        "draw_recall_overlay": overlay_metrics["draw_recall"],
        "draw_precision_overlay": overlay_metrics["draw_precision"],
        "draw_f1_overlay": overlay_metrics["draw_f1"],
        "home_pred_rate_argmax": _label_rate(scored["argmax_prediction"], "H"),
        "draw_pred_rate_argmax": _label_rate(scored["argmax_prediction"], "D"),
        "away_pred_rate_argmax": _label_rate(scored["argmax_prediction"], "A"),
        "home_actual_rate": _label_rate(scored["actual_result"], "H"),
        "draw_actual_rate": _label_rate(scored["actual_result"], "D"),
        "away_actual_rate": _label_rate(scored["actual_result"], "A"),
        "notes": ";".join(notes) if notes else None,
    }
    print("Computed production model health metrics:")
    print(_format_mapping(metrics))
    return metrics


def write_health_log(conn, metrics) -> int:
    query = text(
        f"""
        INSERT INTO {HEALTH_TABLE} (
            computed_at,
            target_season,
            target_gameweek,
            model_name,
            prediction_count,
            scored_count,
            accuracy_argmax,
            accuracy_overlay,
            log_loss,
            brier,
            draw_recall_argmax,
            draw_precision_argmax,
            draw_f1_argmax,
            draw_recall_overlay,
            draw_precision_overlay,
            draw_f1_overlay,
            home_pred_rate_argmax,
            draw_pred_rate_argmax,
            away_pred_rate_argmax,
            home_actual_rate,
            draw_actual_rate,
            away_actual_rate,
            notes
        )
        VALUES (
            :computed_at,
            :target_season,
            :target_gameweek,
            :model_name,
            :prediction_count,
            :scored_count,
            :accuracy_argmax,
            :accuracy_overlay,
            :log_loss,
            :brier,
            :draw_recall_argmax,
            :draw_precision_argmax,
            :draw_f1_argmax,
            :draw_recall_overlay,
            :draw_precision_overlay,
            :draw_f1_overlay,
            :home_pred_rate_argmax,
            :draw_pred_rate_argmax,
            :away_pred_rate_argmax,
            :home_actual_rate,
            :draw_actual_rate,
            :away_actual_rate,
            :notes
        )
        RETURNING health_log_id
        """
    )
    with conn.begin() as db_conn:
        health_log_id = int(
            db_conn.execute(query, _db_safe_record(metrics)).scalar_one()
        )
    print(f"Wrote {HEALTH_TABLE} row health_log_id={health_log_id}")
    return health_log_id


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


def assert_counts_unchanged_except_scoring(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    unexpected = {
        table_name: counts
        for table_name, counts in changed.items()
        if table_name not in ALLOWED_ROW_COUNT_CHANGE_TABLES
    }
    if unexpected:
        raise RuntimeError(f"Unexpected watched table count changes: {unexpected}")

    health_change = changed.get(HEALTH_TABLE)
    if health_change and health_change[1] < health_change[0]:
        raise RuntimeError(f"{HEALTH_TABLE} row count decreased: {health_change}")

    print("PASS: watched table counts unchanged except allowed scoring health logs.")
    if changed:
        print("Allowed scoring table count changes:")
        for table_name in sorted(changed):
            print(f"- {table_name}: {changed[table_name][0]} -> {changed[table_name][1]}")
    else:
        print("No watched table counts changed.")


def main() -> None:
    args = parse_args()
    conn = get_db_connection()
    create_health_table(conn)

    if args.init_schema_only:
        print("=== Production P4A scoring schema initialization only ===")
        print_scoring_table_counts(conn)
        print("No predictions were scored and no health metrics were written.")
        return

    before_counts = capture_watched_table_counts(conn)
    scored_count = 0
    health_log_rows_written = 0
    skip_reason: str | None = None

    pred_df = load_unscored_predictions(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
    )
    if pred_df.empty:
        skip_reason = "SKIPPED_NO_UNSCORED_PREDICTIONS"
        print(skip_reason)
    else:
        results_df = load_actual_results(conn, args.target_season)
        scored_candidates = join_predictions_to_results(pred_df, results_df)
        if scored_candidates.empty:
            skip_reason = "SKIPPED_NO_COMPLETED_RESULTS_FOR_PREDICTIONS"
            print(skip_reason)
        else:
            scored_df = score_prediction_rows(scored_candidates)
            scored_count = update_prediction_scores(conn, scored_df)
            metrics = compute_health_metrics(
                scored_df,
                target_season=args.target_season,
                target_gameweek=args.target_gameweek,
            )
            write_health_log(conn, metrics)
            health_log_rows_written = 1

    after_counts = capture_watched_table_counts(conn)
    assert_counts_unchanged_except_scoring(before_counts, after_counts)
    print_watched_count_comparison(before_counts, after_counts)
    print(
        "Scoring run summary: "
        f"unscored_predictions={len(pred_df)}, "
        f"scored_predictions={scored_count}, "
        f"health_log_rows_written={health_log_rows_written}"
    )
    if skip_reason:
        print(f"Skip reason: {skip_reason}")
    print("No fake predictions, results, or scores were created.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production P4A prediction scorer")
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    return parser.parse_args()


def print_scoring_table_counts(conn) -> None:
    with conn.connect() as db_conn:
        for table_name in [PREDICTION_RUN_TABLE, PREDICTION_TABLE, HEALTH_TABLE]:
            count = int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
            print(f"{table_name}: {count}")


def print_watched_count_comparison(before_counts, after_counts) -> None:
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        print(f"- {table_name}: {before_counts.get(table_name)} -> {after_counts.get(table_name)}")


def _verify_health_table(conn) -> None:
    _verify_table_columns(conn, HEALTH_TABLE, HEALTH_REQUIRED_COLUMNS)


def _verify_table_columns(conn, table_name: str, required_columns: list[str]) -> None:
    with conn.connect() as db_conn:
        if not _table_exists(db_conn, table_name):
            raise RuntimeError(f"{table_name} table does not exist")
        existing_columns = [
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
    missing = sorted(set(required_columns) - set(existing_columns))
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


def _validate_result_rows(results_df: pandas.DataFrame) -> None:
    if results_df.empty:
        return
    errors: list[str] = []
    if results_df[["target_season", "match_date", "home_team", "away_team", "actual_result"]].isna().any().any():
        errors.append("actual result rows contain null required fields")
    invalid_results = sorted(set(results_df["actual_result"]) - set(LABELS))
    if invalid_results:
        errors.append(f"invalid actual result labels: {invalid_results}")
    if (results_df["home_team"] == results_df["away_team"]).any():
        errors.append("actual result rows contain home_team = away_team")
    key_columns = ["target_season", "match_date", "home_team", "away_team"]
    duplicate_keys = int(results_df.duplicated(key_columns).sum())
    if duplicate_keys:
        errors.append(f"duplicate actual result key count: {duplicate_keys}")
    if errors:
        raise ValueError("Actual result validation failed: " + "; ".join(errors))


def _validate_scoring_input(scored: pandas.DataFrame) -> None:
    required = [
        "prediction_id",
        "actual_result",
        "argmax_prediction",
        "overlay_prediction",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "target_season",
        "model_name",
    ]
    missing = sorted(set(required) - set(scored.columns))
    if missing:
        raise ValueError(f"Missing scoring column(s): {missing}")
    invalid_actuals = sorted(set(scored["actual_result"].dropna()) - set(LABELS))
    invalid_argmax = sorted(set(scored["argmax_prediction"].dropna()) - set(LABELS))
    invalid_overlay = sorted(set(scored["overlay_prediction"].dropna()) - set(LABELS))
    errors: list[str] = []
    if invalid_actuals:
        errors.append(f"invalid actual_result labels: {invalid_actuals}")
    if invalid_argmax:
        errors.append(f"invalid argmax_prediction labels: {invalid_argmax}")
    if invalid_overlay:
        errors.append(f"invalid overlay_prediction labels: {invalid_overlay}")
    if scored[required].isna().any().any():
        null_columns = sorted(column for column in required if scored[column].isna().any())
        errors.append(f"null scoring value(s): {null_columns}")
    _normalized_probability_matrix(scored)
    if errors:
        raise ValueError("Scoring input validation failed: " + "; ".join(errors))


def _normalized_probability_matrix(df: pandas.DataFrame) -> numpy.ndarray:
    columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    probabilities = df[columns].to_numpy(dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError("Probability matrix shape does not match labels")
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


def _draw_metrics(predicted: pandas.Series, actual: pandas.Series) -> dict[str, float]:
    predicted_draw = predicted == "D"
    actual_draw = actual == "D"
    true_positive = int((predicted_draw & actual_draw).sum())
    false_positive = int((predicted_draw & ~actual_draw).sum())
    false_negative = int((~predicted_draw & actual_draw).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "draw_recall": float(recall),
        "draw_precision": float(precision),
        "draw_f1": float(f1),
    }


def _label_rate(values: pandas.Series, label: str) -> float:
    if len(values) == 0:
        return 0.0
    return float((values == label).mean())


def _single_or_multiple(values: pandas.Series) -> str:
    unique_values = sorted(str(value) for value in values.dropna().unique())
    if len(unique_values) == 1:
        return unique_values[0]
    return "multiple"


def _empty_scored_frame(source_df: pandas.DataFrame) -> pandas.DataFrame:
    columns = list(source_df.columns)
    for column in ["actual_result", "was_correct_argmax", "was_correct_overlay", "scored_at"]:
        if column not in columns:
            columns.append(column)
    return pandas.DataFrame(columns=columns)


def _nullable_int(value):
    if value is None or pandas.isna(value):
        return None
    return int(value)


def _db_safe_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _db_safe_value(value) for key, value in row.items()}


def _db_safe_value(value):
    if value is None:
        return None
    if isinstance(value, pandas.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)
    if pandas.isna(value):
        return None
    if isinstance(value, numpy.generic):
        return value.item()
    return value


def _format_mapping(values: dict[str, Any]) -> str:
    lines = []
    for key, value in values.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
