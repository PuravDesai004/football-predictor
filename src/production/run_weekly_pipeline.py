from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_SEASON = "2026-27"
PYTHON_EXECUTABLE = sys.executable
PRODUCTION_SCRIPTS = {
    "ingest": PROJECT_ROOT / "src" / "production" / "weekly_ingest.py",
    "fpl_history": PROJECT_ROOT / "src" / "production" / "ingest_fpl_gameweek_v3.py",
    "build_features": PROJECT_ROOT / "src" / "production" / "build_upcoming_features.py",
    "predict": PROJECT_ROOT / "src" / "production" / "predict_production_matches.py",
    "score": PROJECT_ROOT / "src" / "production" / "score_predictions.py",
    "fpl_predictions": PROJECT_ROOT / "src" / "production" / "run_fpl_predictions_v3.py",
}
PIPELINE_STAGES = ["ingest", "fpl_history", "build_features", "predict", "score", "fpl_predictions"]
DEFAULT_FPL_ARTIFACT_DIR = Path(
    os.getenv("FPL_V3_ARTIFACT_DIR", str(PROJECT_ROOT / "data" / "production_artifacts"))
)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402


SKIP_TOKENS = [
    "SKIPPED_",
    "SKIP:",
    "DRY_RUN_NO_PREDICTIONS_WRITTEN",
    "Upcoming fixtures found: 0",
    "Wrote 0 upcoming feature rows",
    "SKIPPED_FPL_ARTIFACT_UNAVAILABLE",
    "SKIPPED_NO_UPCOMING_FIXTURES",
    "SKIPPED_STALE_FIXTURE_SNAPSHOT",
    "SKIPPED_NO_COMPLETED_GAMEWEEK",
    "SKIPPED_NO_FPL_SNAPSHOTS",
    "SKIPPED_FPL_LIVE_UNAVAILABLE",
]


def run_command(command, stage_name) -> dict:
    print(f"=== Stage: {stage_name} ===")
    print("Command: " + _format_command(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())

    status = _classify_stage_status(stage_name, completed.returncode, output)
    print(f"Stage {stage_name} status: {status} (returncode={completed.returncode})")
    return {
        "stage": stage_name,
        "command": command,
        "returncode": int(completed.returncode),
        "status": status,
        "output": output,
    }


def run_ingest(
    target_season,
    target_gameweek=None,
    skip_fpl=False,
    skip_football_data=False,
    skip_understat=False,
) -> dict:
    command = _base_command("ingest", target_season, target_gameweek)
    if skip_fpl:
        command.append("--skip-fpl")
    if skip_football_data:
        command.append("--skip-football-data")
    if skip_understat:
        command.append("--skip-understat")
    return run_command(command, "ingest")


def run_build_features(target_season, target_gameweek=None, replace=False) -> dict:
    command = _base_command("build_features", target_season, target_gameweek)
    if replace:
        command.append("--replace")
    return run_command(command, "build_features")


def run_fpl_history(target_season, target_gameweek=None) -> dict:
    command = _base_command("fpl_history", target_season, target_gameweek)
    return run_command(command, "fpl_history")


def run_predict(target_season, target_gameweek=None) -> dict:
    command = _base_command("predict", target_season, target_gameweek)
    return run_command(command, "predict")


def run_score(target_season, target_gameweek=None) -> dict:
    command = _base_command("score", target_season, target_gameweek)
    return run_command(command, "score")


def run_fpl_predictions(target_season, target_gameweek=None, artifact_dir=None, dry_run=False) -> dict:
    artifact_dir = Path(artifact_dir or DEFAULT_FPL_ARTIFACT_DIR)
    model_path = artifact_dir / "fpl_points_v3_candidate.pkl"
    features_path = artifact_dir / "fpl_points_v3_candidate_features.json"
    if not model_path.exists() or not features_path.exists():
        reason = f"SKIPPED_FPL_ARTIFACT_UNAVAILABLE: {artifact_dir}"
        print(reason)
        return _stage_result("fpl_predictions", "skipped", reason)
    command = _base_command("fpl_predictions", target_season, target_gameweek)
    command.extend(["--artifact-dir", str(artifact_dir)])
    if dry_run:
        command.append("--dry-run")
    return run_command(command, "fpl_predictions")


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_pipeline_status(conn, target_season, target_gameweek=None) -> dict:
    status: dict[str, Any] = {
        "target_season": target_season,
        "target_gameweek": target_gameweek,
    }
    with conn.connect() as db_conn:
        status["latest_ingestion_run"] = _latest_row(
            db_conn,
            "production_ingestion_runs",
            "run_id",
            target_season,
            target_gameweek,
        )
        status["latest_prediction_run"] = _latest_row(
            db_conn,
            "production_prediction_runs",
            "prediction_run_id",
            target_season,
            target_gameweek,
        )
        status["latest_health_log"] = _latest_row(
            db_conn,
            "production_model_health_log",
            "health_log_id",
            target_season,
            target_gameweek,
        )
        status["upcoming_feature_row_count"] = _count_rows(
            db_conn,
            "production_upcoming_match_features_v3",
            target_season,
            target_gameweek,
        )
        status["prediction_row_count"] = _count_rows(
            db_conn,
            "production_match_predictions",
            target_season,
            target_gameweek,
        )
        status["unscored_prediction_count"] = _count_rows(
            db_conn,
            "production_match_predictions",
            target_season,
            target_gameweek,
            extra_where="scored_at IS NULL",
        )
        status["scored_prediction_count"] = _count_rows(
            db_conn,
            "production_match_predictions",
            target_season,
            target_gameweek,
            extra_where="scored_at IS NOT NULL",
        )
        status["model_health_log_row_count"] = _count_rows(
            db_conn,
            "production_model_health_log",
            target_season,
            target_gameweek,
        )
        status["fpl_prediction_row_count"] = _count_rows(
            db_conn,
            "fpl_player_predictions_v3",
            target_season,
            target_gameweek,
        )
        status["fpl_optimizer_row_count"] = _count_rows(
            db_conn,
            "fpl_optimizer_outputs_v3",
            target_season,
            target_gameweek,
        )
        status["fpl_gameweek_snapshot_row_count"] = _count_rows(
            db_conn,
            "production_fpl_gameweek_snapshots_v3",
            target_season,
            target_gameweek,
        )
    return status


def print_pipeline_summary(stage_results, status) -> None:
    print("=== Weekly Production Pipeline Summary ===")
    print("Stage results:")
    for stage in PIPELINE_STAGES:
        result = stage_results.get(stage)
        if result is None:
            result = _stage_result(stage, "skipped", "NOT_RUN")
        print(
            f"- {stage}: {result['status']} "
            f"(returncode={result.get('returncode', 0)})"
        )
        reason = _extract_reason(result.get("output", ""))
        if reason:
            print(f"  reason: {reason}")

    latest_ingestion = status.get("latest_ingestion_run")
    latest_prediction = status.get("latest_prediction_run")
    latest_health = status.get("latest_health_log")
    print("Latest ingestion status:")
    if latest_ingestion:
        print(
            f"- run_id={latest_ingestion.get('run_id')}, "
            f"run_status={latest_ingestion.get('run_status')}"
        )
        print(
            "- sources: "
            f"fpl_bootstrap={latest_ingestion.get('fpl_bootstrap_status')}, "
            f"fpl_fixtures={latest_ingestion.get('fpl_fixtures_status')}, "
            f"football_data={latest_ingestion.get('football_data_status')}, "
            f"understat={latest_ingestion.get('understat_status')}"
        )
    else:
        print("- none")

    print(f"Upcoming feature row count: {status.get('upcoming_feature_row_count')}")
    print("Latest prediction status:")
    if latest_prediction:
        print(
            f"- prediction_run_id={latest_prediction.get('prediction_run_id')}, "
            f"run_status={latest_prediction.get('run_status')}, "
            f"rows_loaded={latest_prediction.get('rows_loaded')}, "
            f"rows_predicted={latest_prediction.get('rows_predicted')}"
        )
        if latest_prediction.get("error_message"):
            print(f"- message={latest_prediction.get('error_message')}")
    else:
        print("- none")

    print(f"Prediction row count: {status.get('prediction_row_count')}")
    print(f"Unscored prediction count: {status.get('unscored_prediction_count')}")
    print(f"Scored prediction count: {status.get('scored_prediction_count')}")
    print(f"Model health log row count: {status.get('model_health_log_row_count')}")
    print(f"FPL prediction row count: {status.get('fpl_prediction_row_count')}")
    print(f"FPL optimizer row count: {status.get('fpl_optimizer_row_count')}")
    print(f"FPL gameweek snapshot row count: {status.get('fpl_gameweek_snapshot_row_count')}")
    print("Latest scoring status:")
    if latest_health:
        print(
            f"- health_log_id={latest_health.get('health_log_id')}, "
            f"scored_count={latest_health.get('scored_count')}, "
            f"computed_at={latest_health.get('computed_at')}"
        )
    else:
        print("- none")
    print(f"Final pipeline status: {status.get('final_pipeline_status')}")


def main() -> None:
    args = parse_args()
    stage_results: dict[str, dict] = {}

    if args.skip_ingest:
        stage_results["ingest"] = _skipped_stage("ingest", "SKIPPED_BY_CLI")
    else:
        stage_results["ingest"] = run_ingest(
            args.target_season,
            target_gameweek=args.target_gameweek,
            skip_fpl=args.skip_fpl,
            skip_football_data=args.skip_football_data,
            skip_understat=args.skip_understat,
        )
        if stage_results["ingest"]["status"] == "failed":
            _finish_failed(stage_results, args)

    if args.skip_fpl_history:
        stage_results["fpl_history"] = _skipped_stage("fpl_history", "SKIPPED_BY_CLI")
    else:
        stage_results["fpl_history"] = run_fpl_history(
            args.target_season,
            target_gameweek=args.target_gameweek,
        )
        if stage_results["fpl_history"]["status"] == "failed":
            _finish_failed(stage_results, args)

    if args.skip_build_features:
        stage_results["build_features"] = _skipped_stage(
            "build_features",
            "SKIPPED_BY_CLI",
        )
    else:
        stage_results["build_features"] = run_build_features(
            args.target_season,
            target_gameweek=args.target_gameweek,
            replace=args.replace_features,
        )
        if stage_results["build_features"]["status"] == "failed":
            _finish_failed(stage_results, args)

    if args.skip_predict:
        stage_results["predict"] = _skipped_stage("predict", "SKIPPED_BY_CLI")
    else:
        stage_results["predict"] = run_predict(
            args.target_season,
            target_gameweek=args.target_gameweek,
        )
    if stage_results["predict"]["status"] == "failed":
            _finish_failed(stage_results, args)

    if args.skip_score:
        stage_results["score"] = _skipped_stage("score", "SKIPPED_BY_CLI")
    else:
        stage_results["score"] = run_score(
            args.target_season,
            target_gameweek=args.target_gameweek,
        )
    if stage_results["score"]["status"] == "failed":
            _finish_failed(stage_results, args)

    if args.skip_fpl_predictions:
        stage_results["fpl_predictions"] = _skipped_stage(
            "fpl_predictions", "SKIPPED_BY_CLI"
        )
    else:
        stage_results["fpl_predictions"] = run_fpl_predictions(
            args.target_season,
            target_gameweek=args.target_gameweek,
            artifact_dir=args.fpl_artifact_dir,
            dry_run=args.dry_run_fpl,
        )
        if stage_results["fpl_predictions"]["status"] == "failed":
            _finish_failed(stage_results, args)

    final_status = _final_pipeline_status(stage_results)
    conn = get_db_connection()
    status = load_pipeline_status(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
    )
    status["final_pipeline_status"] = final_status
    print_pipeline_summary(stage_results, status)
    print("No fake data, predictions, results, or scores were created.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production P5A weekly pipeline runner")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-fpl-history", action="store_true")
    parser.add_argument("--skip-build-features", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-fpl-predictions", action="store_true")
    parser.add_argument("--skip-fpl", action="store_true")
    parser.add_argument("--skip-football-data", action="store_true")
    parser.add_argument("--skip-understat", action="store_true")
    parser.add_argument("--replace-features", action="store_true")
    parser.add_argument(
        "--fpl-artifact-dir",
        type=Path,
        default=DEFAULT_FPL_ARTIFACT_DIR,
    )
    parser.add_argument("--dry-run-fpl", action="store_true")
    return parser.parse_args()


def _base_command(stage_name: str, target_season: str, target_gameweek=None) -> list[str]:
    command = [
        PYTHON_EXECUTABLE,
        str(PRODUCTION_SCRIPTS[stage_name]),
        "--target-season",
        target_season,
    ]
    if target_gameweek is not None:
        command.extend(["--target-gameweek", str(int(target_gameweek))])
    return command


def _classify_stage_status(stage_name: str, returncode: int, output: str) -> str:
    if returncode != 0:
        return "failed"
    if stage_name == "ingest" and "PARTIAL:" in output:
        return "partial"
    if any(token in output for token in SKIP_TOKENS):
        return "skipped"
    return "success"


def _final_pipeline_status(stage_results: dict[str, dict]) -> str:
    statuses = [stage_results.get(stage, {}).get("status") for stage in PIPELINE_STAGES]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status in {"partial", "skipped"} for status in statuses):
        return "completed_with_skips"
    return "completed"


def _finish_failed(stage_results: dict[str, dict], args: argparse.Namespace) -> None:
    conn = get_db_connection()
    status = load_pipeline_status(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
    )
    status["final_pipeline_status"] = "failed"
    print_pipeline_summary(stage_results, status)
    raise SystemExit(1)


def _skipped_stage(stage_name: str, reason: str) -> dict:
    print(f"Stage {stage_name} skipped: {reason}")
    return _stage_result(stage_name, "skipped", reason)


def _stage_result(stage_name: str, status: str, output: str) -> dict:
    return {
        "stage": stage_name,
        "command": [],
        "returncode": 0,
        "status": status,
        "output": output,
    }


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


def _latest_row(
    db_conn,
    table_name: str,
    order_column: str,
    target_season: str,
    target_gameweek=None,
) -> dict | None:
    if not _table_exists(db_conn, table_name):
        return None
    where_clauses = ["target_season = :target_season"]
    params: dict[str, Any] = {"target_season": target_season}
    if target_gameweek is not None:
        where_clauses.append("target_gameweek = :target_gameweek")
        params["target_gameweek"] = int(target_gameweek)
    row = db_conn.execute(
        text(
            f"""
            SELECT *
            FROM {table_name}
            WHERE {" AND ".join(where_clauses)}
            ORDER BY {order_column} DESC
            LIMIT 1
            """
        ),
        params,
    ).mappings().first()
    return dict(row) if row else None


def _count_rows(
    db_conn,
    table_name: str,
    target_season: str,
    target_gameweek=None,
    extra_where: str | None = None,
) -> int | str:
    if not _table_exists(db_conn, table_name):
        return "MISSING"
    where_clauses = ["target_season = :target_season"]
    params: dict[str, Any] = {"target_season": target_season}
    if target_gameweek is not None:
        where_clauses.append("target_gameweek = :target_gameweek")
        params["target_gameweek"] = int(target_gameweek)
    if extra_where:
        where_clauses.append(extra_where)
    return int(
        db_conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {table_name}
                WHERE {" AND ".join(where_clauses)}
                """
            ),
            params,
        ).scalar_one()
    )


def _extract_reason(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("SKIPPED_")
            or stripped.startswith("SKIP:")
            or stripped.startswith("PARTIAL:")
            or stripped == "SKIPPED_BY_CLI"
            or "Upcoming fixtures found: 0" in stripped
        ):
            return stripped
    return None


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


if __name__ == "__main__":
    main()
