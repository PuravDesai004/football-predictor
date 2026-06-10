from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "production_logistic_elo_v3"
TARGET_SEASON = "2026-27"
TRAINING_SEASONS = "2021-22 through 2025-26"
EXPECTED_FEATURE_COUNT = 32
EXPECTED_DRAW_THRESHOLD = 0.30
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
ARTIFACTS = {
    "Model": MODEL_DIR / "production_logistic_elo_v3.pkl",
    "Features": MODEL_DIR / "production_features_v3.json",
    "Draw Threshold": MODEL_DIR / "production_draw_threshold_v3.json",
    "Metadata": MODEL_DIR / "production_metadata_v3.json",
}
REPORTS = {
    "Tier 3 Experiment Summary": PROJECT_ROOT / "docs" / "tier3_experiment_summary.md",
    "Final Holdout Report": PROJECT_ROOT / "docs" / "tier3_final_holdout_report.md",
    "Final Error Analysis": PROJECT_ROOT / "docs" / "tier3_final_error_analysis.md",
}
COUNT_TABLES = {
    "Upcoming Features": "production_upcoming_match_features_v3",
    "Predictions": "production_match_predictions",
    "Ingestion Runs": "production_ingestion_runs",
    "Health Logs": "production_model_health_log",
}
PRODUCTION_TABLES = [
    "production_ingestion_runs",
    "production_data_freshness",
    "production_fpl_bootstrap_snapshots",
    "production_fpl_fixture_snapshots",
    "production_team_name_mapping",
    "production_upcoming_match_features_v3",
    "production_prediction_runs",
    "production_match_predictions",
    "production_model_health_log",
    "elo_current_v3",
]
HOLDOUT_ARGMAX = {
    "Accuracy": "0.4868",
    "Log Loss": "1.0601",
    "Brier": "0.6372",
    "Draw F1": "0.0000",
}
HOLDOUT_OVERLAY = {
    "Accuracy": "0.4684",
    "Log Loss": "1.0601",
    "Brier": "0.6372",
    "Draw F1": "0.1159",
}
ACTUAL_HOLDOUT_DISTRIBUTION = {"Home": 162, "Draw": 104, "Away": 114}
ARGMAX_HOLDOUT_DISTRIBUTION = {"Home": 250, "Draw": 1, "Away": 129}
OVERLAY_HOLDOUT_DISTRIBUTION = {"Home": 229, "Draw": 34, "Away": 117}


st.set_page_config(
    page_title="Production Football Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_db_connection():
    try:
        database_url = None
        try:
            if "DATABASE_URL" in st.secrets:
                database_url = str(st.secrets["DATABASE_URL"])
        except Exception:
            database_url = None

        if database_url is None:
            database_url = os.getenv("DATABASE_URL")

        if database_url:
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            make_url(database_url)
            connect_args = {"connect_timeout": 5}
            if "sslmode" not in database_url.lower():
                connect_args["sslmode"] = "require"
            engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
        else:
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_pass = os.getenv("DB_PASS")
            if not all([db_host, db_port, db_name, db_user, db_pass]):
                return None
            if db_host.lower() == "localhost":
                db_host = "127.0.0.1"
            engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}",
                connect_args={"connect_timeout": 5},
                pool_pre_ping=True,
            )

        with engine.connect():
            pass
        return engine
    except Exception:
        return None


def safe_table_count(conn, table_name):
    if conn is None:
        return None
    if table_name not in PRODUCTION_TABLES:
        return None
    try:
        with conn.connect() as db_conn:
            if not _table_exists(db_conn, table_name):
                return None
            return int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
    except Exception:
        return None


def load_latest_ingestion_run(conn):
    return _load_latest_row(conn, "production_ingestion_runs", "run_id")


def load_latest_prediction_run(conn):
    return _load_latest_row(conn, "production_prediction_runs", "prediction_run_id")


def load_data_freshness(conn):
    if conn is None:
        return pd.DataFrame()
    query = """
        SELECT
            source_name,
            last_successful_run_id,
            last_successful_update_at,
            latest_event_id,
            latest_deadline_time,
            latest_completed_match_date,
            latest_error_message,
            updated_at
        FROM production_data_freshness
        ORDER BY source_name
    """
    return _read_dataframe(conn, "production_data_freshness", query)


def load_prediction_rows(conn):
    if conn is None:
        return pd.DataFrame()
    query = """
        SELECT
            prediction_id,
            prediction_run_id,
            target_season,
            target_gameweek,
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
            draw_risk_flag,
            confidence,
            prediction_created_at,
            actual_result,
            was_correct_argmax,
            was_correct_overlay,
            scored_at
        FROM production_match_predictions
        ORDER BY match_date, kickoff_time, prediction_id
        LIMIT 100
    """
    return _read_dataframe(conn, "production_match_predictions", query)


def load_health_logs(conn):
    if conn is None:
        return pd.DataFrame()
    query = """
        SELECT
            health_log_id,
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
            notes,
            created_at
        FROM production_model_health_log
        ORDER BY computed_at DESC, health_log_id DESC
        LIMIT 100
    """
    return _read_dataframe(conn, "production_model_health_log", query)


def check_artifacts():
    results: dict[str, dict[str, Any]] = {}
    for label, path in ARTIFACTS.items():
        results[label] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "details": "",
        }

    feature_path = ARTIFACTS["Features"]
    if feature_path.exists():
        try:
            payload = json.loads(feature_path.read_text(encoding="utf-8"))
            features = payload.get("features")
            results["Features"]["details"] = (
                f"{len(features)} features" if isinstance(features, list) else "invalid payload"
            )
        except Exception:
            results["Features"]["details"] = "could not read metadata"

    threshold_path = ARTIFACTS["Draw Threshold"]
    if threshold_path.exists():
        try:
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = payload.get("selected_draw_threshold", payload.get("draw_threshold"))
            results["Draw Threshold"]["details"] = f"{float(threshold):.2f}"
        except Exception:
            results["Draw Threshold"]["details"] = "could not read metadata"

    metadata_path = ARTIFACTS["Metadata"]
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_name = payload.get("model_name", "unknown")
            feature_count = payload.get("feature_count", "unknown")
            results["Metadata"]["details"] = (
                f"model={model_name}, feature_count={feature_count}"
            )
        except Exception:
            results["Metadata"]["details"] = "could not read metadata"

    return results


def render_status_badge(status):
    normalized = str(status or "unknown").lower()
    color = {
        "success": "#16803c",
        "passed": "#16803c",
        "available": "#16803c",
        "partial": "#9a6700",
        "skipped": "#57606a",
        "unavailable": "#9a6700",
        "failed": "#cf222e",
        "error": "#cf222e",
        "missing": "#cf222e",
        "unknown": "#57606a",
    }.get(normalized, "#57606a")
    st.markdown(
        (
            f"<span style='display:inline-block;padding:0.18rem 0.55rem;"
            f"border-radius:999px;background:{color};color:white;"
            f"font-size:0.82rem;font-weight:600;'>{normalized}</span>"
        ),
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, help_text=None):
    st.metric(label=label, value="-" if value is None else value, help=help_text)


def load_markdown_file(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def main() -> None:
    st.title("Production Premier League Model Dashboard")
    st.caption("Read-only local dashboard for production model and weekly pipeline status.")

    conn = get_db_connection()
    artifact_results = check_artifacts()
    counts = {label: safe_table_count(conn, table) for label, table in COUNT_TABLES.items()}
    latest_ingestion = load_latest_ingestion_run(conn)
    latest_prediction = load_latest_prediction_run(conn)
    freshness_df = load_data_freshness(conn)
    predictions_df = load_prediction_rows(conn)
    health_df = load_health_logs(conn)

    if conn is None:
        st.error("Database connection unavailable. Check local environment configuration.")

    tabs = st.tabs(
        [
            "Overview",
            "Pipeline Status",
            "Predictions",
            "Model Health",
            "Reports",
            "How To Run",
        ]
    )

    with tabs[0]:
        render_overview(artifact_results, counts, freshness_df)

    with tabs[1]:
        render_pipeline_status(latest_ingestion, latest_prediction, freshness_df, counts)

    with tabs[2]:
        render_predictions(predictions_df)

    with tabs[3]:
        render_model_health(health_df)

    with tabs[4]:
        render_reports()

    with tabs[5]:
        render_how_to_run()


def render_overview(artifact_results, counts, freshness_df) -> None:
    st.subheader("Production Overview")
    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Model", MODEL_NAME)
    with cols[1]:
        render_metric_card("Training Seasons", TRAINING_SEASONS)
    with cols[2]:
        render_metric_card("Feature Count", EXPECTED_FEATURE_COUNT)
    with cols[3]:
        render_metric_card("Draw Threshold", f"{EXPECTED_DRAW_THRESHOLD:.2f}")

    st.divider()
    st.subheader("Artifact Checks")
    artifact_rows = []
    for label, result in artifact_results.items():
        artifact_rows.append(
            {
                "artifact": label,
                "exists": "yes" if result["exists"] else "no",
                "path": result["path"],
                "details": result["details"],
            }
        )
    st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
    if not all(result["exists"] for result in artifact_results.values()):
        st.error("One or more production artifacts are missing.")

    count_cols = st.columns(4)
    for index, (label, value) in enumerate(counts.items()):
        with count_cols[index]:
            render_metric_card(label, value)

    if counts.get("Predictions") == 0:
        st.warning(
            "No production predictions yet. Run the weekly pipeline after upcoming fixtures are available."
        )

    if _freshness_has_warning(freshness_df):
        st.warning("One or more production data sources are stale or unavailable.")


def render_pipeline_status(latest_ingestion, latest_prediction, freshness_df, counts) -> None:
    st.subheader("Pipeline Status")
    left, right = st.columns(2)

    with left:
        st.markdown("**Latest ingestion run**")
        if latest_ingestion:
            render_status_badge(latest_ingestion.get("run_status"))
            st.json(_compact_mapping(latest_ingestion), expanded=False)
        else:
            st.info("No ingestion runs found.")

    with right:
        st.markdown("**Latest prediction run**")
        if latest_prediction:
            render_status_badge(latest_prediction.get("run_status"))
            st.json(_compact_mapping(latest_prediction), expanded=False)
        else:
            st.info("No prediction runs found.")

    st.subheader("Data Freshness")
    if freshness_df.empty:
        st.info("No data freshness rows found.")
    else:
        freshness_display = freshness_df.copy()
        st.dataframe(freshness_display, use_container_width=True, hide_index=True)
        for row in freshness_df.to_dict(orient="records"):
            if row.get("latest_error_message"):
                st.warning(f"{row['source_name']}: {row['latest_error_message']}")

    st.subheader("Derived pipeline state")
    status_rows = [
        {
            "area": "FPL snapshots",
            "status": _source_status(latest_ingestion, ["fpl_bootstrap_status", "fpl_fixtures_status"]),
        },
        {
            "area": "football-data",
            "status": (latest_ingestion or {}).get("football_data_status", "unknown"),
        },
        {
            "area": "Understat",
            "status": (latest_ingestion or {}).get("understat_status", "unknown"),
        },
        {
            "area": "Upcoming features",
            "status": "skipped" if counts.get("Upcoming Features") == 0 else "success",
        },
        {
            "area": "Predictions",
            "status": (latest_prediction or {}).get("run_status", "unknown"),
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)


def render_predictions(predictions_df) -> None:
    st.subheader("Production Predictions")
    if predictions_df.empty:
        st.info(
            "No production predictions yet. Run the weekly pipeline after upcoming fixtures are available."
        )
        return

    display_df = predictions_df.copy()
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win", "confidence"]
    for column in probability_columns:
        if column in display_df:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_model_health(health_df) -> None:
    st.subheader("Final Tier 3 Holdout Metrics")
    argmax_cols = st.columns(4)
    for index, (label, value) in enumerate(HOLDOUT_ARGMAX.items()):
        with argmax_cols[index]:
            render_metric_card(f"Argmax {label}", value)
    overlay_cols = st.columns(4)
    for index, (label, value) in enumerate(HOLDOUT_OVERLAY.items()):
        with overlay_cols[index]:
            render_metric_card(f"Overlay {label}", value)

    st.warning(
        "Known final-holdout weakness: draw underprediction, home bias, and high-confidence wrong predictions."
    )

    distribution_df = pd.DataFrame(
        {
            "Actual": ACTUAL_HOLDOUT_DISTRIBUTION,
            "Argmax Predicted": ARGMAX_HOLDOUT_DISTRIBUTION,
            "Overlay Predicted": OVERLAY_HOLDOUT_DISTRIBUTION,
        }
    ).reset_index(names="class")
    st.dataframe(distribution_df, use_container_width=True, hide_index=True)

    st.subheader("Production Health Logs")
    if health_df.empty:
        st.info("No production model health logs yet. Scoring will populate this after predictions meet completed results.")
        return
    st.dataframe(health_df, use_container_width=True, hide_index=True)


def render_reports() -> None:
    st.subheader("Tier 3 Reports")
    for title, path in REPORTS.items():
        with st.expander(title, expanded=False):
            markdown = load_markdown_file(path)
            if markdown:
                st.markdown(markdown)
            else:
                st.warning(f"Report not found: {path.relative_to(PROJECT_ROOT)}")


def render_how_to_run() -> None:
    st.subheader("Local Commands")
    commands = [
        "python src/production/run_weekly_pipeline.py --target-season 2026-27",
        "python src/production/weekly_ingest.py --target-season 2026-27",
        "python src/production/build_upcoming_features.py --target-season 2026-27",
        "python src/production/predict_production_matches.py --target-season 2026-27",
        "python src/production/score_predictions.py --target-season 2026-27",
    ]
    for command in commands:
        st.code(command, language="powershell")
    st.info("This dashboard is read-only. It does not run the weekly pipeline automatically.")


def _read_dataframe(conn, table_name: str, query: str) -> pd.DataFrame:
    try:
        with conn.connect() as db_conn:
            if not _table_exists(db_conn, table_name):
                return pd.DataFrame()
        return pd.read_sql(text(query), conn)
    except Exception:
        return pd.DataFrame()


def _load_latest_row(conn, table_name: str, order_column: str) -> dict | None:
    if conn is None:
        return None
    if table_name not in PRODUCTION_TABLES:
        return None
    try:
        with conn.connect() as db_conn:
            if not _table_exists(db_conn, table_name):
                return None
            row = db_conn.execute(
                text(
                    f"""
                    SELECT *
                    FROM {table_name}
                    ORDER BY {order_column} DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
            return dict(row) if row else None
    except Exception:
        return None


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


def _freshness_has_warning(freshness_df: pd.DataFrame) -> bool:
    if freshness_df.empty:
        return True
    if "latest_error_message" in freshness_df and freshness_df["latest_error_message"].notna().any():
        return True
    if "last_successful_update_at" in freshness_df and freshness_df["last_successful_update_at"].isna().any():
        return True
    return False


def _source_status(latest_ingestion: dict | None, keys: list[str]) -> str:
    if not latest_ingestion:
        return "unknown"
    values = {str(latest_ingestion.get(key, "unknown")).lower() for key in keys}
    if values == {"success"}:
        return "success"
    if "failed" in values:
        return "failed"
    if "unavailable" in values or "skipped" in values:
        return "partial"
    return "unknown"


def _compact_mapping(row: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            compact[key] = value.isoformat()
        else:
            compact[key] = value
    return compact


if __name__ == "__main__":
    main()
