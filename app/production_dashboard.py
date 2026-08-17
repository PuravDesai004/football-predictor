from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from html import escape
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
FPL_PREDICTION_TABLE = "fpl_player_predictions_v3"
FPL_OPTIMIZER_TABLE = "fpl_optimizer_outputs_v3"
FPL_SNAPSHOT_TABLE = "production_fpl_gameweek_snapshots_v3"
FPL_TEAM_MAPPING_TABLE = "production_team_name_mapping"
FPL_POSITION_NAMES = {
    1: "Goalkeeper",
    2: "Defender",
    3: "Midfielder",
    4: "Forward",
}
PIPELINE_TIMEOUT_SECONDS = 600
PIPELINE_DEADLINE_GUARD_HOURS = 4
FPL_SNAPSHOT_STALE_HOURS = 48
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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111d;
            --panel: #0d1724;
            --panel-2: #111d2c;
            --border: rgba(148, 163, 184, 0.18);
            --muted: #94a3b8;
            --text: #f8fafc;
            --blue: #38bdf8;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
        }
        .stApp {
            background:
                linear-gradient(180deg, #07111d 0%, #0a1320 52%, #07111d 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        header[data-testid="stHeader"] {
            background: rgba(7, 17, 29, 0.88);
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        }
        div[data-testid="stTabs"] button {
            color: #cbd5e1;
            font-weight: 700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--blue);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        .fp-hero {
            border: 1px solid rgba(56, 189, 248, 0.20);
            background: linear-gradient(135deg, rgba(13, 23, 36, 0.96), rgba(17, 29, 44, 0.92));
            border-radius: 14px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        }
        .fp-hero-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .fp-eyebrow {
            color: var(--blue);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .fp-title {
            margin: 0;
            font-size: clamp(1.7rem, 3vw, 2.45rem);
            line-height: 1.08;
            font-weight: 850;
            letter-spacing: 0;
        }
        .fp-subtitle {
            max-width: 880px;
            color: #cbd5e1;
            margin-top: 0.65rem;
            font-size: 0.98rem;
            line-height: 1.55;
        }
        .fp-status-row {
            display: flex;
            gap: 0.55rem;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .fp-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0 1.25rem;
        }
        .fp-grid-3 {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .fp-card {
            background: rgba(13, 23, 36, 0.92);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            min-height: 106px;
        }
        .fp-card-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.55rem;
        }
        .fp-card-value {
            color: var(--text);
            font-size: 1.6rem;
            font-weight: 850;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }
        .fp-card-detail {
            color: #aebdd0;
            font-size: 0.86rem;
            margin-top: 0.55rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .fp-section {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.8rem 0 0.8rem;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        }
        .fp-section h2 {
            font-size: 1.25rem;
            margin: 0;
        }
        .fp-section p {
            color: var(--muted);
            margin: 0;
            font-size: 0.9rem;
        }
        .fp-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 800;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .fp-pill-success {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.13);
            border-color: rgba(34, 197, 94, 0.32);
        }
        .fp-pill-warning {
            color: #fde68a;
            background: rgba(245, 158, 11, 0.13);
            border-color: rgba(245, 158, 11, 0.32);
        }
        .fp-pill-muted {
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.12);
            border-color: rgba(148, 163, 184, 0.24);
        }
        .fp-pill-danger {
            color: #fecdd3;
            background: rgba(244, 63, 94, 0.13);
            border-color: rgba(244, 63, 94, 0.32);
        }
        .fp-alert {
            border: 1px solid rgba(245, 158, 11, 0.30);
            background: rgba(245, 158, 11, 0.10);
            color: #fde68a;
            border-radius: 12px;
            padding: 0.95rem 1rem;
            margin: 1rem 0;
            font-weight: 650;
        }
        .fp-muted-box {
            border: 1px dashed rgba(148, 163, 184, 0.24);
            background: rgba(15, 23, 42, 0.56);
            border-radius: 12px;
            padding: 1.2rem;
            color: #cbd5e1;
        }
        .fp-code-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.7rem;
        }
        @media (max-width: 1100px) {
            .fp-grid, .fp-grid-3 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .fp-grid, .fp-grid-3 {
                grid-template-columns: 1fr;
            }
            .fp-status-row {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def load_fpl_prediction_rows(conn):
    if conn is None:
        return pd.DataFrame()
    query = f"""
        WITH latest_target AS (
            SELECT target_season, target_gameweek
            FROM {FPL_PREDICTION_TABLE}
            GROUP BY target_season, target_gameweek
            ORDER BY MAX(created_at) DESC NULLS LAST,
                     target_season DESC,
                     target_gameweek DESC
            LIMIT 1
        )
        SELECT
            p.prediction_id,
            p.target_season,
            p.target_gameweek,
            p.player_id,
            p.player_name,
            p.team_name,
            p.position_id,
            p.predicted_points,
            p.availability_factor,
            p.expected_points,
            p.status,
            p.chance_of_playing,
            p.created_at
        FROM {FPL_PREDICTION_TABLE} p
        INNER JOIN latest_target t
            ON p.target_season = t.target_season
            AND p.target_gameweek = t.target_gameweek
        ORDER BY p.expected_points DESC, p.player_name, p.player_id
    """
    return _read_dataframe(conn, FPL_PREDICTION_TABLE, query)


def load_latest_fpl_optimizer(conn) -> dict[str, Any] | None:
    if conn is None:
        return None
    query = f"""
        SELECT
            target_season,
            target_gameweek,
            generated_at,
            budget_limit,
            squad_cost,
            squad_json,
            starting_xi_json,
            captain_player_id,
            vice_captain_player_id,
            objective_value,
            source_prediction_count
        FROM {FPL_OPTIMIZER_TABLE}
        ORDER BY generated_at DESC NULLS LAST,
                 target_season DESC,
                 target_gameweek DESC
        LIMIT 1
    """
    optimizer_df = _read_dataframe(conn, FPL_OPTIMIZER_TABLE, query)
    if optimizer_df.empty:
        return None
    return optimizer_df.iloc[0].to_dict()


def load_fpl_snapshot_status(conn):
    if conn is None:
        return pd.DataFrame()
    query = f"""
        SELECT
            target_season,
            gameweek,
            COUNT(*) AS player_rows,
            MAX(snapshot_time) AS snapshot_time
        FROM {FPL_SNAPSHOT_TABLE}
        GROUP BY target_season, gameweek
        ORDER BY target_season DESC, gameweek DESC
    """
    return _read_dataframe(conn, FPL_SNAPSHOT_TABLE, query)


def load_fpl_cold_start_teams(conn) -> list[str]:
    if conn is None:
        return []
    query = f"""
        SELECT DISTINCT fpl_team_name
        FROM {FPL_TEAM_MAPPING_TABLE}
        WHERE is_active = TRUE
          AND source LIKE '%:new_team_cold_start'
        ORDER BY fpl_team_name
    """
    mapping_df = _read_dataframe(conn, FPL_TEAM_MAPPING_TABLE, query)
    if mapping_df.empty or "fpl_team_name" not in mapping_df:
        return []
    return [
        str(team_name)
        for team_name in mapping_df["fpl_team_name"].dropna().tolist()
    ]


def load_latest_fpl_bootstrap_deadline(conn) -> dict[str, Any] | None:
    if conn is None:
        return None
    query = """
        SELECT
            run_id,
            MAX(deadline_time) AS latest_deadline,
            MAX(snapshot_time) AS latest_snapshot_time,
            COUNT(*) AS snapshot_rows
        FROM production_fpl_bootstrap_snapshots
        GROUP BY run_id
        ORDER BY run_id DESC
        LIMIT 1
    """
    deadline_df = _read_dataframe(
        conn,
        "production_fpl_bootstrap_snapshots",
        query,
    )
    if deadline_df.empty:
        return None
    return deadline_df.iloc[0].to_dict()


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
    st.markdown(status_badge_html(normalized), unsafe_allow_html=True)


def status_badge_html(status, label: str | None = None) -> str:
    normalized = str(status or "unknown").lower()
    css_class = {
        "success": "fp-pill-success",
        "passed": "fp-pill-success",
        "available": "fp-pill-success",
        "db connected": "fp-pill-success",
        "artifacts ready": "fp-pill-success",
        "db unavailable": "fp-pill-warning",
        "artifacts missing": "fp-pill-danger",
        "ready": "fp-pill-success",
        "partial": "fp-pill-warning",
        "waiting for fixtures": "fp-pill-warning",
        "skipped": "fp-pill-muted",
        "unavailable": "fp-pill-warning",
        "failed": "fp-pill-danger",
        "error": "fp-pill-danger",
        "missing": "fp-pill-danger",
        "unknown": "fp-pill-muted",
    }.get(normalized, "fp-pill-muted")
    display = label if label is not None else normalized
    return f"<span class='fp-pill {css_class}'>{escape(display)}</span>"


def render_metric_card(label, value, help_text=None):
    detail = f"<div class='fp-card-detail'>{escape(str(help_text))}</div>" if help_text else ""
    st.markdown(
        (
            "<div class='fp-card'>"
            f"<div class='fp-card-label'>{escape(str(label))}</div>"
            f"<div class='fp-card-value'>{escape('-' if value is None else str(value))}</div>"
            f"{detail}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_card_grid(cards: list[dict[str, Any]], columns: int = 4) -> None:
    grid_class = "fp-grid fp-grid-3" if columns == 3 else "fp-grid"
    html_parts = [f"<div class='{grid_class}'>"]
    for card in cards:
        status_html = card.get("status_html", "")
        detail = card.get("detail", "")
        html_parts.append(
            "<div class='fp-card'>"
            f"<div class='fp-card-label'>{escape(str(card.get('label', '')))}</div>"
            f"<div class='fp-card-value'>{escape(str(card.get('value', '-')))}</div>"
            f"{status_html}"
            f"<div class='fp-card-detail'>{escape(str(detail))}</div>"
            "</div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_section(title: str, caption: str = "") -> None:
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(
        f"<div class='fp-section'><h2>{escape(title)}</h2>{caption_html}</div>",
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, detail: str) -> None:
    st.markdown(
        (
            "<div class='fp-muted-box'>"
            f"<strong>{escape(title)}</strong><br>"
            f"<span>{escape(detail)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_run_card(title: str, row: dict[str, Any], id_key: str) -> None:
    status = row.get("run_status", row.get("status", "unknown"))
    run_id = row.get(id_key, "-")
    started_at = row.get("started_at") or row.get("created_at") or row.get("prediction_created_at")
    completed_at = row.get("completed_at") or row.get("finished_at") or row.get("updated_at")
    detail = []
    if started_at:
        detail.append(f"started: {_format_value(started_at)}")
    if completed_at:
        detail.append(f"updated: {_format_value(completed_at)}")
    if row.get("error_message"):
        detail.append(f"error: {row.get('error_message')}")
    render_card_grid(
        [
            {
                "label": title,
                "value": f"#{run_id}",
                "status_html": status_badge_html(status),
                "detail": " | ".join(detail) if detail else "latest run metadata",
            }
        ],
        columns=3,
    )
    with st.expander(f"{title} details", expanded=False):
        st.json(_compact_mapping(row), expanded=False)


def render_alert(message: str) -> None:
    st.markdown(f"<div class='fp-alert'>{escape(message)}</div>", unsafe_allow_html=True)


def format_bool(value: bool) -> str:
    return "available" if value else "missing"


def _format_value(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def load_markdown_file(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def render_operator_controls(conn, latest_fpl_deadline: dict[str, Any] | None) -> None:
    st.session_state.setdefault("pipeline_active", False)
    render_section(
        "Operator Controls",
        "Read-only refresh and guarded weekly pipeline execution.",
    )

    reasons = _pipeline_guard_reasons(conn, latest_fpl_deadline)
    control_left, control_right = st.columns([1, 2])
    with control_left:
        if st.button("Refresh Dashboard", key="refresh_dashboard_button"):
            st.rerun()

    with control_right:
        _render_pipeline_deadline(latest_fpl_deadline)
        if reasons:
            render_alert("Pipeline blocked: " + " ".join(reasons))
        else:
            st.success("Pipeline guard passed. The weekly pipeline is ready to run.")

        if st.button(
            "Run Weekly Pipeline",
            key="run_weekly_pipeline_button",
            disabled=bool(reasons),
        ):
            result = _execute_weekly_pipeline()
            st.session_state["pipeline_result"] = result
            st.rerun()

    _render_pipeline_result(st.session_state.get("pipeline_result"))


def _render_pipeline_deadline(deadline_info: dict[str, Any] | None) -> None:
    if not deadline_info:
        render_empty_state(
            "Latest FPL deadline unavailable",
            "The newest FPL bootstrap run has not returned a usable deadline.",
        )
        return

    deadline = _coerce_utc_timestamp(deadline_info.get("latest_deadline"))
    snapshot_time = _coerce_utc_timestamp(deadline_info.get("latest_snapshot_time"))
    snapshot_age = _timestamp_age_text(snapshot_time)
    render_card_grid(
        [
            {
                "label": "Latest FPL deadline",
                "value": _format_value(deadline) if deadline else "Unavailable",
                "detail": f"bootstrap run #{deadline_info.get('run_id', '-')}",
            },
            {
                "label": "Latest bootstrap snapshot",
                "value": _format_value(snapshot_time) if snapshot_time else "Unavailable",
                "detail": snapshot_age or "snapshot age unavailable",
            },
        ],
        columns=3,
    )


def _pipeline_guard_reasons(
    conn,
    deadline_info: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if st.session_state.get("pipeline_active", False):
        reasons.append("another pipeline run is active in this Streamlit session.")
    if conn is None:
        reasons.append("PostgreSQL is unavailable, so the latest FPL deadline cannot be verified.")
        return reasons
    if not deadline_info:
        reasons.append("no deadline exists in the newest FPL bootstrap run.")
        return reasons

    now = pd.Timestamp.now(tz="UTC")
    deadline = _coerce_utc_timestamp(deadline_info.get("latest_deadline"))
    snapshot_time = _coerce_utc_timestamp(deadline_info.get("latest_snapshot_time"))
    if deadline is None:
        reasons.append("no deadline exists in the newest FPL bootstrap run.")
    else:
        remaining = deadline - now
        if remaining <= pd.Timedelta(hours=PIPELINE_DEADLINE_GUARD_HOURS):
            if remaining.total_seconds() < 0:
                reasons.append(
                    "the FPL deadline has passed "
                    f"({_format_timedelta(-remaining)} ago; deadline {_format_value(deadline)})."
                )
            else:
                reasons.append(
                    "the FPL deadline is less than four hours away "
                    f"({_format_timedelta(remaining)} remaining; deadline {_format_value(deadline)})."
                )

    if snapshot_time is None:
        reasons.append("the latest FPL bootstrap snapshot has no timestamp and is treated as stale.")
    elif now - snapshot_time > pd.Timedelta(hours=FPL_SNAPSHOT_STALE_HOURS):
        reasons.append(
            "the latest FPL bootstrap snapshot is clearly stale "
            f"({_format_timedelta(now - snapshot_time)} old; threshold {FPL_SNAPSHOT_STALE_HOURS} hours)."
        )
    return reasons


def _execute_weekly_pipeline() -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "production" / "run_weekly_pipeline.py"),
        "--target-season",
        TARGET_SEASON,
        "--fpl-artifact-dir",
        str(PROJECT_ROOT / "data" / "production_artifacts"),
    ]
    st.session_state["pipeline_active"] = True
    result: dict[str, Any]
    with st.status("Running weekly pipeline…", expanded=True) as status:
        st.write("The guarded weekly pipeline is running. Dashboard queries will refresh when it finishes.")
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=PIPELINE_TIMEOUT_SECONDS,
                check=False,
            )
            stdout = _redact_pipeline_output(completed.stdout)
            stderr = _redact_pipeline_output(completed.stderr)
            full_output = _combine_pipeline_output(stdout, stderr)
            pipeline_status = _classify_pipeline_result(completed.returncode, full_output)
            result = {
                "status": pipeline_status,
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "full_output": full_output,
                "concise_output": _concise_pipeline_output(full_output),
                "completed_at": datetime.now(timezone.utc),
            }
            status.update(
                label=f"Weekly pipeline {pipeline_status}",
                state="error" if pipeline_status == "failure" else "complete",
                expanded=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _redact_pipeline_output(_output_text(exc.stdout))
            stderr = _redact_pipeline_output(_output_text(exc.stderr))
            full_output = _combine_pipeline_output(stdout, stderr)
            result = {
                "status": "failure",
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "full_output": _combine_pipeline_output(
                    full_output,
                    f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS} seconds.",
                ),
                "concise_output": _concise_pipeline_output(full_output),
                "completed_at": datetime.now(timezone.utc),
            }
            status.update(label="Weekly pipeline failed: timeout", state="error", expanded=False)
        except Exception as exc:
            message = _redact_pipeline_output(str(exc))
            result = {
                "status": "failure",
                "exit_code": None,
                "stdout": "",
                "stderr": message,
                "full_output": message,
                "concise_output": message,
                "completed_at": datetime.now(timezone.utc),
            }
            status.update(label="Weekly pipeline failed", state="error", expanded=False)
        finally:
            st.session_state["pipeline_active"] = False
    return result


def _render_pipeline_result(result: dict[str, Any] | None) -> None:
    if not result:
        return
    status = result.get("status", "failure")
    if status == "success":
        st.success("Weekly pipeline completed successfully.")
    elif status == "partial":
        st.warning("Weekly pipeline completed with skips or partial source status.")
    else:
        st.error("Weekly pipeline failed.")
    st.caption(f"Pipeline state: {status} | Exit code: {result.get('exit_code', '-')}")
    if result.get("concise_output"):
        st.code(result["concise_output"], language="text")
    with st.expander("Full pipeline output", expanded=False):
        st.code(result.get("full_output") or "No output captured.", language="text")


def _classify_pipeline_result(exit_code: int | None, output: str) -> str:
    if exit_code not in (0, None):
        return "failure"
    normalized = output.lower()
    if "completed_with_skips" in normalized or "run_status=partial" in normalized:
        return "partial"
    return "success"


def _concise_pipeline_output(output: str, max_lines: int = 18, max_chars: int = 4000) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    concise = "\n".join(lines[-max_lines:])
    if len(concise) > max_chars:
        concise = concise[-max_chars:]
    return concise


def _combine_pipeline_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append("STDOUT\n" + stdout.strip())
    if stderr.strip():
        parts.append("STDERR\n" + stderr.strip())
    return "\n\n".join(parts)


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _redact_pipeline_output(value: Any) -> str:
    output = _output_text(value)
    output = re.sub(
        r"(?i)(postgres(?:ql)?(?:\+[^:/\s]+)?://)[^\s]+",
        r"\1[redacted]",
        output,
    )
    output = re.sub(
        r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        output,
    )
    return output


def _coerce_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(timestamp) else timestamp


def _timestamp_age_text(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None
    age = pd.Timestamp.now(tz="UTC") - timestamp
    if age.total_seconds() < 0:
        return "snapshot timestamp is in the future"
    return f"snapshot age: {_format_timedelta(age)}"


def _format_timedelta(value: Any) -> str:
    seconds = max(0, int(value.total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def main() -> None:
    inject_css()

    conn = get_db_connection()
    artifact_results = check_artifacts()
    counts = {label: safe_table_count(conn, table) for label, table in COUNT_TABLES.items()}
    latest_ingestion = load_latest_ingestion_run(conn)
    latest_prediction = load_latest_prediction_run(conn)
    freshness_df = load_data_freshness(conn)
    predictions_df = load_prediction_rows(conn)
    fpl_predictions_df = load_fpl_prediction_rows(conn)
    fpl_optimizer_row = load_latest_fpl_optimizer(conn)
    fpl_snapshot_df = load_fpl_snapshot_status(conn)
    fpl_cold_start_teams = load_fpl_cold_start_teams(conn)
    latest_fpl_deadline = load_latest_fpl_bootstrap_deadline(conn)
    health_df = load_health_logs(conn)

    render_hero(conn, artifact_results, counts)

    render_operator_controls(conn, latest_fpl_deadline)

    if conn is None:
        render_alert("Database connection unavailable. Artifact checks still work, but live production tables cannot be read.")

    tabs = st.tabs(
        [
            "Overview",
            "Pipeline Status",
            "Predictions",
            "FPL Predictions",
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
        render_fpl_predictions(
            fpl_predictions_df,
            fpl_optimizer_row,
            fpl_snapshot_df,
            latest_ingestion,
            freshness_df,
            fpl_cold_start_teams,
        )

    with tabs[4]:
        render_model_health(health_df)

    with tabs[5]:
        render_reports()

    with tabs[6]:
        render_how_to_run()


def render_hero(conn, artifact_results, counts) -> None:
    db_status = "db connected" if conn is not None else "db unavailable"
    artifact_status = "artifacts ready" if all(result["exists"] for result in artifact_results.values()) else "artifacts missing"
    prediction_count = counts.get("Predictions")
    prediction_status = "ready" if prediction_count and prediction_count > 0 else "waiting for fixtures"
    st.markdown(
        (
            "<div class='fp-hero'>"
            "<div class='fp-hero-top'>"
            "<div>"
            "<div class='fp-eyebrow'>Tier 3 production console</div>"
            "<h1 class='fp-title'>Premier League model control room</h1>"
            "<div class='fp-subtitle'>Read-only view for the production logistic Elo model, weekly ingestion pipeline, prediction tables, and final validation reports.</div>"
            "</div>"
            "<div class='fp-status-row'>"
            f"{status_badge_html(db_status)}"
            f"{status_badge_html(artifact_status)}"
            f"{status_badge_html(prediction_status)}"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_overview(artifact_results, counts, freshness_df) -> None:
    render_section("Production Overview", "Frozen production model configuration and local artifact health.")
    render_card_grid(
        [
            {
                "label": "Model",
                "value": "Logistic Elo v3",
                "detail": f"Artifact: {MODEL_NAME}",
            },
            {
                "label": "Training Seasons",
                "value": "2021-22 - 2025-26",
                "detail": "Production fit includes the former 2025-26 holdout.",
            },
            {
                "label": "Feature Count",
                "value": EXPECTED_FEATURE_COUNT,
                "detail": "Base rolling features plus pre-match Elo.",
            },
            {
                "label": "Draw Threshold",
                "value": f"{EXPECTED_DRAW_THRESHOLD:.2f}",
                "detail": "Optional hard-label draw-risk helper.",
            },
        ]
    )

    render_section("Artifact Checks", "Production artifacts are local-only and ignored by Git.")
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
    artifact_cards = [
        {
            "label": row["artifact"],
            "value": row["exists"],
            "status_html": status_badge_html(format_bool(row["exists"] == "yes")),
            "detail": row["details"] or "local artifact verified",
        }
        for row in artifact_rows
    ]
    render_card_grid(artifact_cards)
    if not all(result["exists"] for result in artifact_results.values()):
        render_alert("One or more production artifacts are missing.")

    render_section("Live Production Tables", "Counts from local production tables when PostgreSQL is reachable.")
    render_card_grid(
        [
            {
                "label": label,
                "value": "-" if value is None else value,
                "detail": "table unavailable" if value is None else "rows currently available",
            }
            for label, value in counts.items()
        ]
    )

    if counts.get("Predictions") == 0:
        render_alert(
            "No production predictions yet. Run the weekly pipeline after upcoming fixtures are available."
        )

    if _freshness_has_warning(freshness_df):
        render_alert("One or more production data sources are stale or unavailable.")


def render_pipeline_status(latest_ingestion, latest_prediction, freshness_df, counts) -> None:
    render_section("Pipeline Status", "Latest ingestion, prediction, and freshness state.")
    left, right = st.columns(2)

    with left:
        if latest_ingestion:
            render_run_card("Latest ingestion run", latest_ingestion, "run_id")
        else:
            render_empty_state("No ingestion runs found", "Run the weekly pipeline after sources are available.")

    with right:
        if latest_prediction:
            render_run_card("Latest prediction run", latest_prediction, "prediction_run_id")
        else:
            render_empty_state("No prediction runs found", "Predictions will appear after upcoming feature rows exist.")

    render_section("Data Freshness", "Source snapshots and latest source errors.")
    if freshness_df.empty:
        render_empty_state("No freshness rows found", "The ingestion foundation is present, but no source freshness rows were returned.")
    else:
        freshness_display = freshness_df.copy()
        st.dataframe(freshness_display, use_container_width=True, hide_index=True)
        for row in freshness_df.to_dict(orient="records"):
            if row.get("latest_error_message"):
                render_alert(f"{row['source_name']}: {row['latest_error_message']}")

    render_section("Derived Pipeline State", "Readable summary of the pipeline gates.")
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
    render_section("Production Predictions", "Upcoming matches scored by the frozen production model.")
    if predictions_df.empty:
        render_empty_state(
            "No predictions yet",
            "Run the weekly pipeline after 2026-27 upcoming fixtures are available. The app will not create fake fixtures.",
        )
        return

    display_df = predictions_df.copy()
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win", "confidence"]
    for column in probability_columns:
        if column in display_df:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_fpl_predictions(
    fpl_predictions_df: pd.DataFrame,
    optimizer_row: dict[str, Any] | None,
    snapshot_df: pd.DataFrame,
    latest_ingestion: dict[str, Any] | None,
    freshness_df: pd.DataFrame,
    cold_start_teams: list[str],
) -> None:
    render_section(
        "FPL Predictions",
        "Read-only FPL v3 player forecasts and the latest optimized squad.",
    )

    if fpl_predictions_df.empty and optimizer_row is None:
        render_empty_state(
            "No FPL predictions available",
            "Run the FPL production prediction stage after the current bootstrap snapshot is available.",
        )
        _render_fpl_warnings(
            fpl_predictions_df,
            optimizer_row,
            snapshot_df,
            latest_ingestion,
            freshness_df,
            cold_start_teams,
        )
        return

    target_season = _first_value(fpl_predictions_df, "target_season") or (
        optimizer_row or {}
    ).get("target_season")
    target_gameweek = _first_value(fpl_predictions_df, "target_gameweek") or (
        optimizer_row or {}
    ).get("target_gameweek")
    generated_at = None
    if not fpl_predictions_df.empty and "created_at" in fpl_predictions_df:
        generated_at = fpl_predictions_df["created_at"].max()
    if generated_at is None and optimizer_row:
        generated_at = optimizer_row.get("generated_at")

    squad_records = _json_records((optimizer_row or {}).get("squad_json"))
    starting_records = _json_records((optimizer_row or {}).get("starting_xi_json"))
    squad_cost = (optimizer_row or {}).get("squad_cost")
    budget_limit = (optimizer_row or {}).get("budget_limit")
    budget_remaining = _subtract_if_numeric(budget_limit, squad_cost)
    captain_name = _lookup_player_name(
        (optimizer_row or {}).get("captain_player_id"),
        fpl_predictions_df,
        squad_records,
    )
    vice_captain_name = _lookup_player_name(
        (optimizer_row or {}).get("vice_captain_player_id"),
        fpl_predictions_df,
        squad_records,
    )

    render_card_grid(
        [
            {
                "label": "Target Season",
                "value": target_season or "-",
                "detail": "latest FPL prediction target",
            },
            {
                "label": "Target Gameweek",
                "value": target_gameweek or "-",
                "detail": "latest FPL prediction target",
            },
            {
                "label": "Prediction Rows",
                "value": len(fpl_predictions_df),
                "detail": "player forecasts returned",
            },
            {
                "label": "Generated",
                "value": _format_value(generated_at) if generated_at else "-",
                "detail": "latest prediction generation time",
            },
            {
                "label": "Optimized Squad Cost",
                "value": squad_cost if squad_cost is not None else "-",
                "detail": f"budget limit: {budget_limit}" if budget_limit is not None else "optimizer output unavailable",
            },
            {
                "label": "Budget Remaining",
                "value": budget_remaining if budget_remaining is not None else "-",
                "detail": "budget units remaining",
            },
            {
                "label": "Captain",
                "value": captain_name,
                "detail": "optimized captain",
            },
            {
                "label": "Vice-captain",
                "value": vice_captain_name,
                "detail": "optimized vice-captain",
            },
        ]
    )

    _render_fpl_warnings(
        fpl_predictions_df,
        optimizer_row,
        snapshot_df,
        latest_ingestion,
        freshness_df,
        cold_start_teams,
    )

    if optimizer_row is None:
        render_empty_state(
            "No optimizer output available",
            "Player predictions exist, but no optimized squad has been generated for the latest target.",
        )
    else:
        render_section("Optimized Squad", "The latest read-only optimizer output.")
        if starting_records:
            st.subheader("Starting XI")
            st.dataframe(
                _squad_display_frame(starting_records, fpl_predictions_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_state("No starting XI available", "The optimizer output did not contain a starting XI.")

        if squad_records:
            st.subheader("Full 15-player squad")
            st.dataframe(
                _squad_display_frame(squad_records, fpl_predictions_df),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_empty_state("No full squad available", "The optimizer output did not contain a squad list.")

    render_section(
        "Player Predictions",
        "All current FPL player forecasts sorted by expected points descending.",
    )
    if fpl_predictions_df.empty:
        render_empty_state("No player prediction rows", "No FPL player forecasts were returned for the latest target.")
        return

    display_df = fpl_predictions_df.copy()
    display_df["position"] = display_df["position_id"].map(FPL_POSITION_NAMES).fillna("Unknown")
    display_df["chance_of_playing_display"] = display_df["chance_of_playing"].apply(
        lambda value: "Unknown" if pd.isna(value) else f"{int(value)}%"
    )
    display_df["cold_start_warning"] = display_df["team_name"].apply(
        lambda team: "Cold-start team" if team in cold_start_teams else ""
    )
    display_df = display_df.sort_values(
        ["expected_points", "player_name"],
        ascending=[False, True],
    )
    display_df = display_df.rename(
        columns={
            "player_name": "Player Name",
            "team_name": "Team",
            "position": "Position",
            "predicted_points": "Predicted Points",
            "availability_factor": "Availability Factor",
            "expected_points": "Expected Points",
            "status": "Player Status",
            "chance_of_playing_display": "Chance of Playing",
            "cold_start_warning": "Cold-start Warning",
        }
    )
    table_columns = [
        "Player Name",
        "Team",
        "Position",
        "Predicted Points",
        "Availability Factor",
        "Expected Points",
        "Player Status",
        "Chance of Playing",
        "Cold-start Warning",
    ]
    st.dataframe(display_df[table_columns], use_container_width=True, hide_index=True)


def _render_fpl_warnings(
    fpl_predictions_df: pd.DataFrame,
    optimizer_row: dict[str, Any] | None,
    snapshot_df: pd.DataFrame,
    latest_ingestion: dict[str, Any] | None,
    freshness_df: pd.DataFrame,
    cold_start_teams: list[str],
) -> None:
    if cold_start_teams:
        render_alert("Cold-start teams were used: " + ", ".join(cold_start_teams) + ".")

    ingestion_status = str((latest_ingestion or {}).get("run_status", "unknown")).lower()
    if ingestion_status == "partial":
        render_alert("The latest ingestion run is partial; some production sources did not complete.")

    football_data_status = str((latest_ingestion or {}).get("football_data_status", "unknown")).lower()
    if football_data_status in {"failed", "unavailable", "skipped"}:
        render_alert("football-data ingestion is unavailable in the latest production run.")

    understat_status = str((latest_ingestion or {}).get("understat_status", "unknown")).lower()
    if understat_status in {"failed", "unavailable", "skipped"}:
        render_alert("Understat data is unavailable in the latest production run.")

    stale_sources = _freshness_warning_sources(freshness_df)
    if stale_sources:
        render_alert("Stale or unavailable data sources: " + ", ".join(stale_sources) + ".")

    if not fpl_predictions_df.empty:
        target_season = _first_value(fpl_predictions_df, "target_season")
        target_gameweek = _first_value(fpl_predictions_df, "target_gameweek")
        target_snapshots = snapshot_df.loc[
            (snapshot_df.get("target_season") == target_season)
            & (snapshot_df.get("gameweek") == target_gameweek)
        ] if not snapshot_df.empty else pd.DataFrame()
        if target_snapshots.empty:
            render_alert(
                "FPL predictions exist, but current gameweek results have not been scored; "
                "gameweek snapshots are not available yet."
            )


def _squad_display_frame(records: list[dict[str, Any]], predictions_df: pd.DataFrame) -> pd.DataFrame:
    prediction_lookup = {}
    if not predictions_df.empty and "player_id" in predictions_df:
        prediction_lookup = predictions_df.set_index("player_id").to_dict(orient="index")

    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        player_id = _safe_int(record.get("player_id"))
        prediction = prediction_lookup.get(player_id, {})
        rows.append(
            {
                "Player Name": record.get("player_name") or prediction.get("player_name") or "Unavailable",
                "Team": record.get("team_name") or prediction.get("team_name") or "Unavailable",
                "Position": record.get("position") or FPL_POSITION_NAMES.get(
                    _safe_int(record.get("position_id") or prediction.get("position_id")),
                    "Unknown",
                ),
                "Expected Points": _numeric_display(
                    record.get("expected_points", prediction.get("expected_points"))
                ),
                "Cost": record.get("now_cost", "-") or "-",
            }
        )
    return pd.DataFrame(rows, columns=["Player Name", "Team", "Position", "Expected Points", "Cost"])


def _json_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _lookup_player_name(player_id: Any, predictions_df: pd.DataFrame, squad_records: list[dict[str, Any]]) -> str:
    normalized_id = _safe_int(player_id)
    if normalized_id is None:
        return "Unavailable"
    if not predictions_df.empty:
        matches = predictions_df.loc[predictions_df["player_id"] == normalized_id, "player_name"]
        if not matches.empty:
            return str(matches.iloc[0])
    for record in squad_records:
        if _safe_int(record.get("player_id")) == normalized_id:
            return str(record.get("player_name") or f"Player {normalized_id}")
    return f"Player {normalized_id}"


def _first_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame:
        return None
    value = frame.iloc[0][column]
    return None if pd.isna(value) else value


def _safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_display(value: Any) -> Any:
    if value is None or pd.isna(value):
        return "-"
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return value


def _subtract_if_numeric(left: Any, right: Any) -> Any:
    left_value = _numeric_value(left)
    right_value = _numeric_value(right)
    if left_value is None or right_value is None:
        return None
    result = left_value - right_value
    return int(result) if result.is_integer() else round(result, 2)


def _numeric_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_model_health(health_df) -> None:
    render_section("Final Tier 3 Holdout Metrics", "Official frozen evaluation before the production refit.")
    render_card_grid(
        [
            {"label": f"Argmax {label}", "value": value, "detail": "final 2025-26 holdout"}
            for label, value in HOLDOUT_ARGMAX.items()
        ]
    )
    render_card_grid(
        [
            {"label": f"Overlay {label}", "value": value, "detail": "hard-label helper only"}
            for label, value in HOLDOUT_OVERLAY.items()
        ]
    )

    render_alert(
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

    render_section("Production Health Logs", "Weekly scoring will populate this after predictions meet completed results.")
    if health_df.empty:
        render_empty_state(
            "No health logs yet",
            "This is expected until there are production predictions and completed match results to score.",
        )
        return
    st.dataframe(health_df, use_container_width=True, hide_index=True)


def render_reports() -> None:
    render_section("Tier 3 Reports", "Research notes, final holdout report, and error analysis.")
    for title, path in REPORTS.items():
        with st.expander(title, expanded=False):
            markdown = load_markdown_file(path)
            if markdown:
                st.markdown(markdown)
            else:
                st.warning(f"Report not found: {path.relative_to(PROJECT_ROOT)}")


def render_how_to_run() -> None:
    render_section("Local Commands", "Run these from the project root in the active Python environment.")
    commands = [
        "python src/production/run_weekly_pipeline.py --target-season 2026-27",
        "python src/production/weekly_ingest.py --target-season 2026-27",
        "python src/production/build_upcoming_features.py --target-season 2026-27",
        "python src/production/predict_production_matches.py --target-season 2026-27",
        "python src/production/score_predictions.py --target-season 2026-27",
    ]
    st.markdown(
        "<div class='fp-code-note'>The dashboard is read-only. It shows pipeline state but does not execute pipeline jobs.</div>",
        unsafe_allow_html=True,
    )
    for command in commands:
        st.code(command, language="powershell")


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
    return bool(_freshness_warning_sources(freshness_df))


def _freshness_warning_sources(freshness_df: pd.DataFrame) -> list[str]:
    if freshness_df.empty:
        return ["freshness unavailable"]

    warnings: list[str] = []
    now = pd.Timestamp.now(tz="UTC")
    for row in freshness_df.to_dict(orient="records"):
        source_name = str(row.get("source_name") or "unknown source")
        error_message = row.get("latest_error_message")
        if error_message is not None and not pd.isna(error_message) and str(error_message).strip():
            warnings.append(source_name)
            continue

        timestamp = pd.to_datetime(row.get("last_successful_update_at"), errors="coerce", utc=True)
        if pd.isna(timestamp) or now - timestamp > pd.Timedelta(hours=48):
            warnings.append(source_name)
    return sorted(set(warnings))


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
