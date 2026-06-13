import json
import math
import sys
import warnings
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sqlalchemy
import streamlit as st
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_FEATURES_PATH = MODELS_DIR / "model_features.json"

FALLBACK_MODEL_FEATURES_16 = [
    "home_form_scored",
    "home_form_conceded",
    "home_clean_sheet_rate",
    "away_form_scored",
    "away_form_conceded",
    "away_clean_sheet_rate",
    "home_fdr",
    "away_fdr",
    "strength_overall_home",
    "strength_overall_away",
    "home_team_away_str",
    "away_team_home_str",
    "home_xg_last5",
    "home_xga_last5",
    "away_xg_last5",
    "away_xga_last5",
]

POSITION_LABELS = {
    1: "Goalkeepers",
    2: "Defenders",
    3: "Midfielders",
    4: "Forwards",
}

st.set_page_config(
    page_title="Premier League ML Lab",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --app-bg: #0b0c0f;
            --panel-bg: #15181d;
            --panel-soft: #101318;
            --panel-border: #2a3038;
            --muted: #9aa4b2;
            --text: #f7f8fa;
            --accent: #2dd4bf;
            --accent-strong: #14b8a6;
            --blue: #60a5fa;
            --amber: #f59e0b;
            --danger: #f87171;
            --success: #22c55e;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: rgba(11, 12, 15, 0.78);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] {
            background: #111318;
            border-right: 1px solid var(--panel-border);
        }

        [data-testid="stSidebar"] h1 {
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--panel-border);
            margin: 1.55rem 0;
        }

        [data-testid="stSidebar"] .stRadio label {
            border-radius: 8px;
            padding: 0.18rem 0;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 5.4rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        h3 {
            margin-top: 1.2rem;
        }

        .page-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        .page-title {
            font-size: clamp(2rem, 4vw, 3.4rem);
            font-weight: 780;
            line-height: 1.05;
            margin-bottom: 0.75rem;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.55;
            max-width: 860px;
            margin-bottom: 1.7rem;
        }

        .section-rule {
            border-top: 1px solid var(--panel-border);
            margin: 1.6rem 0 1.8rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.8rem;
        }

        .metric-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            min-width: 0;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--text);
            font-size: clamp(1.35rem, 2vw, 2.05rem);
            font-weight: 740;
            line-height: 1.2;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.4;
            margin-top: 0.35rem;
        }

        .status-grid {
            display: grid;
            gap: 0.65rem;
            margin-top: 0.6rem;
        }

        .status-chip {
            align-items: center;
            background: var(--panel-soft);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.7rem 0.8rem;
        }

        .status-chip-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-chip-value {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 700;
            text-align: right;
        }

        .status-panel {
            background: rgba(45, 212, 191, 0.1);
            border: 1px solid rgba(45, 212, 191, 0.3);
            border-radius: 8px;
            color: #ccfbf1;
            padding: 0.95rem 1rem;
            margin: 1rem 0 1.25rem;
        }

        .scoreboard {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            gap: 1rem;
            margin: 1rem 0 1.35rem;
            overflow: hidden;
            padding: clamp(1rem, 3vw, 1.45rem);
        }

        .score-team {
            min-width: 0;
        }

        .score-team.away {
            text-align: right;
        }

        .score-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .score-name {
            color: var(--text);
            font-size: clamp(1.25rem, 2.8vw, 2.1rem);
            font-weight: 780;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .score-core {
            align-items: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 150px;
        }

        .scoreline {
            color: var(--text);
            font-size: clamp(2.2rem, 5vw, 4rem);
            font-weight: 800;
            line-height: 1;
            white-space: nowrap;
        }

        .score-outcome {
            color: var(--accent);
            font-size: 0.9rem;
            font-weight: 760;
            margin-top: 0.55rem;
            text-align: center;
        }

        .result-card {
            background:
                linear-gradient(135deg, rgba(45, 212, 191, 0.16), transparent 40%),
                linear-gradient(180deg, #15181d, #101318);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin: 1rem 0 1.25rem;
            padding: clamp(1.1rem, 3vw, 1.6rem);
        }

        .fixture-title {
            color: var(--text);
            font-size: clamp(1.45rem, 3.4vw, 2.6rem);
            font-weight: 800;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }

        .outcome-badge {
            background: rgba(45, 212, 191, 0.13);
            border: 1px solid rgba(45, 212, 191, 0.45);
            border-radius: 999px;
            color: #ccfbf1;
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin: 1rem 0 0.65rem;
            padding: 0.45rem 0.75rem;
            text-transform: uppercase;
        }

        .result-meta {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .cred-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.8rem;
        }

        .cred-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            min-width: 0;
            padding: 1rem;
        }

        .cred-title {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 780;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .cred-value {
            color: var(--text);
            font-size: clamp(1.2rem, 2vw, 1.7rem);
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.4rem;
            overflow-wrap: anywhere;
        }

        .cred-note {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 0.35rem;
        }

        .disclaimer {
            background: rgba(245, 158, 11, 0.11);
            border: 1px solid rgba(245, 158, 11, 0.38);
            border-radius: 8px;
            color: #fde68a;
            font-weight: 720;
            line-height: 1.45;
            margin: 1rem 0 1.5rem;
            padding: 1rem;
        }

        .prob-grid {
            display: grid;
            gap: 0.8rem;
            margin: 1rem 0 1.8rem;
        }

        .prob-row {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.95rem 1rem;
        }

        .prob-topline {
            align-items: center;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }

        .prob-label {
            color: var(--text);
            font-weight: 730;
            overflow-wrap: anywhere;
        }

        .prob-value {
            color: var(--text);
            font-weight: 800;
            white-space: nowrap;
        }

        .prob-track {
            background: #222832;
            border-radius: 999px;
            height: 0.65rem;
            overflow: hidden;
        }

        .prob-fill {
            background: linear-gradient(90deg, var(--accent), var(--blue));
            border-radius: 999px;
            height: 100%;
        }

        .stacked-bar {
            background: #222832;
            border-radius: 999px;
            display: flex;
            height: 1rem;
            margin: 1rem 0 0.45rem;
            overflow: hidden;
        }

        .bar-home {
            background: #38bdf8;
        }

        .bar-draw {
            background: #f59e0b;
        }

        .bar-away {
            background: #ef4444;
        }

        .bar-legend {
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            font-size: 0.84rem;
            margin-bottom: 1.35rem;
        }

        .bar-dot {
            border-radius: 999px;
            display: inline-block;
            height: 0.65rem;
            margin-right: 0.35rem;
            width: 0.65rem;
        }

        .pitch {
            background:
                linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
                linear-gradient(180deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
                linear-gradient(180deg, rgba(20, 184, 166, 0.12), rgba(17, 24, 39, 0.88));
            background-size: 48px 48px, 48px 48px, auto;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin: 1rem 0 1.5rem;
            padding: 1.2rem;
        }

        .pitch-row {
            display: grid;
            gap: 0.8rem;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            margin: 0.95rem 0;
        }

        .pitch-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            margin-top: 1rem;
            text-align: center;
            text-transform: uppercase;
        }

        .player-card {
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid rgba(45, 212, 191, 0.28);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            min-width: 0;
            padding: 0.8rem;
        }

        .player-name {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 780;
            line-height: 1.2;
            min-height: 2.35rem;
            overflow-wrap: anywhere;
        }

        .player-meta {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .player-statline {
            display: flex;
            gap: 0.75rem;
            justify-content: space-between;
            margin-top: 0.7rem;
        }

        .player-stat {
            min-width: 0;
        }

        .player-stat-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .player-stat-value {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 780;
            margin-top: 0.1rem;
        }

        .captain-panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            margin: 1rem 0 1.5rem;
            padding: 1.2rem;
        }

        .captain-name {
            color: var(--text);
            font-size: clamp(1.5rem, 3vw, 2.4rem);
            font-weight: 790;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .captain-meta {
            color: var(--muted);
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .captain-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-top: 1.15rem;
        }

        .captain-stat {
            border-top: 1px solid var(--panel-border);
            padding-top: 0.8rem;
        }

        .captain-stat-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 780;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .caption-muted {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.8rem;
        }

        .stButton > button[kind="primary"] {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #05100e;
        }

        .stButton > button[kind="primary"]:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: #05100e;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 8px;
        }

        .stDataFrame {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 3.5rem;
            }

            .scoreboard {
                grid-template-columns: 1fr;
            }

            .score-team.away {
                text-align: left;
            }

            .score-core {
                align-items: flex-start;
            }

            .pitch {
                padding: 0.85rem;
            }

            .pitch-row {
                grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Loads the PostgreSQL engine once for the Streamlit session.
@st.cache_resource
def load_engine():
    try:
        from src.data_pipeline import get_engine

        engine = get_engine()
        if engine is None:
            st.error("Could not connect to PostgreSQL. Check your .env settings.")
        return engine
    except Exception as error:
        st.error(f"Database connection failed: {error}")
        return None


# Loads saved match models without retraining.
@st.cache_resource
def load_models():
    try:
        required_model_files = {
            "scaler": "scaler.pkl",
            "label_encoder": "label_encoder.pkl",
            "home_goals": "xgb_home_goals.pkl",
            "away_goals": "xgb_away_goals.pkl",
        }

        models = {}
        for key, filename in required_model_files.items():
            path = MODELS_DIR / filename
            if not path.exists():
                st.error(f"Missing model file: {path}")
                return None
            models[key] = joblib.load(path)

        xgb_path = MODELS_DIR / "xgb_classifier.pkl"
        lr_path = MODELS_DIR / "logistic_classifier.pkl"

        if xgb_path.exists():
            models["xgb_clf"] = joblib.load(xgb_path)
            models["active_classifier"] = "XGBoost"
        elif lr_path.exists():
            models["lr"] = joblib.load(lr_path)
            models["active_classifier"] = "Logistic Regression fallback"
            st.warning("XGBoost classifier missing. Using Logistic Regression fallback.")
        else:
            st.error("Missing classifier files: xgb_classifier.pkl and logistic_classifier.pkl")
            return None

        if lr_path.exists() and "lr" not in models:
            models["lr"] = joblib.load(lr_path)

        return models
    except Exception as error:
        st.error(f"Failed to load models: {error}")
        return None


# Loads the saved match feature order, with the 16-feature xG fallback.
@st.cache_data
def load_model_features():
    try:
        if MODEL_FEATURES_PATH.exists():
            with MODEL_FEATURES_PATH.open("r", encoding="utf-8") as file:
                features = json.load(file)

            if isinstance(features, list) and features:
                return features

            st.error("model_features.json is invalid. Using fallback features.")
        else:
            st.warning("model_features.json is missing. Using fallback 16-feature list.")
    except Exception as error:
        st.error(f"Could not load model_features.json: {error}")

    return FALLBACK_MODEL_FEATURES_16


# Returns the classifier used for visible Win/Draw/Loss probabilities.
def get_active_classifier(models):
    if models is None:
        return None, None

    if models.get("active_classifier") == "XGBoost" and "xgb_clf" in models:
        return models["xgb_clf"], "XGBoost"

    if "lr" in models:
        return models["lr"], "Logistic Regression fallback"

    return None, None


# Keeps progress values safe for Streamlit.
def clip_probability(value):
    return float(np.clip(float(value), 0.0, 1.0))


def render_header(kicker, title, subtitle):
    st.markdown(
        (
            f'<div class="page-kicker">{escape(kicker)}</div>'
            f'<div class="page-title">{escape(title)}</div>'
            f'<div class="page-subtitle">{escape(subtitle)}</div>'
            '<div class="section-rule"></div>'
        ),
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics):
    cards = []
    for label, value, note in metrics:
        note_html = f'<div class="metric-note">{escape(note)}</div>' if note else ""
        cards.append(
            (
                '<div class="metric-card">'
                f'<div class="metric-label">{escape(label)}</div>'
                f'<div class="metric-value">{escape(str(value))}</div>'
                f"{note_html}"
                "</div>"
            )
        )

    st.markdown(
        f'<div class="metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_status_chips(items):
    chips = []
    for label, value in items:
        chips.append(
            (
                '<div class="status-chip">'
                f'<div class="status-chip-label">{escape(label)}</div>'
                f'<div class="status-chip-value">{escape(str(value))}</div>'
                "</div>"
            )
        )

    st.markdown(
        f'<div class="status-grid">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def render_status_panel(message):
    st.markdown(
        f'<div class="status-panel">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_match_scoreboard(home_team, away_team, pred_home, pred_away, result_label):
    st.markdown(
        (
            '<div class="scoreboard">'
            '<div class="score-team">'
            '<div class="score-label">Home</div>'
            f'<div class="score-name">{escape(home_team)}</div>'
            "</div>"
            '<div class="score-core">'
            f'<div class="scoreline">{pred_home} - {pred_away}</div>'
            f'<div class="score-outcome">{escape(result_label)}</div>'
            "</div>"
            '<div class="score-team away">'
            '<div class="score-label">Away</div>'
            f'<div class="score-name">{escape(away_team)}</div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_match_result_card(home_team, away_team, outcome_label, confidence, scoreline):
    fixture = f"{home_team} vs {away_team}"
    st.markdown(
        (
            '<div class="result-card">'
            f'<div class="fixture-title">{escape(fixture)}</div>'
            f'<div class="outcome-badge">Predicted Outcome: {escape(outcome_label)}</div>'
            f'<div class="result-meta">Projected score: {escape(scoreline)} | {escape(confidence)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_probability_bars(probabilities):
    rows = []
    for label, value in probabilities:
        percent = max(0.0, min(float(value), 1.0)) * 100
        rows.append(
            (
                '<div class="prob-row">'
                '<div class="prob-topline">'
                f'<div class="prob-label">{escape(label)}</div>'
                f'<div class="prob-value">{percent:.1f}%</div>'
                "</div>"
                '<div class="prob-track">'
                f'<div class="prob-fill" style="width: {percent:.1f}%"></div>'
                "</div>"
                "</div>"
            )
        )

    st.markdown(
        f'<div class="prob-grid">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def confidence_label(probability):
    if probability >= 0.65:
        return "High confidence"
    if probability >= 0.45:
        return "Medium confidence"
    return "Low confidence"


def confidence_from_margin(probabilities):
    ordered = sorted([float(value) for value in probabilities], reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    if margin >= 0.20:
        return "High confidence"
    if margin >= 0.10:
        return "Medium confidence"
    return "Low confidence"


def render_stacked_probability_bar(home_team, away_team, home_prob, draw_prob, away_prob):
    home_width = max(0.0, min(float(home_prob), 1.0)) * 100
    draw_width = max(0.0, min(float(draw_prob), 1.0)) * 100
    away_width = max(0.0, min(float(away_prob), 1.0)) * 100

    st.markdown(
        (
            '<div class="stacked-bar">'
            f'<div class="bar-home" style="width: {home_width:.1f}%"></div>'
            f'<div class="bar-draw" style="width: {draw_width:.1f}%"></div>'
            f'<div class="bar-away" style="width: {away_width:.1f}%"></div>'
            "</div>"
            '<div class="bar-legend">'
            f'<span><span class="bar-dot bar-home"></span>{escape(home_team)} {home_width:.1f}%</span>'
            f'<span><span class="bar-dot bar-draw"></span>Draw {draw_width:.1f}%</span>'
            f'<span><span class="bar-dot bar-away"></span>{escape(away_team)} {away_width:.1f}%</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def explain_match_result(best_result, home_team, away_team, home_prob, draw_prob, away_prob):
    confidence = confidence_from_margin([home_prob, draw_prob, away_prob])
    if best_result == "H":
        result_text = f"The model favors {home_team}"
    elif best_result == "A":
        result_text = f"The model favors {away_team}"
    else:
        result_text = "The model sees this fixture as draw-leaning"

    return f"{result_text} with {confidence.lower()} based on the current feature set."


def render_model_credibility_cards():
    cards = [
        ("Tier 2 Accuracy", "0.570", "XGBoost match classifier"),
        ("FPL MAE / RMSE", "0.926 / not tracked", "RMSE is not stored in the current artifacts"),
        ("Data Sources", "FPL API + Understat", "Players, fixtures, xG, rolling form, and history"),
        ("Model Type", "XGBoost", "Classifier, regressors, and PuLP squad optimizer"),
    ]
    rendered = []
    for title, value, note in cards:
        rendered.append(
            (
                '<div class="cred-card">'
                f'<div class="cred-title">{escape(title)}</div>'
                f'<div class="cred-value">{escape(value)}</div>'
                f'<div class="cred-note">{escape(note)}</div>'
                "</div>"
            )
        )

    st.markdown(
        f'<div class="cred-grid">{"".join(rendered)}</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        '<div class="disclaimer">Analytics tool only. Not betting advice. Football is highly variant.</div>',
        unsafe_allow_html=True,
    )



def render_captain_panel(captain):
    start_prob = captain.get("start_probability", 1.0)
    if pd.isna(start_prob):
        start_prob = 1.0

    captain_name = f"{captain['first_name']} {captain['second_name']}"
    st.markdown(
        (
            '<div class="captain-panel">'
            '<div class="metric-label">Captain Recommendation</div>'
            f'<div class="captain-name">{escape(captain_name)}</div>'
            f'<div class="captain-meta">{escape(str(captain["team_name"]))}</div>'
            '<div class="captain-stats">'
            '<div class="captain-stat">'
            '<div class="metric-label">Price</div>'
            f'<div class="captain-stat-value">GBP {float(captain["price"]):.1f}m</div>'
            "</div>"
            '<div class="captain-stat">'
            '<div class="metric-label">Form</div>'
            f'<div class="captain-stat-value">{float(captain["form"]):.1f}</div>'
            "</div>"
            '<div class="captain-stat">'
            '<div class="metric-label">Start Probability</div>'
            f'<div class="captain-stat-value">{float(start_prob):.0%}</div>'
            "</div>"
            '<div class="captain-stat">'
            '<div class="metric-label">Estimated Points</div>'
            f'<div class="captain-stat-value">{float(captain["estimated_points"]):.1f}</div>'
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def player_position_label(position):
    return {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
    }.get(int(position), "Player")


def render_player_card(row):
    player_name = f"{row['first_name']} {row['second_name']}"
    team_name = row.get("team_name", "")
    position = player_position_label(row.get("position", 0))
    role = row.get("squad_role", "Squad")
    price = float(row.get("price", 0.0))
    points = float(row.get("estimated_points", 0.0))

    return (
        '<div class="player-card">'
        f'<div class="player-name">{escape(player_name)}</div>'
        f'<div class="player-meta">{escape(str(team_name))} | {escape(position)} | {escape(str(role))}</div>'
        '<div class="player-statline">'
        '<div class="player-stat">'
        '<div class="player-stat-label">Price</div>'
        f'<div class="player-stat-value">GBP {price:.1f}m</div>'
        "</div>"
        '<div class="player-stat">'
        '<div class="player-stat-label">Points</div>'
        f'<div class="player-stat-value">{points:.1f}</div>'
        "</div>"
        "</div>"
        "</div>"
    )


def render_squad_pitch(squad):
    rows = []
    for position in [1, 2, 3, 4]:
        pos_players = squad[squad["position"] == position].copy()
        pos_players = pos_players.sort_values(
            ["is_starter", "estimated_points"],
            ascending=[False, False],
        )
        cards = "".join(render_player_card(row) for _, row in pos_players.iterrows())
        rows.append(
            (
                f'<div class="pitch-label">{escape(POSITION_LABELS[position])}</div>'
                f'<div class="pitch-row">{cards}</div>'
            )
        )

    st.markdown(
        f'<div class="pitch">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


# Loads team names for the match predictor.
@st.cache_data
def load_teams(_engine):
    try:
        if _engine is None:
            return []

        query = """
            SELECT DISTINCT home_team_name
            FROM match_features
            ORDER BY home_team_name
        """
        df = pd.read_sql(query, _engine)
        return sorted(df["home_team_name"].dropna().unique().tolist())
    except Exception as error:
        st.error(f"Could not load teams: {error}")
        return []


# Loads optimizer-ready FPL player data through the optimizer serving path.
@st.cache_data
def load_player_data(_engine):
    try:
        if _engine is None:
            st.error("Cannot load player data because the database connection failed.")
            return pd.DataFrame(), "unavailable"

        from src.fpl_optimizer import get_player_points_for_optimizer

        result = get_player_points_for_optimizer(_engine)
        if isinstance(result, tuple):
            df, points_mode = result
        else:
            df = result
            points_mode = "unknown"

        if df is None or df.empty:
            st.error("Could not load optimizer-ready FPL player data.")
            return pd.DataFrame(), points_mode

        return df, points_mode
    except Exception as error:
        st.warning(f"FPL XGBoost loading failed. Falling back to rule-based optimizer: {error}")
        try:
            from src.feature_engineering import load_player_features
            from src.fpl_optimizer import predict_player_points

            fallback_df = load_player_features(_engine)
            if fallback_df is None or fallback_df.empty:
                st.error("Could not load fallback player_fpl_features.")
                return pd.DataFrame(), "rule-based fallback"

            return predict_player_points(fallback_df), "rule-based fallback"
        except Exception as fallback_error:
            st.error(f"Could not load fallback FPL player data: {fallback_error}")
            return pd.DataFrame(), "unavailable"


# Refreshes current FPL player metadata without touching history, models, or match features.
def refresh_latest_fpl_player_data():
    try:
        from src.data_pipeline import refresh_player_data_only

        return refresh_player_data_only()
    except Exception as error:
        raise RuntimeError(str(error)) from error


# Builds one match prediction row in the exact saved feature order.
def build_match_feature_row(engine, home_team, away_team, feature_cols):
    try:
        home_query = sqlalchemy.text(
            """
            SELECT home_form_scored, home_form_conceded,
                   home_clean_sheet_rate, home_fdr,
                   strength_overall_home, home_team_away_str,
                   home_xg_last5, home_xga_last5
            FROM match_features
            WHERE home_team_name = :home_team
            ORDER BY gameweek DESC, fixture_id DESC
            LIMIT 1
            """
        )

        away_query = sqlalchemy.text(
            """
            SELECT away_form_scored, away_form_conceded,
                   away_clean_sheet_rate, away_fdr,
                   strength_overall_away, away_team_home_str,
                   away_xg_last5, away_xga_last5
            FROM match_features
            WHERE away_team_name = :away_team
            ORDER BY gameweek DESC, fixture_id DESC
            LIMIT 1
            """
        )

        home_data = pd.read_sql(home_query, engine, params={"home_team": home_team})
        away_data = pd.read_sql(away_query, engine, params={"away_team": away_team})

        if home_data.empty or away_data.empty:
            st.error("Could not find recent data for one or both teams.")
            return None

        row = {
            "home_form_scored": home_data["home_form_scored"].iloc[0],
            "home_form_conceded": home_data["home_form_conceded"].iloc[0],
            "home_clean_sheet_rate": home_data["home_clean_sheet_rate"].iloc[0],
            "away_form_scored": away_data["away_form_scored"].iloc[0],
            "away_form_conceded": away_data["away_form_conceded"].iloc[0],
            "away_clean_sheet_rate": away_data["away_clean_sheet_rate"].iloc[0],
            "home_fdr": home_data["home_fdr"].iloc[0],
            "away_fdr": away_data["away_fdr"].iloc[0],
            "strength_overall_home": home_data["strength_overall_home"].iloc[0],
            "strength_overall_away": away_data["strength_overall_away"].iloc[0],
            "home_team_away_str": home_data["home_team_away_str"].iloc[0],
            "away_team_home_str": away_data["away_team_home_str"].iloc[0],
            "home_xg_last5": home_data["home_xg_last5"].iloc[0],
            "home_xga_last5": home_data["home_xga_last5"].iloc[0],
            "away_xg_last5": away_data["away_xg_last5"].iloc[0],
            "away_xga_last5": away_data["away_xga_last5"].iloc[0],
        }

        missing_features = [col for col in feature_cols if col not in row]
        if missing_features:
            st.warning(f"Some model features defaulted to 0: {missing_features}")
            for feature in missing_features:
                row[feature] = 0.0

        prediction_df = pd.DataFrame([row]).reindex(columns=feature_cols).fillna(0.0)
        if list(prediction_df.columns) != list(feature_cols):
            st.error("Prediction feature order does not match model_features.json.")
            return None

        return prediction_df
    except Exception as error:
        st.error(f"Could not build match feature row: {error}")
        return None


# Forces scoreline display to agree with the highest probability result.
def align_score_with_result(best_result, pred_home, pred_away):
    if best_result == "H" and pred_home <= pred_away:
        pred_home = pred_away + 1
    elif best_result == "A" and pred_away <= pred_home:
        pred_away = pred_home + 1
    elif best_result == "D":
        avg = math.floor((pred_home + pred_away) / 2)
        pred_home = avg
        pred_away = avg

    return pred_home, pred_away


# Normalizes squad numeric columns and computes captain score safely.
def prepare_squad_for_display(squad):
    squad = squad.copy()
    squad["estimated_points"] = pd.to_numeric(
        squad["estimated_points"],
        errors="coerce",
    ).fillna(0)

    if "form" in squad.columns:
        squad["form"] = pd.to_numeric(squad["form"], errors="coerce").fillna(0)
    else:
        squad["form"] = 0.0

    default_captain_score = squad["estimated_points"] + (0.4 * squad["form"])
    if "captain_score" in squad.columns:
        squad["captain_score"] = pd.to_numeric(
            squad["captain_score"],
            errors="coerce",
        ).fillna(default_captain_score)
    else:
        squad["captain_score"] = default_captain_score

    return squad


# Returns a display-friendly squad table with optional model serving fields.
def format_squad_table(pos_players):
    table = pos_players.copy()
    first_name = table.get("first_name", pd.Series("", index=table.index)).fillna("")
    second_name = table.get("second_name", pd.Series("", index=table.index)).fillna("")
    table["Player"] = (first_name.astype(str) + " " + second_name.astype(str)).str.strip()

    rename_map = {
        "team_name": "Club",
        "squad_role": "Role",
        "price": "Price",
        "form": "Form",
        "raw_estimated_points": "Raw Est. Pts",
        "start_probability": "Start Prob.",
        "estimated_points": "Est. Pts",
    }
    table = table.rename(columns=rename_map)
    if "Start Prob." in table.columns:
        table["Start Prob."] = pd.to_numeric(
            table["Start Prob."],
            errors="coerce",
        ).fillna(0) * 100

    display_cols = [
        "Player",
        "Club",
        "Role",
        "Price",
        "Form",
        "Raw Est. Pts",
        "Start Prob.",
        "Est. Pts",
    ]
    display_cols = [col for col in display_cols if col in table.columns]

    return table[display_cols]


def squad_table_config():
    return {
        "Player": st.column_config.TextColumn("Player", width="large"),
        "Club": st.column_config.TextColumn("Club", width="medium"),
        "Role": st.column_config.TextColumn("Role", width="small"),
        "Price": st.column_config.NumberColumn("Price", format="%.1f"),
        "Form": st.column_config.NumberColumn("Form", format="%.1f"),
        "Raw Est. Pts": st.column_config.NumberColumn("Raw Est. Pts", format="%.2f"),
        "Start Prob.": st.column_config.NumberColumn("Start Prob.", format="%.0f%%"),
        "Est. Pts": st.column_config.NumberColumn("Est. Pts", format="%.2f"),
    }


engine = load_engine()
teams_list = load_teams(engine)
models_for_sidebar = load_models()
_, active_classifier_label = get_active_classifier(models_for_sidebar)

with st.sidebar:
    st.title("Football Predictor")
    st.caption("Premier League ML Lab")
    st.markdown("---")

    page = st.radio(
        "Pages",
        ["Match Predictor", "FPL Optimizer", "Model Info"],
    )

    st.markdown("---")
    st.caption("Status")
    render_status_chips(
        [
            ("Database", "connected" if engine is not None else "disconnected"),
            ("Classifier", active_classifier_label or "unavailable"),
        ]
    )

    st.markdown("---")
    st.caption("Data: FPL Official API + Understat")
    st.caption("Models: XGBoost + Logistic Regression")

if page == "Match Predictor":
    render_header(
        "Match Predictor",
        "Premier League Match Predictor",
        (
            "Predict Win, Draw, and Loss probabilities using rolling form, team strength, "
            "xG, and fixture difficulty."
        ),
    )

    if engine is None:
        st.stop()

    if len(teams_list) < 2:
        st.error("Not enough teams found in match_features.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams_list, key="home_team")
    with col2:
        away_team = st.selectbox(
            "Away Team",
            teams_list,
            index=1,
            key="away_team",
        )

    same_team = home_team == away_team
    if same_team:
        st.warning("Home and Away team cannot be the same.")

    predict_btn = st.button(
        "Predict Match",
        type="primary",
        use_container_width=True,
    )

    if predict_btn and not same_team:
        with st.spinner("Calculating prediction..."):
            models = load_models()
            if models is None:
                st.stop()

            feature_cols = load_model_features()
            X_pred = build_match_feature_row(engine, home_team, away_team, feature_cols)
            if X_pred is None:
                st.stop()

            try:
                classifier, classifier_label = get_active_classifier(models)
                if classifier is None:
                    st.error("No classifier is available for prediction.")
                    st.stop()

                scaler = models["scaler"]
                le = models["label_encoder"]
                home_model = models["home_goals"]
                away_model = models["away_goals"]

                if classifier_label == "XGBoost":
                    proba = classifier.predict_proba(X_pred)[0]
                else:
                    X_scaled = scaler.transform(X_pred)
                    proba = classifier.predict_proba(X_scaled)[0]

                classes = le.classes_
                prob_dict = dict(zip(classes, proba))
                home_prob = clip_probability(prob_dict.get("H", 0.0))
                draw_prob = clip_probability(prob_dict.get("D", 0.0))
                away_prob = clip_probability(prob_dict.get("A", 0.0))

                best_result = max(prob_dict, key=prob_dict.get)
                result_label = {
                    "H": f"{home_team} Win",
                    "D": "Draw",
                    "A": f"{away_team} Win",
                }[best_result]

                raw_home = float(home_model.predict(X_pred)[0])
                raw_away = float(away_model.predict(X_pred)[0])
                pred_home = max(0, round(raw_home))
                pred_away = max(0, round(raw_away))
                pred_home, pred_away = align_score_with_result(
                    best_result,
                    pred_home,
                    pred_away,
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.stop()

        st.markdown("### Match Result")
        scoreline = f"{home_team} {pred_home} - {pred_away} {away_team}"
        margin_confidence = confidence_from_margin([home_prob, draw_prob, away_prob])
        render_match_result_card(
            home_team,
            away_team,
            result_label,
            margin_confidence,
            scoreline,
        )
        render_match_scoreboard(home_team, away_team, pred_home, pred_away, result_label)
        confidence = max(home_prob, draw_prob, away_prob)
        render_metric_cards(
            [
                ("Model", classifier_label, "Active classifier"),
                ("Confidence", margin_confidence, f"Top probability: {confidence:.1%}"),
            ]
        )

        st.markdown("### Win Probabilities")
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        prob_col1.metric(f"{home_team} Win", f"{home_prob:.1%}")
        prob_col2.metric("Draw", f"{draw_prob:.1%}")
        prob_col3.metric(f"{away_team} Win", f"{away_prob:.1%}")
        render_probability_bars(
            [
                (f"{home_team} Win", home_prob),
                ("Draw", draw_prob),
                (f"{away_team} Win", away_prob),
            ]
        )
        render_stacked_probability_bar(home_team, away_team, home_prob, draw_prob, away_prob)
        render_status_panel(
            explain_match_result(
                best_result,
                home_team,
                away_team,
                home_prob,
                draw_prob,
                away_prob,
            )
        )
        st.markdown(
            '<div class="caption-muted">Score is a rough estimate. Win, draw, and loss probabilities are the primary output.</div>',
            unsafe_allow_html=True,
        )

elif page == "FPL Optimizer":
    render_header(
        "Squad Optimizer",
        "FPL Optimizer",
        (
            "Build an optimized 15-player squad within budget using the FPL points model "
            "and linear programming."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget", 95.0, 100.0, 100.0, 0.5, format="GBP %.1fm")
    with col2:
        chip = st.selectbox(
            "Active Chip",
            ["None", "Bench Boost", "Wildcard", "Free Hit"],
        )

    if chip in ["Wildcard", "Free Hit"]:
        st.warning(
            "Wildcard and Free Hit are not fully implemented yet. "
            "Running standard squad optimization."
        )

    refresh_message = st.session_state.pop("fpl_refresh_success", None)
    if refresh_message:
        st.success(refresh_message)

    refresh_btn = st.button(
        "Refresh latest FPL player data",
        use_container_width=True,
    )

    if refresh_btn:
        try:
            with st.spinner("Refreshing latest FPL player data..."):
                refreshed_rows = refresh_latest_fpl_player_data()
            st.session_state["fpl_refresh_success"] = (
                "Latest FPL player data refreshed successfully. "
                f"Players table rows: {refreshed_rows}"
            )
            st.cache_data.clear()
            st.rerun()
        except Exception as error:
            st.error(f"Refresh failed: {error}")

    player_df, points_mode = load_player_data(engine)
    render_status_panel(f"Using FPL points model: {points_mode}")
    if points_mode == "rule-based fallback":
        st.warning("XGBoost model unavailable. Using rule-based fallback points.")

    optimize_btn = st.button(
        "Pick My Team",
        type="primary",
        use_container_width=True,
    )

    if optimize_btn:
        with st.spinner("Optimizing squad..."):
            from src.fpl_optimizer import optimize_squad

            chip_map = {
                "None": None,
                "Bench Boost": "bench_boost",
                "Wildcard": None,
                "Free Hit": None,
            }
            if player_df.empty:
                st.error("Could not load FPL player data for optimization.")
                st.stop()

            try:
                squad = optimize_squad(player_df, budget=budget, chip=chip_map[chip])
            except Exception as error:
                st.error(f"FPL optimization failed: {error}")
                st.stop()

        if squad is None or len(squad) == 0:
            st.error("Could not find a valid squad with the current budget and constraints.")
        else:
            squad = prepare_squad_for_display(squad)

            captain_pool = squad
            if "is_starter" in squad.columns and (squad["is_starter"] == 1).any():
                captain_pool = squad[squad["is_starter"] == 1]

            captain = captain_pool.sort_values("captain_score", ascending=False).iloc[0]
            total_cost = squad["price"].sum()
            total_pts = squad["estimated_points"].sum()
            remaining_budget = budget - total_cost

            captain_name = f"{captain['first_name']} {captain['second_name']}"

            st.markdown("### Squad Summary")
            render_metric_cards(
                [
                    ("Total Predicted Points", f"{total_pts:.1f}", None),
                    ("Budget Used", f"GBP {total_cost:.1f}m", f"Remaining: GBP {remaining_budget:.1f}m"),
                    ("Captain", captain_name, captain.get("team_name", "")),
                ]
            )

            st.markdown("---")
            st.markdown("### Squad Builder")
            render_squad_pitch(squad)

            st.markdown("---")
            st.markdown("### Captain Recommendation")
            render_captain_panel(captain)
            st.markdown(
                (
                    '<div class="caption-muted">'
                    'Captain score uses estimated points with a form adjustment. '
                    'Starter roles are preferred when available.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            with st.expander("View Detailed Table"):
                for pos in [1, 2, 3, 4]:
                    pos_players = squad[squad["position"] == pos].copy()
                    pos_players = pos_players.sort_values(
                        "estimated_points",
                        ascending=False,
                    )

                    st.subheader(POSITION_LABELS[pos])
                    st.dataframe(
                        format_squad_table(pos_players),
                        use_container_width=True,
                        hide_index=True,
                        column_config=squad_table_config(),
                    )

elif page == "Model Info":
    render_header(
        "Model Report",
        "Premier League ML Lab",
        "A Tier 2 football analytics system for match probabilities and FPL squad optimization.",
    )

    st.markdown("### Architecture")
    render_metric_cards(
        [
            ("Tier 1", "Baseline", "Core FPL data and logistic model foundation"),
            ("Tier 2", "XGBoost + xG", "Understat features, rolling form, and FPL history"),
            ("Serving", "Streamlit + Supabase", "Cloud dashboard backed by Postgres"),
        ]
    )

    st.markdown("### Credibility Snapshot")
    render_model_credibility_cards()

    st.markdown("### Model Stack")
    st.write("- XGBoost classifier for match Win, Draw, and Loss probabilities")
    st.write("- Logistic Regression fallback for match classification")
    st.write("- XGBoost regressors for home and away goal estimates")
    st.write("- XGBoost regressor for FPL player points")
    st.write("- PuLP linear optimizer for squad selection under FPL constraints")

    st.markdown("### Current Metrics")
    render_metric_cards(
        [
            ("XGBoost Match Accuracy", "0.570", None),
            ("Logistic Match Accuracy", "0.532", None),
            ("FPL XGBoost MAE", "0.926", None),
            ("FPL XGBoost R2", "0.334", None),
        ]
    )

    st.markdown("### Data Sources")
    render_status_panel(
        "The app combines official FPL player, fixture, price, availability, and gameweek data with Understat xG and tactical team-history features."
    )

    st.markdown("### Limitations")
    render_disclaimer()

    st.markdown("### Built By")
    st.write("Purav Desai, B.Tech IT, SCET Surat")
