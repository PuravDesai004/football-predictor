import json
import math
import sys
import warnings
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
    page_title="Football ML Predictor",
    layout="wide",
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
    display_cols = [
        "first_name",
        "second_name",
        "team_name",
        "squad_role",
        "price",
        "form",
        "raw_estimated_points",
        "start_probability",
        "estimated_points",
    ]
    display_cols = [col for col in display_cols if col in pos_players.columns]

    table = pos_players[display_cols].rename(
        columns={
            "first_name": "First",
            "second_name": "Last",
            "team_name": "Club",
            "squad_role": "Role",
            "price": "Price",
            "form": "Form",
            "raw_estimated_points": "Raw Est. Pts",
            "start_probability": "Start Prob.",
            "estimated_points": "Est. Pts",
        }
    )

    return table


engine = load_engine()
teams_list = load_teams(engine)
models_for_sidebar = load_models()
_, active_classifier_label = get_active_classifier(models_for_sidebar)

with st.sidebar:
    st.title("Football ML")
    st.caption("Premier League Predictor + FPL Optimizer")
    st.markdown("---")

    page = st.radio(
        "Pages",
        ["Match Predictor", "FPL Team Selector", "About"],
    )

    st.markdown("---")
    st.caption("Status")
    st.write(f"Database: {'connected' if engine is not None else 'disconnected'}")
    st.write(f"Classifier: {active_classifier_label or 'unavailable'}")

    st.markdown("---")
    st.caption("Data: FPL Official API + Understat")
    st.caption("Models: XGBoost + Logistic Regression")

if page == "Match Predictor":
    st.title("Premier League Match Predictor")
    st.caption(
        "Predicts Win/Draw/Loss probability and expected score using rolling form, "
        "team strength, xG, and fixture difficulty."
    )
    st.markdown("---")

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

        st.markdown("### Prediction")
        result_col, score_col = st.columns(2)
        result_col.metric("Predicted Result", result_label)
        score_col.metric(
            "Predicted Score",
            f"{home_team} {pred_home} - {pred_away} {away_team}",
        )

        st.markdown("### Win Probabilities")
        col_h, col_d, col_a = st.columns(3)
        with col_h:
            st.metric(f"{home_team} Win", f"{home_prob:.1%}")
            st.progress(home_prob)
        with col_d:
            st.metric("Draw", f"{draw_prob:.1%}")
            st.progress(draw_prob)
        with col_a:
            st.metric(f"{away_team} Win", f"{away_prob:.1%}")
            st.progress(away_prob)

        st.caption("Score is a rough estimate. Win/Draw/Loss probabilities are more reliable.")

elif page == "FPL Team Selector":
    st.title("FPL Team Selector")
    st.caption(
        "Picks an optimized 15-man squad within budget using the FPL points model "
        "and PuLP linear programming."
    )
    st.markdown("---")

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
    st.info(f"Using FPL points model: {points_mode}")
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

            st.markdown("### Squad Summary")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Cost", f"GBP {total_cost:.1f}m")
            mc2.metric("Est. Total Points", f"{total_pts:.1f}")
            mc3.metric("Captain", f"{captain['first_name']} {captain['second_name']}")

            st.markdown("---")
            st.markdown("### Squad")
            for pos in [1, 2, 3, 4]:
                pos_players = squad[squad["position"] == pos].copy()
                pos_players = pos_players.sort_values("estimated_points", ascending=False)

                st.subheader(POSITION_LABELS[pos])
                st.dataframe(
                    format_squad_table(pos_players),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("---")
            st.markdown("### Captain Recommendation")
            start_prob = captain.get("start_probability", 1.0)
            if pd.isna(start_prob):
                start_prob = 1.0

            st.info(
                f"{captain['first_name']} {captain['second_name']} "
                f"({captain['team_name']})\n\n"
                f"Price: GBP {captain['price']:.1f}m\n\n"
                f"Form: {captain['form']:.1f}\n\n"
                f"Start Probability: {start_prob:.0%}\n\n"
                f"Estimated Points: {captain['estimated_points']:.1f}"
            )
            st.caption(
                "Captain score = Estimated Points + 0.4 x Form. "
                "Only starters are considered when starter roles are available."
            )

elif page == "About":
    st.title("About This Project")
    st.caption("Football ML Prediction System")
    st.markdown("---")

    st.markdown("### Project Summary")
    st.write(
        "This app predicts Premier League match outcomes and builds optimized FPL squads. "
        "It combines official FPL API data, Understat xG data, time-safe feature engineering, "
        "machine learning models, and linear programming."
    )

    st.markdown("### Models")
    st.write("- XGBoost Classifier for match Win/Draw/Loss probabilities")
    st.write("- Logistic Regression as match classifier fallback")
    st.write("- XGBoost Regressors for home and away goal estimates")
    st.write("- XGBoost Regressor for FPL player points")
    st.write("- PuLP optimizer for FPL squad selection")

    st.markdown("### Current Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("XGBoost Match Accuracy", "0.570")
    m2.metric("Logistic Match Accuracy", "0.532")
    m3.metric("FPL XGBoost MAE", "0.926")
    m4.metric("FPL XGBoost R2", "0.334")

    st.markdown("### Data Sources")
    st.write("- FPL Official API: players, prices, availability, fixtures, gameweek history")
    st.write("- Understat: xG, xGA, tactical team-history data")

    st.markdown("### Built By")
    st.write("Purav Desai, B.Tech IT, SCET Surat")
