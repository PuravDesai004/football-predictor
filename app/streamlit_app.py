import os
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

st.set_page_config(
    page_title="Football ML Predictor",
    page_icon="⚽",
    layout="wide",
)


# Loads the PostgreSQL engine once and keeps it cached for the app session.
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


# Loads all saved models from models/saved so predictions can run without retraining.
@st.cache_resource
def load_models():
    try:
        model_files = {
            "lr": "logistic_classifier.pkl",
            "xgb_clf": "xgb_classifier.pkl",
            "scaler": "scaler.pkl",
            "label_encoder": "label_encoder.pkl",
            "home_goals": "xgb_home_goals.pkl",
            "away_goals": "xgb_away_goals.pkl",
        }

        models = {}
        for key, filename in model_files.items():
            path = MODELS_DIR / filename
            if not path.exists():
                st.error(f"Missing model file: {path}")
                return None
            models[key] = joblib.load(path)

        return models
    except Exception as error:
        st.error(f"Failed to load models: {error}")
        return None


# Loads the team names used by the match predictor dropdowns.
@st.cache_data
def load_teams(_engine):
    try:
        if _engine is None:
            st.error("Cannot load teams because the database connection failed.")
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


# Loads player features and estimates next-gameweek FPL points.
@st.cache_data
def load_player_data(_engine):
    try:
        if _engine is None:
            st.error("Cannot load player data because the database connection failed.")
            return pd.DataFrame()

        from src.feature_engineering import load_player_features
        from src.fpl_optimizer import predict_player_points

        df = load_player_features(_engine)
        if df is None or df.empty:
            st.error("Could not load player features from PostgreSQL.")
            return pd.DataFrame()

        return predict_player_points(df)
    except Exception as error:
        st.error(f"Could not load player data: {error}")
        return pd.DataFrame()


# Pulls the latest feature row for the selected home and away teams.
def build_match_feature_row(engine, home_team, away_team):
    home_query = sqlalchemy.text(
        """
        SELECT home_form_scored, home_form_conceded,
               home_clean_sheet_rate, home_fdr,
               strength_overall_home, home_team_away_str
        FROM match_features
        WHERE home_team_name = :home_team
        ORDER BY gameweek DESC
        LIMIT 1
        """
    )

    away_query = sqlalchemy.text(
        """
        SELECT away_form_scored, away_form_conceded,
               away_clean_sheet_rate, away_fdr,
               strength_overall_away, away_team_home_str
        FROM match_features
        WHERE away_team_name = :away_team
        ORDER BY gameweek DESC
        LIMIT 1
        """
    )

    home_data = pd.read_sql(home_query, engine, params={"home_team": home_team})
    away_data = pd.read_sql(away_query, engine, params={"away_team": away_team})

    if home_data.empty or away_data.empty:
        return None

    feature_cols = [
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
    ]

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
    }

    return pd.DataFrame([row])[feature_cols].fillna(0)


# Forces the displayed scoreline to agree with the highest probability result.
def align_score_with_result(best_result, pred_home, pred_away):
    if best_result == "H" and pred_home <= pred_away:
        pred_home = pred_away + 1
    elif best_result == "A" and pred_away <= pred_home:
        pred_away = pred_home + 1
    elif best_result == "D":
        avg = round((pred_home + pred_away) / 2)
        pred_home = avg
        pred_away = avg

    return pred_home, pred_away


engine = load_engine()
teams_list = load_teams(engine)

st.sidebar.title("⚽ Football ML")
st.sidebar.markdown("Premier League Predictor + FPL Optimizer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Match Predictor", "FPL Team Selector", "About"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: FPL Official API + football-data.org")
st.sidebar.caption("Model: Logistic Regression + XGBoost")

if page == "Match Predictor":
    st.title("⚽ Premier League Match Predictor")
    st.markdown(
        "Predicts Win/Draw/Loss probability and expected score "
        "using rolling form, team strength, and fixture difficulty."
    )
    st.markdown("---")

    if engine is None:
        st.stop()

    if len(teams_list) < 2:
        st.error("Not enough teams found in match_features.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select Home Team", teams_list, key="home_team")
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox(
            "Select Away Team",
            teams_list,
            index=1,
            key="away_team",
        )

    if home_team == away_team:
        st.warning("⚠️ Home and Away team are the same.")

    st.markdown("---")
    predict_btn = st.button(
        "🔮 Predict Match",
        type="primary",
        use_container_width=True,
    )

    if predict_btn:
        if home_team == away_team:
            st.error("Home and Away team cannot be the same.")
        else:
            with st.spinner("Calculating prediction..."):
                models = load_models()
                if models is None:
                    st.stop()

                X_pred = build_match_feature_row(engine, home_team, away_team)
                if X_pred is None:
                    st.error("Could not find recent data for one or both teams.")
                    st.stop()

                lr = models["lr"]
                scaler = models["scaler"]
                le = models["label_encoder"]
                home_model = models["home_goals"]
                away_model = models["away_goals"]

                X_scaled = scaler.transform(X_pred)
                proba = lr.predict_proba(X_scaled)[0]
                classes = le.classes_

                prob_dict = dict(zip(classes, proba))
                home_prob = prob_dict.get("H", 0)
                draw_prob = prob_dict.get("D", 0)
                away_prob = prob_dict.get("A", 0)

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

                st.success(f"**Predicted Result: {result_label}**")

                st.markdown("### Win Probabilities")
                col_h, col_d, col_a = st.columns(3)
                with col_h:
                    st.metric(f"🏠 {home_team}", f"{home_prob:.1%}")
                    st.progress(home_prob)
                with col_d:
                    st.metric("🤝 Draw", f"{draw_prob:.1%}")
                    st.progress(draw_prob)
                with col_a:
                    st.metric(f"✈️ {away_team}", f"{away_prob:.1%}")
                    st.progress(away_prob)

                st.markdown("### Predicted Score")
                st.markdown(
                    f"<h2 style='text-align:center'>"
                    f"{home_team} &nbsp; {pred_home} — {pred_away} "
                    f"&nbsp; {away_team}</h2>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "⚠️ Score is a rough estimate. "
                    "Win/Draw/Loss probabilities are more reliable."
                )

elif page == "FPL Team Selector":
    st.title("🏆 FPL Team Selector")
    st.markdown(
        "Picks the optimal 15-man squad within your budget "
        "using current player form and availability."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget (£m)", 95.0, 100.0, 100.0, 0.5)
    with col2:
        chip = st.selectbox(
            "Active Chip",
            ["None", "Bench Boost", "Wildcard", "Free Hit"],
        )

    st.markdown("---")
    optimize_btn = st.button(
        "⚡ Pick My Team",
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
            player_df = load_player_data(engine)
            if player_df.empty:
                st.stop()

            squad = optimize_squad(player_df, budget=budget, chip=chip_map[chip])

            if squad is None or len(squad) == 0:
                st.error("Could not find a valid squad. Try increasing budget.")
            else:
                if "captain_score" not in squad.columns:
                    squad["captain_score"] = (
                        squad["estimated_points"] + 0.4 * squad["form"]
                    )

                captain_pool = squad
                if "is_starter" in squad.columns and (squad["is_starter"] == 1).any():
                    captain_pool = squad[squad["is_starter"] == 1]

                captain = captain_pool.sort_values("captain_score", ascending=False).iloc[0]

                total_cost = squad["price"].sum()
                total_pts = squad["estimated_points"].sum()

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total Cost", f"£{total_cost:.1f}m")
                mc2.metric("Est. Total Points", f"{total_pts:.1f}")
                mc3.metric(
                    "Captain",
                    f"{captain['first_name']} {captain['second_name']}",
                )

                st.markdown("---")

                pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
                pos_emojis = {1: "🧤", 2: "🛡️", 3: "⚙️", 4: "⚽"}

                for pos in [1, 2, 3, 4]:
                    pos_players = squad[squad["position"] == pos].copy()
                    pos_players = pos_players.sort_values(
                        "estimated_points",
                        ascending=False,
                    )
                    st.subheader(f"{pos_emojis[pos]} {pos_names[pos]}")
                    display_cols = [
                        "first_name",
                        "second_name",
                        "team_name",
                        "squad_role",
                        "price",
                        "form",
                        "estimated_points",
                    ]
                    display_cols = [
                        col for col in display_cols if col in pos_players.columns
                    ]
                    st.dataframe(
                        pos_players[display_cols].rename(
                            columns={
                                "first_name": "First",
                                "second_name": "Last",
                                "team_name": "Club",
                                "squad_role": "Role",
                                "price": "Price (£m)",
                                "form": "Form",
                                "estimated_points": "Est. Pts",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("---")
                st.subheader("⭐ Captain Pick")
                cap_col1, cap_col2 = st.columns(2)
                with cap_col1:
                    st.info(
                        f"**{captain['first_name']} "
                        f"{captain['second_name']}**\n\n"
                        f"Club: {captain['team_name']}\n\n"
                        f"Price: £{captain['price']:.1f}m\n\n"
                        f"Form: {captain['form']}\n\n"
                        f"Est. Points: {captain['estimated_points']:.1f}"
                    )
                with cap_col2:
                    st.markdown(
                        "**Why this captain?**\n\n"
                        "Captain score = Estimated Points + (0.4 × Form)\n\n"
                        "This picks the highest ceiling player, not just the "
                        "highest average, because captains score double points."
                    )

elif page == "About":
    st.title("📊 About This Project")
    st.markdown(
        """
        ## Football ML Prediction System — Tier 1

        Built by **Purav Desai** | B.Tech IT, SCET Surat

        ---

        ### What This App Does
        - **Match Predictor**: Predicts Win/Draw/Loss probability and expected
          score for any PL fixture
        - **FPL Team Selector**: Picks the mathematically optimal 15-man fantasy
          squad within budget using PuLP optimizer

        ### Data Sources
        - FPL Official API (player stats, prices, form)
        - football-data.org (match results, fixtures)

        ### Models Used
        - **Logistic Regression** — Win/Draw/Loss probabilities
        - **XGBoost Regressor** — Goal predictions
        - **PuLP Linear Programming** — Squad optimization

        ### Key Features Engineered
        - Rolling form last 5 matches (home and away separately)
        - Clean sheet probability per team
        - Fixture difficulty ratings
        - Team strength ratings
        - Points per million value score

        ### Domain Problems Solved
        - **Style bias**: Teams like Arsenal with low xG but high defensive
          value are handled via strength ratings
        - **Data leakage fixed**: Time-based train/test split, H2H features
          removed until multi-season data available
        - **Overfitting prevented**: 5-fold cross-validation on all models

        ---

        🔗 [GitHub](https://github.com/PuravDesai004) |
        💼 [LinkedIn](https://linkedin.com/in/puravdesai41)

        *Tier 1 of 5 — Sentiment layer, LSTM, and UCL predictor coming in
        future tiers*
        """
    )
