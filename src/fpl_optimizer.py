import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pulp
import sqlalchemy
from dotenv import load_dotenv


warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

FPL_MODEL_PATH = MODELS_DIR / "fpl_points_xgb.pkl"
FPL_FEATURES_PATH = MODELS_DIR / "fpl_points_features.json"


# Loads the trained FPL XGBoost points model and exact feature order.
def load_fpl_points_model():
    try:
        if not FPL_MODEL_PATH.exists() or not FPL_FEATURES_PATH.exists():
            print("FPL XGBoost model unavailable. Falling back to rule-based points.")
            return None

        model = joblib.load(FPL_MODEL_PATH)
        with FPL_FEATURES_PATH.open("r", encoding="utf-8") as file:
            feature_names = json.load(file)

        if not isinstance(feature_names, list) or not feature_names:
            print("FPL XGBoost model unavailable. Falling back to rule-based points.")
            return None

        return model, feature_names
    except Exception:
        print("FPL XGBoost model unavailable. Falling back to rule-based points.")
        return None


# Loads one latest leakage-safe feature row per player for model-backed estimates.
def load_latest_player_gameweek_features(engine):
    try:
        columns_df = pd.read_sql(
            sqlalchemy.text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'players'
                """
            ),
            engine,
        )
        player_columns = set(columns_df["column_name"].tolist())
    except Exception:
        player_columns = set()

    status_select = "p.status" if "status" in player_columns else "NULL AS status"
    chance_this_select = (
        "p.chance_of_playing_this_round"
        if "chance_of_playing_this_round" in player_columns
        else "NULL::FLOAT AS chance_of_playing_this_round"
    )
    chance_next_select = (
        "p.chance_of_playing_next_round"
        if "chance_of_playing_next_round" in player_columns
        else "NULL::FLOAT AS chance_of_playing_next_round"
    )

    query = sqlalchemy.text(
        f"""
        WITH ranked_features AS (
            SELECT
                pgf.*,
                ROW_NUMBER() OVER (
                    PARTITION BY pgf.player_id
                    ORDER BY
                        pgf.gameweek DESC,
                        pgf.kickoff_time DESC NULLS LAST,
                        pgf.fixture DESC NULLS LAST
                ) AS row_rank
            FROM player_gameweek_features pgf
        )
        SELECT
            ranked_features.*,
            pff.first_name AS optimizer_first_name,
            pff.second_name AS optimizer_second_name,
            pff.team,
            pff.team_name,
            pff.position AS optimizer_position,
            pff.price,
            pff.is_available,
            pff.form,
            {status_select},
            {chance_this_select},
            {chance_next_select},
            pff.minutes AS season_minutes,
            pff.goals_scored,
            pff.assists,
            pff.clean_sheets,
            pff.minutes_ratio
        FROM ranked_features
        LEFT JOIN player_fpl_features pff
            ON ranked_features.player_id = pff.player_id
        LEFT JOIN players p
            ON ranked_features.player_id = p.player_id
        WHERE ranked_features.row_rank = 1
        ORDER BY ranked_features.player_id
        """
    )

    df = pd.read_sql(query, engine)
    print(f"Loaded latest FPL ML feature rows: {len(df)}")

    if "optimizer_first_name" in df.columns:
        df["first_name"] = df["optimizer_first_name"].fillna(df["first_name"])
    if "optimizer_second_name" in df.columns:
        df["second_name"] = df["optimizer_second_name"].fillna(df["second_name"])
    if "optimizer_position" in df.columns:
        df["position"] = df["optimizer_position"].fillna(df["position"])

    return df


# Calculates a transparent rule-based likelihood that a player starts the next match.
def calculate_start_probability(row):
    def has_value(value):
        return value is not None and not pd.isna(value)

    is_available = row.get("is_available", False)
    if not has_value(is_available) or not bool(is_available):
        return 0.0

    status = row.get("status", None)
    status = str(status).lower().strip() if has_value(status) else None
    if status in ["i", "s"]:
        return 0.0

    next_round_chance = row.get("chance_of_playing_next_round", None)
    this_round_chance = row.get("chance_of_playing_this_round", None)

    if has_value(next_round_chance):
        base = float(next_round_chance) / 100.0
    elif has_value(this_round_chance):
        base = float(this_round_chance) / 100.0
    elif status == "a" or status is None:
        base = 0.95
    elif status == "d":
        base = 0.50
    else:
        base = 0.50

    minutes_avg_last5 = pd.to_numeric(
        pd.Series([row.get("minutes_avg_last5", 0.0)]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    starts_avg_last5 = pd.to_numeric(
        pd.Series([row.get("starts_avg_last5", 0.0)]),
        errors="coerce",
    ).fillna(0.0).iloc[0]
    season_minutes = pd.to_numeric(
        pd.Series([row.get("season_minutes", 0.0)]),
        errors="coerce",
    ).fillna(0.0).iloc[0]

    if starts_avg_last5 >= 0.8 or minutes_avg_last5 >= 70:
        base += 0.05
    if minutes_avg_last5 < 30 and season_minutes > 0:
        base -= 0.10
    if season_minutes == 0:
        base = min(base, 0.20)

    return float(np.clip(base, 0.0, 1.0))


# Adjusts point estimates by a rule-based start probability before optimization.
def apply_start_probability_adjustment(df):
    df = df.copy()

    if "estimated_points" not in df.columns:
        df["estimated_points"] = 0.0

    df["estimated_points"] = pd.to_numeric(
        df["estimated_points"],
        errors="coerce",
    ).fillna(0.0)

    if "raw_estimated_points" not in df.columns:
        df["raw_estimated_points"] = df["estimated_points"]

    df["start_probability"] = df.apply(calculate_start_probability, axis=1)
    df["estimated_points"] = np.clip(
        df["estimated_points"] * df["start_probability"],
        0.0,
        None,
    )

    print(f"FPL start probability avg: {df['start_probability'].mean():.3f}")
    print(f"FPL start probability below 0.5: {int((df['start_probability'] < 0.5).sum())}")
    print(f"FPL start probability zero: {int((df['start_probability'] == 0).sum())}")

    return df


# Applies practical guardrails to stop cold-start players from dominating ML rankings.
def apply_fpl_prediction_sanity_rules(df):
    df = df.copy()

    defaults = {
        "season_minutes": 0.0,
        "minutes_avg_last5": 0.0,
        "starts_avg_last5": 0.0,
        "history_matches_last5": 0.0,
        "estimated_points": 0.0,
        "is_available": False,
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    numeric_columns = [
        "season_minutes",
        "minutes_avg_last5",
        "starts_avg_last5",
        "history_matches_last5",
        "estimated_points",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["is_available"] = df["is_available"].fillna(False).astype(bool)
    df["is_playing_candidate"] = (
        (df["season_minutes"] >= 90)
        | (df["minutes_avg_last5"] >= 20)
        | (df["starts_avg_last5"] >= 0.4)
    )

    low_minutes_mask = ~df["is_playing_candidate"]
    low_minutes_capped = int(
        (low_minutes_mask & (df["estimated_points"] > 0.5)).sum()
    )
    df.loc[low_minutes_mask, "estimated_points"] = np.minimum(
        df.loc[low_minutes_mask, "estimated_points"],
        0.5,
    )

    immature_mask = df["history_matches_last5"] < 3
    immature_capped = int((immature_mask & (df["estimated_points"] > 1.0)).sum())
    df.loc[immature_mask, "estimated_points"] = np.minimum(
        df.loc[immature_mask, "estimated_points"],
        1.0,
    )

    unavailable_mask = ~df["is_available"]
    unavailable_zeroed = int((unavailable_mask & (df["estimated_points"] > 0.0)).sum())
    df.loc[unavailable_mask, "estimated_points"] = 0.0

    print(f"FPL sanity guard low-minutes capped players: {low_minutes_capped}")
    print(f"FPL sanity guard immature-history capped players: {immature_capped}")
    print(f"FPL sanity guard unavailable zeroed players: {unavailable_zeroed}")

    return df


# Uses the trained XGBoost model to estimate player points for the optimizer.
def predict_player_points_ml(player_features_df, model, feature_names):
    df = player_features_df.copy()

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        print(f"Missing FPL model features defaulted to 0: {missing_features}")
        for feature in missing_features:
            df[feature] = 0.0

    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].fillna(False).astype(float)

    for feature in feature_names:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float)

    predictions = model.predict(df[feature_names])
    df["raw_estimated_points"] = np.clip(predictions, 0.0, None)
    df["estimated_points"] = df["raw_estimated_points"]
    df["points_model"] = "xgboost"
    df = apply_fpl_prediction_sanity_rules(df)
    df = apply_start_probability_adjustment(df)

    return df


# Uses a rule-based formula as a fallback when the trained model is unavailable.
def predict_player_points_rule_based(df):
    df = df.copy()

    numeric_cols = [
        "form",
        "clean_sheets",
        "minutes_ratio",
        "goals_scored",
        "assists",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["is_available"] = df["is_available"].fillna(False).astype(bool)

    def calculate_points(row):
        base = row["form"]
        bonus_pts = 0

        if row["position"] == 1:
            bonus_pts = row["clean_sheets"] * 6 * row["minutes_ratio"]
        elif row["position"] == 2:
            bonus_pts = (
                (row["goals_scored"] * 6)
                + (row["assists"] * 3)
                + (row["clean_sheets"] * 6 * row["minutes_ratio"])
            )
        elif row["position"] == 3:
            bonus_pts = (row["goals_scored"] * 5) + (row["assists"] * 3)
        elif row["position"] == 4:
            bonus_pts = (row["goals_scored"] * 4) + (row["assists"] * 3)

        availability_penalty = 1 if row["is_available"] else 0
        return (base + (bonus_pts / 38.0)) * availability_penalty

    df["estimated_points"] = df.apply(calculate_points, axis=1)
    df["points_model"] = "rule_based"

    print(f"Points estimated for {len(df)} players")
    print(
        df[
            [
                "first_name",
                "second_name",
                "team_name",
                "position",
                "price",
                "estimated_points",
            ]
        ]
        .sort_values("estimated_points", ascending=False)
        .head(10)
    )

    return df


# Backward-compatible rule-based function used by existing app code.
def predict_player_points(df):
    return predict_player_points_rule_based(df)


# Loads player rows with ML point estimates, falling back to rule-based estimates.
def get_player_points_for_optimizer(engine):
    model_bundle = load_fpl_points_model()

    if model_bundle is not None:
        try:
            model, feature_names = model_bundle
            df = load_latest_player_gameweek_features(engine)

            if df.empty:
                raise ValueError("No latest player_gameweek_features rows loaded.")

            df = predict_player_points_ml(df, model, feature_names)
            print("FPL points mode: XGBoost")
            return df, "XGBoost"
        except Exception as error:
            print(f"Warning: FPL XGBoost prediction failed: {error}")

    from src.feature_engineering import load_player_features

    fallback_df = load_player_features(engine)
    if fallback_df is None or fallback_df.empty:
        print("Failed to load fallback player_fpl_features.")
        return pd.DataFrame(), "rule-based fallback"

    fallback_df = predict_player_points_rule_based(fallback_df)
    print("FPL points mode: rule-based fallback")
    return fallback_df, "rule-based fallback"


# Uses PuLP linear programming to find the mathematically optimal 15-man FPL squad.
def solve_squad_problem(df, budget=100.0, chip=None):
    df = df.reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), "Empty"

    players_list = list(df.index)
    x = pulp.LpVariable.dicts("player", players_list, cat="Binary")
    starter = pulp.LpVariable.dicts("starter", players_list, cat="Binary")

    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

    if chip == "bench_boost":
        # Maximize all 15 players' points.
        prob += pulp.lpSum(df.loc[i, "estimated_points"] * x[i] for i in players_list)
    else:
        # Maximize the starting XI, with a small bench value to break close ties.
        prob += pulp.lpSum(
            (df.loc[i, "estimated_points"] * starter[i])
            + (0.05 * df.loc[i, "estimated_points"] * (x[i] - starter[i]))
            for i in players_list
        )

    prob += pulp.lpSum(x[i] for i in players_list) == 15
    prob += pulp.lpSum(df.loc[i, "price"] * x[i] for i in players_list) <= budget

    gks = df[df["position"] == 1].index.tolist()
    defs = df[df["position"] == 2].index.tolist()
    mids = df[df["position"] == 3].index.tolist()
    fwds = df[df["position"] == 4].index.tolist()

    prob += pulp.lpSum(x[i] for i in gks) == 2
    prob += pulp.lpSum(x[i] for i in defs) == 5
    prob += pulp.lpSum(x[i] for i in mids) == 5
    prob += pulp.lpSum(x[i] for i in fwds) == 3

    # Starting XI constraints for a valid FPL formation.
    for i in players_list:
        prob += starter[i] <= x[i]

    prob += pulp.lpSum(starter[i] for i in players_list) == 11
    prob += pulp.lpSum(starter[i] for i in gks) == 1
    prob += pulp.lpSum(starter[i] for i in defs) >= 3
    prob += pulp.lpSum(starter[i] for i in defs) <= 5
    prob += pulp.lpSum(starter[i] for i in mids) >= 2
    prob += pulp.lpSum(starter[i] for i in mids) <= 5
    prob += pulp.lpSum(starter[i] for i in fwds) >= 1
    prob += pulp.lpSum(starter[i] for i in fwds) <= 3

    for team_id in df["team"].unique():
        team_players = df[df["team"] == team_id].index.tolist()
        prob += pulp.lpSum(x[i] for i in team_players) <= 3

    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    except Exception as error:
        print(f"Error solving FPL optimization: {error}")
        return pd.DataFrame(), "Error"

    solver_status = pulp.LpStatus[prob.status]
    if solver_status != "Optimal":
        return pd.DataFrame(), solver_status

    selected = [i for i in players_list if x[i].value() == 1]
    squad_df = df.loc[selected].copy()
    squad_df["is_starter"] = [int(starter[i].value() == 1) for i in selected]
    squad_df["squad_role"] = np.where(squad_df["is_starter"] == 1, "Starter", "Bench")
    squad_df = squad_df.sort_values(
        ["is_starter", "position", "estimated_points"],
        ascending=[False, True, False],
    )

    return squad_df, solver_status


# Uses PuLP linear programming to find the mathematically optimal 15-man FPL squad.
def optimize_squad(df, budget=100.0, chip=None):
    df = df.copy()

    required_columns = [
        "player_id",
        "first_name",
        "second_name",
        "team",
        "team_name",
        "position",
        "price",
        "is_available",
        "estimated_points",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"Missing optimizer columns: {missing_columns}")
        return pd.DataFrame()

    df["is_available"] = df["is_available"].fillna(False).astype(bool)
    if "is_playing_candidate" not in df.columns:
        df["is_playing_candidate"] = True
    df["is_playing_candidate"] = df["is_playing_candidate"].fillna(False).astype(bool)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(int)
    df["team"] = pd.to_numeric(df["team"], errors="coerce").fillna(0).astype(int)
    df["estimated_points"] = pd.to_numeric(
        df["estimated_points"],
        errors="coerce",
    ).fillna(0)

    candidate_df = df[df["is_available"] & df["is_playing_candidate"]].copy()
    squad_df, solver_status = solve_squad_problem(candidate_df, budget=budget, chip=chip)

    if solver_status == "Optimal":
        return squad_df

    print(
        "WARNING: Playing-candidate filter made optimization infeasible. "
        "Falling back to available players."
    )

    available_df = df[df["is_available"]].copy()
    squad_df, solver_status = solve_squad_problem(available_df, budget=budget, chip=chip)

    if solver_status != "Optimal":
        print(f"Warning: Solver status is {solver_status}. Squad may not be optimal.")

    return squad_df


# Prints the selected squad in a readable format grouped by position.
def display_squad(squad_df):
    if squad_df.empty:
        print("No squad to display.")
        return

    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    total_price = squad_df["price"].sum()
    total_pts = squad_df["estimated_points"].sum()

    print("\n-- OPTIMAL FPL SQUAD --------------------------------")
    print(f"  Total Cost: GBP {total_price:.1f}m  |  Estimated Points: {total_pts:.1f}")
    print("-" * 55)

    for pos in [1, 2, 3, 4]:
        pos_players = squad_df[squad_df["position"] == pos]
        print(f"\n  {pos_names[pos]}:")
        for _, row in pos_players.iterrows():
            role = row.get("squad_role", "Squad")
            print(
                f"    {row['first_name']} {row['second_name']:<20}"
                f"  {row['team_name']:<15}"
                f"  {role:<7}"
                f"  GBP {row['price']:.1f}m"
                f"  est. {row['estimated_points']:.1f}pts"
            )


# Picks the captain using ceiling logic instead of just average points.
def pick_captain(squad_df):
    if squad_df.empty:
        print("No squad available for captain recommendation.")
        return None

    squad_df = squad_df.copy()
    if "form" in squad_df.columns:
        squad_df["form"] = pd.to_numeric(squad_df["form"], errors="coerce").fillna(0)
    else:
        squad_df["form"] = 0.0
    squad_df["captain_score"] = squad_df["estimated_points"] + (0.4 * squad_df["form"])

    captain_pool = squad_df
    if "is_starter" in squad_df.columns and (squad_df["is_starter"] == 1).any():
        captain_pool = squad_df[squad_df["is_starter"] == 1]

    captain = captain_pool.sort_values("captain_score", ascending=False).iloc[0]

    print("\n-- CAPTAIN RECOMMENDATION ---------------------------")
    print(
        f"  Captain : {captain['first_name']} {captain['second_name']}"
        f"  ({captain['team_name']})  GBP {captain['price']:.1f}m"
    )
    print(
        f"  Form: {captain['form']}  |  "
        f"Est. Points: {captain['estimated_points']:.1f}  |  "
        f"Captain Score: {captain['captain_score']:.2f}"
    )

    return captain


if __name__ == "__main__":
    from src.data_pipeline import get_engine

    engine = get_engine()
    if engine is None:
        print("Failed to connect to PostgreSQL.")
        sys.exit(1)

    df, points_mode = get_player_points_for_optimizer(engine)
    if df.empty:
        print("Failed to load player data")
        sys.exit(1)
    print(f"Active FPL points mode: {points_mode}")

    print("\n=== NORMAL SQUAD (no chip) ===")
    squad = optimize_squad(df, budget=100.0, chip=None)
    display_squad(squad)
    pick_captain(squad)

    print("\n=== BENCH BOOST SQUAD ===")
    bb_squad = optimize_squad(df, budget=100.0, chip="bench_boost")
    display_squad(bb_squad)
    pick_captain(bb_squad)

    print("\nDay 6 complete. FPL optimizer working.")
