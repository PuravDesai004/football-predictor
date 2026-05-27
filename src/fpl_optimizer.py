import os
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


# Uses player stats to estimate expected FPL points next gameweek.
# We do not have a trained points model yet, so this rule-based formula
# approximates FPL scoring closely until it is replaced by XGBoost in Tier 2.
def predict_player_points(df):
    df = df.copy()

    numeric_cols = [
        "form",
        "clean_sheets",
        "minutes_ratio",
        "goals_scored",
        "assists",
        "is_available",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

        availability_penalty = 1 if row["is_available"] == 1 else 0
        return (base + (bonus_pts / 38.0)) * availability_penalty

    df["estimated_points"] = df.apply(calculate_points, axis=1)

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


# Uses PuLP linear programming to find the mathematically optimal
# 15-man FPL squad within all official rules.
def optimize_squad(df, budget=100.0, chip=None):
    df = df[df["is_available"] == 1].copy()
    df = df.reset_index(drop=True)

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
        return pd.DataFrame()

    solver_status = pulp.LpStatus[prob.status]
    if solver_status != "Optimal":
        print(f"Warning: Solver status is {solver_status}. Squad may not be optimal.")

    selected = [i for i in players_list if x[i].value() == 1]
    squad_df = df.loc[selected].copy()
    squad_df["is_starter"] = [int(starter[i].value() == 1) for i in selected]
    squad_df["squad_role"] = np.where(squad_df["is_starter"] == 1, "Starter", "Bench")
    squad_df = squad_df.sort_values(
        ["is_starter", "position", "estimated_points"], ascending=[False, True, False]
    )

    return squad_df


# Prints the selected squad in a readable format grouped by position.
def display_squad(squad_df):
    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    total_price = squad_df["price"].sum()
    total_pts = squad_df["estimated_points"].sum()

    print("\n── OPTIMAL FPL SQUAD ───────────────────────────────")
    print(f"  Total Cost: £{total_price:.1f}m  |  Estimated Points: {total_pts:.1f}")
    print("─" * 55)

    for pos in [1, 2, 3, 4]:
        pos_players = squad_df[squad_df["position"] == pos]
        print(f"\n  {pos_names[pos]}:")
        for _, row in pos_players.iterrows():
            role = row.get("squad_role", "Squad")
            print(
                f"    {row['first_name']} {row['second_name']:<20}"
                f"  {row['team_name']:<15}"
                f"  {role:<7}"
                f"  £{row['price']:.1f}m"
                f"  est. {row['estimated_points']:.1f}pts"
            )


# Picks the captain using ceiling logic instead of just average points.
def pick_captain(squad_df):
    squad_df = squad_df.copy()
    squad_df["captain_score"] = squad_df["estimated_points"] + (0.4 * squad_df["form"])
    captain_pool = squad_df
    if "is_starter" in squad_df.columns and (squad_df["is_starter"] == 1).any():
        captain_pool = squad_df[squad_df["is_starter"] == 1]

    captain = captain_pool.sort_values("captain_score", ascending=False).iloc[0]

    print("\n── CAPTAIN RECOMMENDATION ──────────────────────────")
    print(
        f"  Captain : {captain['first_name']} {captain['second_name']}"
        f"  ({captain['team_name']})  £{captain['price']:.1f}m"
    )
    print(
        f"  Form: {captain['form']}  |  "
        f"Est. Points: {captain['estimated_points']:.1f}  |  "
        f"Captain Score: {captain['captain_score']:.2f}"
    )

    return captain


if __name__ == "__main__":
    from src.data_pipeline import get_engine
    from src.feature_engineering import load_player_features

    engine = get_engine()
    df = load_player_features(engine)
    if df is None:
        print("Failed to load player data")
        sys.exit()

    # Run normal squad optimization.
    print("\n=== NORMAL SQUAD (no chip) ===")
    df = predict_player_points(df)
    squad = optimize_squad(df, budget=100.0, chip=None)
    display_squad(squad)
    pick_captain(squad)

    # Run bench boost version for comparison.
    print("\n=== BENCH BOOST SQUAD ===")
    bb_squad = optimize_squad(df, budget=100.0, chip="bench_boost")
    display_squad(bb_squad)
    pick_captain(bb_squad)

    print("\nDay 6 complete. FPL optimizer working.")
