"""Generate current FPL v3 player predictions and an optimized squad."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pulp
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from build_fpl_features_v3 import calculate_rolling_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEASON = "2026-27"
BOOTSTRAP_TABLE = "production_fpl_bootstrap_snapshots"
FIXTURE_TABLE = "production_fpl_fixture_snapshots"
PREDICTION_TABLE = "fpl_player_predictions_v3"
OPTIMIZER_TABLE = "fpl_optimizer_outputs_v3"
HISTORY_TABLE = "fpl_player_gameweek_history_v3"
IDENTITY_TABLE = "fpl_player_identity_map_v3"
LIVE_GAMEWEEK_TABLE = "production_fpl_gameweek_snapshots_v3"
FINAL_HOLDOUT_SEASON = "2025-26"
DB_CONNECT_TIMEOUT_SECONDS = 5
SQUAD_BUDGET = 1000
MAX_PLAYERS_PER_TEAM = 3

MODEL_FEATURE_COLUMNS = [
    "prior_points_last1", "prior_points_last3", "prior_points_last5",
    "prior_points_last10", "prior_points_season", "prior_minutes_last3",
    "prior_minutes_last5", "prior_minutes_last10", "prior_appearances_last5",
    "prior_starts_last5", "prior_goals_last5", "prior_assists_last5",
    "prior_bonus_last5", "prior_clean_sheets_last5", "prior_saves_last5",
    "prior_xg_last5", "prior_xa_last5", "prior_points_per_90",
    "prior_minutes_total", "prior_gameweeks_played",
]

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_LIMITS = {1: 2, 2: 5, 3: 5, 4: 3}
STARTER_LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def get_database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        url = make_url(database_url)
        if url.host and url.host.lower() == "localhost":
            url = url.set(host="127.0.0.1")
        return url.render_as_string(hide_password=False)
    values = {key: os.getenv(key) for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS")}
    if values["DB_HOST"] and values["DB_HOST"].lower() == "localhost":
        values["DB_HOST"] = "127.0.0.1"
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing database settings: {missing}")
    return "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        values["DB_USER"], values["DB_PASS"], values["DB_HOST"], values["DB_PORT"], values["DB_NAME"]
    )


def get_engine():
    database_url = get_database_url()
    url = make_url(database_url)
    connect_args: dict[str, Any] = {"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS}
    if url.host and url.host not in {"127.0.0.1", "localhost"} and "sslmode" not in database_url.lower():
        connect_args["sslmode"] = "require"
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    with engine.connect():
        pass
    return engine


def init_schema(engine) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS fpl_player_predictions_v3 (
            prediction_id BIGSERIAL PRIMARY KEY,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            fpl_code INTEGER NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NULL,
            position_id INTEGER NOT NULL,
            now_cost INTEGER NOT NULL,
            predicted_points FLOAT NOT NULL,
            availability_factor FLOAT NOT NULL,
            expected_points FLOAT NOT NULL,
            status TEXT NOT NULL,
            chance_of_playing INTEGER NULL,
            source_feature_season TEXT NULL,
            source_feature_gameweek INTEGER NULL,
            source_bootstrap_run_id INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (target_season, target_gameweek, player_id),
            CHECK (target_gameweek > 0),
            CHECK (position_id IN (1, 2, 3, 4)),
            CHECK (now_cost >= 0),
            CHECK (availability_factor >= 0 AND availability_factor <= 1),
            CHECK (expected_points >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fpl_optimizer_outputs_v3 (
            optimizer_output_id BIGSERIAL PRIMARY KEY,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NOT NULL,
            generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            budget_limit INTEGER NOT NULL,
            squad_cost INTEGER NOT NULL,
            squad_json JSONB NOT NULL,
            starting_xi_json JSONB NOT NULL,
            captain_player_id INTEGER NOT NULL,
            vice_captain_player_id INTEGER NOT NULL,
            objective_value FLOAT NOT NULL,
            source_prediction_count INTEGER NOT NULL,
            UNIQUE (target_season, target_gameweek),
            CHECK (target_gameweek > 0),
            CHECK (budget_limit >= 0),
            CHECK (squad_cost >= 0),
            CHECK (source_prediction_count > 0)
        )
        """,
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _count(engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def _latest_target_gameweek(engine, run_id: int) -> int | None:
    with engine.connect() as connection:
        value = connection.execute(text(f"""
            SELECT MIN(event_id)
            FROM {FIXTURE_TABLE}
            WHERE run_id = :run_id
              AND finished = FALSE
              AND event_id IS NOT NULL
        """), {"run_id": run_id}).scalar_one()
    return int(value) if value is not None else None


def load_latest_snapshots(engine, target_gameweek: int | None) -> tuple[pd.DataFrame, int, int]:
    with engine.connect() as connection:
        bootstrap_run_id = connection.execute(text(f"SELECT MAX(run_id) FROM {BOOTSTRAP_TABLE}")).scalar_one()
        fixture_run_id = connection.execute(text(f"SELECT MAX(run_id) FROM {FIXTURE_TABLE}")).scalar_one()
    if bootstrap_run_id is None or fixture_run_id is None:
        raise RuntimeError("Production FPL snapshots are empty")
    if target_gameweek is None:
        target_gameweek = _latest_target_gameweek(engine, int(fixture_run_id))
    if target_gameweek is None:
        return pd.DataFrame(), int(bootstrap_run_id), None

    query = text(f"""
        SELECT DISTINCT ON (player_id)
            player_id, player_name, team_id, team_name, position_id, now_cost,
            status, chance_of_playing_this_round, chance_of_playing_next_round,
            event_id, raw_player_json
        FROM {BOOTSTRAP_TABLE}
        WHERE run_id = :run_id
          AND (event_id = :gameweek OR event_id IS NULL)
        ORDER BY player_id, snapshot_id DESC
    """)
    frame = pd.read_sql(query, engine, params={"run_id": int(bootstrap_run_id), "gameweek": target_gameweek})
    if frame.empty:
        return pd.DataFrame(), int(bootstrap_run_id), int(target_gameweek)
    frame["fpl_code"] = frame["raw_player_json"].map(_extract_fpl_code)
    frame["position_id"] = pd.to_numeric(frame["position_id"], errors="coerce").astype("Int64")
    frame = frame[frame["position_id"].isin(POSITION_NAMES)].copy()
    frame["target_gameweek"] = int(target_gameweek)
    frame["bootstrap_run_id"] = int(bootstrap_run_id)
    return frame, int(bootstrap_run_id), int(target_gameweek)


def _extract_fpl_code(raw: Any) -> int | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("code")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _season_order(season: str) -> int:
    try:
        return int(str(season)[:4])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid FPL season label: {season}") from exc


def _sum_preserving_null(series: pd.Series) -> float | None:
    if series.notna().any():
        return float(series.fillna(0).sum())
    return None


RUNTIME_STAT_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "bonus",
    "starts",
    "expected_goals",
    "expected_assists",
]


def _load_runtime_history(engine, target_season: str, target_gameweek: int) -> pd.DataFrame:
    history_query = text(f"""
        SELECT
            h.id AS source_history_row_id,
            h.season,
            h.gameweek,
            h.player_source_id,
            h.player_name,
            im.fpl_code,
            h.total_points,
            h.minutes,
            h.goals_scored,
            h.assists,
            h.clean_sheets,
            h.saves,
            h.bonus,
            h.starts,
            h.expected_goals,
            h.expected_assists
        FROM {HISTORY_TABLE} h
        JOIN {IDENTITY_TABLE} im
          ON im.season = h.season
         AND im.player_source_id = h.player_source_id
        WHERE im.fpl_code IS NOT NULL
    """)
    live_query = text(f"""
        SELECT
            snapshot_id AS source_history_row_id,
            target_season AS season,
            gameweek,
            CAST(fpl_code AS TEXT) AS player_source_id,
            player_name,
            fpl_code,
            total_points,
            minutes,
            goals_scored,
            assists,
            clean_sheets,
            saves,
            bonus,
            starts,
            expected_goals,
            expected_assists
        FROM {LIVE_GAMEWEEK_TABLE}
        WHERE target_season = :target_season
          AND gameweek < :target_gameweek
          AND identity_status = 'matched'
          AND fpl_code IS NOT NULL
    """)
    history = pd.read_sql(history_query, engine)
    live = pd.read_sql(
        live_query,
        engine,
        params={"target_season": target_season, "target_gameweek": target_gameweek},
    )
    if not live.empty and (pd.to_numeric(live["gameweek"], errors="coerce") >= target_gameweek).any():
        raise RuntimeError("Current-gameweek live rows entered the prior-only runtime feature set")
    if history.empty and live.empty:
        raise RuntimeError("No historical or current-season FPL rows are available for runtime features")
    combined = pd.concat([history, live], ignore_index=True)
    combined["fpl_code"] = pd.to_numeric(combined["fpl_code"], errors="coerce")
    combined = combined[combined["fpl_code"].notna()].copy()
    combined["fpl_code"] = combined["fpl_code"].astype(int)
    combined["season_order"] = combined["season"].map(_season_order)
    for column in RUNTIME_STAT_COLUMNS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def _aggregate_runtime_rows(rows: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, Any] = {
        "source_history_row_id": ("source_history_row_id", "min"),
        "player_source_id": ("player_source_id", "first"),
        "player_name": ("player_name", "first"),
        "season_order": ("season_order", "first"),
        "total_points": ("total_points", _sum_preserving_null),
        "minutes": ("minutes", _sum_preserving_null),
        "goals_scored": ("goals_scored", _sum_preserving_null),
        "assists": ("assists", _sum_preserving_null),
        "clean_sheets": ("clean_sheets", _sum_preserving_null),
        "saves": ("saves", _sum_preserving_null),
        "bonus": ("bonus", _sum_preserving_null),
        "starts": ("starts", _sum_preserving_null),
        "expected_goals": ("expected_goals", _sum_preserving_null),
        "expected_assists": ("expected_assists", _sum_preserving_null),
    }
    return (
        rows.groupby(["fpl_code", "season", "gameweek"], as_index=False, sort=False)
        .agg(**aggregations)
        .sort_values(["fpl_code", "season_order", "gameweek", "source_history_row_id"])
        .reset_index(drop=True)
    )


def load_current_prior_features(
    engine,
    target_season: str,
    target_gameweek: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build the pre-target feature row from history plus completed live GWs."""
    rows = _aggregate_runtime_rows(_load_runtime_history(engine, target_season, target_gameweek))
    feature_parts: list[pd.DataFrame] = []
    for fpl_code, group in rows.groupby("fpl_code", sort=False):
        group = group.copy()
        latest_name = str(group["player_name"].dropna().iloc[-1]) if group["player_name"].notna().any() else ""
        synthetic = {column: np.nan for column in group.columns}
        synthetic.update({
            "fpl_code": int(fpl_code),
            "season": target_season,
            "gameweek": int(target_gameweek),
            "season_order": _season_order(target_season),
            "source_history_row_id": int(group["source_history_row_id"].max()) + 1,
            "player_source_id": str(fpl_code),
            "player_name": latest_name,
        })
        target_row = pd.DataFrame([synthetic], columns=group.columns)
        rolled = calculate_rolling_features(pd.concat([group, target_row], ignore_index=True))
        feature_parts.append(rolled.iloc[[-1]][["fpl_code", "season", "gameweek", *feature_columns]])
    if not feature_parts:
        return pd.DataFrame(columns=["fpl_code", "season", "gameweek", *feature_columns])
    features = pd.concat(feature_parts, ignore_index=True)
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features[["fpl_code", "season", "gameweek", *feature_columns]]


def load_model_artifacts(artifact_dir: Path) -> tuple[Any, list[str]]:
    model_path = artifact_dir / "fpl_points_v3_candidate.pkl"
    features_path = artifact_dir / "fpl_points_v3_candidate_features.json"
    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"Missing v3 candidate artifacts in {artifact_dir}")
    with features_path.open("r", encoding="utf-8") as handle:
        feature_columns = json.load(handle)
    if feature_columns != MODEL_FEATURE_COLUMNS:
        raise RuntimeError("Candidate feature artifact does not match the v3 feature contract")
    return joblib.load(model_path), feature_columns


def build_predictions(players: pd.DataFrame, prior_features: pd.DataFrame, model: Any, feature_columns: list[str]) -> pd.DataFrame:
    merged = players.merge(prior_features, on="fpl_code", how="left", suffixes=("", "_prior"))
    raw_chance = merged["chance_of_playing_this_round"].fillna(100)
    merged["availability_factor"] = (pd.to_numeric(raw_chance, errors="coerce").fillna(100) / 100).clip(0, 1)
    unavailable = merged["status"].astype(str).str.lower().isin({"i", "n", "s", "u"})
    merged.loc[unavailable, "availability_factor"] = 0.0
    values = merged[feature_columns].copy()
    predictions = np.asarray(model.predict(values), dtype=float)
    merged["predicted_points"] = np.maximum(predictions, 0.0)
    merged["expected_points"] = merged["predicted_points"] * merged["availability_factor"]
    merged["source_feature_season"] = merged["season"]
    merged["source_feature_gameweek"] = merged["gameweek"]
    merged["fpl_code"] = pd.to_numeric(merged["fpl_code"], errors="coerce").astype("Int64")
    return merged


def optimize_squad(predictions: pd.DataFrame) -> dict[str, Any]:
    eligible = predictions[predictions["availability_factor"] > 0].copy()
    if len(eligible) < 15:
        raise RuntimeError(f"Only {len(eligible)} eligible players are available; 15 are required")
    problem = pulp.LpProblem("fpl_v3_squad", pulp.LpMaximize)
    choices = {int(row.player_id): pulp.LpVariable(f"player_{int(row.player_id)}", cat="Binary") for row in eligible.itertuples()}
    by_id = eligible.set_index("player_id")
    problem += pulp.lpSum(float(by_id.loc[player_id, "expected_points"]) * variable for player_id, variable in choices.items())
    problem += pulp.lpSum(choices.values()) == 15
    problem += pulp.lpSum(int(by_id.loc[player_id, "now_cost"]) * variable for player_id, variable in choices.items()) <= SQUAD_BUDGET
    for position_id, quota in POSITION_LIMITS.items():
        problem += pulp.lpSum(variable for player_id, variable in choices.items() if int(by_id.loc[player_id, "position_id"]) == position_id) == quota
    for team_id in eligible["team_id"].dropna().unique():
        problem += pulp.lpSum(variable for player_id, variable in choices.items() if int(by_id.loc[player_id, "team_id"]) == int(team_id)) <= MAX_PLAYERS_PER_TEAM
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Squad optimizer status: {pulp.LpStatus[status]}")
    squad = eligible[eligible["player_id"].map(lambda player_id: choices[int(player_id)].value() == 1)].copy()
    return _select_starting_xi(squad)


def _select_starting_xi(squad: pd.DataFrame) -> dict[str, Any]:
    starter_problem = pulp.LpProblem("fpl_v3_starting_xi", pulp.LpMaximize)
    choices = {int(row.player_id): pulp.LpVariable(f"starter_{int(row.player_id)}", cat="Binary") for row in squad.itertuples()}
    by_id = squad.set_index("player_id")
    starter_problem += pulp.lpSum(float(by_id.loc[player_id, "expected_points"]) * variable for player_id, variable in choices.items())
    starter_problem += pulp.lpSum(choices.values()) == 11
    for position_id, (minimum, maximum) in STARTER_LIMITS.items():
        expression = pulp.lpSum(variable for player_id, variable in choices.items() if int(by_id.loc[player_id, "position_id"]) == position_id)
        starter_problem += expression >= minimum
        starter_problem += expression <= maximum
    status = starter_problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Starting XI optimizer status: {pulp.LpStatus[status]}")
    starters = squad[squad["player_id"].map(lambda player_id: choices[int(player_id)].value() == 1)].copy()
    ranked = starters.sort_values("expected_points", ascending=False)
    captain = int(ranked.iloc[0]["player_id"])
    vice_captain = int(ranked.iloc[1]["player_id"])
    return {
        "squad": _records(squad),
        "starting_xi": _records(starters),
        "captain_player_id": captain,
        "vice_captain_player_id": vice_captain,
        "budget_limit": SQUAD_BUDGET,
        "squad_cost": int(squad["now_cost"].sum()),
        "objective_value": float(squad["expected_points"].sum()),
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["player_id", "fpl_code", "player_name", "team_id", "team_name", "position_id", "now_cost", "expected_points", "availability_factor"]
    output = frame[columns].copy()
    output["position"] = output["position_id"].map(POSITION_NAMES)
    return json.loads(output.to_json(orient="records"))


def write_outputs(engine, predictions: pd.DataFrame, optimized: dict[str, Any], target_season: str, target_gameweek: int, bootstrap_run_id: int) -> None:
    prediction_rows = predictions[[
        "player_id", "fpl_code", "player_name", "team_id", "team_name", "position_id", "now_cost",
        "predicted_points", "availability_factor", "expected_points", "status",
        "chance_of_playing_this_round", "source_feature_season", "source_feature_gameweek",
    ]].copy()
    prediction_rows = prediction_rows.where(pd.notna(prediction_rows), None)
    with engine.begin() as connection:
        connection.execute(text(f"DELETE FROM {PREDICTION_TABLE} WHERE target_season = :season AND target_gameweek = :gameweek"), {"season": target_season, "gameweek": target_gameweek})
        for row in prediction_rows.to_dict(orient="records"):
            connection.execute(text(f"""
                INSERT INTO {PREDICTION_TABLE} (
                    target_season, target_gameweek, player_id, fpl_code, player_name,
                    team_id, team_name, position_id, now_cost, predicted_points,
                    availability_factor, expected_points, status, chance_of_playing,
                    source_feature_season, source_feature_gameweek, source_bootstrap_run_id
                ) VALUES (
                    :season, :gameweek, :player_id, :fpl_code, :player_name,
                    :team_id, :team_name, :position_id, :now_cost, :predicted_points,
                    :availability_factor, :expected_points, :status, :chance_of_playing,
                    :source_feature_season, :source_feature_gameweek, :run_id
                )
            """), {**row, "season": target_season, "gameweek": target_gameweek, "run_id": bootstrap_run_id, "chance_of_playing": row.pop("chance_of_playing_this_round")})
        connection.execute(text(f"DELETE FROM {OPTIMIZER_TABLE} WHERE target_season = :season AND target_gameweek = :gameweek"), {"season": target_season, "gameweek": target_gameweek})
        connection.execute(text(f"""
            INSERT INTO {OPTIMIZER_TABLE} (
                target_season, target_gameweek, budget_limit, squad_cost,
                squad_json, starting_xi_json, captain_player_id,
                vice_captain_player_id, objective_value, source_prediction_count
            ) VALUES (
                :season, :gameweek, :budget, :cost, CAST(:squad AS JSONB),
                CAST(:starting AS JSONB), :captain, :vice, :objective, :count
            )
        """), {
            "season": target_season, "gameweek": target_gameweek, "budget": optimized["budget_limit"],
            "cost": optimized["squad_cost"], "squad": json.dumps(optimized["squad"]),
            "starting": json.dumps(optimized["starting_xi"]), "captain": optimized["captain_player_id"],
            "vice": optimized["vice_captain_player_id"], "objective": optimized["objective_value"],
            "count": len(predictions),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Tier 3 FPL predictions and optimize a squad")
    parser.add_argument("--target-season", default=DEFAULT_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_season == FINAL_HOLDOUT_SEASON:
        raise RuntimeError("2025-26 is reserved as a final holdout and cannot be a production target")
    engine = get_engine()
    init_schema(engine)
    if args.init_schema_only:
        print("PASS: v3 prediction and optimizer tables verified")
        return
    players, bootstrap_run_id, target_gameweek = load_latest_snapshots(engine, args.target_gameweek)
    if target_gameweek is None or players.empty:
        print("SKIPPED_NO_UPCOMING_FIXTURES")
        return
    model, feature_columns = load_model_artifacts(args.artifact_dir)
    prior_features = load_current_prior_features(
        engine,
        args.target_season,
        target_gameweek,
        feature_columns,
    )
    predictions = build_predictions(players, prior_features, model, feature_columns)
    optimized = optimize_squad(predictions)
    print(f"Predictions generated: {len(predictions)}")
    print(f"Target: {args.target_season} GW{target_gameweek}")
    print(f"Squad cost: {optimized['squad_cost']} / {optimized['budget_limit']}")
    print(f"Captain player_id: {optimized['captain_player_id']}")
    print(f"Vice-captain player_id: {optimized['vice_captain_player_id']}")
    if not args.dry_run:
        before_predictions = _count(engine, PREDICTION_TABLE)
        before_optimizer = _count(engine, OPTIMIZER_TABLE)
        write_outputs(engine, predictions, optimized, args.target_season, target_gameweek, bootstrap_run_id)
        print(f"Prediction rows: {before_predictions} -> {_count(engine, PREDICTION_TABLE)}")
        print(f"Optimizer rows: {before_optimizer} -> {_count(engine, OPTIMIZER_TABLE)}")
    else:
        print("DRY RUN: no prediction or optimizer rows written")


if __name__ == "__main__":
    main()
