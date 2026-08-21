"""Ingest completed current-season FPL gameweek live stats into an isolated v3 table."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEASON = "2026-27"
FIXTURE_TABLE = "production_fpl_fixture_snapshots"
BOOTSTRAP_TABLE = "production_fpl_bootstrap_snapshots"
OUTPUT_TABLE = "production_fpl_gameweek_snapshots_v3"
HTTP_TIMEOUT_SECONDS = 30
DB_CONNECT_TIMEOUT_SECONDS = 5
LIVE_EVENT_URL = "https://fantasy.premierleague.com/api/event/{gameweek}/live/"

STAT_FIELDS = {
    "minutes": ("minutes", 0),
    "total_points": ("total_points", 0),
    "goals_scored": ("goals_scored", 0),
    "assists": ("assists", 0),
    "clean_sheets": ("clean_sheets", 0),
    "saves": ("saves", 0),
    "bonus": ("bonus", 0),
    "starts": ("starts", 0),
    "expected_goals": ("expected_goals", None),
    "expected_assists": ("expected_assists", None),
    "expected_goal_involvements": ("expected_goal_involvements", None),
    "expected_goals_conceded": ("expected_goals_conceded", None),
}


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
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
                snapshot_id BIGSERIAL PRIMARY KEY,
                ingestion_run_id INTEGER NULL REFERENCES production_ingestion_runs(run_id),
                target_season TEXT NOT NULL,
                gameweek INTEGER NOT NULL,
                snapshot_time TIMESTAMP NOT NULL,
                player_id INTEGER NOT NULL,
                fpl_code INTEGER NULL,
                player_name TEXT NOT NULL,
                team_id INTEGER NULL,
                position_id INTEGER NULL,
                minutes INTEGER NOT NULL DEFAULT 0,
                total_points INTEGER NOT NULL DEFAULT 0,
                goals_scored INTEGER NOT NULL DEFAULT 0,
                assists INTEGER NOT NULL DEFAULT 0,
                clean_sheets INTEGER NOT NULL DEFAULT 0,
                saves INTEGER NOT NULL DEFAULT 0,
                bonus INTEGER NOT NULL DEFAULT 0,
                starts INTEGER NOT NULL DEFAULT 0,
                expected_goals FLOAT NULL,
                expected_assists FLOAT NULL,
                expected_goal_involvements FLOAT NULL,
                expected_goals_conceded FLOAT NULL,
                identity_status TEXT NOT NULL,
                raw_stats_json JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (target_season, gameweek, player_id),
                CHECK (gameweek > 0),
                CHECK (minutes >= 0),
                CHECK (identity_status IN ('matched', 'unmatched'))
            )
        """))


def _latest_ingestion_run_id(engine, season: str) -> int | None:
    with engine.connect() as connection:
        value = connection.execute(text("""
            SELECT MAX(run_id)
            FROM production_ingestion_runs
            WHERE target_season = :season
        """), {"season": season}).scalar_one()
    return int(value) if value is not None else None


def _latest_fixture_run_id(engine, season: str) -> int | None:
    with engine.connect() as connection:
        value = connection.execute(text(f"""
            SELECT MAX(f.run_id)
            FROM {FIXTURE_TABLE} f
            JOIN production_ingestion_runs r ON r.run_id = f.run_id
            WHERE r.target_season = :season
        """), {"season": season}).scalar_one()
    return int(value) if value is not None else None


def resolve_completed_gameweek(engine, fixture_run_id: int, requested: int | None) -> int | None:
    season_start_year = None
    with engine.connect() as connection:
        run_season = connection.execute(text("""
            SELECT target_season FROM production_ingestion_runs WHERE run_id = :run_id
        """), {"run_id": fixture_run_id}).scalar_one_or_none()
        latest_kickoff = connection.execute(text(f"""
            SELECT MAX(kickoff_time) FROM {FIXTURE_TABLE} WHERE run_id = :run_id
        """), {"run_id": fixture_run_id}).scalar_one()
    if run_season:
        season_start_year = int(str(run_season)[:4])
    if season_start_year is None or latest_kickoff is None:
        return None
    if latest_kickoff.date() < datetime(season_start_year, 7, 1).date():
        print("SKIPPED_STALE_FIXTURE_SNAPSHOT")
        return None
    if requested is None:
        query = text(f"""
            SELECT MAX(event_id)
            FROM {FIXTURE_TABLE}
            WHERE run_id = :run_id AND event_id IS NOT NULL AND finished = TRUE
        """)
        with engine.connect() as connection:
            value = connection.execute(query, {"run_id": fixture_run_id}).scalar_one()
        return int(value) if value is not None else None

    query = text(f"""
        SELECT COUNT(*) AS fixture_count,
               SUM(CASE WHEN finished THEN 1 ELSE 0 END) AS finished_count
        FROM {FIXTURE_TABLE}
        WHERE run_id = :run_id AND event_id = :gameweek
    """)
    with engine.connect() as connection:
        row = connection.execute(query, {"run_id": fixture_run_id, "gameweek": requested}).mappings().one()
    if not row["fixture_count"] or row["finished_count"] != row["fixture_count"]:
        return None
    return int(requested)


def fetch_live_event(gameweek: int) -> list[dict[str, Any]]:
    url = LIVE_EVENT_URL.format(gameweek=gameweek)
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"FPL live GW{gameweek} request failed: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"FPL live GW{gameweek} returned invalid JSON") from error
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list) or not elements:
        raise RuntimeError(f"FPL live GW{gameweek} returned no player elements")
    return elements


def load_player_identity(engine, bootstrap_run_id: int) -> dict[int, dict[str, Any]]:
    query = text(f"""
        SELECT DISTINCT ON (player_id)
            player_id, player_name, team_id, position_id, raw_player_json
        FROM {BOOTSTRAP_TABLE}
        WHERE run_id = :run_id
        ORDER BY player_id, snapshot_id DESC
    """)
    with engine.connect() as connection:
        rows = connection.execute(query, {"run_id": bootstrap_run_id}).mappings().all()
    result = {}
    for row in rows:
        raw = row["raw_player_json"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        code = raw.get("code") if isinstance(raw, dict) else None
        try:
            code = int(code) if code is not None else None
        except (TypeError, ValueError):
            code = None
        result[int(row["player_id"])] = {**dict(row), "fpl_code": code}
    return result


def _number(value: Any, integer: bool = False):
    if value in (None, ""):
        return None if not integer else 0
    try:
        return int(float(value)) if integer else float(value)
    except (TypeError, ValueError):
        return None if not integer else 0


def build_rows(elements, identity: dict[int, dict[str, Any]], season: str, gameweek: int, run_id: int | None) -> list[dict[str, Any]]:
    snapshot_time = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for element in elements:
        player_id = int(element.get("id"))
        stats = element.get("stats") or {}
        source = identity.get(player_id, {})
        row = {
            "ingestion_run_id": run_id,
            "target_season": season,
            "gameweek": gameweek,
            "snapshot_time": snapshot_time,
            "player_id": player_id,
            "fpl_code": source.get("fpl_code"),
            "player_name": source.get("player_name") or str(player_id),
            "team_id": source.get("team_id"),
            "position_id": source.get("position_id"),
            "identity_status": "matched" if source.get("fpl_code") is not None else "unmatched",
            "raw_stats_json": json.dumps(element, sort_keys=True),
        }
        for column, (source_key, default) in STAT_FIELDS.items():
            value = stats.get(source_key, default)
            row[column] = _number(value, integer=column in {"minutes", "total_points", "goals_scored", "assists", "clean_sheets", "saves", "bonus", "starts"})
        rows.append(row)
    return rows


def write_rows(engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns if column not in {"target_season", "gameweek", "player_id", "created_at"})
    query = text(f"""
        INSERT INTO {OUTPUT_TABLE} ({', '.join(columns)})
        VALUES ({', '.join(':' + column for column in columns)})
        ON CONFLICT (target_season, gameweek, player_id)
        DO UPDATE SET {assignments}, updated_at = CURRENT_TIMESTAMP
    """)
    with engine.begin() as connection:
        connection.execute(query, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest completed FPL gameweek live stats into v3")
    parser.add_argument("--target-season", default=DEFAULT_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    parser.add_argument("--init-schema-only", action="store_true")
    args = parser.parse_args()
    if args.target_season == "2025-26":
        raise RuntimeError("2025-26 is reserved as a final holdout and cannot be a production target")
    engine = get_engine()
    init_schema(engine)
    if args.init_schema_only:
        print("PASS: production_fpl_gameweek_snapshots_v3 exists")
        return
    fixture_run_id = _latest_fixture_run_id(engine, args.target_season)
    bootstrap_run_id = _latest_ingestion_run_id(engine, args.target_season)
    if fixture_run_id is None or bootstrap_run_id is None:
        print("SKIPPED_NO_FPL_SNAPSHOTS")
        return
    gameweek = resolve_completed_gameweek(engine, fixture_run_id, args.target_gameweek)
    if gameweek is None:
        print("SKIPPED_NO_COMPLETED_GAMEWEEK")
        return
    try:
        elements = fetch_live_event(gameweek)
    except RuntimeError as error:
        print(f"SKIPPED_FPL_LIVE_UNAVAILABLE: {error}")
        return
    identity = load_player_identity(engine, bootstrap_run_id)
    rows = build_rows(elements, identity, args.target_season, gameweek, bootstrap_run_id)
    before = _count(engine)
    written = write_rows(engine, rows)
    after = _count(engine)
    print(f"Completed gameweek: {gameweek}")
    print(f"Live player rows received: {len(elements)}")
    print(f"Rows upserted: {written}")
    print(f"Snapshot table rows: {before} -> {after}")
    print(f"Matched fpl_code rows: {sum(row['identity_status'] == 'matched' for row in rows)}")
    print("No Tier 2 or historical v3 tables were modified.")


def _count(engine) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {OUTPUT_TABLE}")).scalar_one())


if __name__ == "__main__":
    main()
