from __future__ import annotations

import argparse
from io import StringIO
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas
import requests
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
DEFAULT_TARGET_SEASON = "2026-27"
RUN_TYPE = "weekly_ingest"
HTTP_TIMEOUT_SECONDS = 30
FOOTBALL_DATA_2026_27_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
FOOTBALL_DATA_2026_27_FILENAME = "E0_2026-27.csv"
TARGET_MATCH_SEASON = "2026-27"
REQUIRED_FOOTBALL_DATA_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402
from multi_season_pipeline import normalize_team_name  # noqa: E402
from tier3_understat_pipeline import (  # noqa: E402
    fetch_understat_league_data,
    parse_understat_matches,
)


PRODUCTION_TABLES = [
    "production_ingestion_runs",
    "production_fpl_bootstrap_snapshots",
    "production_fpl_fixture_snapshots",
    "production_football_data_match_staging",
    "production_understat_xg_staging",
    "production_data_freshness",
]
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "elo_current_v3",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
    *PRODUCTION_TABLES,
]
ALLOWED_CHANGED_TABLES = {
    *PRODUCTION_TABLES,
    "historical_matches",
    "historical_understat_xg",
}
RUN_STATUSES = {"started", "success", "partial", "failed"}

BOOTSTRAP_REQUIRED_FIELDS = [
    "id",
    "first_name",
    "second_name",
    "web_name",
    "team",
    "element_type",
    "now_cost",
    "total_points",
    "status",
]
FIXTURE_REQUIRED_FIELDS = [
    "id",
    "event",
    "kickoff_time",
    "team_h",
    "team_a",
    "finished",
    "started",
]
SCHEMA_REQUIRED_COLUMNS = {
    "production_ingestion_runs": [
        "run_id",
        "run_started_at",
        "run_finished_at",
        "run_status",
        "run_type",
        "target_season",
        "target_gameweek",
        "deadline_time",
        "fpl_bootstrap_status",
        "fpl_fixtures_status",
        "football_data_status",
        "understat_status",
        "rows_fpl_players",
        "rows_fpl_fixtures",
        "rows_new_results",
        "rows_new_xg",
        "error_message",
        "created_at",
    ],
    "production_fpl_bootstrap_snapshots": [
        "snapshot_id",
        "run_id",
        "snapshot_time",
        "event_id",
        "deadline_time",
        "player_id",
        "player_name",
        "web_name",
        "team_id",
        "team_name",
        "position_id",
        "now_cost",
        "selected_by_percent",
        "total_points",
        "form",
        "status",
        "chance_of_playing_next_round",
        "chance_of_playing_this_round",
        "news",
        "news_added",
        "raw_player_json",
        "created_at",
    ],
    "production_fpl_fixture_snapshots": [
        "snapshot_id",
        "run_id",
        "snapshot_time",
        "event_id",
        "fixture_id",
        "kickoff_time",
        "team_h_id",
        "team_a_id",
        "team_h_name",
        "team_a_name",
        "finished",
        "started",
        "finished_provisional",
        "team_h_score",
        "team_a_score",
        "raw_fixture_json",
        "created_at",
    ],
    "production_football_data_match_staging": [
        "staging_id",
        "run_id",
        "source_name",
        "target_season",
        "source_url",
        "source_file_name",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "raw_row_json",
        "validation_status",
        "validation_error",
        "created_at",
    ],
    "production_understat_xg_staging": [
        "staging_id",
        "run_id",
        "source_name",
        "target_season",
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "raw_row_json",
        "validation_status",
        "validation_error",
        "created_at",
    ],
    "production_data_freshness": [
        "source_name",
        "last_successful_run_id",
        "last_successful_update_at",
        "latest_event_id",
        "latest_deadline_time",
        "latest_completed_match_date",
        "latest_error_message",
        "updated_at",
    ],
}


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def create_production_ingestion_tables(conn) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS production_ingestion_runs (
            run_id SERIAL PRIMARY KEY,
            run_started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            run_finished_at TIMESTAMP NULL,
            run_status TEXT NOT NULL,
            run_type TEXT NOT NULL,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NULL,
            deadline_time TIMESTAMP NULL,
            fpl_bootstrap_status TEXT NULL,
            fpl_fixtures_status TEXT NULL,
            football_data_status TEXT NULL,
            understat_status TEXT NULL,
            rows_fpl_players INTEGER NOT NULL DEFAULT 0,
            rows_fpl_fixtures INTEGER NOT NULL DEFAULT 0,
            rows_new_results INTEGER NOT NULL DEFAULT 0,
            rows_new_xg INTEGER NOT NULL DEFAULT 0,
            error_message TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (run_status IN ('started', 'success', 'partial', 'failed')),
            CHECK (target_gameweek IS NULL OR target_gameweek > 0),
            CHECK (rows_fpl_players >= 0),
            CHECK (rows_fpl_fixtures >= 0),
            CHECK (rows_new_results >= 0),
            CHECK (rows_new_xg >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_fpl_bootstrap_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES production_ingestion_runs(run_id),
            snapshot_time TIMESTAMP NOT NULL,
            event_id INTEGER NULL,
            deadline_time TIMESTAMP NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            web_name TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NULL,
            position_id INTEGER NOT NULL,
            now_cost INTEGER NOT NULL,
            selected_by_percent FLOAT NULL,
            total_points INTEGER NOT NULL,
            form FLOAT NULL,
            status TEXT NOT NULL,
            chance_of_playing_next_round INTEGER NULL,
            chance_of_playing_this_round INTEGER NULL,
            news TEXT NULL,
            news_added TIMESTAMP NULL,
            raw_player_json JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_fpl_fixture_snapshots (
            snapshot_id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES production_ingestion_runs(run_id),
            snapshot_time TIMESTAMP NOT NULL,
            event_id INTEGER NULL,
            fixture_id INTEGER NOT NULL,
            kickoff_time TIMESTAMP NULL,
            team_h_id INTEGER NOT NULL,
            team_a_id INTEGER NOT NULL,
            team_h_name TEXT NULL,
            team_a_name TEXT NULL,
            finished BOOLEAN NOT NULL,
            started BOOLEAN NOT NULL,
            finished_provisional BOOLEAN NULL,
            team_h_score INTEGER NULL,
            team_a_score INTEGER NULL,
            raw_fixture_json JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_football_data_match_staging (
            staging_id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES production_ingestion_runs(run_id),
            source_name TEXT NOT NULL,
            target_season TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_file_name TEXT NOT NULL,
            match_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            result TEXT NOT NULL,
            raw_row_json JSONB NOT NULL,
            validation_status TEXT NOT NULL,
            validation_error TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (home_team <> away_team),
            CHECK (home_goals >= 0),
            CHECK (away_goals >= 0),
            CHECK (result IN ('H', 'D', 'A'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_understat_xg_staging (
            staging_id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES production_ingestion_runs(run_id),
            source_name TEXT NOT NULL,
            target_season TEXT NOT NULL,
            match_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_xg FLOAT NOT NULL,
            away_xg FLOAT NOT NULL,
            raw_row_json JSONB NOT NULL,
            validation_status TEXT NOT NULL,
            validation_error TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (home_team <> away_team),
            CHECK (home_xg >= 0),
            CHECK (away_xg >= 0)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS production_data_freshness (
            source_name TEXT PRIMARY KEY,
            last_successful_run_id INTEGER NULL REFERENCES production_ingestion_runs(run_id),
            last_successful_update_at TIMESTAMP NULL,
            latest_event_id INTEGER NULL,
            latest_deadline_time TIMESTAMP NULL,
            latest_completed_match_date DATE NULL,
            latest_error_message TEXT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    with conn.begin() as db_conn:
        for statement in statements:
            db_conn.execute(text(statement))
    verify_production_ingestion_tables(conn)
    print("PASS: production ingestion tables exist and required columns are present.")


def start_ingestion_run(conn, target_season, target_gameweek=None) -> int:
    query = text(
        """
        INSERT INTO production_ingestion_runs (
            run_status,
            run_type,
            target_season,
            target_gameweek,
            fpl_bootstrap_status,
            fpl_fixtures_status,
            football_data_status,
            understat_status
        )
        VALUES (
            'started',
            :run_type,
            :target_season,
            :target_gameweek,
            'pending',
            'pending',
            'not_run',
            'not_run'
        )
        RETURNING run_id
        """
    )
    with conn.begin() as db_conn:
        run_id = int(
            db_conn.execute(
                query,
                {
                    "run_type": RUN_TYPE,
                    "target_season": target_season,
                    "target_gameweek": target_gameweek,
                },
            ).scalar_one()
        )
    print(f"Started production ingestion run_id={run_id}")
    return run_id


def finish_ingestion_run(conn, run_id, status, summary, error_message=None) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Invalid run status: {status}")
    query = text(
        """
        UPDATE production_ingestion_runs
        SET
            run_finished_at = CURRENT_TIMESTAMP,
            run_status = :run_status,
            deadline_time = :deadline_time,
            fpl_bootstrap_status = :fpl_bootstrap_status,
            fpl_fixtures_status = :fpl_fixtures_status,
            football_data_status = :football_data_status,
            understat_status = :understat_status,
            rows_fpl_players = :rows_fpl_players,
            rows_fpl_fixtures = :rows_fpl_fixtures,
            rows_new_results = :rows_new_results,
            rows_new_xg = :rows_new_xg,
            error_message = :error_message
        WHERE run_id = :run_id
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(
            query,
            {
                "run_id": run_id,
                "run_status": status,
                "deadline_time": summary.get("deadline_time"),
                "fpl_bootstrap_status": summary.get("fpl_bootstrap_status", "not_run"),
                "fpl_fixtures_status": summary.get("fpl_fixtures_status", "not_run"),
                "football_data_status": summary.get("football_data_status", "not_run"),
                "understat_status": summary.get("understat_status", "not_run"),
                "rows_fpl_players": int(summary.get("rows_fpl_players", 0)),
                "rows_fpl_fixtures": int(summary.get("rows_fpl_fixtures", 0)),
                "rows_new_results": int(summary.get("rows_new_results", 0)),
                "rows_new_xg": int(summary.get("rows_new_xg", 0)),
                "error_message": error_message,
            },
        )
    print(f"Finished production ingestion run_id={run_id} status={status}")


def fetch_json(url, source_name) -> dict | list:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as error:
        raise RuntimeError(f"{source_name} request timed out after {HTTP_TIMEOUT_SECONDS}s") from error
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", "unknown")
        raise RuntimeError(f"{source_name} HTTP error status={status_code}") from error
    except requests.RequestException as error:
        raise RuntimeError(f"{source_name} request failed: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{source_name} returned invalid JSON") from error


def fetch_football_data_csv(url, source_name) -> pandas.DataFrame:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as error:
        raise RuntimeError(
            f"{source_name} request timed out after {HTTP_TIMEOUT_SECONDS}s"
        ) from error
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", "unknown")
        raise RuntimeError(f"{source_name} HTTP error status={status_code}") from error
    except requests.RequestException as error:
        raise RuntimeError(f"{source_name} request failed: {error}") from error

    try:
        csv_df = pandas.read_csv(StringIO(response.text))
    except Exception as error:
        raise RuntimeError(f"{source_name} CSV parse failed: {error}") from error
    if csv_df.empty:
        raise ValueError(f"{source_name} CSV returned zero rows")
    return csv_df


def parse_football_data_results(
    csv_df,
    target_season,
    source_url,
    source_file_name,
) -> pandas.DataFrame:
    missing_columns = [
        column for column in REQUIRED_FOOTBALL_DATA_COLUMNS if column not in csv_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "football-data CSV missing required column(s): "
            + ", ".join(missing_columns)
        )

    raw_df = csv_df.copy()
    raw_df["raw_row_json"] = [
        _json_ready_record(row) for row in raw_df.to_dict(orient="records")
    ]
    completed_df = filter_completed_matches_only(raw_df)
    if completed_df.empty:
        return pandas.DataFrame(
            columns=[
                "source_name",
                "target_season",
                "source_url",
                "source_file_name",
                "match_date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
                "raw_row_json",
                "validation_status",
                "validation_error",
            ]
        )

    completed_df["match_date"] = pandas.to_datetime(
        completed_df["Date"],
        dayfirst=True,
        errors="coerce",
    ).dt.date
    bad_dates = completed_df.loc[completed_df["match_date"].isna(), "Date"].unique()
    if len(bad_dates) > 0:
        raise ValueError(f"football-data CSV has unparseable Date value(s): {list(bad_dates)}")

    home_goals = pandas.to_numeric(completed_df["FTHG"], errors="coerce")
    away_goals = pandas.to_numeric(completed_df["FTAG"], errors="coerce")
    if home_goals.isna().any() or away_goals.isna().any():
        raise ValueError("football-data CSV has non-numeric completed match goals")

    completed_df["home_goals"] = home_goals.astype(int)
    completed_df["away_goals"] = away_goals.astype(int)
    completed_df["home_team"] = completed_df["HomeTeam"].apply(normalize_team_name)
    completed_df["away_team"] = completed_df["AwayTeam"].apply(normalize_team_name)
    completed_df["result"] = [
        _result_from_goals(home_goal, away_goal)
        for home_goal, away_goal in zip(
            completed_df["home_goals"],
            completed_df["away_goals"],
        )
    ]
    completed_df["source_result"] = completed_df["FTR"].astype(str).str.strip().str.upper()

    results_df = pandas.DataFrame(
        {
            "source_name": "football_data",
            "target_season": target_season,
            "source_url": source_url,
            "source_file_name": source_file_name,
            "match_date": completed_df["match_date"],
            "home_team": completed_df["home_team"],
            "away_team": completed_df["away_team"],
            "home_goals": completed_df["home_goals"],
            "away_goals": completed_df["away_goals"],
            "result": completed_df["result"],
            "source_result": completed_df["source_result"],
            "raw_row_json": completed_df["raw_row_json"],
            "validation_status": "valid",
            "validation_error": None,
        }
    )
    return results_df.reset_index(drop=True)


def filter_completed_matches_only(results_df) -> pandas.DataFrame:
    required_non_null = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    existing_required = [column for column in required_non_null if column in results_df.columns]
    completed = results_df.dropna(subset=existing_required).copy()
    for column in existing_required:
        completed = completed.loc[completed[column].astype(str).str.strip() != ""]
    return completed.reset_index(drop=True)


def validate_football_data_results(results_df) -> None:
    if results_df.empty:
        print("football-data completed results: 0 rows")
        return
    errors: list[str] = []
    required_columns = {
        "target_season",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "source_result",
        "raw_row_json",
    }
    missing_columns = sorted(required_columns - set(results_df.columns))
    if missing_columns:
        errors.append(f"missing parsed column(s): {missing_columns}")
    invalid_results = sorted(set(results_df["source_result"]) - {"H", "D", "A"})
    if invalid_results:
        errors.append(f"invalid FTR value(s): {invalid_results}")
    mismatches = results_df.loc[results_df["source_result"] != results_df["result"]]
    if not mismatches.empty:
        examples = mismatches[
            ["match_date", "home_team", "away_team", "home_goals", "away_goals", "source_result"]
        ].head(5)
        errors.append(
            "FTR values do not match goals. Examples: "
            + str(examples.to_dict(orient="records"))
        )
    same_team = results_df.loc[results_df["home_team"] == results_df["away_team"]]
    if not same_team.empty:
        errors.append(f"home_team equals away_team for {len(same_team)} row(s)")
    duplicate_count = int(
        results_df.duplicated(
            subset=["target_season", "match_date", "home_team", "away_team"]
        ).sum()
    )
    if duplicate_count:
        errors.append(f"duplicate completed match row(s): {duplicate_count}")
    if errors:
        raise ValueError("football-data result validation failed: " + "; ".join(errors))
    print(f"PASS: football-data completed results validated ({len(results_df)} rows).")


def write_football_data_staging(conn, run_id, results_df) -> int:
    if results_df.empty:
        print(f"Wrote 0 football-data staging rows for run_id={run_id}")
        return 0
    query = text(
        """
        INSERT INTO production_football_data_match_staging (
            run_id,
            source_name,
            target_season,
            source_url,
            source_file_name,
            match_date,
            home_team,
            away_team,
            home_goals,
            away_goals,
            result,
            raw_row_json,
            validation_status,
            validation_error
        )
        VALUES (
            :run_id,
            :source_name,
            :target_season,
            :source_url,
            :source_file_name,
            :match_date,
            :home_team,
            :away_team,
            :home_goals,
            :away_goals,
            :result,
            CAST(:raw_row_json AS JSONB),
            :validation_status,
            :validation_error
        )
        """
    )
    records = []
    for row in results_df.to_dict(orient="records"):
        records.append(
            {
                "run_id": run_id,
                "source_name": row["source_name"],
                "target_season": row["target_season"],
                "source_url": row["source_url"],
                "source_file_name": row["source_file_name"],
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "result": row["result"],
                "raw_row_json": json.dumps(row["raw_row_json"], sort_keys=True),
                "validation_status": row["validation_status"],
                "validation_error": row["validation_error"],
            }
        )
    with conn.begin() as db_conn:
        db_conn.execute(query, records)
    print(f"Wrote {len(records)} football-data staging rows for run_id={run_id}")
    return len(records)


def upsert_new_historical_matches(conn, results_df) -> int:
    if results_df.empty:
        print("Inserted 0 new historical_matches rows from football-data")
        return 0
    for season_id in sorted(set(results_df["target_season"])):
        _ensure_season_row(conn, season_id)
    query = text(
        """
        INSERT INTO historical_matches (
            season_id,
            match_date,
            kickoff_time,
            home_team,
            away_team,
            home_goals,
            away_goals,
            result,
            source,
            source_file
        )
        VALUES (
            :season_id,
            :match_date,
            NULL,
            :home_team,
            :away_team,
            :home_goals,
            :away_goals,
            :result,
            'football_data',
            :source_file
        )
        ON CONFLICT (season_id, home_team, away_team) DO NOTHING
        """
    )
    records = [
        {
            "season_id": row["target_season"],
            "match_date": row["match_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "result": row["result"],
            "source_file": row["source_file_name"],
        }
        for row in results_df.to_dict(orient="records")
    ]
    with conn.begin() as db_conn:
        before_count = _table_count(db_conn, "historical_matches")
        db_conn.execute(query, records)
        after_count = _table_count(db_conn, "historical_matches")
    inserted = int(after_count - before_count)
    print(f"Inserted {inserted} new historical_matches rows from football-data")
    return inserted


def fetch_understat_xg_for_season(target_season) -> pandas.DataFrame:
    understat_season = target_season[:4]
    return fetch_understat_league_data(understat_season)


def parse_understat_xg_for_ingestion(understat_df, target_season) -> pandas.DataFrame:
    if understat_df.empty:
        return pandas.DataFrame(
            columns=[
                "understat_match_id",
                "target_season",
                "match_date",
                "home_team",
                "away_team",
                "home_xg",
                "away_xg",
                "home_goals",
                "away_goals",
                "source_name",
                "raw_row_json",
                "validation_status",
                "validation_error",
            ]
        )
    raw_by_id = {
        str(row.get("id")).strip(): _json_ready_record(row)
        for row in understat_df.to_dict(orient="records")
        if row.get("id") is not None
    }
    parsed_df = parse_understat_matches(understat_df, target_season)
    if parsed_df.empty:
        return pandas.DataFrame(
            columns=[
                "understat_match_id",
                "target_season",
                "match_date",
                "home_team",
                "away_team",
                "home_xg",
                "away_xg",
                "home_goals",
                "away_goals",
                "source_name",
                "raw_row_json",
                "validation_status",
                "validation_error",
            ]
        )
    parsed_df = parsed_df.rename(columns={"season_id": "target_season", "source": "source_name"})
    parsed_df["raw_row_json"] = parsed_df["understat_match_id"].map(raw_by_id)
    parsed_df["validation_status"] = "valid"
    parsed_df["validation_error"] = None
    return parsed_df.reset_index(drop=True)


def validate_understat_xg(xg_df, historical_results_df) -> None:
    if xg_df.empty:
        print("Understat completed xG rows: 0")
        return
    errors: list[str] = []
    required_columns = {
        "understat_match_id",
        "target_season",
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "home_goals",
        "away_goals",
        "raw_row_json",
    }
    missing_columns = sorted(required_columns - set(xg_df.columns))
    if missing_columns:
        errors.append(f"missing parsed xG column(s): {missing_columns}")
    if (xg_df["home_xg"] < 0).any() or (xg_df["away_xg"] < 0).any():
        errors.append("negative xG value found")
    duplicate_count = int(
        xg_df.duplicated(
            subset=["target_season", "match_date", "home_team", "away_team"]
        ).sum()
    )
    if duplicate_count:
        errors.append(f"duplicate Understat xG row(s): {duplicate_count}")

    if historical_results_df.empty:
        errors.append("Understat xG rows exist but no completed historical results exist")
    else:
        join_columns_xg = ["target_season", "match_date", "home_team", "away_team"]
        join_columns_hist = ["season_id", "match_date", "home_team", "away_team"]
        merged = xg_df.merge(
            historical_results_df[join_columns_hist + ["home_goals", "away_goals"]],
            left_on=join_columns_xg,
            right_on=join_columns_hist,
            how="left",
            suffixes=("_xg", "_historical"),
        )
        unmatched = merged.loc[merged["season_id"].isna()]
        if not unmatched.empty:
            examples = unmatched[
                ["target_season", "match_date", "home_team", "away_team"]
            ].head(5)
            errors.append(
                "Understat xG rows did not match completed historical results. Examples: "
                + str(examples.to_dict(orient="records"))
            )
        goal_mismatch = merged.loc[
            merged["season_id"].notna()
            & (
                (merged["home_goals_xg"] != merged["home_goals_historical"])
                | (merged["away_goals_xg"] != merged["away_goals_historical"])
            )
        ]
        if not goal_mismatch.empty:
            errors.append(f"Understat goal mismatch row(s): {len(goal_mismatch)}")

    if errors:
        raise ValueError("Understat xG validation failed: " + "; ".join(errors))
    print(f"PASS: Understat xG rows validated ({len(xg_df)} rows).")


def write_understat_xg_staging(conn, run_id, xg_df) -> int:
    if xg_df.empty:
        print(f"Wrote 0 Understat xG staging rows for run_id={run_id}")
        return 0
    query = text(
        """
        INSERT INTO production_understat_xg_staging (
            run_id,
            source_name,
            target_season,
            match_date,
            home_team,
            away_team,
            home_xg,
            away_xg,
            raw_row_json,
            validation_status,
            validation_error
        )
        VALUES (
            :run_id,
            :source_name,
            :target_season,
            :match_date,
            :home_team,
            :away_team,
            :home_xg,
            :away_xg,
            CAST(:raw_row_json AS JSONB),
            :validation_status,
            :validation_error
        )
        """
    )
    records = []
    for row in xg_df.to_dict(orient="records"):
        records.append(
            {
                "run_id": run_id,
                "source_name": row.get("source_name", "understat"),
                "target_season": row["target_season"],
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_xg": float(row["home_xg"]),
                "away_xg": float(row["away_xg"]),
                "raw_row_json": json.dumps(row["raw_row_json"], sort_keys=True),
                "validation_status": row["validation_status"],
                "validation_error": row["validation_error"],
            }
        )
    with conn.begin() as db_conn:
        db_conn.execute(query, records)
    print(f"Wrote {len(records)} Understat xG staging rows for run_id={run_id}")
    return len(records)


def upsert_new_historical_understat_xg(conn, xg_df) -> int:
    if xg_df.empty:
        print("Inserted 0 new historical_understat_xg rows")
        return 0
    for season_id in sorted(set(xg_df["target_season"])):
        _ensure_season_row(conn, season_id)
    query = text(
        """
        INSERT INTO historical_understat_xg (
            understat_match_id,
            season_id,
            match_date,
            home_team,
            away_team,
            home_xg,
            away_xg,
            home_goals,
            away_goals,
            source
        )
        VALUES (
            :understat_match_id,
            :season_id,
            :match_date,
            :home_team,
            :away_team,
            :home_xg,
            :away_xg,
            :home_goals,
            :away_goals,
            'understat'
        )
        ON CONFLICT (season_id, match_date, home_team, away_team) DO NOTHING
        """
    )
    records = [
        {
            "understat_match_id": str(row["understat_match_id"]),
            "season_id": row["target_season"],
            "match_date": row["match_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_xg": float(row["home_xg"]),
            "away_xg": float(row["away_xg"]),
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
        }
        for row in xg_df.to_dict(orient="records")
    ]
    with conn.begin() as db_conn:
        before_count = _table_count(db_conn, "historical_understat_xg")
        db_conn.execute(query, records)
        after_count = _table_count(db_conn, "historical_understat_xg")
    inserted = int(after_count - before_count)
    print(f"Inserted {inserted} new historical_understat_xg rows")
    return inserted


def update_match_xg_freshness(conn, source_name, run_id, metadata, error_message=None) -> None:
    update_data_freshness(conn, source_name, run_id, metadata, error_message)


def parse_fpl_bootstrap(bootstrap_json) -> tuple[pandas.DataFrame, dict]:
    if not isinstance(bootstrap_json, dict):
        raise ValueError("FPL bootstrap payload was not a JSON object")
    elements = bootstrap_json.get("elements")
    teams = bootstrap_json.get("teams")
    events = bootstrap_json.get("events")
    if not isinstance(elements, list) or not elements:
        raise ValueError("FPL bootstrap payload has no player elements")
    if not isinstance(teams, list) or not teams:
        raise ValueError("FPL bootstrap payload has no teams metadata")
    if not isinstance(events, list):
        raise ValueError("FPL bootstrap payload has no events metadata")

    missing_by_field = {
        field: sum(field not in player for player in elements)
        for field in BOOTSTRAP_REQUIRED_FIELDS
    }
    missing = {field: count for field, count in missing_by_field.items() if count}
    if missing:
        raise ValueError(f"FPL bootstrap player field(s) missing: {missing}")

    team_name_by_id = {
        int(team["id"]): str(team.get("name") or team.get("short_name") or team["id"])
        for team in teams
        if "id" in team
    }
    if not team_name_by_id:
        raise ValueError("FPL bootstrap teams metadata did not map team IDs to names")

    current_event = _select_relevant_event(events)
    event_id = _nullable_int(current_event.get("id")) if current_event else None
    deadline_time = _parse_timestamp(current_event.get("deadline_time")) if current_event else None
    snapshot_time = _utcnow_naive()

    rows = []
    for player in elements:
        first_name = str(player.get("first_name") or "").strip()
        second_name = str(player.get("second_name") or "").strip()
        player_name = " ".join(value for value in [first_name, second_name] if value).strip()
        team_id = int(player["team"])
        rows.append(
            {
                "snapshot_time": snapshot_time,
                "event_id": event_id,
                "deadline_time": deadline_time,
                "player_id": int(player["id"]),
                "player_name": player_name or str(player.get("web_name") or player["id"]),
                "web_name": str(player["web_name"]),
                "team_id": team_id,
                "team_name": team_name_by_id.get(team_id),
                "position_id": int(player["element_type"]),
                "now_cost": int(player["now_cost"]),
                "selected_by_percent": _nullable_float(player.get("selected_by_percent")),
                "total_points": int(player["total_points"]),
                "form": _nullable_float(player.get("form")),
                "status": str(player["status"]),
                "chance_of_playing_next_round": _nullable_int(
                    player.get("chance_of_playing_next_round")
                ),
                "chance_of_playing_this_round": _nullable_int(
                    player.get("chance_of_playing_this_round")
                ),
                "news": _nullable_text(player.get("news")),
                "news_added": _parse_timestamp(player.get("news_added")),
                "raw_player_json": player,
            }
        )
    players_df = pandas.DataFrame(rows)
    metadata = {
        "snapshot_time": snapshot_time,
        "event_id": event_id,
        "deadline_time": deadline_time,
        "team_name_by_id": team_name_by_id,
        "source_required_fields": BOOTSTRAP_REQUIRED_FIELDS.copy(),
    }
    players_df.attrs["bootstrap_metadata"] = metadata
    return players_df, metadata


def parse_fpl_fixtures(fixtures_json, bootstrap_metadata) -> pandas.DataFrame:
    if not isinstance(fixtures_json, list):
        raise ValueError("FPL fixtures payload was not a JSON list")
    if not fixtures_json:
        raise ValueError(
            "FPL fixtures API returned zero rows; offseason or unavailable fixture list"
        )

    missing_by_field = {
        field: sum(field not in fixture for fixture in fixtures_json)
        for field in FIXTURE_REQUIRED_FIELDS
    }
    missing = {field: count for field, count in missing_by_field.items() if count}
    if missing:
        raise ValueError(f"FPL fixture field(s) missing: {missing}")

    team_name_by_id = bootstrap_metadata["team_name_by_id"]
    snapshot_time = bootstrap_metadata["snapshot_time"]
    rows = []
    for fixture in fixtures_json:
        team_h_id = int(fixture["team_h"])
        team_a_id = int(fixture["team_a"])
        rows.append(
            {
                "snapshot_time": snapshot_time,
                "event_id": _nullable_int(fixture.get("event")),
                "fixture_id": int(fixture["id"]),
                "kickoff_time": _parse_timestamp(fixture.get("kickoff_time")),
                "team_h_id": team_h_id,
                "team_a_id": team_a_id,
                "team_h_name": team_name_by_id.get(team_h_id),
                "team_a_name": team_name_by_id.get(team_a_id),
                "finished": bool(fixture["finished"]),
                "started": bool(fixture["started"]),
                "finished_provisional": _nullable_bool(fixture.get("finished_provisional")),
                "team_h_score": _nullable_int(fixture.get("team_h_score")),
                "team_a_score": _nullable_int(fixture.get("team_a_score")),
                "raw_fixture_json": fixture,
            }
        )
    fixtures_df = pandas.DataFrame(rows)
    fixtures_df.attrs["source_required_fields"] = FIXTURE_REQUIRED_FIELDS.copy()
    return fixtures_df


def write_fpl_bootstrap_snapshot(conn, run_id, players_df) -> int:
    if players_df.empty:
        return 0
    query = text(
        """
        INSERT INTO production_fpl_bootstrap_snapshots (
            run_id,
            snapshot_time,
            event_id,
            deadline_time,
            player_id,
            player_name,
            web_name,
            team_id,
            team_name,
            position_id,
            now_cost,
            selected_by_percent,
            total_points,
            form,
            status,
            chance_of_playing_next_round,
            chance_of_playing_this_round,
            news,
            news_added,
            raw_player_json
        )
        VALUES (
            :run_id,
            :snapshot_time,
            :event_id,
            :deadline_time,
            :player_id,
            :player_name,
            :web_name,
            :team_id,
            :team_name,
            :position_id,
            :now_cost,
            :selected_by_percent,
            :total_points,
            :form,
            :status,
            :chance_of_playing_next_round,
            :chance_of_playing_this_round,
            :news,
            :news_added,
            CAST(:raw_player_json AS JSONB)
        )
        """
    )
    records = [
        {
            **_record_without_raw(row),
            "run_id": run_id,
            "raw_player_json": json.dumps(row["raw_player_json"], sort_keys=True),
        }
        for row in players_df.to_dict(orient="records")
    ]
    with conn.begin() as db_conn:
        db_conn.execute(query, records)
    print(f"Wrote {len(records)} FPL bootstrap snapshot rows for run_id={run_id}")
    return len(records)


def write_fpl_fixture_snapshot(conn, run_id, fixtures_df) -> int:
    if fixtures_df.empty:
        return 0
    query = text(
        """
        INSERT INTO production_fpl_fixture_snapshots (
            run_id,
            snapshot_time,
            event_id,
            fixture_id,
            kickoff_time,
            team_h_id,
            team_a_id,
            team_h_name,
            team_a_name,
            finished,
            started,
            finished_provisional,
            team_h_score,
            team_a_score,
            raw_fixture_json
        )
        VALUES (
            :run_id,
            :snapshot_time,
            :event_id,
            :fixture_id,
            :kickoff_time,
            :team_h_id,
            :team_a_id,
            :team_h_name,
            :team_a_name,
            :finished,
            :started,
            :finished_provisional,
            :team_h_score,
            :team_a_score,
            CAST(:raw_fixture_json AS JSONB)
        )
        """
    )
    records = [
        {
            **_record_without_raw(row),
            "run_id": run_id,
            "raw_fixture_json": json.dumps(row["raw_fixture_json"], sort_keys=True),
        }
        for row in fixtures_df.to_dict(orient="records")
    ]
    with conn.begin() as db_conn:
        db_conn.execute(query, records)
    print(f"Wrote {len(records)} FPL fixture snapshot rows for run_id={run_id}")
    return len(records)


def update_data_freshness(conn, source_name, run_id, metadata, error_message=None) -> None:
    if error_message:
        query = text(
            """
            INSERT INTO production_data_freshness (
                source_name,
                latest_error_message,
                updated_at
            )
            VALUES (:source_name, :latest_error_message, CURRENT_TIMESTAMP)
            ON CONFLICT (source_name) DO UPDATE SET
                latest_error_message = EXCLUDED.latest_error_message,
                updated_at = CURRENT_TIMESTAMP
            """
        )
        params = {
            "source_name": source_name,
            "latest_error_message": error_message,
        }
    else:
        query = text(
            """
            INSERT INTO production_data_freshness (
                source_name,
                last_successful_run_id,
                last_successful_update_at,
                latest_event_id,
                latest_deadline_time,
                latest_completed_match_date,
                latest_error_message,
                updated_at
            )
            VALUES (
                :source_name,
                :run_id,
                CURRENT_TIMESTAMP,
                :latest_event_id,
                :latest_deadline_time,
                :latest_completed_match_date,
                NULL,
                CURRENT_TIMESTAMP
            )
            ON CONFLICT (source_name) DO UPDATE SET
                last_successful_run_id = EXCLUDED.last_successful_run_id,
                last_successful_update_at = EXCLUDED.last_successful_update_at,
                latest_event_id = EXCLUDED.latest_event_id,
                latest_deadline_time = EXCLUDED.latest_deadline_time,
                latest_completed_match_date = EXCLUDED.latest_completed_match_date,
                latest_error_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            """
        )
        params = {
            "source_name": source_name,
            "run_id": run_id,
            "latest_event_id": metadata.get("event_id"),
            "latest_deadline_time": metadata.get("deadline_time"),
            "latest_completed_match_date": metadata.get("latest_completed_match_date"),
        }
    with conn.begin() as db_conn:
        db_conn.execute(query, params)


def validate_fpl_snapshot(players_df, fixtures_df) -> None:
    errors: list[str] = []
    if players_df.empty:
        errors.append("FPL bootstrap players dataframe has zero rows")
    if fixtures_df.empty:
        errors.append("FPL fixtures dataframe has zero rows")

    player_required = {
        "player_id",
        "player_name",
        "web_name",
        "team_id",
        "team_name",
        "position_id",
        "now_cost",
        "total_points",
        "status",
    }
    fixture_required = {
        "fixture_id",
        "event_id",
        "kickoff_time",
        "team_h_id",
        "team_a_id",
        "finished",
        "started",
    }
    missing_player_columns = sorted(player_required - set(players_df.columns))
    missing_fixture_columns = sorted(fixture_required - set(fixtures_df.columns))
    if missing_player_columns:
        errors.append(f"missing parsed player column(s): {missing_player_columns}")
    if missing_fixture_columns:
        errors.append(f"missing parsed fixture column(s): {missing_fixture_columns}")
    if players_df["team_name"].isna().any():
        errors.append(
            f"players with unmapped team_id: {int(players_df['team_name'].isna().sum())}"
        )
    fixture_unmapped = int(
        fixtures_df["team_h_name"].isna().sum() + fixtures_df["team_a_name"].isna().sum()
    )
    if fixture_unmapped:
        errors.append(f"fixture team IDs not mapped to names: {fixture_unmapped}")
    if errors:
        raise ValueError("FPL snapshot validation failed: " + "; ".join(errors))

    print("PASS: FPL bootstrap and fixture snapshots validated.")
    print(f"FPL player snapshot rows: {len(players_df)}")
    print(f"FPL fixture snapshot rows: {len(fixtures_df)}")


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            exists = _table_exists(db_conn, table_name)
            if exists:
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_counts_unchanged_except_allowed(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    unexpected = {
        table_name: counts
        for table_name, counts in changed.items()
        if table_name not in ALLOWED_CHANGED_TABLES
    }
    if unexpected:
        raise RuntimeError(f"Unexpected watched table count changes: {unexpected}")

    print("PASS: watched Tier 2/Tier 3 source table counts unchanged.")
    if changed:
        print("Allowed production ingestion table count changes:")
        for table_name in sorted(changed):
            print(f"- {table_name}: {changed[table_name][0]} -> {changed[table_name][1]}")
    else:
        print("No watched table counts changed.")


def assert_historical_deltas_match_summary(before, after, summary) -> None:
    historical_match_delta = _count_delta(before, after, "historical_matches")
    historical_xg_delta = _count_delta(before, after, "historical_understat_xg")
    expected_match_delta = int(summary.get("rows_new_results", 0))
    expected_xg_delta = int(summary.get("rows_new_xg", 0))

    errors: list[str] = []
    if historical_match_delta != expected_match_delta:
        errors.append(
            "historical_matches delta "
            f"{historical_match_delta} != rows_new_results {expected_match_delta}"
        )
    if historical_xg_delta != expected_xg_delta:
        errors.append(
            "historical_understat_xg delta "
            f"{historical_xg_delta} != rows_new_xg {expected_xg_delta}"
        )
    if errors:
        raise RuntimeError("Historical table delta assertion failed: " + "; ".join(errors))

    print("PASS: historical table deltas match inserted row summary.")


def run_fpl_ingestion(conn, run_id, summary, source_errors) -> dict[str, Any]:
    if summary.get("skip_fpl"):
        summary["fpl_bootstrap_status"] = "skipped"
        summary["fpl_fixtures_status"] = "skipped"
        print("SKIP: FPL bootstrap and fixture ingestion skipped by CLI flag.")
        return {}

    bootstrap_metadata: dict[str, Any] = {}
    try:
        bootstrap_json = fetch_json(FPL_BOOTSTRAP_URL, "FPL bootstrap")
        players_df, bootstrap_metadata = parse_fpl_bootstrap(bootstrap_json)
        rows_players = write_fpl_bootstrap_snapshot(conn, run_id, players_df)
        summary.update(
            {
                "fpl_bootstrap_status": "success",
                "rows_fpl_players": rows_players,
                "deadline_time": bootstrap_metadata.get("deadline_time"),
            }
        )
        update_data_freshness(
            conn,
            "fpl_bootstrap",
            run_id,
            {
                "event_id": bootstrap_metadata.get("event_id"),
                "deadline_time": bootstrap_metadata.get("deadline_time"),
            },
        )
    except Exception as error:
        error_message = f"FPL bootstrap failed: {type(error).__name__}: {error}"
        summary["fpl_bootstrap_status"] = _source_failure_status(error_message)
        summary["fpl_fixtures_status"] = "not_run"
        source_errors.append(error_message)
        update_data_freshness(conn, "fpl_bootstrap", run_id, {}, error_message)
        print(error_message)
        return {}

    try:
        fixtures_json = fetch_json(FPL_FIXTURES_URL, "FPL fixtures")
        fixtures_df = parse_fpl_fixtures(fixtures_json, bootstrap_metadata)
        validate_fpl_snapshot(players_df, fixtures_df)
        rows_fixtures = write_fpl_fixture_snapshot(conn, run_id, fixtures_df)
        summary.update(
            {
                "fpl_fixtures_status": "success",
                "rows_fpl_fixtures": rows_fixtures,
            }
        )
        update_data_freshness(
            conn,
            "fpl_fixtures",
            run_id,
            {
                "event_id": _latest_event_id(fixtures_df),
                "deadline_time": bootstrap_metadata.get("deadline_time"),
                "latest_completed_match_date": _latest_completed_match_date(fixtures_df),
            },
        )
        return bootstrap_metadata
    except Exception as error:
        error_message = f"FPL fixtures failed: {type(error).__name__}: {error}"
        summary["fpl_fixtures_status"] = _source_failure_status(error_message)
        source_errors.append(error_message)
        update_data_freshness(conn, "fpl_fixtures", run_id, {}, error_message)
        print(error_message)
        return bootstrap_metadata


def run_football_data_ingestion(conn, run_id, target_season, summary, source_errors) -> None:
    if summary.get("skip_football_data"):
        summary["football_data_status"] = "skipped"
        print("SKIP: football-data result ingestion skipped by CLI flag.")
        return

    try:
        csv_df = fetch_football_data_csv(
            FOOTBALL_DATA_2026_27_URL,
            "football-data 2026-27 Premier League CSV",
        )
        results_df = parse_football_data_results(
            csv_df,
            target_season,
            FOOTBALL_DATA_2026_27_URL,
            FOOTBALL_DATA_2026_27_FILENAME,
        )
        validate_football_data_results(results_df)
        write_football_data_staging(conn, run_id, results_df)
        rows_new_results = upsert_new_historical_matches(conn, results_df)
        summary["football_data_status"] = "success"
        summary["rows_new_results"] = rows_new_results
        update_match_xg_freshness(
            conn,
            "football_data_results",
            run_id,
            {
                "latest_completed_match_date": _latest_match_date(results_df),
            },
        )
    except Exception as error:
        error_message = f"football-data results failed: {type(error).__name__}: {error}"
        summary["football_data_status"] = _source_failure_status(error_message)
        source_errors.append(error_message)
        update_match_xg_freshness(conn, "football_data_results", run_id, {}, error_message)
        print(error_message)


def run_understat_xg_ingestion(conn, run_id, target_season, summary, source_errors) -> None:
    if summary.get("skip_understat"):
        summary["understat_status"] = "skipped"
        print("SKIP: Understat xG ingestion skipped by CLI flag.")
        return

    try:
        understat_df = fetch_understat_xg_for_season(target_season)
        xg_df = parse_understat_xg_for_ingestion(understat_df, target_season)
        historical_results_df = _load_completed_historical_results(conn, target_season)
        validate_understat_xg(xg_df, historical_results_df)
        write_understat_xg_staging(conn, run_id, xg_df)
        rows_new_xg = upsert_new_historical_understat_xg(conn, xg_df)
        summary["understat_status"] = "success"
        summary["rows_new_xg"] = rows_new_xg
        update_match_xg_freshness(
            conn,
            "understat_xg",
            run_id,
            {
                "latest_completed_match_date": _latest_match_date(xg_df),
            },
        )
    except Exception as error:
        error_message = f"Understat xG failed: {type(error).__name__}: {error}"
        summary["understat_status"] = _source_failure_status(error_message)
        source_errors.append(error_message)
        update_match_xg_freshness(conn, "understat_xg", run_id, {}, error_message)
        print(error_message)


def main() -> None:
    args = parse_args()
    conn = get_db_connection()
    create_production_ingestion_tables(conn)

    if args.init_schema_only:
        print("=== Production P2B schema initialization only ===")
        print_production_table_counts(conn)
        return

    before_counts = capture_watched_table_counts(conn)
    run_id = start_ingestion_run(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
    )
    summary = _empty_summary()
    summary["skip_fpl"] = args.skip_fpl
    summary["skip_football_data"] = args.skip_football_data
    summary["skip_understat"] = args.skip_understat
    source_errors: list[str] = []

    run_fpl_ingestion(conn, run_id, summary, source_errors)
    run_football_data_ingestion(
        conn,
        run_id,
        target_season=args.target_season,
        summary=summary,
        source_errors=source_errors,
    )
    run_understat_xg_ingestion(
        conn,
        run_id,
        target_season=args.target_season,
        summary=summary,
        source_errors=source_errors,
    )

    status = _derive_run_status(summary)
    error_message = "; ".join(source_errors) if source_errors else None
    finish_ingestion_run(conn, run_id, status, summary, error_message)
    after_counts = capture_watched_table_counts(conn)
    assert_counts_unchanged_except_allowed(before_counts, after_counts)
    assert_historical_deltas_match_summary(before_counts, after_counts, summary)
    print_latest_ingestion_state(conn)
    print_watched_count_comparison(before_counts, after_counts)

    if status == "success":
        print("PASS: Production P2B weekly ingestion completed successfully.")
    elif status == "partial":
        print("PARTIAL: Production P2B ingestion completed with source failure(s).")
    else:
        print("FAILED: Production P2B ingestion failed for all attempted sources.")
        raise SystemExit(1)


def verify_production_ingestion_tables(conn) -> None:
    with conn.connect() as db_conn:
        for table_name, required_columns in SCHEMA_REQUIRED_COLUMNS.items():
            if not _table_exists(db_conn, table_name):
                raise RuntimeError(f"{table_name} table does not exist")
            columns = [
                row["column_name"]
                for row in db_conn.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = CURRENT_SCHEMA()
                            AND table_name = :table_name
                        """
                    ),
                    {"table_name": table_name},
                ).mappings().all()
            ]
            missing = sorted(set(required_columns) - set(columns))
            if missing:
                raise RuntimeError(f"{table_name} missing required column(s): {missing}")


def print_production_table_counts(conn) -> None:
    with conn.connect() as db_conn:
        for table_name in PRODUCTION_TABLES:
            count = int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
            print(f"{table_name}: {count}")


def print_latest_ingestion_state(conn) -> None:
    with conn.connect() as db_conn:
        latest_run = db_conn.execute(
            text(
                """
                SELECT *
                FROM production_ingestion_runs
                ORDER BY run_id DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        bootstrap_counts = db_conn.execute(
            text(
                """
                SELECT run_id, COUNT(*) AS row_count
                FROM production_fpl_bootstrap_snapshots
                GROUP BY run_id
                ORDER BY run_id
                """
            )
        ).mappings().all()
        fixture_counts = db_conn.execute(
            text(
                """
                SELECT run_id, COUNT(*) AS row_count
                FROM production_fpl_fixture_snapshots
                GROUP BY run_id
                ORDER BY run_id
                """
            )
        ).mappings().all()
        football_data_staging_counts = db_conn.execute(
            text(
                """
                SELECT run_id, COUNT(*) AS row_count
                FROM production_football_data_match_staging
                GROUP BY run_id
                ORDER BY run_id
                """
            )
        ).mappings().all()
        understat_staging_counts = db_conn.execute(
            text(
                """
                SELECT run_id, COUNT(*) AS row_count
                FROM production_understat_xg_staging
                GROUP BY run_id
                ORDER BY run_id
                """
            )
        ).mappings().all()
        freshness_rows = db_conn.execute(
            text(
                """
                SELECT *
                FROM production_data_freshness
                ORDER BY source_name
                """
            )
        ).mappings().all()

    print("Latest production_ingestion_runs row:")
    print(_json_dumps_mapping(latest_run))
    print("FPL bootstrap snapshot row count by run_id:")
    for row in bootstrap_counts:
        print(f"- run_id={row['run_id']}: {row['row_count']}")
    print("FPL fixture snapshot row count by run_id:")
    for row in fixture_counts:
        print(f"- run_id={row['run_id']}: {row['row_count']}")
    print("football-data staging row count by run_id:")
    for row in football_data_staging_counts:
        print(f"- run_id={row['run_id']}: {row['row_count']}")
    print("Understat xG staging row count by run_id:")
    for row in understat_staging_counts:
        print(f"- run_id={row['run_id']}: {row['row_count']}")
    print("production_data_freshness rows:")
    for row in freshness_rows:
        print(_json_dumps_mapping(row))


def print_watched_count_comparison(before_counts, after_counts) -> None:
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        print(f"- {table_name}: {before_counts.get(table_name)} -> {after_counts.get(table_name)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production P2B weekly ingestion foundation")
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    parser.add_argument("--skip-fpl", action="store_true")
    parser.add_argument("--skip-football-data", action="store_true")
    parser.add_argument("--skip-understat", action="store_true")
    return parser.parse_args()


def _empty_summary() -> dict[str, Any]:
    return {
        "fpl_bootstrap_status": "not_run",
        "fpl_fixtures_status": "not_run",
        "football_data_status": "not_run",
        "understat_status": "not_run",
        "rows_fpl_players": 0,
        "rows_fpl_fixtures": 0,
        "rows_new_results": 0,
        "rows_new_xg": 0,
        "deadline_time": None,
    }


def _derive_run_status(summary) -> str:
    statuses = [
        summary.get("fpl_bootstrap_status", "not_run"),
        summary.get("fpl_fixtures_status", "not_run"),
        summary.get("football_data_status", "not_run"),
        summary.get("understat_status", "not_run"),
    ]
    success_count = sum(status == "success" for status in statuses)
    skipped_count = sum(status == "skipped" for status in statuses)
    failed_count = sum(status in {"failed", "unavailable"} for status in statuses)

    if failed_count and not success_count and not skipped_count:
        return "failed"
    if failed_count or skipped_count:
        return "partial"
    if success_count:
        return "success"
    return "partial"


def _source_failure_status(error_message: str) -> str:
    unavailable_fragments = [
        "status=404",
        "zero rows",
        "no dates data",
        "offseason",
        "unavailable",
    ]
    error_text = error_message.lower()
    if any(fragment in error_text for fragment in unavailable_fragments):
        return "unavailable"
    return "failed"


def _count_delta(before, after, table_name: str) -> int:
    before_count = before.get(table_name)
    after_count = after.get(table_name)
    if isinstance(before_count, int) and isinstance(after_count, int):
        return after_count - before_count
    if before_count == "MISSING" and isinstance(after_count, int):
        return after_count
    return 0


def _load_completed_historical_results(conn, target_season: str) -> pandas.DataFrame:
    query = text(
        """
        SELECT
            season_id,
            match_date,
            home_team,
            away_team,
            home_goals,
            away_goals
        FROM historical_matches
        WHERE season_id = :target_season
        """
    )
    with conn.connect() as db_conn:
        rows = db_conn.execute(query, {"target_season": target_season}).mappings().all()
    results_df = pandas.DataFrame(rows)
    if results_df.empty:
        return pandas.DataFrame(
            columns=[
                "season_id",
                "match_date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ]
        )
    results_df["match_date"] = pandas.to_datetime(results_df["match_date"]).dt.date
    return results_df


def _latest_match_date(df: pandas.DataFrame):
    if df.empty or "match_date" not in df:
        return None
    dates = pandas.to_datetime(df["match_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date()


def _table_count(db_conn, table_name: str) -> int:
    return int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _ensure_season_row(conn, season_id: str) -> None:
    start_date, end_date = _season_bounds(season_id)
    query = text(
        """
        INSERT INTO seasons (season_id, season_name, start_date, end_date, is_current)
        VALUES (:season_id, :season_name, :start_date, :end_date, FALSE)
        ON CONFLICT (season_id) DO NOTHING
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(
            query,
            {
                "season_id": season_id,
                "season_name": f"{season_id} Premier League",
                "start_date": start_date,
                "end_date": end_date,
            },
        )


def _season_bounds(season_id: str) -> tuple[str, str]:
    try:
        start_year = int(season_id[:4])
        end_year = int(f"20{season_id[-2:]}")
    except ValueError as error:
        raise ValueError(f"Invalid season_id: {season_id}") from error
    return f"{start_year}-08-01", f"{end_year}-05-31"


def _result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _json_ready_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready_value(value) for key, value in row.items()}


def _json_ready_value(value):
    if isinstance(value, dict):
        return {str(key): _json_ready_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_value(item) for item in value]
    if isinstance(value, pandas.Timestamp):
        return value.isoformat() if not pandas.isna(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    try:
        if pandas.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


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


def _select_relevant_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("is_next"):
            return event
    for event in events:
        if event.get("is_current"):
            return event
    for event in events:
        if event.get("is_previous"):
            return event
    return events[0] if events else None


def _latest_event_id(fixtures_df: pandas.DataFrame) -> int | None:
    event_values = pandas.to_numeric(fixtures_df["event_id"], errors="coerce").dropna()
    if event_values.empty:
        return None
    return int(event_values.max())


def _latest_completed_match_date(fixtures_df: pandas.DataFrame):
    if fixtures_df.empty or "kickoff_time" not in fixtures_df:
        return None
    completed = fixtures_df.loc[
        fixtures_df["finished"].fillna(False) & fixtures_df["kickoff_time"].notna()
    ]
    if completed.empty:
        return None
    return pandas.to_datetime(completed["kickoff_time"]).max().date()


def _parse_timestamp(value):
    if value in (None, "") or pandas.isna(value):
        return None
    parsed = pandas.to_datetime(value, utc=True, errors="coerce")
    if pandas.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(tzinfo=None)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _nullable_int(value):
    if value in (None, "") or pandas.isna(value):
        return None
    return int(value)


def _nullable_float(value):
    if value in (None, "") or pandas.isna(value):
        return None
    return float(value)


def _nullable_bool(value):
    if value in (None, "") or pandas.isna(value):
        return None
    return bool(value)


def _nullable_text(value):
    if value is None or pandas.isna(value):
        return None
    text_value = str(value)
    return text_value if text_value else None


def _record_without_raw(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _db_safe_value(value)
        for key, value in row.items()
        if not key.startswith("raw_")
    }


def _db_safe_value(value):
    if isinstance(value, pandas.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.to_pydatetime()
    if isinstance(value, float) and pandas.isna(value):
        return None
    if pandas.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _json_dumps_mapping(row) -> str:
    if row is None:
        return "{}"
    return json.dumps(
        {key: _json_safe(value) for key, value in dict(row).items()},
        sort_keys=True,
        default=str,
    )


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
