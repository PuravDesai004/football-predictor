import json
import os
import pathlib

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


# Creates and returns a SQLAlchemy engine for the PostgreSQL database.
def get_engine():
    mode = "local_env"
    try:
        database_url = None

        try:
            import streamlit as st

            if "DATABASE_URL" in st.secrets:
                database_url = str(st.secrets["DATABASE_URL"])
                mode = "streamlit_secrets"
        except Exception:
            database_url = None

        if database_url is None:
            env_database_url = os.getenv("DATABASE_URL")
            if env_database_url:
                database_url = env_database_url
                mode = "database_url"

        if database_url:
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)

            url = make_url(database_url)
            connect_args = {}
            if "sslmode" not in database_url.lower():
                connect_args["sslmode"] = "require"

            engine = create_engine(database_url, connect_args=connect_args)
            db_name = url.database or "unknown"
        else:
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_pass = os.getenv("DB_PASS")

            missing = [
                name
                for name, value in {
                    "DB_HOST": db_host,
                    "DB_PORT": db_port,
                    "DB_NAME": db_name,
                    "DB_USER": db_user,
                    "DB_PASS": db_pass,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"Missing local database settings: {missing}")

            connection_string = (
                f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            )
            engine = create_engine(connection_string)

        with engine.connect():
            pass

        print(f"Connected to PostgreSQL database: {db_name}")
        print(f"Connection mode: {mode}")
        return engine
    except Exception as error:
        print(f"Error connecting to PostgreSQL database using {mode}: {type(error).__name__}")
        return None


# Drops derived SQL views so base FPL tables can be replaced cleanly.
def drop_dependent_views(engine):
    views = [
        "player_fpl_features",
        "match_features",
        "team_style_form",
        "team_tactical_match_stats",
        "away_xg_form",
        "home_xg_form",
        "team_xg_stats",
        "h2h_stats",
        "away_form",
        "home_form",
        "team_season_stats",
        "match_results",
    ]

    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS player_gameweek_features"))
            print(
                "Dropped generated table player_gameweek_features. "
                "Re-run fpl_feature_engineering.py before FPL ML training/serving if needed."
            )
            for view in views:
                conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
    except Exception as error:
        print(f"Warning: Could not drop dependent views before reload: {error}")


# Converts FPL kickoff_time values to plain datetimes before database storage.
def convert_kickoff_time(df, context):
    if "kickoff_time" not in df.columns:
        print(f"Warning: kickoff_time missing from {context}")
        return df

    try:
        converted = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
        df["kickoff_time"] = converted.dt.tz_localize(None)
        null_count = df["kickoff_time"].isna().sum()
        if null_count > 0:
            print(f"Warning: {context} kickoff_time has {null_count} null rows")
    except Exception as error:
        print(f"Warning: Could not convert {context} kickoff_time to datetime: {error}")

    return df


# Fetches Fantasy Premier League bootstrap data and saves the raw API response.
def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        print(f"Error fetching FPL data: {error}")
        return None
    except json.JSONDecodeError as error:
        print(f"Error decoding FPL data: {error}")
        return None

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    gameweeks = pd.DataFrame(data["events"])

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_file_path = RAW_DATA_DIR / "fpl_bootstrap.json"

    with raw_file_path.open("w", encoding="utf-8") as raw_file:
        json.dump(data, raw_file, indent=2)

    print(f"Players shape: {players.shape}")
    print(f"Players columns: {list(players.columns)}")
    print(f"Teams shape: {teams.shape}")
    print(f"Teams columns: {list(teams.columns)}")
    print(f"Gameweeks shape: {gameweeks.shape}")
    print(f"Gameweeks columns: {list(gameweeks.columns)}")

    return players, teams, gameweeks


# Cleans the players DataFrame for modeling by selecting, renaming, and typing fields.
def clean_players(df):
    columns_to_keep = [
        "id",
        "first_name",
        "second_name",
        "team",
        "element_type",
        "now_cost",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "form",
        "selected_by_percent",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "status",
        "news",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "starts",
    ]

    for column in columns_to_keep:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[columns_to_keep].copy()
    df = df.rename(columns={"element_type": "position"})
    df["now_cost"] = df["now_cost"] / 10
    df["form"] = df["form"].astype(float)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float)
    df["chance_of_playing_this_round"] = pd.to_numeric(
        df["chance_of_playing_this_round"],
        errors="coerce",
    )
    df["chance_of_playing_next_round"] = pd.to_numeric(
        df["chance_of_playing_next_round"],
        errors="coerce",
    )
    df["is_available"] = df["status"].apply(lambda status: True if status == "a" else False)

    print(f"Players cleaned: {len(df)} rows, {len(df.columns)} columns")

    return df


# Fetches Fantasy Premier League fixture data and saves the raw API response.
def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        print(f"Error fetching fixtures: {error}")
        return None
    except json.JSONDecodeError as error:
        print(f"Error decoding fixtures: {error}")
        return None

    fixtures = pd.DataFrame(data)
    columns_to_keep = [
        "id",
        "event",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
        "kickoff_time",
        "team_h_score",
        "team_a_score",
    ]

    fixtures = fixtures[columns_to_keep].copy()
    fixtures = convert_kickoff_time(fixtures, "fixtures API")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_file_path = RAW_DATA_DIR / "fpl_fixtures.json"

    with raw_file_path.open("w", encoding="utf-8") as raw_file:
        json.dump(data, raw_file, indent=2)

    print(f"Fixtures shape: {fixtures.shape}")
    print(fixtures.head(3))

    return fixtures


# Stores cleaned FPL players, teams, fixtures, and gameweeks data in PostgreSQL.
def store_all_data(players_df, teams_df, fixtures_df, gameweeks_df, engine):
    players_columns = [
        "player_id",
        "first_name",
        "second_name",
        "team",
        "position",
        "price",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "form",
        "selected_by_percent",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "is_available",
        "status",
        "news",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "starts",
    ]
    teams_columns = [
        "team_id",
        "name",
        "short_name",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
        "played",
        "win",
        "draw",
        "loss",
        "points",
    ]
    fixtures_columns = [
        "fixture_id",
        "gameweek",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
        "kickoff_time",
        "team_h_score",
        "team_a_score",
    ]
    gameweeks_columns = [
        "gw_id",
        "name",
        "deadline_time",
        "finished",
        "is_current",
        "is_next",
        "is_previous",
        "average_entry_score",
        "highest_score",
    ]

    drop_dependent_views(engine)

    players_to_store = players_df.rename(columns={"id": "player_id", "now_cost": "price"})
    players_to_store = players_to_store[players_columns].copy()
    players_to_store.to_sql("players", engine, if_exists="replace", index=False)
    print(f"Stored {len(players_to_store)} players")

    teams_to_store = teams_df.rename(columns={"id": "team_id"})
    teams_to_store = teams_to_store[teams_columns].copy()
    teams_to_store.to_sql("teams", engine, if_exists="replace", index=False)
    print(f"Stored {len(teams_to_store)} teams")

    fixtures_to_store = fixtures_df.rename(
        columns={"id": "fixture_id", "event": "gameweek"}
    )
    fixtures_to_store = fixtures_to_store[fixtures_columns].copy()
    fixtures_to_store = convert_kickoff_time(fixtures_to_store, "fixtures store")

    fixtures_to_store.to_sql("fixtures", engine, if_exists="replace", index=False)
    print(f"Stored {len(fixtures_to_store)} fixtures")

    gameweeks_to_store = gameweeks_df.rename(columns={"id": "gw_id"})
    gameweeks_to_store = gameweeks_to_store[gameweeks_columns].copy()
    gameweeks_to_store.to_sql("gameweeks", engine, if_exists="replace", index=False)
    print(f"Stored {len(gameweeks_to_store)} gameweeks")

    print("All tables stored in PostgreSQL successfully.")


# Recreates only the player FPL feature view after a player metadata refresh.
def rebuild_player_fpl_features_view(engine):
    query = """
    CREATE OR REPLACE VIEW player_fpl_features AS
    SELECT
        p.player_id,
        p.first_name,
        p.second_name,
        p.team,
        t.name AS team_name,
        p.position,
        p.price,
        p.total_points,
        p.minutes,
        p.goals_scored,
        p.assists,
        p.clean_sheets,
        p.goals_conceded,
        p.bonus,
        p.form,
        p.selected_by_percent,
        p.chance_of_playing_this_round,
        p.chance_of_playing_next_round,
        p.is_available,
        p.status,
        p.news,
        p.expected_goals,
        p.expected_assists,
        p.expected_goal_involvements,
        p.expected_goals_conceded,
        p.starts,
        p.influence,
        p.creativity,
        p.threat,
        p.ict_index,
        ROUND((p.total_points::FLOAT / NULLIF(p.price, 0))::NUMERIC, 2)
            AS points_per_million,
        ROUND((p.minutes::FLOAT / NULLIF(38 * 90, 0))::NUMERIC, 2)
            AS minutes_ratio,
        t.strength_overall_home,
        t.strength_overall_away
    FROM players p
    LEFT JOIN teams t ON p.team = t.team_id;
    """

    with engine.begin() as conn:
        conn.execute(text(query))


# Refreshes only current FPL player metadata for pre-deadline optimizer serving.
def refresh_player_data_only():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")

    fpl_data = fetch_fpl_data()
    if fpl_data is None:
        raise RuntimeError("Could not fetch latest FPL bootstrap data.")

    players, _, _ = fpl_data
    cleaned_players = clean_players(players)

    players_columns = [
        "player_id",
        "first_name",
        "second_name",
        "team",
        "position",
        "price",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "form",
        "selected_by_percent",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "is_available",
        "status",
        "news",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "starts",
    ]

    players_to_store = cleaned_players.rename(
        columns={"id": "player_id", "now_cost": "price"}
    )
    players_to_store = players_to_store[players_columns].copy()

    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS player_fpl_features CASCADE"))

    players_to_store.to_sql("players", engine, if_exists="replace", index=False)
    rebuild_player_fpl_features_view(engine)

    print(f"Refreshed players table: {len(players_to_store)} rows")
    return len(players_to_store)


if __name__ == "__main__":
    # Step 1: Create the PostgreSQL connection engine from .env values.
    engine = get_engine()

    # Step 2: Fetch players, teams, and gameweeks from the FPL bootstrap API.
    fpl_data = fetch_fpl_data()

    if fpl_data is not None:
        players, teams, gameweeks = fpl_data

        # Step 3: Clean the players DataFrame before storing it.
        cleaned_players = clean_players(players)

        # Step 4: Fetch fixtures from the FPL fixtures API.
        fixtures = fetch_fixtures()

        # Step 5: Store all four DataFrames in PostgreSQL.
        if engine is not None and fixtures is not None:
            try:
                store_all_data(cleaned_players, teams, fixtures, gameweeks, engine)
                # Step 6: Confirm the Day 2 database pipeline is complete.
                print("Day 2 complete. All data in PostgreSQL.")
            except Exception as error:
                print(f"Error storing data in PostgreSQL: {error}")
