import json
import os
import pathlib

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


# Creates and returns a SQLAlchemy engine for the PostgreSQL database.
def get_engine():
    try:
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASS")

        connection_string = (
            f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        )
        engine = create_engine(connection_string)

        with engine.connect():
            pass

        print("Connected to PostgreSQL: football_db")
        return engine
    except Exception as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None


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

    df = df[columns_to_keep].copy()
    df = df.rename(columns={"element_type": "position"})
    df["now_cost"] = df["now_cost"] / 10
    df["form"] = df["form"].astype(float)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float)
    df["is_available"] = df["status"].apply(lambda status: 1 if status == "a" else 0)

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
        "team_h_score",
        "team_a_score",
    ]

    fixtures = fixtures[columns_to_keep].copy()

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

    players_to_store = players_df.rename(
        columns={"id": "player_id", "now_cost": "price"}
    )
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
    fixtures_to_store.to_sql("fixtures", engine, if_exists="replace", index=False)
    print(f"Stored {len(fixtures_to_store)} fixtures")

    gameweeks_to_store = gameweeks_df.rename(columns={"id": "gw_id"})
    gameweeks_to_store = gameweeks_to_store[gameweeks_columns].copy()
    gameweeks_to_store.to_sql("gameweeks", engine, if_exists="replace", index=False)
    print(f"Stored {len(gameweeks_to_store)} gameweeks")

    print("All tables stored in PostgreSQL successfully.")


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
