import time
from pathlib import Path

import pandas as pd
import requests
from data_pipeline import get_engine
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPL_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

HISTORY_COLUMNS = [
    "player_id",
    "fixture",
    "opponent_team",
    "total_points",
    "was_home",
    "kickoff_time",
    "team_h_score",
    "team_a_score",
    "gameweek",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value",
    "selected",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
]


# Converts empty strings and missing optional values to NULL-friendly values.
def nullable_value(row, key):
    value = row.get(key)
    if value == "":
        return None
    return value


# Fetches one player's FPL element-summary payload with retries.
def fetch_player_summary(player_id, session=None, max_retries=3):
    client = session or requests.Session()
    url = FPL_SUMMARY_URL.format(player_id=player_id)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            print(
                f"Warning: player {player_id} request failed "
                f"(attempt {attempt}/{max_retries}): {error}"
            )
        except ValueError as error:
            print(
                f"Warning: player {player_id} JSON parse failed "
                f"(attempt {attempt}/{max_retries}): {error}"
            )

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    print(f"Warning: giving up on player {player_id}")
    return None


# Extracts raw per-gameweek history rows from a player summary payload.
def extract_player_history(player_id, payload):
    if not payload:
        return []

    history = payload.get("history", [])
    rows = []

    for item in history:
        rows.append(
            {
                "player_id": player_id,
                "fixture": nullable_value(item, "fixture"),
                "opponent_team": nullable_value(item, "opponent_team"),
                "total_points": nullable_value(item, "total_points"),
                "was_home": nullable_value(item, "was_home"),
                "kickoff_time": nullable_value(item, "kickoff_time"),
                "team_h_score": nullable_value(item, "team_h_score"),
                "team_a_score": nullable_value(item, "team_a_score"),
                "gameweek": nullable_value(item, "round"),
                "minutes": nullable_value(item, "minutes"),
                "goals_scored": nullable_value(item, "goals_scored"),
                "assists": nullable_value(item, "assists"),
                "clean_sheets": nullable_value(item, "clean_sheets"),
                "goals_conceded": nullable_value(item, "goals_conceded"),
                "own_goals": nullable_value(item, "own_goals"),
                "penalties_saved": nullable_value(item, "penalties_saved"),
                "penalties_missed": nullable_value(item, "penalties_missed"),
                "yellow_cards": nullable_value(item, "yellow_cards"),
                "red_cards": nullable_value(item, "red_cards"),
                "saves": nullable_value(item, "saves"),
                "bonus": nullable_value(item, "bonus"),
                "bps": nullable_value(item, "bps"),
                "influence": nullable_value(item, "influence"),
                "creativity": nullable_value(item, "creativity"),
                "threat": nullable_value(item, "threat"),
                "ict_index": nullable_value(item, "ict_index"),
                "starts": nullable_value(item, "starts"),
                "expected_goals": nullable_value(item, "expected_goals"),
                "expected_assists": nullable_value(item, "expected_assists"),
                "expected_goal_involvements": nullable_value(
                    item, "expected_goal_involvements"
                ),
                "expected_goals_conceded": nullable_value(
                    item, "expected_goals_conceded"
                ),
                "value": nullable_value(item, "value"),
                "selected": nullable_value(item, "selected"),
                "transfers_balance": nullable_value(item, "transfers_balance"),
                "transfers_in": nullable_value(item, "transfers_in"),
                "transfers_out": nullable_value(item, "transfers_out"),
            }
        )

    return rows


# Fetches and combines gameweek history for every player in the players table.
def fetch_all_player_histories(engine, delay_seconds=0.08):
    player_ids = pd.read_sql(
        "SELECT player_id FROM players ORDER BY player_id",
        engine,
    )["player_id"].tolist()

    total = len(player_ids)
    all_rows = []
    failed_player_ids = []

    with requests.Session() as session:
        for index, player_id in enumerate(player_ids, start=1):
            payload = fetch_player_summary(player_id, session=session)

            if payload is None:
                failed_player_ids.append(player_id)
            else:
                rows = extract_player_history(player_id, payload)
                all_rows.extend(rows)

            if index % 50 == 0 or index == total:
                print(
                    f"Processed {index}/{total} players, "
                    f"rows collected: {len(all_rows)}"
                )

            time.sleep(delay_seconds)

    if failed_player_ids:
        print(f"Failed player IDs: {failed_player_ids}")
    else:
        print("Failed player IDs: []")

    return pd.DataFrame(all_rows, columns=HISTORY_COLUMNS)


# Creates or truncates the player gameweek history table, loads rows, and indexes it.
def load_player_gameweek_history(df, engine):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS player_gameweek_history (
            id SERIAL PRIMARY KEY,
            player_id INTEGER,
            fixture INTEGER,
            opponent_team INTEGER,
            total_points INTEGER,
            was_home BOOLEAN,
            kickoff_time TIMESTAMP,
            team_h_score INTEGER,
            team_a_score INTEGER,
            gameweek INTEGER,
            minutes INTEGER,
            goals_scored INTEGER,
            assists INTEGER,
            clean_sheets INTEGER,
            goals_conceded INTEGER,
            own_goals INTEGER,
            penalties_saved INTEGER,
            penalties_missed INTEGER,
            yellow_cards INTEGER,
            red_cards INTEGER,
            saves INTEGER,
            bonus INTEGER,
            bps INTEGER,
            influence FLOAT,
            creativity FLOAT,
            threat FLOAT,
            ict_index FLOAT,
            starts INTEGER,
            expected_goals FLOAT,
            expected_assists FLOAT,
            expected_goal_involvements FLOAT,
            expected_goals_conceded FLOAT,
            value INTEGER,
            selected INTEGER,
            transfers_balance INTEGER,
            transfers_in INTEGER,
            transfers_out INTEGER
        );
    """

    index_sql = [
        """
        CREATE INDEX IF NOT EXISTS idx_player_gameweek_history_player_id
        ON player_gameweek_history (player_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_player_gameweek_history_gameweek
        ON player_gameweek_history (gameweek);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_player_gameweek_history_fixture
        ON player_gameweek_history (fixture);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_player_gameweek_history_kickoff_time
        ON player_gameweek_history (kickoff_time);
        """,
    ]

    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        conn.execute(text("TRUNCATE TABLE player_gameweek_history RESTART IDENTITY;"))

    df.to_sql("player_gameweek_history", engine, if_exists="append", index=False)

    with engine.begin() as conn:
        for statement in index_sql:
            conn.execute(text(statement))

    print(f"Loaded {len(df)} rows into player_gameweek_history")


# Prints row counts, coverage, null checks, and sample rows for the loaded table.
def verify_player_gameweek_history(engine):
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM player_gameweek_history")
        ).scalar()
        player_count = conn.execute(
            text("SELECT COUNT(DISTINCT player_id) FROM player_gameweek_history")
        ).scalar()
        min_gw, max_gw = conn.execute(
            text("SELECT MIN(gameweek), MAX(gameweek) FROM player_gameweek_history")
        ).one()
        min_date, max_date = conn.execute(
            text("SELECT MIN(kickoff_time), MAX(kickoff_time) FROM player_gameweek_history")
        ).one()
        null_total_points = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM player_gameweek_history
                WHERE total_points IS NULL
                """
            )
        ).scalar()

    print(f"player_gameweek_history row count: {count}")
    print(f"player_gameweek_history players: {player_count}")
    print(f"player_gameweek_history gameweeks: GW{min_gw} to GW{max_gw}")
    print(f"player_gameweek_history date range: {min_date} to {max_date}")
    print(f"player_gameweek_history null total_points rows: {null_total_points}")

    sample = pd.read_sql(
        """
        SELECT player_id, gameweek, minutes, total_points, expected_goals,
               expected_assists, ict_index, value, selected
        FROM player_gameweek_history
        ORDER BY player_id, gameweek
        LIMIT 5
        """,
        engine,
    )
    print(sample)


if __name__ == "__main__":
    engine = get_engine()

    if engine is None:
        print("Failed to connect to PostgreSQL.")
        raise SystemExit(1)

    history_df = fetch_all_player_histories(engine)

    if history_df.empty:
        print("No player gameweek history rows fetched.")
        raise SystemExit(1)

    load_player_gameweek_history(history_df, engine)
    verify_player_gameweek_history(engine)
