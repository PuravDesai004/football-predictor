import json
import re
import time

import pandas as pd
import requests
from data_pipeline import get_engine  # reuse existing engine
from sqlalchemy import text


UNDERSTAT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://understat.com/league/EPL/2025",
    "X-Requested-With": "XMLHttpRequest",
}

TEAM_MAP = {
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham": "West Ham",
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "Spurs": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton": "Brighton",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Crystal Palace": "Crystal Palace",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
    "Southampton": "Southampton",
    "Bournemouth": "Bournemouth",
    "Burnley": "Burnley",
    "Leeds": "Leeds",
    "Sunderland": "Sunderland",
}


# Fetches the full Understat league payload once so all extraction functions reuse it.
def fetch_understat_payload(season=2025):
    url = f"https://understat.com/getLeagueData/EPL/{season}/"
    headers = {
        **UNDERSTAT_HEADERS,
        "Referer": f"https://understat.com/league/EPL/{season}",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        time.sleep(1)
        return response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Failed to fetch Understat payload: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Failed to parse Understat JSON payload: {error}") from error


# Applies the Understat-to-project team name mapping and warns about unknown teams.
def normalize_team_names(df):
    teams = set(df["home_team"].dropna()).union(set(df["away_team"].dropna()))
    unknown_teams = sorted(team for team in teams if team not in TEAM_MAP)

    for team in unknown_teams:
        print(f"Warning: Team name not found in TEAM_MAP: {team}")

    df["home_team"] = df["home_team"].apply(lambda team: TEAM_MAP.get(team, team))
    df["away_team"] = df["away_team"].apply(lambda team: TEAM_MAP.get(team, team))

    return df


# Normalizes a single Understat team name for the team-history table.
def normalize_team_title(team_title):
    if team_title not in TEAM_MAP:
        print(f"Warning: Team name not found in TEAM_MAP: {team_title}")

    return TEAM_MAP.get(team_title, team_title)


# Checks normalized Understat match team names against the PostgreSQL teams table.
def warn_unmapped_against_teams(df, engine):
    teams_df = pd.read_sql(text("SELECT name FROM teams"), engine)
    valid_team_names = set(teams_df["name"].dropna())
    scraped_team_names = set(df["home_team"].dropna()).union(set(df["away_team"].dropna()))

    for name in sorted(scraped_team_names):
        if name not in valid_team_names:
            print(f"UNMAPPED: {name}")


# Checks normalized team-history names against the PostgreSQL teams table.
def warn_unmapped_team_history_against_teams(df, engine):
    teams_df = pd.read_sql(text("SELECT name FROM teams"), engine)
    valid_team_names = set(teams_df["name"].dropna())

    for name in sorted(set(df["team_name"].dropna())):
        if name not in valid_team_names:
            print(f"UNMAPPED: {name}")


# Converts missing or malformed numeric values to NULL-friendly None.
def to_float(value):
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Converts missing or malformed integer values to NULL-friendly None.
def to_int(value):
    parsed_value = to_float(value)

    if parsed_value is None:
        return None

    return int(parsed_value)


# Converts Understat datetime strings to YYYY-MM-DD date strings.
def parse_match_date(value):
    if value is None or value == "":
        return None

    return str(value).split(" ")[0]


# Converts Understat PPDA dictionaries into att / def, with safe NULL handling.
def parse_ppda(value, team_name, match_date, field_name):
    if value is None or value == "":
        return None

    if isinstance(value, dict):
        att = to_float(value.get("att"))
        defensive_actions = to_float(value.get("def"))

        if defensive_actions is None or defensive_actions == 0:
            print(
                f"WARNING: Null {field_name} for {team_name} on {match_date} "
                "because def is missing or zero"
            )
            return None

        if att is None:
            print(
                f"WARNING: Null {field_name} for {team_name} on {match_date} "
                "because att is missing"
            )
            return None

        return att / defensive_actions

    return to_float(value)


# Reads xG values from either the current Understat format or the fallback format.
def get_xg_value(match_data, side):
    team_key = "h" if side == "home" else "a"

    value = (
        match_data.get(team_key, {}).get("xg")
        or match_data.get(team_key, {}).get("xG")
        or match_data.get("xG", {}).get(team_key)
    )

    if value is None:
        raise ValueError(f"Missing xG value for match: {match_data}")

    return float(value)


# Builds the understat_xg DataFrame from the payload's completed fixture rows.
def scrape_understat_epl(season=2025, payload=None):
    if payload is None:
        payload = fetch_understat_payload(season=season)

    data = payload.get("dates", [])

    if not data:
        pattern = r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, json.dumps(payload))

        if match is None:
            raise ValueError("Could not find Understat dates data.")

        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        data = json.loads(raw)

    rows = []
    for match_data in data:
        if not match_data.get("isResult"):
            continue

        rows.append(
            {
                "match_date": parse_match_date(match_data.get("datetime")),
                "home_team": match_data["h"]["title"],
                "away_team": match_data["a"]["title"],
                "home_xg": get_xg_value(match_data, "home"),
                "away_xg": get_xg_value(match_data, "away"),
                "home_goals": to_int(match_data["goals"]["h"]),
                "away_goals": to_int(match_data["goals"]["a"]),
                "season": int(season),
            }
        )

    df = pd.DataFrame(rows)
    print(f"Scraped {len(rows)} completed matches from Understat")

    if not df.empty:
        df = normalize_team_names(df)

    return df


# Extracts one tactical history row per team per match from payload["teams"].
def extract_team_history(payload, season=2025):
    teams_data = payload.get("teams", {})

    if isinstance(teams_data, dict):
        team_items = teams_data.items()
    elif isinstance(teams_data, list):
        team_items = [(team.get("id"), team) for team in teams_data if isinstance(team, dict)]
    else:
        raise ValueError("Understat payload['teams'] must be a dict or list.")

    rows = []
    for team_key, team in team_items:
        if not isinstance(team, dict):
            continue

        understat_team_id = team.get("id") or team_key
        raw_team_title = team.get("title")
        team_name = normalize_team_title(raw_team_title)

        for history_row in team.get("history", []):
            match_date = parse_match_date(history_row.get("date"))

            rows.append(
                {
                    "season": int(season),
                    "understat_team_id": to_int(understat_team_id),
                    "team_name": team_name,
                    "match_date": match_date,
                    "venue": history_row.get("h_a"),
                    "result": history_row.get("result"),
                    "xg": to_float(history_row.get("xG")),
                    "xga": to_float(history_row.get("xGA")),
                    "npxg": to_float(history_row.get("npxG")),
                    "npxga": to_float(history_row.get("npxGA")),
                    "npxgd": to_float(history_row.get("npxGD")),
                    "ppda": parse_ppda(
                        history_row.get("ppda"),
                        team_name,
                        match_date,
                        "ppda",
                    ),
                    "ppda_allowed": parse_ppda(
                        history_row.get("ppda_allowed"),
                        team_name,
                        match_date,
                        "ppda_allowed",
                    ),
                    "deep": to_float(history_row.get("deep")),
                    "deep_allowed": to_float(history_row.get("deep_allowed")),
                    "scored": to_int(history_row.get("scored")),
                    "missed": to_int(history_row.get("missed")),
                    "xpts": to_float(history_row.get("xpts")),
                    "pts": to_int(history_row.get("pts")),
                    "wins": to_int(history_row.get("wins")),
                    "draws": to_int(history_row.get("draws")),
                    "loses": to_int(history_row.get("loses")),
                }
            )

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} team-history rows from Understat")
    return df


# Creates the PostgreSQL table and loads Understat xG rows.
def load_to_postgres(df, engine):
    warn_unmapped_against_teams(df, engine)

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS understat_xg (
            id SERIAL PRIMARY KEY,
            match_date DATE,
            home_team VARCHAR(100),
            away_team VARCHAR(100),
            home_xg FLOAT,
            away_xg FLOAT,
            home_goals INT,
            away_goals INT,
            season INT
        );
    """

    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        conn.execute(text("TRUNCATE TABLE understat_xg RESTART IDENTITY"))
        df.to_sql("understat_xg", conn, if_exists="append", index=False)

    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM understat_xg")).scalar()

    print(f"Loaded {n} rows into understat_xg")


# Creates the PostgreSQL table and loads tactical team-history rows.
def load_team_history_to_postgres(df, engine):
    warn_unmapped_team_history_against_teams(df, engine)

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS understat_team_history (
            id SERIAL PRIMARY KEY,
            season INT,
            understat_team_id INT,
            team_name VARCHAR(100),
            match_date DATE,
            venue VARCHAR(5),
            result VARCHAR(5),
            xg FLOAT,
            xga FLOAT,
            npxg FLOAT,
            npxga FLOAT,
            npxgd FLOAT,
            ppda FLOAT,
            ppda_allowed FLOAT,
            deep FLOAT,
            deep_allowed FLOAT,
            scored INT,
            missed INT,
            xpts FLOAT,
            pts INT,
            wins INT,
            draws INT,
            loses INT
        );
    """

    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        conn.execute(text("TRUNCATE TABLE understat_team_history RESTART IDENTITY"))
        df.to_sql("understat_team_history", conn, if_exists="append", index=False)

    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM understat_team_history")).scalar()

    print(f"Loaded {n} rows into understat_team_history")


# Prints a small sample and total row count from the loaded Understat xG table.
def verify(engine):
    sample_query = """
        SELECT home_team, away_team, home_xg, away_xg
        FROM understat_xg
        LIMIT 5
    """
    count_query = "SELECT COUNT(*) FROM understat_xg"

    sample = pd.read_sql(text(sample_query), engine)
    print(sample)

    with engine.connect() as conn:
        count = conn.execute(text(count_query)).scalar()

    print(f"understat_xg row count: {count}")


# Prints verification checks and sample rows for the tactical team-history table.
def verify_team_history(engine):
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM understat_team_history")).scalar()
        team_count = conn.execute(
            text("SELECT COUNT(DISTINCT team_name) FROM understat_team_history")
        ).scalar()
        date_range = conn.execute(
            text("SELECT MIN(match_date), MAX(match_date) FROM understat_team_history")
        ).fetchone()
        null_ppda = conn.execute(
            text("SELECT COUNT(*) FROM understat_team_history WHERE ppda IS NULL")
        ).scalar()
        null_ppda_allowed = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM understat_team_history
                WHERE ppda_allowed IS NULL
                """
            )
        ).scalar()

    print(f"understat_team_history row count: {count}")
    print(f"understat_team_history teams: {team_count}")
    print(
        "understat_team_history date range: "
        f"{date_range[0]} to {date_range[1]}"
    )
    print(f"understat_team_history null ppda rows: {null_ppda}")
    print(f"understat_team_history null ppda_allowed rows: {null_ppda_allowed}")

    if null_ppda:
        affected_ppda = pd.read_sql(
            text(
                """
                SELECT team_name, match_date
                FROM understat_team_history
                WHERE ppda IS NULL
                ORDER BY team_name, match_date
                """
            ),
            engine,
        )
        print("Rows with null ppda:")
        print(affected_ppda)

    if null_ppda_allowed:
        affected_ppda_allowed = pd.read_sql(
            text(
                """
                SELECT team_name, match_date
                FROM understat_team_history
                WHERE ppda_allowed IS NULL
                ORDER BY team_name, match_date
                """
            ),
            engine,
        )
        print("Rows with null ppda_allowed:")
        print(affected_ppda_allowed)

    sample = pd.read_sql(
        text(
            """
            SELECT team_name, match_date, venue, xg, xga, ppda,
                   ppda_allowed, deep, deep_allowed
            FROM understat_team_history
            ORDER BY match_date, team_name
            LIMIT 5
            """
        ),
        engine,
    )
    print(sample)


if __name__ == "__main__":
    engine = get_engine()
    payload = fetch_understat_payload(season=2025)

    xg_df = scrape_understat_epl(season=2025, payload=payload)
    print(xg_df.head())
    load_to_postgres(xg_df, engine)
    verify(engine)

    team_history_df = extract_team_history(payload, season=2025)
    load_team_history_to_postgres(team_history_df, engine)
    verify_team_history(engine)
