from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from sqlalchemy import text

from data_pipeline import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

TARGET_UNDERSTAT_SEASONS: dict[str, str] = {
    "2021-22": "2021",
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
    "2025-26": "2025",
}

UNDERSTAT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}

UNDERSTAT_TEAM_MAPPING: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}

OUTPUT_COLUMNS = [
    "understat_match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_goals",
    "away_goals",
    "source",
]

REQUIRED_OUTPUT_COLUMNS = [
    "understat_match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_goals",
    "away_goals",
    "source",
]


def normalize_understat_team_name(raw_name: str) -> str:
    source_name = str(raw_name).strip()
    try:
        return UNDERSTAT_TEAM_MAPPING[source_name]
    except KeyError as error:
        raise ValueError(f"Unknown Understat team name: {raw_name}") from error


def fetch_understat_league_data(understat_season: str) -> pd.DataFrame:
    url = f"https://understat.com/getLeagueData/EPL/{understat_season}/"
    headers = {
        **UNDERSTAT_HEADERS,
        "Referer": f"https://understat.com/league/EPL/{understat_season}",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Network error fetching Understat season {understat_season}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON from Understat season {understat_season}: {error}"
        ) from error

    data = payload.get("dates")
    if not isinstance(data, list) or not data:
        raise ValueError(
            f"Understat season {understat_season} response has no dates data"
        )

    print(f"Fetched Understat season {understat_season}: {len(data)} raw rows")
    return pd.DataFrame(data)


def _is_completed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _required_dict(value: Any, field_name: str, match_id: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Match {match_id} has invalid {field_name}: {value}")
    return value


def _to_float(value: Any, field_name: str, match_id: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Match {match_id} is missing {field_name}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Match {match_id} has non-numeric {field_name}: {value}"
        ) from error
    return parsed


def _to_int(value: Any, field_name: str, match_id: str) -> int:
    parsed = _to_float(value, field_name, match_id)
    if not parsed.is_integer():
        raise ValueError(f"Match {match_id} has non-integer {field_name}: {value}")
    return int(parsed)


def _extract_xg(match_data: dict, side: str, match_id: str) -> float:
    team_key = "h" if side == "home" else "a"
    value = None

    team_data = match_data.get(team_key)
    if isinstance(team_data, dict):
        value = team_data.get("xG")
        if value is None:
            value = team_data.get("xg")

    if value is None and isinstance(match_data.get("xG"), dict):
        value = match_data["xG"].get(team_key)

    return _to_float(value, f"{side}_xg", match_id)


def _extract_match_date(match_data: dict, match_id: str):
    raw_date = match_data.get("datetime") or match_data.get("date")
    parsed = pd.to_datetime(raw_date, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Match {match_id} has unparseable match date: {raw_date}")
    return parsed.date()


def parse_understat_matches(raw_df: pd.DataFrame, season_id: str) -> pd.DataFrame:
    rows: list[dict] = []
    unknown_teams: set[str] = set()

    for match_data in raw_df.to_dict(orient="records"):
        if not _is_completed(match_data.get("isResult")):
            continue

        raw_match_id = match_data.get("id")
        if raw_match_id is None or str(raw_match_id).strip() == "":
            raise ValueError(f"Understat match in {season_id} is missing id")
        match_id = str(raw_match_id).strip()

        home_data = _required_dict(match_data.get("h"), "home team data", match_id)
        away_data = _required_dict(match_data.get("a"), "away team data", match_id)
        goals_data = _required_dict(match_data.get("goals"), "goals", match_id)

        raw_home_team = home_data.get("title")
        raw_away_team = away_data.get("title")
        for raw_team_name in [raw_home_team, raw_away_team]:
            if str(raw_team_name).strip() not in UNDERSTAT_TEAM_MAPPING:
                unknown_teams.add(str(raw_team_name).strip())

        if unknown_teams:
            continue

        rows.append(
            {
                "understat_match_id": match_id,
                "season_id": season_id,
                "match_date": _extract_match_date(match_data, match_id),
                "home_team": normalize_understat_team_name(raw_home_team),
                "away_team": normalize_understat_team_name(raw_away_team),
                "home_xg": _extract_xg(match_data, "home", match_id),
                "away_xg": _extract_xg(match_data, "away", match_id),
                "home_goals": _to_int(goals_data.get("h"), "home_goals", match_id),
                "away_goals": _to_int(goals_data.get("a"), "away_goals", match_id),
                "source": "understat",
            }
        )

    if unknown_teams:
        missing = ", ".join(repr(team) for team in sorted(unknown_teams))
        raise ValueError(f"Unknown Understat team name(s) in {season_id}: {missing}")

    parsed_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return parsed_df


def validate_understat_season(
    df: pd.DataFrame,
    season_id: str,
    expected_matches: int = 380,
) -> None:
    season_df = df.loc[df["season_id"] == season_id].copy()
    errors: list[str] = []

    match_count = len(season_df)
    if expected_matches >= 0 and match_count != expected_matches:
        errors.append(f"expected {expected_matches} matches, found {match_count}")

    unique_teams = set(season_df["home_team"].dropna()) | set(
        season_df["away_team"].dropna()
    )
    if expected_matches == 380 and len(unique_teams) != 20:
        errors.append(f"expected 20 unique teams, found {len(unique_teams)}")

    for column in REQUIRED_OUTPUT_COLUMNS:
        null_count = season_df[column].isna().sum()
        if null_count > 0:
            errors.append(f"{column} has {null_count} null value(s)")

    duplicate_count = season_df.duplicated(
        subset=["season_id", "match_date", "home_team", "away_team"]
    ).sum()
    if duplicate_count > 0:
        errors.append(
            f"found {duplicate_count} duplicate season/date/home/away row(s)"
        )

    same_team_count = (season_df["home_team"] == season_df["away_team"]).sum()
    if same_team_count > 0:
        errors.append(f"found {same_team_count} home_team = away_team row(s)")

    negative_counts = {
        "home_xg": (season_df["home_xg"] < 0).sum(),
        "away_xg": (season_df["away_xg"] < 0).sum(),
        "home_goals": (season_df["home_goals"] < 0).sum(),
        "away_goals": (season_df["away_goals"] < 0).sum(),
    }
    for column, count in negative_counts.items():
        if count > 0:
            errors.append(f"{column} has {count} negative value(s)")

    non_integer_goal_count = (
        season_df["home_goals"].apply(lambda value: float(value).is_integer())
        & season_df["away_goals"].apply(lambda value: float(value).is_integer())
    ).eq(False).sum()
    if non_integer_goal_count > 0:
        errors.append(f"found {non_integer_goal_count} non-integer scoreline row(s)")

    if errors:
        raise ValueError(
            f"Understat validation failed for {season_id}: {'; '.join(errors)}"
        )

    min_date = season_df["match_date"].min() if not season_df.empty else "n/a"
    max_date = season_df["match_date"].max() if not season_df.empty else "n/a"
    print(
        f"Understat validation passed for {season_id}: "
        f"{match_count} matches, {len(unique_teams)} teams, {min_date} to {max_date}"
    )


def create_tier3_schema(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)


def verify_historical_understat_schema(engine) -> None:
    with engine.connect() as conn:
        table_exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'historical_understat_xg'
                )
                """
            )
        ).scalar_one()
        if not table_exists:
            raise RuntimeError("historical_understat_xg table does not exist")

        existing_columns = set(
            conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'historical_understat_xg'
                    """
                )
            ).scalars()
        )

    required_columns = {
        "understat_match_id",
        "season_id",
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "home_goals",
        "away_goals",
        "source",
        "created_at",
        "updated_at",
    }
    missing_columns = sorted(required_columns - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "historical_understat_xg is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    print("historical_understat_xg schema verification passed.")


def store_historical_understat_xg(
    df: pd.DataFrame,
    engine,
    season_id: str,
) -> None:
    season_df = df.loc[df["season_id"] == season_id, OUTPUT_COLUMNS].copy()
    records = season_df.to_dict(orient="records")

    insert_sql = text(
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
            :source
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM historical_understat_xg WHERE season_id = :season_id"),
            {"season_id": season_id},
        )
        if records:
            conn.execute(insert_sql, records)

    print(f"Stored {len(records)} historical_understat_xg rows for {season_id}")


def _get_imported_understat_seasons(engine) -> list[str]:
    with engine.connect() as conn:
        seasons = conn.execute(
            text(
                """
                SELECT season_id
                FROM historical_understat_xg
                GROUP BY season_id
                ORDER BY MIN(match_date), season_id
                """
            )
        ).scalars().all()
    return list(seasons)


def verify_understat_join_coverage(engine) -> None:
    print("=== Understat Join Coverage ===")
    imported_seasons = _get_imported_understat_seasons(engine)
    if not imported_seasons:
        raise RuntimeError("No historical_understat_xg seasons found for coverage check")

    with engine.connect() as conn:
        coverage_rows = conn.execute(
            text(
                """
                SELECT
                    hm.season_id,
                    COUNT(*) AS matched_rows
                FROM historical_matches hm
                INNER JOIN historical_understat_xg ux
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                WHERE hm.season_id = ANY(:seasons)
                GROUP BY hm.season_id
                ORDER BY hm.season_id
                """
            ),
            {"seasons": imported_seasons},
        ).mappings().all()

        football_unmatched_rows = conn.execute(
            text(
                """
                SELECT hm.season_id, hm.match_date, hm.home_team, hm.away_team
                FROM historical_matches hm
                LEFT JOIN historical_understat_xg ux
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                WHERE hm.season_id = ANY(:seasons)
                    AND ux.understat_match_id IS NULL
                ORDER BY hm.season_id, hm.match_date, hm.home_team, hm.away_team
                LIMIT 10
                """
            ),
            {"seasons": imported_seasons},
        ).mappings().all()

        football_unmatched_count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM historical_matches hm
                LEFT JOIN historical_understat_xg ux
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                WHERE hm.season_id = ANY(:seasons)
                    AND ux.understat_match_id IS NULL
                """
            ),
            {"seasons": imported_seasons},
        ).scalar_one()

        understat_unmatched_rows = conn.execute(
            text(
                """
                SELECT ux.season_id, ux.match_date, ux.home_team, ux.away_team
                FROM historical_understat_xg ux
                LEFT JOIN historical_matches hm
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                WHERE hm.match_id IS NULL
                ORDER BY ux.season_id, ux.match_date, ux.home_team, ux.away_team
                LIMIT 10
                """
            )
        ).mappings().all()

        understat_unmatched_count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM historical_understat_xg ux
                LEFT JOIN historical_matches hm
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                WHERE hm.match_id IS NULL
                """
            )
        ).scalar_one()

        duplicate_join_key_count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT season_id, match_date, home_team, away_team
                    FROM historical_matches
                    WHERE season_id = ANY(:seasons)
                    GROUP BY season_id, match_date, home_team, away_team
                    HAVING COUNT(*) > 1
                    UNION ALL
                    SELECT season_id, match_date, home_team, away_team
                    FROM historical_understat_xg
                    GROUP BY season_id, match_date, home_team, away_team
                    HAVING COUNT(*) > 1
                ) duplicate_keys
                """
            ),
            {"seasons": imported_seasons},
        ).scalar_one()

    coverage_by_season = {row["season_id"]: row["matched_rows"] for row in coverage_rows}
    for season_id in imported_seasons:
        matched_rows = coverage_by_season.get(season_id, 0)
        print(f"{season_id}: {matched_rows} matched rows")
        if matched_rows != 380:
            print(f"Warning: {season_id} expected 380 matched rows")

    print(f"Unmatched football-data matches: {football_unmatched_count}")
    if football_unmatched_rows:
        print(f"Unmatched football-data examples: {list(map(dict, football_unmatched_rows))}")

    print(f"Unmatched Understat matches: {understat_unmatched_count}")
    if understat_unmatched_rows:
        print(f"Unmatched Understat examples: {list(map(dict, understat_unmatched_rows))}")

    print(f"Duplicate join key count: {duplicate_join_key_count}")
    if (
        football_unmatched_count
        or understat_unmatched_count
        or duplicate_join_key_count
        or any(coverage_by_season.get(season_id, 0) != 380 for season_id in imported_seasons)
    ):
        raise ValueError("Understat join coverage validation failed")

    print("Understat join coverage validation passed.")


def print_understat_summary(engine) -> None:
    with engine.connect() as conn:
        total_rows = conn.execute(
            text("SELECT COUNT(*) FROM historical_understat_xg")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                """
                SELECT
                    ux.season_id,
                    COUNT(*) AS row_count,
                    (
                        SELECT COUNT(DISTINCT team_name)
                        FROM (
                            SELECT home_team AS team_name
                            FROM historical_understat_xg home_rows
                            WHERE home_rows.season_id = ux.season_id
                            UNION
                            SELECT away_team AS team_name
                            FROM historical_understat_xg away_rows
                            WHERE away_rows.season_id = ux.season_id
                        ) teams
                    ) AS unique_team_count,
                    MIN(ux.match_date) AS min_match_date,
                    MAX(ux.match_date) AS max_match_date,
                    COUNT(hm.match_id) AS matched_historical_rows
                FROM historical_understat_xg ux
                LEFT JOIN historical_matches hm
                    ON hm.season_id = ux.season_id
                    AND hm.match_date = ux.match_date
                    AND hm.home_team = ux.home_team
                    AND hm.away_team = ux.away_team
                GROUP BY ux.season_id
                ORDER BY ux.season_id
                """
            )
        ).mappings().all()

    print("=== Historical Understat Summary ===")
    print(f"historical_understat_xg total rows: {total_rows}")
    for row in season_rows:
        print(
            f"{row['season_id']}: {row['row_count']} rows, "
            f"{row['unique_team_count']} teams, "
            f"{row['min_match_date']} to {row['max_match_date']}, "
            f"{row['matched_historical_rows']} matched historical rows"
        )


def _verify_historical_matches_ready(engine) -> None:
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'historical_matches'
                )
                """
            )
        ).scalar_one()
        if not exists:
            raise RuntimeError("historical_matches table does not exist")

        count = conn.execute(text("SELECT COUNT(*) FROM historical_matches")).scalar_one()

    if count <= 0:
        raise RuntimeError("historical_matches has no rows")
    if count != 1900:
        print(f"Warning: historical_matches row count is {count}, expected 1900")
    else:
        print("historical_matches row count verified: 1900")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    try:
        _verify_historical_matches_ready(engine)
        create_tier3_schema(engine)
        verify_historical_understat_schema(engine)

        parsed_frames: list[pd.DataFrame] = []
        for season_id, understat_season in TARGET_UNDERSTAT_SEASONS.items():
            raw_df = fetch_understat_league_data(understat_season)
            parsed_df = parse_understat_matches(raw_df, season_id)
            validate_understat_season(parsed_df, season_id, expected_matches=380)
            parsed_frames.append(parsed_df)
            time.sleep(0.5)

        all_understat_df = pd.concat(parsed_frames, ignore_index=True)
        for season_id in TARGET_UNDERSTAT_SEASONS:
            store_historical_understat_xg(all_understat_df, engine, season_id)

        verify_understat_join_coverage(engine)
        print_understat_summary(engine)
    except Exception as error:
        print(f"Tier 3 Understat import failed: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    print("Tier 2 understat_xg and understat_team_history were not touched.")
    print("Tier 2 tables, model artifacts, Streamlit, and training behavior were not touched.")


if __name__ == "__main__":
    main()
