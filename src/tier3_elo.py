from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from data_pipeline import get_engine
from tier3_validation import validate_historical_match_integrity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

INITIAL_ELO = 1500.0
PROMOTED_ELO = 1400.0
HOME_ADVANTAGE = 50.0
K_FACTOR = 20.0
EXPECTED_MATCH_COUNT = 1900
EXPECTED_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
INITIALIZATION_TYPES = {"initial", "carried", "promoted_or_returning"}

MATCH_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]

ELO_TABLE_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_elo_before",
    "away_elo_before",
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "elo_diff_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
    "actual_home_score",
    "actual_away_score",
    "k_factor",
    "home_advantage",
    "home_initialization",
    "away_initialization",
]

REQUIRED_ELO_VALIDATION_COLUMNS = [
    column for column in ELO_TABLE_COLUMNS if column != "kickoff_time"
]

SAFETY_COUNT_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
]


def _table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return conn.execute(
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


def _count_table_rows(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def _query_mappings(
    engine,
    query: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(query), params or {}).mappings().all()
    return [dict(row) for row in rows]


def _record_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def load_historical_matches(engine) -> pd.DataFrame:
    query = text(
        """
        WITH season_order AS (
            SELECT season_id, MIN(match_date) AS season_start
            FROM historical_matches
            GROUP BY season_id
        )
        SELECT
            hm.match_id,
            hm.season_id,
            hm.match_date,
            hm.kickoff_time,
            hm.home_team,
            hm.away_team,
            hm.home_goals,
            hm.away_goals,
            hm.result
        FROM historical_matches hm
        INNER JOIN season_order so
            ON hm.season_id = so.season_id
        ORDER BY
            so.season_start,
            COALESCE(hm.kickoff_time, hm.match_date::timestamp),
            hm.match_id
        """
    )
    matches_df = pd.read_sql(query, engine)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"]).dt.date
    matches_df["kickoff_time"] = pd.to_datetime(
        matches_df["kickoff_time"],
        errors="coerce",
    )
    matches_df["event_time"] = matches_df["kickoff_time"].fillna(
        pd.to_datetime(matches_df["match_date"])
    )

    errors: list[str] = []
    if len(matches_df) != EXPECTED_MATCH_COUNT:
        errors.append(f"expected {EXPECTED_MATCH_COUNT} rows, found {len(matches_df)}")
    if matches_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(matches_df['match_id'].duplicated().sum())}"
        )

    required_columns = [column for column in MATCH_COLUMNS if column != "kickoff_time"]
    null_counts = matches_df[required_columns].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"null required match columns: {bad_nulls}")

    expected_result = np.where(
        matches_df["home_goals"] > matches_df["away_goals"],
        "H",
        np.where(matches_df["home_goals"] < matches_df["away_goals"], "A", "D"),
    )
    mismatches = matches_df.loc[matches_df["result"].to_numpy() != expected_result]
    if not mismatches.empty:
        examples = mismatches[
            ["match_id", "home_team", "away_team", "home_goals", "away_goals", "result"]
        ].head(5).to_dict(orient="records")
        errors.append(
            f"result/goals mismatch count: {len(mismatches)}, examples: {examples}"
        )

    if errors:
        raise ValueError("Historical match Elo load validation failed: " + "; ".join(errors))

    print("=== Historical Matches For Elo ===")
    season_counts = matches_df.groupby("season_id").size().sort_index()
    for season_id, count in season_counts.items():
        print(f"{season_id}: {int(count)} rows")
    print(f"Loaded {len(matches_df)} historical match rows for Elo.")

    return matches_df


def get_season_team_sets(matches_df: pd.DataFrame) -> dict[str, set[str]]:
    season_team_sets: dict[str, set[str]] = {}
    season_order = matches_df.groupby("season_id")["event_time"].min().sort_values()
    for season_id in season_order.index:
        season_df = matches_df.loc[matches_df["season_id"] == season_id]
        season_team_sets[season_id] = set(season_df["home_team"]) | set(
            season_df["away_team"]
        )
    return season_team_sets


def initialize_season_elos(
    season_id: str,
    season_index: int,
    teams_in_season: set[str],
    previous_season_teams: set[str],
    elo_state: dict[str, float],
) -> dict[str, str]:
    initialization_types: dict[str, str] = {}
    for team in sorted(teams_in_season):
        if season_index == 0:
            elo_state[team] = INITIAL_ELO
            initialization_types[team] = "initial"
        elif team in previous_season_teams and team in elo_state:
            initialization_types[team] = "carried"
        else:
            elo_state[team] = PROMOTED_ELO
            initialization_types[team] = "promoted_or_returning"

    print(
        f"{season_id} Elo initialization: "
        f"initial={sum(value == 'initial' for value in initialization_types.values())}, "
        f"carried={sum(value == 'carried' for value in initialization_types.values())}, "
        "promoted_or_returning="
        f"{sum(value == 'promoted_or_returning' for value in initialization_types.values())}"
    )
    return initialization_types


def expected_score(
    home_elo: float,
    away_elo: float,
    home_advantage: float,
) -> tuple[float, float]:
    adjusted_home_elo = home_elo + home_advantage
    expected_home = 1 / (1 + 10 ** ((away_elo - adjusted_home_elo) / 400))
    expected_away = 1 - expected_home
    return float(expected_home), float(expected_away)


def actual_scores(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals == away_goals:
        return 0.5, 0.5
    return 0.0, 1.0


def compute_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    season_team_sets = get_season_team_sets(matches_df)
    season_order = list(season_team_sets.keys())
    elo_state: dict[str, float] = {}
    previous_season_teams: set[str] = set()
    rows: list[dict[str, Any]] = []

    for season_index, season_id in enumerate(season_order):
        teams_in_season = season_team_sets[season_id]
        initialization_types = initialize_season_elos(
            season_id,
            season_index,
            teams_in_season,
            previous_season_teams,
            elo_state,
        )
        season_df = matches_df.loc[matches_df["season_id"] == season_id].sort_values(
            ["event_time", "match_id"]
        )

        for match in season_df.itertuples(index=False):
            home_elo_before = float(elo_state[match.home_team])
            away_elo_before = float(elo_state[match.away_team])
            expected_home, expected_away = expected_score(
                home_elo_before,
                away_elo_before,
                HOME_ADVANTAGE,
            )
            actual_home, actual_away = actual_scores(match.home_goals, match.away_goals)
            home_delta = K_FACTOR * (actual_home - expected_home)
            away_delta = K_FACTOR * (actual_away - expected_away)
            home_elo_after = home_elo_before + home_delta
            away_elo_after = away_elo_before + away_delta

            rows.append(
                {
                    "match_id": int(match.match_id),
                    "season_id": match.season_id,
                    "match_date": match.match_date,
                    "kickoff_time": None
                    if pd.isna(match.kickoff_time)
                    else match.kickoff_time.to_pydatetime(),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": int(match.home_goals),
                    "away_goals": int(match.away_goals),
                    "result": match.result,
                    "home_elo_before": home_elo_before,
                    "away_elo_before": away_elo_before,
                    "home_elo_after": home_elo_after,
                    "away_elo_after": away_elo_after,
                    "home_elo_delta": home_delta,
                    "away_elo_delta": away_delta,
                    "elo_diff_before": home_elo_before - away_elo_before,
                    "elo_diff_home_adjusted": (
                        home_elo_before + HOME_ADVANTAGE
                    )
                    - away_elo_before,
                    "expected_home_score": expected_home,
                    "expected_away_score": expected_away,
                    "actual_home_score": actual_home,
                    "actual_away_score": actual_away,
                    "k_factor": K_FACTOR,
                    "home_advantage": HOME_ADVANTAGE,
                    "home_initialization": initialization_types[match.home_team],
                    "away_initialization": initialization_types[match.away_team],
                }
            )

            elo_state[match.home_team] = home_elo_after
            elo_state[match.away_team] = away_elo_after

        previous_season_teams = teams_in_season

    elo_df = pd.DataFrame(rows, columns=ELO_TABLE_COLUMNS)
    print(f"Computed {len(elo_df)} Elo rating rows.")
    return elo_df


def _validate_season_initializations(elo_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    season_team_sets: dict[str, set[str]] = {}
    for season_id in EXPECTED_SEASONS:
        season_df = elo_df.loc[elo_df["season_id"] == season_id]
        season_team_sets[season_id] = set(season_df["home_team"]) | set(
            season_df["away_team"]
        )

    first_season = EXPECTED_SEASONS[0]
    first_season_df = elo_df.loc[elo_df["season_id"] == first_season]
    first_types = set(first_season_df["home_initialization"]) | set(
        first_season_df["away_initialization"]
    )
    if first_types != {"initial"}:
        errors.append(f"first season initialization types were {sorted(first_types)}")

    for season_id in EXPECTED_SEASONS[1:]:
        season_df = elo_df.loc[elo_df["season_id"] == season_id]
        for column in ["home_initialization", "away_initialization"]:
            unknown_types = set(season_df[column]) - INITIALIZATION_TYPES
            if unknown_types:
                errors.append(
                    f"{season_id} {column} unknown initialization types: "
                    f"{sorted(unknown_types)}"
                )
        carried_or_promoted = set(season_df["home_initialization"]) | set(
            season_df["away_initialization"]
        )
        if "initial" in carried_or_promoted:
            errors.append(f"{season_id} contains unexpected initial initialization")

        previous_teams = season_team_sets[EXPECTED_SEASONS[EXPECTED_SEASONS.index(season_id) - 1]]
        for team in sorted(season_team_sets[season_id]):
            expected_type = (
                "carried" if team in previous_teams else "promoted_or_returning"
            )
            team_rows = season_df.loc[
                (season_df["home_team"] == team) | (season_df["away_team"] == team)
            ]
            actual_types = set(
                team_rows.loc[team_rows["home_team"] == team, "home_initialization"]
            ) | set(
                team_rows.loc[team_rows["away_team"] == team, "away_initialization"]
            )
            if actual_types != {expected_type}:
                errors.append(
                    f"{season_id} {team} expected {expected_type}, "
                    f"found {sorted(actual_types)}"
                )

    return errors


def _validate_chronological_before_values(
    elo_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    season_team_sets = get_season_team_sets(matches_df)
    previous_season_teams: set[str] = set()
    elo_state: dict[str, float] = {}
    rows_by_match_id = elo_df.set_index("match_id")

    for season_index, season_id in enumerate(season_team_sets.keys()):
        initialization_types = initialize_season_elos(
            season_id,
            season_index,
            season_team_sets[season_id],
            previous_season_teams,
            elo_state,
        )
        del initialization_types
        season_matches = matches_df.loc[matches_df["season_id"] == season_id].sort_values(
            ["event_time", "match_id"]
        )
        for match in season_matches.itertuples(index=False):
            elo_row = rows_by_match_id.loc[match.match_id]
            home_expected_before = elo_state[match.home_team]
            away_expected_before = elo_state[match.away_team]
            if not np.isclose(elo_row["home_elo_before"], home_expected_before):
                errors.append(
                    f"match {match.match_id} home before "
                    f"{elo_row['home_elo_before']} != state {home_expected_before}"
                )
                break
            if not np.isclose(elo_row["away_elo_before"], away_expected_before):
                errors.append(
                    f"match {match.match_id} away before "
                    f"{elo_row['away_elo_before']} != state {away_expected_before}"
                )
                break

            elo_state[match.home_team] = float(elo_row["home_elo_after"])
            elo_state[match.away_team] = float(elo_row["away_elo_after"])

        previous_season_teams = season_team_sets[season_id]
        if errors:
            break

    return errors


def validate_elo_ratings(elo_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    print("=== Elo Validation ===")
    errors: list[str] = []

    if len(elo_df) != len(matches_df):
        errors.append(f"Elo row count {len(elo_df)} != match row count {len(matches_df)}")
    if elo_df["match_id"].nunique() != len(matches_df):
        errors.append("Elo table does not have one row per match_id")
    if elo_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate Elo match_id count: {int(elo_df['match_id'].duplicated().sum())}"
        )

    null_counts = elo_df[REQUIRED_ELO_VALIDATION_COLUMNS].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"null required Elo columns: {bad_nulls}")

    if not np.allclose(
        elo_df["expected_home_score"] + elo_df["expected_away_score"],
        1.0,
    ):
        errors.append("expected_home_score + expected_away_score check failed")
    if not np.allclose(
        elo_df["actual_home_score"] + elo_df["actual_away_score"],
        1.0,
    ):
        errors.append("actual_home_score + actual_away_score check failed")
    if not np.allclose(elo_df["home_elo_delta"] + elo_df["away_elo_delta"], 0.0):
        errors.append("home_elo_delta + away_elo_delta check failed")
    if not np.allclose(
        elo_df["home_elo_after"],
        elo_df["home_elo_before"] + elo_df["home_elo_delta"],
    ):
        errors.append("home_elo_after formula check failed")
    if not np.allclose(
        elo_df["away_elo_after"],
        elo_df["away_elo_before"] + elo_df["away_elo_delta"],
    ):
        errors.append("away_elo_after formula check failed")

    errors.extend(_validate_chronological_before_values(elo_df, matches_df))
    errors.extend(_validate_season_initializations(elo_df))

    expected_result = np.where(
        elo_df["actual_home_score"] == 1.0,
        "H",
        np.where(elo_df["actual_away_score"] == 1.0, "A", "D"),
    )
    if (elo_df["result"].to_numpy() != expected_result).any():
        errors.append("result does not match actual Elo scores")

    season_counts = elo_df.groupby("season_id").size().sort_index()
    bad_season_counts = {
        season_id: int(count)
        for season_id, count in season_counts.items()
        if int(count) != 380
    }
    missing_seasons = [season_id for season_id in EXPECTED_SEASONS if season_id not in season_counts]
    if bad_season_counts:
        errors.append(f"bad Elo row counts by season: {bad_season_counts}")
    if missing_seasons:
        errors.append(f"missing Elo seasons: {missing_seasons}")

    print("Elo before summary by season:")
    summary = elo_df.groupby("season_id").agg(
        min_home_elo_before=("home_elo_before", "min"),
        max_home_elo_before=("home_elo_before", "max"),
        mean_home_elo_before=("home_elo_before", "mean"),
        min_away_elo_before=("away_elo_before", "min"),
        max_away_elo_before=("away_elo_before", "max"),
        mean_away_elo_before=("away_elo_before", "mean"),
    )
    for season_id, row in summary.iterrows():
        print(
            f"{season_id}: "
            f"home min/max/mean={row['min_home_elo_before']:.1f}/"
            f"{row['max_home_elo_before']:.1f}/{row['mean_home_elo_before']:.1f}; "
            f"away min/max/mean={row['min_away_elo_before']:.1f}/"
            f"{row['max_away_elo_before']:.1f}/{row['mean_away_elo_before']:.1f}"
        )

    if errors:
        print("Elo validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Elo validation failed")

    print("PASS: Elo row count matches historical_matches")
    print("PASS: one Elo row per match_id and no duplicate match_id")
    print("PASS: no null required Elo fields")
    print("PASS: expected score sum check")
    print("PASS: actual score sum check")
    print("PASS: Elo delta sum check")
    print("PASS: Elo after formulas")
    print("PASS: chronological before-Elo replay")
    print("PASS: initialization rules")
    print("PASS: result matches actual Elo scores")
    print("PASS: row counts by season are 380 each")
    print("Elo validation passed.")


def create_or_verify_elo_table(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)

    if not _table_exists(engine, "elo_ratings_v3"):
        raise RuntimeError("elo_ratings_v3 table does not exist")

    rows = _query_mappings(
        engine,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = CURRENT_SCHEMA()
            AND table_name = 'elo_ratings_v3'
        """,
    )
    existing_columns = {row["column_name"] for row in rows}
    required_columns = set(ELO_TABLE_COLUMNS + ["created_at"])
    missing_columns = sorted(required_columns - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "elo_ratings_v3 is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    print("elo_ratings_v3 schema verification passed.")


def store_elo_ratings(elo_df: pd.DataFrame, engine) -> None:
    records = [
        {column: _record_value(row[column]) for column in ELO_TABLE_COLUMNS}
        for row in elo_df[ELO_TABLE_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(ELO_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in ELO_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO elo_ratings_v3 (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM elo_ratings_v3"))
        conn.execute(insert_sql, records)

    print(f"Stored {len(records)} rows in elo_ratings_v3")


def capture_table_counts(engine, table_names: list[str]) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    for table_name in table_names:
        if _table_exists(engine, table_name):
            counts[table_name] = _count_table_rows(engine, table_name)
        else:
            counts[table_name] = "MISSING"
    return counts


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name, count in counts.items():
        print(f"{table_name}: {count}")


def _verify_counts_unchanged(
    before_counts: dict[str, int | str],
    after_counts: dict[str, int | str],
) -> None:
    changed = {
        table_name: (before_counts.get(table_name), after_counts.get(table_name))
        for table_name in sorted(set(before_counts) | set(after_counts))
        if before_counts.get(table_name) != after_counts.get(table_name)
    }
    if changed:
        raise RuntimeError(f"Safety table counts changed unexpectedly: {changed}")
    print("Safety counts unchanged for Tier 2 and Tier 3 source/feature tables.")


def print_elo_summary(engine) -> None:
    with engine.connect() as conn:
        total_rows = conn.execute(
            text("SELECT COUNT(*) FROM elo_ratings_v3")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                """
                SELECT
                    season_id,
                    COUNT(*) AS row_count
                FROM elo_ratings_v3
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        elo_summary = conn.execute(
            text(
                """
                SELECT
                    season_id,
                    MIN(home_elo_before) AS min_home_elo_before,
                    MAX(home_elo_before) AS max_home_elo_before,
                    AVG(home_elo_before) AS mean_home_elo_before,
                    MIN(away_elo_before) AS min_away_elo_before,
                    MAX(away_elo_before) AS max_away_elo_before,
                    AVG(away_elo_before) AS mean_away_elo_before
                FROM elo_ratings_v3
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        initialization_counts = conn.execute(
            text(
                """
                SELECT season_id, initialization, COUNT(DISTINCT team) AS team_count
                FROM (
                    SELECT season_id, home_team AS team, home_initialization AS initialization
                    FROM elo_ratings_v3
                    UNION
                    SELECT season_id, away_team AS team, away_initialization AS initialization
                    FROM elo_ratings_v3
                ) team_initializations
                GROUP BY season_id, initialization
                ORDER BY season_id, initialization
                """
            )
        ).mappings().all()
        top_teams = conn.execute(
            text(
                """
                WITH last_team_rows AS (
                    SELECT
                        team,
                        elo_after,
                        ROW_NUMBER() OVER (
                            PARTITION BY team
                            ORDER BY event_time DESC, match_id DESC
                        ) AS row_num
                    FROM (
                        SELECT
                            home_team AS team,
                            home_elo_after AS elo_after,
                            COALESCE(kickoff_time, match_date::timestamp) AS event_time,
                            match_id
                        FROM elo_ratings_v3
                        UNION ALL
                        SELECT
                            away_team AS team,
                            away_elo_after AS elo_after,
                            COALESCE(kickoff_time, match_date::timestamp) AS event_time,
                            match_id
                        FROM elo_ratings_v3
                    ) team_rows
                )
                SELECT team, elo_after
                FROM last_team_rows
                WHERE row_num = 1
                ORDER BY elo_after DESC, team
                LIMIT 10
                """
            )
        ).mappings().all()

    print("=== Elo Summary ===")
    print(f"elo_ratings_v3 total rows: {total_rows}")
    print("Rows by season:")
    for row in season_rows:
        print(f"- {row['season_id']}: {row['row_count']}")

    print("Home/Away Elo before min/max/mean by season:")
    for row in elo_summary:
        print(
            f"- {row['season_id']}: "
            f"home {row['min_home_elo_before']:.1f}/"
            f"{row['max_home_elo_before']:.1f}/"
            f"{row['mean_home_elo_before']:.1f}; "
            f"away {row['min_away_elo_before']:.1f}/"
            f"{row['max_away_elo_before']:.1f}/"
            f"{row['mean_away_elo_before']:.1f}"
        )

    print("Initialization counts by season:")
    for row in initialization_counts:
        print(
            f"- {row['season_id']} {row['initialization']}: "
            f"{row['team_count']}"
        )

    print("Top 10 teams by latest Elo after final imported match:")
    for index, row in enumerate(top_teams, start=1):
        print(f"{index}. {row['team']}: {row['elo_after']:.1f}")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    print("=== Tier 3 Elo Build ===")
    print(
        "Constants: "
        f"INITIAL_ELO={INITIAL_ELO}, PROMOTED_ELO={PROMOTED_ELO}, "
        f"HOME_ADVANTAGE={HOME_ADVANTAGE}, K_FACTOR={K_FACTOR}"
    )
    validate_historical_match_integrity(engine)

    matches_df = load_historical_matches(engine)
    elo_df = compute_elo_ratings(matches_df)
    validate_elo_ratings(elo_df, matches_df)

    before_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts before Elo write", before_counts)

    create_or_verify_elo_table(engine)
    store_elo_ratings(elo_df, engine)

    after_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts after Elo write", after_counts)
    _verify_counts_unchanged(before_counts, after_counts)

    print_elo_summary(engine)
    print("2025-26 received Elo rows but remains reserved as final test for modeling.")
    print("No model training occurred.")
    print("Tier 2 tables, match_features_v3_base, Streamlit, and model artifacts were not touched.")


if __name__ == "__main__":
    main()
