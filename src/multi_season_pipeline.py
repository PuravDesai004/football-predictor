from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import text

from data_pipeline import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "historical"
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

TARGET_SEASONS: dict[str, str] = {
    "2021-22": "E0_2021-22.csv",
    "2022-23": "E0_2022-23.csv",
    "2023-24": "E0_2023-24.csv",
    "2024-25": "E0_2024-25.csv",
    "2025-26": "E0_2025-26.csv",
}

FOOTBALL_DATA_TEAM_MAPPING: dict[str, str] = {
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
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Man City",
    "Man United": "Man Utd",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "Watford": "Watford",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
TIER3_TABLES = ["seasons", "team_name_mapping", "historical_matches"]
REQUIRED_HISTORICAL_MATCH_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "source",
    "source_file",
    "created_at",
    "updated_at",
]
INSERT_COLUMNS = [
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "source",
    "source_file",
]


def normalize_team_name(raw_name: str, source: str = "football_data") -> str:
    if source != "football_data":
        raise ValueError(f"Unsupported team source: {source}")

    source_name = str(raw_name).strip()
    try:
        return FOOTBALL_DATA_TEAM_MAPPING[source_name]
    except KeyError as error:
        raise ValueError(f"Unknown football_data team name: {raw_name}") from error


def _result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _format_expected_files(season_config: dict[str, str]) -> str:
    return "\n".join(f"- {filename}" for filename in season_config.values())


def _extract_source_team_names(series: pd.Series) -> set[str]:
    return set(series.dropna().astype(str).str.strip())


def parse_football_data_csv(filepath: Path, season_id: str) -> pd.DataFrame:
    raw_df = pd.read_csv(filepath)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(
            f"{filepath.name} is missing required column(s): {', '.join(missing_columns)}"
        )

    completed_df = raw_df.dropna(subset=["FTHG", "FTAG", "FTR"]).copy()
    completed_df["match_date"] = pd.to_datetime(
        completed_df["Date"],
        dayfirst=True,
        errors="coerce",
    ).dt.date

    bad_dates = completed_df.loc[completed_df["match_date"].isna(), "Date"].unique()
    if len(bad_dates) > 0:
        raise ValueError(
            f"{filepath.name} has unparseable Date value(s) in season {season_id}: "
            f"{list(bad_dates)}"
        )

    home_goal_values = pd.to_numeric(completed_df["FTHG"], errors="coerce")
    away_goal_values = pd.to_numeric(completed_df["FTAG"], errors="coerce")
    if home_goal_values.isna().any() or away_goal_values.isna().any():
        raise ValueError(f"{filepath.name} has non-numeric goals in season {season_id}")

    completed_df["home_goals"] = home_goal_values.astype(int)
    completed_df["away_goals"] = away_goal_values.astype(int)

    source_home_names = _extract_source_team_names(completed_df["HomeTeam"])
    source_away_names = _extract_source_team_names(completed_df["AwayTeam"])
    unknown_team_names = sorted(
        (source_home_names | source_away_names) - set(FOOTBALL_DATA_TEAM_MAPPING)
    )
    if unknown_team_names:
        missing = ", ".join(repr(name) for name in unknown_team_names)
        raise ValueError(
            f"Unknown football_data team name(s) in season {season_id}: {missing}"
        )

    completed_df["home_team"] = completed_df["HomeTeam"].apply(normalize_team_name)
    completed_df["away_team"] = completed_df["AwayTeam"].apply(normalize_team_name)
    completed_df["result"] = [
        _result_from_goals(home_goals, away_goals)
        for home_goals, away_goals in zip(
            completed_df["home_goals"],
            completed_df["away_goals"],
        )
    ]

    source_results = completed_df["FTR"].astype(str).str.strip().str.upper()
    invalid_results = sorted(set(source_results) - {"H", "D", "A"})
    if invalid_results:
        raise ValueError(
            f"{filepath.name} has invalid FTR value(s) in season {season_id}: "
            f"{invalid_results}"
        )

    mismatches = completed_df.loc[source_results != completed_df["result"]]
    if not mismatches.empty:
        examples = mismatches[
            ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        ].head(5)
        raise ValueError(
            f"{filepath.name} has FTR values that do not match goals in season "
            f"{season_id}. Examples: {examples.to_dict(orient='records')}"
        )

    kickoff_time = pd.Series(pd.NaT, index=completed_df.index, dtype="datetime64[ns]")
    if "Time" in completed_df.columns:
        time_values = completed_df["Time"].astype("string").str.strip()
        has_time = completed_df["Time"].notna() & (time_values != "")
        if has_time.any():
            combined_values = (
                completed_df.loc[has_time, "match_date"].astype(str)
                + " "
                + time_values.loc[has_time]
            )
            parsed_kickoff = pd.to_datetime(combined_values, errors="coerce")
            bad_times = completed_df.loc[has_time].loc[
                parsed_kickoff.isna(),
                ["Date", "Time", "HomeTeam", "AwayTeam"],
            ]
            if not bad_times.empty:
                raise ValueError(
                    f"{filepath.name} has unparseable Time value(s) in season "
                    f"{season_id}: {bad_times.to_dict(orient='records')}"
                )
            kickoff_time.loc[has_time] = parsed_kickoff

    parsed_df = pd.DataFrame(
        {
            "season_id": season_id,
            "match_date": completed_df["match_date"],
            "kickoff_time": kickoff_time,
            "home_team": completed_df["home_team"],
            "away_team": completed_df["away_team"],
            "home_goals": completed_df["home_goals"],
            "away_goals": completed_df["away_goals"],
            "result": completed_df["result"],
            "source": "football_data",
            "source_file": filepath.name,
        }
    )

    return parsed_df[INSERT_COLUMNS].reset_index(drop=True)


def validate_season(
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

    for column in ["home_team", "away_team", "home_goals", "away_goals", "result"]:
        null_count = season_df[column].isna().sum()
        if null_count > 0:
            errors.append(f"{column} has {null_count} null value(s)")

    if season_df["match_date"].isna().any():
        errors.append("match_date has unparsed value(s)")

    duplicate_count = season_df.duplicated(
        subset=["season_id", "home_team", "away_team"]
    ).sum()
    if duplicate_count > 0:
        errors.append(
            f"found {duplicate_count} duplicate season/home_team/away_team row(s)"
        )

    same_team_count = (season_df["home_team"] == season_df["away_team"]).sum()
    if same_team_count > 0:
        errors.append(f"found {same_team_count} row(s) with identical home and away team")

    invalid_results = sorted(set(season_df["result"]) - {"H", "D", "A"})
    if invalid_results:
        errors.append(f"invalid result value(s): {invalid_results}")

    season_df["expected_result"] = [
        _result_from_goals(home_goals, away_goals)
        for home_goals, away_goals in zip(
            season_df["home_goals"],
            season_df["away_goals"],
        )
    ]
    result_mismatches = season_df.loc[
        season_df["result"].fillna("<NA>").astype(str) != season_df["expected_result"],
        [
            "match_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "expected_result",
        ],
    ]
    if not result_mismatches.empty:
        examples = result_mismatches.head(5).to_dict(orient="records")
        errors.append(
            "result does not match home_goals and away_goals for "
            f"{len(result_mismatches)} row(s). Examples: {examples}"
        )

    negative_goals = (
        (season_df["home_goals"] < 0) | (season_df["away_goals"] < 0)
    ).sum()
    if negative_goals > 0:
        errors.append(f"found {negative_goals} row(s) with negative goals")

    if errors:
        raise ValueError(f"Validation failed for {season_id}: {'; '.join(errors)}")

    min_date = season_df["match_date"].min() if not season_df.empty else "n/a"
    max_date = season_df["match_date"].max() if not season_df.empty else "n/a"
    print(
        f"Validation passed for {season_id}: "
        f"{match_count} matches, {len(unique_teams)} teams, {min_date} to {max_date}"
    )


def create_tier3_schema(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(schema_sql)
    except Exception as error:
        print(f"Tier 3 schema creation failed: {type(error).__name__}: {error}")
        raise


def verify_tier3_schema(engine) -> None:
    missing_tables: list[str] = []
    with engine.connect() as conn:
        for table_name in TIER3_TABLES:
            exists = conn.execute(
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
            if not exists:
                missing_tables.append(table_name)

        if missing_tables:
            raise RuntimeError(
                f"Missing Tier 3 table(s): {', '.join(missing_tables)}"
            )

        existing_columns = set(
            conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'historical_matches'
                    """
                )
            ).scalars()
        )

    missing_columns = [
        column
        for column in REQUIRED_HISTORICAL_MATCH_COLUMNS
        if column not in existing_columns
    ]
    if missing_columns:
        raise RuntimeError(
            "historical_matches is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    print("Tier 3 schema verification passed.")


def _season_bounds(season_id: str) -> tuple[str, str]:
    start_year = int(season_id[:4])
    end_year = int(f"20{season_id[-2:]}")
    return f"{start_year}-08-01", f"{end_year}-05-31"


def seed_seasons(engine, imported_seasons: list[str]) -> None:
    if not imported_seasons:
        return

    current_season = sorted(imported_seasons)[-1]
    upsert_sql = text(
        """
        INSERT INTO seasons (season_id, season_name, start_date, end_date, is_current)
        VALUES (:season_id, :season_name, :start_date, :end_date, :is_current)
        ON CONFLICT (season_id) DO UPDATE SET
            season_name = EXCLUDED.season_name,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            is_current = EXCLUDED.is_current
        """
    )

    with engine.begin() as conn:
        conn.execute(text("UPDATE seasons SET is_current = FALSE"))
        for season_id in imported_seasons:
            start_date, end_date = _season_bounds(season_id)
            conn.execute(
                upsert_sql,
                {
                    "season_id": season_id,
                    "season_name": f"{season_id} Premier League",
                    "start_date": start_date,
                    "end_date": end_date,
                    "is_current": season_id == current_season,
                },
            )


def seed_team_name_mapping(engine) -> None:
    upsert_sql = text(
        """
        INSERT INTO team_name_mapping (source, source_name, canonical_name)
        VALUES (:source, :source_name, :canonical_name)
        ON CONFLICT (source, source_name) DO UPDATE SET
            canonical_name = EXCLUDED.canonical_name
        """
    )
    rows = [
        {
            "source": "football_data",
            "source_name": source_name,
            "canonical_name": canonical_name,
        }
        for source_name, canonical_name in sorted(FOOTBALL_DATA_TEAM_MAPPING.items())
    ]

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)


def store_historical_matches(df: pd.DataFrame, engine, season_id: str) -> None:
    season_df = df.loc[df["season_id"] == season_id, INSERT_COLUMNS].copy()
    records = season_df.to_dict(orient="records")
    for record in records:
        if pd.isna(record["kickoff_time"]):
            record["kickoff_time"] = None

    insert_sql = text(
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
            :kickoff_time,
            :home_team,
            :away_team,
            :home_goals,
            :away_goals,
            :result,
            :source,
            :source_file
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM historical_matches WHERE season_id = :season_id"),
            {"season_id": season_id},
        )
        if records:
            conn.execute(insert_sql, records)

    print(f"Stored {len(records)} historical_matches rows for {season_id}")


def load_all_historical_csvs(
    data_dir: Path,
    season_config: dict[str, str],
) -> pd.DataFrame:
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created missing directory: {data_dir}")
        print("Place local football-data.co.uk CSV files here with these names:")
        print(_format_expected_files(season_config))

    available: list[tuple[str, Path]] = []
    missing: list[str] = []
    for season_id, filename in season_config.items():
        filepath = data_dir / filename
        if filepath.exists():
            available.append((season_id, filepath))
        else:
            missing.append(filename)

    if missing:
        print("Missing CSV file(s):")
        for filename in missing:
            print(f"- {filename}")

    if not available:
        print("No target historical CSV files found. Nothing to import.")
        print("Expected file names:")
        print(_format_expected_files(season_config))
        return pd.DataFrame(columns=INSERT_COLUMNS)

    parsed_frames: list[pd.DataFrame] = []
    for season_id, filepath in available:
        parsed_df = parse_football_data_csv(filepath, season_id)
        if season_id == "2025-26" and len(parsed_df) < 380:
            print(
                f"Warning: {season_id} appears incomplete with "
                f"{len(parsed_df)} completed match(es). Importing available rows only."
            )
            validate_season(parsed_df, season_id, expected_matches=len(parsed_df))
        else:
            validate_season(parsed_df, season_id, expected_matches=380)
        parsed_frames.append(parsed_df)

    all_matches_df = pd.concat(parsed_frames, ignore_index=True)
    imported_seasons = sorted(all_matches_df["season_id"].unique())
    print(f"Imported season(s) from CSV: {', '.join(imported_seasons)}")
    return all_matches_df


def print_ingestion_summary(engine) -> None:
    with engine.connect() as conn:
        seasons_count = conn.execute(text("SELECT COUNT(*) FROM seasons")).scalar_one()
        mapping_count = conn.execute(
            text("SELECT COUNT(*) FROM team_name_mapping")
        ).scalar_one()
        total_matches = conn.execute(
            text("SELECT COUNT(*) FROM historical_matches")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                """
                SELECT
                    hm.season_id,
                    COUNT(*) AS match_count,
                    (
                        SELECT COUNT(DISTINCT team_name)
                        FROM (
                            SELECT home_team AS team_name
                            FROM historical_matches hm_home
                            WHERE hm_home.season_id = hm.season_id
                            UNION
                            SELECT away_team AS team_name
                            FROM historical_matches hm_away
                            WHERE hm_away.season_id = hm.season_id
                        ) teams
                    ) AS unique_team_count,
                    MIN(hm.match_date) AS min_match_date,
                    MAX(hm.match_date) AS max_match_date
                FROM historical_matches hm
                GROUP BY hm.season_id
                ORDER BY hm.season_id
                """
            )
        ).mappings().all()

    print("Tier 3 ingestion summary:")
    print(f"- seasons row count: {seasons_count}")
    print(f"- team_name_mapping row count: {mapping_count}")
    print(f"- historical_matches total row count: {total_matches}")
    for row in season_rows:
        print(
            f"- {row['season_id']}: {row['match_count']} matches, "
            f"{row['unique_team_count']} teams, "
            f"{row['min_match_date']} to {row['max_match_date']}"
        )


def print_tier3_table_counts(engine) -> dict[str, int]:
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for table_name in TIER3_TABLES:
            counts[table_name] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()

    print("Tier 3 table row counts:")
    for table_name in TIER3_TABLES:
        print(f"- {table_name}: {counts[table_name]}")

    return counts


def run_init_schema_only() -> None:
    engine = get_engine()
    if engine is None:
        print("Could not connect to PostgreSQL. Check local DB settings and retry.")
        raise SystemExit(1)

    try:
        create_tier3_schema(engine)
        verify_tier3_schema(engine)
        seed_team_name_mapping(engine)
        print_tier3_table_counts(engine)
    except Exception as error:
        print(f"Tier 3 schema initialization failed: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    print("No historical_matches rows were written.")
    print("Tier 2 tables were not modified.")
    print("Model artifacts were not touched.")


def _write_self_check_csv(filepath: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(filepath, index=False)


def _require_self_check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_parser_self_check() -> None:
    season_id = "self-check"
    valid_rows = [
        {
            "Date": "13/08/2021",
            "Time": "20:00",
            "HomeTeam": "Brentford",
            "AwayTeam": "Arsenal",
            "FTHG": 2,
            "FTAG": 0,
            "FTR": "H",
        },
        {
            "Date": "14/08/2021",
            "Time": "",
            "HomeTeam": "Man United",
            "AwayTeam": "Leeds",
            "FTHG": 5,
            "FTAG": 1,
            "FTR": "H",
        },
        {
            "Date": "15/08/2021",
            "Time": "14:00",
            "HomeTeam": "Newcastle",
            "AwayTeam": "West Ham",
            "FTHG": 2,
            "FTAG": 4,
            "FTR": "A",
        },
        {
            "Date": "16/08/2021",
            "Time": "20:00",
            "HomeTeam": "Everton",
            "AwayTeam": "Southampton",
            "FTHG": 1,
            "FTAG": 1,
            "FTR": "D",
        },
    ]

    with TemporaryDirectory(prefix="tier3_parser_self_check_") as temp_dir:
        valid_csv = Path(temp_dir) / "valid_sample.csv"
        wrong_ftr_csv = Path(temp_dir) / "wrong_ftr_sample.csv"

        _write_self_check_csv(valid_csv, valid_rows)
        parsed_df = parse_football_data_csv(valid_csv, season_id)

        _require_self_check(len(parsed_df) == 4, "valid sample row count mismatch")
        _require_self_check(
            parsed_df.loc[0, "match_date"].isoformat() == "2021-08-13",
            "Date parsing failed",
        )
        _require_self_check(
            pd.notna(parsed_df.loc[0, "kickoff_time"])
            and parsed_df.loc[0, "kickoff_time"].hour == 20,
            "Time parsing failed",
        )
        _require_self_check(
            pd.isna(parsed_df.loc[1, "kickoff_time"]),
            "missing Time should produce null kickoff_time",
        )
        _require_self_check(
            parsed_df.loc[1, "home_team"] == "Man Utd",
            "team normalization failed",
        )
        _require_self_check(
            parsed_df["result"].tolist() == ["H", "H", "A", "D"],
            "result derivation failed",
        )
        validate_season(parsed_df, season_id, expected_matches=len(parsed_df))
        print("Parser self-check valid sample passed.")

        wrong_ftr_rows = [dict(row) for row in valid_rows]
        wrong_ftr_rows[0]["FTR"] = "A"
        _write_self_check_csv(wrong_ftr_csv, wrong_ftr_rows)
        try:
            parse_football_data_csv(wrong_ftr_csv, season_id)
        except ValueError as error:
            print(f"Parser self-check caught deliberately wrong FTR: {error}")
        else:
            raise AssertionError("wrong FTR sample did not fail parsing")

        invalid_validation_df = parsed_df.copy()
        invalid_validation_df.loc[0, "result"] = "A"
        try:
            validate_season(
                invalid_validation_df,
                season_id,
                expected_matches=len(invalid_validation_df),
            )
        except ValueError as error:
            print(f"Parser self-check caught validation mismatch: {error}")
        else:
            raise AssertionError("validation mismatch sample did not fail")

    print("Parser self-check complete. No database writes were performed.")
    print("Tier 2 tables were not modified.")
    print("Model artifacts were not touched.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier 3 Phase 1A local historical match ingestion."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--init-schema-only",
        action="store_true",
        help="Create and verify Tier 3 schema tables without reading CSVs.",
    )
    mode_group.add_argument(
        "--self-check-parser",
        action="store_true",
        help="Run parser and validation checks against temporary sample CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.init_schema_only:
        run_init_schema_only()
        return

    if args.self_check_parser:
        run_parser_self_check()
        return

    try:
        all_matches_df = load_all_historical_csvs(DATA_DIR, TARGET_SEASONS)
    except Exception as error:
        print(f"Historical CSV parsing failed: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    if all_matches_df.empty:
        print("No database writes were performed.")
        print("Tier 2 tables were not modified.")
        print("Model artifacts were not touched.")
        return

    engine = get_engine()
    if engine is None:
        print("Could not connect to PostgreSQL. Check local DB settings and retry.")
        raise SystemExit(1)

    imported_seasons = sorted(all_matches_df["season_id"].unique())
    try:
        create_tier3_schema(engine)
        verify_tier3_schema(engine)
        seed_seasons(engine, imported_seasons)
        seed_team_name_mapping(engine)
        for season_id in imported_seasons:
            store_historical_matches(all_matches_df, engine, season_id)
        print_ingestion_summary(engine)
    except Exception as error:
        print(f"Tier 3 historical import failed: {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    print("Tier 2 tables were not modified.")
    print("Model artifacts were not touched.")


if __name__ == "__main__":
    main()
