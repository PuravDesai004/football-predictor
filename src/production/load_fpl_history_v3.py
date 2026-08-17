from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "vaastav_fpl_history"
DB_CONNECT_TIMEOUT_SECONDS = 5

HISTORY_TABLE = "fpl_player_gameweek_history_v3"
FEATURES_TABLE = "fpl_player_features_v3"
TRAINING_RUNS_TABLE = "fpl_model_training_runs_v3"
V3_TABLES = [HISTORY_TABLE, FEATURES_TABLE, TRAINING_RUNS_TABLE]
WATCHED_TIER2_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "player_gameweek_history",
    "player_gameweek_features",
]

NORMALIZED_COLUMNS = [
    "source",
    "season",
    "gameweek",
    "player_source_id",
    "player_name",
    "player_slug",
    "team_name",
    "opponent_team_name",
    "fixture_id",
    "kickoff_time",
    "was_home",
    "position",
    "minutes",
    "total_points",
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
    "transfers_in",
    "transfers_out",
    "selected",
    "value",
    "raw_player_key",
    "source_file",
]

INTEGER_COLUMNS = [
    "gameweek",
    "minutes",
    "total_points",
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
    "starts",
    "transfers_in",
    "transfers_out",
    "selected",
]

FLOAT_COLUMNS = [
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value",
]

TEXT_COLUMNS = [
    "source",
    "season",
    "player_source_id",
    "player_name",
    "player_slug",
    "team_name",
    "opponent_team_name",
    "fixture_id",
    "position",
    "raw_player_key",
    "source_file",
]

HISTORY_DDL = f"""
CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'vaastav',
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    player_source_id TEXT NULL,
    player_name TEXT NOT NULL,
    player_slug TEXT NULL,
    team_name TEXT NULL,
    opponent_team_name TEXT NULL,
    fixture_id TEXT NULL,
    kickoff_time TIMESTAMP NULL,
    was_home BOOLEAN NULL,
    position TEXT NULL,
    minutes INTEGER NULL,
    total_points INTEGER NULL,
    goals_scored INTEGER NULL,
    assists INTEGER NULL,
    clean_sheets INTEGER NULL,
    goals_conceded INTEGER NULL,
    own_goals INTEGER NULL,
    penalties_saved INTEGER NULL,
    penalties_missed INTEGER NULL,
    yellow_cards INTEGER NULL,
    red_cards INTEGER NULL,
    saves INTEGER NULL,
    bonus INTEGER NULL,
    bps INTEGER NULL,
    influence FLOAT NULL,
    creativity FLOAT NULL,
    threat FLOAT NULL,
    ict_index FLOAT NULL,
    starts INTEGER NULL,
    expected_goals FLOAT NULL,
    expected_assists FLOAT NULL,
    expected_goal_involvements FLOAT NULL,
    expected_goals_conceded FLOAT NULL,
    transfers_in INTEGER NULL,
    transfers_out INTEGER NULL,
    selected INTEGER NULL,
    value FLOAT NULL,
    raw_player_key TEXT NULL,
    source_file TEXT NOT NULL,
    loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (gameweek > 0),
    CHECK (TRIM(player_name) <> ''),
    CHECK (TRIM(source_file) <> '')
)
"""

FEATURES_DDL = f"""
CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
    feature_id BIGSERIAL PRIMARY KEY,
    source_history_id INTEGER NULL REFERENCES {HISTORY_TABLE}(id),
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    feature_generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (gameweek > 0),
    CHECK (TRIM(player_name) <> '')
)
"""

TRAINING_RUNS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TRAINING_RUNS_TABLE} (
    training_run_id BIGSERIAL PRIMARY KEY,
    run_started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    run_finished_at TIMESTAMP NULL,
    run_status TEXT NOT NULL,
    model_name TEXT NOT NULL,
    train_seasons TEXT[] NULL,
    validation_season TEXT NULL,
    final_holdout_season TEXT NULL,
    feature_table TEXT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    metrics_json JSONB NULL,
    notes TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (run_status IN ('started', 'success', 'failed', 'skipped')),
    CHECK (row_count >= 0)
)
"""

INDEX_STATEMENTS = [
    "DROP INDEX IF EXISTS idx_fpl_pgh_v3_unique_fixture",
    "DROP INDEX IF EXISTS idx_fpl_pgh_v3_unique_no_fixture",
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_fpl_pgh_v3_unique_fixture_player_id
    ON {HISTORY_TABLE} (
        season,
        gameweek,
        player_source_id,
        fixture_id,
        source_file
    )
    WHERE fixture_id IS NOT NULL AND player_source_id IS NOT NULL
    """,
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_fpl_pgh_v3_unique_fixture_name
    ON {HISTORY_TABLE} (
        season,
        gameweek,
        player_name,
        fixture_id,
        source_file
    )
    WHERE fixture_id IS NOT NULL AND player_source_id IS NULL
    """,
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_fpl_pgh_v3_unique_no_fixture_player_id
    ON {HISTORY_TABLE} (
        season,
        gameweek,
        player_source_id,
        source_file
    )
    WHERE fixture_id IS NULL AND player_source_id IS NOT NULL
    """,
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_fpl_pgh_v3_unique_no_fixture_name
    ON {HISTORY_TABLE} (
        season,
        gameweek,
        player_name,
        source_file
    )
    WHERE fixture_id IS NULL AND player_source_id IS NULL
    """,
    f"CREATE INDEX IF NOT EXISTS idx_fpl_pgh_v3_season ON {HISTORY_TABLE} (season)",
    f"CREATE INDEX IF NOT EXISTS idx_fpl_pgh_v3_gameweek ON {HISTORY_TABLE} (gameweek)",
    f"CREATE INDEX IF NOT EXISTS idx_fpl_pgh_v3_player_name ON {HISTORY_TABLE} (player_name)",
    f"CREATE INDEX IF NOT EXISTS idx_fpl_pgh_v3_team_name ON {HISTORY_TABLE} (team_name)",
    f"CREATE INDEX IF NOT EXISTS idx_fpl_pgh_v3_position ON {HISTORY_TABLE} (position)",
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_season_gw
    ON {FEATURES_TABLE} (season, gameweek)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_player_name
    ON {FEATURES_TABLE} (player_name)
    """,
]


def get_database_url():
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        url = make_url(database_url)
        if url.host and url.host.lower() == "localhost":
            url = url.set(host="127.0.0.1")
        return url.render_as_string(hide_password=False)

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    if db_host and db_host.lower() == "localhost":
        db_host = "127.0.0.1"

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
        raise RuntimeError(f"Missing local database settings: {missing}")

    return f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"


def get_engine():
    database_url = get_database_url()
    url = make_url(database_url)
    connect_args: dict[str, Any] = {"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS}
    if (
        url.host
        and url.host not in {"127.0.0.1", "localhost"}
        and "sslmode" not in database_url.lower()
    ):
        connect_args["sslmode"] = "require"

    engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    with engine.connect():
        pass

    print(f"Connected to PostgreSQL database: {url.database or 'unknown'}")
    return engine


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


def _table_columns(engine, table_name: str) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = CURRENT_SCHEMA()
                  AND table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"table_name": table_name},
        ).fetchall()
    return [row[0] for row in rows]


def init_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(HISTORY_DDL))
        conn.execute(text(FEATURES_DDL))
        conn.execute(text(TRAINING_RUNS_DDL))
        for statement in INDEX_STATEMENTS:
            conn.execute(text(statement))

    missing_tables = [table for table in V3_TABLES if not _table_exists(engine, table)]
    if missing_tables:
        raise RuntimeError(f"Missing FPL v3 table(s): {missing_tables}")

    history_columns = set(_table_columns(engine, HISTORY_TABLE))
    missing_columns = [
        column for column in [*NORMALIZED_COLUMNS, "id", "loaded_at"]
        if column not in history_columns
    ]
    if missing_columns:
        raise RuntimeError(f"{HISTORY_TABLE} missing column(s): {missing_columns}")

    print("PASS: FPL v3 schema created/verified.")


def discover_vaastav_gw_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    candidates = set()
    for pattern in [
        "data/*/gws/gw*.csv",
        "*/gws/gw*.csv",
        "**/gws/gw*.csv",
    ]:
        candidates.update(path for path in base_dir.glob(pattern) if path.is_file())

    return sorted(candidates, key=lambda path: (parse_season_from_path(path), parse_gameweek_from_filename(path), str(path)))


def parse_season_from_path(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"\d{4}-\d{2}", part):
            return part
    raise ValueError(f"Could not parse season from path: {path}")


def parse_gameweek_from_filename(path: Path) -> int:
    match = re.fullmatch(r"gw(\d+)\.csv", path.name.lower())
    if not match:
        raise ValueError(f"Could not parse gameweek from filename: {path.name}")

    gameweek = int(match.group(1))
    if gameweek <= 0:
        raise ValueError(f"Invalid gameweek parsed from filename: {path.name}")
    return gameweek


def _canonical_column_name(column: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(column).strip().lower()).strip("_")


def _canonicalize_dataframe(df: pandas.DataFrame) -> pandas.DataFrame:
    renamed = df.copy()
    seen: set[str] = set()
    columns = []
    for column in renamed.columns:
        canonical = _canonical_column_name(column)
        if canonical in seen:
            suffix = 2
            while f"{canonical}_{suffix}" in seen:
                suffix += 1
            canonical = f"{canonical}_{suffix}"
        seen.add(canonical)
        columns.append(canonical)
    renamed.columns = columns
    return renamed


def _series_from_candidates(
    df: pandas.DataFrame,
    candidates: list[str],
    default: Any = None,
) -> pandas.Series:
    for candidate in candidates:
        canonical = _canonical_column_name(candidate)
        if canonical in df.columns:
            return df[canonical]
    return pandas.Series([default] * len(df), index=df.index)


def _text_series(series: pandas.Series) -> pandas.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pandas.NA, "nan": pandas.NA, "None": pandas.NA})


def _integer_series(series: pandas.Series) -> pandas.Series:
    numeric = pandas.to_numeric(series, errors="coerce")
    return numeric.apply(lambda value: int(value) if pandas.notna(value) else None)


def _float_series(series: pandas.Series) -> pandas.Series:
    numeric = pandas.to_numeric(series, errors="coerce")
    return numeric.apply(lambda value: float(value) if pandas.notna(value) else None)


def _boolean_value(value: Any):
    if pandas.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text_value = str(value).strip().lower()
    if text_value in {"true", "t", "yes", "y", "1"}:
        return True
    if text_value in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _relative_source_file(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path_text.replace("\\", "/")


def normalize_vaastav_gameweek_df(
    df: pandas.DataFrame,
    season: str,
    gameweek: int,
    source_file: str,
) -> pandas.DataFrame:
    if df.empty:
        raise ValueError(f"{source_file} has zero rows")

    source_df = _canonicalize_dataframe(df)
    normalized = pandas.DataFrame(index=source_df.index)

    if "name" in source_df.columns or "player_name" in source_df.columns:
        player_name = _series_from_candidates(source_df, ["player_name", "name", "web_name"])
    elif {"first_name", "second_name"}.issubset(source_df.columns):
        player_name = (
            source_df["first_name"].astype("string").fillna("")
            + " "
            + source_df["second_name"].astype("string").fillna("")
        )
    else:
        player_name = pandas.Series([pandas.NA] * len(source_df), index=source_df.index)

    normalized["source"] = "vaastav"
    normalized["season"] = season
    normalized["gameweek"] = gameweek
    normalized["player_source_id"] = _series_from_candidates(
        source_df,
        ["element", "player_id", "id"],
    )
    normalized["player_name"] = player_name
    normalized["player_slug"] = _series_from_candidates(
        source_df,
        ["player_slug", "slug", "cleaned_name"],
    )
    normalized["team_name"] = _series_from_candidates(source_df, ["team_name", "team"])
    normalized["opponent_team_name"] = _series_from_candidates(
        source_df,
        ["opponent_team_name", "opponent_team"],
    )
    normalized["fixture_id"] = _series_from_candidates(source_df, ["fixture_id", "fixture"])
    normalized["kickoff_time"] = pandas.to_datetime(
        _series_from_candidates(source_df, ["kickoff_time", "kickoff"]),
        errors="coerce",
        utc=False,
    )
    normalized["was_home"] = _series_from_candidates(source_df, ["was_home"]).apply(_boolean_value)
    normalized["position"] = _series_from_candidates(
        source_df,
        ["position", "element_type", "element_type_name"],
    )

    for column in INTEGER_COLUMNS:
        if column == "gameweek":
            continue
        normalized[column] = _integer_series(_series_from_candidates(source_df, [column]))

    for column in FLOAT_COLUMNS:
        normalized[column] = _float_series(_series_from_candidates(source_df, [column]))

    normalized["raw_player_key"] = _series_from_candidates(
        source_df,
        ["raw_player_key", "element", "player_id", "id", "name"],
    )
    normalized["source_file"] = _relative_source_file(source_file)

    for column in TEXT_COLUMNS:
        normalized[column] = _text_series(normalized[column])

    normalized["gameweek"] = _integer_series(normalized["gameweek"])
    normalized["kickoff_time"] = pandas.to_datetime(normalized["kickoff_time"], errors="coerce")

    return normalized[NORMALIZED_COLUMNS]


def validate_normalized_df(df: pandas.DataFrame) -> None:
    missing_columns = [column for column in NORMALIZED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Normalized dataframe missing column(s): {missing_columns}")

    if df.empty:
        raise ValueError("Normalized dataframe has zero rows")

    errors = []
    missing_player = df["player_name"].isna() | (df["player_name"].astype(str).str.strip() == "")
    if missing_player.any():
        errors.append(f"missing player_name rows: {int(missing_player.sum())}")

    gameweek_values = pandas.to_numeric(df["gameweek"], errors="coerce")
    invalid_gameweek = gameweek_values.isna() | (gameweek_values <= 0)
    if invalid_gameweek.any():
        errors.append(f"invalid gameweek rows: {int(invalid_gameweek.sum())}")

    missing_source_file = df["source_file"].isna() | (
        df["source_file"].astype(str).str.strip() == ""
    )
    if missing_source_file.any():
        errors.append(f"missing source_file rows: {int(missing_source_file.sum())}")

    fixture_text = df["fixture_id"].fillna("").astype(str).str.strip()
    player_source_text = df["player_source_id"].fillna("").astype(str).str.strip()
    with_fixture_and_id = df[(fixture_text != "") & (player_source_text != "")]
    with_fixture_without_id = df[(fixture_text != "") & (player_source_text == "")]
    without_fixture_with_id = df[(fixture_text == "") & (player_source_text != "")]
    without_fixture_without_id = df[(fixture_text == "") & (player_source_text == "")]

    duplicate_frames = []
    if not with_fixture_and_id.empty:
        duplicates = with_fixture_and_id[
            with_fixture_and_id.duplicated(
                ["season", "gameweek", "player_source_id", "fixture_id", "source_file"],
                keep=False,
            )
        ]
        if not duplicates.empty:
            duplicate_frames.append(duplicates)

    if not with_fixture_without_id.empty:
        duplicates = with_fixture_without_id[
            with_fixture_without_id.duplicated(
                ["season", "gameweek", "player_name", "fixture_id", "source_file"],
                keep=False,
            )
        ]
        if not duplicates.empty:
            duplicate_frames.append(duplicates)

    if not without_fixture_with_id.empty:
        duplicates = without_fixture_with_id[
            without_fixture_with_id.duplicated(
                ["season", "gameweek", "player_source_id", "source_file"],
                keep=False,
            )
        ]
        if not duplicates.empty:
            duplicate_frames.append(duplicates)

    if not without_fixture_without_id.empty:
        duplicates = without_fixture_without_id[
            without_fixture_without_id.duplicated(
                ["season", "gameweek", "player_name", "source_file"],
                keep=False,
            )
        ]
        if not duplicates.empty:
            duplicate_frames.append(duplicates)

    if duplicate_frames:
        duplicate_df = pandas.concat(duplicate_frames, ignore_index=True)
        examples = duplicate_df[
            ["season", "gameweek", "player_name", "fixture_id", "source_file"]
        ].head(5).to_dict("records")
        errors.append(
            f"duplicate normalized rows: {len(duplicate_df)}; examples: {examples}"
        )

    if errors:
        raise ValueError("FPL v3 normalized validation failed: " + "; ".join(errors))


def _records_for_insert(df: pandas.DataFrame) -> list[dict]:
    insert_df = df.copy()
    insert_df["kickoff_time"] = insert_df["kickoff_time"].apply(
        lambda value: value.to_pydatetime()
        if isinstance(value, pandas.Timestamp) and pandas.notna(value)
        else None
    )
    insert_df = insert_df.astype(object).where(pandas.notna(insert_df), None)
    return insert_df.to_dict("records")


def _read_vaastav_csv(path: Path) -> pandas.DataFrame:
    errors = []
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pandas.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {path}; attempts: {errors}",
    )


def load_gameweek_file(engine, path: Path) -> int:
    season = parse_season_from_path(path)
    gameweek = parse_gameweek_from_filename(path)
    source_file = path.resolve().relative_to(PROJECT_ROOT).as_posix()

    raw_df = _read_vaastav_csv(path)
    if raw_df.empty:
        with engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {HISTORY_TABLE} WHERE source_file = :source_file"),
                {"source_file": source_file},
            )
        print(f"SKIP: {source_file} has zero rows.")
        return 0

    normalized_df = normalize_vaastav_gameweek_df(
        raw_df,
        season=season,
        gameweek=gameweek,
        source_file=source_file,
    )
    before_dedup = len(normalized_df)
    normalized_df = normalized_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before_dedup - len(normalized_df)
    if removed_duplicates:
        print(f"Removed {removed_duplicates} exact duplicate row(s) from {source_file}")

    validate_normalized_df(normalized_df)

    records = _records_for_insert(normalized_df)
    columns = NORMALIZED_COLUMNS
    column_sql = ", ".join(columns)
    value_sql = ", ".join(f":{column}" for column in columns)
    insert_sql = text(
        f"""
        INSERT INTO {HISTORY_TABLE} ({column_sql})
        VALUES ({value_sql})
        """
    )

    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {HISTORY_TABLE} WHERE source_file = :source_file"),
            {"source_file": source_file},
        )
        if records:
            conn.execute(insert_sql, records)

    print(f"Loaded {len(records)} row(s) from {source_file}")
    return len(records)


def _row_count(engine, table_name: str):
    if not _table_exists(engine, table_name):
        return "MISSING"
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def capture_watched_tier2_counts(engine) -> dict[str, int | str]:
    return {table: _row_count(engine, table) for table in WATCHED_TIER2_TABLES}


def assert_tier2_counts_unchanged(before, after) -> None:
    changed = {
        table: (before.get(table), after.get(table))
        for table in WATCHED_TIER2_TABLES
        if before.get(table) != after.get(table)
    }
    if changed:
        raise RuntimeError(f"Protected Tier 2 table count(s) changed: {changed}")
    print("PASS: protected Tier 2 table row counts unchanged.")


def print_row_counts(engine) -> None:
    print("FPL v3 table row counts:")
    for table in V3_TABLES:
        print(f"- {table}: {_row_count(engine, table)}")

    print("Protected Tier 2 table row counts:")
    for table, count in capture_watched_tier2_counts(engine).items():
        print(f"- {table}: {count}")


def self_check_parser() -> None:
    sample_source = "data/vaastav_fpl_history/data/2024-25/gws/gw1.csv"
    season = parse_season_from_path(Path(sample_source))
    gameweek = parse_gameweek_from_filename(Path(sample_source))
    if season != "2024-25" or gameweek != 1:
        raise AssertionError("Season/gameweek path parsing failed")

    sample_df = pandas.DataFrame(
        [
            {
                "name": "Erling Haaland",
                "element": "355",
                "team": "Man City",
                "opponent_team": "Chelsea",
                "fixture": "1",
                "kickoff_time": "2024-08-18T15:30:00Z",
                "was_home": "False",
                "position": "FWD",
                "minutes": "90",
                "total_points": "13",
                "goals_scored": "2",
                "assists": "0",
                "clean_sheets": "0",
                "goals_conceded": "0",
                "own_goals": "0",
                "penalties_saved": "0",
                "penalties_missed": "0",
                "yellow_cards": "0",
                "red_cards": "0",
                "saves": "0",
                "bonus": "3",
                "bps": "42",
                "influence": "59.4",
                "creativity": "12.1",
                "threat": "88.0",
                "ict_index": "15.9",
                "starts": "1",
                "expected_goals": "1.21",
                "expected_assists": "0.02",
                "expected_goal_involvements": "1.23",
                "expected_goals_conceded": "0.65",
                "transfers_in": "125000",
                "transfers_out": "12000",
                "selected": "5200000",
                "value": "150",
            },
            {
                "name": "Bukayo Saka",
                "element": "19",
                "team": "Arsenal",
                "opponent_team": "Wolves",
                "fixture": "2",
                "kickoff_time": "2024-08-17T14:00:00Z",
                "was_home": "True",
                "position": "MID",
                "minutes": "90",
                "total_points": "10",
                "bps": "36",
                "influence": "41.2",
                "creativity": "33.0",
                "threat": "55.5",
                "ict_index": "12.9",
                "value": "100",
            },
        ]
    )

    normalized = normalize_vaastav_gameweek_df(
        sample_df,
        season=season,
        gameweek=gameweek,
        source_file=sample_source,
    )
    validate_normalized_df(normalized)

    if normalized.loc[0, "minutes"] != 90 or bool(normalized.loc[0, "was_home"]):
        raise AssertionError("Type coercion self-check failed")
    if not pandas.api.types.is_datetime64_any_dtype(normalized["kickoff_time"]):
        raise AssertionError("kickoff_time parsing self-check failed")

    duplicate_df = pandas.concat([normalized, normalized.iloc[[0]]], ignore_index=True)
    try:
        validate_normalized_df(duplicate_df)
    except ValueError as error:
        if "duplicate normalized rows" not in str(error):
            raise
    else:
        raise AssertionError("Duplicate detection self-check failed")

    missing_name_df = normalized.copy()
    missing_name_df.loc[0, "player_name"] = pandas.NA
    try:
        validate_normalized_df(missing_name_df)
    except ValueError as error:
        if "missing player_name" not in str(error):
            raise
    else:
        raise AssertionError("Required player_name validation self-check failed")

    print("PASS: parser self-check completed without DB writes.")


def _print_count_comparison(before, after) -> None:
    print("Protected Tier 2 before/after counts:")
    for table in WATCHED_TIER2_TABLES:
        print(f"- {table}: {before.get(table)} -> {after.get(table)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load local Vaastav FPL gameweek history into isolated Tier 3 v3 tables."
    )
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--self-check-parser", action="store_true")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    if args.self_check_parser:
        self_check_parser()
        return

    base_dir = args.base_dir
    engine = get_engine()

    if args.init_schema_only:
        init_schema(engine)
        print_row_counts(engine)
        return

    before_counts = capture_watched_tier2_counts(engine)

    if not base_dir.exists():
        print(f"WARNING: Vaastav history folder not found: {base_dir}")
        print("SKIPPED_MISSING_VAASTAV_HISTORY_FOLDER")
        after_counts = capture_watched_tier2_counts(engine)
        _print_count_comparison(before_counts, after_counts)
        assert_tier2_counts_unchanged(before_counts, after_counts)
        print_row_counts(engine)
        return

    files = discover_vaastav_gw_files(base_dir)
    if not files:
        print(f"WARNING: No Vaastav gameweek CSV files found under: {base_dir}")
        print("SKIPPED_NO_VAASTAV_GAMEWEEK_FILES")
        after_counts = capture_watched_tier2_counts(engine)
        _print_count_comparison(before_counts, after_counts)
        assert_tier2_counts_unchanged(before_counts, after_counts)
        print_row_counts(engine)
        return

    init_schema(engine)
    total_rows = 0
    print(f"Discovered {len(files)} Vaastav gameweek CSV file(s).")
    for file_path in files:
        total_rows += load_gameweek_file(engine, file_path)

    after_counts = capture_watched_tier2_counts(engine)
    _print_count_comparison(before_counts, after_counts)
    assert_tier2_counts_unchanged(before_counts, after_counts)
    print_row_counts(engine)
    print(f"Loaded {total_rows} total FPL v3 history row(s).")


if __name__ == "__main__":
    main()
