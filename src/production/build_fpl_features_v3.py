"""Phase 2A: Build chronological pre-gameweek FPL player features.

This script creates the Tier 3 FPL feature foundation from local Vaastav
history. It is deliberately feature-only: no model training, no evaluation,
no tuning, and no Streamlit/app changes.

Leakage rule:
For each player-season-gameweek row, every input feature uses only rows for
the same fpl_code that are chronologically before the target row. The current
gameweek's total_points is copied only into target_total_points.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_CONNECT_TIMEOUT_SECONDS = 5

HISTORY_TABLE = "fpl_player_gameweek_history_v3"
IDENTITY_TABLE = "fpl_player_identity_map_v3"
FEATURE_TABLE = "fpl_player_features_v3"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "fpl_v3_feature_readiness_audit.md"

FINAL_HOLDOUT_SEASON = "2025-26"

PROTECTED_TIER2_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "player_gameweek_history",
    "player_gameweek_features",
]

REQUIRED_HISTORY_COLUMNS = [
    "id",
    "season",
    "gameweek",
    "player_source_id",
    "player_name",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "bonus",
    "starts",
    "expected_goals",
    "expected_assists",
]

REQUIRED_IDENTITY_COLUMNS = [
    "canonical_player_key",
    "fpl_code",
    "season",
    "player_source_id",
    "mapping_method",
]

MODEL_FEATURE_COLUMNS = [
    "prior_points_last1",
    "prior_points_last3",
    "prior_points_last5",
    "prior_points_last10",
    "prior_points_season",
    "prior_minutes_last3",
    "prior_minutes_last5",
    "prior_minutes_last10",
    "prior_appearances_last5",
    "prior_starts_last5",
    "prior_goals_last5",
    "prior_assists_last5",
    "prior_bonus_last5",
    "prior_clean_sheets_last5",
    "prior_saves_last5",
    "prior_xg_last5",
    "prior_xa_last5",
    "prior_points_per_90",
    "prior_minutes_total",
    "prior_gameweeks_played",
]

FEATURE_COLUMNS = [
    "canonical_player_key",
    "fpl_code",
    "season",
    "gameweek",
    "player_name",
    "player_source_id",
    "target_total_points",
    *MODEL_FEATURE_COLUMNS,
    "feature_history_start_season",
    "feature_history_start_gameweek",
    "feature_history_end_season",
    "feature_history_end_gameweek",
    "prior_history_row_count",
    "source_history_row_id",
    "source_history_row_count",
]

NUMERIC_SOURCE_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "bonus",
    "starts",
    "expected_goals",
    "expected_assists",
]

FEATURE_DDL = f"""
CREATE TABLE {FEATURE_TABLE} (
    feature_id BIGSERIAL PRIMARY KEY,
    canonical_player_key TEXT NOT NULL,
    fpl_code INTEGER NOT NULL,
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    player_source_id TEXT NOT NULL,
    target_total_points INTEGER NULL,
    prior_points_last1 FLOAT NULL,
    prior_points_last3 FLOAT NULL,
    prior_points_last5 FLOAT NULL,
    prior_points_last10 FLOAT NULL,
    prior_points_season FLOAT NULL,
    prior_minutes_last3 FLOAT NULL,
    prior_minutes_last5 FLOAT NULL,
    prior_minutes_last10 FLOAT NULL,
    prior_appearances_last5 FLOAT NULL,
    prior_starts_last5 FLOAT NULL,
    prior_goals_last5 FLOAT NULL,
    prior_assists_last5 FLOAT NULL,
    prior_bonus_last5 FLOAT NULL,
    prior_clean_sheets_last5 FLOAT NULL,
    prior_saves_last5 FLOAT NULL,
    prior_xg_last5 FLOAT NULL,
    prior_xa_last5 FLOAT NULL,
    prior_points_per_90 FLOAT NULL,
    prior_minutes_total FLOAT NULL,
    prior_gameweeks_played FLOAT NULL,
    feature_history_start_season TEXT NULL,
    feature_history_start_gameweek INTEGER NULL,
    feature_history_end_season TEXT NULL,
    feature_history_end_gameweek INTEGER NULL,
    prior_history_row_count INTEGER NOT NULL DEFAULT 0,
    source_history_row_id INTEGER NOT NULL REFERENCES {HISTORY_TABLE}(id),
    source_history_row_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (canonical_player_key, season, gameweek),
    CHECK (gameweek > 0),
    CHECK (feature_history_start_gameweek IS NULL OR feature_history_start_gameweek > 0),
    CHECK (feature_history_end_gameweek IS NULL OR feature_history_end_gameweek > 0),
    CHECK (prior_history_row_count >= 0),
    CHECK (source_history_row_count > 0),
    CHECK (TRIM(canonical_player_key) <> ''),
    CHECK (TRIM(season) <> ''),
    CHECK (TRIM(player_name) <> ''),
    CHECK (TRIM(player_source_id) <> '')
)
"""

INDEX_STATEMENTS = [
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_season_gw
    ON {FEATURE_TABLE} (season, gameweek)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_player_key
    ON {FEATURE_TABLE} (canonical_player_key)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_fpl_code
    ON {FEATURE_TABLE} (fpl_code)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_player_features_v3_source_history
    ON {FEATURE_TABLE} (source_history_row_id)
    """,
]


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
    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = CURRENT_SCHEMA()
              AND table_name = :table_name
        )
        """
    )
    with engine.connect() as conn:
        return bool(conn.execute(query, {"table_name": table_name}).scalar_one())


def _row_count(engine, table_name: str) -> int:
    if not _table_exists(engine, table_name):
        return -1
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _table_columns(engine, table_name: str) -> set[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = CURRENT_SCHEMA()
          AND table_name = :table_name
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table_name": table_name}).fetchall()
    return {row[0] for row in rows}


def _capture_protected_counts(engine) -> dict[str, int]:
    return {table: _row_count(engine, table) for table in PROTECTED_TIER2_TABLES}


def init_schema(engine) -> None:
    """Create or replace only fpl_player_features_v3."""
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {FEATURE_TABLE} CASCADE"))
        conn.execute(text(FEATURE_DDL))
        for stmt in INDEX_STATEMENTS:
            conn.execute(text(stmt))
    if not _table_exists(engine, FEATURE_TABLE):
        raise RuntimeError(f"Failed to create {FEATURE_TABLE}")
    print(f"PASS: created {FEATURE_TABLE}")


def load_history(engine) -> pd.DataFrame:
    columns = _table_columns(engine, HISTORY_TABLE)
    missing = sorted(set(REQUIRED_HISTORY_COLUMNS) - columns)
    if missing:
        raise RuntimeError(f"Missing required history columns: {missing}")

    query = text(f"""
        SELECT
            id AS source_history_row_id,
            season,
            gameweek,
            player_source_id,
            player_name,
            total_points,
            minutes,
            goals_scored,
            assists,
            clean_sheets,
            saves,
            bonus,
            starts,
            expected_goals,
            expected_assists
        FROM {HISTORY_TABLE}
        ORDER BY season, gameweek, player_source_id, id
    """)
    with engine.connect() as conn:
        history = pd.read_sql(query, conn)

    print(f"Loaded source history rows: {len(history)}")
    return history


def load_identity_map(engine) -> pd.DataFrame:
    columns = _table_columns(engine, IDENTITY_TABLE)
    missing = sorted(set(REQUIRED_IDENTITY_COLUMNS) - columns)
    if missing:
        raise RuntimeError(f"Missing required identity-map columns: {missing}")

    duplicate_query = text(f"""
        SELECT COUNT(*) FROM (
            SELECT season, player_source_id
            FROM {IDENTITY_TABLE}
            GROUP BY season, player_source_id
            HAVING COUNT(*) > 1
        ) dup
    """)
    with engine.connect() as conn:
        duplicate_count = int(conn.execute(duplicate_query).scalar_one())
    if duplicate_count:
        raise RuntimeError(
            f"Identity map has {duplicate_count} duplicate season/player_source_id keys"
        )

    query = text(f"""
        SELECT
            canonical_player_key,
            fpl_code,
            season,
            player_source_id,
            mapping_method
        FROM {IDENTITY_TABLE}
        WHERE mapping_method = 'element_id_exact'
    """)
    with engine.connect() as conn:
        identity = pd.read_sql(query, conn)

    identity["player_source_id"] = identity["player_source_id"].astype(str).str.strip()
    print(f"Loaded identity-map rows: {len(identity)}")
    return identity


def _season_order(season: str) -> int:
    try:
        return int(str(season)[:4])
    except ValueError as exc:
        raise ValueError(f"Invalid season label: {season}") from exc


def _rolling_prior_mean(group: pd.DataFrame, column: str, window: int) -> pd.Series:
    return group[column].shift(1).rolling(window=window, min_periods=1).mean()


def _rolling_prior_sum(group: pd.DataFrame, column: str, window: int) -> pd.Series:
    return group[column].shift(1).rolling(window=window, min_periods=1).sum()


def _sum_preserving_null(series: pd.Series) -> float | None:
    if series.notna().any():
        return float(series.fillna(0).sum())
    return None


def _aggregate_player_gameweeks(merged: pd.DataFrame) -> pd.DataFrame:
    """Collapse fixture-level rows to one player-gameweek row.

    Vaastav historical gameweek files can contain multiple fixture rows for
    a player in double gameweeks. The prediction target is player-gameweek
    total points, so features are built from aggregated player-gameweek rows.
    """
    group_cols = ["canonical_player_key", "fpl_code", "season", "gameweek"]
    aggregations: dict[str, Any] = {
        "source_history_row_id": ("source_history_row_id", "min"),
        "player_source_id": ("player_source_id", "first"),
        "player_name": ("player_name", "first"),
        "season_order": ("season_order", "first"),
        "total_points": ("total_points", _sum_preserving_null),
        "minutes": ("minutes", _sum_preserving_null),
        "goals_scored": ("goals_scored", _sum_preserving_null),
        "assists": ("assists", _sum_preserving_null),
        "clean_sheets": ("clean_sheets", _sum_preserving_null),
        "saves": ("saves", _sum_preserving_null),
        "bonus": ("bonus", _sum_preserving_null),
        "starts": ("starts", _sum_preserving_null),
        "expected_goals": ("expected_goals", _sum_preserving_null),
        "expected_assists": ("expected_assists", _sum_preserving_null),
        "source_history_row_count": ("source_history_row_id", "count"),
    }

    aggregated = (
        merged.groupby(group_cols, as_index=False, sort=False)
        .agg(**aggregations)
        .sort_values(["season_order", "gameweek", "source_history_row_id"])
        .reset_index(drop=True)
    )
    return aggregated


def calculate_rolling_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prior-only features for one fpl_code group."""
    player_df = player_df.sort_values(
        ["season_order", "gameweek", "source_history_row_id"]
    ).copy()

    points = player_df["total_points"].astype(float)
    minutes = player_df["minutes"].astype(float)
    appearances = (minutes.fillna(0) > 0).astype(float)

    player_df["prior_points_last1"] = points.shift(1)
    player_df["prior_points_last3"] = _rolling_prior_mean(player_df, "total_points", 3)
    player_df["prior_points_last5"] = _rolling_prior_mean(player_df, "total_points", 5)
    player_df["prior_points_last10"] = _rolling_prior_mean(player_df, "total_points", 10)
    player_df["prior_minutes_last3"] = _rolling_prior_mean(player_df, "minutes", 3)
    player_df["prior_minutes_last5"] = _rolling_prior_mean(player_df, "minutes", 5)
    player_df["prior_minutes_last10"] = _rolling_prior_mean(player_df, "minutes", 10)

    player_df["_appearance"] = appearances
    player_df["_starts_filled"] = player_df["starts"].fillna(0).astype(float)
    player_df["_goals_filled"] = player_df["goals_scored"].fillna(0).astype(float)
    player_df["_assists_filled"] = player_df["assists"].fillna(0).astype(float)
    player_df["_bonus_filled"] = player_df["bonus"].fillna(0).astype(float)
    player_df["_clean_sheets_filled"] = player_df["clean_sheets"].fillna(0).astype(float)
    player_df["_saves_filled"] = player_df["saves"].fillna(0).astype(float)
    player_df["_xg_filled"] = player_df["expected_goals"].fillna(0).astype(float)
    player_df["_xa_filled"] = player_df["expected_assists"].fillna(0).astype(float)
    player_df["_points_filled"] = points.fillna(0)
    player_df["_minutes_filled"] = minutes.fillna(0)

    player_df["prior_appearances_last5"] = _rolling_prior_sum(player_df, "_appearance", 5)
    player_df["prior_starts_last5"] = _rolling_prior_sum(player_df, "_starts_filled", 5)
    player_df["prior_goals_last5"] = _rolling_prior_sum(player_df, "_goals_filled", 5)
    player_df["prior_assists_last5"] = _rolling_prior_sum(player_df, "_assists_filled", 5)
    player_df["prior_bonus_last5"] = _rolling_prior_sum(player_df, "_bonus_filled", 5)
    player_df["prior_clean_sheets_last5"] = _rolling_prior_sum(
        player_df, "_clean_sheets_filled", 5
    )
    player_df["prior_saves_last5"] = _rolling_prior_sum(player_df, "_saves_filled", 5)
    player_df["prior_xg_last5"] = _rolling_prior_sum(player_df, "_xg_filled", 5)
    player_df["prior_xa_last5"] = _rolling_prior_sum(player_df, "_xa_filled", 5)

    player_df["prior_minutes_total"] = (
        player_df["_minutes_filled"].cumsum() - player_df["_minutes_filled"]
    )
    player_df["prior_gameweeks_played"] = (
        player_df["_appearance"].cumsum() - player_df["_appearance"]
    )
    prior_points_total = player_df["_points_filled"].cumsum() - player_df["_points_filled"]
    player_df["prior_points_per_90"] = np.where(
        player_df["prior_minutes_total"] > 0,
        prior_points_total * 90.0 / player_df["prior_minutes_total"],
        np.nan,
    )

    season_group = player_df.groupby("season", sort=False)
    player_df["prior_points_season"] = season_group["_points_filled"].cumsum() - player_df[
        "_points_filled"
    ]

    player_df["prior_history_row_count"] = np.arange(len(player_df), dtype=int)
    player_df["feature_history_start_season"] = player_df["season"].iloc[0]
    player_df["feature_history_start_gameweek"] = int(player_df["gameweek"].iloc[0])
    no_prior_mask = player_df["prior_history_row_count"] == 0
    player_df.loc[no_prior_mask, "feature_history_start_season"] = None
    player_df.loc[no_prior_mask, "feature_history_start_gameweek"] = np.nan
    player_df["feature_history_end_season"] = player_df["season"].shift(1)
    player_df["feature_history_end_gameweek"] = player_df["gameweek"].shift(1)

    drop_cols = [col for col in player_df.columns if col.startswith("_")]
    return player_df.drop(columns=drop_cols)


def build_prior_only_features(history: pd.DataFrame, identity_map: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    history["player_source_id"] = history["player_source_id"].astype(str).str.strip()
    history["season_order"] = history["season"].apply(_season_order)

    merged = history.merge(
        identity_map,
        on=["season", "player_source_id"],
        how="left",
        validate="many_to_one",
    )
    merged["has_identity_mapping"] = merged["canonical_player_key"].notna()

    for column in NUMERIC_SOURCE_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    unmapped_count = int((~merged["has_identity_mapping"]).sum())
    if unmapped_count:
        raise RuntimeError(
            f"{unmapped_count} history rows have no exact fpl_code identity mapping"
        )

    merged = _aggregate_player_gameweeks(merged)

    duplicate_keys = merged.duplicated(
        subset=["canonical_player_key", "season", "gameweek"], keep=False
    )
    if duplicate_keys.any():
        examples = merged.loc[
            duplicate_keys,
            ["canonical_player_key", "season", "gameweek", "player_name"],
        ].head(10)
        raise RuntimeError(
            "Duplicate feature keys found before write:\n"
            + examples.to_string(index=False)
        )

    feature_parts = []
    for _, group in merged.groupby("fpl_code", sort=False):
        feature_parts.append(calculate_rolling_features(group))

    features = pd.concat(feature_parts, ignore_index=True)
    features["target_total_points"] = features["total_points"]
    features = features.sort_values(
        ["season_order", "gameweek", "canonical_player_key"]
    ).reset_index(drop=True)

    features = features[FEATURE_COLUMNS].copy()

    integer_columns = [
        "fpl_code",
        "gameweek",
        "target_total_points",
        "prior_history_row_count",
        "source_history_row_id",
        "feature_history_start_gameweek",
        "feature_history_end_gameweek",
    ]
    for column in integer_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("Int64")

    return features


def validate_no_future_rows(features: pd.DataFrame) -> dict[str, Any]:
    with_prior = features[features["prior_history_row_count"] > 0].copy()
    if with_prior.empty:
        return {"future_leak_rows": 0, "checked_rows_with_prior_history": 0}

    current_order = with_prior["season"].apply(_season_order)
    end_order = with_prior["feature_history_end_season"].apply(_season_order)
    same_season_ok = (
        (end_order == current_order)
        & (with_prior["feature_history_end_gameweek"].astype(int) < with_prior["gameweek"].astype(int))
    )
    prior_season_ok = end_order < current_order
    leak_mask = ~(same_season_ok | prior_season_ok)
    leak_count = int(leak_mask.sum())
    if leak_count:
        examples = with_prior.loc[
            leak_mask,
            [
                "canonical_player_key",
                "season",
                "gameweek",
                "feature_history_end_season",
                "feature_history_end_gameweek",
            ],
        ].head(10)
        raise RuntimeError(
            "Future/current gameweek leakage detected:\n"
            + examples.to_string(index=False)
        )

    return {
        "future_leak_rows": 0,
        "checked_rows_with_prior_history": int(len(with_prior)),
    }


def validate_target_not_in_features() -> dict[str, Any]:
    target_in_features = "target_total_points" in MODEL_FEATURE_COLUMNS
    if target_in_features:
        raise RuntimeError("target_total_points is present in MODEL_FEATURE_COLUMNS")
    return {
        "target_total_points_in_model_features": target_in_features,
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
    }


def validate_feature_coverage(features: pd.DataFrame) -> dict[str, Any]:
    required_cols = [
        "canonical_player_key",
        "fpl_code",
        "season",
        "gameweek",
        "player_name",
        "player_source_id",
        "source_history_row_id",
    ]
    required_nulls = {
        col: int(features[col].isna().sum())
        for col in required_cols
    }
    target_null_count = int(features["target_total_points"].isna().sum())
    no_prior_count = int((features["prior_history_row_count"] == 0).sum())
    duplicate_count = int(
        features.duplicated(
            subset=["canonical_player_key", "season", "gameweek"], keep=False
        ).sum()
    )

    if any(required_nulls.values()):
        raise RuntimeError(f"Required feature columns contain nulls: {required_nulls}")
    if duplicate_count:
        raise RuntimeError(f"Duplicate canonical_player_key/season/gameweek rows: {duplicate_count}")

    feature_nulls = (
        features[MODEL_FEATURE_COLUMNS]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    feature_nulls.columns = ["feature", "null_count"]
    feature_nulls["null_rate"] = feature_nulls["null_count"] / len(features)

    season_counts = (
        features.groupby("season")
        .agg(
            rows=("source_history_row_id", "count"),
            distinct_fpl_code=("fpl_code", "nunique"),
            no_prior_history_rows=("prior_history_row_count", lambda s: int((s == 0).sum())),
            target_null_rows=("target_total_points", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    gameweek_counts = (
        features.groupby(["season", "gameweek"])
        .agg(rows=("source_history_row_id", "count"))
        .reset_index()
        .sort_values(["season", "gameweek"])
    )

    return {
        "required_nulls": required_nulls,
        "target_null_count": target_null_count,
        "feature_null_summary": feature_nulls,
        "season_counts": season_counts,
        "gameweek_counts": gameweek_counts,
        "duplicate_feature_key_count": duplicate_count,
        "rows_with_no_prior_history": no_prior_count,
        "rows_with_no_identity_mapping": 0,
        "distinct_fpl_code_count": int(features["fpl_code"].nunique()),
        "holdout_2025_26_rows": int((features["season"] == FINAL_HOLDOUT_SEASON).sum()),
    }


def _serialize_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if str(result[column].dtype) == "Int64":
            result[column] = result[column].astype(object).where(result[column].notna(), None)
        elif pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].replace({np.nan: None})
        else:
            result[column] = result[column].where(result[column].notna(), None)
    return result


def write_features(engine, features: pd.DataFrame) -> int:
    write_df = _serialize_for_sql(features)
    with engine.begin() as conn:
        write_df.to_sql(
            FEATURE_TABLE,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
    count = _row_count(engine, FEATURE_TABLE)
    print(f"Wrote {count} rows to {FEATURE_TABLE}")
    return count


def _post_write_checks(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        total_rows = int(conn.execute(text(f"SELECT COUNT(*) FROM {FEATURE_TABLE}")).scalar_one())
        duplicate_count = int(conn.execute(text(f"""
            SELECT COUNT(*) FROM (
                SELECT canonical_player_key, season, gameweek
                FROM {FEATURE_TABLE}
                GROUP BY canonical_player_key, season, gameweek
                HAVING COUNT(*) > 1
            ) dup
        """)).scalar_one())
        target_mismatch = int(conn.execute(text(f"""
            WITH source_points AS (
                SELECT
                    im.canonical_player_key,
                    h.season,
                    h.gameweek,
                    SUM(h.total_points) AS target_total_points
                FROM {HISTORY_TABLE} h
                JOIN {IDENTITY_TABLE} im
                  ON h.season = im.season
                 AND h.player_source_id = im.player_source_id
                 AND im.mapping_method = 'element_id_exact'
                GROUP BY im.canonical_player_key, h.season, h.gameweek
            )
            SELECT COUNT(*)
            FROM {FEATURE_TABLE} f
            JOIN source_points sp
              ON f.canonical_player_key = sp.canonical_player_key
             AND f.season = sp.season
             AND f.gameweek = sp.gameweek
            WHERE f.target_total_points IS DISTINCT FROM sp.target_total_points
        """)).scalar_one())
        no_identity = int(conn.execute(text(f"""
            SELECT COUNT(*)
            FROM {FEATURE_TABLE} f
            LEFT JOIN {IDENTITY_TABLE} im
              ON f.season = im.season
             AND f.player_source_id = im.player_source_id
             AND f.fpl_code = im.fpl_code
            WHERE im.identity_id IS NULL
        """)).scalar_one())

    if duplicate_count or target_mismatch or no_identity:
        raise RuntimeError(
            "Post-write checks failed: "
            f"duplicates={duplicate_count}, target_mismatch={target_mismatch}, "
            f"no_identity={no_identity}"
        )
    return {
        "feature_table_row_count": total_rows,
        "post_write_duplicate_count": duplicate_count,
        "post_write_target_mismatch_count": target_mismatch,
        "post_write_no_identity_count": no_identity,
    }


def _manual_sample_rechecks(features: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    del history
    source = features[
        [
            "source_history_row_id",
            "canonical_player_key",
            "fpl_code",
            "season",
            "gameweek",
            "target_total_points",
        ]
    ].copy()
    source["season_order"] = source["season"].apply(_season_order)
    candidates = features[features["prior_history_row_count"] >= 5].head(5)
    rows = []

    for _, row in candidates.iterrows():
        prior = source[
            (source["fpl_code"] == row["fpl_code"])
            & (
                (source["season_order"] < _season_order(row["season"]))
                | (
                    (source["season"] == row["season"])
                    & (source["gameweek"] < int(row["gameweek"]))
                )
            )
        ].sort_values(["season_order", "gameweek", "source_history_row_id"])

        last3 = prior.tail(3)
        expected_last3 = (
            float(last3["target_total_points"].mean()) if not last3.empty else math.nan
        )
        actual_last3 = row["prior_points_last3"]
        rows.append(
            {
                "canonical_player_key": row["canonical_player_key"],
                "season": row["season"],
                "gameweek": int(row["gameweek"]),
                "player_name": row["player_name"],
                "expected_prior_points_last3": expected_last3,
                "actual_prior_points_last3": actual_last3,
                "check_passed": (
                    pd.isna(expected_last3)
                    and pd.isna(actual_last3)
                )
                or abs(float(actual_last3) - expected_last3) < 1e-9,
            }
        )

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| (none) |\n|--------|"
    columns = list(df.columns)

    def _format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(_format_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_markdown_report(results: dict[str, Any], output_path: Path = REPORT_OUTPUT_PATH) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    season_counts = results["season_counts"]
    gameweek_counts = results["gameweek_counts"]
    feature_nulls = results["feature_null_summary"]
    sample_rows = results["sample_rows"]
    sample_checks = results["sample_checks"]

    lines = [
        "# FPL v3 Feature Readiness Audit",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        "Phase 2A builds chronological pre-gameweek player features only. No model training, evaluation, tuning, Streamlit changes, or production model artifact work was performed.",
        "",
        "2025-26 is present only as raw feature output metadata and was not loaded into any model dataframe.",
        "",
        "## Feature Definitions",
        "",
        "| Feature | Source column | Lookback | Time cutoff | Null behavior |",
        "|---------|---------------|----------|-------------|---------------|",
        "| target_total_points | total_points | current row target | current gameweek target only | preserved null if source null |",
        "| prior_points_last1 | total_points | last 1 prior player row | strictly before target row | null if no prior row |",
        "| prior_points_last3/5/10 | total_points | mean over last N prior player rows | strictly before target row | null if no prior row |",
        "| prior_points_season | total_points | same-season cumulative sum before target GW | gameweeks < current GW | 0 at first same-season GW |",
        "| prior_minutes_last3/5/10 | minutes | mean over last N prior player rows | strictly before target row | null if no prior row |",
        "| prior_appearances_last5 | minutes > 0 | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_starts_last5 | starts | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_goals_last5 | goals_scored | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_assists_last5 | assists | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_bonus_last5 | bonus | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_clean_sheets_last5 | clean_sheets | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_saves_last5 | saves | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_xg_last5 | expected_goals | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_xa_last5 | expected_assists | sum over last 5 prior player rows | strictly before target row | null if no prior row |",
        "| prior_points_per_90 | total_points, minutes | all prior player rows | strictly before target row | null if no prior minutes |",
        "| prior_minutes_total | minutes | all prior player rows | strictly before target row | 0 if no prior minutes |",
        "| prior_gameweeks_played | minutes > 0 | all prior player rows | strictly before target row | 0 if no prior appearances |",
        "",
        "No final-season players_raw.csv team or position metadata is used as a historical model feature.",
        "",
        "## Global Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Source history rows | {results['source_history_rows']} |",
        f"| Identity-map rows | {results['identity_map_rows']} |",
        f"| Feature table rows | {results['feature_table_row_count']} |",
        f"| Distinct fpl_code | {results['distinct_fpl_code_count']} |",
        f"| Rows with no identity mapping | {results['rows_with_no_identity_mapping']} |",
        f"| Rows with no prior history | {results['rows_with_no_prior_history']} |",
        f"| Duplicate canonical_player_key/season/gameweek | {results['duplicate_feature_key_count']} |",
        f"| Target null count | {results['target_null_count']} |",
        f"| Target/result mismatch count | {results['post_write_target_mismatch_count']} |",
        f"| 2025-26 feature rows | {results['holdout_2025_26_rows']} |",
        "",
        "## Leakage Audit",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Rows checked with prior history | {results['checked_rows_with_prior_history']} |",
        f"| Future/current gameweek leak rows | {results['future_leak_rows']} |",
        f"| target_total_points absent from model feature list | {not results['target_total_points_in_model_features']} |",
        "| No future-season rows in prior windows | True |",
        "| No current-gameweek source values in input features | True |",
        "| No final-season metadata used as historical features | True |",
        "",
        "## Rows By Season",
        "",
        _df_to_markdown(season_counts),
        "",
        "## Rows By Gameweek",
        "",
        _df_to_markdown(gameweek_counts),
        "",
        "## Feature Null Summary",
        "",
        _df_to_markdown(feature_nulls),
        "",
        "## Required Null Counts",
        "",
        "| Column | Null count |",
        "|--------|------------|",
    ]
    for column, count in results["required_nulls"].items():
        lines.append(f"| {column} | {count} |")

    lines.extend([
        "",
        "## Manual Sample Rechecks",
        "",
        _df_to_markdown(sample_checks),
        "",
        "## Sample Feature Rows",
        "",
        _df_to_markdown(sample_rows),
        "",
        "## Protected Tier 2 Counts",
        "",
        "| Table | Before | After | Status |",
        "|-------|--------|-------|--------|",
    ])
    for table in PROTECTED_TIER2_TABLES:
        before = results["protected_before"].get(table)
        after = results["protected_after"].get(table)
        status = "UNCHANGED" if before == after else "CHANGED"
        lines.append(f"| {table} | {before} | {after} | {status} |")

    lines.extend([
        "",
        "## Model Artifact Status",
        "",
        f"`git status --short models/saved` output during final validation should remain empty. In-script model artifact status: {results['model_artifact_status']}.",
        "",
        "## Confirmation",
        "",
        "- No model training happened.",
        "- No model evaluation happened.",
        "- No tuning happened.",
        "- No Streamlit changes happened.",
        "- No production model artifact work happened.",
        "- Only fpl_player_features_v3 was written.",
        "",
    ])

    content = "\n".join(lines)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Audit report written to: {output_path}")
    except PermissionError as exc:
        print(f"WARNING: Could not write audit report due filesystem permission: {exc}")
        print("REPORT_CONTENT_BEGIN")
        print(content)
        print("REPORT_CONTENT_END")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Tier 3 FPL prior-only player features v3"
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_OUTPUT_PATH,
        help="Markdown report output path",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Tier 3 FPL Phase 2A: Prior-only feature foundation")
    print("=" * 72)

    engine = get_engine()
    protected_before = _capture_protected_counts(engine)
    source_history_rows = _row_count(engine, HISTORY_TABLE)
    identity_map_rows = _row_count(engine, IDENTITY_TABLE)
    print(f"Source history rows: {source_history_rows}")
    print(f"Identity-map rows: {identity_map_rows}")

    history = load_history(engine)
    identity_map = load_identity_map(engine)

    features = build_prior_only_features(history, identity_map)
    leakage_stats = validate_no_future_rows(features)
    target_stats = validate_target_not_in_features()
    coverage_stats = validate_feature_coverage(features)

    print("PASS: prior-only feature leakage checks passed before write")
    print("PASS: target_total_points is absent from model feature columns")

    init_schema(engine)
    rows_written = write_features(engine, features)
    post_write_stats = _post_write_checks(engine)
    protected_after = _capture_protected_counts(engine)

    for table in PROTECTED_TIER2_TABLES:
        before = protected_before.get(table)
        after = protected_after.get(table)
        if before != after:
            raise RuntimeError(f"Protected Tier 2 table changed: {table} {before} -> {after}")

    sample_rows = features[
        [
            "canonical_player_key",
            "fpl_code",
            "season",
            "gameweek",
            "player_name",
            "target_total_points",
            "prior_points_last3",
            "prior_minutes_last3",
            "prior_points_season",
            "feature_history_end_season",
            "feature_history_end_gameweek",
        ]
    ].head(10)
    sample_checks = _manual_sample_rechecks(features, history)

    results: dict[str, Any] = {
        "source_history_rows": source_history_rows,
        "identity_map_rows": identity_map_rows,
        "feature_table_row_count": rows_written,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "sample_rows": sample_rows,
        "sample_checks": sample_checks,
        "model_artifact_status": "not touched by this script",
        **leakage_stats,
        **target_stats,
        **coverage_stats,
        **post_write_stats,
    }
    write_markdown_report(results, args.report_path)

    print("=" * 72)
    print("PHASE 2A COMPLETE")
    print("=" * 72)
    print(f"Feature rows: {rows_written}")
    print(f"Distinct fpl_code: {coverage_stats['distinct_fpl_code_count']}")
    print(f"Rows with no prior history: {coverage_stats['rows_with_no_prior_history']}")
    print(f"2025-26 rows: {coverage_stats['holdout_2025_26_rows']} (metadata only)")
    print("Protected Tier 2 counts unchanged.")
    print("No training, evaluation, tuning, Streamlit, or production model work happened.")


if __name__ == "__main__":
    main()
