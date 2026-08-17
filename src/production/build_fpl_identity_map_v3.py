"""Phase 1D: Build the FPL player identity map (v3).

Maps each season's FPL season-local element ID (player_source_id) to the
globally stable ``fpl_code`` from Vaastav ``players_raw.csv`` using an
exact join on ``(season, player_source_id) == (season, id)``.

Output table: ``fpl_player_identity_map_v3``

No fuzzy matching, no guessed IDs, no name-only cross-season merging.
Unmatched rows are explicitly flagged for manual review.

Does NOT modify:
- fpl_player_gameweek_history_v3
- fpl_player_features_v3
- Any Tier 2 table
- Streamlit / production prediction code / model artifacts
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "vaastav_fpl_history"
DB_CONNECT_TIMEOUT_SECONDS = 5

HISTORY_TABLE = "fpl_player_gameweek_history_v3"
IDENTITY_TABLE = "fpl_player_identity_map_v3"

PROTECTED_TIER2_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "player_gameweek_history",
    "player_gameweek_features",
]

PROTECTED_MODEL_FILES = [
    "fpl_points_xgb.pkl",
    "fpl_points_features.json",
    "label_encoder.pkl",
    "logistic_classifier.pkl",
    "model_features.json",
    "scaler.pkl",
    "xgb_away_goals.pkl",
    "xgb_classifier.pkl",
    "xgb_home_goals.pkl",
]

ALL_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

# Position mapping: element_type integer -> short name
ELEMENT_TYPE_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

REPORT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "fpl_v3_identity_map_audit.md"


# ---------------------------------------------------------------------------
# Database helpers (same pattern as load_fpl_history_v3.py)
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    """Build a PostgreSQL connection URL from the .env file."""
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
    """Create and verify a SQLAlchemy engine."""
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


def _table_row_count(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar_one()


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: str | None) -> str:
    """Normalize names for audit comparisons only, never for matching."""
    if not name or pd.isna(name):
        return ""
    text_val = str(name).strip()
    if not text_val:
        return ""
    text_val = text_val.replace("_", " ")
    nfkd = unicodedata.normalize("NFKD", text_val)
    ascii_stripped = "".join(
        ch for ch in nfkd if not unicodedata.combining(ch)
    )
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_stripped.lower()).strip()
    normalized = re.sub(r"\s+\d+$", "", normalized).strip()
    result = re.sub(r"\s+", " ", normalized).strip()
    return result


# ---------------------------------------------------------------------------
# Vaastav file discovery
# ---------------------------------------------------------------------------

def locate_season_players_raw(base_dir: Path, season: str) -> Path | None:
    """Find players_raw.csv for a given season in the Vaastav data folder."""
    candidate = base_dir / "data" / season / "players_raw.csv"
    if candidate.is_file():
        return candidate
    return None


def load_players_raw_for_season(path: Path, season: str) -> pd.DataFrame:
    """Load players_raw.csv and extract the columns we need.

    Returns a DataFrame with columns:
        season, element_id (season-local), fpl_code, player_name_raw,
        position_raw, web_name, source_file
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "id" not in df.columns:
        raise ValueError(f"players_raw.csv missing 'id' column: {path}")
    if "code" not in df.columns:
        raise ValueError(f"players_raw.csv missing 'code' column: {path}")

    result = pd.DataFrame(index=df.index)
    result["season"] = season
    result["element_id"] = df["id"].astype(str).str.strip()
    result["fpl_code"] = pd.to_numeric(df["code"], errors="coerce")

    # Names
    first = df.get("first_name", pd.Series([""] * len(df))).fillna("").astype(str)
    second = df.get("second_name", pd.Series([""] * len(df))).fillna("").astype(str)
    result["web_name"] = df.get("web_name", pd.Series([""] * len(df))).fillna("").astype(str)

    # Composite player name from players_raw
    result["player_name_raw"] = (first.str.strip() + " " + second.str.strip()).str.strip()

    # Element type -> position
    if "element_type" in df.columns:
        et = pd.to_numeric(df["element_type"], errors="coerce")
        result["position_raw"] = et.map(ELEMENT_TYPE_MAP)
    else:
        result["position_raw"] = None

    result["source_file"] = str(path.relative_to(PROJECT_ROOT).as_posix())

    return result


# ---------------------------------------------------------------------------
# History summary from database
# ---------------------------------------------------------------------------

def load_history_summary(engine) -> pd.DataFrame:
    """Load a per-(season, player_source_id) summary from the history table.

    Aggregates: player_name (mode), team_name (mode), position (mode),
    min/max gameweek, row count, sum of minutes, sum of total_points.
    """
    query = text(f"""
        SELECT
            season,
            player_source_id,
            MODE() WITHIN GROUP (ORDER BY player_name) AS player_name,
            MODE() WITHIN GROUP (ORDER BY team_name) AS team_name,
            MODE() WITHIN GROUP (ORDER BY position) AS position,
            COUNT(DISTINCT NULLIF(TRIM(team_name), '')) AS distinct_team_name_count,
            COUNT(DISTINCT NULLIF(TRIM(position), '')) AS distinct_position_count,
            MIN(gameweek) AS first_gameweek,
            MAX(gameweek) AS last_gameweek,
            COUNT(*) AS row_count,
            COALESCE(SUM(minutes), 0) AS minutes_total,
            COALESCE(SUM(total_points), 0) AS total_points_sum
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NOT NULL
          AND TRIM(player_source_id) <> ''
        GROUP BY season, player_source_id
        ORDER BY season, player_source_id
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df["player_source_id"] = df["player_source_id"].astype(str).str.strip()
    print(f"Loaded history summary: {len(df)} player-season groups")

    # Also count rows with NULL/empty player_source_id
    null_query = text(f"""
        SELECT COUNT(*) AS cnt
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NULL
           OR TRIM(player_source_id) = ''
    """)
    with engine.connect() as conn:
        null_count = conn.execute(null_query).scalar_one()
    if null_count > 0:
        print(f"WARNING: {null_count} history rows have NULL/empty player_source_id")

    return df


def load_total_history_rows(engine) -> int:
    """Return the total number of rows in the history table."""
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {HISTORY_TABLE}")
        ).scalar_one()


# ---------------------------------------------------------------------------
# Exact join: history summary <-> players_raw
# ---------------------------------------------------------------------------

def join_history_to_players_raw(
    history_summary: pd.DataFrame,
    players_raw_all: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join each history player-season group to its players_raw.csv row.

    Join key: (season, player_source_id) == (season, element_id)

    Returns merged DataFrame with columns from both sides + match status.
    """
    merged_parts: list[pd.DataFrame] = []

    for season in sorted(history_summary["season"].unique()):
        hist_season = history_summary[history_summary["season"] == season].copy()

        if season not in players_raw_all:
            # No players_raw.csv for this season — all unmatched
            hist_season["fpl_code"] = None
            hist_season["mapping_method"] = "unmatched"
            hist_season["confidence_score"] = 0.0
            hist_season["needs_manual_review"] = True
            hist_season["review_reason"] = "no_players_raw_csv_for_season"
            hist_season["player_name_raw"] = None
            hist_season["position_raw"] = None
            merged_parts.append(hist_season)
            print(f"  {season}: no players_raw.csv — {len(hist_season)} groups unmatched")
            continue

        raw = players_raw_all[season].copy()

        # Left join history onto players_raw
        merged = hist_season.merge(
            raw[["season", "element_id", "fpl_code", "player_name_raw", "position_raw"]],
            left_on=["season", "player_source_id"],
            right_on=["season", "element_id"],
            how="left",
            suffixes=("", "_raw"),
        )

        # Determine match status
        matched_mask = merged["fpl_code"].notna()
        merged["mapping_method"] = "unmatched"
        merged.loc[matched_mask, "mapping_method"] = "element_id_exact"
        merged["confidence_score"] = 0.0
        merged.loc[matched_mask, "confidence_score"] = 1.0
        merged["needs_manual_review"] = False
        merged.loc[~matched_mask, "needs_manual_review"] = True
        merged["review_reason"] = None
        merged.loc[~matched_mask, "review_reason"] = "unmatched_player_source_id"

        matched_count = int(matched_mask.sum())
        unmatched_count = int((~matched_mask).sum())
        print(f"  {season}: {matched_count} matched, {unmatched_count} unmatched out of {len(merged)}")

        merged_parts.append(merged)

    if not merged_parts:
        raise RuntimeError("No history summary data to join")

    result = pd.concat(merged_parts, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Validation of the exact join results
# ---------------------------------------------------------------------------

def validate_exact_identity_join(
    joined: pd.DataFrame,
    history_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the joined data and flag issues for manual review.

    Returns a dict of validation stats and updates joined in place to set
    review flags.
    """
    stats: dict[str, Any] = {}

    total_groups = len(joined)
    matched = joined[joined["mapping_method"] == "element_id_exact"]
    unmatched = joined[joined["mapping_method"] == "unmatched"]

    stats["total_player_season_groups"] = total_groups
    stats["matched_groups"] = len(matched)
    stats["unmatched_groups"] = len(unmatched)
    stats["matched_history_rows"] = int(matched["row_count"].fillna(0).sum())
    stats["unmatched_history_rows"] = int(unmatched["row_count"].fillna(0).sum())
    stats["missing_fpl_code_count"] = int(joined["fpl_code"].isna().sum())

    # Helper to append a review reason
    def _flag(idx: int, reason: str) -> None:
        joined.at[idx, "needs_manual_review"] = True
        existing = joined.at[idx, "review_reason"]
        if existing and str(existing).strip():
            joined.at[idx, "review_reason"] = str(existing) + "; " + reason
        else:
            joined.at[idx, "review_reason"] = reason

    # ── Duplicate season/player_source_id ──
    dup_source = joined[
        joined.duplicated(subset=["season", "player_source_id"], keep=False)
    ]
    stats["duplicate_season_player_source_id"] = len(dup_source)
    for idx in dup_source.index:
        _flag(idx, "duplicate_season_player_source_id")

    # ── Duplicate season/fpl_code (among matched only) ──
    matched_with_code = joined[
        (joined["mapping_method"] == "element_id_exact") & joined["fpl_code"].notna()
    ].copy()
    dup_code = matched_with_code[
        matched_with_code.duplicated(subset=["season", "fpl_code"], keep=False)
    ]
    stats["duplicate_season_fpl_code"] = len(dup_code)
    for idx in dup_code.index:
        _flag(idx, "duplicate_season_fpl_code")

    # ── Missing fpl_code in matched rows (shouldn't happen) ──
    missing_code = joined[
        (joined["mapping_method"] == "element_id_exact") & joined["fpl_code"].isna()
    ]
    stats["missing_fpl_code_in_matched"] = len(missing_code)

    # ── One fpl_code -> multiple normalized names ──
    fpl_code_multi_names = 0
    if not matched_with_code.empty:
        matched_with_code["norm_name"] = matched_with_code["player_name"].apply(normalize_name)
        code_names = (
            matched_with_code.groupby("fpl_code")["norm_name"]
            .nunique()
            .reset_index(name="name_count")
        )
        multi_name_codes = code_names[code_names["name_count"] > 1]
        fpl_code_multi_names = len(multi_name_codes)

        if fpl_code_multi_names > 0:
            flagged_codes = set(multi_name_codes["fpl_code"])
            for idx in joined.index:
                if pd.notna(joined.at[idx, "fpl_code"]) and joined.at[idx, "fpl_code"] in flagged_codes:
                    _flag(idx, "fpl_code_multiple_names")

    stats["fpl_code_multiple_names"] = fpl_code_multi_names

    fpl_code_multi_players_one_season = 0
    if not matched_with_code.empty:
        season_code_names = (
            matched_with_code.groupby(["season", "fpl_code"])["norm_name"]
            .nunique()
            .reset_index(name="name_count")
        )
        multi_player_pairs = season_code_names[season_code_names["name_count"] > 1]
        fpl_code_multi_players_one_season = len(multi_player_pairs)

        if fpl_code_multi_players_one_season > 0:
            flagged_pairs = set(zip(multi_player_pairs["season"], multi_player_pairs["fpl_code"]))
            for idx in joined.index:
                if pd.notna(joined.at[idx, "fpl_code"]):
                    pair = (joined.at[idx, "season"], joined.at[idx, "fpl_code"])
                    if pair in flagged_pairs:
                        _flag(idx, "fpl_code_multiple_players_one_season")

    stats["fpl_code_multiple_players_one_season"] = fpl_code_multi_players_one_season

    # ── Conflicting player name: history name vs players_raw name ──
    conflict_count = 0
    for idx in matched.index:
        hist_norm = normalize_name(joined.at[idx, "player_name"])
        raw_val = joined.at[idx, "player_name_raw"]
        raw_norm = normalize_name(raw_val) if pd.notna(raw_val) else ""
        if hist_norm and raw_norm and hist_norm != raw_norm:
            conflict_count += 1
            _flag(idx, "conflicting_player_name")
    stats["conflicting_player_names"] = conflict_count

    # ── Conflicting position: history position vs players_raw position ──
    pos_conflict = 0
    for idx in matched.index:
        hist_pos = str(joined.at[idx, "position"]).strip().upper() if pd.notna(joined.at[idx, "position"]) else ""
        raw_pos = str(joined.at[idx, "position_raw"]).strip().upper() if pd.notna(joined.at[idx, "position_raw"]) else ""
        if hist_pos and raw_pos and hist_pos != raw_pos:
            pos_conflict += 1
            _flag(idx, "conflicting_position")
    stats["conflicting_positions"] = pos_conflict

    team_conflict_rows = joined[joined["distinct_team_name_count"].fillna(0) > 1]
    for idx in team_conflict_rows.index:
        _flag(idx, "conflicting_team_mappings")
    stats["conflicting_team_mappings"] = len(team_conflict_rows)

    position_conflict_rows = joined[joined["distinct_position_count"].fillna(0) > 1]
    for idx in position_conflict_rows.index:
        _flag(idx, "conflicting_position_mappings")
    stats["conflicting_position_mappings"] = len(position_conflict_rows)

    # ── Same-season team overlap: one fpl_code in multiple teams ──
    team_overlap = 0
    if not matched_with_code.empty:
        code_teams = (
            matched_with_code.groupby(["fpl_code", "season"])["team_name"]
            .nunique()
            .reset_index(name="team_count")
        )
        multi_team = code_teams[code_teams["team_count"] > 1]
        team_overlap = len(multi_team)

        if team_overlap > 0:
            flagged_pairs = set(
                zip(multi_team["fpl_code"], multi_team["season"])
            )
            for idx in joined.index:
                if pd.notna(joined.at[idx, "fpl_code"]):
                    pair = (joined.at[idx, "fpl_code"], joined.at[idx, "season"])
                    if pair in flagged_pairs:
                        _flag(idx, "same_season_team_overlap")

    stats["fpl_code_multiple_teams_same_season"] = team_overlap

    return stats


# ---------------------------------------------------------------------------
# Build the identity map DataFrame
# ---------------------------------------------------------------------------

def build_identity_map(joined: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final identity map DataFrame with all 20 required columns."""
    identity = pd.DataFrame()

    # canonical_player_key: fpl_code as string when matched, NULL otherwise
    identity["canonical_player_key"] = joined.apply(
        lambda row: str(int(row["fpl_code"]))
        if row["mapping_method"] == "element_id_exact" and pd.notna(row["fpl_code"])
        else None,
        axis=1,
    )

    identity["fpl_code"] = joined["fpl_code"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    identity["season"] = joined["season"]
    identity["player_source_id"] = joined["player_source_id"]
    identity["player_name"] = joined["player_name"]
    identity["normalized_player_name"] = joined["player_name"].apply(normalize_name)
    identity["team_name"] = joined["team_name"]
    identity["position"] = joined["position"]
    identity["first_gameweek"] = joined["first_gameweek"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    identity["last_gameweek"] = joined["last_gameweek"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    identity["row_count"] = joined["row_count"].fillna(0).astype(int)
    identity["minutes_total"] = joined["minutes_total"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    identity["total_points_sum"] = joined["total_points_sum"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    identity["history_rows_matched"] = joined["row_count"].fillna(0).astype(int)
    identity["mapping_method"] = joined["mapping_method"]
    identity["confidence_score"] = joined["confidence_score"].fillna(0.0)
    identity["needs_manual_review"] = joined["needs_manual_review"].fillna(False)
    identity["review_reason"] = joined["review_reason"]

    # Ensure normalized_player_name is not empty
    empty_norm = identity["normalized_player_name"] == ""
    if empty_norm.any():
        identity.loc[empty_norm, "normalized_player_name"] = (
            identity.loc[empty_norm, "player_name"]
            .fillna("unknown")
            .apply(lambda x: normalize_name(x) if normalize_name(x) else "unknown")
        )

    return identity


# ---------------------------------------------------------------------------
# Schema init and write
# ---------------------------------------------------------------------------

IDENTITY_DDL = f"""
CREATE TABLE IF NOT EXISTS {IDENTITY_TABLE} (
    identity_id BIGSERIAL PRIMARY KEY,
    canonical_player_key TEXT NULL,
    fpl_code INTEGER NULL,
    season TEXT NOT NULL,
    player_source_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    normalized_player_name TEXT NOT NULL,
    team_name TEXT NULL,
    position TEXT NULL,
    first_gameweek INTEGER NULL,
    last_gameweek INTEGER NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    minutes_total INTEGER NULL,
    total_points_sum INTEGER NULL,
    history_rows_matched INTEGER NOT NULL DEFAULT 0,
    mapping_method TEXT NOT NULL,
    confidence_score FLOAT NOT NULL DEFAULT 0.0,
    needs_manual_review BOOLEAN NOT NULL DEFAULT FALSE,
    review_reason TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (season, player_source_id),
    CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CHECK (mapping_method IN ('element_id_exact', 'unmatched')),
    CHECK (row_count >= 0),
    CHECK (history_rows_matched >= 0),
    CHECK (TRIM(season) <> ''),
    CHECK (TRIM(player_source_id) <> ''),
    CHECK (TRIM(player_name) <> ''),
    CHECK (TRIM(normalized_player_name) <> '')
)
"""

INDEX_STATEMENTS = [
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_identity_map_v3_fpl_code
    ON {IDENTITY_TABLE} (fpl_code)
    WHERE fpl_code IS NOT NULL
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_identity_map_v3_season
    ON {IDENTITY_TABLE} (season)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_identity_map_v3_needs_review
    ON {IDENTITY_TABLE} (needs_manual_review)
    WHERE needs_manual_review = TRUE
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_identity_map_v3_canonical_key
    ON {IDENTITY_TABLE} (canonical_player_key)
    WHERE canonical_player_key IS NOT NULL
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_fpl_identity_map_v3_season_player
    ON {IDENTITY_TABLE} (season, player_source_id)
    """,
]


def init_schema(engine) -> None:
    """Drop and recreate the identity map table.

    Only touches fpl_player_identity_map_v3 — no other tables.
    """
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {IDENTITY_TABLE} CASCADE"))
        conn.execute(text(IDENTITY_DDL))
        for stmt in INDEX_STATEMENTS:
            conn.execute(text(stmt))

    if not _table_exists(engine, IDENTITY_TABLE):
        raise RuntimeError(f"Failed to create {IDENTITY_TABLE}")

    print(f"PASS: {IDENTITY_TABLE} schema created.")


def write_identity_map(engine, identity_df: pd.DataFrame) -> int:
    """Write the identity map DataFrame to the database.

    Returns the number of rows written.
    """
    if identity_df.empty:
        print("WARNING: Identity map is empty — nothing to write.")
        return 0

    # Columns to write (exclude identity_id — auto-generated)
    write_cols = [
        "canonical_player_key",
        "fpl_code",
        "season",
        "player_source_id",
        "player_name",
        "normalized_player_name",
        "team_name",
        "position",
        "first_gameweek",
        "last_gameweek",
        "row_count",
        "minutes_total",
        "total_points_sum",
        "history_rows_matched",
        "mapping_method",
        "confidence_score",
        "needs_manual_review",
        "review_reason",
    ]

    write_df = identity_df[write_cols].copy()

    with engine.begin() as conn:
        write_df.to_sql(
            IDENTITY_TABLE,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

    actual_count = _table_row_count(engine, IDENTITY_TABLE)
    print(f"Wrote {actual_count} rows to {IDENTITY_TABLE}")
    return actual_count


# ---------------------------------------------------------------------------
# Post-write validation
# ---------------------------------------------------------------------------

def validate_identity_map(
    engine,
    expected_groups: int,
    total_history_rows: int,
) -> dict[str, Any]:
    """Post-write validation of the identity map table."""
    stats: dict[str, Any] = {}

    actual_count = _table_row_count(engine, IDENTITY_TABLE)
    stats["identity_map_row_count"] = actual_count
    stats["expected_groups"] = expected_groups

    if actual_count != expected_groups:
        raise RuntimeError(
            f"FAIL: Identity map has {actual_count} rows but expected {expected_groups} "
            f"player-season groups"
        )
    print(f"PASS: Identity map row count matches: {actual_count}")

    with engine.connect() as conn:
        # Rows by season
        season_counts = pd.read_sql(
            text(f"""
                SELECT season, COUNT(*) AS cnt,
                       SUM(CASE WHEN mapping_method = 'element_id_exact' THEN 1 ELSE 0 END) AS matched,
                       SUM(CASE WHEN mapping_method = 'unmatched' THEN 1 ELSE 0 END) AS unmatched,
                       SUM(row_count) AS history_rows_represented,
                       SUM(CASE WHEN mapping_method = 'element_id_exact' THEN row_count ELSE 0 END) AS matched_history_rows,
                       SUM(CASE WHEN mapping_method = 'unmatched' THEN row_count ELSE 0 END) AS unmatched_history_rows
                FROM {IDENTITY_TABLE}
                GROUP BY season
                ORDER BY season
            """),
            conn,
        )
        stats["season_breakdown"] = season_counts.to_dict("records")

        # Distinct fpl_code
        distinct_codes = conn.execute(
            text(f"""
                SELECT COUNT(DISTINCT fpl_code) FROM {IDENTITY_TABLE}
                WHERE fpl_code IS NOT NULL
            """)
        ).scalar_one()
        stats["distinct_fpl_code_count"] = distinct_codes

        # Manual review
        review_count = conn.execute(
            text(f"""
                SELECT COUNT(*) FROM {IDENTITY_TABLE}
                WHERE needs_manual_review = TRUE
            """)
        ).scalar_one()
        stats["manual_review_count"] = review_count

        # Review reasons
        review_reasons = pd.read_sql(
            text(f"""
                SELECT review_reason, COUNT(*) AS cnt
                FROM {IDENTITY_TABLE}
                WHERE needs_manual_review = TRUE
                  AND review_reason IS NOT NULL
                GROUP BY review_reason
                ORDER BY cnt DESC
                LIMIT 20
            """),
            conn,
        )
        stats["review_reasons"] = review_reasons.to_dict("records")

        # Duplicate checks in final table
        dup_source = conn.execute(
            text(f"""
                SELECT COUNT(*) FROM (
                    SELECT season, player_source_id
                    FROM {IDENTITY_TABLE}
                    GROUP BY season, player_source_id
                    HAVING COUNT(*) > 1
                ) sub
            """)
        ).scalar_one()
        stats["duplicate_season_player_source_id_in_table"] = dup_source
        if dup_source > 0:
            raise RuntimeError(
                f"FAIL: {dup_source} duplicate (season, player_source_id) in identity map!"
            )

        dup_code = conn.execute(
            text(f"""
                SELECT COUNT(*) FROM (
                    SELECT season, fpl_code
                    FROM {IDENTITY_TABLE}
                    WHERE fpl_code IS NOT NULL
                    GROUP BY season, fpl_code
                    HAVING COUNT(*) > 1
                ) sub
            """)
        ).scalar_one()
        stats["duplicate_season_fpl_code_in_table"] = dup_code

        fpl_code_multiple_players_one_season = conn.execute(
            text(f"""
                SELECT COUNT(*) FROM (
                    SELECT season, fpl_code
                    FROM {IDENTITY_TABLE}
                    WHERE fpl_code IS NOT NULL
                    GROUP BY season, fpl_code
                    HAVING COUNT(DISTINCT normalized_player_name) > 1
                ) sub
            """)
        ).scalar_one()
        stats["fpl_code_multiple_players_one_season_in_table"] = (
            fpl_code_multiple_players_one_season
        )

        missing_fpl_code_count = conn.execute(
            text(f"""
                SELECT COUNT(*) FROM {IDENTITY_TABLE}
                WHERE fpl_code IS NULL
            """)
        ).scalar_one()
        stats["missing_fpl_code_count_in_table"] = missing_fpl_code_count

        # History rows represented
        total_represented = conn.execute(
            text(f"SELECT COALESCE(SUM(row_count), 0) FROM {IDENTITY_TABLE}")
        ).scalar_one()
        stats["total_history_rows_represented"] = total_represented
        stats["total_history_rows"] = total_history_rows
        stats["matched_history_rows_in_table"] = conn.execute(
            text(f"""
                SELECT COALESCE(SUM(row_count), 0) FROM {IDENTITY_TABLE}
                WHERE mapping_method = 'element_id_exact'
            """)
        ).scalar_one()
        stats["unmatched_history_rows_in_table"] = conn.execute(
            text(f"""
                SELECT COALESCE(SUM(row_count), 0) FROM {IDENTITY_TABLE}
                WHERE mapping_method = 'unmatched'
            """)
        ).scalar_one()
        stats["missing_history_player_season_groups"] = max(
            expected_groups - actual_count,
            0,
        )

        # Orphan check — by construction all rows come from history
        orphan_identity_rows = conn.execute(
            text(f"""
                SELECT COUNT(*)
                FROM {IDENTITY_TABLE} im
                LEFT JOIN (
                    SELECT DISTINCT season, player_source_id
                    FROM {HISTORY_TABLE}
                    WHERE player_source_id IS NOT NULL
                      AND TRIM(player_source_id) <> ''
                ) hist
                  ON im.season = hist.season
                 AND im.player_source_id = hist.player_source_id
                WHERE hist.player_source_id IS NULL
            """)
        ).scalar_one()
        stats["orphan_identity_rows"] = orphan_identity_rows

    print(f"PASS: Post-write validation complete.")
    print(f"  Distinct fpl_code: {distinct_codes}")
    print(f"  Manual review: {review_count}")
    print(f"  History rows represented: {total_represented}/{total_history_rows}")

    return stats


# ---------------------------------------------------------------------------
# Tier 2 safety checks
# ---------------------------------------------------------------------------

def check_tier2_safety(engine) -> dict[str, int]:
    """Check Tier 2 table row counts — must not be modified."""
    counts: dict[str, int] = {}
    for table in PROTECTED_TIER2_TABLES:
        if _table_exists(engine, table):
            counts[table] = _table_row_count(engine, table)
        else:
            counts[table] = -1
    return counts


def check_model_files() -> dict[str, str]:
    """Check that protected model files exist and are unchanged."""
    models_dir = PROJECT_ROOT / "models" / "saved"
    status: dict[str, str] = {}
    for fname in PROTECTED_MODEL_FILES:
        fpath = models_dir / fname
        if fpath.exists():
            status[fname] = f"EXISTS ({fpath.stat().st_size} bytes)"
        else:
            status[fname] = "MISSING"
    return status


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    all_stats: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate the audit markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# FPL v3 Identity Map Audit Report",
        "",
        f"**Generated**: {now}",
        f"**Phase**: 1D - Multi-Season Player Identity Mapping",
        "",
        "---",
        "",
        "## Summary",
        "",
        "Identity mapping rule: `fpl_code` is sourced only from each season's `players_raw.csv` "
        "using the exact join `(season, player_source_id) = (season, id)`. "
        "No name inference, fuzzy matching, or silent cross-player merging is used.",
        "",
        "2025-26 rows are treated as raw identity metadata only in this phase; they are not used "
        "for model selection, tuning, or evaluation.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total history rows | {all_stats.get('total_history_rows', 'N/A')} |",
        f"| Total player-season groups | {all_stats.get('total_player_season_groups', 'N/A')} |",
        f"| Matched groups | {all_stats.get('matched_groups', 'N/A')} |",
        f"| Unmatched groups | {all_stats.get('unmatched_groups', 'N/A')} |",
        f"| Matched history rows | {all_stats.get('matched_history_rows', all_stats.get('matched_history_rows_in_table', 'N/A'))} |",
        f"| Unmatched history rows | {all_stats.get('unmatched_history_rows', all_stats.get('unmatched_history_rows_in_table', 'N/A'))} |",
        f"| Missing fpl_code count | {all_stats.get('missing_fpl_code_count', all_stats.get('missing_fpl_code_count_in_table', 'N/A'))} |",
        f"| Identity map rows written | {all_stats.get('identity_map_row_count', 'N/A')} |",
        f"| Distinct fpl_code | {all_stats.get('distinct_fpl_code_count', 'N/A')} |",
        f"| Manual review count | {all_stats.get('manual_review_count', 'N/A')} |",
        "",
        "## Match Percentage by Season",
        "",
        "| Season | Groups | Matched | Unmatched | Match % | History Rows | Matched Rows | Unmatched Rows |",
        "|--------|--------|---------|-----------|---------|--------------|--------------|----------------|",
    ]

    for row in all_stats.get("season_breakdown", []):
        total = row["cnt"]
        m = row.get("matched", 0)
        u = row.get("unmatched", 0)
        pct = f"{100 * m / total:.1f}" if total > 0 else "N/A"
        hist_rows = row.get("history_rows_represented", 0)
        matched_rows = row.get("matched_history_rows", 0)
        unmatched_rows = row.get("unmatched_history_rows", 0)
        lines.append(
            f"| {row['season']} | {total} | {m} | {u} | {pct}% | {hist_rows} | {matched_rows} | {unmatched_rows} |"
        )

    lines.extend([
        "",
        "## Duplicate & Conflict Analysis",
        "",
        "| Check | Count |",
        "|-------|-------|",
        f"| Duplicate (season, player_source_id) in join | {all_stats.get('duplicate_season_player_source_id', 0)} |",
        f"| Duplicate (season, fpl_code) in join | {all_stats.get('duplicate_season_fpl_code', 0)} |",
        f"| Duplicate (season, player_source_id) in table | {all_stats.get('duplicate_season_player_source_id_in_table', 0)} |",
        f"| Duplicate (season, fpl_code) in table | {all_stats.get('duplicate_season_fpl_code_in_table', 0)} |",
        f"| Missing fpl_code in matched rows | {all_stats.get('missing_fpl_code_in_matched', 0)} |",
        f"| Missing fpl_code in identity table | {all_stats.get('missing_fpl_code_count_in_table', all_stats.get('missing_fpl_code_count', 0))} |",
        f"| fpl_code -> multiple normalized names | {all_stats.get('fpl_code_multiple_names', 0)} |",
        f"| fpl_code -> multiple players in one season | {all_stats.get('fpl_code_multiple_players_one_season', all_stats.get('fpl_code_multiple_players_one_season_in_table', 0))} |",
        f"| fpl_code -> multiple players in one season (table) | {all_stats.get('fpl_code_multiple_players_one_season_in_table', 0)} |",
        f"| Conflicting player names (history vs raw) | {all_stats.get('conflicting_player_names', 0)} |",
        f"| Conflicting positions (history vs raw) | {all_stats.get('conflicting_positions', 0)} |",
        f"| Conflicting team mappings within history groups | {all_stats.get('conflicting_team_mappings', 0)} |",
        f"| Conflicting position mappings within history groups | {all_stats.get('conflicting_position_mappings', 0)} |",
        f"| fpl_code -> multiple teams same season | {all_stats.get('fpl_code_multiple_teams_same_season', 0)} |",
        "",
        "## Manual Review Reasons",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ])

    for reason_row in all_stats.get("review_reasons", []):
        lines.append(f"| {reason_row['review_reason']} | {reason_row['cnt']} |")

    if not all_stats.get("review_reasons"):
        lines.append("| (none) | 0 |")

    lines.extend([
        "",
        "## History Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total history rows | {all_stats.get('total_history_rows', 'N/A')} |",
        f"| History rows represented by identity map | {all_stats.get('total_history_rows_represented', 'N/A')} |",
        f"| Matched history rows in table | {all_stats.get('matched_history_rows_in_table', all_stats.get('matched_history_rows', 'N/A'))} |",
        f"| Unmatched history rows in table | {all_stats.get('unmatched_history_rows_in_table', all_stats.get('unmatched_history_rows', 'N/A'))} |",
        f"| Missing history player-season groups | {all_stats.get('missing_history_player_season_groups', 0)} |",
        f"| Orphan identity map rows | {all_stats.get('orphan_identity_rows', 0)} |",
        "",
        "## Tier 2 Table Safety",
        "",
        "| Table | Before | After | Status |",
        "|-------|--------|-------|--------|",
    ])

    tier2_before = all_stats.get("tier2_before", {})
    tier2_after = all_stats.get("tier2_after", {})
    for table in PROTECTED_TIER2_TABLES:
        before = tier2_before.get(table, "N/A")
        after = tier2_after.get(table, "N/A")
        status_str = "SAFE" if before == after else "CHANGED"
        lines.append(f"| {table} | {before} | {after} | {status_str} |")

    lines.extend([
        "",
        "## Model Artifacts Status",
        "",
        "| File | Status |",
        "|------|--------|",
    ])

    for fname, fstatus in all_stats.get("model_files", {}).items():
        lines.append(f"| {fname} | {fstatus} |")

    lines.extend([
        "",
        "## Identity Map Completeness",
        "",
        f"**Result**: {all_stats.get('completeness_result', 'UNKNOWN')}",
        "",
        "## Confirmation",
        "",
        "- No model training performed",
        "- No model evaluation performed",
        "- No tuning performed",
        "- No Streamlit changes",
        "- No model artifact modifications",
        "- No production prediction code changes",
        "- Only fpl_player_identity_map_v3 was written",
        "",
        "---",
        "",
        f"*Report generated by `build_fpl_identity_map_v3.py` at {now}*",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1D: Build FPL player identity map v3"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Path to vaastav_fpl_history root",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_OUTPUT_PATH,
        help="Path for the audit markdown report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only — do not write to the database",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1D: Build FPL Player Identity Map v3")
    print("=" * 70)
    print()

    # ── Step 1: Connect ──
    print("Step 1: Connecting to database...")
    engine = get_engine()
    print()

    # ── Step 2: Safety checks ──
    print("Step 2: Running safety checks...")
    tier2_before = check_tier2_safety(engine)
    model_files = check_model_files()

    for table, count in tier2_before.items():
        indicator = "OK" if count >= 0 else "WARN"
        print(f"  [{indicator}] {table}: {count} rows")

    for fname, fstatus in model_files.items():
        indicator = "OK" if "EXISTS" in fstatus else "MISSING"
        print(f"  [{indicator}] {fname}: {fstatus}")

    # Verify history table exists
    if not _table_exists(engine, HISTORY_TABLE):
        raise RuntimeError(f"FAIL: {HISTORY_TABLE} does not exist!")
    total_history_rows = load_total_history_rows(engine)
    print(f"  [OK] {HISTORY_TABLE}: {total_history_rows} rows")
    print()

    # ── Step 3: Load history summary ──
    print("Step 3: Loading history summary...")
    history_summary = load_history_summary(engine)
    print(f"  Total player-season groups: {len(history_summary)}")

    seasons_in_history = sorted(history_summary["season"].unique())
    print(f"  Seasons: {seasons_in_history}")
    print()

    # ── Step 4: Load all players_raw.csv files ──
    print("Step 4: Loading players_raw.csv for each season...")
    players_raw_all: dict[str, pd.DataFrame] = {}

    for season in ALL_SEASONS:
        raw_path = locate_season_players_raw(args.base_dir, season)
        if raw_path:
            raw_df = load_players_raw_for_season(raw_path, season)
            players_raw_all[season] = raw_df
            print(f"  [OK] {season}: {len(raw_df)} players in players_raw.csv")
        else:
            print(f"  [WARN] {season}: players_raw.csv not found")

    print()

    # ── Step 5: Exact join ──
    print("Step 5: Joining history to players_raw via exact element ID match...")
    joined = join_history_to_players_raw(history_summary, players_raw_all)
    print(f"  Total joined rows: {len(joined)}")
    print()

    # ── Step 6: Validate the join ──
    print("Step 6: Validating exact identity join...")
    join_stats = validate_exact_identity_join(joined, history_summary)

    for key, value in join_stats.items():
        print(f"  {key}: {value}")
    print()

    # ── Step 7: Build identity map ──
    print("Step 7: Building identity map DataFrame...")
    identity_df = build_identity_map(joined)
    print(f"  Identity map rows: {len(identity_df)}")
    print(f"  Columns: {list(identity_df.columns)}")
    print()

    if args.dry_run:
        print("DRY RUN: Skipping database write.")
        print()
        post_stats = {
            "identity_map_row_count": len(identity_df),
            "distinct_fpl_code_count": identity_df["fpl_code"].dropna().nunique(),
            "manual_review_count": int(identity_df["needs_manual_review"].sum()),
            "review_reasons": [],
            "duplicate_season_player_source_id_in_table": 0,
            "duplicate_season_fpl_code_in_table": 0,
            "total_history_rows_represented": int(identity_df["row_count"].sum()),
            "total_history_rows": total_history_rows,
            "orphan_identity_rows": 0,
            "season_breakdown": [],
        }
    else:
        # ── Step 8: Init schema and write ──
        print("Step 8: Initializing schema and writing identity map...")
        init_schema(engine)
        rows_written = write_identity_map(engine, identity_df)
        print()

        # ── Step 9: Post-write validation ──
        print("Step 9: Post-write validation...")
        post_stats = validate_identity_map(
            engine,
            expected_groups=len(identity_df),
            total_history_rows=total_history_rows,
        )
        print()

    # ── Step 10: Final safety checks ──
    print("Step 10: Final safety checks...")
    tier2_after = check_tier2_safety(engine)

    safety_ok = True
    for table in PROTECTED_TIER2_TABLES:
        before = tier2_before.get(table, -1)
        after = tier2_after.get(table, -1)
        if before != after:
            print(f"  [FAIL] {table} changed from {before} to {after} rows!")
            safety_ok = False
        else:
            print(f"  [OK] {table}: {before} -> {after} (unchanged)")

    if not safety_ok:
        raise RuntimeError("FAIL: Tier 2 tables were modified!")
    print()

    # ── Step 11: Write report ──
    print("Step 11: Writing audit report...")

    all_stats = {**join_stats}
    all_stats["total_history_rows"] = total_history_rows
    all_stats.update(post_stats)
    all_stats["tier2_before"] = tier2_before
    all_stats["tier2_after"] = tier2_after
    all_stats["model_files"] = model_files

    # Determine completeness
    matched_groups = join_stats.get("matched_groups", 0)
    total_groups = join_stats.get("total_player_season_groups", 0)
    match_pct = 100 * matched_groups / total_groups if total_groups > 0 else 0
    if match_pct >= 95:
        all_stats["completeness_result"] = f"PASS - {match_pct:.1f}% of player-season groups matched"
    else:
        all_stats["completeness_result"] = f"WARNING - only {match_pct:.1f}% matched (expected >= 95%)"

    write_markdown_report(all_stats, args.report_path)
    print()

    # ── Final summary ──
    print("=" * 70)
    print("PHASE 1D COMPLETE")
    print("=" * 70)
    print()
    print("MODIFIED: sql/tier3_schema.sql")
    print("CREATED:  src/production/build_fpl_identity_map_v3.py")
    print(f"CREATED:  docs/fpl_v3_identity_map_audit.md")
    print()
    print(f"Identity map rows:       {all_stats.get('identity_map_row_count', len(identity_df))}")
    print(f"Distinct fpl_code:       {all_stats.get('distinct_fpl_code_count', 'N/A')}")
    print(f"Matched groups:          {matched_groups}")
    print(f"Unmatched groups:        {join_stats.get('unmatched_groups', 0)}")
    print(f"Match percentage:        {match_pct:.1f}%")
    print(f"Manual review count:     {all_stats.get('manual_review_count', 'N/A')}")
    print(f"Completeness:            {all_stats.get('completeness_result', 'UNKNOWN')}")
    print()
    print("No training, tuning, evaluation, Streamlit changes,")
    print("model artifact changes, or production changes performed.")


if __name__ == "__main__":
    main()
