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

EXPECTED_ROW_COUNT = 1900
HOME_ADVANTAGE = 50.0
EXPECTED_SEASON_COUNTS = {
    "2021-22": 380,
    "2022-23": 380,
    "2023-24": 380,
    "2024-25": 380,
    "2025-26": 380,
}

BASE_FEATURE_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_win",
    "is_draw",
    "away_win",
    "home_home_matches_last5",
    "away_away_matches_last5",
    "home_overall_matches_last5",
    "away_overall_matches_last5",
    "home_overall_matches_last10",
    "away_overall_matches_last10",
    "home_goals_scored_home_last5",
    "home_goals_conceded_home_last5",
    "home_clean_sheet_rate_home_last5",
    "away_goals_scored_away_last5",
    "away_goals_conceded_away_last5",
    "away_clean_sheet_rate_away_last5",
    "home_xg_home_last5",
    "home_xga_home_last5",
    "away_xg_away_last5",
    "away_xga_away_last5",
    "home_points_overall_last5",
    "away_points_overall_last5",
    "home_points_overall_last10",
    "away_points_overall_last10",
    "home_goal_diff_overall_last5",
    "away_goal_diff_overall_last5",
    "home_xg_overall_last5",
    "home_xga_overall_last5",
    "away_xg_overall_last5",
    "away_xga_overall_last5",
    "created_at",
]

ID_TARGET_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "result",
    "home_win",
    "is_draw",
    "away_win",
]

ELO_FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
]

ELO_AUDIT_COLUMNS = [
    "home_initialization",
    "away_initialization",
]

FORBIDDEN_POST_MATCH_ELO_COLUMNS = [
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
]

ELO_FEATURE_TABLE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *ELO_FEATURE_COLUMNS,
    *ELO_AUDIT_COLUMNS,
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
    "elo_ratings_v3",
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


def _verify_table_count(engine, table_name: str, expected_count: int) -> None:
    if not _table_exists(engine, table_name):
        raise RuntimeError(f"{table_name} table does not exist")
    row_count = _count_table_rows(engine, table_name)
    if row_count != expected_count:
        raise RuntimeError(
            f"{table_name} expected {expected_count} rows, found {row_count}"
        )
    print(f"{table_name} row count verified: {row_count}")


def load_base_features_with_elo(engine) -> pd.DataFrame:
    base_count = _count_table_rows(engine, "match_features_v3_base")
    elo_count = _count_table_rows(engine, "elo_ratings_v3")
    if base_count != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"match_features_v3_base expected {EXPECTED_ROW_COUNT} rows, found {base_count}"
        )
    if elo_count != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"elo_ratings_v3 expected {EXPECTED_ROW_COUNT} rows, found {elo_count}"
        )

    base_select = ",\n            ".join(f"b.{column}" for column in BASE_FEATURE_COLUMNS)
    elo_select = ",\n            ".join(f"e.{column}" for column in ELO_FEATURE_COLUMNS)
    audit_select = ",\n            ".join(f"e.{column}" for column in ELO_AUDIT_COLUMNS)
    query = text(
        f"""
        SELECT
            {base_select},
            {elo_select},
            {audit_select}
        FROM match_features_v3_base b
        INNER JOIN elo_ratings_v3 e
            ON b.match_id = e.match_id
        ORDER BY b.match_date, b.kickoff_time, b.match_id
        """
    )
    features_df = pd.read_sql(query, engine)

    errors: list[str] = []
    if len(features_df) != base_count:
        errors.append(
            f"joined row count {len(features_df)} != match_features_v3_base count {base_count}"
        )
    if len(features_df) != elo_count:
        errors.append(
            f"joined row count {len(features_df)} != elo_ratings_v3 count {elo_count}"
        )
    if features_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(features_df['match_id'].duplicated().sum())}"
        )

    missing_base_ids = _query_mappings(
        engine,
        """
        SELECT COUNT(*) AS missing_count
        FROM match_features_v3_base b
        LEFT JOIN elo_ratings_v3 e
            ON b.match_id = e.match_id
        WHERE e.match_id IS NULL
        """,
    )[0]["missing_count"]
    missing_elo_ids = _query_mappings(
        engine,
        """
        SELECT COUNT(*) AS missing_count
        FROM elo_ratings_v3 e
        LEFT JOIN match_features_v3_base b
            ON e.match_id = b.match_id
        WHERE b.match_id IS NULL
        """,
    )[0]["missing_count"]
    if missing_base_ids:
        errors.append(f"base rows missing Elo coverage: {missing_base_ids}")
    if missing_elo_ids:
        errors.append(f"Elo rows missing base coverage: {missing_elo_ids}")

    null_elo_counts = features_df[ELO_FEATURE_COLUMNS + ELO_AUDIT_COLUMNS].isna().sum()
    bad_elo_nulls = {
        column: int(count)
        for column, count in null_elo_counts.items()
        if int(count) > 0
    }
    if bad_elo_nulls:
        errors.append(f"null Elo feature/audit columns: {bad_elo_nulls}")

    null_id_counts = features_df[ID_TARGET_COLUMNS].isna().sum()
    bad_id_nulls = {
        column: int(count)
        for column, count in null_id_counts.items()
        if int(count) > 0
    }
    if bad_id_nulls:
        errors.append(f"null ID/target columns: {bad_id_nulls}")

    forbidden_present = [
        column for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS if column in features_df.columns
    ]
    if forbidden_present:
        errors.append(f"forbidden post-match Elo columns present: {forbidden_present}")

    if errors:
        raise ValueError("Base + Elo feature join validation failed: " + "; ".join(errors))

    print(
        "Base + pre-match Elo join coverage passed: "
        f"{len(features_df)} rows loaded"
    )
    return features_df


def validate_elo_feature_frame(features_df: pd.DataFrame) -> None:
    print("=== Elo Feature Frame Validation ===")
    errors: list[str] = []

    if len(features_df) != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} rows, found {len(features_df)}")
    if features_df["match_id"].nunique() != len(features_df):
        errors.append("feature frame does not have one row per match_id")
    if features_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(features_df['match_id'].duplicated().sum())}"
        )

    season_counts = features_df.groupby("season_id").size().sort_index()
    for season_id, expected_count in EXPECTED_SEASON_COUNTS.items():
        actual_count = int(season_counts.get(season_id, 0))
        if actual_count != expected_count:
            errors.append(
                f"{season_id} expected {expected_count} rows, found {actual_count}"
            )

    required_columns = ID_TARGET_COLUMNS + ELO_FEATURE_COLUMNS + ELO_AUDIT_COLUMNS
    null_counts = features_df[required_columns].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"null required feature columns: {bad_nulls}")

    expected_targets = pd.DataFrame(
        {
            "home_win": (features_df["result"] == "H").astype(int),
            "is_draw": (features_df["result"] == "D").astype(int),
            "away_win": (features_df["result"] == "A").astype(int),
        }
    )
    target_mismatches = (
        features_df[["home_win", "is_draw", "away_win"]] != expected_targets
    ).any(axis=1)
    if target_mismatches.any():
        errors.append(f"target/result mismatch count: {int(target_mismatches.sum())}")

    if not np.allclose(
        features_df["expected_home_score"] + features_df["expected_away_score"],
        1.0,
    ):
        errors.append("expected_home_score + expected_away_score check failed")
    if not np.allclose(
        features_df["elo_diff_before"],
        features_df["home_elo_before"] - features_df["away_elo_before"],
    ):
        errors.append("elo_diff_before formula check failed")
    if not np.allclose(
        features_df["elo_diff_home_adjusted"],
        features_df["home_elo_before"] + HOME_ADVANTAGE - features_df["away_elo_before"],
    ):
        errors.append("elo_diff_home_adjusted formula check failed")

    for column in ["expected_home_score", "expected_away_score"]:
        bad_count = (~features_df[column].between(0, 1)).sum()
        if bad_count:
            errors.append(f"{column} has {int(bad_count)} value(s) outside 0..1")

    forbidden_present = [
        column for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS if column in features_df.columns
    ]
    if forbidden_present:
        errors.append(f"forbidden post-match Elo columns present: {forbidden_present}")

    print("Initialization counts by season:")
    initialization_counts = (
        pd.concat(
            [
                features_df[["season_id", "home_team", "home_initialization"]].rename(
                    columns={
                        "home_team": "team",
                        "home_initialization": "initialization",
                    }
                ),
                features_df[["season_id", "away_team", "away_initialization"]].rename(
                    columns={
                        "away_team": "team",
                        "away_initialization": "initialization",
                    }
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .groupby(["season_id", "initialization"])
        .size()
        .sort_index()
    )
    for (season_id, initialization), count in initialization_counts.items():
        print(f"- {season_id} {initialization}: {int(count)}")

    print("Elo min/max/mean by season:")
    elo_summary = features_df.groupby("season_id").agg(
        min_home_elo_before=("home_elo_before", "min"),
        max_home_elo_before=("home_elo_before", "max"),
        mean_home_elo_before=("home_elo_before", "mean"),
        min_away_elo_before=("away_elo_before", "min"),
        max_away_elo_before=("away_elo_before", "max"),
        mean_away_elo_before=("away_elo_before", "mean"),
        min_elo_diff_before=("elo_diff_before", "min"),
        max_elo_diff_before=("elo_diff_before", "max"),
        mean_elo_diff_before=("elo_diff_before", "mean"),
    )
    for season_id, row in elo_summary.iterrows():
        print(
            f"- {season_id}: "
            f"home {row['min_home_elo_before']:.1f}/"
            f"{row['max_home_elo_before']:.1f}/"
            f"{row['mean_home_elo_before']:.1f}; "
            f"away {row['min_away_elo_before']:.1f}/"
            f"{row['max_away_elo_before']:.1f}/"
            f"{row['mean_away_elo_before']:.1f}; "
            f"diff {row['min_elo_diff_before']:.1f}/"
            f"{row['max_elo_diff_before']:.1f}/"
            f"{row['mean_elo_diff_before']:.1f}"
        )

    print("Elo feature null counts:")
    elo_null_counts = features_df[ELO_FEATURE_COLUMNS + ELO_AUDIT_COLUMNS].isna().sum()
    for column, count in elo_null_counts.items():
        print(f"- {column}: {int(count)}")

    if errors:
        print("Elo feature frame validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Elo feature frame validation failed")

    print("PASS: total rows and one row per match_id")
    print("PASS: season counts are 380 each")
    print("PASS: no null required ID/target/Elo feature columns")
    print("PASS: target columns match result")
    print("PASS: expected score sum check")
    print("PASS: Elo diff checks")
    print("PASS: expected scores are between 0 and 1")
    print("PASS: forbidden post-match Elo columns absent from DataFrame")
    print("Elo feature frame validation passed.")


def create_or_verify_elo_feature_table(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)

    if not _table_exists(engine, "match_features_v3_elo"):
        raise RuntimeError("match_features_v3_elo table does not exist")

    rows = _query_mappings(
        engine,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = CURRENT_SCHEMA()
            AND table_name = 'match_features_v3_elo'
        """,
    )
    existing_columns = {row["column_name"] for row in rows}
    missing_columns = sorted(set(ELO_FEATURE_TABLE_COLUMNS) - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_elo is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    forbidden_present = sorted(set(FORBIDDEN_POST_MATCH_ELO_COLUMNS) & existing_columns)
    if forbidden_present:
        raise RuntimeError(
            "match_features_v3_elo has forbidden post-match Elo column(s): "
            f"{', '.join(forbidden_present)}"
        )

    print("match_features_v3_elo schema verification passed.")
    print("Forbidden post-match Elo column check passed.")


def store_elo_features(features_df: pd.DataFrame, engine) -> None:
    records = [
        {column: _record_value(row[column]) for column in ELO_FEATURE_TABLE_COLUMNS}
        for row in features_df[ELO_FEATURE_TABLE_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(ELO_FEATURE_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in ELO_FEATURE_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO match_features_v3_elo (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM match_features_v3_elo"))
        conn.execute(insert_sql, records)

    print(f"Stored {len(records)} rows in match_features_v3_elo")


def print_elo_feature_summary(engine) -> None:
    with engine.connect() as conn:
        total_rows = conn.execute(
            text("SELECT COUNT(*) FROM match_features_v3_elo")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                """
                SELECT
                    season_id,
                    COUNT(*) AS row_count,
                    MIN(match_date) AS min_match_date,
                    MAX(match_date) AS max_match_date
                FROM match_features_v3_elo
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        null_select = ",\n                ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in ELO_FEATURE_COLUMNS + ELO_AUDIT_COLUMNS
        )
        null_counts = conn.execute(
            text(f"SELECT {null_select} FROM match_features_v3_elo")
        ).mappings().one()
        initialization_counts = conn.execute(
            text(
                """
                SELECT season_id, initialization, COUNT(DISTINCT team) AS team_count
                FROM (
                    SELECT season_id, home_team AS team, home_initialization AS initialization
                    FROM match_features_v3_elo
                    UNION
                    SELECT season_id, away_team AS team, away_initialization AS initialization
                    FROM match_features_v3_elo
                ) team_initializations
                GROUP BY season_id, initialization
                ORDER BY season_id, initialization
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
                    AVG(away_elo_before) AS mean_away_elo_before,
                    MIN(elo_diff_before) AS min_elo_diff_before,
                    MAX(elo_diff_before) AS max_elo_diff_before,
                    AVG(elo_diff_before) AS mean_elo_diff_before
                FROM match_features_v3_elo
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()

    print("=== match_features_v3_elo Summary ===")
    print(f"match_features_v3_elo total rows: {total_rows}")
    print("Rows by season:")
    for row in season_rows:
        print(
            f"- {row['season_id']}: {row['row_count']} rows, "
            f"{row['min_match_date']} to {row['max_match_date']}"
        )

    print("Elo feature null counts:")
    for column in ELO_FEATURE_COLUMNS + ELO_AUDIT_COLUMNS:
        print(f"- {column}: {null_counts[column]}")

    print("Initialization counts by season:")
    for row in initialization_counts:
        print(
            f"- {row['season_id']} {row['initialization']}: "
            f"{row['team_count']}"
        )

    print("Elo min/max/mean by season:")
    for row in elo_summary:
        print(
            f"- {row['season_id']}: "
            f"home {row['min_home_elo_before']:.1f}/"
            f"{row['max_home_elo_before']:.1f}/"
            f"{row['mean_home_elo_before']:.1f}; "
            f"away {row['min_away_elo_before']:.1f}/"
            f"{row['max_away_elo_before']:.1f}/"
            f"{row['mean_away_elo_before']:.1f}; "
            f"diff {row['min_elo_diff_before']:.1f}/"
            f"{row['max_elo_diff_before']:.1f}/"
            f"{row['mean_elo_diff_before']:.1f}"
        )


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
    print("Safety counts unchanged for Tier 2 and Tier 3 source/base tables.")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    print("=== Tier 3 Elo Feature Build ===")
    validate_historical_match_integrity(engine)
    _verify_table_count(engine, "match_features_v3_base", EXPECTED_ROW_COUNT)
    _verify_table_count(engine, "elo_ratings_v3", EXPECTED_ROW_COUNT)

    features_df = load_base_features_with_elo(engine)
    validate_elo_feature_frame(features_df)

    before_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts before Elo feature write", before_counts)

    create_or_verify_elo_feature_table(engine)
    store_elo_features(features_df, engine)

    after_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts after Elo feature write", after_counts)
    _verify_counts_unchanged(before_counts, after_counts)

    print_elo_feature_summary(engine)
    print("2025-26 is included in the feature table but remains reserved as final test.")
    print("No model training occurred.")
    print(
        "Tier 2 tables, match_features_v3_base, elo_ratings_v3, Streamlit, "
        "and model artifacts were not touched."
    )


if __name__ == "__main__":
    main()
