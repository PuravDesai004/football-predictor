from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data_pipeline import get_engine


EXPECTED_COMPLETED_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

REQUIRED_HISTORICAL_MATCH_COLUMNS = [
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "source_file",
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


def _query_scalar(engine, query: str, params: dict | None = None):
    with engine.connect() as conn:
        return conn.execute(text(query), params or {}).scalar_one()


def _query_mappings(engine, query: str, params: dict | None = None) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(query), params or {}).mappings().all()
    return [dict(row) for row in rows]


def get_historical_match_summary(engine) -> pd.DataFrame:
    rows = _query_mappings(
        engine,
        """
        SELECT
            hm.season_id,
            COUNT(*) AS match_count,
            (
                SELECT COUNT(DISTINCT team_name)
                FROM (
                    SELECT home_team AS team_name
                    FROM historical_matches home_matches
                    WHERE home_matches.season_id = hm.season_id
                    UNION
                    SELECT away_team AS team_name
                    FROM historical_matches away_matches
                    WHERE away_matches.season_id = hm.season_id
                ) teams
            ) AS unique_team_count,
            MIN(hm.match_date) AS min_match_date,
            MAX(hm.match_date) AS max_match_date
        FROM historical_matches hm
        GROUP BY hm.season_id
        ORDER BY MIN(hm.match_date), hm.season_id
        """,
    )
    return pd.DataFrame(
        rows,
        columns=[
            "season_id",
            "match_count",
            "unique_team_count",
            "min_match_date",
            "max_match_date",
        ],
    )


def validate_historical_match_integrity(engine) -> None:
    print("=== Historical Match Integrity ===")
    errors: list[str] = []

    if not _table_exists(engine, "historical_matches"):
        raise RuntimeError("FAIL: historical_matches table does not exist")
    print("PASS: historical_matches table exists")

    row_count = _query_scalar(engine, "SELECT COUNT(*) FROM historical_matches")
    if row_count <= 0:
        errors.append("historical_matches row count is 0")
    else:
        print(f"PASS: historical_matches row count is {row_count}")

    duplicate_count = _query_scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM (
            SELECT season_id, home_team, away_team
            FROM historical_matches
            GROUP BY season_id, home_team, away_team
            HAVING COUNT(*) > 1
        ) duplicate_groups
        """,
    )
    if duplicate_count:
        errors.append(
            f"found {duplicate_count} duplicate season/home/away group(s)"
        )
    else:
        print("PASS: no duplicate season_id/home_team/away_team rows")

    null_counts = _query_mappings(
        engine,
        """
        SELECT
            SUM(CASE WHEN season_id IS NULL THEN 1 ELSE 0 END) AS season_id,
            SUM(CASE WHEN match_date IS NULL THEN 1 ELSE 0 END) AS match_date,
            SUM(CASE WHEN home_team IS NULL THEN 1 ELSE 0 END) AS home_team,
            SUM(CASE WHEN away_team IS NULL THEN 1 ELSE 0 END) AS away_team,
            SUM(CASE WHEN home_goals IS NULL THEN 1 ELSE 0 END) AS home_goals,
            SUM(CASE WHEN away_goals IS NULL THEN 1 ELSE 0 END) AS away_goals,
            SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) AS result,
            SUM(CASE WHEN source_file IS NULL THEN 1 ELSE 0 END) AS source_file
        FROM historical_matches
        """,
    )[0]
    null_errors = {
        column: count
        for column, count in null_counts.items()
        if count is not None and count > 0
    }
    if null_errors:
        errors.append(f"required-column nulls found: {null_errors}")
    else:
        print(
            "PASS: no nulls in required columns "
            f"({', '.join(REQUIRED_HISTORICAL_MATCH_COLUMNS)})"
        )

    same_team_count = _query_scalar(
        engine,
        "SELECT COUNT(*) FROM historical_matches WHERE home_team = away_team",
    )
    if same_team_count:
        errors.append(f"found {same_team_count} row(s) where home_team = away_team")
    else:
        print("PASS: no home_team = away_team rows")

    result_mismatch_count = _query_scalar(
        engine,
        """
        SELECT COUNT(*)
        FROM historical_matches
        WHERE
            (result = 'H' AND home_goals <= away_goals)
            OR (result = 'A' AND away_goals <= home_goals)
            OR (result = 'D' AND home_goals <> away_goals)
        """,
    )
    if result_mismatch_count:
        errors.append(f"found {result_mismatch_count} result/goals mismatch row(s)")
    else:
        print("PASS: result matches home_goals and away_goals")

    summary_df = get_historical_match_summary(engine)
    summary_by_season = {
        row.season_id: row
        for row in summary_df.itertuples(index=False)
    }
    for season_id in EXPECTED_COMPLETED_SEASONS:
        if season_id not in summary_by_season:
            continue

        row = summary_by_season[season_id]
        if row.match_count != 380 or row.unique_team_count != 20:
            errors.append(
                f"{season_id} expected 380 matches and 20 teams, found "
                f"{row.match_count} matches and {row.unique_team_count} teams"
            )
        else:
            print(f"PASS: {season_id} has 380 matches and 20 teams")

    if errors:
        print("Historical integrity validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Historical match integrity validation failed")

    print("Historical integrity validation passed.")


def get_ordered_seasons(engine) -> list[str]:
    rows = _query_mappings(
        engine,
        """
        SELECT season_id
        FROM historical_matches
        GROUP BY season_id
        ORDER BY MIN(match_date), season_id
        """,
    )
    return [row["season_id"] for row in rows]


def split_development_and_final_test(
    seasons: list[str],
    final_test_season: str | None = None,
) -> dict:
    if not seasons:
        raise ValueError("No seasons available for development/final-test split")

    if final_test_season is None:
        final_test_season = seasons[-1]

    if final_test_season not in seasons:
        raise ValueError(f"Final test season not found: {final_test_season}")

    final_test_index = seasons.index(final_test_season)
    development_seasons = seasons[:final_test_index]
    if not development_seasons:
        raise ValueError("No development seasons remain before final test season")

    return {
        "development_seasons": development_seasons,
        "final_test_season": final_test_season,
    }


def build_walk_forward_season_splits(
    development_seasons: list[str],
    min_train_seasons: int = 2,
    validation_window: int = 1,
) -> list[dict]:
    if min_train_seasons <= 0:
        raise ValueError("min_train_seasons must be positive")
    if validation_window <= 0:
        raise ValueError("validation_window must be positive")

    splits: list[dict] = []
    fold_number = 1
    max_start = len(development_seasons) - validation_window
    for validation_start in range(min_train_seasons, max_start + 1):
        train_seasons = development_seasons[:validation_start]
        validation_seasons = development_seasons[
            validation_start : validation_start + validation_window
        ]
        if len(validation_seasons) < validation_window:
            continue
        splits.append(
            {
                "fold": fold_number,
                "train_seasons": train_seasons,
                "validation_seasons": validation_seasons,
            }
        )
        fold_number += 1

    return splits


def _season_window_stats(engine, seasons: list[str]) -> dict:
    if not seasons:
        raise ValueError("At least one season is required for date window checks")

    rows = _query_mappings(
        engine,
        """
        SELECT
            COUNT(*) AS row_count,
            MIN(match_date) AS min_match_date,
            MAX(match_date) AS max_match_date
        FROM historical_matches
        WHERE season_id = ANY(:seasons)
        """,
        {"seasons": seasons},
    )
    return rows[0]


def validate_walk_forward_splits(engine, splits: list[dict]) -> None:
    print("=== Walk-Forward Splits ===")
    if not splits:
        raise ValueError("No walk-forward splits were built")

    ordered_seasons = get_ordered_seasons(engine)
    season_order = {season_id: index for index, season_id in enumerate(ordered_seasons)}

    for split in splits:
        fold = split["fold"]
        train_seasons = split["train_seasons"]
        validation_seasons = split["validation_seasons"]

        overlap = sorted(set(train_seasons) & set(validation_seasons))
        if overlap:
            raise ValueError(f"Fold {fold} train/validation overlap: {overlap}")

        latest_train_order = max(season_order[season] for season in train_seasons)
        earliest_validation_order = min(
            season_order[season] for season in validation_seasons
        )
        if latest_train_order >= earliest_validation_order:
            raise ValueError(
                f"Fold {fold} training seasons are not earlier than validation seasons"
            )

        train_stats = _season_window_stats(engine, train_seasons)
        validation_stats = _season_window_stats(engine, validation_seasons)
        if train_stats["row_count"] <= 0:
            raise ValueError(f"Fold {fold} has no training rows")
        if validation_stats["row_count"] <= 0:
            raise ValueError(f"Fold {fold} has no validation rows")
        if train_stats["max_match_date"] >= validation_stats["min_match_date"]:
            raise ValueError(
                f"Fold {fold} has date leakage: max train date "
                f"{train_stats['max_match_date']} >= min validation date "
                f"{validation_stats['min_match_date']}"
            )

        print(f"Fold {fold}")
        print(f"- Train seasons: {', '.join(train_seasons)}")
        print(f"- Validate seasons: {', '.join(validation_seasons)}")
        print(
            f"- Train rows: {train_stats['row_count']}, "
            f"max train date: {train_stats['max_match_date']}"
        )
        print(
            f"- Validation rows: {validation_stats['row_count']}, "
            f"min validation date: {validation_stats['min_match_date']}"
        )

    print("Walk-forward leakage checks passed.")


def validate_final_test_holdout(
    engine,
    development_seasons: list[str],
    final_test_season: str,
) -> None:
    print("=== Final Test Holdout ===")
    if final_test_season in development_seasons:
        raise ValueError(
            f"Final test season {final_test_season} is included in development seasons"
        )

    development_stats = _season_window_stats(engine, development_seasons)
    final_test_stats = _season_window_stats(engine, [final_test_season])
    if development_stats["row_count"] <= 0:
        raise ValueError("Development window has no rows")
    if final_test_stats["row_count"] <= 0:
        raise ValueError("Final test window has no rows")
    if development_stats["max_match_date"] >= final_test_stats["min_match_date"]:
        raise ValueError(
            "Final test holdout has date leakage: max development date "
            f"{development_stats['max_match_date']} >= min final test date "
            f"{final_test_stats['min_match_date']}"
        )

    print(f"Development seasons: {', '.join(development_seasons)}")
    print(f"Reserved final test season: {final_test_season}")
    print(f"Development rows: {development_stats['row_count']}")
    print(f"Final test rows: {final_test_stats['row_count']}")
    print(f"Max development date: {development_stats['max_match_date']}")
    print(f"Min final test date: {final_test_stats['min_match_date']}")
    print("Final test holdout validation passed.")


def _print_summary(summary_df: pd.DataFrame) -> None:
    print("=== Season Summary ===")
    if summary_df.empty:
        print("No historical match seasons found.")
        return

    for row in summary_df.itertuples(index=False):
        print(
            f"{row.season_id}: {row.match_count} matches, "
            f"{row.unique_team_count} teams, "
            f"{row.min_match_date} to {row.max_match_date}"
        )


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    validate_historical_match_integrity(engine)
    summary_df = get_historical_match_summary(engine)
    _print_summary(summary_df)

    ordered_seasons = get_ordered_seasons(engine)
    print(f"Ordered seasons: {', '.join(ordered_seasons)}")

    split_config = split_development_and_final_test(ordered_seasons)
    development_seasons = split_config["development_seasons"]
    final_test_season = split_config["final_test_season"]
    print(f"Development seasons: {', '.join(development_seasons)}")
    print(f"Reserved final test season: {final_test_season}")

    splits = build_walk_forward_season_splits(development_seasons)
    validate_walk_forward_splits(engine, splits)
    validate_final_test_holdout(engine, development_seasons, final_test_season)

    print("No database writes or model training occurred.")


if __name__ == "__main__":
    main()
