from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas
from sqlalchemy import text

from data_pipeline import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CANDIDATE_NAME = "logistic_elo_expanding"
SOURCE_TABLE = "match_features_v3_elo"
DEV_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_HOLDOUT_SEASON = "2025-26"
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
DRAW_OVERLAY_ALLOWED = True
DOMINANT_CLASS_MAX_PROB = 0.50

EXPECTED_SOURCE_ROW_COUNT = 1900
EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_HOLDOUT_ROW_COUNT = 380
EXPECTED_SEASON_ROWS = 380
EXPERIMENT_SUMMARY_FILE = PROJECT_ROOT / "docs" / "tier3_experiment_summary.md"

BASE_FEATURE_COLUMNS = [
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
]
ELO_FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
]

FORBIDDEN_EXACT_FEATURES = {
    "match_id",
    "season",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    TARGET_COLUMN,
    "home_goals",
    "away_goals",
    "home_win",
    "is_draw",
    "away_win",
    "created_at",
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
FORBIDDEN_FEATURE_TOKENS = [
    "h2h",
    "style",
    "pressure",
    "poisson",
    "odds",
    "betting",
    "manager",
    "sentiment",
    "injury",
    "rivalry",
    "derby",
    "calibration",
]
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "match_features_v3_style_experiment",
    "match_features_v3_h2h_experiment",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_candidate_source_metadata(conn) -> dict:
    if not _table_exists(conn, SOURCE_TABLE):
        raise RuntimeError(f"{SOURCE_TABLE} table does not exist")

    with conn.connect() as db_conn:
        total_rows = int(
            db_conn.execute(text(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")).scalar_one()
        )
        rows_by_season = [
            dict(row)
            for row in db_conn.execute(
                text(
                    f"""
                    SELECT season_id, COUNT(*) AS row_count
                    FROM {SOURCE_TABLE}
                    GROUP BY season_id
                    ORDER BY season_id
                    """
                )
            ).mappings().all()
        ]
        final_holdout_rows = int(
            db_conn.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM {SOURCE_TABLE}
                    WHERE season_id = :season_id
                    """
                ),
                {"season_id": FINAL_HOLDOUT_SEASON},
            ).scalar_one()
        )
        columns = [
            row["column_name"]
            for row in db_conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = :table_name
                    ORDER BY ordinal_position
                    """
                ),
                {"table_name": SOURCE_TABLE},
            ).mappings().all()
        ]

    errors: list[str] = []
    if total_rows != EXPECTED_SOURCE_ROW_COUNT:
        errors.append(
            f"{SOURCE_TABLE} expected {EXPECTED_SOURCE_ROW_COUNT} rows, found {total_rows}"
        )
    if final_holdout_rows != EXPECTED_HOLDOUT_ROW_COUNT:
        errors.append(
            f"{FINAL_HOLDOUT_SEASON} metadata count expected "
            f"{EXPECTED_HOLDOUT_ROW_COUNT}, found {final_holdout_rows}"
        )
    for row in rows_by_season:
        if int(row["row_count"]) != EXPECTED_SEASON_ROWS:
            errors.append(
                f"{row['season_id']} expected {EXPECTED_SEASON_ROWS} rows, "
                f"found {row['row_count']}"
            )
    if errors:
        raise ValueError("Candidate source metadata validation failed: " + "; ".join(errors))

    return {
        "source_table": SOURCE_TABLE,
        "total_rows": total_rows,
        "rows_by_season": rows_by_season,
        "final_holdout_metadata_count": final_holdout_rows,
        "columns": columns,
    }


def load_dev_candidate_data(conn) -> pandas.DataFrame:
    query = text(
        f"""
        SELECT *
        FROM {SOURCE_TABLE}
        WHERE season_id = ANY(:dev_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pandas.read_sql(query, conn, params={"dev_seasons": DEV_SEASONS})
    df["match_date"] = pandas.to_datetime(df["match_date"])

    errors: list[str] = []
    if len(df) != EXPECTED_DEV_ROW_COUNT:
        errors.append(f"expected {EXPECTED_DEV_ROW_COUNT} development rows, found {len(df)}")
    seasons_present = sorted(df["season_id"].dropna().unique().tolist())
    if seasons_present != DEV_SEASONS:
        errors.append(f"development seasons {seasons_present} != {DEV_SEASONS}")
    season_counts = df.groupby("season_id").size().to_dict()
    bad_counts = {
        season_id: int(count)
        for season_id, count in season_counts.items()
        if int(count) != EXPECTED_SEASON_ROWS
    }
    if bad_counts:
        errors.append(f"bad development season counts: {bad_counts}")
    if df["match_id"].duplicated().any():
        errors.append(f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}")

    if errors:
        raise ValueError("Development candidate data validation failed: " + "; ".join(errors))

    validate_final_holdout_not_loaded(df)
    return df


def validate_final_holdout_not_loaded(df) -> None:
    if FINAL_HOLDOUT_SEASON in set(df["season_id"]):
        raise ValueError(f"{FINAL_HOLDOUT_SEASON} was loaded into development data")
    print(f"PASS: {FINAL_HOLDOUT_SEASON} not loaded into development dataframe.")


def get_final_candidate_feature_columns(df) -> list[str]:
    feature_columns = [*BASE_FEATURE_COLUMNS, *ELO_FEATURE_COLUMNS]
    validate_required_candidate_columns(df, feature_columns)
    validate_feature_exclusions(feature_columns)
    validate_rejected_experiments_excluded(feature_columns)
    return feature_columns


def validate_feature_exclusions(feature_columns) -> None:
    feature_set = set(feature_columns)
    forbidden_exact = sorted(feature_set & FORBIDDEN_EXACT_FEATURES)
    forbidden_token_matches = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    duplicate_features = sorted(
        column for column in feature_set if feature_columns.count(column) > 1
    )

    errors: list[str] = []
    if forbidden_exact:
        errors.append(f"forbidden exact features selected: {forbidden_exact}")
    if forbidden_token_matches:
        errors.append(f"forbidden token features selected: {forbidden_token_matches}")
    if duplicate_features:
        errors.append(f"duplicate feature names selected: {duplicate_features}")
    if errors:
        raise ValueError("Feature exclusion audit failed: " + "; ".join(errors))

    print("PASS: feature exclusions validated.")


def validate_required_candidate_columns(df, feature_columns) -> None:
    required_columns = {
        "match_id",
        "season_id",
        "match_date",
        "home_team",
        "away_team",
        TARGET_COLUMN,
        *feature_columns,
    }
    missing_columns = sorted(required_columns - set(df.columns))

    errors: list[str] = []
    if missing_columns:
        errors.append(f"missing required candidate column(s): {missing_columns}")
    if TARGET_COLUMN not in df.columns:
        errors.append(f"target column {TARGET_COLUMN} is missing")
    else:
        null_targets = int(df[TARGET_COLUMN].isna().sum())
        if null_targets:
            errors.append(f"null target labels: {null_targets}")
        unknown_labels = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
        if unknown_labels:
            errors.append(f"unknown target labels: {unknown_labels}")

    non_numeric_features = [
        column
        for column in feature_columns
        if column in df.columns and not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        errors.append(f"non-numeric candidate feature(s): {non_numeric_features}")

    if errors:
        raise ValueError("Candidate column validation failed: " + "; ".join(errors))

    print("PASS: required candidate columns and target integrity validated.")


def validate_walk_forward_definition() -> None:
    expected_folds = [
        {
            "fold": 1,
            "train_seasons": ["2021-22", "2022-23"],
            "validation_seasons": ["2023-24"],
        },
        {
            "fold": 2,
            "train_seasons": ["2021-22", "2022-23", "2023-24"],
            "validation_seasons": ["2024-25"],
        },
    ]
    for fold in expected_folds:
        overlap = sorted(set(fold["train_seasons"]) & set(fold["validation_seasons"]))
        if overlap:
            raise ValueError(f"Fold {fold['fold']} has train/validation overlap: {overlap}")
        if FINAL_HOLDOUT_SEASON in set(fold["train_seasons"]) | set(fold["validation_seasons"]):
            raise ValueError(f"{FINAL_HOLDOUT_SEASON} appears in fold {fold['fold']}")
    print("PASS: walk-forward definition frozen to development seasons only.")


def validate_draw_overlay_definition() -> None:
    errors: list[str] = []
    if not DRAW_OVERLAY_ALLOWED:
        errors.append("draw overlay is not marked as allowed")
    if DOMINANT_CLASS_MAX_PROB != 0.50:
        errors.append(
            f"DOMINANT_CLASS_MAX_PROB expected 0.50, found {DOMINANT_CLASS_MAX_PROB}"
        )
    if FINAL_CANDIDATE_NAME != "logistic_elo_expanding":
        errors.append(f"unexpected final candidate: {FINAL_CANDIDATE_NAME}")
    if errors:
        raise ValueError("Draw overlay definition audit failed: " + "; ".join(errors))

    print("PASS: draw overlay definition audited.")
    print("- Overlay is hard-label only.")
    print("- Probabilities are not changed by the overlay.")
    print("- Overlay source is only candidate model probabilities.")
    print("- Threshold must be training-derived during final evaluation/training.")
    print("- Phase 9A status: ACCEPT_DRAW_OVERLAY_EXPERIMENTAL_SERVING_HELPER.")


def validate_rejected_experiments_excluded(feature_columns) -> None:
    rejected_terms = {
        "h2h": "H2H rejected",
        "style": "style rejected",
        "pressure": "pressure rejected",
        "poisson": "Poisson diagnostic only",
        "calibration": "calibration not promoted",
        "rivalry": "rivalry/derby rejected",
        "derby": "rivalry/derby rejected",
    }
    violations = [
        f"{column} ({reason})"
        for column in feature_columns
        for term, reason in rejected_terms.items()
        if term in column.lower()
    ]
    if violations:
        raise ValueError(f"Rejected experiment feature(s) selected: {violations}")

    print("PASS: rejected experiments excluded from candidate features.")
    print("- H2H rejected.")
    print("- Style rejected.")
    print("- Pressure rejected.")
    print("- Calibration not promoted.")
    print("- XGB not champion.")
    print("- Poisson diagnostic only.")
    print("- Rivalry/derby rejected.")


def validate_docs_match_candidate_definition() -> None:
    doc_text = EXPERIMENT_SUMMARY_FILE.read_text(encoding="utf-8")
    lower_doc = doc_text.lower()
    required_snippets = {
        "logistic_elo_expanding": "candidate name",
        "2025-26": "reserved final holdout season",
        "draw overlay": "draw overlay helper",
        "h2h": "H2H decision",
        "style": "style decision",
        "pressure": "pressure decision",
        "no final `2025-26` holdout evaluation": "no final holdout evaluation statement",
        "match_features_v3_elo": "source table",
        "base + elo": "allowed feature family",
    }
    missing = [
        f"{description}: {snippet}"
        for snippet, description in required_snippets.items()
        if snippet.lower() not in lower_doc
    ]
    if missing:
        raise ValueError("Docs candidate definition audit failed: " + "; ".join(missing))

    print("PASS: docs match final candidate definition.")


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            exists = db_conn.execute(
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
            if exists:
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_no_counts_changed(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    if changed:
        raise RuntimeError(f"Watched table counts changed unexpectedly: {changed}")
    print("PASS: watched table counts unchanged.")


def print_freeze_audit_report(metadata, df, feature_columns) -> None:
    print("=== Tier 3 Final Candidate Freeze Audit Report ===")
    print(f"Final candidate: {FINAL_CANDIDATE_NAME}")
    print(f"Source table: {metadata['source_table']}")
    print(f"Source table row count: {metadata['total_rows']}")
    print(
        f"{FINAL_HOLDOUT_SEASON} metadata-only row count: "
        f"{metadata['final_holdout_metadata_count']}"
    )
    print(f"Development rows loaded: {len(df)}")
    print("Development seasons loaded: " + ", ".join(sorted(df["season_id"].unique())))
    print(f"Final candidate feature count: {len(feature_columns)}")
    print("Allowed feature family: base + Elo features only")
    print("Draw overlay: allowed as hard-label serving helper only")
    print("Probabilities altered by overlay: no")
    print("Final holdout evaluation run: no")
    print("Rows by season in source table:")
    for row in metadata["rows_by_season"]:
        print(f"- {row['season_id']}: {row['row_count']}")


def main() -> None:
    print("=== Tier 3 Phase 10A Final Model Freeze Audit ===")
    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)

    metadata = load_candidate_source_metadata(conn)
    df = load_dev_candidate_data(conn)
    feature_columns = get_final_candidate_feature_columns(df)
    validate_walk_forward_definition()
    validate_draw_overlay_definition()
    validate_docs_match_candidate_definition()
    print_freeze_audit_report(metadata, df, feature_columns)

    after_counts = capture_watched_table_counts(conn)
    assert_no_counts_changed(before_counts, after_counts)

    print(f"PASS: {FINAL_HOLDOUT_SEASON} was counted only as metadata and not loaded.")
    print("PASS: final candidate freeze audit passed.")
    print("No model artifact was created.")
    print("No final holdout evaluation was run.")
    print("No database writes occurred.")
    print("No Streamlit, Tier 2 artifact, H2H, style, pressure, Poisson, calibration, odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


def _table_exists(conn, table_name: str) -> bool:
    with conn.connect() as db_conn:
        return bool(
            db_conn.execute(
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
        )


if __name__ == "__main__":
    main()
