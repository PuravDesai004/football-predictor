from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from data_pipeline import get_engine
from tier3_validation import validate_historical_match_integrity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

TIER2_COUNT_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]

SOURCE_TABLES = ["historical_matches", "historical_understat_xg"]

ID_TARGET_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_win",
    "is_draw",
    "away_win",
]

COUNT_COLUMNS = [
    "home_home_matches_last5",
    "away_away_matches_last5",
    "home_overall_matches_last5",
    "away_overall_matches_last5",
    "home_overall_matches_last10",
    "away_overall_matches_last10",
]

FEATURE_COLUMNS = [
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

NON_NEGATIVE_FEATURE_COLUMNS = [
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
    "home_xg_overall_last5",
    "home_xga_overall_last5",
    "away_xg_overall_last5",
    "away_xga_overall_last5",
]

FEATURE_TABLE_COLUMNS = [
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
    *COUNT_COLUMNS,
    *FEATURE_COLUMNS,
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


def capture_table_counts(engine, tables: list[str], label: str) -> dict[str, int | str]:
    print(f"=== {label} ===")
    counts: dict[str, int | str] = {}
    for table_name in tables:
        if _table_exists(engine, table_name):
            counts[table_name] = _count_table_rows(engine, table_name)
        else:
            counts[table_name] = "MISSING"
        print(f"{table_name}: {counts[table_name]}")
    return counts


def verify_counts_unchanged(
    before_counts: dict[str, int | str],
    after_counts: dict[str, int | str],
    label: str,
) -> None:
    if before_counts != after_counts:
        changed = {
            table_name: (before_counts.get(table_name), after_counts.get(table_name))
            for table_name in sorted(set(before_counts) | set(after_counts))
            if before_counts.get(table_name) != after_counts.get(table_name)
        }
        raise RuntimeError(f"{label} changed unexpectedly: {changed}")
    print(f"{label} unchanged.")


def _prepare_match_times(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = df.copy()
    prepared_df["match_date"] = pd.to_datetime(prepared_df["match_date"]).dt.date
    prepared_df["match_date_ts"] = pd.to_datetime(prepared_df["match_date"])
    prepared_df["kickoff_time"] = pd.to_datetime(
        prepared_df["kickoff_time"],
        errors="coerce",
    )
    prepared_df["event_time"] = prepared_df["kickoff_time"].fillna(
        prepared_df["match_date_ts"]
    )
    return prepared_df


def load_historical_matches_with_xg(engine) -> pd.DataFrame:
    historical_count = _count_table_rows(engine, "historical_matches")
    query = """
        SELECT
            hm.match_id,
            hm.season_id,
            hm.match_date,
            hm.kickoff_time,
            hm.home_team,
            hm.away_team,
            hm.home_goals,
            hm.away_goals,
            hm.result,
            ux.home_xg,
            ux.away_xg
        FROM historical_matches hm
        INNER JOIN historical_understat_xg ux
            ON hm.season_id = ux.season_id
            AND hm.match_date = ux.match_date
            AND hm.home_team = ux.home_team
            AND hm.away_team = ux.away_team
        ORDER BY hm.match_date, hm.kickoff_time, hm.match_id
    """
    matches_df = pd.read_sql(text(query), engine)
    matches_df = _prepare_match_times(matches_df)

    errors: list[str] = []
    if len(matches_df) != historical_count:
        errors.append(
            f"joined row count {len(matches_df)} != historical_matches count "
            f"{historical_count}"
        )
    if matches_df[["home_xg", "away_xg"]].isna().any().any():
        errors.append("joined data has null xG values")
    if matches_df["match_id"].duplicated().any():
        errors.append("joined data has duplicate match_id values")
    if matches_df.duplicated(
        subset=["season_id", "match_date", "home_team", "away_team"]
    ).any():
        errors.append("joined data has duplicate season/date/home/away keys")

    if errors:
        raise ValueError(
            "Historical match + xG join validation failed: " + "; ".join(errors)
        )

    print(
        "Historical match + xG join validation passed: "
        f"{len(matches_df)} rows loaded"
    )
    return matches_df


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_team_perspective_rows(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in matches_df.sort_values(["event_time", "match_id"]).itertuples(
        index=False
    ):
        rows.append(
            {
                "match_id": match.match_id,
                "team": match.home_team,
                "opponent": match.away_team,
                "venue": "H",
                "match_date": match.match_date,
                "kickoff_time": match.kickoff_time,
                "event_time": match.event_time,
                "goals_for": match.home_goals,
                "goals_against": match.away_goals,
                "xg_for": match.home_xg,
                "xg_against": match.away_xg,
                "points": _points_for(match.home_goals, match.away_goals),
            }
        )
        rows.append(
            {
                "match_id": match.match_id,
                "team": match.away_team,
                "opponent": match.home_team,
                "venue": "A",
                "match_date": match.match_date,
                "kickoff_time": match.kickoff_time,
                "event_time": match.event_time,
                "goals_for": match.away_goals,
                "goals_against": match.home_goals,
                "xg_for": match.away_xg,
                "xg_against": match.home_xg,
                "points": _points_for(match.away_goals, match.home_goals),
            }
        )

    team_rows = pd.DataFrame(rows)
    team_rows["event_time"] = pd.to_datetime(team_rows["event_time"])
    print(f"Built {len(team_rows)} team-perspective rows")
    return team_rows


def get_prior_team_matches(
    team_rows: pd.DataFrame,
    team: str,
    current_event_time,
    venue: str | None = None,
) -> pd.DataFrame:
    event_time = pd.Timestamp(current_event_time)
    prior_rows = team_rows.loc[
        (team_rows["team"] == team) & (team_rows["event_time"] < event_time)
    ]
    if venue is not None:
        prior_rows = prior_rows.loc[prior_rows["venue"] == venue]
    return prior_rows.sort_values(
        ["event_time", "match_id"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _mean_or_none(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(series.mean())


def _clean_sheet_rate(prior_rows: pd.DataFrame) -> float | None:
    if len(prior_rows) == 0:
        return None
    return float((prior_rows["goals_against"] == 0).mean())


def _goal_diff_mean(prior_rows: pd.DataFrame) -> float | None:
    if len(prior_rows) == 0:
        return None
    return float((prior_rows["goals_for"] - prior_rows["goals_against"]).mean())


def build_match_features(
    matches_df: pd.DataFrame,
    team_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sorted_matches = matches_df.sort_values(["event_time", "match_id"]).reset_index(
        drop=True
    )

    for match in sorted_matches.itertuples(index=False):
        home_home_last5 = get_prior_team_matches(
            team_rows,
            match.home_team,
            match.event_time,
            venue="H",
        ).head(5)
        away_away_last5 = get_prior_team_matches(
            team_rows,
            match.away_team,
            match.event_time,
            venue="A",
        ).head(5)
        home_overall_prior = get_prior_team_matches(
            team_rows,
            match.home_team,
            match.event_time,
        )
        away_overall_prior = get_prior_team_matches(
            team_rows,
            match.away_team,
            match.event_time,
        )
        home_overall_last5 = home_overall_prior.head(5)
        away_overall_last5 = away_overall_prior.head(5)
        home_overall_last10 = home_overall_prior.head(10)
        away_overall_last10 = away_overall_prior.head(10)

        rows.append(
            {
                "match_id": match.match_id,
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
                "home_win": 1 if match.result == "H" else 0,
                "is_draw": 1 if match.result == "D" else 0,
                "away_win": 1 if match.result == "A" else 0,
                "home_home_matches_last5": len(home_home_last5),
                "away_away_matches_last5": len(away_away_last5),
                "home_overall_matches_last5": len(home_overall_last5),
                "away_overall_matches_last5": len(away_overall_last5),
                "home_overall_matches_last10": len(home_overall_last10),
                "away_overall_matches_last10": len(away_overall_last10),
                "home_goals_scored_home_last5": _mean_or_none(
                    home_home_last5["goals_for"]
                ),
                "home_goals_conceded_home_last5": _mean_or_none(
                    home_home_last5["goals_against"]
                ),
                "home_clean_sheet_rate_home_last5": _clean_sheet_rate(home_home_last5),
                "away_goals_scored_away_last5": _mean_or_none(
                    away_away_last5["goals_for"]
                ),
                "away_goals_conceded_away_last5": _mean_or_none(
                    away_away_last5["goals_against"]
                ),
                "away_clean_sheet_rate_away_last5": _clean_sheet_rate(away_away_last5),
                "home_xg_home_last5": _mean_or_none(home_home_last5["xg_for"]),
                "home_xga_home_last5": _mean_or_none(home_home_last5["xg_against"]),
                "away_xg_away_last5": _mean_or_none(away_away_last5["xg_for"]),
                "away_xga_away_last5": _mean_or_none(away_away_last5["xg_against"]),
                "home_points_overall_last5": _mean_or_none(
                    home_overall_last5["points"]
                ),
                "away_points_overall_last5": _mean_or_none(
                    away_overall_last5["points"]
                ),
                "home_points_overall_last10": _mean_or_none(
                    home_overall_last10["points"]
                ),
                "away_points_overall_last10": _mean_or_none(
                    away_overall_last10["points"]
                ),
                "home_goal_diff_overall_last5": _goal_diff_mean(home_overall_last5),
                "away_goal_diff_overall_last5": _goal_diff_mean(away_overall_last5),
                "home_xg_overall_last5": _mean_or_none(home_overall_last5["xg_for"]),
                "home_xga_overall_last5": _mean_or_none(
                    home_overall_last5["xg_against"]
                ),
                "away_xg_overall_last5": _mean_or_none(away_overall_last5["xg_for"]),
                "away_xga_overall_last5": _mean_or_none(
                    away_overall_last5["xg_against"]
                ),
            }
        )

    features_df = pd.DataFrame(rows, columns=FEATURE_TABLE_COLUMNS)
    print(f"Built {len(features_df)} match feature rows")
    return features_df


def validate_match_features(
    features_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    team_rows: pd.DataFrame,
) -> None:
    print("=== Feature Validation ===")
    errors: list[str] = []

    if len(features_df) != len(matches_df):
        errors.append(
            f"feature row count {len(features_df)} != match row count {len(matches_df)}"
        )
    if features_df["match_id"].duplicated().any():
        errors.append("duplicate match_id values found")
    if features_df["match_id"].nunique() != len(features_df):
        errors.append("feature table does not have one row per match_id")

    null_id_target_counts = features_df[ID_TARGET_COLUMNS].isna().sum()
    bad_nulls = null_id_target_counts[null_id_target_counts > 0]
    if not bad_nulls.empty:
        errors.append(f"nulls in ID/target columns: {bad_nulls.to_dict()}")

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

    for column in [
        "home_home_matches_last5",
        "away_away_matches_last5",
        "home_overall_matches_last5",
        "away_overall_matches_last5",
    ]:
        bad_count = (~features_df[column].between(0, 5)).sum()
        if bad_count > 0:
            errors.append(f"{column} has {int(bad_count)} value(s) outside 0..5")

    for column in ["home_overall_matches_last10", "away_overall_matches_last10"]:
        bad_count = (~features_df[column].between(0, 10)).sum()
        if bad_count > 0:
            errors.append(f"{column} has {int(bad_count)} value(s) outside 0..10")

    for column in NON_NEGATIVE_FEATURE_COLUMNS:
        bad_count = (features_df[column].dropna() < 0).sum()
        if bad_count > 0:
            errors.append(f"{column} has {int(bad_count)} negative value(s)")

    for column in [
        "home_clean_sheet_rate_home_last5",
        "away_clean_sheet_rate_away_last5",
    ]:
        non_null = features_df[column].dropna()
        bad_count = (~non_null.between(0, 1)).sum()
        if bad_count > 0:
            errors.append(f"{column} has {int(bad_count)} value(s) outside 0..1")

    matches_by_id = matches_df.set_index("match_id")
    leakage_errors = 0
    count_mismatches = 0
    for feature_row in features_df.itertuples(index=False):
        match = matches_by_id.loc[feature_row.match_id]
        current_event_time = match["event_time"]
        windows = {
            "home_home_matches_last5": get_prior_team_matches(
                team_rows,
                feature_row.home_team,
                current_event_time,
                venue="H",
            ).head(5),
            "away_away_matches_last5": get_prior_team_matches(
                team_rows,
                feature_row.away_team,
                current_event_time,
                venue="A",
            ).head(5),
            "home_overall_matches_last5": get_prior_team_matches(
                team_rows,
                feature_row.home_team,
                current_event_time,
            ).head(5),
            "away_overall_matches_last5": get_prior_team_matches(
                team_rows,
                feature_row.away_team,
                current_event_time,
            ).head(5),
            "home_overall_matches_last10": get_prior_team_matches(
                team_rows,
                feature_row.home_team,
                current_event_time,
            ).head(10),
            "away_overall_matches_last10": get_prior_team_matches(
                team_rows,
                feature_row.away_team,
                current_event_time,
            ).head(10),
        }
        for count_column, window_df in windows.items():
            if (window_df["event_time"] >= current_event_time).any():
                leakage_errors += 1
            if getattr(feature_row, count_column) != len(window_df):
                count_mismatches += 1

    if leakage_errors:
        errors.append(f"leakage audit found {leakage_errors} bad window(s)")
    if count_mismatches:
        errors.append(f"leakage audit found {count_mismatches} count mismatch(es)")

    print("Feature null counts:")
    null_counts = features_df[FEATURE_COLUMNS].isna().sum()
    for column, count in null_counts.items():
        print(f"- {column}: {int(count)}")

    print("Feature rows by season:")
    season_counts = features_df.groupby("season_id").size().sort_index()
    for season_id, count in season_counts.items():
        print(f"- {season_id}: {int(count)}")

    if errors:
        print("Feature validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Feature validation failed")

    print("Feature validation passed.")
    print("Leakage audit passed: all rolling windows use event_time < current event_time.")


def create_or_verify_feature_table(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)

    with engine.connect() as conn:
        existing_columns = set(
            conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'match_features_v3_base'
                    """
                )
            ).scalars()
        )

    required_columns = set(FEATURE_TABLE_COLUMNS + ["created_at"])
    missing_columns = sorted(required_columns - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_base is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    print("match_features_v3_base schema verification passed.")


def _record_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def store_match_features(features_df: pd.DataFrame, engine) -> None:
    records = [
        {column: _record_value(row[column]) for column in FEATURE_TABLE_COLUMNS}
        for row in features_df[FEATURE_TABLE_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(FEATURE_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in FEATURE_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO match_features_v3_base (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM match_features_v3_base"))
        conn.execute(insert_sql, records)

    print(f"Stored {len(records)} rows in match_features_v3_base")


def print_feature_summary(engine) -> None:
    with engine.connect() as conn:
        total_rows = conn.execute(
            text("SELECT COUNT(*) FROM match_features_v3_base")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                """
                SELECT
                    season_id,
                    COUNT(*) AS row_count,
                    MIN(match_date) AS min_match_date,
                    MAX(match_date) AS max_match_date
                FROM match_features_v3_base
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        null_select = ",\n                ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in FEATURE_COLUMNS
        )
        null_counts = conn.execute(
            text(f"SELECT {null_select} FROM match_features_v3_base")
        ).mappings().one()
        zero_home = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM match_features_v3_base
                WHERE home_home_matches_last5 = 0
                """
            )
        ).scalar_one()
        zero_away = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM match_features_v3_base
                WHERE away_away_matches_last5 = 0
                """
            )
        ).scalar_one()

    print("=== match_features_v3_base Summary ===")
    print(f"match_features_v3_base total rows: {total_rows}")
    for row in season_rows:
        print(
            f"{row['season_id']}: {row['row_count']} rows, "
            f"{row['min_match_date']} to {row['max_match_date']}"
        )

    print("Feature null counts:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}: {null_counts[column]}")

    print(f"Rows with zero prior home history: {zero_home}")
    print(f"Rows with zero prior away history: {zero_away}")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    validate_historical_match_integrity(engine)

    matches_df = load_historical_matches_with_xg(engine)
    team_rows = build_team_perspective_rows(matches_df)
    features_df = build_match_features(matches_df, team_rows)
    validate_match_features(features_df, matches_df, team_rows)

    tier2_before = capture_table_counts(engine, TIER2_COUNT_TABLES, "Tier 2 counts before")
    source_before = capture_table_counts(
        engine,
        SOURCE_TABLES,
        "Historical source counts before",
    )

    create_or_verify_feature_table(engine)
    store_match_features(features_df, engine)

    tier2_after = capture_table_counts(engine, TIER2_COUNT_TABLES, "Tier 2 counts after")
    source_after = capture_table_counts(
        engine,
        SOURCE_TABLES,
        "Historical source counts after",
    )
    verify_counts_unchanged(tier2_before, tier2_after, "Tier 2 table counts")
    verify_counts_unchanged(source_before, source_after, "Historical source table counts")

    print_feature_summary(engine)
    print("No model training occurred.")
    print("Tier 2 tables, Streamlit, and model artifacts were not touched.")


if __name__ == "__main__":
    main()
