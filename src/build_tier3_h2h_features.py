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
H2H_TABLE_NAME = "match_features_v3_h2h_experiment"

ELO_INHERITED_COLUMNS = [
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
    "home_elo_before",
    "away_elo_before",
    "elo_diff_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
    "home_initialization",
    "away_initialization",
]

INHERITED_ID_TARGET_COLUMNS = [
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

H2H_COUNT_COLUMNS = [
    "h2h_matches_prior",
    "h2h_home_wins_prior",
    "h2h_draws_prior",
    "h2h_away_wins_prior",
]

H2H_AVERAGE_COLUMNS = [
    "h2h_home_goals_avg_prior",
    "h2h_away_goals_avg_prior",
    "h2h_home_points_avg_prior",
    "h2h_away_points_avg_prior",
    "h2h_goal_diff_avg_prior",
]

H2H_RATE_COLUMNS = [
    "h2h_home_win_rate_prior",
    "h2h_draw_rate_prior",
    "h2h_away_win_rate_prior",
]

H2H_LAST_MEETING_COLUMNS = [
    "h2h_last_meeting_days",
    "h2h_last_meeting_home_goals",
    "h2h_last_meeting_away_goals",
    "h2h_last_meeting_result",
]

H2H_COLUMNS = [
    *H2H_COUNT_COLUMNS,
    *H2H_AVERAGE_COLUMNS,
    *H2H_RATE_COLUMNS,
    *H2H_LAST_MEETING_COLUMNS,
]

H2H_TABLE_COLUMNS = [*ELO_INHERITED_COLUMNS, *H2H_COLUMNS]

FORBIDDEN_POST_MATCH_ELO_COLUMNS = [
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
]

SAFETY_COUNT_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]


def _table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
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


def _count_table_rows(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _query_mappings(
    engine,
    query: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(query), params or {}).mappings().all()
    return [dict(row) for row in rows]


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


def _expected_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _row_value(row, column: str):
    if isinstance(row, pd.Series):
        return row[column]
    return getattr(row, column)


def load_matches_for_h2h(engine) -> pd.DataFrame:
    query = """
        SELECT
            match_id,
            season_id,
            match_date,
            kickoff_time,
            home_team,
            away_team,
            home_goals,
            away_goals,
            result
        FROM historical_matches
        ORDER BY match_date, kickoff_time, match_id
    """
    matches_df = pd.read_sql(text(query), engine)
    matches_df = _prepare_match_times(matches_df)

    errors: list[str] = []
    if len(matches_df) != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} rows, found {len(matches_df)}")
    if matches_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(matches_df['match_id'].duplicated().sum())}"
        )

    required_columns = [
        "match_id",
        "season_id",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "event_time",
    ]
    null_counts = matches_df[required_columns].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"null required columns: {bad_nulls}")

    expected_results = matches_df.apply(
        lambda row: _expected_result(int(row["home_goals"]), int(row["away_goals"])),
        axis=1,
    )
    mismatches = matches_df["result"] != expected_results
    if mismatches.any():
        errors.append(f"result/goals mismatch count: {int(mismatches.sum())}")

    if errors:
        raise ValueError("Historical matches H2H load validation failed: " + "; ".join(errors))

    matches_df = matches_df.sort_values(["event_time", "match_id"]).reset_index(
        drop=True
    )
    print(f"Loaded {len(matches_df)} historical matches for H2H.")
    return matches_df


def load_elo_features(engine) -> pd.DataFrame:
    elo_features_df = pd.read_sql(
        text(
            """
            SELECT *
            FROM match_features_v3_elo
            ORDER BY match_date, kickoff_time, match_id
            """
        ),
        engine,
    )

    errors: list[str] = []
    if len(elo_features_df) != EXPECTED_ROW_COUNT:
        errors.append(
            f"expected {EXPECTED_ROW_COUNT} rows, found {len(elo_features_df)}"
        )
    if elo_features_df["match_id"].duplicated().any():
        errors.append(
            "duplicate match_id count: "
            f"{int(elo_features_df['match_id'].duplicated().sum())}"
        )
    if elo_features_df["match_id"].nunique() != len(elo_features_df):
        errors.append("match_features_v3_elo does not have one row per match_id")

    forbidden_present = [
        column for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS if column in elo_features_df.columns
    ]
    if forbidden_present:
        errors.append(f"forbidden post-match Elo columns present: {forbidden_present}")

    missing_inherited = sorted(set(ELO_INHERITED_COLUMNS) - set(elo_features_df.columns))
    if missing_inherited:
        errors.append(f"missing inherited Elo feature columns: {missing_inherited}")

    if errors:
        raise ValueError("match_features_v3_elo load validation failed: " + "; ".join(errors))

    print(f"Loaded {len(elo_features_df)} rows from match_features_v3_elo.")
    return elo_features_df[ELO_INHERITED_COLUMNS].copy()


def get_prior_h2h_matches(
    matches_df: pd.DataFrame,
    current_home_team: str,
    current_away_team: str,
    current_event_time,
) -> pd.DataFrame:
    event_time = pd.Timestamp(current_event_time)
    same_pair = (
        (
            (matches_df["home_team"] == current_home_team)
            & (matches_df["away_team"] == current_away_team)
        )
        | (
            (matches_df["home_team"] == current_away_team)
            & (matches_df["away_team"] == current_home_team)
        )
    )
    prior_rows = matches_df.loc[same_pair & (matches_df["event_time"] < event_time)]
    return prior_rows.sort_values(
        ["event_time", "match_id"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _goals_from_current_orientation(
    prior_row,
    current_home_team: str,
    current_away_team: str,
) -> tuple[int, int]:
    if _row_value(prior_row, "home_team") == current_home_team:
        return int(_row_value(prior_row, "home_goals")), int(
            _row_value(prior_row, "away_goals")
        )
    if _row_value(prior_row, "away_team") == current_home_team:
        return int(_row_value(prior_row, "away_goals")), int(
            _row_value(prior_row, "home_goals")
        )
    raise ValueError(
        "Prior H2H row does not contain current fixture teams: "
        f"{current_home_team} vs {current_away_team}"
    )


def compute_h2h_features_for_match(match_row, prior_h2h_df: pd.DataFrame) -> dict:
    current_home_team = _row_value(match_row, "home_team")
    current_away_team = _row_value(match_row, "away_team")

    if prior_h2h_df.empty:
        return {
            "h2h_matches_prior": 0,
            "h2h_home_wins_prior": 0,
            "h2h_draws_prior": 0,
            "h2h_away_wins_prior": 0,
            "h2h_home_goals_avg_prior": None,
            "h2h_away_goals_avg_prior": None,
            "h2h_home_points_avg_prior": None,
            "h2h_away_points_avg_prior": None,
            "h2h_goal_diff_avg_prior": None,
            "h2h_home_win_rate_prior": None,
            "h2h_draw_rate_prior": None,
            "h2h_away_win_rate_prior": None,
            "h2h_last_meeting_days": None,
            "h2h_last_meeting_home_goals": None,
            "h2h_last_meeting_away_goals": None,
            "h2h_last_meeting_result": None,
        }

    home_goals_values: list[int] = []
    away_goals_values: list[int] = []
    home_points_values: list[int] = []
    away_points_values: list[int] = []
    current_home_results: list[str] = []

    for prior_row in prior_h2h_df.itertuples(index=False):
        home_goals, away_goals = _goals_from_current_orientation(
            prior_row,
            current_home_team,
            current_away_team,
        )
        home_goals_values.append(home_goals)
        away_goals_values.append(away_goals)
        home_points_values.append(_points_for(home_goals, away_goals))
        away_points_values.append(_points_for(away_goals, home_goals))
        current_home_results.append(_expected_result(home_goals, away_goals))

    h2h_matches_prior = len(prior_h2h_df)
    h2h_home_wins_prior = current_home_results.count("H")
    h2h_draws_prior = current_home_results.count("D")
    h2h_away_wins_prior = current_home_results.count("A")

    last_prior_meeting = prior_h2h_df.iloc[0]
    last_home_goals, last_away_goals = _goals_from_current_orientation(
        last_prior_meeting,
        current_home_team,
        current_away_team,
    )
    current_match_date = pd.Timestamp(_row_value(match_row, "match_date")).date()
    last_match_date = pd.Timestamp(last_prior_meeting["match_date"]).date()
    last_meeting_days = (current_match_date - last_match_date).days

    return {
        "h2h_matches_prior": h2h_matches_prior,
        "h2h_home_wins_prior": h2h_home_wins_prior,
        "h2h_draws_prior": h2h_draws_prior,
        "h2h_away_wins_prior": h2h_away_wins_prior,
        "h2h_home_goals_avg_prior": float(np.mean(home_goals_values)),
        "h2h_away_goals_avg_prior": float(np.mean(away_goals_values)),
        "h2h_home_points_avg_prior": float(np.mean(home_points_values)),
        "h2h_away_points_avg_prior": float(np.mean(away_points_values)),
        "h2h_goal_diff_avg_prior": float(
            np.mean(np.array(home_goals_values) - np.array(away_goals_values))
        ),
        "h2h_home_win_rate_prior": h2h_home_wins_prior / h2h_matches_prior,
        "h2h_draw_rate_prior": h2h_draws_prior / h2h_matches_prior,
        "h2h_away_win_rate_prior": h2h_away_wins_prior / h2h_matches_prior,
        "h2h_last_meeting_days": int(last_meeting_days),
        "h2h_last_meeting_home_goals": int(last_home_goals),
        "h2h_last_meeting_away_goals": int(last_away_goals),
        "h2h_last_meeting_result": _expected_result(last_home_goals, last_away_goals),
    }


def build_h2h_feature_frame(
    matches_df: pd.DataFrame,
    elo_features_df: pd.DataFrame,
) -> pd.DataFrame:
    h2h_rows: list[dict[str, Any]] = []
    for match_row in matches_df.itertuples(index=False):
        prior_h2h_df = get_prior_h2h_matches(
            matches_df,
            match_row.home_team,
            match_row.away_team,
            match_row.event_time,
        )
        h2h_rows.append(
            {
                "match_id": int(match_row.match_id),
                **compute_h2h_features_for_match(match_row, prior_h2h_df),
            }
        )

    h2h_df = pd.DataFrame(h2h_rows)
    features_df = elo_features_df.merge(h2h_df, on="match_id", how="inner")
    if len(features_df) != len(elo_features_df):
        raise ValueError(
            "H2H feature join changed row count: "
            f"{len(features_df)} != {len(elo_features_df)}"
        )

    features_df = features_df.sort_values(
        ["match_date", "kickoff_time", "match_id"],
        na_position="last",
    ).reset_index(drop=True)
    print(f"Built {len(features_df)} H2H experiment feature rows.")
    return features_df[H2H_TABLE_COLUMNS].copy()


def _print_h2h_distribution(features_df: pd.DataFrame) -> None:
    zero_prior = int((features_df["h2h_matches_prior"] == 0).sum())
    one_prior = int((features_df["h2h_matches_prior"] == 1).sum())
    two_plus_prior = int((features_df["h2h_matches_prior"] >= 2).sum())
    max_prior = int(features_df["h2h_matches_prior"].max())
    print("H2H prior-meeting distribution:")
    print(f"- rows with 0 prior meetings: {zero_prior}")
    print(f"- rows with 1 prior meeting: {one_prior}")
    print(f"- rows with 2+ prior meetings: {two_plus_prior}")
    print(f"- max prior meetings: {max_prior}")

    print("Average prior meetings by season:")
    averages = (
        features_df.groupby("season_id")["h2h_matches_prior"]
        .mean()
        .sort_index()
    )
    for season_id, average in averages.items():
        print(f"- {season_id}: {float(average):.4f}")


def validate_h2h_feature_frame(
    features_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> None:
    print("=== H2H Feature Frame Validation ===")
    errors: list[str] = []

    if len(features_df) != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} rows, found {len(features_df)}")
    if features_df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(features_df['match_id'].duplicated().sum())}"
        )
    if features_df["match_id"].nunique() != len(features_df):
        errors.append("feature frame does not have one row per match_id")

    inherited_null_counts = features_df[INHERITED_ID_TARGET_COLUMNS].isna().sum()
    inherited_nulls = {
        column: int(count)
        for column, count in inherited_null_counts.items()
        if int(count) > 0
    }
    if inherited_nulls:
        errors.append(f"null inherited ID/target columns: {inherited_nulls}")

    negative_counts = {
        column: int((features_df[column] < 0).sum())
        for column in H2H_COUNT_COLUMNS
        if int((features_df[column] < 0).sum()) > 0
    }
    if negative_counts:
        errors.append(f"negative H2H count columns: {negative_counts}")

    count_sum = (
        features_df["h2h_home_wins_prior"]
        + features_df["h2h_draws_prior"]
        + features_df["h2h_away_wins_prior"]
    )
    count_mismatches = features_df["h2h_matches_prior"] != count_sum
    if count_mismatches.any():
        errors.append(
            "h2h_matches_prior count mismatch rows: "
            f"{int(count_mismatches.sum())}"
        )

    for column in H2H_RATE_COLUMNS:
        non_null = features_df[column].dropna()
        bad_count = int((~non_null.between(0, 1)).sum())
        if bad_count:
            errors.append(f"{column} has {bad_count} value(s) outside 0..1")

    rows_with_prior = features_df["h2h_matches_prior"] > 0
    rows_without_prior = features_df["h2h_matches_prior"] == 0
    rate_nulls_with_prior = features_df.loc[rows_with_prior, H2H_RATE_COLUMNS].isna().sum()
    bad_rate_nulls = {
        column: int(count)
        for column, count in rate_nulls_with_prior.items()
        if int(count) > 0
    }
    if bad_rate_nulls:
        errors.append(f"rate nulls when prior meetings exist: {bad_rate_nulls}")

    rate_sums = features_df.loc[rows_with_prior, H2H_RATE_COLUMNS].sum(axis=1)
    if not np.allclose(rate_sums, 1.0):
        errors.append("H2H rate columns do not sum to 1 for every row with prior meetings")

    for column in [
        "h2h_home_goals_avg_prior",
        "h2h_away_goals_avg_prior",
    ]:
        non_null = features_df[column].dropna()
        bad_count = int((non_null < 0).sum())
        if bad_count:
            errors.append(f"{column} has {bad_count} negative value(s)")

    null_expected_when_no_prior = [
        *H2H_AVERAGE_COLUMNS,
        *H2H_RATE_COLUMNS,
        *H2H_LAST_MEETING_COLUMNS,
    ]
    bad_null_expected = {
        column: int(features_df.loc[rows_without_prior, column].notna().sum())
        for column in null_expected_when_no_prior
        if int(features_df.loc[rows_without_prior, column].notna().sum()) > 0
    }
    if bad_null_expected:
        errors.append(
            "columns expected null when no prior meetings are populated: "
            f"{bad_null_expected}"
        )

    last_nulls_with_prior = features_df.loc[
        rows_with_prior,
        H2H_LAST_MEETING_COLUMNS,
    ].isna().sum()
    bad_last_nulls = {
        column: int(count)
        for column, count in last_nulls_with_prior.items()
        if int(count) > 0
    }
    if bad_last_nulls:
        errors.append(f"last meeting nulls when prior meetings exist: {bad_last_nulls}")

    bad_last_days = int(
        (
            features_df.loc[features_df["h2h_last_meeting_days"].notna(), "h2h_last_meeting_days"]
            <= 0
        ).sum()
    )
    if bad_last_days:
        errors.append(f"h2h_last_meeting_days has {bad_last_days} non-positive value(s)")

    matches_by_id = matches_df.set_index("match_id")
    features_by_id = features_df.set_index("match_id")
    leakage_errors = 0
    prior_count_mismatches = 0
    for match_row in matches_df.itertuples(index=False):
        prior_h2h_df = get_prior_h2h_matches(
            matches_df,
            match_row.home_team,
            match_row.away_team,
            match_row.event_time,
        )
        if (prior_h2h_df["event_time"] >= match_row.event_time).any():
            leakage_errors += 1
        expected_prior_count = len(prior_h2h_df)
        actual_prior_count = int(features_by_id.loc[match_row.match_id, "h2h_matches_prior"])
        if expected_prior_count != actual_prior_count:
            prior_count_mismatches += 1
        if match_row.match_id in set(prior_h2h_df["match_id"]):
            leakage_errors += 1

    if leakage_errors:
        errors.append(f"H2H leakage audit found {leakage_errors} bad row(s)")
    if prior_count_mismatches:
        errors.append(
            "H2H leakage audit found prior-count mismatches: "
            f"{prior_count_mismatches}"
        )

    _print_h2h_distribution(features_df)

    if errors:
        print("H2H feature frame validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("H2H feature frame validation failed")

    print("PASS: total rows and one row per match_id")
    print("PASS: no null inherited ID/target columns")
    print("PASS: H2H count columns are non-negative")
    print("PASS: h2h_matches_prior equals wins + draws + losses")
    print("PASS: H2H rates are valid and sum to 1 when prior meetings exist")
    print("PASS: H2H average goals are non-negative when populated")
    print("PASS: last meeting fields are valid")
    print("PASS: H2H leakage audit passed")
    print("H2H feature frame validation passed.")


def create_or_verify_h2h_table(engine) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)

    if not _table_exists(engine, H2H_TABLE_NAME):
        raise RuntimeError(f"{H2H_TABLE_NAME} table does not exist")

    h2h_columns = {
        row["column_name"]
        for row in _query_mappings(
            engine,
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = CURRENT_SCHEMA()
                AND table_name = :table_name
            """,
            {"table_name": H2H_TABLE_NAME},
        )
    }
    elo_columns = {
        row["column_name"]
        for row in _query_mappings(
            engine,
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = CURRENT_SCHEMA()
                AND table_name = 'match_features_v3_elo'
            """,
        )
    }

    missing_inherited = sorted(elo_columns - h2h_columns)
    missing_h2h = sorted(set(H2H_COLUMNS) - h2h_columns)
    if missing_inherited:
        raise RuntimeError(
            f"{H2H_TABLE_NAME} missing inherited column(s): {missing_inherited}"
        )
    if missing_h2h:
        raise RuntimeError(f"{H2H_TABLE_NAME} missing H2H column(s): {missing_h2h}")

    print(f"{H2H_TABLE_NAME} schema verification passed.")


def _record_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def store_h2h_features(features_df: pd.DataFrame, engine) -> None:
    records = [
        {column: _record_value(row[column]) for column in H2H_TABLE_COLUMNS}
        for row in features_df[H2H_TABLE_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(H2H_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in H2H_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO {H2H_TABLE_NAME} (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {H2H_TABLE_NAME}"))
        conn.execute(insert_sql, records)

    print(f"Stored {len(records)} rows in {H2H_TABLE_NAME}.")


def print_h2h_summary(engine) -> None:
    with engine.connect() as conn:
        total_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {H2H_TABLE_NAME}")
        ).scalar_one()
        season_rows = conn.execute(
            text(
                f"""
                SELECT season_id, COUNT(*) AS row_count
                FROM {H2H_TABLE_NAME}
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        distribution = conn.execute(
            text(
                f"""
                SELECT
                    SUM(CASE WHEN h2h_matches_prior = 0 THEN 1 ELSE 0 END) AS zero_prior,
                    SUM(CASE WHEN h2h_matches_prior = 1 THEN 1 ELSE 0 END) AS one_prior,
                    SUM(CASE WHEN h2h_matches_prior >= 2 THEN 1 ELSE 0 END) AS two_plus_prior,
                    MAX(h2h_matches_prior) AS max_prior
                FROM {H2H_TABLE_NAME}
                """
            )
        ).mappings().one()
        average_rows = conn.execute(
            text(
                f"""
                SELECT season_id, AVG(h2h_matches_prior) AS avg_prior
                FROM {H2H_TABLE_NAME}
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        null_select = ",\n                    ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in H2H_COLUMNS
        )
        null_counts = conn.execute(
            text(f"SELECT {null_select} FROM {H2H_TABLE_NAME}")
        ).mappings().one()
        zero_samples = conn.execute(
            text(
                f"""
                SELECT match_id, season_id, match_date, home_team, away_team,
                    h2h_matches_prior
                FROM {H2H_TABLE_NAME}
                WHERE h2h_matches_prior = 0
                ORDER BY match_date, match_id
                LIMIT 5
                """
            )
        ).mappings().all()
        two_plus_samples = conn.execute(
            text(
                f"""
                SELECT match_id, season_id, match_date, home_team, away_team,
                    h2h_matches_prior, h2h_home_wins_prior, h2h_draws_prior,
                    h2h_away_wins_prior, h2h_last_meeting_result
                FROM {H2H_TABLE_NAME}
                WHERE h2h_matches_prior >= 2
                ORDER BY match_date, match_id
                LIMIT 5
                """
            )
        ).mappings().all()

    print(f"=== {H2H_TABLE_NAME} Summary ===")
    print(f"{H2H_TABLE_NAME} total rows: {total_rows}")
    print("Rows by season:")
    for row in season_rows:
        print(f"- {row['season_id']}: {row['row_count']}")

    print("Prior-meeting distribution:")
    print(f"- rows with 0 prior meetings: {distribution['zero_prior']}")
    print(f"- rows with 1 prior meeting: {distribution['one_prior']}")
    print(f"- rows with 2+ prior meetings: {distribution['two_plus_prior']}")
    print(f"- max prior meetings: {distribution['max_prior']}")

    print("Average prior meetings by season:")
    for row in average_rows:
        print(f"- {row['season_id']}: {row['avg_prior']:.4f}")

    print("H2H null counts:")
    for column in H2H_COLUMNS:
        print(f"- {column}: {null_counts[column]}")

    print("Sample rows with 0 prior meetings:")
    for row in zero_samples:
        print(
            f"- {row['match_id']} {row['season_id']} {row['match_date']} "
            f"{row['home_team']} vs {row['away_team']} prior={row['h2h_matches_prior']}"
        )

    print("Sample rows with 2+ prior meetings:")
    for row in two_plus_samples:
        print(
            f"- {row['match_id']} {row['season_id']} {row['match_date']} "
            f"{row['home_team']} vs {row['away_team']} prior={row['h2h_matches_prior']} "
            f"H/D/A={row['h2h_home_wins_prior']}/{row['h2h_draws_prior']}/"
            f"{row['h2h_away_wins_prior']} last={row['h2h_last_meeting_result']}"
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
    print("Safety counts unchanged for watched Tier 2 and Tier 3 tables.")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    print("=== Tier 3 H2H Experimental Feature Build ===")
    validate_historical_match_integrity(engine)
    matches_df = load_matches_for_h2h(engine)
    elo_features_df = load_elo_features(engine)
    features_df = build_h2h_feature_frame(matches_df, elo_features_df)
    validate_h2h_feature_frame(features_df, matches_df)

    before_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts before H2H experiment write", before_counts)

    create_or_verify_h2h_table(engine)
    store_h2h_features(features_df, engine)

    after_counts = capture_table_counts(engine, SAFETY_COUNT_TABLES)
    _print_table_counts("Safety counts after H2H experiment write", after_counts)
    _verify_counts_unchanged(before_counts, after_counts)

    print_h2h_summary(engine)
    print("2025-26 is included in the H2H feature table but remains reserved as final test.")
    print("No model training occurred.")
    print(
        "Tier 2 tables, match_features_v3_elo, match_features_v3_base, "
        "elo_ratings_v3, historical source tables, Streamlit, and model artifacts "
        "were not touched."
    )


if __name__ == "__main__":
    main()
