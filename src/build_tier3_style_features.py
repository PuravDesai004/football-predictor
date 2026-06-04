from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas
from sqlalchemy import text

from data_pipeline import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

STYLE_TABLE = "match_features_v3_style_experiment"
EXPECTED_ROW_COUNT = 1900

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

STYLE_COLUMNS = [
    "home_attack_xg_last5",
    "away_attack_xg_last5",
    "home_defense_xga_last5",
    "away_defense_xga_last5",
    "home_attack_xg_last10",
    "away_attack_xg_last10",
    "home_defense_xga_last10",
    "away_defense_xga_last10",
    "home_attack_minus_away_defense_last5",
    "away_attack_minus_home_defense_last5",
    "home_attack_minus_away_defense_last10",
    "away_attack_minus_home_defense_last10",
    "style_xg_pressure_diff_last5",
    "style_xg_pressure_diff_last10",
    "home_style_prior_matches",
    "away_style_prior_matches",
]

STYLE_TABLE_COLUMNS = [*ELO_TABLE_COLUMNS, *STYLE_COLUMNS]

FORBIDDEN_FINAL_COLUMNS = {
    "home_xg",
    "away_xg",
    "home_xga",
    "away_xga",
    "same_match_xg",
    "same_match_xga",
    "xg_for",
    "xg_against",
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
    "style_cluster",
    "team_style_cluster",
    "home_style_cluster",
    "away_style_cluster",
    "style_label",
    "home_style_label",
    "away_style_label",
    "ppda",
    "deep",
    "deep_completions",
    "possession",
    "corners",
    "shots",
    "fdr",
    "strength",
    "odds",
}

WATCHED_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "team_style_clusters",
    "player_gameweek_history",
    "player_gameweek_features",
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "match_features_v3_h2h_experiment",
    STYLE_TABLE,
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def _table_exists(conn, table_name: str) -> bool:
    with conn.connect() as db:
        return db.execute(
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


def _get_table_columns(conn, table_name: str) -> set[str]:
    with conn.connect() as db:
        columns = db.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = CURRENT_SCHEMA()
                    AND table_name = :table_name
                """
            ),
            {"table_name": table_name},
        ).scalars().all()
    return set(columns)


def _count_table_rows(conn, table_name: str) -> int:
    with conn.connect() as db:
        return int(db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _record_value(value: Any):
    if pandas.isna(value):
        return None
    if isinstance(value, pandas.Timestamp):
        return value.to_pydatetime()
    return value


def _prepare_match_times(df: pandas.DataFrame) -> pandas.DataFrame:
    prepared_df = df.copy()
    prepared_df["match_date"] = pandas.to_datetime(prepared_df["match_date"]).dt.date
    prepared_df["match_date_ts"] = pandas.to_datetime(prepared_df["match_date"])
    prepared_df["kickoff_time"] = pandas.to_datetime(
        prepared_df["kickoff_time"],
        errors="coerce",
    )
    prepared_df["event_time"] = prepared_df["kickoff_time"].fillna(
        prepared_df["match_date_ts"]
    )
    return prepared_df


def _mean_if_enough(prior_df: pandas.DataFrame, column: str, window: int) -> float | None:
    if len(prior_df) < window:
        return None
    return float(prior_df.head(window)[column].mean())


def _diff_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def load_match_base(conn) -> pandas.DataFrame:
    if not _table_exists(conn, "match_features_v3_elo"):
        raise RuntimeError("match_features_v3_elo table does not exist")

    existing_columns = _get_table_columns(conn, "match_features_v3_elo")
    missing_columns = sorted(set(ELO_TABLE_COLUMNS) - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_elo is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    column_list = ",\n            ".join(ELO_TABLE_COLUMNS)
    query = text(
        f"""
        SELECT
            {column_list}
        FROM match_features_v3_elo
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    matches_df = pandas.read_sql(query, conn)
    matches_df = _prepare_match_times(matches_df)

    errors: list[str] = []
    if len(matches_df) != EXPECTED_ROW_COUNT:
        errors.append(
            f"match_features_v3_elo expected {EXPECTED_ROW_COUNT} rows, "
            f"found {len(matches_df)}"
        )
    duplicate_count = int(matches_df["match_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"match_features_v3_elo duplicate match_id count: {duplicate_count}")

    null_counts = matches_df[ID_TARGET_COLUMNS].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"match_features_v3_elo target/id nulls: {bad_nulls}")

    expected_targets = pandas.DataFrame(
        {
            "home_win": (matches_df["result"] == "H").astype(int),
            "is_draw": (matches_df["result"] == "D").astype(int),
            "away_win": (matches_df["result"] == "A").astype(int),
        }
    )
    target_mismatch_count = int(
        (
            matches_df[["home_win", "is_draw", "away_win"]].reset_index(drop=True)
            != expected_targets
        )
        .any(axis=1)
        .sum()
    )
    if target_mismatch_count:
        errors.append(
            f"match_features_v3_elo has {target_mismatch_count} target mismatch row(s)"
        )

    if errors:
        raise ValueError("Match base validation failed: " + "; ".join(errors))

    print(f"Loaded match_features_v3_elo base rows: {len(matches_df)}")
    return matches_df


def load_understat_team_match_history(conn) -> pandas.DataFrame:
    if not _table_exists(conn, "historical_understat_xg"):
        raise RuntimeError("historical_understat_xg table does not exist")

    query = text(
        """
        SELECT
            understat_match_id,
            season_id,
            match_date,
            home_team,
            away_team,
            home_xg,
            away_xg
        FROM historical_understat_xg
        ORDER BY match_date, understat_match_id
        """
    )
    understat_df = pandas.read_sql(query, conn)
    understat_df["match_date"] = pandas.to_datetime(understat_df["match_date"]).dt.date

    errors: list[str] = []
    if len(understat_df) != EXPECTED_ROW_COUNT:
        errors.append(
            f"historical_understat_xg expected {EXPECTED_ROW_COUNT} rows, "
            f"found {len(understat_df)}"
        )
    duplicate_count = int(
        understat_df.duplicated(
            subset=["season_id", "match_date", "home_team", "away_team"]
        ).sum()
    )
    if duplicate_count:
        errors.append(
            "historical_understat_xg duplicate season/date/home/away count: "
            f"{duplicate_count}"
        )
    null_counts = understat_df[
        ["season_id", "match_date", "home_team", "away_team", "home_xg", "away_xg"]
    ].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"historical_understat_xg nulls: {bad_nulls}")
    negative_xg_count = int(
        ((understat_df["home_xg"] < 0) | (understat_df["away_xg"] < 0)).sum()
    )
    if negative_xg_count:
        errors.append(f"historical_understat_xg negative xG rows: {negative_xg_count}")

    if errors:
        raise ValueError("Understat source validation failed: " + "; ".join(errors))

    print(f"Loaded historical_understat_xg source rows: {len(understat_df)}")
    return understat_df


def build_team_match_history(
    matches_df: pandas.DataFrame,
    understat_df: pandas.DataFrame,
) -> pandas.DataFrame:
    join_columns = ["season_id", "match_date", "home_team", "away_team"]
    merged_df = matches_df.merge(
        understat_df,
        on=join_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_understat"),
    )

    errors: list[str] = []
    missing_xg_count = int(merged_df[["home_xg", "away_xg"]].isna().any(axis=1).sum())
    if missing_xg_count:
        examples = merged_df.loc[
            merged_df[["home_xg", "away_xg"]].isna().any(axis=1),
            ["season_id", "match_date", "home_team", "away_team"],
        ].head(5)
        errors.append(
            f"{missing_xg_count} match_features_v3_elo rows lack historical_understat_xg; "
            f"examples: {examples.to_dict(orient='records')}"
        )

    rows: list[dict[str, Any]] = []
    for match in merged_df.sort_values(["event_time", "match_id"]).itertuples(
        index=False
    ):
        rows.append(
            {
                "match_id": match.match_id,
                "season_id": match.season_id,
                "match_date": match.match_date,
                "event_time": match.event_time,
                "team": match.home_team,
                "opponent": match.away_team,
                "venue": "home",
                "xg_for": float(match.home_xg),
                "xg_against": float(match.away_xg),
            }
        )
        rows.append(
            {
                "match_id": match.match_id,
                "season_id": match.season_id,
                "match_date": match.match_date,
                "event_time": match.event_time,
                "team": match.away_team,
                "opponent": match.home_team,
                "venue": "away",
                "xg_for": float(match.away_xg),
                "xg_against": float(match.home_xg),
            }
        )

    team_history_df = pandas.DataFrame(rows)
    if len(team_history_df) != len(matches_df) * 2:
        errors.append(
            f"team history expected {len(matches_df) * 2} rows, "
            f"found {len(team_history_df)}"
        )
    if team_history_df[["xg_for", "xg_against", "event_time"]].isna().any().any():
        errors.append("team history has null xG or event_time values")

    if errors:
        raise ValueError("Team match history validation failed: " + "; ".join(errors))

    print(f"Built team-match history rows: {len(team_history_df)}")
    return team_history_df


def compute_prior_style_features(
    team_history_df: pandas.DataFrame,
    matches_df: pandas.DataFrame,
) -> pandas.DataFrame:
    ordered_history = team_history_df.sort_values(["event_time", "match_id"]).copy()
    history_by_team = {
        team: team_df.sort_values(["event_time", "match_id"]).copy()
        for team, team_df in ordered_history.groupby("team")
    }

    rows: list[dict[str, Any]] = []
    leakage_errors: list[dict[str, Any]] = []
    checked_source_rows = 0

    for match in matches_df.sort_values(["event_time", "match_id"]).itertuples(
        index=False
    ):
        current_event_time = match.event_time
        home_prior = history_by_team.get(match.home_team, pandas.DataFrame())
        away_prior = history_by_team.get(match.away_team, pandas.DataFrame())
        home_prior = home_prior.loc[home_prior["event_time"] < current_event_time]
        away_prior = away_prior.loc[away_prior["event_time"] < current_event_time]
        home_prior = home_prior.sort_values(["event_time", "match_id"], ascending=False)
        away_prior = away_prior.sort_values(["event_time", "match_id"], ascending=False)

        for side_name, prior_df in [("home", home_prior), ("away", away_prior)]:
            for window in [5, 10]:
                source_rows = prior_df.head(window)
                checked_source_rows += len(source_rows)
                bad_sources = source_rows.loc[
                    source_rows["event_time"] >= current_event_time,
                    ["match_id", "event_time"],
                ]
                if not bad_sources.empty:
                    leakage_errors.append(
                        {
                            "target_match_id": match.match_id,
                            "side": side_name,
                            "window": window,
                            "sources": bad_sources.to_dict(orient="records"),
                        }
                    )

        home_attack_xg_last5 = _mean_if_enough(home_prior, "xg_for", 5)
        away_attack_xg_last5 = _mean_if_enough(away_prior, "xg_for", 5)
        home_defense_xga_last5 = _mean_if_enough(home_prior, "xg_against", 5)
        away_defense_xga_last5 = _mean_if_enough(away_prior, "xg_against", 5)
        home_attack_xg_last10 = _mean_if_enough(home_prior, "xg_for", 10)
        away_attack_xg_last10 = _mean_if_enough(away_prior, "xg_for", 10)
        home_defense_xga_last10 = _mean_if_enough(home_prior, "xg_against", 10)
        away_defense_xga_last10 = _mean_if_enough(away_prior, "xg_against", 10)

        home_attack_minus_away_defense_last5 = _diff_or_none(
            home_attack_xg_last5,
            away_defense_xga_last5,
        )
        away_attack_minus_home_defense_last5 = _diff_or_none(
            away_attack_xg_last5,
            home_defense_xga_last5,
        )
        home_attack_minus_away_defense_last10 = _diff_or_none(
            home_attack_xg_last10,
            away_defense_xga_last10,
        )
        away_attack_minus_home_defense_last10 = _diff_or_none(
            away_attack_xg_last10,
            home_defense_xga_last10,
        )

        row = {column: getattr(match, column) for column in ELO_TABLE_COLUMNS}
        row.update(
            {
                "home_attack_xg_last5": home_attack_xg_last5,
                "away_attack_xg_last5": away_attack_xg_last5,
                "home_defense_xga_last5": home_defense_xga_last5,
                "away_defense_xga_last5": away_defense_xga_last5,
                "home_attack_xg_last10": home_attack_xg_last10,
                "away_attack_xg_last10": away_attack_xg_last10,
                "home_defense_xga_last10": home_defense_xga_last10,
                "away_defense_xga_last10": away_defense_xga_last10,
                "home_attack_minus_away_defense_last5": (
                    home_attack_minus_away_defense_last5
                ),
                "away_attack_minus_home_defense_last5": (
                    away_attack_minus_home_defense_last5
                ),
                "home_attack_minus_away_defense_last10": (
                    home_attack_minus_away_defense_last10
                ),
                "away_attack_minus_home_defense_last10": (
                    away_attack_minus_home_defense_last10
                ),
                "style_xg_pressure_diff_last5": _diff_or_none(
                    home_attack_minus_away_defense_last5,
                    away_attack_minus_home_defense_last5,
                ),
                "style_xg_pressure_diff_last10": _diff_or_none(
                    home_attack_minus_away_defense_last10,
                    away_attack_minus_home_defense_last10,
                ),
                "home_style_prior_matches": int(len(home_prior)),
                "away_style_prior_matches": int(len(away_prior)),
            }
        )
        rows.append(row)

    style_df = pandas.DataFrame(rows, columns=STYLE_TABLE_COLUMNS)
    style_df.attrs["leakage_checked_source_rows"] = checked_source_rows
    style_df.attrs["leakage_errors"] = leakage_errors
    style_df.attrs["leakage_rule"] = "source event_time < target match event_time"

    print(f"Computed style feature rows: {len(style_df)}")
    return style_df


def validate_style_features(style_df: pandas.DataFrame) -> None:
    errors: list[str] = []

    if len(style_df) != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} style rows, found {len(style_df)}")

    missing_columns = sorted(set(STYLE_TABLE_COLUMNS) - set(style_df.columns))
    if missing_columns:
        errors.append(
            "style feature DataFrame missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    extra_columns = sorted(set(style_df.columns) - set(STYLE_TABLE_COLUMNS))
    forbidden_extra_columns = sorted(
        column
        for column in extra_columns
        if column in FORBIDDEN_FINAL_COLUMNS
        or any(token in column for token in ["cluster", "label", "ppda", "deep"])
    )
    if extra_columns:
        errors.append(
            "style feature DataFrame has unexpected column(s): "
            f"{', '.join(extra_columns)}"
        )
    if forbidden_extra_columns:
        errors.append(
            "style feature DataFrame has forbidden column(s): "
            f"{', '.join(forbidden_extra_columns)}"
        )

    forbidden_present = sorted(FORBIDDEN_FINAL_COLUMNS & set(style_df.columns))
    if forbidden_present:
        errors.append(
            "style feature DataFrame contains forbidden final column(s): "
            f"{', '.join(forbidden_present)}"
        )

    duplicate_match_id_count = int(style_df["match_id"].duplicated().sum())
    if duplicate_match_id_count:
        errors.append(f"duplicate match_id count: {duplicate_match_id_count}")

    target_expected = pandas.DataFrame(
        {
            "home_win": (style_df["result"] == "H").astype(int),
            "is_draw": (style_df["result"] == "D").astype(int),
            "away_win": (style_df["result"] == "A").astype(int),
        }
    )
    target_mismatch_count = int(
        (
            style_df[["home_win", "is_draw", "away_win"]].reset_index(drop=True)
            != target_expected
        )
        .any(axis=1)
        .sum()
    )
    if target_mismatch_count:
        errors.append(f"target/result mismatch count: {target_mismatch_count}")

    prior_negative_count = int(
        (
            (style_df["home_style_prior_matches"] < 0)
            | (style_df["away_style_prior_matches"] < 0)
        ).sum()
    )
    if prior_negative_count:
        errors.append(f"negative style prior match count rows: {prior_negative_count}")

    for column in [
        "home_attack_minus_away_defense_last5",
        "away_attack_minus_home_defense_last5",
        "home_attack_minus_away_defense_last10",
        "away_attack_minus_home_defense_last10",
        "style_xg_pressure_diff_last5",
        "style_xg_pressure_diff_last10",
    ]:
        infinite_count = int(pandas.Series(style_df[column]).isin([float("inf"), float("-inf")]).sum())
        if infinite_count:
            errors.append(f"{column} has {infinite_count} infinite value(s)")

    leakage_errors = style_df.attrs.get("leakage_errors", [])
    if leakage_errors:
        errors.append(
            "leakage audit found source rows with event_time >= target event_time: "
            f"{leakage_errors[:5]}"
        )

    season_counts = (
        style_df.groupby("season_id")["match_id"].count().sort_index().to_dict()
    )
    for season_id, row_count in season_counts.items():
        if int(row_count) != 380:
            errors.append(f"{season_id} expected 380 rows, found {int(row_count)}")

    if errors:
        raise ValueError("Style feature validation failed: " + "; ".join(errors))

    null_counts = style_df[STYLE_COLUMNS].isna().sum()
    print("Style feature validation passed.")
    print(f"Duplicate match_id count: {duplicate_match_id_count}")
    print(
        "Leakage audit passed: "
        f"{style_df.attrs.get('leakage_checked_source_rows', 0)} source rows checked "
        f"with rule {style_df.attrs.get('leakage_rule')}"
    )
    print("Style feature null summary:")
    for column, count in null_counts.items():
        print(f"- {column}: {int(count)}")
    print("Style feature rows by season:")
    for season_id, row_count in season_counts.items():
        print(f"- {season_id}: {int(row_count)}")
    print("Style feature min/max/mean by season:")
    feature_summary = (
        style_df.groupby("season_id")[STYLE_COLUMNS]
        .agg(["min", "max", "mean"])
        .round(4)
    )
    print(feature_summary.to_string())


def create_style_table(conn) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with conn.begin() as db:
        db.exec_driver_sql(schema_sql)

    if not _table_exists(conn, STYLE_TABLE):
        raise RuntimeError(f"{STYLE_TABLE} table does not exist after schema init")

    existing_columns = _get_table_columns(conn, STYLE_TABLE)
    missing_columns = sorted(set(STYLE_TABLE_COLUMNS) - existing_columns)
    if missing_columns:
        raise RuntimeError(
            f"{STYLE_TABLE} is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    forbidden_columns = sorted(FORBIDDEN_FINAL_COLUMNS & existing_columns)
    if forbidden_columns:
        raise RuntimeError(
            f"{STYLE_TABLE} has forbidden column(s): {', '.join(forbidden_columns)}"
        )

    print(f"{STYLE_TABLE} schema verification passed.")


def write_style_features(conn, style_df: pandas.DataFrame) -> None:
    records = [
        {column: _record_value(row[column]) for column in STYLE_TABLE_COLUMNS}
        for row in style_df.to_dict(orient="records")
    ]
    column_list = ",\n            ".join(STYLE_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in STYLE_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO {STYLE_TABLE} (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )

    with conn.begin() as db:
        db.execute(text(f"DELETE FROM {STYLE_TABLE}"))
        db.execute(insert_sql, records)

    print(f"Stored {len(records)} rows in {STYLE_TABLE}")


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    for table_name in WATCHED_TABLES:
        if _table_exists(conn, table_name):
            counts[table_name] = _count_table_rows(conn, table_name)
        else:
            counts[table_name] = "MISSING"
    return counts


def assert_watched_counts_unchanged_except_style(before, after) -> None:
    changed_unexpectedly = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if table_name != STYLE_TABLE and before.get(table_name) != after.get(table_name)
    }
    if changed_unexpectedly:
        raise RuntimeError(
            "Watched table counts changed unexpectedly: "
            f"{changed_unexpectedly}"
        )

    style_after = after.get(STYLE_TABLE)
    if style_after != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"{STYLE_TABLE} expected {EXPECTED_ROW_COUNT} rows after write, "
            f"found {style_after}"
        )

    print("Watched table counts unchanged except style experiment table.")
    print(f"{STYLE_TABLE}: {before.get(STYLE_TABLE)} -> {style_after}")


def _print_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name in WATCHED_TABLES:
        print(f"{table_name}: {counts.get(table_name)}")


def _verify_style_table_targets_preserved(conn) -> None:
    with conn.connect() as db:
        mismatch_count = db.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {STYLE_TABLE} s
                INNER JOIN match_features_v3_elo e
                    ON s.match_id = e.match_id
                WHERE s.season_id <> e.season_id
                    OR s.match_date <> e.match_date
                    OR s.home_team <> e.home_team
                    OR s.away_team <> e.away_team
                    OR s.home_goals <> e.home_goals
                    OR s.away_goals <> e.away_goals
                    OR s.result <> e.result
                    OR s.home_win <> e.home_win
                    OR s.is_draw <> e.is_draw
                    OR s.away_win <> e.away_win
                """
            )
        ).scalar_one()
        style_missing_elo_count = db.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM {STYLE_TABLE} s
                LEFT JOIN match_features_v3_elo e
                    ON s.match_id = e.match_id
                WHERE e.match_id IS NULL
                """
            )
        ).scalar_one()
        elo_missing_style_count = db.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM match_features_v3_elo e
                LEFT JOIN {STYLE_TABLE} s
                    ON s.match_id = e.match_id
                WHERE s.match_id IS NULL
                """
            )
        ).scalar_one()
        style_count = db.execute(text(f"SELECT COUNT(*) FROM {STYLE_TABLE}")).scalar_one()
        elo_count = db.execute(text("SELECT COUNT(*) FROM match_features_v3_elo")).scalar_one()

    if int(mismatch_count) != 0:
        raise RuntimeError(
            f"{STYLE_TABLE} has {mismatch_count} row(s) with target/base mismatches"
        )
    if int(style_missing_elo_count) != 0:
        raise RuntimeError(
            f"{STYLE_TABLE} has {style_missing_elo_count} row(s) absent from match_features_v3_elo"
        )
    if int(elo_missing_style_count) != 0:
        raise RuntimeError(
            f"match_features_v3_elo has {elo_missing_style_count} row(s) absent from {STYLE_TABLE}"
        )
    if int(style_count) != int(elo_count):
        raise RuntimeError(
            f"{STYLE_TABLE} row count {style_count} != match_features_v3_elo {elo_count}"
        )
    print(
        "Target/result preservation verified against match_features_v3_elo: "
        f"{mismatch_count} mismatches, "
        f"{style_missing_elo_count} style-only rows, "
        f"{elo_missing_style_count} Elo-only rows"
    )


def _print_db_style_summary(conn) -> None:
    with conn.connect() as db:
        total_rows = db.execute(text(f"SELECT COUNT(*) FROM {STYLE_TABLE}")).scalar_one()
        rows_by_season = db.execute(
            text(
                f"""
                SELECT season_id, COUNT(*) AS row_count
                FROM {STYLE_TABLE}
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        null_select = ",\n                ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in STYLE_COLUMNS
        )
        null_counts = db.execute(
            text(f"SELECT {null_select} FROM {STYLE_TABLE}")
        ).mappings().one()

    print(f"{STYLE_TABLE} row count: {total_rows}")
    print(f"{STYLE_TABLE} rows by season:")
    for row in rows_by_season:
        print(f"- {row['season_id']}: {row['row_count']}")
    print(f"{STYLE_TABLE} database null summary:")
    for column in STYLE_COLUMNS:
        print(f"- {column}: {null_counts[column]}")


def main() -> None:
    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)
    _print_counts("Watched table counts before style write", before_counts)

    matches_df = load_match_base(conn)
    understat_df = load_understat_team_match_history(conn)
    team_history_df = build_team_match_history(matches_df, understat_df)
    style_df = compute_prior_style_features(team_history_df, matches_df)
    validate_style_features(style_df)

    create_style_table(conn)
    write_style_features(conn, style_df)
    _verify_style_table_targets_preserved(conn)
    _print_db_style_summary(conn)

    after_counts = capture_watched_table_counts(conn)
    _print_counts("Watched table counts after style write", after_counts)
    assert_watched_counts_unchanged_except_style(before_counts, after_counts)

    print("Source tables used: historical_matches, historical_understat_xg, match_features_v3_elo")
    print("No understat_team_history, team_style_clusters, H2H, odds, PPDA, deep, shots, possession, or FPL strength/FDR data used.")
    print("No model training, model artifacts, Streamlit changes, or final holdout evaluation occurred.")


if __name__ == "__main__":
    main()
