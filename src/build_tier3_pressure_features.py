from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas
from sqlalchemy import text

from data_pipeline import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIER3_SCHEMA_FILE = PROJECT_ROOT / "sql" / "tier3_schema.sql"

EXPECTED_MATCH_ROWS = 1900
EXPECTED_STANDINGS_ROWS = 3800
EXPECTED_SEASON_MATCH_ROWS = 380
EXPECTED_TEAMS_PER_SEASON = 20

MIN_GAMES_FOR_PRESSURE = 8
TITLE_WINDOW_POINTS = 6
TOP4_WINDOW_POINTS = 6
TOP6_WINDOW_POINTS = 6
RELEGATION_WINDOW_POINTS = 6
TOTAL_LEAGUE_MATCHES_PER_TEAM = 38

STANDINGS_TABLE = "standings_before_match_v3"
PRESSURE_TABLE = "match_features_v3_pressure_experiment"

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

STANDINGS_COLUMNS = [
    "match_id",
    "season",
    "match_date",
    "team_name",
    "opponent_team",
    "venue",
    "is_home",
    "games_played_before",
    "wins_before",
    "draws_before",
    "losses_before",
    "points_before",
    "goals_for_before",
    "goals_against_before",
    "goal_diff_before",
    "ppg_before",
    "rank_before",
    "points_to_1st_before",
    "points_to_4th_before",
    "points_to_6th_before",
    "points_above_18th_before",
]

PRESSURE_COLUMNS = [
    "home_games_played_before",
    "away_games_played_before",
    "home_points_before",
    "away_points_before",
    "home_ppg_before",
    "away_ppg_before",
    "home_rank_before",
    "away_rank_before",
    "home_goal_diff_before",
    "away_goal_diff_before",
    "rank_diff_before",
    "points_diff_before",
    "ppg_diff_before",
    "goal_diff_table_diff_before",
    "home_title_pressure_before",
    "away_title_pressure_before",
    "home_top4_pressure_before",
    "away_top4_pressure_before",
    "home_top6_pressure_before",
    "away_top6_pressure_before",
    "home_relegation_pressure_before",
    "away_relegation_pressure_before",
    "home_pressure_index_before",
    "away_pressure_index_before",
    "match_pressure_index_before",
    "pressure_diff_before",
    "season_progress_before",
]

PRESSURE_TABLE_COLUMNS = [*ELO_TABLE_COLUMNS, *PRESSURE_COLUMNS]

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

PRESSURE_COMPONENT_COLUMNS = [
    "home_title_pressure_before",
    "away_title_pressure_before",
    "home_top4_pressure_before",
    "away_top4_pressure_before",
    "home_top6_pressure_before",
    "away_top6_pressure_before",
    "home_relegation_pressure_before",
    "away_relegation_pressure_before",
    "home_pressure_index_before",
    "away_pressure_index_before",
    "match_pressure_index_before",
]

FORBIDDEN_POST_MATCH_ELO_COLUMNS = {
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}

FORBIDDEN_TOKENS = [
    "h2h",
    "style",
    "poisson",
    "odds",
    "rivalry",
    "derby",
    "manager",
    "sentiment",
    "injury",
]

WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
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
    STANDINGS_TABLE,
    PRESSURE_TABLE,
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
        return set(
            db.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = :table_name
                    """
                ),
                {"table_name": table_name},
            ).scalars()
        )


def _count_table_rows(conn, table_name: str) -> int:
    with conn.connect() as db:
        return int(db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _record_value(value):
    if pandas.isna(value):
        return None
    if isinstance(value, pandas.Timestamp):
        return value.to_pydatetime()
    return value


def _print_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name in WATCHED_TABLES:
        print(f"{table_name}: {counts.get(table_name)}")


def _result_for(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _empty_team_stats() -> dict[str, int]:
    return {
        "games_played_before": 0,
        "wins_before": 0,
        "draws_before": 0,
        "losses_before": 0,
        "points_before": 0,
        "goals_for_before": 0,
        "goals_against_before": 0,
    }


def _apply_team_result(
    stats: dict[str, dict[str, int]],
    team: str,
    goals_for: int,
    goals_against: int,
) -> None:
    row = stats[team]
    row["games_played_before"] += 1
    row["goals_for_before"] += int(goals_for)
    row["goals_against_before"] += int(goals_against)
    if goals_for > goals_against:
        row["wins_before"] += 1
        row["points_before"] += 3
    elif goals_for == goals_against:
        row["draws_before"] += 1
        row["points_before"] += 1
    else:
        row["losses_before"] += 1


def _ranked_table_from_stats(stats: dict[str, dict[str, int]]) -> dict[str, dict]:
    table_rows: list[dict[str, Any]] = []
    for team_name, values in stats.items():
        games_played = values["games_played_before"]
        goal_diff = values["goals_for_before"] - values["goals_against_before"]
        table_rows.append(
            {
                "team_name": team_name,
                **values,
                "goal_diff_before": goal_diff,
                "ppg_before": None
                if games_played == 0
                else values["points_before"] / games_played,
            }
        )

    ranked_rows = sorted(
        table_rows,
        key=lambda row: (
            -row["points_before"],
            -row["goal_diff_before"],
            -row["goals_for_before"],
            row["team_name"],
        ),
    )
    if len(ranked_rows) < 18:
        raise ValueError(
            f"Cannot compute reference gaps with fewer than 18 teams: {len(ranked_rows)}"
        )

    first_points = ranked_rows[0]["points_before"]
    fourth_points = ranked_rows[3]["points_before"]
    sixth_points = ranked_rows[5]["points_before"]
    eighteenth_points = ranked_rows[17]["points_before"]

    ranked_by_team: dict[str, dict] = {}
    for rank, row in enumerate(ranked_rows, start=1):
        row = row.copy()
        row["rank_before"] = rank
        row["points_to_1st_before"] = first_points - row["points_before"]
        row["points_to_4th_before"] = fourth_points - row["points_before"]
        row["points_to_6th_before"] = sixth_points - row["points_before"]
        row["points_above_18th_before"] = row["points_before"] - eighteenth_points
        ranked_by_team[row["team_name"]] = row
    return ranked_by_team


def _standings_for_date(
    season_matches: pandas.DataFrame,
    teams: list[str],
    current_match_date,
) -> dict[str, dict]:
    stats = {team: _empty_team_stats() for team in teams}
    prior_matches = season_matches.loc[
        season_matches["match_date"] < current_match_date
    ].sort_values(["match_date", "kickoff_time", "match_id"])

    for match in prior_matches.itertuples(index=False):
        _apply_team_result(
            stats,
            match.home_team,
            int(match.home_goals),
            int(match.away_goals),
        )
        _apply_team_result(
            stats,
            match.away_team,
            int(match.away_goals),
            int(match.home_goals),
        )

    return _ranked_table_from_stats(stats)


def _stats_from_prior_team_matches(prior_matches: pandas.DataFrame, team: str) -> dict:
    stats = {team: _empty_team_stats()}
    for match in prior_matches.sort_values(["match_date", "kickoff_time", "match_id"]).itertuples(
        index=False
    ):
        if match.home_team == team:
            _apply_team_result(
                stats,
                team,
                int(match.home_goals),
                int(match.away_goals),
            )
        elif match.away_team == team:
            _apply_team_result(
                stats,
                team,
                int(match.away_goals),
                int(match.home_goals),
            )
    row = stats[team]
    row = {
        **row,
        "goal_diff_before": row["goals_for_before"] - row["goals_against_before"],
        "ppg_before": None
        if row["games_played_before"] == 0
        else row["points_before"] / row["games_played_before"],
    }
    return row


def _team_pressure_values(standings_row) -> dict[str, float | None]:
    games_played = int(standings_row.games_played_before)
    if games_played < MIN_GAMES_FOR_PRESSURE:
        return {
            "title_pressure": None,
            "top4_pressure": None,
            "top6_pressure": None,
            "relegation_pressure": None,
            "pressure_index": None,
        }

    phase_weight = min(1.0, max(0.0, (games_played - MIN_GAMES_FOR_PRESSURE) / 22))

    title_gap = float(standings_row.points_to_1st_before)
    title_pressure_raw = max(0.0, (TITLE_WINDOW_POINTS - title_gap) / TITLE_WINDOW_POINTS)
    title_pressure = phase_weight * title_pressure_raw

    top4_gap = abs(float(standings_row.points_to_4th_before))
    top4_pressure_raw = max(0.0, (TOP4_WINDOW_POINTS - top4_gap) / TOP4_WINDOW_POINTS)
    top4_pressure = phase_weight * top4_pressure_raw

    top6_gap = abs(float(standings_row.points_to_6th_before))
    top6_pressure_raw = max(0.0, (TOP6_WINDOW_POINTS - top6_gap) / TOP6_WINDOW_POINTS)
    top6_pressure = phase_weight * top6_pressure_raw

    relegation_gap = abs(float(standings_row.points_above_18th_before))
    relegation_pressure_raw = max(
        0.0,
        (RELEGATION_WINDOW_POINTS - relegation_gap) / RELEGATION_WINDOW_POINTS,
    )
    relegation_pressure = phase_weight * relegation_pressure_raw

    return {
        "title_pressure": title_pressure,
        "top4_pressure": top4_pressure,
        "top6_pressure": top6_pressure,
        "relegation_pressure": relegation_pressure,
        "pressure_index": max(
            title_pressure,
            top4_pressure,
            top6_pressure,
            relegation_pressure,
        ),
    }


def _diff_or_none(left, right) -> float | None:
    if pandas.isna(left) or pandas.isna(right):
        return None
    return float(left - right)


def _max_or_none(values: list[float | None]) -> float | None:
    non_null_values = [float(value) for value in values if not pandas.isna(value)]
    if not non_null_values:
        return None
    return max(non_null_values)


def load_historical_matches(conn) -> pandas.DataFrame:
    if not _table_exists(conn, "historical_matches"):
        raise RuntimeError("historical_matches table does not exist")

    query = text(
        """
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
        ORDER BY season_id, match_date, kickoff_time, match_id
        """
    )
    matches_df = pandas.read_sql(query, conn)
    matches_df["match_date"] = pandas.to_datetime(matches_df["match_date"]).dt.date
    matches_df["kickoff_time"] = pandas.to_datetime(
        matches_df["kickoff_time"],
        errors="coerce",
    )

    errors: list[str] = []
    if len(matches_df) != EXPECTED_MATCH_ROWS:
        errors.append(
            f"historical_matches expected {EXPECTED_MATCH_ROWS} rows, found {len(matches_df)}"
        )
    duplicate_count = int(matches_df["match_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"historical_matches duplicate match_id count: {duplicate_count}")
    same_team_count = int((matches_df["home_team"] == matches_df["away_team"]).sum())
    if same_team_count:
        errors.append(f"historical_matches same-team rows: {same_team_count}")

    result_mismatches = [
        {
            "match_id": int(row.match_id),
            "home_goals": int(row.home_goals),
            "away_goals": int(row.away_goals),
            "result": row.result,
            "expected": _result_for(int(row.home_goals), int(row.away_goals)),
        }
        for row in matches_df.itertuples(index=False)
        if row.result != _result_for(int(row.home_goals), int(row.away_goals))
    ]
    if result_mismatches:
        errors.append(f"result mismatch examples: {result_mismatches[:5]}")

    season_counts = matches_df.groupby("season_id").size().to_dict()
    bad_season_counts = {
        season: int(count)
        for season, count in season_counts.items()
        if int(count) != EXPECTED_SEASON_MATCH_ROWS
    }
    if bad_season_counts:
        errors.append(f"bad season match counts: {bad_season_counts}")

    if errors:
        raise ValueError("Historical match load validation failed: " + "; ".join(errors))

    print(f"Loaded historical_matches rows: {len(matches_df)}")
    return matches_df


def load_elo_features(conn) -> pandas.DataFrame:
    if not _table_exists(conn, "match_features_v3_elo"):
        raise RuntimeError("match_features_v3_elo table does not exist")

    existing_columns = _get_table_columns(conn, "match_features_v3_elo")
    missing_columns = sorted(set(ELO_TABLE_COLUMNS) - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_elo missing required column(s): "
            f"{', '.join(missing_columns)}"
        )
    forbidden_columns = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & existing_columns)
    if forbidden_columns:
        raise RuntimeError(
            "match_features_v3_elo has forbidden post-match Elo column(s): "
            f"{', '.join(forbidden_columns)}"
        )

    column_list = ",\n            ".join(ELO_TABLE_COLUMNS)
    query = text(
        f"""
        SELECT
            {column_list}
        FROM match_features_v3_elo
        ORDER BY season_id, match_date, kickoff_time, match_id
        """
    )
    elo_df = pandas.read_sql(query, conn)
    elo_df["match_date"] = pandas.to_datetime(elo_df["match_date"]).dt.date
    elo_df["kickoff_time"] = pandas.to_datetime(elo_df["kickoff_time"], errors="coerce")

    errors: list[str] = []
    if len(elo_df) != EXPECTED_MATCH_ROWS:
        errors.append(
            f"match_features_v3_elo expected {EXPECTED_MATCH_ROWS} rows, found {len(elo_df)}"
        )
    duplicate_count = int(elo_df["match_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"match_features_v3_elo duplicate match_id count: {duplicate_count}")
    null_counts = elo_df[ID_TARGET_COLUMNS].isna().sum()
    bad_nulls = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if bad_nulls:
        errors.append(f"match_features_v3_elo target/id nulls: {bad_nulls}")

    if errors:
        raise ValueError("Elo feature load validation failed: " + "; ".join(errors))

    print(f"Loaded match_features_v3_elo rows: {len(elo_df)}")
    return elo_df


def get_season_teams(matches_df, season) -> list[str]:
    season_df = matches_df.loc[matches_df["season_id"] == season]
    teams = sorted(set(season_df["home_team"]) | set(season_df["away_team"]))
    if len(teams) < 18:
        raise ValueError(
            f"{season} has fewer than 18 teams; cannot compute table gaps: {len(teams)}"
        )
    if len(teams) != EXPECTED_TEAMS_PER_SEASON:
        raise ValueError(
            f"{season} expected {EXPECTED_TEAMS_PER_SEASON} teams, found {len(teams)}"
        )
    return teams


def compute_standings_before_match(matches_df) -> pandas.DataFrame:
    table_cache: dict[tuple[str, Any], dict[str, dict]] = {}
    rows: list[dict[str, Any]] = []

    for season, season_matches in matches_df.groupby("season_id", sort=True):
        teams = get_season_teams(matches_df, season)
        season_matches = season_matches.sort_values(["match_date", "kickoff_time", "match_id"])
        for match_date in sorted(season_matches["match_date"].unique()):
            table_cache[(season, match_date)] = _standings_for_date(
                season_matches,
                teams,
                match_date,
            )

        for match in season_matches.itertuples(index=False):
            table = table_cache[(match.season_id, match.match_date)]
            for team_name, opponent_team, venue, is_home in [
                (match.home_team, match.away_team, "home", 1),
                (match.away_team, match.home_team, "away", 0),
            ]:
                standing = table[team_name]
                rows.append(
                    {
                        "match_id": int(match.match_id),
                        "season": match.season_id,
                        "match_date": match.match_date,
                        "team_name": team_name,
                        "opponent_team": opponent_team,
                        "venue": venue,
                        "is_home": is_home,
                        **{
                            column: standing[column]
                            for column in STANDINGS_COLUMNS
                            if column
                            not in {
                                "match_id",
                                "season",
                                "match_date",
                                "team_name",
                                "opponent_team",
                                "venue",
                                "is_home",
                            }
                        },
                    }
                )

    standings_df = pandas.DataFrame(rows, columns=STANDINGS_COLUMNS)
    print(f"Computed standings_before_match rows: {len(standings_df)}")
    return standings_df


def compute_pressure_features(standings_df, elo_df) -> pandas.DataFrame:
    home_standings = standings_df.loc[standings_df["is_home"] == 1].set_index("match_id")
    away_standings = standings_df.loc[standings_df["is_home"] == 0].set_index("match_id")

    rows: list[dict[str, Any]] = []
    for match in elo_df.sort_values(["season_id", "match_date", "kickoff_time", "match_id"]).itertuples(
        index=False
    ):
        home = home_standings.loc[match.match_id]
        away = away_standings.loc[match.match_id]
        if home.team_name != match.home_team or away.team_name != match.away_team:
            raise ValueError(
                f"Standings join mismatch for match_id {match.match_id}: "
                f"{home.team_name}/{away.team_name} vs {match.home_team}/{match.away_team}"
            )

        home_pressure = _team_pressure_values(home)
        away_pressure = _team_pressure_values(away)
        match_pressure = _max_or_none(
            [home_pressure["pressure_index"], away_pressure["pressure_index"]]
        )

        row = {column: getattr(match, column) for column in ELO_TABLE_COLUMNS}
        row.update(
            {
                "home_games_played_before": int(home.games_played_before),
                "away_games_played_before": int(away.games_played_before),
                "home_points_before": int(home.points_before),
                "away_points_before": int(away.points_before),
                "home_ppg_before": home.ppg_before,
                "away_ppg_before": away.ppg_before,
                "home_rank_before": int(home.rank_before),
                "away_rank_before": int(away.rank_before),
                "home_goal_diff_before": int(home.goal_diff_before),
                "away_goal_diff_before": int(away.goal_diff_before),
                "rank_diff_before": int(home.rank_before) - int(away.rank_before),
                "points_diff_before": int(home.points_before) - int(away.points_before),
                "ppg_diff_before": _diff_or_none(home.ppg_before, away.ppg_before),
                "goal_diff_table_diff_before": int(home.goal_diff_before)
                - int(away.goal_diff_before),
                "home_title_pressure_before": home_pressure["title_pressure"],
                "away_title_pressure_before": away_pressure["title_pressure"],
                "home_top4_pressure_before": home_pressure["top4_pressure"],
                "away_top4_pressure_before": away_pressure["top4_pressure"],
                "home_top6_pressure_before": home_pressure["top6_pressure"],
                "away_top6_pressure_before": away_pressure["top6_pressure"],
                "home_relegation_pressure_before": home_pressure[
                    "relegation_pressure"
                ],
                "away_relegation_pressure_before": away_pressure[
                    "relegation_pressure"
                ],
                "home_pressure_index_before": home_pressure["pressure_index"],
                "away_pressure_index_before": away_pressure["pressure_index"],
                "match_pressure_index_before": match_pressure,
                "pressure_diff_before": _diff_or_none(
                    home_pressure["pressure_index"],
                    away_pressure["pressure_index"],
                ),
                "season_progress_before": (
                    (int(home.games_played_before) + int(away.games_played_before))
                    / 2
                    / TOTAL_LEAGUE_MATCHES_PER_TEAM
                ),
            }
        )
        rows.append(row)

    pressure_df = pandas.DataFrame(rows, columns=PRESSURE_TABLE_COLUMNS)
    print(f"Computed match pressure feature rows: {len(pressure_df)}")
    return pressure_df


def validate_standings(standings_df, matches_df) -> None:
    print("=== standings_before_match_v3 Validation ===")
    errors: list[str] = []

    if len(standings_df) != EXPECTED_STANDINGS_ROWS:
        errors.append(
            f"expected {EXPECTED_STANDINGS_ROWS} rows, found {len(standings_df)}"
        )

    rows_per_match = standings_df.groupby("match_id").size()
    bad_match_row_counts = rows_per_match.loc[rows_per_match != 2]
    if not bad_match_row_counts.empty:
        errors.append(
            f"matches without exactly 2 standings rows: {bad_match_row_counts.head().to_dict()}"
        )

    home_counts = standings_df.groupby("match_id")["is_home"].sum()
    bad_home_counts = home_counts.loc[home_counts != 1]
    if not bad_home_counts.empty:
        errors.append(
            f"matches without exactly one home row: {bad_home_counts.head().to_dict()}"
        )

    venue_counts = standings_df.groupby(["match_id", "venue"]).size().unstack(fill_value=0)
    if "home" not in venue_counts.columns or "away" not in venue_counts.columns:
        errors.append("home/away venue columns missing from standings rows")
    else:
        bad_venue_rows = venue_counts.loc[
            (venue_counts["home"] != 1) | (venue_counts["away"] != 1)
        ]
        if not bad_venue_rows.empty:
            errors.append(
                f"matches without one home and one away venue row: {bad_venue_rows.head().to_dict(orient='index')}"
            )

    duplicate_count = int(
        standings_df.duplicated(subset=["match_id", "team_name"]).sum()
    )
    if duplicate_count:
        errors.append(f"duplicate match_id/team_name rows: {duplicate_count}")

    numeric_non_negative_columns = [
        "games_played_before",
        "wins_before",
        "draws_before",
        "losses_before",
        "points_before",
        "goals_for_before",
        "goals_against_before",
    ]
    for column in numeric_non_negative_columns:
        bad_count = int((standings_df[column] < 0).sum())
        if bad_count:
            errors.append(f"{column} has {bad_count} negative value(s)")

    points_mismatch_count = int(
        (
            standings_df["points_before"]
            != standings_df["wins_before"] * 3 + standings_df["draws_before"]
        ).sum()
    )
    if points_mismatch_count:
        errors.append(f"points formula mismatch rows: {points_mismatch_count}")

    games_mismatch_count = int(
        (
            standings_df["games_played_before"]
            != standings_df["wins_before"]
            + standings_df["draws_before"]
            + standings_df["losses_before"]
        ).sum()
    )
    if games_mismatch_count:
        errors.append(f"games formula mismatch rows: {games_mismatch_count}")

    goal_diff_mismatch_count = int(
        (
            standings_df["goal_diff_before"]
            != standings_df["goals_for_before"] - standings_df["goals_against_before"]
        ).sum()
    )
    if goal_diff_mismatch_count:
        errors.append(f"goal difference formula mismatch rows: {goal_diff_mismatch_count}")

    bad_rank_count = int((~standings_df["rank_before"].between(1, 20)).sum())
    if bad_rank_count:
        errors.append(f"rank_before outside 1..20 rows: {bad_rank_count}")

    matches_by_id = matches_df.set_index("match_id")
    team_mismatch_count = 0
    stat_mismatch_examples: list[dict[str, Any]] = []
    rank_gap_mismatch_examples: list[dict[str, Any]] = []
    leakage_checked_rows = 0
    table_cache: dict[tuple[str, Any], dict[str, dict]] = {}

    for standing in standings_df.itertuples(index=False):
        match = matches_by_id.loc[standing.match_id]
        expected_team = match["home_team"] if standing.is_home == 1 else match["away_team"]
        expected_opponent = (
            match["away_team"] if standing.is_home == 1 else match["home_team"]
        )
        if standing.team_name != expected_team or standing.opponent_team != expected_opponent:
            team_mismatch_count += 1

        prior_matches = matches_df.loc[
            (matches_df["season_id"] == standing.season)
            & (matches_df["match_date"] < standing.match_date)
            & (
                (matches_df["home_team"] == standing.team_name)
                | (matches_df["away_team"] == standing.team_name)
            )
        ]
        expected_stats = _stats_from_prior_team_matches(
            prior_matches,
            standing.team_name,
        )
        leakage_checked_rows += 1
        for column in [
            "games_played_before",
            "wins_before",
            "draws_before",
            "losses_before",
            "points_before",
            "goals_for_before",
            "goals_against_before",
            "goal_diff_before",
        ]:
            if getattr(standing, column) != expected_stats[column]:
                stat_mismatch_examples.append(
                    {
                        "match_id": standing.match_id,
                        "team_name": standing.team_name,
                        "column": column,
                        "actual": getattr(standing, column),
                        "expected": expected_stats[column],
                    }
                )
                break
        if pandas.isna(standing.ppg_before) != pandas.isna(expected_stats["ppg_before"]):
            stat_mismatch_examples.append(
                {
                    "match_id": standing.match_id,
                    "team_name": standing.team_name,
                    "column": "ppg_before",
                    "actual": standing.ppg_before,
                    "expected": expected_stats["ppg_before"],
                }
            )
        elif not pandas.isna(standing.ppg_before) and not math.isclose(
            float(standing.ppg_before),
            float(expected_stats["ppg_before"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            stat_mismatch_examples.append(
                {
                    "match_id": standing.match_id,
                    "team_name": standing.team_name,
                    "column": "ppg_before",
                    "actual": standing.ppg_before,
                    "expected": expected_stats["ppg_before"],
                }
            )

        cache_key = (standing.season, standing.match_date)
        if cache_key not in table_cache:
            teams = get_season_teams(matches_df, standing.season)
            season_matches = matches_df.loc[matches_df["season_id"] == standing.season]
            table_cache[cache_key] = _standings_for_date(
                season_matches,
                teams,
                standing.match_date,
            )
        expected_rank_row = table_cache[cache_key][standing.team_name]
        for column in [
            "rank_before",
            "points_to_1st_before",
            "points_to_4th_before",
            "points_to_6th_before",
            "points_above_18th_before",
        ]:
            if getattr(standing, column) != expected_rank_row[column]:
                rank_gap_mismatch_examples.append(
                    {
                        "match_id": standing.match_id,
                        "team_name": standing.team_name,
                        "column": column,
                        "actual": getattr(standing, column),
                        "expected": expected_rank_row[column],
                    }
                )
                break

    if team_mismatch_count:
        errors.append(f"home/away standings team mismatch rows: {team_mismatch_count}")
    if stat_mismatch_examples:
        errors.append(f"stat mismatch examples: {stat_mismatch_examples[:5]}")
    if rank_gap_mismatch_examples:
        errors.append(f"rank/gap mismatch examples: {rank_gap_mismatch_examples[:5]}")

    for season, season_df in matches_df.groupby("season_id"):
        first_match_date = min(season_df["match_date"])
        first_date_rows = standings_df.loc[
            (standings_df["season"] == season)
            & (standings_df["match_date"] == first_match_date)
        ]
        non_zero_first_date = int((first_date_rows["games_played_before"] != 0).sum())
        if non_zero_first_date:
            errors.append(
                f"{season} first match date has {non_zero_first_date} non-zero prior games row(s)"
            )

    season_rows = standings_df.groupby("season").size().sort_index()
    print("Standings rows by season:")
    for season, row_count in season_rows.items():
        print(f"- {season}: {int(row_count)}")

    if errors:
        print("Standings validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Standings validation failed")

    print("Standings validation passed.")
    print(
        "Leakage audit passed: "
        f"{leakage_checked_rows} standings rows recomputed using only "
        "same-season matches with prior_match.match_date < current_match.match_date."
    )
    print("Same-date result exclusion passed by recomputation against strict date-less-than windows.")


def validate_pressure_features(pressure_df, elo_df) -> None:
    print("=== match_features_v3_pressure_experiment Validation ===")
    errors: list[str] = []

    if len(pressure_df) != len(elo_df) or len(pressure_df) != EXPECTED_MATCH_ROWS:
        errors.append(
            f"pressure rows {len(pressure_df)} != elo rows {len(elo_df)} "
            f"or expected {EXPECTED_MATCH_ROWS}"
        )
    duplicate_count = int(pressure_df["match_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"duplicate match_id count: {duplicate_count}")
    if pressure_df["match_id"].nunique() != len(pressure_df):
        errors.append("pressure DataFrame does not have one row per match_id")

    missing_columns = sorted(set(PRESSURE_TABLE_COLUMNS) - set(pressure_df.columns))
    if missing_columns:
        errors.append(f"missing pressure table columns: {missing_columns}")

    extra_columns = sorted(set(pressure_df.columns) - set(PRESSURE_TABLE_COLUMNS))
    if extra_columns:
        errors.append(f"unexpected pressure DataFrame columns: {extra_columns}")

    forbidden_columns = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & set(pressure_df.columns))
    if forbidden_columns:
        errors.append(f"forbidden post-match Elo columns present: {forbidden_columns}")

    forbidden_token_columns = sorted(
        column
        for column in pressure_df.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )
    if forbidden_token_columns:
        errors.append(f"forbidden feature-family columns present: {forbidden_token_columns}")

    comparison_columns = ID_TARGET_COLUMNS
    merged = pressure_df[comparison_columns].merge(
        elo_df[comparison_columns],
        on="match_id",
        suffixes=("_pressure", "_elo"),
        how="outer",
        indicator=True,
    )
    non_matched = merged.loc[merged["_merge"] != "both"]
    if not non_matched.empty:
        errors.append(f"target preservation anti-join rows: {len(non_matched)}")

    mismatch_count = 0
    for column in comparison_columns:
        if column == "match_id":
            continue
        mismatch_count += int(
            (
                merged[f"{column}_pressure"] != merged[f"{column}_elo"]
            ).sum()
        )
    if mismatch_count:
        errors.append(f"target/result preservation mismatch count: {mismatch_count}")

    for column in PRESSURE_COMPONENT_COLUMNS:
        non_null = pressure_df[column].dropna()
        bad_count = int((~non_null.between(0, 1)).sum())
        if bad_count:
            errors.append(f"{column} has {bad_count} value(s) outside 0..1")

    bad_progress_count = int((~pressure_df["season_progress_before"].between(0, 1)).sum())
    if bad_progress_count:
        errors.append(f"season_progress_before outside 0..1 rows: {bad_progress_count}")

    for side in ["home", "away"]:
        early_rows = pressure_df.loc[
            pressure_df[f"{side}_games_played_before"] < MIN_GAMES_FOR_PRESSURE
        ]
        late_rows = pressure_df.loc[
            pressure_df[f"{side}_games_played_before"] >= MIN_GAMES_FOR_PRESSURE
        ]
        component_columns = [
            f"{side}_title_pressure_before",
            f"{side}_top4_pressure_before",
            f"{side}_top6_pressure_before",
            f"{side}_relegation_pressure_before",
            f"{side}_pressure_index_before",
        ]
        early_non_null = int(early_rows[component_columns].notna().sum().sum())
        late_null = int(late_rows[component_columns].isna().sum().sum())
        if early_non_null:
            errors.append(
                f"{side} pressure components have {early_non_null} non-null early-game value(s)"
            )
        if late_null:
            errors.append(
                f"{side} pressure components have {late_null} null post-threshold value(s)"
            )

    expected_progress = (
        pressure_df["home_games_played_before"] + pressure_df["away_games_played_before"]
    ) / 2 / TOTAL_LEAGUE_MATCHES_PER_TEAM
    if not expected_progress.equals(pressure_df["season_progress_before"]):
        max_delta = (expected_progress - pressure_df["season_progress_before"]).abs().max()
        if max_delta > 1e-12:
            errors.append(f"season_progress_before formula max delta: {max_delta}")

    expected_rank_diff = pressure_df["home_rank_before"] - pressure_df["away_rank_before"]
    if not expected_rank_diff.equals(pressure_df["rank_diff_before"]):
        errors.append("rank_diff_before formula mismatch")

    expected_points_diff = (
        pressure_df["home_points_before"] - pressure_df["away_points_before"]
    )
    if not expected_points_diff.equals(pressure_df["points_diff_before"]):
        errors.append("points_diff_before formula mismatch")

    expected_goal_diff = (
        pressure_df["home_goal_diff_before"] - pressure_df["away_goal_diff_before"]
    )
    if not expected_goal_diff.equals(pressure_df["goal_diff_table_diff_before"]):
        errors.append("goal_diff_table_diff_before formula mismatch")

    rows_by_season = pressure_df.groupby("season_id").size().sort_index()
    for season, row_count in rows_by_season.items():
        if int(row_count) != EXPECTED_SEASON_MATCH_ROWS:
            errors.append(f"{season} expected 380 pressure rows, found {int(row_count)}")

    print("Pressure null summary:")
    null_counts = pressure_df[PRESSURE_COLUMNS].isna().sum()
    for column, count in null_counts.items():
        print(f"- {column}: {int(count)}")

    print("Pressure rows by season:")
    for season, row_count in rows_by_season.items():
        print(f"- {season}: {int(row_count)}")

    print("Pressure min/max/mean by season:")
    summary = (
        pressure_df.groupby("season_id")[PRESSURE_COLUMNS]
        .agg(["min", "max", "mean"])
        .round(4)
    )
    print(summary.to_string())

    if errors:
        print("Pressure validation failed:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Pressure validation failed")

    print("Pressure feature validation passed.")
    print("Target/result preservation vs match_features_v3_elo: 0 mismatches.")
    print("Forbidden feature-family column check passed.")


def create_pressure_tables(conn) -> None:
    schema_sql = TIER3_SCHEMA_FILE.read_text(encoding="utf-8")
    with conn.begin() as db:
        db.exec_driver_sql(schema_sql)

    for table_name, required_columns in [
        (STANDINGS_TABLE, [*STANDINGS_COLUMNS, "created_at"]),
        (PRESSURE_TABLE, PRESSURE_TABLE_COLUMNS),
    ]:
        if not _table_exists(conn, table_name):
            raise RuntimeError(f"{table_name} table does not exist after schema init")
        existing_columns = _get_table_columns(conn, table_name)
        missing_columns = sorted(set(required_columns) - existing_columns)
        if missing_columns:
            raise RuntimeError(
                f"{table_name} missing required column(s): {', '.join(missing_columns)}"
            )
    print("Pressure table schema verification passed.")


def write_standings(conn, standings_df) -> None:
    records = [
        {column: _record_value(row[column]) for column in STANDINGS_COLUMNS}
        for row in standings_df[STANDINGS_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(STANDINGS_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in STANDINGS_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO {STANDINGS_TABLE} (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )
    with conn.begin() as db:
        db.execute(text(f"DELETE FROM {STANDINGS_TABLE}"))
        db.execute(insert_sql, records)
    print(f"Stored {len(records)} rows in {STANDINGS_TABLE}")


def write_pressure_features(conn, pressure_df) -> None:
    records = [
        {column: _record_value(row[column]) for column in PRESSURE_TABLE_COLUMNS}
        for row in pressure_df[PRESSURE_TABLE_COLUMNS].to_dict(orient="records")
    ]
    column_list = ",\n            ".join(PRESSURE_TABLE_COLUMNS)
    value_list = ",\n            ".join(f":{column}" for column in PRESSURE_TABLE_COLUMNS)
    insert_sql = text(
        f"""
        INSERT INTO {PRESSURE_TABLE} (
            {column_list}
        )
        VALUES (
            {value_list}
        )
        """
    )
    with conn.begin() as db:
        db.execute(text(f"DELETE FROM {PRESSURE_TABLE}"))
        db.execute(insert_sql, records)
    print(f"Stored {len(records)} rows in {PRESSURE_TABLE}")


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    for table_name in WATCHED_TABLES:
        if _table_exists(conn, table_name):
            counts[table_name] = _count_table_rows(conn, table_name)
        else:
            counts[table_name] = "MISSING"
    return counts


def assert_watched_counts_unchanged_except_pressure(before, after) -> None:
    allowed_changes = {STANDINGS_TABLE, PRESSURE_TABLE}
    changed_unexpectedly = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if table_name not in allowed_changes
        and before.get(table_name) != after.get(table_name)
    }
    if changed_unexpectedly:
        raise RuntimeError(
            "Watched table counts changed unexpectedly: "
            f"{changed_unexpectedly}"
        )
    if after.get(STANDINGS_TABLE) != EXPECTED_STANDINGS_ROWS:
        raise RuntimeError(
            f"{STANDINGS_TABLE} expected {EXPECTED_STANDINGS_ROWS} rows, "
            f"found {after.get(STANDINGS_TABLE)}"
        )
    if after.get(PRESSURE_TABLE) != EXPECTED_MATCH_ROWS:
        raise RuntimeError(
            f"{PRESSURE_TABLE} expected {EXPECTED_MATCH_ROWS} rows, "
            f"found {after.get(PRESSURE_TABLE)}"
        )
    print("Watched table counts unchanged except pressure experiment tables.")
    print(f"{STANDINGS_TABLE}: {before.get(STANDINGS_TABLE)} -> {after.get(STANDINGS_TABLE)}")
    print(f"{PRESSURE_TABLE}: {before.get(PRESSURE_TABLE)} -> {after.get(PRESSURE_TABLE)}")


def _print_db_summary(conn) -> None:
    with conn.connect() as db:
        standings_count = db.execute(
            text(f"SELECT COUNT(*) FROM {STANDINGS_TABLE}")
        ).scalar_one()
        pressure_count = db.execute(
            text(f"SELECT COUNT(*) FROM {PRESSURE_TABLE}")
        ).scalar_one()
        pressure_rows_by_season = db.execute(
            text(
                f"""
                SELECT season_id, COUNT(*) AS row_count
                FROM {PRESSURE_TABLE}
                GROUP BY season_id
                ORDER BY season_id
                """
            )
        ).mappings().all()
        null_select = ",\n                ".join(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
            for column in PRESSURE_COLUMNS
        )
        null_counts = db.execute(
            text(f"SELECT {null_select} FROM {PRESSURE_TABLE}")
        ).mappings().one()

    print(f"{STANDINGS_TABLE} row count: {standings_count}")
    print(f"{PRESSURE_TABLE} row count: {pressure_count}")
    print(f"{PRESSURE_TABLE} rows by season:")
    for row in pressure_rows_by_season:
        print(f"- {row['season_id']}: {row['row_count']}")
    print(f"{PRESSURE_TABLE} null summary:")
    for column in PRESSURE_COLUMNS:
        print(f"- {column}: {null_counts[column]}")


def main() -> None:
    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)
    _print_counts("Watched table counts before pressure write", before_counts)

    matches_df = load_historical_matches(conn)
    elo_df = load_elo_features(conn)
    standings_df = compute_standings_before_match(matches_df)
    validate_standings(standings_df, matches_df)
    pressure_df = compute_pressure_features(standings_df, elo_df)
    validate_pressure_features(pressure_df, elo_df)

    create_pressure_tables(conn)
    write_standings(conn, standings_df)
    write_pressure_features(conn, pressure_df)
    _print_db_summary(conn)

    after_counts = capture_watched_table_counts(conn)
    _print_counts("Watched table counts after pressure write", after_counts)
    assert_watched_counts_unchanged_except_pressure(before_counts, after_counts)

    print("Source tables used: historical_matches, match_features_v3_elo")
    print("No final league table, final rank, future matches, same-date results, or same-match result features were used.")
    print("No rivalry/derby, H2H, style, Poisson, odds, manager, sentiment, injury, deployment, app work, model training, or model artifacts occurred.")


if __name__ == "__main__":
    main()
