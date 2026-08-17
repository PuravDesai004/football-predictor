from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import pandas
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_SEASON = "2026-27"
FEATURE_ARTIFACT = PROJECT_ROOT / "models" / "saved" / "production_features_v3.json"
SOURCE_FIXTURE_TABLE = "production_fpl_fixture_snapshots"
SOURCE_BOOTSTRAP_TABLE = "production_fpl_bootstrap_snapshots"
ELO_CURRENT_TABLE = "elo_current_v3"
OUTPUT_TABLE = "production_upcoming_match_features_v3"
TEAM_MAPPING_TABLE = "production_team_name_mapping"
HOME_ADVANTAGE = 50.0
PROMOTED_ELO = 1400.0

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import get_engine  # noqa: E402


EXPECTED_FEATURE_COUNT = 32
COUNT_FEATURE_COLUMNS = {
    "home_home_matches_last5",
    "away_away_matches_last5",
    "home_overall_matches_last5",
    "away_overall_matches_last5",
    "home_overall_matches_last10",
    "away_overall_matches_last10",
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
    "league_code",
]
FORBIDDEN_EXACT_FEATURES = {
    "match_id",
    "season",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    "result",
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
FPL_MODEL_TEAM_ALIASES = {
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
}
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "elo_current_v3",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "match_features_v3_style_experiment",
    "match_features_v3_h2h_experiment",
    "production_ingestion_runs",
    "production_fpl_bootstrap_snapshots",
    "production_fpl_fixture_snapshots",
    "production_football_data_match_staging",
    "production_understat_xg_staging",
    "production_data_freshness",
    "production_prediction_runs",
    "production_match_predictions",
    TEAM_MAPPING_TABLE,
    OUTPUT_TABLE,
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]
ALLOWED_CHANGED_TABLES = {TEAM_MAPPING_TABLE, OUTPUT_TABLE}
TEAM_STATE_COLUMNS = [
    "team_name",
    "home_home_matches_last5",
    "home_goals_scored_home_last5",
    "home_goals_conceded_home_last5",
    "home_clean_sheet_rate_home_last5",
    "home_xg_home_last5",
    "home_xga_home_last5",
    "away_away_matches_last5",
    "away_goals_scored_away_last5",
    "away_goals_conceded_away_last5",
    "away_clean_sheet_rate_away_last5",
    "away_xg_away_last5",
    "away_xga_away_last5",
    "overall_matches_last5",
    "overall_matches_last10",
    "points_overall_last5",
    "points_overall_last10",
    "goal_diff_overall_last5",
    "xg_overall_last5",
    "xga_overall_last5",
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_feature_artifact() -> list[str]:
    if not FEATURE_ARTIFACT.exists():
        raise FileNotFoundError(f"Missing production feature artifact: {FEATURE_ARTIFACT}")
    with FEATURE_ARTIFACT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    feature_columns = payload.get("features")
    if not isinstance(feature_columns, list):
        raise ValueError("Feature artifact missing list field: features")
    feature_columns = [str(column) for column in feature_columns]

    errors: list[str] = []
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        errors.append(
            f"feature count expected {EXPECTED_FEATURE_COUNT}, found {len(feature_columns)}"
        )
    duplicate_features = sorted(
        {column for column in feature_columns if feature_columns.count(column) > 1}
    )
    forbidden_exact = sorted(set(feature_columns) & FORBIDDEN_EXACT_FEATURES)
    forbidden_tokens = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if duplicate_features:
        errors.append(f"duplicate feature(s): {duplicate_features}")
    if forbidden_exact:
        errors.append(f"forbidden exact feature(s): {forbidden_exact}")
    if forbidden_tokens:
        errors.append(f"forbidden feature family token(s): {forbidden_tokens}")
    if errors:
        raise ValueError("Production feature artifact validation failed: " + "; ".join(errors))

    print(f"PASS: loaded production feature artifact with {len(feature_columns)} features.")
    return feature_columns


def create_upcoming_feature_tables(conn, feature_columns) -> None:
    feature_definitions = ",\n            ".join(
        f"{column} {_feature_sql_type(column)} NULL" for column in feature_columns
    )
    statements = [
        """
        CREATE TABLE IF NOT EXISTS production_team_name_mapping (
            fpl_team_id INTEGER PRIMARY KEY,
            fpl_team_name TEXT NOT NULL,
            model_team_name TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            source TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            upcoming_feature_id BIGSERIAL PRIMARY KEY,
            target_season TEXT NOT NULL,
            target_gameweek INTEGER NULL,
            fixture_id INTEGER NOT NULL,
            kickoff_time TIMESTAMP NOT NULL,
            match_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            {feature_definitions},
            feature_generated_at TIMESTAMP NOT NULL,
            source_snapshot_run_id INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (target_season, fixture_id),
            CHECK (target_gameweek IS NULL OR target_gameweek > 0),
            CHECK (home_team <> away_team)
        )
        """,
    ]
    with conn.begin() as db_conn:
        for statement in statements:
            db_conn.execute(text(statement))
    _verify_upcoming_feature_tables(conn, feature_columns)
    print("PASS: production upcoming feature tables exist and required columns are present.")


def load_latest_fpl_snapshot_metadata(conn) -> dict:
    with conn.connect() as db_conn:
        fixture_row = db_conn.execute(
            text(
                f"""
                SELECT
                    run_id,
                    COUNT(*) AS fixture_rows,
                    MIN(kickoff_time) AS min_kickoff_time,
                    MAX(kickoff_time) AS max_kickoff_time,
                    SUM(
                        CASE
                            WHEN finished = FALSE AND kickoff_time IS NOT NULL
                            THEN 1 ELSE 0
                        END
                    ) AS upcoming_rows
                FROM {SOURCE_FIXTURE_TABLE}
                GROUP BY run_id
                ORDER BY run_id DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        bootstrap_row = db_conn.execute(
            text(
                f"""
                SELECT
                    run_id,
                    COUNT(*) AS bootstrap_rows,
                    COUNT(DISTINCT team_id) AS team_count,
                    MAX(snapshot_time) AS snapshot_time,
                    MAX(deadline_time) AS deadline_time
                FROM {SOURCE_BOOTSTRAP_TABLE}
                GROUP BY run_id
                ORDER BY run_id DESC
                LIMIT 1
                """
            )
        ).mappings().first()

    if fixture_row is None:
        raise RuntimeError(f"{SOURCE_FIXTURE_TABLE} has no fixture snapshots")
    if bootstrap_row is None:
        raise RuntimeError(f"{SOURCE_BOOTSTRAP_TABLE} has no bootstrap snapshots")

    metadata = {
        "fixture_run_id": int(fixture_row["run_id"]),
        "fixture_rows": int(fixture_row["fixture_rows"]),
        "fixture_upcoming_rows": int(fixture_row["upcoming_rows"] or 0),
        "fixture_min_kickoff_time": fixture_row["min_kickoff_time"],
        "fixture_max_kickoff_time": fixture_row["max_kickoff_time"],
        "bootstrap_run_id": int(bootstrap_row["run_id"]),
        "bootstrap_rows": int(bootstrap_row["bootstrap_rows"]),
        "bootstrap_team_count": int(bootstrap_row["team_count"]),
        "bootstrap_snapshot_time": bootstrap_row["snapshot_time"],
        "bootstrap_deadline_time": bootstrap_row["deadline_time"],
    }
    print("Latest FPL snapshot metadata:")
    print(_json_dumps_mapping(metadata))
    return metadata


def seed_or_validate_team_mapping(conn, snapshot_metadata) -> None:
    bootstrap_run_id = int(snapshot_metadata["bootstrap_run_id"])
    with conn.connect() as db_conn:
        team_rows = db_conn.execute(
            text(
                f"""
                SELECT
                    team_id AS fpl_team_id,
                    team_name AS fpl_team_name,
                    COUNT(*) AS player_rows
                FROM {SOURCE_BOOTSTRAP_TABLE}
                WHERE run_id = :run_id
                GROUP BY team_id, team_name
                ORDER BY team_id
                """
            ),
            {"run_id": bootstrap_run_id},
        ).mappings().all()
        model_teams = {
            row["team_name"]
            for row in db_conn.execute(
                text(
                    """
                    SELECT team_name
                    FROM elo_current_v3
                    UNION
                    SELECT home_team AS team_name FROM historical_matches
                    UNION
                    SELECT away_team AS team_name FROM historical_matches
                    """
                )
            ).mappings().all()
        }

    if not team_rows:
        raise RuntimeError(f"No teams found in bootstrap run_id={bootstrap_run_id}")
    if len(team_rows) != 20:
        raise RuntimeError(
            f"Expected 20 FPL teams in bootstrap run_id={bootstrap_run_id}, "
            f"found {len(team_rows)}"
        )

    records = []
    cold_start_teams: list[str] = []
    for row in team_rows:
        fpl_team_name = str(row["fpl_team_name"]).strip()
        model_team_name = _resolve_model_team_name(fpl_team_name, model_teams)
        if model_team_name is None:
            model_team_name = fpl_team_name
            cold_start_teams.append(fpl_team_name)
            source = f"fpl_bootstrap_run_{bootstrap_run_id}:new_team_cold_start"
            print(
                "WARNING: new FPL team has no historical model-team match; "
                f"using exact-name cold-start mapping for {fpl_team_name}"
            )
        else:
            source = f"fpl_bootstrap_run_{bootstrap_run_id}"
        records.append(
            {
                "fpl_team_id": int(row["fpl_team_id"]),
                "fpl_team_name": fpl_team_name,
                "model_team_name": model_team_name,
                "source": source,
            }
        )

    query = text(
        f"""
        INSERT INTO {TEAM_MAPPING_TABLE} (
            fpl_team_id,
            fpl_team_name,
            model_team_name,
            is_active,
            source
        )
        VALUES (
            :fpl_team_id,
            :fpl_team_name,
            :model_team_name,
            TRUE,
            :source
        )
        ON CONFLICT (fpl_team_id) DO UPDATE SET
            fpl_team_name = EXCLUDED.fpl_team_name,
            model_team_name = EXCLUDED.model_team_name,
            is_active = TRUE,
            source = EXCLUDED.source
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(query, records)

    mapping_df = _load_active_team_mapping(conn)
    mapped_model_teams = set(mapping_df["model_team_name"])
    missing_model_teams = sorted(set(record["model_team_name"] for record in records) - mapped_model_teams)
    if missing_model_teams:
        raise RuntimeError(f"Team mapping write failed for: {missing_model_teams}")

    print(f"PASS: seeded/validated {len(records)} active FPL team mappings.")
    if cold_start_teams:
        print("New teams reported as cold-start teams:")
        for team_name in sorted(cold_start_teams):
            print(f"- {team_name}")
    else:
        print("New teams reported as cold-start teams: none")
    for record in records:
        print(
            f"- {record['fpl_team_id']}: "
            f"{record['fpl_team_name']} -> {record['model_team_name']}"
        )


def load_upcoming_fixtures(conn, target_season, target_gameweek=None) -> pandas.DataFrame:
    snapshot_metadata = load_latest_fpl_snapshot_metadata(conn)
    params: dict[str, Any] = {"run_id": int(snapshot_metadata["fixture_run_id"])}
    where_clauses = [
        "run_id = :run_id",
        "finished = FALSE",
        "kickoff_time IS NOT NULL",
    ]
    if target_gameweek is not None:
        where_clauses.append("event_id = :target_gameweek")
        params["target_gameweek"] = int(target_gameweek)

    query = text(
        f"""
        SELECT
            run_id AS source_snapshot_run_id,
            event_id AS target_gameweek,
            fixture_id,
            kickoff_time,
            team_h_id,
            team_a_id,
            team_h_name,
            team_a_name,
            finished,
            started,
            finished_provisional
        FROM {SOURCE_FIXTURE_TABLE}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY kickoff_time, fixture_id
        """
    )
    fixtures_df = pandas.read_sql(query, conn, params=params)
    fixtures_df["target_season"] = target_season
    if not fixtures_df.empty:
        fixtures_df["kickoff_time"] = pandas.to_datetime(fixtures_df["kickoff_time"])
        fixtures_df["match_date"] = fixtures_df["kickoff_time"].dt.date

    print(
        f"Upcoming fixtures found: {len(fixtures_df)} "
        f"from fixture snapshot run_id={snapshot_metadata['fixture_run_id']}"
    )
    _print_upcoming_rows_by_gameweek(fixtures_df)
    return fixtures_df


def map_fixture_teams(conn, fixtures_df) -> pandas.DataFrame:
    if fixtures_df.empty:
        print("No upcoming fixtures to map.")
        return _empty_mapped_fixture_frame()

    mapping_df = _load_active_team_mapping(conn)
    home_mapping = mapping_df.rename(
        columns={
            "fpl_team_id": "team_h_id",
            "model_team_name": "home_team",
            "fpl_team_name": "home_fpl_team_name",
        }
    )[["team_h_id", "home_fpl_team_name", "home_team"]]
    away_mapping = mapping_df.rename(
        columns={
            "fpl_team_id": "team_a_id",
            "model_team_name": "away_team",
            "fpl_team_name": "away_fpl_team_name",
        }
    )[["team_a_id", "away_fpl_team_name", "away_team"]]

    mapped_df = fixtures_df.merge(home_mapping, on="team_h_id", how="left").merge(
        away_mapping,
        on="team_a_id",
        how="left",
    )
    missing_home = mapped_df.loc[mapped_df["home_team"].isna(), "team_h_name"].dropna().unique()
    missing_away = mapped_df.loc[mapped_df["away_team"].isna(), "team_a_name"].dropna().unique()
    missing = sorted(set(missing_home) | set(missing_away))
    if missing:
        raise RuntimeError(
            "Fixture team mapping missing for FPL team name(s): " + ", ".join(missing)
        )
    print(f"PASS: mapped teams for {len(mapped_df)} upcoming fixture row(s).")
    return mapped_df


def load_current_elo(conn) -> pandas.DataFrame:
    if not _table_exists_for_engine(conn, ELO_CURRENT_TABLE):
        raise RuntimeError(f"{ELO_CURRENT_TABLE} table does not exist")
    query = text(
        f"""
        SELECT team_name, elo_rating, source_season, source_match_id
        FROM {ELO_CURRENT_TABLE}
        ORDER BY team_name
        """
    )
    elo_df = pandas.read_sql(query, conn)
    if elo_df.empty:
        raise RuntimeError(f"{ELO_CURRENT_TABLE} has no rows")
    if elo_df["team_name"].duplicated().any():
        raise RuntimeError(f"{ELO_CURRENT_TABLE} has duplicate team_name rows")
    if elo_df[["team_name", "elo_rating"]].isna().any().any():
        raise RuntimeError(f"{ELO_CURRENT_TABLE} has null team_name or elo_rating")
    print(f"Loaded current Elo rows: {len(elo_df)}")
    return elo_df


def load_latest_team_feature_state(conn) -> pandas.DataFrame:
    matches_df = _load_completed_matches_with_xg(conn)
    team_rows = _build_team_perspective_rows(matches_df)
    rows: list[dict[str, Any]] = []
    for team_name in sorted(team_rows["team"].unique()):
        home_last5 = _prior_team_rows(team_rows, team_name, venue="H").head(5)
        away_last5 = _prior_team_rows(team_rows, team_name, venue="A").head(5)
        overall_last5 = _prior_team_rows(team_rows, team_name).head(5)
        overall_last10 = _prior_team_rows(team_rows, team_name).head(10)
        rows.append(
            {
                "team_name": team_name,
                "home_home_matches_last5": len(home_last5),
                "home_goals_scored_home_last5": _mean_or_none(home_last5["goals_for"]),
                "home_goals_conceded_home_last5": _mean_or_none(home_last5["goals_against"]),
                "home_clean_sheet_rate_home_last5": _clean_sheet_rate(home_last5),
                "home_xg_home_last5": _mean_or_none(home_last5["xg_for"]),
                "home_xga_home_last5": _mean_or_none(home_last5["xg_against"]),
                "away_away_matches_last5": len(away_last5),
                "away_goals_scored_away_last5": _mean_or_none(away_last5["goals_for"]),
                "away_goals_conceded_away_last5": _mean_or_none(away_last5["goals_against"]),
                "away_clean_sheet_rate_away_last5": _clean_sheet_rate(away_last5),
                "away_xg_away_last5": _mean_or_none(away_last5["xg_for"]),
                "away_xga_away_last5": _mean_or_none(away_last5["xg_against"]),
                "overall_matches_last5": len(overall_last5),
                "overall_matches_last10": len(overall_last10),
                "points_overall_last5": _mean_or_none(overall_last5["points"]),
                "points_overall_last10": _mean_or_none(overall_last10["points"]),
                "goal_diff_overall_last5": _goal_diff_mean(overall_last5),
                "xg_overall_last5": _mean_or_none(overall_last5["xg_for"]),
                "xga_overall_last5": _mean_or_none(overall_last5["xg_against"]),
            }
        )
    state_df = pandas.DataFrame(rows, columns=TEAM_STATE_COLUMNS)
    print(f"Loaded latest team feature state rows: {len(state_df)}")
    return state_df


def build_feature_rows(
    fixtures_df,
    elo_df,
    team_state_df,
    feature_columns,
    snapshot_metadata,
) -> pandas.DataFrame:
    if fixtures_df.empty:
        _print_missing_team_summary([], [])
        return _empty_feature_frame(feature_columns)

    elo_by_team = elo_df.set_index("team_name")
    state_by_team = team_state_df.set_index("team_name")
    fixture_teams = sorted(set(fixtures_df["home_team"]) | set(fixtures_df["away_team"]))
    missing_elo = sorted(team for team in fixture_teams if team not in elo_by_team.index)
    missing_state = sorted(team for team in fixture_teams if team not in state_by_team.index)
    _print_missing_team_summary(missing_elo, missing_state)
    if missing_elo:
        print(f"WARNING: using in-memory promoted-team Elo fallback {PROMOTED_ELO:.1f} for:")
        for team_name in missing_elo:
            print(f"- {team_name}")
            elo_by_team.loc[team_name, "elo_rating"] = PROMOTED_ELO
    if missing_state:
        print("WARNING: using in-memory zero rolling-state cold start for:")
        cold_start_state = {
            column: 0.0
            for column in TEAM_STATE_COLUMNS
            if column != "team_name"
        }
        for team_name in missing_state:
            print(f"- {team_name}")
            state_by_team.loc[team_name] = cold_start_state

    generated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures_df.sort_values(["kickoff_time", "fixture_id"]).itertuples(index=False):
        home_state = state_by_team.loc[fixture.home_team]
        away_state = state_by_team.loc[fixture.away_team]
        home_elo = float(elo_by_team.loc[fixture.home_team, "elo_rating"])
        away_elo = float(elo_by_team.loc[fixture.away_team, "elo_rating"])
        elo_diff = home_elo - away_elo
        elo_diff_home_adjusted = elo_diff + HOME_ADVANTAGE
        expected_home = 1 / (1 + 10 ** (-elo_diff_home_adjusted / 400))
        row = {
            "target_season": fixture.target_season,
            "target_gameweek": _nullable_int(fixture.target_gameweek),
            "fixture_id": int(fixture.fixture_id),
            "kickoff_time": _to_timestamp(fixture.kickoff_time),
            "match_date": _to_date(fixture.match_date),
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "feature_generated_at": generated_at,
            "source_snapshot_run_id": int(fixture.source_snapshot_run_id),
            "home_home_matches_last5": int(home_state["home_home_matches_last5"]),
            "away_away_matches_last5": int(away_state["away_away_matches_last5"]),
            "home_overall_matches_last5": int(home_state["overall_matches_last5"]),
            "away_overall_matches_last5": int(away_state["overall_matches_last5"]),
            "home_overall_matches_last10": int(home_state["overall_matches_last10"]),
            "away_overall_matches_last10": int(away_state["overall_matches_last10"]),
            "home_goals_scored_home_last5": home_state["home_goals_scored_home_last5"],
            "home_goals_conceded_home_last5": home_state["home_goals_conceded_home_last5"],
            "home_clean_sheet_rate_home_last5": home_state["home_clean_sheet_rate_home_last5"],
            "away_goals_scored_away_last5": away_state["away_goals_scored_away_last5"],
            "away_goals_conceded_away_last5": away_state["away_goals_conceded_away_last5"],
            "away_clean_sheet_rate_away_last5": away_state["away_clean_sheet_rate_away_last5"],
            "home_xg_home_last5": home_state["home_xg_home_last5"],
            "home_xga_home_last5": home_state["home_xga_home_last5"],
            "away_xg_away_last5": away_state["away_xg_away_last5"],
            "away_xga_away_last5": away_state["away_xga_away_last5"],
            "home_points_overall_last5": home_state["points_overall_last5"],
            "away_points_overall_last5": away_state["points_overall_last5"],
            "home_points_overall_last10": home_state["points_overall_last10"],
            "away_points_overall_last10": away_state["points_overall_last10"],
            "home_goal_diff_overall_last5": home_state["goal_diff_overall_last5"],
            "away_goal_diff_overall_last5": away_state["goal_diff_overall_last5"],
            "home_xg_overall_last5": home_state["xg_overall_last5"],
            "home_xga_overall_last5": home_state["xga_overall_last5"],
            "away_xg_overall_last5": away_state["xg_overall_last5"],
            "away_xga_overall_last5": away_state["xga_overall_last5"],
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "elo_diff_before": elo_diff,
            "elo_diff_home_adjusted": elo_diff_home_adjusted,
            "expected_home_score": float(expected_home),
            "expected_away_score": float(1 - expected_home),
        }
        rows.append(row)

    feature_df = pandas.DataFrame(rows)
    missing_feature_columns = sorted(set(feature_columns) - set(feature_df.columns))
    if missing_feature_columns:
        raise RuntimeError(f"Internal missing feature column(s): {missing_feature_columns}")

    output_columns = [
        "target_season",
        "target_gameweek",
        "fixture_id",
        "kickoff_time",
        "match_date",
        "home_team",
        "away_team",
        *feature_columns,
        "feature_generated_at",
        "source_snapshot_run_id",
    ]
    feature_df = feature_df[output_columns].copy()
    print(
        f"Built {len(feature_df)} upcoming feature row(s) "
        f"from FPL fixture snapshot run_id={snapshot_metadata['fixture_run_id']}"
    )
    return feature_df


def validate_feature_rows(feature_df, feature_columns) -> None:
    if feature_df.empty:
        print("Upcoming feature rows: 0")
        print("Missing feature summary: no rows to validate.")
        return

    errors: list[str] = []
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        errors.append(f"feature artifact count is {len(feature_columns)}")
    missing_columns = sorted(set(feature_columns) - set(feature_df.columns))
    if missing_columns:
        errors.append(f"missing output feature column(s): {missing_columns}")
    forbidden_feature_names = sorted(set(feature_columns) & FORBIDDEN_EXACT_FEATURES)
    forbidden_feature_tokens = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    forbidden_output_tokens = sorted(
        column
        for column in feature_df.columns
        if any(token in column.lower() for token in ["league_code"])
    )
    if forbidden_feature_names:
        errors.append(f"forbidden feature name(s): {forbidden_feature_names}")
    if forbidden_feature_tokens:
        errors.append(f"forbidden feature family token(s): {forbidden_feature_tokens}")
    if forbidden_output_tokens:
        errors.append(f"forbidden output column token(s): {forbidden_output_tokens}")
    required_metadata = [
        "target_season",
        "fixture_id",
        "kickoff_time",
        "match_date",
        "home_team",
        "away_team",
        "feature_generated_at",
        "source_snapshot_run_id",
    ]
    metadata_nulls = {
        column: int(feature_df[column].isna().sum())
        for column in required_metadata
        if column in feature_df and int(feature_df[column].isna().sum()) > 0
    }
    if metadata_nulls:
        errors.append(f"null required metadata count(s): {metadata_nulls}")
    if feature_df["fixture_id"].duplicated().any():
        errors.append(f"duplicate fixture_id count: {int(feature_df['fixture_id'].duplicated().sum())}")
    same_team = int((feature_df["home_team"] == feature_df["away_team"]).sum())
    if same_team:
        errors.append(f"home_team equals away_team for {same_team} row(s)")

    missing_feature_summary = {
        column: int(feature_df[column].isna().sum())
        for column in feature_columns
        if column in feature_df and int(feature_df[column].isna().sum()) > 0
    }
    print("Missing feature summary:")
    if missing_feature_summary:
        for column, count in missing_feature_summary.items():
            print(f"- {column}: {count}")
        errors.append(f"missing production feature values: {missing_feature_summary}")
    else:
        print("- none")

    non_numeric_features = [
        column
        for column in feature_columns
        if column in feature_df and not pandas.api.types.is_numeric_dtype(feature_df[column])
    ]
    if non_numeric_features:
        errors.append(f"non-numeric feature column(s): {non_numeric_features}")
    for column in COUNT_FEATURE_COLUMNS:
        if column in feature_df and (~feature_df[column].between(0, 10)).any():
            errors.append(f"{column} has count outside expected range")
    if not numpy.allclose(
        feature_df["expected_home_score"] + feature_df["expected_away_score"],
        1.0,
    ):
        errors.append("expected_home_score + expected_away_score != 1")
    if not numpy.allclose(
        feature_df["elo_diff_before"],
        feature_df["home_elo_before"] - feature_df["away_elo_before"],
    ):
        errors.append("elo_diff_before formula mismatch")
    if not numpy.allclose(
        feature_df["elo_diff_home_adjusted"],
        feature_df["elo_diff_before"] + HOME_ADVANTAGE,
    ):
        errors.append("elo_diff_home_adjusted formula mismatch")

    _print_upcoming_rows_by_gameweek(feature_df)
    if errors:
        raise ValueError("Upcoming feature row validation failed: " + "; ".join(errors))
    print(f"PASS: upcoming feature rows validated ({len(feature_df)} rows).")


def write_upcoming_features(conn, feature_df, replace=False) -> int:
    if feature_df.empty:
        print("Wrote 0 upcoming feature rows.")
        return 0

    _validate_rows_have_source_fixtures(conn, feature_df)
    records = [_db_safe_record(row) for row in feature_df.to_dict(orient="records")]
    if replace:
        with conn.begin() as db_conn:
            for row in records:
                db_conn.execute(
                    text(
                        f"""
                        DELETE FROM {OUTPUT_TABLE}
                        WHERE target_season = :target_season
                            AND fixture_id = :fixture_id
                        """
                    ),
                    {
                        "target_season": row["target_season"],
                        "fixture_id": row["fixture_id"],
                    },
                )
    else:
        _assert_no_existing_upcoming_features(conn, feature_df)

    insert_columns = [
        "target_season",
        "target_gameweek",
        "fixture_id",
        "kickoff_time",
        "match_date",
        "home_team",
        "away_team",
        *[
            column
            for column in feature_df.columns
            if column
            not in {
                "target_season",
                "target_gameweek",
                "fixture_id",
                "kickoff_time",
                "match_date",
                "home_team",
                "away_team",
                "feature_generated_at",
                "source_snapshot_run_id",
            }
        ],
        "feature_generated_at",
        "source_snapshot_run_id",
    ]
    column_sql = ",\n            ".join(insert_columns)
    value_sql = ",\n            ".join(f":{column}" for column in insert_columns)
    query = text(
        f"""
        INSERT INTO {OUTPUT_TABLE} (
            {column_sql}
        )
        VALUES (
            {value_sql}
        )
        """
    )
    with conn.begin() as db_conn:
        db_conn.execute(query, [{column: row[column] for column in insert_columns} for row in records])
    print(f"Wrote {len(records)} upcoming feature rows.")
    return len(records)


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            if _table_exists(db_conn, table_name):
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_counts_unchanged_except_upcoming(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    unexpected = {
        table_name: counts
        for table_name, counts in changed.items()
        if table_name not in ALLOWED_CHANGED_TABLES
    }
    if unexpected:
        raise RuntimeError(f"Unexpected watched table count changes: {unexpected}")

    print("PASS: watched non-upcoming table counts unchanged.")
    if changed:
        print("Allowed upcoming table count changes:")
        for table_name in sorted(changed):
            print(f"- {table_name}: {changed[table_name][0]} -> {changed[table_name][1]}")
    else:
        print("No watched table counts changed.")


def main() -> None:
    args = parse_args()
    conn = get_db_connection()
    feature_columns = load_feature_artifact()
    create_upcoming_feature_tables(conn, feature_columns)
    snapshot_metadata = load_latest_fpl_snapshot_metadata(conn)
    seed_or_validate_team_mapping(conn, snapshot_metadata)

    if args.init_schema_only:
        print("=== Production P3B schema initialization only ===")
        print_upcoming_table_counts(conn)
        return

    before_counts = capture_watched_table_counts(conn)
    fixtures_df = load_upcoming_fixtures(
        conn,
        target_season=args.target_season,
        target_gameweek=args.target_gameweek,
    )
    mapped_fixtures_df = map_fixture_teams(conn, fixtures_df)
    elo_df = load_current_elo(conn)
    team_state_df = load_latest_team_feature_state(conn)
    feature_df = build_feature_rows(
        mapped_fixtures_df,
        elo_df,
        team_state_df,
        feature_columns,
        snapshot_metadata,
    )
    validate_feature_rows(feature_df, feature_columns)
    rows_written = write_upcoming_features(conn, feature_df, replace=args.replace)
    after_counts = capture_watched_table_counts(conn)
    assert_counts_unchanged_except_upcoming(before_counts, after_counts)
    print_watched_count_comparison(before_counts, after_counts)
    print(f"Upcoming feature rows written: {rows_written}")
    print("No fake fixtures, teams, features, or predictions were created.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production P3B upcoming feature builder")
    parser.add_argument("--init-schema-only", action="store_true")
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--target-gameweek", type=int, default=None)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def print_upcoming_table_counts(conn) -> None:
    with conn.connect() as db_conn:
        for table_name in [TEAM_MAPPING_TABLE, OUTPUT_TABLE]:
            count = int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
            print(f"{table_name}: {count}")


def print_watched_count_comparison(before_counts, after_counts) -> None:
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        print(f"- {table_name}: {before_counts.get(table_name)} -> {after_counts.get(table_name)}")


def _verify_upcoming_feature_tables(conn, feature_columns) -> None:
    required_columns = {
        TEAM_MAPPING_TABLE: [
            "fpl_team_id",
            "fpl_team_name",
            "model_team_name",
            "is_active",
            "source",
            "created_at",
        ],
        OUTPUT_TABLE: [
            "upcoming_feature_id",
            "target_season",
            "target_gameweek",
            "fixture_id",
            "kickoff_time",
            "match_date",
            "home_team",
            "away_team",
            *feature_columns,
            "feature_generated_at",
            "source_snapshot_run_id",
            "created_at",
        ],
    }
    with conn.connect() as db_conn:
        for table_name, expected_columns in required_columns.items():
            if not _table_exists(db_conn, table_name):
                raise RuntimeError(f"{table_name} table does not exist")
            existing_columns = [
                row["column_name"]
                for row in db_conn.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = CURRENT_SCHEMA()
                            AND table_name = :table_name
                        """
                    ),
                    {"table_name": table_name},
                ).mappings().all()
            ]
            missing = sorted(set(expected_columns) - set(existing_columns))
            if missing:
                raise RuntimeError(f"{table_name} missing required column(s): {missing}")


def _load_active_team_mapping(conn) -> pandas.DataFrame:
    query = text(
        f"""
        SELECT fpl_team_id, fpl_team_name, model_team_name, is_active, source
        FROM {TEAM_MAPPING_TABLE}
        WHERE is_active = TRUE
        ORDER BY fpl_team_id
        """
    )
    mapping_df = pandas.read_sql(query, conn)
    if mapping_df.empty:
        raise RuntimeError(f"{TEAM_MAPPING_TABLE} has no active team mappings")
    if mapping_df["fpl_team_id"].duplicated().any():
        raise RuntimeError(f"{TEAM_MAPPING_TABLE} has duplicate fpl_team_id rows")
    return mapping_df


def _resolve_model_team_name(fpl_team_name: str, model_teams: set[str]) -> str | None:
    if fpl_team_name in model_teams:
        return fpl_team_name
    alias = FPL_MODEL_TEAM_ALIASES.get(fpl_team_name)
    if alias in model_teams:
        return alias
    normalized_fpl = _normalize_team_text(fpl_team_name)
    for model_team in sorted(model_teams):
        if _normalize_team_text(model_team) == normalized_fpl:
            return model_team
    return None


def _normalize_team_text(value: str) -> str:
    return (
        value.lower()
        .replace("'", "")
        .replace(".", "")
        .replace(" ", "")
        .replace("&", "and")
    )


def _load_completed_matches_with_xg(conn) -> pandas.DataFrame:
    query = text(
        """
        SELECT
            hm.match_id,
            hm.season_id,
            hm.match_date,
            hm.kickoff_time,
            hm.home_team,
            hm.away_team,
            hm.home_goals,
            hm.away_goals,
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
    )
    matches_df = pandas.read_sql(query, conn)
    if matches_df.empty:
        raise RuntimeError("No completed historical match/xG rows available for team state")
    if matches_df[["home_xg", "away_xg"]].isna().any().any():
        raise RuntimeError("Completed historical match/xG state has null xG values")
    matches_df["match_date"] = pandas.to_datetime(matches_df["match_date"]).dt.date
    matches_df["kickoff_time"] = pandas.to_datetime(matches_df["kickoff_time"], errors="coerce")
    matches_df["event_time"] = matches_df["kickoff_time"].fillna(
        pandas.to_datetime(matches_df["match_date"])
    )
    print(f"Loaded completed historical match/xG rows for state: {len(matches_df)}")
    return matches_df


def _build_team_perspective_rows(matches_df: pandas.DataFrame) -> pandas.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in matches_df.sort_values(["event_time", "match_id"]).itertuples(index=False):
        rows.append(
            {
                "match_id": int(match.match_id),
                "team": match.home_team,
                "venue": "H",
                "event_time": match.event_time,
                "goals_for": int(match.home_goals),
                "goals_against": int(match.away_goals),
                "xg_for": float(match.home_xg),
                "xg_against": float(match.away_xg),
                "points": _points_for(int(match.home_goals), int(match.away_goals)),
            }
        )
        rows.append(
            {
                "match_id": int(match.match_id),
                "team": match.away_team,
                "venue": "A",
                "event_time": match.event_time,
                "goals_for": int(match.away_goals),
                "goals_against": int(match.home_goals),
                "xg_for": float(match.away_xg),
                "xg_against": float(match.home_xg),
                "points": _points_for(int(match.away_goals), int(match.home_goals)),
            }
        )
    team_rows = pandas.DataFrame(rows)
    return team_rows.sort_values(["event_time", "match_id"], ascending=[False, False])


def _prior_team_rows(team_rows: pandas.DataFrame, team_name: str, venue: str | None = None) -> pandas.DataFrame:
    prior_rows = team_rows.loc[team_rows["team"] == team_name]
    if venue is not None:
        prior_rows = prior_rows.loc[prior_rows["venue"] == venue]
    return prior_rows.sort_values(["event_time", "match_id"], ascending=[False, False])


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _mean_or_none(series: pandas.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(series.mean())


def _clean_sheet_rate(rows: pandas.DataFrame) -> float | None:
    if rows.empty:
        return None
    return float((rows["goals_against"] == 0).mean())


def _goal_diff_mean(rows: pandas.DataFrame) -> float | None:
    if rows.empty:
        return None
    return float((rows["goals_for"] - rows["goals_against"]).mean())


def _print_missing_team_summary(missing_elo: list[str], missing_state: list[str]) -> None:
    print("Teams without current Elo:")
    print("- none" if not missing_elo else "- " + "\n- ".join(missing_elo))
    print("Teams without rolling feature state:")
    print("- none" if not missing_state else "- " + "\n- ".join(missing_state))


def _print_upcoming_rows_by_gameweek(df: pandas.DataFrame) -> None:
    print("Upcoming rows by gameweek:")
    if df.empty:
        print("- none")
        return
    for gameweek, count in df.groupby("target_gameweek").size().sort_index().items():
        print(f"- {int(gameweek)}: {int(count)}")


def _empty_mapped_fixture_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        columns=[
            "source_snapshot_run_id",
            "target_gameweek",
            "fixture_id",
            "kickoff_time",
            "team_h_id",
            "team_a_id",
            "team_h_name",
            "team_a_name",
            "target_season",
            "match_date",
            "home_team",
            "away_team",
        ]
    )


def _empty_feature_frame(feature_columns: list[str]) -> pandas.DataFrame:
    return pandas.DataFrame(
        columns=[
            "target_season",
            "target_gameweek",
            "fixture_id",
            "kickoff_time",
            "match_date",
            "home_team",
            "away_team",
            *feature_columns,
            "feature_generated_at",
            "source_snapshot_run_id",
        ]
    )


def _assert_no_existing_upcoming_features(conn, feature_df: pandas.DataFrame) -> None:
    with conn.connect() as db_conn:
        for row in feature_df.to_dict(orient="records"):
            duplicate_count = int(
                db_conn.execute(
                    text(
                        f"""
                        SELECT COUNT(*)
                        FROM {OUTPUT_TABLE}
                        WHERE target_season = :target_season
                            AND fixture_id = :fixture_id
                        """
                    ),
                    {
                        "target_season": row["target_season"],
                        "fixture_id": int(row["fixture_id"]),
                    },
                ).scalar_one()
            )
            if duplicate_count:
                raise RuntimeError(
                    "Existing upcoming feature row found for "
                    f"{row['target_season']} fixture_id={row['fixture_id']}; "
                    "use --replace to refresh."
                )


def _validate_rows_have_source_fixtures(conn, feature_df: pandas.DataFrame) -> None:
    missing: list[str] = []
    with conn.connect() as db_conn:
        for row in feature_df.to_dict(orient="records"):
            count = int(
                db_conn.execute(
                    text(
                        f"""
                        SELECT COUNT(*)
                        FROM {SOURCE_FIXTURE_TABLE}
                        WHERE run_id = :run_id
                            AND fixture_id = :fixture_id
                            AND finished = FALSE
                            AND kickoff_time IS NOT NULL
                        """
                    ),
                    {
                        "run_id": int(row["source_snapshot_run_id"]),
                        "fixture_id": int(row["fixture_id"]),
                    },
                ).scalar_one()
            )
            if count != 1:
                missing.append(
                    f"run_id={row['source_snapshot_run_id']} fixture_id={row['fixture_id']}"
                )
    if missing:
        raise RuntimeError(
            "Upcoming feature row(s) missing real unfinished FPL fixture source: "
            + ", ".join(missing[:5])
        )
    print("PASS: every upcoming feature row has a real unfinished FPL fixture source.")


def _feature_sql_type(column: str) -> str:
    return "INTEGER" if column in COUNT_FEATURE_COLUMNS else "FLOAT"


def _table_exists_for_engine(conn, table_name: str) -> bool:
    with conn.connect() as db_conn:
        return _table_exists(db_conn, table_name)


def _table_exists(db_conn, table_name: str) -> bool:
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


def _to_date(value):
    if value is None or pandas.isna(value):
        return None
    parsed = pandas.to_datetime(value, errors="coerce")
    if pandas.isna(parsed):
        raise ValueError(f"Unparseable match_date: {value}")
    return parsed.date()


def _to_timestamp(value):
    if value is None or pandas.isna(value):
        return None
    parsed = pandas.to_datetime(value, utc=True, errors="coerce")
    if pandas.isna(parsed):
        raise ValueError(f"Unparseable kickoff_time: {value}")
    return parsed.to_pydatetime().replace(tzinfo=None)


def _nullable_int(value):
    if value is None or pandas.isna(value):
        return None
    return int(value)


def _db_safe_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _db_safe_value(value) for key, value in row.items()}


def _db_safe_value(value):
    if isinstance(value, pandas.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.to_pydatetime()
    if isinstance(value, float) and pandas.isna(value):
        return None
    if pandas.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _json_dumps_mapping(row) -> str:
    if row is None:
        return "{}"
    return json.dumps(
        {key: _json_safe(value) for key, value in dict(row).items()},
        sort_keys=True,
        default=str,
    )


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
