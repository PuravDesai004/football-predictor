from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pandas
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "fpl_v3_data_quality_audit.md"
DB_CONNECT_TIMEOUT_SECONDS = 5

HISTORY_TABLE = "fpl_player_gameweek_history_v3"
PROTECTED_TIER2_TABLES = [
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "player_gameweek_history",
    "player_gameweek_features",
]

REQUIRED_AUDIT_COLUMNS = [
    "season",
    "gameweek",
    "player_name",
    "source_file",
    "total_points",
    "minutes",
]

OPTIONAL_AUDIT_COLUMNS = [
    "team_name",
    "opponent_team_name",
    "position",
    "player_source_id",
    "fixture_id",
    "kickoff_time",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


def get_database_url():
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


def _row_count(engine, table_name: str):
    if not _table_exists(engine, table_name):
        return "MISSING"
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def _read_sql(engine, query: str, params: dict | None = None) -> pandas.DataFrame:
    with engine.connect() as conn:
        return pandas.read_sql(text(query), conn, params=params or {})


def get_table_counts(engine) -> dict:
    tables = [HISTORY_TABLE, *PROTECTED_TIER2_TABLES]
    return {table: _row_count(engine, table) for table in tables}


def load_history_summary(engine) -> pandas.DataFrame:
    query = f"""
        SELECT 'total_rows' AS metric, COUNT(*)::TEXT AS value FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'seasons', COUNT(DISTINCT season)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'min_season', MIN(season)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'max_season', MAX(season)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'min_gameweek', MIN(gameweek)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'max_gameweek', MAX(gameweek)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'distinct_player_name', COUNT(DISTINCT player_name)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'distinct_player_source_id', COUNT(DISTINCT player_source_id)::TEXT FROM {HISTORY_TABLE}
        UNION ALL
        SELECT 'distinct_source_file', COUNT(DISTINCT source_file)::TEXT FROM {HISTORY_TABLE}
    """
    return _read_sql(engine, query)


def _column_null_audit(engine, columns: list[str]) -> pandas.DataFrame:
    total_rows = int(_row_count(engine, HISTORY_TABLE))
    rows = []
    for column in columns:
        if column in {"season", "player_name", "source_file", "team_name", "position"}:
            condition = f"{column} IS NULL OR TRIM({column}) = ''"
        else:
            condition = f"{column} IS NULL"

        with engine.connect() as conn:
            null_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE {condition}")
            ).scalar_one()
        rows.append(
            {
                "column_name": column,
                "null_count": int(null_count),
                "null_rate": round(int(null_count) / total_rows, 6) if total_rows else 0,
            }
        )
    return pandas.DataFrame(rows)


def audit_required_columns(engine) -> pandas.DataFrame:
    return _column_null_audit(engine, REQUIRED_AUDIT_COLUMNS)


def audit_optional_columns(engine) -> pandas.DataFrame:
    return _column_null_audit(engine, OPTIONAL_AUDIT_COLUMNS).sort_values(
        ["null_count", "column_name"],
        ascending=[False, True],
    )


def audit_season_coverage(engine) -> pandas.DataFrame:
    query = f"""
        SELECT
            season,
            COUNT(*) AS rows,
            COUNT(DISTINCT player_name) AS distinct_players,
            MIN(gameweek) AS min_gameweek,
            MAX(gameweek) AS max_gameweek,
            COUNT(DISTINCT source_file) AS source_files,
            COUNT(*) FILTER (WHERE minutes IS NOT NULL) AS rows_minutes_not_null,
            COUNT(*) FILTER (WHERE total_points IS NOT NULL) AS rows_total_points_not_null,
            COUNT(*) FILTER (WHERE team_name IS NOT NULL AND TRIM(team_name) <> '') AS rows_team_name_not_null,
            COUNT(*) FILTER (WHERE position IS NOT NULL AND TRIM(position) <> '') AS rows_position_not_null,
            COUNT(*) FILTER (WHERE expected_goals IS NOT NULL) AS rows_expected_goals_not_null
        FROM {HISTORY_TABLE}
        GROUP BY season
        ORDER BY season
    """
    return _read_sql(engine, query)


def audit_team_position_coverage(engine) -> pandas.DataFrame:
    coverage = _read_sql(
        engine,
        f"""
        SELECT
            'season_coverage' AS audit_section,
            season,
            NULL::TEXT AS player_name,
            COUNT(*) AS rows,
            ROUND(
                AVG(CASE WHEN team_name IS NULL OR TRIM(team_name) = '' THEN 1 ELSE 0 END)::NUMERIC,
                6
            )::FLOAT AS team_name_null_rate,
            ROUND(
                AVG(CASE WHEN position IS NULL OR TRIM(position) = '' THEN 1 ELSE 0 END)::NUMERIC,
                6
            )::FLOAT AS position_null_rate
        FROM {HISTORY_TABLE}
        GROUP BY season
        """,
    )
    missing_team = _read_sql(
        engine,
        f"""
        SELECT
            'top_missing_team_name' AS audit_section,
            season,
            player_name,
            COUNT(*) AS rows,
            NULL::FLOAT AS team_name_null_rate,
            NULL::FLOAT AS position_null_rate
        FROM {HISTORY_TABLE}
        WHERE team_name IS NULL OR TRIM(team_name) = ''
        GROUP BY season, player_name
        ORDER BY rows DESC, season, player_name
        LIMIT 30
        """,
    )
    missing_position = _read_sql(
        engine,
        f"""
        SELECT
            'top_missing_position' AS audit_section,
            season,
            player_name,
            COUNT(*) AS rows,
            NULL::FLOAT AS team_name_null_rate,
            NULL::FLOAT AS position_null_rate
        FROM {HISTORY_TABLE}
        WHERE position IS NULL OR TRIM(position) = ''
        GROUP BY season, player_name
        ORDER BY rows DESC, season, player_name
        LIMIT 30
        """,
    )
    frames = [df for df in [coverage, missing_team, missing_position] if not df.empty]
    records = []
    for frame in frames:
        records.extend(frame.to_dict("records"))
    return pandas.DataFrame(records)


def audit_player_identity_stability(engine) -> pandas.DataFrame:
    same_name_many_ids = _read_sql(
        engine,
        f"""
        SELECT
            'same_name_multiple_ids_within_season' AS audit_type,
            season,
            player_name,
            NULL::TEXT AS player_source_id,
            COUNT(DISTINCT player_source_id) AS distinct_id_count,
            COUNT(*) AS rows,
            STRING_AGG(DISTINCT player_source_id, ', ' ORDER BY player_source_id) AS examples
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NOT NULL
        GROUP BY season, player_name
        HAVING COUNT(DISTINCT player_source_id) > 1
        ORDER BY distinct_id_count DESC, rows DESC, season, player_name
        LIMIT 30
        """,
    )
    same_id_many_names = _read_sql(
        engine,
        f"""
        SELECT
            'same_id_multiple_names_within_season' AS audit_type,
            season,
            NULL::TEXT AS player_name,
            player_source_id,
            COUNT(DISTINCT player_name) AS distinct_id_count,
            COUNT(*) AS rows,
            STRING_AGG(DISTINCT player_name, ', ' ORDER BY player_name) AS examples
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NOT NULL
        GROUP BY season, player_source_id
        HAVING COUNT(DISTINCT player_name) > 1
        ORDER BY distinct_id_count DESC, rows DESC, season, player_source_id
        LIMIT 30
        """,
    )
    global_id_many_names = _read_sql(
        engine,
        f"""
        SELECT
            'same_id_multiple_names_global' AS audit_type,
            NULL::TEXT AS season,
            NULL::TEXT AS player_name,
            player_source_id,
            COUNT(DISTINCT player_name) AS distinct_id_count,
            COUNT(*) AS rows,
            STRING_AGG(DISTINCT season || ':' || player_name, ', ' ORDER BY season || ':' || player_name) AS examples
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NOT NULL
        GROUP BY player_source_id
        HAVING COUNT(DISTINCT player_name) > 1
        ORDER BY distinct_id_count DESC, rows DESC, player_source_id
        LIMIT 30
        """,
    )

    frames = [same_name_many_ids, same_id_many_names, global_id_many_names]
    return pandas.concat(frames, ignore_index=True) if frames else pandas.DataFrame()


def audit_duplicate_players(engine) -> pandas.DataFrame:
    name_key = _read_sql(
        engine,
        f"""
        SELECT
            'name_fixture_source_file' AS duplicate_key,
            season,
            gameweek,
            player_name,
            NULL::TEXT AS player_source_id,
            fixture_id,
            source_file,
            COUNT(*) AS rows
        FROM {HISTORY_TABLE}
        GROUP BY season, gameweek, player_name, fixture_id, source_file
        HAVING COUNT(*) > 1
        ORDER BY rows DESC, season, gameweek, player_name
        LIMIT 50
        """,
    )
    id_key = _read_sql(
        engine,
        f"""
        SELECT
            'id_aware_source_file' AS duplicate_key,
            season,
            gameweek,
            NULL::TEXT AS player_name,
            player_source_id,
            fixture_id,
            source_file,
            COUNT(*) AS rows
        FROM {HISTORY_TABLE}
        WHERE player_source_id IS NOT NULL
        GROUP BY season, gameweek, player_source_id, fixture_id, source_file
        HAVING COUNT(*) > 1
        ORDER BY rows DESC, season, gameweek, player_source_id
        LIMIT 50
        """,
    )
    return pandas.concat([name_key, id_key], ignore_index=True)


def audit_gameweek_anomalies(engine) -> pandas.DataFrame:
    checks = [
        (
            "gameweek_outside_1_to_47",
            f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE gameweek IS NULL OR gameweek < 1 OR gameweek > 47",
        ),
        (
            "seasons_more_than_38_unique_gameweeks",
            f"""
            SELECT COUNT(*) FROM (
                SELECT season
                FROM {HISTORY_TABLE}
                GROUP BY season
                HAVING COUNT(DISTINCT gameweek) > 38
            ) seasons
            """,
        ),
        ("minutes_less_than_zero", f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE minutes < 0"),
        ("total_points_null", f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE total_points IS NULL"),
        ("total_points_gt_30", f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE total_points > 30"),
        ("value_lte_zero_not_null", f"SELECT COUNT(*) FROM {HISTORY_TABLE} WHERE value IS NOT NULL AND value <= 0"),
    ]
    rows = []
    with engine.connect() as conn:
        for check_name, query in checks:
            rows.append(
                {
                    "check_name": check_name,
                    "count": int(conn.execute(text(query)).scalar_one()),
                }
            )
    return pandas.DataFrame(rows)


def audit_training_readiness(engine) -> dict:
    coverage = audit_season_coverage(engine)
    readiness_rows = []
    for _, row in coverage.iterrows():
        rows = int(row["rows"])
        source_files = int(row["source_files"])
        points_rate = row["rows_total_points_not_null"] / rows if rows else 0
        minutes_rate = row["rows_minutes_not_null"] / rows if rows else 0
        team_rate = row["rows_team_name_not_null"] / rows if rows else 0
        position_rate = row["rows_position_not_null"] / rows if rows else 0

        reasons = []
        if points_rate < 0.999:
            reasons.append("missing target_total_points")
        if minutes_rate < 0.999:
            reasons.append("missing minutes")
        if rows < 15000:
            reasons.append("low row count")
        if source_files < 37:
            reasons.append("low source file coverage")
        if team_rate < 0.95:
            reasons.append("team_name coverage below 95%")
        if position_rate < 0.95:
            reasons.append("position coverage below 95%")

        if not reasons:
            status = "READY"
        elif points_rate >= 0.999 and minutes_rate >= 0.999 and rows >= 15000:
            status = "PARTIAL"
        else:
            status = "NOT_READY"

        if row["season"] == "2025-26":
            reasons.append("reserved: do not use for model selection/tuning")

        readiness_rows.append(
            {
                "season": row["season"],
                "status": status,
                "rows": rows,
                "source_files": source_files,
                "points_not_null_rate": round(points_rate, 6),
                "minutes_not_null_rate": round(minutes_rate, 6),
                "team_name_not_null_rate": round(team_rate, 6),
                "position_not_null_rate": round(position_rate, 6),
                "notes": "; ".join(reasons) if reasons else "ready for modeling after feature build",
            }
        )

    identity = audit_player_identity_stability(engine)
    global_id_reuse = 0
    if not identity.empty:
        global_id_reuse = len(identity[identity["audit_type"] == "same_id_multiple_names_global"])

    return {
        "readiness": pandas.DataFrame(readiness_rows),
        "identity_recommendation": (
            "Create fpl_player_identity_map_v3 before modeling. "
            "player_source_id should be treated as season-local, not globally stable."
            if global_id_reuse > 0
            else "player_source_id appears globally stable in this audit, but a v3 identity map is still recommended before production FPL work."
        ),
        "holdout_warning": (
            "2025-26 is available as raw imported history only. Do not use it for model "
            "selection or tuning; reserve it for final FPL evaluation if selected as holdout."
        ),
    }


def _git_status(pathspec: str) -> str:
    result = subprocess.run(
        ["git", "status", "--short", pathspec],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip()


def _df_to_markdown(df: pandas.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._\n"

    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    display_df = display_df.astype(object).where(pandas.notna(display_df), "")
    columns = [str(column) for column in display_df.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in display_df.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in display_df.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def write_markdown_report(results, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training = results["training_readiness"]
    before_counts = results["table_counts_before"]
    after_counts = results["table_counts_after"]

    lines = [
        "# FPL v3 Data Quality Audit",
        "",
        "Phase: Tier 3 FPL Phase 1C",
        "",
        "This audit is read-only against the database. It does not train models, tune metrics, or modify Tier 2 tables.",
        "",
        "## Holdout Warning",
        "",
        training["holdout_warning"],
        "",
        "## Global Counts",
        "",
        _df_to_markdown(results["history_summary"]),
        "## Season Coverage",
        "",
        _df_to_markdown(results["season_coverage"]),
        "## Required Column Audit",
        "",
        _df_to_markdown(results["required_columns"]),
        "## Optional Column Audit",
        "",
        _df_to_markdown(results["optional_columns"]),
        "## Team And Position Coverage",
        "",
        _df_to_markdown(results["team_position_coverage"], max_rows=80),
        "## Player Identity Stability",
        "",
        training["identity_recommendation"],
        "",
        _df_to_markdown(results["identity_stability"], max_rows=90),
        "## Duplicate And Anomaly Checks",
        "",
        "### Duplicate Player Checks",
        "",
        _df_to_markdown(results["duplicate_players"], max_rows=100),
        "### Gameweek And Value Anomalies",
        "",
        _df_to_markdown(results["gameweek_anomalies"]),
        "## Training Readiness",
        "",
        _df_to_markdown(training["readiness"]),
        "## Protected Tier 2 Table Safety",
        "",
        "| table | before | after |",
        "| --- | ---: | ---: |",
    ]
    for table in PROTECTED_TIER2_TABLES:
        lines.append(f"| {table} | {before_counts.get(table)} | {after_counts.get(table)} |")

    lines.extend(
        [
            "",
            f"`models/saved` git status: `{results['models_saved_status'] or 'clean'}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    engine = get_engine()
    before_counts = get_table_counts(engine)

    results = {
        "table_counts_before": before_counts,
        "history_summary": load_history_summary(engine),
        "required_columns": audit_required_columns(engine),
        "optional_columns": audit_optional_columns(engine),
        "season_coverage": audit_season_coverage(engine),
        "team_position_coverage": audit_team_position_coverage(engine),
        "identity_stability": audit_player_identity_stability(engine),
        "duplicate_players": audit_duplicate_players(engine),
        "gameweek_anomalies": audit_gameweek_anomalies(engine),
        "training_readiness": audit_training_readiness(engine),
    }
    after_counts = get_table_counts(engine)
    results["table_counts_after"] = after_counts
    results["models_saved_status"] = _git_status("models/saved")

    protected_changed = {
        table: (before_counts.get(table), after_counts.get(table))
        for table in PROTECTED_TIER2_TABLES
        if before_counts.get(table) != after_counts.get(table)
    }
    if protected_changed:
        raise RuntimeError(f"Protected Tier 2 table counts changed: {protected_changed}")

    write_markdown_report(results, OUTPUT_PATH)

    total_rows = results["history_summary"].loc[
        results["history_summary"]["metric"] == "total_rows",
        "value",
    ].iloc[0]
    print("PASS: FPL v3 data quality audit completed.")
    print(f"FPL v3 history rows: {total_rows}")
    print("PASS: protected Tier 2 table counts unchanged.")
    print(f"Markdown report written: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"models/saved status: {results['models_saved_status'] or 'clean'}")


if __name__ == "__main__":
    main()
