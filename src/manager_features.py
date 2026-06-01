import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATE_COLUMNS = [
    "team_name",
    "manager_name",
    "start_date",
    "end_date",
    "source_note",
]


# Returns the manager history CSV template path.
def get_template_path():
    return PROJECT_ROOT / "data" / "raw" / "manager_history_template.csv"


# Creates one blank manager-history template row per team in PostgreSQL.
def create_manager_template(engine, force=False):
    path = get_template_path()

    if path.exists() and not force:
        print(f"Manager history template already exists: {path.relative_to(PROJECT_ROOT).as_posix()}")
        return path

    teams_df = pd.read_sql("SELECT name FROM teams ORDER BY team_id", engine)
    template_df = pd.DataFrame(
        {
            "team_name": teams_df["name"],
            "manager_name": "",
            "start_date": "",
            "end_date": "",
            "source_note": "",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(path, index=False)

    print(f"Created manager history template: {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Teams included: {len(template_df)}")
    return path


# Loads and validates the raw CSV shape and date formats.
def load_manager_history_csv():
    path = get_template_path()

    if not path.exists():
        raise FileNotFoundError(f"Manager history template not found: {path}")

    df = pd.read_csv(path, keep_default_na=False)
    missing_columns = [col for col in TEMPLATE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[TEMPLATE_COLUMNS].copy()
    for column in TEMPLATE_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    for idx, row in df.iterrows():
        row_number = idx + 2
        for column in ["start_date", "end_date"]:
            value = row[column]
            if value:
                try:
                    pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid {column} on CSV row {row_number}: {value}. "
                        "Use YYYY-MM-DD."
                    ) from exc

    return df


# Validates completed manager rows against teams, dates, and overlap rules.
def validate_manager_history(df, engine):
    teams_df = pd.read_sql("SELECT name FROM teams", engine)
    valid_teams = set(teams_df["name"].tolist())

    df = df.copy()
    optional_cols = ["manager_name", "start_date", "end_date", "source_note"]
    blank_mask = (
        df["team_name"].ne("")
        & df[optional_cols].eq("").all(axis=1)
    )
    intended_mask = ~blank_mask

    intended_df = df[intended_mask].copy()
    completed_mask = (
        intended_df["team_name"].ne("")
        & intended_df["manager_name"].ne("")
        & intended_df["start_date"].ne("")
    )
    completed_df = intended_df[completed_mask].copy()

    unknown_teams = sorted(
        team for team in df["team_name"].dropna().unique()
        if team and team not in valid_teams
    )

    missing_team_count = int(intended_df["team_name"].eq("").sum())
    missing_manager_count = int((
        intended_df["team_name"].ne("")
        & intended_df["manager_name"].eq("")
        & intended_df[["start_date", "end_date", "source_note"]].ne("").any(axis=1)
    ).sum())
    missing_start_count = int((
        intended_df["team_name"].ne("")
        & intended_df["manager_name"].ne("")
        & intended_df["start_date"].eq("")
    ).sum())

    if missing_team_count:
        print(f"WARNING: Rows intended to load missing team_name: {missing_team_count}")
    if missing_manager_count:
        print(f"WARNING: Rows intended to load missing manager_name: {missing_manager_count}")
    if missing_start_count:
        print(f"WARNING: Rows intended to load missing start_date: {missing_start_count}")

    if len(completed_df) > 0:
        completed_df["start_date_parsed"] = pd.to_datetime(
            completed_df["start_date"], format="%Y-%m-%d"
        ).dt.date
        completed_df["end_date_parsed"] = pd.to_datetime(
            completed_df["end_date"].replace("", pd.NA),
            format="%Y-%m-%d",
            errors="coerce",
        ).dt.date

        invalid_ranges = completed_df[
            completed_df["end_date_parsed"].notna()
            & (completed_df["start_date_parsed"] > completed_df["end_date_parsed"])
        ]
        if len(invalid_ranges) > 0:
            raise ValueError(f"Manager rows with start_date after end_date: {len(invalid_ranges)}")
    else:
        completed_df["start_date_parsed"] = []
        completed_df["end_date_parsed"] = []

    overlap_count = 0
    if len(completed_df) > 0:
        for _, team_df in completed_df.sort_values(
            ["team_name", "start_date_parsed"]
        ).groupby("team_name"):
            previous_end = None
            for _, row in team_df.iterrows():
                start = row["start_date_parsed"]
                end = row["end_date_parsed"] if pd.notna(row["end_date_parsed"]) else date.max
                if previous_end is not None and start <= previous_end:
                    overlap_count += 1
                previous_end = max(previous_end, end) if previous_end else end

    print(f"Manager rows completed: {len(completed_df)}")
    print(f"Blank template rows: {int(blank_mask.sum())}")
    print(f"Unknown teams: {unknown_teams}")
    print(f"Overlapping ranges: {overlap_count}")

    if unknown_teams:
        raise ValueError(f"Unknown team names found: {unknown_teams}")
    if overlap_count:
        raise ValueError(f"Overlapping manager ranges found: {overlap_count}")

    return completed_df[TEMPLATE_COLUMNS].copy()


# Creates/recreates manager_history and loads completed manager rows only.
def load_manager_history_to_postgres(df, engine):
    drop_sql = "DROP TABLE IF EXISTS manager_history;"
    create_sql = """
    CREATE TABLE manager_history (
        id SERIAL PRIMARY KEY,
        team_name VARCHAR(100),
        manager_name VARCHAR(100),
        start_date DATE,
        end_date DATE,
        source_note TEXT
    );
    """

    with engine.begin() as conn:
        conn.execute(text(drop_sql))
        conn.execute(text(create_sql))

    load_df = df.copy()
    for column in ["start_date", "end_date"]:
        load_df[column] = load_df[column].replace("", pd.NA)

    if len(load_df) > 0:
        load_df.to_sql("manager_history", engine, if_exists="append", index=False)

    print(f"Loaded {len(load_df)} manager history rows into manager_history")


# Verifies whether manager_history exists and contains usable rows.
def verify_manager_history(engine):
    count = pd.read_sql("SELECT COUNT(*) AS count FROM manager_history", engine)["count"].iloc[0]

    if count == 0:
        print("manager_history is empty. Fill data/raw/manager_history_template.csv before generating features.")
        return

    teams_covered = pd.read_sql(
        "SELECT COUNT(DISTINCT team_name) AS count FROM manager_history",
        engine,
    )["count"].iloc[0]
    date_range = pd.read_sql(
        "SELECT MIN(start_date) AS min_date, MAX(COALESCE(end_date, start_date)) AS max_date FROM manager_history",
        engine,
    )

    print(f"manager_history row count: {count}")
    print(f"manager_history teams covered: {teams_covered}")
    print(
        "manager_history date range: "
        f"{date_range['min_date'].iloc[0]} to {date_range['max_date'].iloc[0]}"
    )


# Runs the manager-history template, validation, load, and verification workflow.
def main():
    from src.data_pipeline import get_engine

    engine = get_engine()
    if engine is None:
        print("Failed to connect to PostgreSQL.")
        return

    if not get_template_path().exists():
        create_manager_template(engine)

    df = load_manager_history_csv()
    completed_df = validate_manager_history(df, engine)
    load_manager_history_to_postgres(completed_df, engine)
    verify_manager_history(engine)


if __name__ == "__main__":
    main()
