import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


# Prints a row count for a SQL view/table and keeps running if verification fails.
def verify_row_count(engine, object_name):
    try:
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {object_name}")).scalar()

        print(f"{object_name} rows: {row_count}")
    except Exception as error:
        print(f"WARNING: Could not verify {object_name}: {error}")


# Prints the pandas shape for a SQL view and keeps running if verification fails.
def verify_view_shape(engine, object_name):
    try:
        df = pd.read_sql(text(f"SELECT * FROM {object_name}"), engine)
        print(f"{object_name} shape: {df.shape}")
    except Exception as error:
        print(f"WARNING: Could not verify {object_name}: {error}")


# Prints mature style-form row count for clustering and keeps running if verification fails.
def verify_style_mature_rows(engine):
    object_name = "team_style_form rows with style_matches_last5 >= 3"

    try:
        with engine.connect() as conn:
            row_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM team_style_form
                    WHERE style_matches_last5 >= 3
                    """
                )
            ).scalar()

        print(f"{object_name}: {row_count}")
    except Exception as error:
        print(f"WARNING: Could not verify {object_name}: {error}")


# Prints missing style cluster counts from match_features and keeps running if verification fails.
def verify_match_style_missing_rows(engine):
    checks = {
        "home_style_cluster missing rows": """
            SELECT COUNT(*)
            FROM match_features
            WHERE home_style_cluster = -1
        """,
        "away_style_cluster missing rows": """
            SELECT COUNT(*)
            FROM match_features
            WHERE away_style_cluster = -1
        """,
    }

    for object_name, query in checks.items():
        try:
            with engine.connect() as conn:
                row_count = conn.execute(text(query)).scalar()

            print(f"{object_name}: {row_count}")
        except Exception as error:
            print(f"WARNING: Could not verify {object_name}: {error}")


# Verifies the feature views that should exist after sql/feature_queries.sql runs.
def verify_feature_outputs(engine):
    verify_row_count(engine, "team_xg_stats")
    verify_row_count(engine, "team_tactical_match_stats")
    verify_row_count(engine, "team_style_form")
    verify_style_mature_rows(engine)
    verify_row_count(engine, "home_xg_form")
    verify_row_count(engine, "away_xg_form")
    verify_view_shape(engine, "match_features")
    verify_match_style_missing_rows(engine)
    verify_view_shape(engine, "player_fpl_features")


# Runs all SQL feature-view statements from sql/feature_queries.sql.
def run_feature_queries(engine):
    try:
        feature_queries_path = PROJECT_ROOT / "sql" / "feature_queries.sql"
        sql_script = feature_queries_path.read_text(encoding="utf-8")
        statements = sql_script.split(";")

        for statement in statements:
            statement = statement.strip()

            if not statement or len(statement) < 10:
                continue

            executable_lines = [
                line for line in statement.splitlines()
                if not line.strip().startswith("--")
            ]
            executable_statement = "\n".join(executable_lines).strip()

            if not executable_statement:
                continue

            with engine.connect() as conn:
                conn.execute(text(statement))
                conn.commit()

            preview = statement[:60].strip().encode("ascii", "ignore").decode("ascii")
            print(f"Created: {preview}")

        print("Feature queries executed successfully")
        verify_feature_outputs(engine)
    except Exception as error:
        print(f"Error running feature queries: {error}")


# Loads the match_features view into a pandas DataFrame for model training.
def load_match_features(engine):
    try:
        query = "SELECT * FROM match_features"
        df = pd.read_sql(query, engine)

        print(f"Match features shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(3))

        return df
    except Exception as error:
        print(f"Error loading match features: {error}")
        return None


# Loads the player_fpl_features view into a pandas DataFrame for FPL modeling.
def load_player_features(engine):
    try:
        query = "SELECT * FROM player_fpl_features"
        df = pd.read_sql(query, engine)

        print(f"Player features shape: {df.shape}")
        print(df.head(3))

        return df
    except Exception as error:
        print(f"Error loading player features: {error}")
        return None


if __name__ == "__main__":
    from src.data_pipeline import get_engine

    engine = get_engine()

    if engine is None:
        print("Error: Could not create PostgreSQL engine.")
        sys.exit()

    run_feature_queries(engine)
    load_match_features(engine)
    load_player_features(engine)
    print("Day 3 complete. Features ready for modeling.")
