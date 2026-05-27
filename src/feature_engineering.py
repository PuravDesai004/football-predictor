import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


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

            with engine.connect() as conn:
                conn.execute(text(statement))
                conn.commit()

            preview = statement[:60].strip().encode("ascii", "ignore").decode("ascii")
            print(f"Created: {preview}")

        print("All feature views created successfully.")
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
