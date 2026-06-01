import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEAKAGE_COLUMNS = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

HISTORICAL_LEAKAGE_CHECK_COLUMNS = [
    "points_prev1",
    "minutes_prev1",
    "xg_prev1",
    "xa_prev1",
    "points_avg_last5",
    "minutes_avg_last5",
    "xg_avg_last5",
    "xa_avg_last5",
    "ict_avg_last5",
    "history_matches_last5",
]


# Drops an old view/table so the refreshed feature table can be rebuilt cleanly.
def drop_existing_feature_relation(engine):
    relation_query = """
        SELECT c.relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'player_gameweek_features'
    """

    with engine.begin() as conn:
        relation_kind = conn.execute(text(relation_query)).scalar()

        if relation_kind == "v":
            conn.execute(text("DROP VIEW IF EXISTS player_gameweek_features CASCADE"))
        elif relation_kind == "m":
            conn.execute(
                text("DROP MATERIALIZED VIEW IF EXISTS player_gameweek_features CASCADE")
            )
        elif relation_kind == "r":
            conn.execute(text("DROP TABLE IF EXISTS player_gameweek_features"))


# Runs the SQL that creates the leakage-safe FPL player gameweek feature table.
def run_fpl_feature_queries(engine):
    try:
        drop_existing_feature_relation(engine)

        query_path = PROJECT_ROOT / "sql" / "fpl_feature_queries.sql"
        sql_script = query_path.read_text(encoding="utf-8")
        statements = sql_script.split(";")

        for statement in statements:
            statement = statement.strip()

            if not statement or len(statement) < 10:
                continue

            with engine.connect() as conn:
                conn.execute(text(statement))
                conn.commit()

        print("FPL feature queries executed successfully")
    except Exception as error:
        print(f"Error running FPL feature queries: {error}")
        raise


# Loads the player_gameweek_features table and prints coverage checks.
def load_player_gameweek_features(engine):
    try:
        df = pd.read_sql("SELECT * FROM player_gameweek_features", engine)

        player_count = df["player_id"].nunique()
        min_gw = df["gameweek"].min()
        max_gw = df["gameweek"].max()
        mature_count = len(df[df["history_matches_last5"] >= 3])
        null_targets = df["target_total_points"].isna().sum()

        print(f"player_gameweek_features shape: {df.shape}")
        print(f"player_gameweek_features players: {player_count}")
        print(f"player_gameweek_features gameweeks: GW{min_gw} to GW{max_gw}")
        print(f"mature rows history_matches_last5 >= 3: {mature_count}")
        print(f"null target_total_points rows: {null_targets}")

        sample_cols = [
            "player_id",
            "gameweek",
            "opponent_team",
            "was_home",
            "target_total_points",
            "points_avg_last5",
            "minutes_avg_last5",
            "xg_avg_last5",
            "xa_avg_last5",
            "ict_avg_last5",
            "history_matches_last5",
        ]
        print(df[sample_cols].head(5).to_string(index=False))

        return df
    except Exception as error:
        print(f"Error loading player_gameweek_features: {error}")
        return None


# Fails if raw same-gameweek outcome columns appear in the feature table.
def verify_no_same_gw_leakage_columns(df):
    if df is None:
        raise ValueError("Cannot verify leakage columns because DataFrame is None.")

    leaked_columns = sorted(set(df.columns).intersection(LEAKAGE_COLUMNS))

    if leaked_columns:
        raise ValueError(
            "Same-GW leakage columns found in player_gameweek_features: "
            f"{leaked_columns}"
        )

    print("FPL leakage column check passed")


# Fails if double-gameweek rows have different historical features within a gameweek.
def verify_no_same_gameweek_lag_leakage(df):
    if df is None:
        raise ValueError("Cannot verify same-gameweek lag leakage because DataFrame is None.")

    missing_columns = [
        column for column in HISTORICAL_LEAKAGE_CHECK_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing historical leakage check columns: {missing_columns}")

    group_sizes = df.groupby(["player_id", "gameweek"]).size()
    duplicate_index = group_sizes[group_sizes > 1].index
    duplicate_count = len(duplicate_index)
    mismatch_count = 0

    for player_id, gameweek in duplicate_index:
        group = df[
            (df["player_id"] == player_id)
            & (df["gameweek"] == gameweek)
        ]

        for column in HISTORICAL_LEAKAGE_CHECK_COLUMNS:
            if group[column].nunique(dropna=False) > 1:
                mismatch_count += 1
                break

    print(f"duplicate player-gameweek groups: {duplicate_count}")
    print(f"same-gameweek historical feature mismatch groups: {mismatch_count}")

    if mismatch_count > 0:
        raise ValueError(
            "Same-gameweek historical feature leakage detected in "
            f"{mismatch_count} duplicate player-gameweek groups."
        )


def main():
    from src.data_pipeline import get_engine

    engine = get_engine()

    if engine is None:
        print("Error: Could not create PostgreSQL engine.")
        sys.exit(1)

    run_fpl_feature_queries(engine)
    df = load_player_gameweek_features(engine)
    verify_no_same_gw_leakage_columns(df)
    verify_no_same_gameweek_lag_leakage(df)


if __name__ == "__main__":
    main()
