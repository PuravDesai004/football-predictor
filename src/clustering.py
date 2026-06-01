import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STYLE_FEATURES = [
    "ppda_last5",
    "ppda_allowed_last5",
    "deep_last5",
    "deep_allowed_last5",
    "xg_last5",
    "xga_last5",
    "npxg_last5",
    "npxga_last5",
    "npxgd_last5",
    "goals_for_last5",
    "goals_against_last5",
    "xpts_last5",
]


# Loads leakage-safe rolling tactical style features from PostgreSQL.
def load_style_features(engine):
    try:
        query = """
            SELECT *
            FROM team_style_form
            ORDER BY gameweek, fixture_id, team_name
        """
        df = pd.read_sql(text(query), engine)

        df["style_matches_last5"] = pd.to_numeric(
            df["style_matches_last5"],
            errors="coerce",
        ).fillna(0).astype(int)

        for feature in STYLE_FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

        print(f"Loaded team_style_form shape: {df.shape}")
        mature_rows = int((df["style_matches_last5"] >= 3).sum())
        print(f"Mature style rows available: {mature_rows}")
        return df
    except Exception as error:
        print(f"Error loading team_style_form: {error}")
        return None


# Creates a complete-gameweek split so no gameweek appears in both train and test.
def get_time_safe_split(df):
    unique_gameweeks = sorted(df["gameweek"].dropna().unique())
    split_index = int(len(unique_gameweeks) * 0.8)

    if split_index <= 0 or split_index >= len(unique_gameweeks):
        raise ValueError("Not enough gameweeks to create a time-safe style split.")

    train_gameweeks = unique_gameweeks[:split_index]
    test_gameweeks = unique_gameweeks[split_index:]
    overlap = sorted(set(train_gameweeks).intersection(set(test_gameweeks)))

    print(
        f"Style train gameweeks: GW{int(min(train_gameweeks))} "
        f"to GW{int(max(train_gameweeks))}"
    )
    print(
        f"Style test gameweeks: GW{int(min(test_gameweeks))} "
        f"to GW{int(max(test_gameweeks))}"
    )

    if overlap:
        print(f"Style gameweek overlap: {overlap}")
        raise ValueError(f"Style gameweek overlap found: {overlap}")

    print("Style gameweek overlap: none")

    train_df = df[df["gameweek"].isin(train_gameweeks)].copy()
    test_df = df[df["gameweek"].isin(test_gameweeks)].copy()
    return train_df, test_df


# Fits KMeans only on the training gameweeks using imputed and scaled style features.
def train_style_clusters(df):
    try:
        train_df, _ = get_time_safe_split(df)
        train_df = train_df[train_df["style_matches_last5"] >= 3].copy()

        if train_df.empty:
            raise ValueError("No mature training rows found for KMeans.")

        x_train = train_df[STYLE_FEATURES].copy()

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)

        x_train_imputed = imputer.fit_transform(x_train)
        x_train_scaled = scaler.fit_transform(x_train_imputed)
        kmeans.fit(x_train_scaled)

        print("KMeans style clusters trained with 4 clusters")
        print(f"Training rows used for KMeans after maturity filter: {len(train_df)}")

        try:
            if len(set(kmeans.labels_)) > 1 and len(train_df) > 4:
                score = silhouette_score(x_train_scaled, kmeans.labels_)
                print(f"Silhouette score: {score:.3f}")
            else:
                print("WARNING: Could not calculate silhouette score: only one cluster found")
        except Exception as error:
            print(f"WARNING: Could not calculate silhouette score: {error}")

        return imputer, scaler, kmeans
    except Exception as error:
        print(f"Error training style clusters: {error}")
        return None, None, None


# Builds one unique readable label for each numeric KMeans cluster.
def build_style_labels(clustered_df):
    profiles = clustered_df.groupby("style_cluster")[STYLE_FEATURES].mean()
    label_by_cluster = {}
    remaining_clusters = set(profiles.index.tolist())

    high_press_cluster = profiles["ppda_last5"].idxmin()
    label_by_cluster[high_press_cluster] = "High Press"
    remaining_clusters.discard(high_press_cluster)

    direct_attack_cluster = profiles.loc[list(remaining_clusters), "deep_last5"].idxmax()
    label_by_cluster[direct_attack_cluster] = "Direct Attack"
    remaining_clusters.discard(direct_attack_cluster)

    low_control_cluster = profiles.loc[list(remaining_clusters), "xga_last5"].idxmax()
    label_by_cluster[low_control_cluster] = "Low Control"
    remaining_clusters.discard(low_control_cluster)

    for cluster_id in remaining_clusters:
        label_by_cluster[cluster_id] = "Compact Defense"

    return label_by_cluster


# Assigns the learned style clusters to every row, including held-out test gameweeks.
def assign_style_clusters(df, imputer, scaler, kmeans):
    try:
        clustered_df = df.copy()
        x_all = clustered_df[STYLE_FEATURES].copy()
        x_all_imputed = imputer.transform(x_all)
        x_all_scaled = scaler.transform(x_all_imputed)

        clustered_df["style_cluster"] = kmeans.predict(x_all_scaled)
        label_by_cluster = build_style_labels(clustered_df)
        clustered_df["style_label"] = clustered_df["style_cluster"].map(label_by_cluster)

        return clustered_df
    except Exception as error:
        print(f"Error assigning style clusters: {error}")
        return None


# Prints cluster sizes and mean tactical profiles for human inspection.
def summarize_clusters(clustered_df):
    try:
        print("Style cluster counts:")
        print(clustered_df["style_label"].value_counts().sort_index())

        profile_columns = [
            "ppda_last5",
            "deep_last5",
            "xg_last5",
            "xga_last5",
            "xpts_last5",
        ]
        profiles = clustered_df.groupby("style_label")[profile_columns].mean().round(3)

        print("Style cluster profiles:")
        print(profiles)
    except Exception as error:
        print(f"Error summarizing style clusters: {error}")


# Saves the imputer, scaler, KMeans model, and exact style feature order.
def save_cluster_artifacts(imputer, scaler, kmeans, feature_names):
    try:
        joblib.dump(imputer, MODELS_DIR / "style_imputer.pkl")
        joblib.dump(scaler, MODELS_DIR / "style_scaler.pkl")
        joblib.dump(kmeans, MODELS_DIR / "style_kmeans.pkl")

        with (MODELS_DIR / "style_features.json").open("w", encoding="utf-8") as file:
            json.dump(feature_names, file, indent=2)

        print("Saved style_imputer.pkl")
        print("Saved style_scaler.pkl")
        print("Saved style_kmeans.pkl")
        print("Saved style_features.json")
    except Exception as error:
        print(f"Error saving cluster artifacts: {error}")


# Loads all team-match style cluster assignments into PostgreSQL.
def load_clusters_to_postgres(clustered_df, engine):
    try:
        columns_to_store = [
            "fixture_id",
            "gameweek",
            "match_date",
            "team_name",
            "opponent_name",
            "venue",
            "style_matches_last5",
            "style_cluster",
            "style_label",
        ] + STYLE_FEATURES

        create_table_sql = """
            DROP TABLE IF EXISTS team_style_clusters;

            CREATE TABLE team_style_clusters (
                fixture_id INT,
                gameweek INT,
                match_date DATE,
                team_name VARCHAR(100),
                opponent_name VARCHAR(100),
                venue VARCHAR(5),
                style_matches_last5 INT,
                style_cluster INT,
                style_label VARCHAR(50),
                ppda_last5 FLOAT,
                ppda_allowed_last5 FLOAT,
                deep_last5 FLOAT,
                deep_allowed_last5 FLOAT,
                xg_last5 FLOAT,
                xga_last5 FLOAT,
                npxg_last5 FLOAT,
                npxga_last5 FLOAT,
                npxgd_last5 FLOAT,
                goals_for_last5 FLOAT,
                goals_against_last5 FLOAT,
                xpts_last5 FLOAT
            );
        """

        with engine.begin() as conn:
            conn.execute(text(create_table_sql))

        clustered_df[columns_to_store].to_sql(
            "team_style_clusters",
            engine,
            if_exists="append",
            index=False,
        )

        print(f"Loaded {len(clustered_df)} rows into team_style_clusters")
    except Exception as error:
        print(f"Error loading team_style_clusters: {error}")


# Verifies the loaded style cluster table row, team, and cluster counts.
def verify_clusters(engine):
    try:
        with engine.connect() as conn:
            row_count = conn.execute(text("SELECT COUNT(*) FROM team_style_clusters")).scalar()
            team_count = conn.execute(
                text("SELECT COUNT(DISTINCT team_name) FROM team_style_clusters")
            ).scalar()
            cluster_count = conn.execute(
                text("SELECT COUNT(DISTINCT style_cluster) FROM team_style_clusters")
            ).scalar()

        print(f"team_style_clusters row count: {row_count}")
        print(f"team_style_clusters teams: {team_count}")
        print(f"team_style_clusters clusters: {cluster_count}")
    except Exception as error:
        print(f"Error verifying team_style_clusters: {error}")


# Runs the full style clustering pipeline without touching match prediction models.
def main():
    from src.data_pipeline import get_engine

    engine = get_engine()
    if engine is None:
        print("Error: Could not create PostgreSQL engine.")
        return

    df = load_style_features(engine)
    if df is None or df.empty:
        print("Error: No team_style_form rows found. Run feature engineering first.")
        return

    imputer, scaler, kmeans = train_style_clusters(df)
    if imputer is None or scaler is None or kmeans is None:
        print("Error: Could not train style clusters.")
        return

    clustered_df = assign_style_clusters(df, imputer, scaler, kmeans)
    if clustered_df is None or clustered_df.empty:
        print("Error: Could not assign style clusters.")
        return

    summarize_clusters(clustered_df)
    save_cluster_artifacts(imputer, scaler, kmeans, STYLE_FEATURES)
    load_clusters_to_postgres(clustered_df, engine)
    verify_clusters(engine)


if __name__ == "__main__":
    main()
