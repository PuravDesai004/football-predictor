import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

FPL_FEATURES = [
    "gameweek",
    "opponent_team",
    "was_home",
    "value",
    "selected",
    "history_matches_last5",
    "points_prev1",
    "minutes_prev1",
    "starts_prev1",
    "xg_prev1",
    "xa_prev1",
    "xgi_prev1",
    "xgc_prev1",
    "ict_prev1",
    "value_prev1",
    "selected_prev1",
    "points_avg_last3",
    "points_avg_last5",
    "minutes_avg_last3",
    "minutes_avg_last5",
    "starts_avg_last5",
    "xg_avg_last5",
    "xa_avg_last5",
    "xgi_avg_last5",
    "xgc_avg_last5",
    "influence_avg_last5",
    "creativity_avg_last5",
    "threat_avg_last5",
    "ict_avg_last5",
    "bps_avg_last5",
    "bonus_avg_last5",
    "goals_avg_last5",
    "assists_avg_last5",
    "clean_sheets_avg_last5",
    "goals_conceded_avg_last5",
    "saves_avg_last5",
    "yellow_cards_avg_last5",
    "red_cards_avg_last5",
    "transfers_balance_avg_last5",
    "transfers_in_avg_last5",
    "transfers_out_avg_last5",
]

FORBIDDEN_RAW_CURRENT_GW_FEATURES = [
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
    "total_points",
]

REQUIRED_COLUMNS = [
    "player_id",
    "gameweek",
    "fixture",
    "kickoff_time",
    "target_total_points",
    *FPL_FEATURES,
]


# Loads the refreshed leakage-safe FPL feature table from PostgreSQL.
def load_fpl_training_data(engine):
    print("=== FPL training data ===")
    query = "SELECT * FROM player_gameweek_features"
    return pd.read_sql(query, engine)


# Validates, filters, sorts, and types FPL features for model training.
def prepare_fpl_features(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required FPL feature columns: {missing_columns}")

    forbidden_used = [
        column for column in FPL_FEATURES
        if column in FORBIDDEN_RAW_CURRENT_GW_FEATURES
    ]
    if forbidden_used:
        raise ValueError(f"Forbidden raw current-GW features used: {forbidden_used}")

    df = df[df["history_matches_last5"] >= 3].copy()
    df = df.sort_values(
        ["gameweek", "kickoff_time", "fixture", "player_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df["was_home"] = df["was_home"].astype(float)
    df["target_total_points"] = pd.to_numeric(
        df["target_total_points"],
        errors="coerce",
    )

    for feature in FPL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float)

    print(f"Rows after maturity filter: {len(df)}")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Gameweeks: GW{df['gameweek'].min()} to GW{df['gameweek'].max()}")
    print(f"Feature count: {len(FPL_FEATURES)}")

    return df


# Splits by complete gameweeks so future fixtures never leak into training.
def make_fpl_time_safe_split(df):
    print("\n=== FPL time-safe split ===")

    gameweeks = sorted(df["gameweek"].dropna().unique().tolist())
    split_index = int(len(gameweeks) * 0.8)

    if split_index <= 0 or split_index >= len(gameweeks):
        raise ValueError("Not enough gameweeks for a time-safe train/test split.")

    train_gameweeks = set(gameweeks[:split_index])
    test_gameweeks = set(gameweeks[split_index:])
    overlap = train_gameweeks.intersection(test_gameweeks)

    if overlap:
        raise ValueError(f"Train/test gameweek overlap detected: {sorted(overlap)}")

    train_df = df[df["gameweek"].isin(train_gameweeks)].copy()
    test_df = df[df["gameweek"].isin(test_gameweeks)].copy()

    print(f"Training rows: {len(train_df)}")
    print(f"Train gameweeks: GW{train_df['gameweek'].min()} to GW{train_df['gameweek'].max()}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Test gameweeks: GW{test_df['gameweek'].min()} to GW{test_df['gameweek'].max()}")
    print("Gameweek overlap: none")

    return train_df, test_df


# Uses each player's previous five-gameweek points average as the benchmark.
def train_baseline_model(train_df, test_df):
    print("\n=== Baseline last-5 average ===")

    baseline_predictions = test_df["points_avg_last5"].fillna(
        train_df["target_total_points"].mean()
    )
    y_test = test_df["target_total_points"]

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(np.mean((y_test - baseline_predictions) ** 2))
    baseline_r2 = r2_score(y_test, baseline_predictions)

    print(f"Baseline MAE: {baseline_mae:.3f}")
    print(f"Baseline RMSE: {baseline_rmse:.3f}")
    print(f"Baseline R2: {baseline_r2:.3f}")

    return baseline_predictions, baseline_mae


# Trains the XGBoost regressor for FPL points prediction.
def train_xgb_fpl_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model


# Evaluates XGBoost against the baseline predictions.
def evaluate_fpl_model(model, X_test, y_test, baseline_predictions):
    print("\n=== XGBoost FPL points model ===")

    predictions = model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, predictions)
    xgb_rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    xgb_r2 = r2_score(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    print(f"XGBoost MAE: {xgb_mae:.3f}")
    print(f"XGBoost RMSE: {xgb_rmse:.3f}")
    print(f"XGBoost R2: {xgb_r2:.3f}")

    return {
        "xgb_mae": xgb_mae,
        "xgb_rmse": xgb_rmse,
        "xgb_r2": xgb_r2,
        "baseline_mae": baseline_mae,
    }


# Prints the top 15 XGBoost feature importances.
def show_fpl_feature_importance(model, feature_names):
    print("\nTop 15 FPL feature importances:")

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    for feature, score in importances.head(15).items():
        print(f"  {feature:<35} {score:.4f}")


# Saves the FPL model and exact feature order for later prediction use.
def save_fpl_model(model, feature_names):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "fpl_points_xgb.pkl"
    features_path = MODELS_DIR / "fpl_points_features.json"

    joblib.dump(model, model_path)
    with features_path.open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved features: {features_path}")


def main():
    from src.data_pipeline import get_engine

    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    df = load_fpl_training_data(engine)
    df = prepare_fpl_features(df)
    train_df, test_df = make_fpl_time_safe_split(df)

    X_train = train_df[FPL_FEATURES]
    y_train = train_df["target_total_points"]
    X_test = test_df[FPL_FEATURES]
    y_test = test_df["target_total_points"]

    baseline_predictions, baseline_mae = train_baseline_model(train_df, test_df)
    model = train_xgb_fpl_model(X_train, y_train)
    metrics = evaluate_fpl_model(model, X_test, y_test, baseline_predictions)

    print("\n=== FPL model decision ===")
    if metrics["xgb_mae"] < baseline_mae:
        print("KEEP FPL XGBOOST MODEL")
        show_fpl_feature_importance(model, FPL_FEATURES)
        save_fpl_model(model, FPL_FEATURES)
    else:
        print("DO NOT KEEP FPL XGBOOST MODEL")
        print("XGBoost did not beat the last-5 average baseline MAE.")


if __name__ == "__main__":
    main()
