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

from src.train_fpl_model import FPL_FEATURES, FORBIDDEN_RAW_CURRENT_GW_FEATURES


POSITION_NAMES = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

POSITION_MODEL_FILES = {
    "GK": "fpl_points_gk_xgb.pkl",
    "DEF": "fpl_points_def_xgb.pkl",
    "MID": "fpl_points_mid_xgb.pkl",
    "FWD": "fpl_points_fwd_xgb.pkl",
}


# Loads leakage-safe FPL gameweek rows and joins current player position metadata.
def load_position_training_data(engine):
    query = """
    SELECT
        pgf.*,
        COALESCE(p.position, pff.position) AS current_position
    FROM player_gameweek_features pgf
    LEFT JOIN players p
        ON pgf.player_id = p.player_id
    LEFT JOIN player_fpl_features pff
        ON pgf.player_id = pff.player_id
    """
    df = pd.read_sql(query, engine)
    print(f"Loaded FPL training rows: {len(df)}")
    return df


# Validates required columns, filters mature rows, and casts model features.
def prepare_position_features(df):
    if "current_position" in df.columns:
        if "position" in df.columns:
            df["position"] = df["current_position"].fillna(df["position"])
        else:
            df["position"] = df["current_position"]

    required_columns = [
        "player_id",
        "gameweek",
        "fixture",
        "kickoff_time",
        "position",
        "target_total_points",
        *FPL_FEATURES,
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    forbidden_used = [
        feature for feature in FPL_FEATURES
        if feature in FORBIDDEN_RAW_CURRENT_GW_FEATURES
    ]
    if forbidden_used:
        raise ValueError(f"Forbidden raw current-GW features used: {forbidden_used}")

    df = df[df["history_matches_last5"] >= 3].copy()
    df = df.sort_values(
        ["gameweek", "kickoff_time", "fixture", "player_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["target_total_points"] = pd.to_numeric(
        df["target_total_points"],
        errors="coerce",
    )
    df["was_home"] = df["was_home"].fillna(False).astype(float)

    for feature in FPL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype(float)

    print(f"Rows after maturity filter: {len(df)}")
    print(f"Players after maturity filter: {df['player_id'].nunique()}")
    print(f"Feature count: {len(FPL_FEATURES)}")

    return df


# Splits rows by complete gameweeks so future gameweeks never leak into training.
def make_complete_gameweek_split(df, train_fraction=0.8):
    gameweeks = sorted(df["gameweek"].dropna().unique().tolist())
    split_index = int(len(gameweeks) * train_fraction)

    if split_index <= 0 or split_index >= len(gameweeks):
        raise ValueError("Not enough gameweeks for a complete-gameweek split.")

    train_gameweeks = set(gameweeks[:split_index])
    test_gameweeks = set(gameweeks[split_index:])
    overlap = train_gameweeks.intersection(test_gameweeks)

    if overlap:
        raise ValueError(f"Train/test gameweek overlap detected: {sorted(overlap)}")

    train_df = df[df["gameweek"].isin(train_gameweeks)].copy()
    test_df = df[df["gameweek"].isin(test_gameweeks)].copy()

    print(f"Train gameweeks: GW{min(train_gameweeks)} to GW{max(train_gameweeks)}")
    print(f"Test gameweeks: GW{min(test_gameweeks)} to GW{max(test_gameweeks)}")
    print("Gameweek overlap: none")

    return train_df, test_df


# Creates a fresh XGBoost regressor with the agreed FPL points settings.
def create_xgb_model():
    return XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )


# Calculates regression metrics for a set of predictions.
def calculate_metrics(y_true, predictions):
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": float(np.sqrt(np.mean((y_true - predictions) ** 2))),
        "r2": r2_score(y_true, predictions),
    }


# Trains and evaluates one position-specific FPL points model.
def train_position_model(position_name, train_df, test_df):
    X_train = train_df[FPL_FEATURES]
    y_train = train_df["target_total_points"]
    X_test = test_df[FPL_FEATURES]
    y_test = test_df["target_total_points"]

    model = create_xgb_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    model_metrics = calculate_metrics(y_test, predictions)

    baseline_predictions = test_df["points_avg_last5"].fillna(y_train.mean())
    baseline_metrics = calculate_metrics(y_test, baseline_predictions)

    print(f"\n=== Position: {position_name} ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(
        "Baseline points_avg_last5 | "
        f"MAE: {baseline_metrics['mae']:.3f} | "
        f"RMSE: {baseline_metrics['rmse']:.3f} | "
        f"R2: {baseline_metrics['r2']:.3f}"
    )
    print(
        "Position XGBoost         | "
        f"MAE: {model_metrics['mae']:.3f} | "
        f"RMSE: {model_metrics['rmse']:.3f} | "
        f"R2: {model_metrics['r2']:.3f}"
    )

    return model, predictions, model_metrics, baseline_metrics


# Loads and evaluates the current single FPL XGBoost model on the same test rows.
def evaluate_single_model(test_df):
    model_path = MODELS_DIR / "fpl_points_xgb.pkl"
    features_path = MODELS_DIR / "fpl_points_features.json"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError("Current single FPL model or feature file is missing.")

    model = joblib.load(model_path)
    with features_path.open("r", encoding="utf-8") as file:
        single_features = json.load(file)

    missing_features = [
        feature for feature in single_features
        if feature not in test_df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing single-model feature columns: {missing_features}")

    X_test = test_df[single_features].copy()
    for feature in single_features:
        X_test[feature] = pd.to_numeric(X_test[feature], errors="coerce").astype(float)

    predictions = model.predict(X_test)
    metrics = calculate_metrics(test_df["target_total_points"], predictions)

    return predictions, metrics, single_features


# Prints the top 10 feature importances for each position model.
def show_position_feature_importance(position_name, model):
    importances = pd.Series(
        model.feature_importances_,
        index=FPL_FEATURES,
    ).sort_values(ascending=False)

    print(f"\nTop 10 feature importances for {position_name}:")
    for feature, score in importances.head(10).items():
        print(f"  {feature:<35} {score:.4f}")


# Saves the position-specific models and manifest only after the decision says keep.
def save_position_models(models, metrics, single_metrics, combined_metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for position_name, model in models.items():
        path = MODELS_DIR / POSITION_MODEL_FILES[position_name]
        joblib.dump(model, path)
        saved_files.append(str(path))

    features_path = MODELS_DIR / "fpl_position_model_features.json"
    with features_path.open("w", encoding="utf-8") as file:
        json.dump(FPL_FEATURES, file, indent=2)
    saved_files.append(str(features_path))

    manifest = {
        "position_model_files": POSITION_MODEL_FILES,
        "features_file": "fpl_position_model_features.json",
        "positions": POSITION_NAMES,
        "single_model_metrics": single_metrics,
        "combined_position_metrics": combined_metrics,
        "per_position_metrics": metrics,
        "decision": "KEEP POSITION-SPECIFIC FPL MODELS",
    }

    manifest_path = MODELS_DIR / "fpl_position_model_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    saved_files.append(str(manifest_path))

    print("\nSaved position-specific files:")
    for path in saved_files:
        print(f"  {path}")

    return saved_files


def main():
    from src.data_pipeline import get_engine

    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    print("=== FPL position-specific training data ===")
    df = load_position_training_data(engine)
    df = prepare_position_features(df)

    print("\n=== FPL complete-gameweek split ===")
    train_df, test_df = make_complete_gameweek_split(df)

    position_models = {}
    per_position_metrics = {}
    combined_predictions = pd.Series(index=test_df.index, dtype=float)

    print("\n=== Per-position model results ===")
    for position_id, position_name in POSITION_NAMES.items():
        train_pos = train_df[train_df["position"] == position_id].copy()
        test_pos = test_df[test_df["position"] == position_id].copy()

        if train_pos.empty or test_pos.empty:
            print(f"\n=== Position: {position_name} ===")
            print("Skipped because train or test rows are empty.")
            continue

        model, predictions, model_metrics, baseline_metrics = train_position_model(
            position_name,
            train_pos,
            test_pos,
        )
        position_models[position_name] = model
        combined_predictions.loc[test_pos.index] = predictions
        per_position_metrics[position_name] = {
            "train_rows": len(train_pos),
            "test_rows": len(test_pos),
            "baseline": baseline_metrics,
            "xgboost": model_metrics,
        }

    if combined_predictions.isna().any():
        missing = int(combined_predictions.isna().sum())
        raise ValueError(f"Missing combined position predictions for {missing} rows.")

    print("\n=== Single vs position-specific comparison ===")
    _, single_metrics, single_features = evaluate_single_model(test_df)
    combined_metrics = calculate_metrics(
        test_df["target_total_points"],
        combined_predictions,
    )

    print(
        "Single model              | "
        f"MAE: {single_metrics['mae']:.3f} | "
        f"RMSE: {single_metrics['rmse']:.3f} | "
        f"R2: {single_metrics['r2']:.3f}"
    )
    print(
        "Position-specific combined | "
        f"MAE: {combined_metrics['mae']:.3f} | "
        f"RMSE: {combined_metrics['rmse']:.3f} | "
        f"R2: {combined_metrics['r2']:.3f}"
    )
    print(f"Single model feature count: {len(single_features)}")
    print(f"Position model feature count: {len(FPL_FEATURES)}")

    for position_name, model in position_models.items():
        show_position_feature_importance(position_name, model)

    print("\n=== FPL position-model decision ===")
    mae_improvement = single_metrics["mae"] - combined_metrics["mae"]
    print(f"MAE improvement vs single model: {mae_improvement:.3f}")

    if mae_improvement >= 0.01:
        print("KEEP POSITION-SPECIFIC FPL MODELS")
        save_position_models(
            position_models,
            per_position_metrics,
            single_metrics,
            combined_metrics,
        )
    else:
        print("DO NOT KEEP POSITION-SPECIFIC FPL MODELS")
        print("Position-specific MAE did not improve by at least 0.01.")


if __name__ == "__main__":
    main()
