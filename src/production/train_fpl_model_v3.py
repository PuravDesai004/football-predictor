"""Train and validate the Tier 3 FPL points baseline chronologically."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOCAL_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "production_artifacts"
FEATURE_TABLE = "fpl_player_features_v3"
TRAINING_RUNS_TABLE = "fpl_model_training_runs_v3"
FINAL_HOLDOUT_SEASON = "2025-26"
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALIDATION_SEASON = "2024-25"
DB_CONNECT_TIMEOUT_SECONDS = 5

MODEL_FEATURE_COLUMNS = [
    "prior_points_last1",
    "prior_points_last3",
    "prior_points_last5",
    "prior_points_last10",
    "prior_points_season",
    "prior_minutes_last3",
    "prior_minutes_last5",
    "prior_minutes_last10",
    "prior_appearances_last5",
    "prior_starts_last5",
    "prior_goals_last5",
    "prior_assists_last5",
    "prior_bonus_last5",
    "prior_clean_sheets_last5",
    "prior_saves_last5",
    "prior_xg_last5",
    "prior_xa_last5",
    "prior_points_per_90",
    "prior_minutes_total",
    "prior_gameweeks_played",
]

IDENTIFIER_COLUMNS = [
    "feature_id",
    "canonical_player_key",
    "fpl_code",
    "season",
    "gameweek",
    "player_name",
    "player_source_id",
    "target_total_points",
    "feature_history_start_season",
    "feature_history_start_gameweek",
    "feature_history_end_season",
    "feature_history_end_gameweek",
    "prior_history_row_count",
    "source_history_row_id",
    "source_history_row_count",
    "created_at",
]

CANDIDATE_ARTIFACTS = {
    "model": LOCAL_ARTIFACTS_DIR / "fpl_points_v3_candidate.pkl",
    "features": LOCAL_ARTIFACTS_DIR / "fpl_points_v3_candidate_features.json",
    "metadata": LOCAL_ARTIFACTS_DIR / "fpl_points_v3_candidate_metadata.json",
}


def get_database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        url = make_url(database_url)
        if url.host and url.host.lower() == "localhost":
            url = url.set(host="127.0.0.1")
        return url.render_as_string(hide_password=False)

    values = {
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASS": os.getenv("DB_PASS"),
    }
    if values["DB_HOST"] and values["DB_HOST"].lower() == "localhost":
        values["DB_HOST"] = "127.0.0.1"
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Missing database settings: {missing}")
    return (
        f"postgresql+psycopg2://{values['DB_USER']}:{values['DB_PASS']}"
        f"@{values['DB_HOST']}:{values['DB_PORT']}/{values['DB_NAME']}"
    )


def get_engine():
    database_url = get_database_url()
    url = make_url(database_url)
    connect_args: dict[str, Any] = {"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS}
    if url.host and url.host not in {"127.0.0.1", "localhost"} and "sslmode" not in database_url.lower():
        connect_args["sslmode"] = "require"
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    with engine.connect():
        pass
    return engine


def _table_count(engine, table_name: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def load_feature_frame(engine) -> pd.DataFrame:
    columns = [
        "feature_id",
        "canonical_player_key",
        "fpl_code",
        "season",
        "gameweek",
        "target_total_points",
        *MODEL_FEATURE_COLUMNS,
    ]
    query = text(
        f"SELECT {', '.join(columns)} FROM {FEATURE_TABLE} "
        "WHERE season = ANY(:seasons) ORDER BY season, gameweek, canonical_player_key"
    )
    frame = pd.read_sql(query, engine, params={"seasons": TRAIN_SEASONS + [VALIDATION_SEASON]})
    if frame.empty:
        raise RuntimeError("No development or validation rows were found in fpl_player_features_v3")
    required = set(columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Feature table is missing required columns: {missing}")
    if FINAL_HOLDOUT_SEASON in set(frame["season"].astype(str)):
        raise RuntimeError("2025-26 entered the model dataframe")
    frame["target_total_points"] = pd.to_numeric(frame["target_total_points"], errors="coerce")
    if frame["target_total_points"].isna().any():
        raise RuntimeError("Model rows contain null target_total_points values")
    for column in MODEL_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["canonical_player_key", "season", "gameweek"]).any():
        raise RuntimeError("Duplicate player-season-gameweek feature rows detected")
    return frame


def split_development_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["season"].isin(TRAIN_SEASONS)].copy()
    validation = frame[frame["season"] == VALIDATION_SEASON].copy()
    if train.empty or validation.empty:
        raise RuntimeError("Training or validation split is empty")
    if set(train["season"]).intersection(validation["season"]):
        raise RuntimeError("Training and validation seasons overlap")
    return train, validation


def build_mean_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    return np.full(len(validation), float(train["target_total_points"].mean()))


def build_linear_baseline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", Ridge(alpha=10.0)),
    ])


def build_xgb_candidate() -> Pipeline:
    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _group_ndcg(frame: pd.DataFrame, prediction_column: str, k: int) -> float:
    values = []
    for _, group in frame.groupby(["season", "gameweek"], sort=False):
        if group.empty:
            continue
        actual = group["target_total_points"].to_numpy(dtype=float)
        predicted = group[prediction_column].to_numpy(dtype=float)
        # FPL points can be negative; NDCG requires non-negative relevance.
        # Shift each gameweek's gains by its minimum without changing ranks.
        gains = actual - np.min(actual)
        if np.allclose(gains, 0.0):
            values.append(1.0)
        else:
            values.append(float(ndcg_score([gains], [predicted], k=min(k, len(group)))))
    return float(np.mean(values)) if values else float("nan")


def _group_top_points_capture(frame: pd.DataFrame, prediction_column: str, k: int) -> float:
    captures = []
    for _, group in frame.groupby(["season", "gameweek"], sort=False):
        actual_top = group.nlargest(min(k, len(group)), "target_total_points")["target_total_points"].sum()
        predicted_top = group.nlargest(min(k, len(group)), prediction_column)["target_total_points"].sum()
        if actual_top > 0:
            captures.append(float(predicted_top / actual_top))
    return float(np.mean(captures)) if captures else float("nan")


def compute_fpl_metrics(actual: pd.Series, predictions: np.ndarray, validation: pd.DataFrame) -> dict[str, float]:
    scored = validation[["season", "gameweek", "target_total_points"]].copy()
    scored["prediction"] = predictions
    actual_values = actual.to_numpy(dtype=float)
    if np.ptp(actual_values) == 0.0 or np.ptp(predictions) == 0.0:
        spearman = 0.0
    else:
        spearman = spearmanr(actual_values, predictions).statistic
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
        "ndcg_at_10": _group_ndcg(scored, "prediction", 10),
        "ndcg_at_25": _group_ndcg(scored, "prediction", 25),
        "ndcg_at_50": _group_ndcg(scored, "prediction", 50),
        "top10_points_captured": _group_top_points_capture(scored, "prediction", 10),
        "top25_points_captured": _group_top_points_capture(scored, "prediction", 25),
        "top50_points_captured": _group_top_points_capture(scored, "prediction", 50),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "validation_rows": int(len(validation)),
        "validation_players": int(validation["canonical_player_key"].nunique()),
    }


def train_candidates(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    x_train = train[MODEL_FEATURE_COLUMNS]
    y_train = train["target_total_points"]
    x_validation = validation[MODEL_FEATURE_COLUMNS]
    y_validation = validation["target_total_points"]
    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    mean_predictions = np.full(len(validation), float(y_train.mean()))
    metrics["mean_baseline"] = compute_fpl_metrics(y_validation, mean_predictions, validation)

    for name, estimator in {
        "ridge_baseline": build_linear_baseline(),
        "xgb_candidate": build_xgb_candidate(),
    }.items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_validation)
        metrics[name] = compute_fpl_metrics(y_validation, predictions, validation)
        fitted[name] = estimator
    return metrics, fitted


def choose_candidate(metrics: dict[str, dict[str, float]]) -> str | None:
    baseline = metrics["mean_baseline"]
    candidates = [name for name in ("ridge_baseline", "xgb_candidate") if name in metrics]
    better = [name for name in candidates if metrics[name]["mae"] < baseline["mae"]]
    if not better:
        return None
    return min(better, key=lambda name: (metrics[name]["mae"], -metrics[name]["ndcg_at_25"]))


def save_candidate_artifact(
    model: Any,
    selected_name: str,
    metrics: dict[str, dict[str, float]],
    requested_dir: Path | None = None,
) -> None:
    artifact_dir = requested_dir or LOCAL_ARTIFACTS_DIR
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        artifact_dir = Path.home() / "football_predictor_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        print(f"Project artifact directory is not writable; using {artifact_dir}")
    artifact_paths = {
        "model": artifact_dir / "fpl_points_v3_candidate.pkl",
        "features": artifact_dir / "fpl_points_v3_candidate_features.json",
        "metadata": artifact_dir / "fpl_points_v3_candidate_metadata.json",
    }
    existing = [path for path in artifact_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite candidate artifacts: {existing}")
    joblib.dump(model, artifact_paths["model"])
    artifact_paths["features"].write_text(json.dumps(MODEL_FEATURE_COLUMNS, indent=2), encoding="utf-8")
    metadata = {
        "model_name": selected_name,
        "feature_table": FEATURE_TABLE,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "train_seasons": TRAIN_SEASONS,
        "validation_season": VALIDATION_SEASON,
        "final_holdout_season": FINAL_HOLDOUT_SEASON,
        "validation_metrics": metrics[selected_name],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_production",
    }
    metadata["artifact_directory"] = str(artifact_dir)
    artifact_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Candidate artifacts written to: {artifact_dir}")


def write_training_run_metadata(engine, metrics: dict[str, dict[str, float]], selected_name: str | None, row_count: int) -> None:
    payload = {
        "metrics": metrics,
        "selected_candidate": selected_name,
        "holdout_used": False,
        "holdout_season": FINAL_HOLDOUT_SEASON,
    }
    with engine.begin() as connection:
        connection.execute(text(f"""
            INSERT INTO {TRAINING_RUNS_TABLE}
            (run_finished_at, run_status, model_name, train_seasons,
             validation_season, final_holdout_season, feature_table,
             row_count, metrics_json, notes)
            VALUES
            (CURRENT_TIMESTAMP, 'success', :model_name, :train_seasons,
             :validation_season, :holdout_season, :feature_table,
             :row_count, CAST(:metrics_json AS JSONB), :notes)
        """), {
            "model_name": selected_name or "fpl_v3_baseline_comparison",
            "train_seasons": TRAIN_SEASONS,
            "validation_season": VALIDATION_SEASON,
            "holdout_season": FINAL_HOLDOUT_SEASON,
            "feature_table": FEATURE_TABLE,
            "row_count": row_count,
            "metrics_json": json.dumps(payload),
            "notes": "Phase 2B chronological development/validation only; 2025-26 not loaded.",
        })


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Tier 3 FPL chronological baselines")
    parser.add_argument(
        "--save-artifact",
        action="store_true",
        help="Save the selected candidate artifact after validation; disabled by default.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Writable directory for candidate artifacts; defaults to the project-local v3 directory.",
    )
    args = parser.parse_args()
    print("Tier 3 FPL Phase 2B: chronological baseline training")
    print(f"Training seasons: {', '.join(TRAIN_SEASONS)}")
    print(f"Validation season: {VALIDATION_SEASON}")
    print(f"Final holdout excluded: {FINAL_HOLDOUT_SEASON}")
    engine = get_engine()
    training_runs_before = _table_count(engine, TRAINING_RUNS_TABLE)
    frame = load_feature_frame(engine)
    train, validation = split_development_validation(frame)
    metrics, fitted = train_candidates(train, validation)
    selected_name = choose_candidate(metrics)
    if selected_name is not None and args.save_artifact:
        save_candidate_artifact(fitted[selected_name], selected_name, metrics, args.artifact_dir)
    write_training_run_metadata(engine, metrics, selected_name, len(train))
    training_runs_after = _table_count(engine, TRAINING_RUNS_TABLE)
    if training_runs_after != training_runs_before + 1:
        raise RuntimeError("Training-run metadata row count did not increase by one")

    for name, result in metrics.items():
        print(
            f"{name}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, "
            f"Spearman={result['spearman']:.4f}, NDCG@25={result['ndcg_at_25']:.4f}"
        )
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Selected candidate: {selected_name or 'NONE_BEATS_MEAN_BASELINE'}")
    print(f"Candidate artifact saved: {bool(selected_name and args.save_artifact)}")
    print(f"Training-run metadata rows: {training_runs_before} -> {training_runs_after}")
    print("2025-26 was not loaded, tuned, evaluated, or used for artifact selection.")


if __name__ == "__main__":
    main()
