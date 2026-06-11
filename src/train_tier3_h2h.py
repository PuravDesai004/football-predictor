from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_pipeline import get_engine
from tier3_validation import validate_historical_match_integrity


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_TEST_SEASON = "2025-26"
FOLDS = [
    {
        "fold": 1,
        "train_seasons": ["2021-22", "2022-23"],
        "validation_seasons": ["2023-24"],
    },
    {
        "fold": 2,
        "train_seasons": ["2021-22", "2022-23", "2023-24"],
        "validation_seasons": ["2024-25"],
    },
]
RANDOM_STATE = 42
MIN_H2H_PRIOR_MATCHES = 3
TARGET_COLUMN = "result"

CLASS_LABELS = ["H", "D", "A"]
EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_FOLD_ROWS = {
    1: (760, 380),
    2: (1140, 380),
}

WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "match_features_v3_h2h_experiment",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]

FORBIDDEN_POST_MATCH_ELO_COLUMNS = [
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
]

EXCLUDED_FEATURE_COLUMNS = {
    "created_at",
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
    "h2h_last_meeting_result",
    "h2h_home_points_avg_prior",
    "h2h_away_points_avg_prior",
    *FORBIDDEN_POST_MATCH_ELO_COLUMNS,
}


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_h2h_feature_data(conn) -> pd.DataFrame:
    if FINAL_TEST_SEASON in DEV_SEASONS:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be a development season")

    query = text(
        """
        SELECT *
        FROM match_features_v3_h2h_experiment
        WHERE season_id = ANY(:dev_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pd.read_sql(query, conn, params={"dev_seasons": DEV_SEASONS})

    errors: list[str] = []
    if len(df) != EXPECTED_DEV_ROW_COUNT:
        errors.append(f"expected {EXPECTED_DEV_ROW_COUNT} rows, found {len(df)}")
    if df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}"
        )
    if df[TARGET_COLUMN].isna().any():
        errors.append(f"null target count: {int(df[TARGET_COLUMN].isna().sum())}")

    unknown_results = sorted(set(df[TARGET_COLUMN].dropna()) - set(CLASS_LABELS))
    if unknown_results:
        errors.append(f"unknown result labels: {unknown_results}")

    seasons_present = sorted(df["season_id"].unique().tolist())
    if seasons_present != DEV_SEASONS:
        errors.append(f"development seasons {seasons_present} != {DEV_SEASONS}")

    if any(column in df.columns for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS):
        forbidden = [
            column for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS if column in df.columns
        ]
        errors.append(f"forbidden post-match Elo columns present: {forbidden}")

    if errors:
        raise ValueError("H2H development dataset validation failed: " + "; ".join(errors))

    df["match_date"] = pd.to_datetime(df["match_date"])
    validate_no_final_holdout_loaded(df)

    print("=== H2H Development Dataset ===")
    print(f"Development dataframe row count: {len(df)}")
    for season_id, count in df.groupby("season_id").size().sort_index().items():
        print(f"- {season_id}: {int(count)} rows")
    print(f"{FINAL_TEST_SEASON} was not loaded into the modeling dataframe.")
    return df


def validate_no_final_holdout_loaded(df) -> None:
    if FINAL_TEST_SEASON in set(df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} was loaded into development data")
    print(f"PASS: {FINAL_TEST_SEASON} not loaded.")


def get_base_elo_feature_columns(df) -> list[str]:
    feature_columns = [
        "home_home_matches_last5",
        "away_away_matches_last5",
        "home_overall_matches_last5",
        "away_overall_matches_last5",
        "home_overall_matches_last10",
        "away_overall_matches_last10",
        "home_goals_scored_home_last5",
        "home_goals_conceded_home_last5",
        "home_clean_sheet_rate_home_last5",
        "away_goals_scored_away_last5",
        "away_goals_conceded_away_last5",
        "away_clean_sheet_rate_away_last5",
        "home_xg_home_last5",
        "home_xga_home_last5",
        "away_xg_away_last5",
        "away_xga_away_last5",
        "home_points_overall_last5",
        "away_points_overall_last5",
        "home_points_overall_last10",
        "away_points_overall_last10",
        "home_goal_diff_overall_last5",
        "away_goal_diff_overall_last5",
        "home_xg_overall_last5",
        "home_xga_overall_last5",
        "away_xg_overall_last5",
        "away_xga_overall_last5",
        "home_elo_before",
        "away_elo_before",
        "elo_diff_before",
        "elo_diff_home_adjusted",
        "expected_home_score",
        "expected_away_score",
    ]
    missing_columns = sorted(set(feature_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing base + Elo feature columns: {missing_columns}")
    forbidden_selected = sorted(set(feature_columns) & EXCLUDED_FEATURE_COLUMNS)
    if forbidden_selected:
        raise ValueError(f"Forbidden columns selected as base + Elo features: {forbidden_selected}")
    odds_columns = [column for column in feature_columns if "odds" in column.lower()]
    if odds_columns:
        raise ValueError(f"Odds columns selected unexpectedly: {odds_columns}")
    return feature_columns


def get_h2h_feature_columns() -> list[str]:
    return [
        "h2h_matches_prior",
        "h2h_home_win_rate_prior",
        "h2h_draw_rate_prior",
        "h2h_away_win_rate_prior",
        "h2h_home_goals_avg_prior",
        "h2h_away_goals_avg_prior",
        "h2h_goal_diff_avg_prior",
        "h2h_last_meeting_days",
        "h2h_last_meeting_home_goals",
        "h2h_last_meeting_away_goals",
    ]


def apply_h2h_threshold(df, h2h_features) -> pd.DataFrame:
    thresholded_df = df.copy()
    missing_columns = sorted(set(h2h_features) - set(thresholded_df.columns))
    if missing_columns:
        raise ValueError(f"Missing H2H feature columns: {missing_columns}")

    threshold_mask = thresholded_df["h2h_matches_prior"] < MIN_H2H_PRIOR_MATCHES
    thresholded_df["h2h_thresholded"] = threshold_mask
    thresholded_df["h2h_usable_after_threshold"] = ~threshold_mask
    thresholded_df.loc[threshold_mask, h2h_features] = np.nan

    thresholded_count = int(threshold_mask.sum())
    usable_count = int((~threshold_mask).sum())
    total_count = len(thresholded_df)
    print("=== H2H Threshold Diagnostics ===")
    print(
        f"Rows thresholded overall (< {MIN_H2H_PRIOR_MATCHES} prior meetings): "
        f"{thresholded_count}/{total_count} ({thresholded_count / total_count:.2%})"
    )
    print(
        f"Rows with usable H2H evidence after thresholding: "
        f"{usable_count}/{total_count} ({usable_count / total_count:.2%})"
    )
    for season_id, season_df in thresholded_df.groupby("season_id", sort=True):
        season_thresholded = int(season_df["h2h_thresholded"].sum())
        print(
            f"- {season_id}: thresholded={season_thresholded}/{len(season_df)} "
            f"({season_thresholded / len(season_df):.2%})"
        )

    return thresholded_df


def build_models() -> dict:
    logistic_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    xgb_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=3,
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )
    return {
        "logistic_elo": clone(logistic_model),
        "logistic_elo_h2h": clone(logistic_model),
        "xgb_elo": clone(xgb_model),
        "xgb_elo_h2h": clone(xgb_model),
    }


def evaluate_predictions(y_true, y_pred, y_prob, labels) -> dict:
    encoded_labels = list(range(len(labels)))
    draw_index = labels.index("D")
    y_prob = _normalize_probabilities(np.asarray(y_prob, dtype=float))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[draw_index],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=encoded_labels)),
        "brier_score": _multiclass_brier_score(y_true, y_prob, labels),
        "draw_recall": float(recall[0]),
        "draw_precision": float(precision[0]),
        "draw_f1": float(f1[0]),
        "confusion_counts": _confusion_counts(y_true, y_pred, labels),
    }


def run_fold(df, fold_config, feature_sets, models) -> dict:
    fold = fold_config["fold"]
    train_seasons = fold_config["train_seasons"]
    validation_seasons = fold_config["validation_seasons"]
    train_df = df.loc[df["season_id"].isin(train_seasons)].copy()
    valid_df = df.loc[df["season_id"].isin(validation_seasons)].copy()

    if FINAL_TEST_SEASON in set(train_df["season_id"]) | set(valid_df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} entered fold {fold}")

    expected_train_rows, expected_valid_rows = EXPECTED_FOLD_ROWS[fold]
    if len(train_df) != expected_train_rows or len(valid_df) != expected_valid_rows:
        raise ValueError(
            f"Fold {fold} expected train/valid rows "
            f"{expected_train_rows}/{expected_valid_rows}, found "
            f"{len(train_df)}/{len(valid_df)}"
        )

    y_train = _encode_targets(train_df[TARGET_COLUMN], CLASS_LABELS)
    y_valid = _encode_targets(valid_df[TARGET_COLUMN], CLASS_LABELS)

    print(f"=== Fold {fold} ===")
    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Validation seasons: {', '.join(validation_seasons)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(valid_df)}")

    for label, fold_df in [("train", train_df), ("validation", valid_df)]:
        thresholded = int(fold_df["h2h_thresholded"].sum())
        usable = int(fold_df["h2h_usable_after_threshold"].sum())
        print(
            f"H2H threshold diagnostics ({label}): "
            f"thresholded={thresholded}/{len(fold_df)} ({thresholded / len(fold_df):.2%}), "
            f"usable={usable}/{len(fold_df)} ({usable / len(fold_df):.2%})"
        )

    result_rows: list[dict[str, Any]] = []
    for model_name, model_template in models.items():
        feature_columns = feature_sets[model_name]
        forbidden_features = sorted(set(feature_columns) & EXCLUDED_FEATURE_COLUMNS)
        if forbidden_features:
            raise ValueError(f"{model_name} selected forbidden features: {forbidden_features}")
        odds_features = [column for column in feature_columns if "odds" in column.lower()]
        if odds_features:
            raise ValueError(f"{model_name} selected odds features: {odds_features}")

        model = clone(model_template)
        X_train = train_df[feature_columns].copy()
        X_valid = valid_df[feature_columns].copy()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        y_prob = _align_predict_proba(model, X_valid, CLASS_LABELS)
        metrics = evaluate_predictions(y_valid, y_pred, y_prob, CLASS_LABELS)

        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}, "
            f"brier_score={metrics['brier_score']:.4f}, "
            f"draw_recall={metrics['draw_recall']:.4f}, "
            f"draw_precision={metrics['draw_precision']:.4f}, "
            f"draw_f1={metrics['draw_f1']:.4f}"
        )
        _print_confusion_counts(metrics["confusion_counts"])

        result_rows.append(
            {
                "fold": fold,
                "model_name": model_name,
                "train_seasons": ", ".join(train_seasons),
                "validation_seasons": ", ".join(validation_seasons),
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
                "train_h2h_thresholded_rows": int(train_df["h2h_thresholded"].sum()),
                "valid_h2h_thresholded_rows": int(valid_df["h2h_thresholded"].sum()),
                "train_h2h_usable_rows": int(train_df["h2h_usable_after_threshold"].sum()),
                "valid_h2h_usable_rows": int(valid_df["h2h_usable_after_threshold"].sum()),
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "draw_recall": metrics["draw_recall"],
                "draw_precision": metrics["draw_precision"],
                "draw_f1": metrics["draw_f1"],
                "confusion_counts": metrics["confusion_counts"],
            }
        )

    return {"fold": fold, "results": result_rows}


def aggregate_results(fold_results) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_result in fold_results:
        rows.extend(fold_result["results"])
    results_df = pd.DataFrame(rows)

    print("=== Aggregate Walk-Forward Metrics ===")
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]
    for model_name, model_df in results_df.groupby("model_name", sort=False):
        print(model_name)
        for metric in metric_columns:
            mean_value = float(model_df[metric].mean())
            std_value = model_df[metric].std(ddof=1)
            if pd.isna(std_value):
                std_value = 0.0
            print(f"- {metric}: mean={mean_value:.4f}, std={float(std_value):.4f}")

    return results_df


def print_h2h_diagnostics(df) -> None:
    print("=== H2H Evidence Diagnostics ===")
    print(f"Development dataframe row count: {len(df)}")
    print("Raw H2H prior-meeting distribution:")
    print(f"- rows with 0 prior meetings: {int((df['h2h_matches_prior'] == 0).sum())}")
    print(f"- rows with 1 prior meeting: {int((df['h2h_matches_prior'] == 1).sum())}")
    print(f"- rows with 2 prior meetings: {int((df['h2h_matches_prior'] == 2).sum())}")
    print(
        f"- rows with >= {MIN_H2H_PRIOR_MATCHES} prior meetings: "
        f"{int((df['h2h_matches_prior'] >= MIN_H2H_PRIOR_MATCHES).sum())}"
    )
    print(f"- max prior meetings: {int(df['h2h_matches_prior'].max())}")
    print("Average prior meetings by development season:")
    for season_id, average in df.groupby("season_id")["h2h_matches_prior"].mean().sort_index().items():
        print(f"- {season_id}: {float(average):.4f}")

    for fold_config in FOLDS:
        fold = fold_config["fold"]
        for label, seasons in [
            ("train", fold_config["train_seasons"]),
            ("validation", fold_config["validation_seasons"]),
        ]:
            fold_df = df.loc[df["season_id"].isin(seasons)]
            usable = int((fold_df["h2h_matches_prior"] >= MIN_H2H_PRIOR_MATCHES).sum())
            print(
                f"Fold {fold} {label} usable H2H rows before threshold: "
                f"{usable}/{len(fold_df)} ({usable / len(fold_df):.2%})"
            )


def print_comparison_and_verdict(results) -> None:
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]
    comparisons = [
        ("logistic_elo", "logistic_elo_h2h"),
        ("xgb_elo", "xgb_elo_h2h"),
    ]

    print("=== Base vs H2H Deltas ===")
    verdict_notes: list[str] = []
    for base_model, h2h_model in comparisons:
        print(f"{h2h_model} - {base_model}")
        fold_deltas: dict[int, dict[str, float]] = {}
        for fold in sorted(results["fold"].unique()):
            base_row = results.loc[
                (results["fold"] == fold) & (results["model_name"] == base_model)
            ].iloc[0]
            h2h_row = results.loc[
                (results["fold"] == fold) & (results["model_name"] == h2h_model)
            ].iloc[0]
            fold_deltas[fold] = {
                metric: float(h2h_row[metric] - base_row[metric])
                for metric in metric_columns
            }
            print(f"Fold {fold}:")
            for metric in metric_columns:
                print(f"- {metric}: {fold_deltas[fold][metric]:+.4f}")

        aggregate_base = results.loc[results["model_name"] == base_model, metric_columns].mean()
        aggregate_h2h = results.loc[results["model_name"] == h2h_model, metric_columns].mean()
        aggregate_delta = aggregate_h2h - aggregate_base
        print("Aggregate:")
        for metric in metric_columns:
            print(f"- {metric}: {aggregate_delta[metric]:+.4f}")

        log_loss_improves_both = all(
            fold_deltas[fold]["log_loss"] <= -0.003 for fold in fold_deltas
        )
        accuracy_improves_both = all(
            fold_deltas[fold]["accuracy"] >= 0.005 for fold in fold_deltas
        )
        no_log_loss_regression_worse_than_limit = all(
            fold_deltas[fold]["log_loss"] <= 0.005 for fold in fold_deltas
        )
        either_fold_draw_recall_worse = any(
            fold_deltas[fold]["draw_recall"] < -0.02 for fold in fold_deltas
        )
        aggregate_draw_recall_worse = aggregate_delta["draw_recall"] < -0.02
        log_loss_worsens_materially = any(
            fold_deltas[fold]["log_loss"] > 0.005 for fold in fold_deltas
        )

        candidate = log_loss_improves_both or (
            accuracy_improves_both and no_log_loss_regression_worse_than_limit
        )
        reject_reasons: list[str] = []
        if not candidate:
            reject_reasons.append("acceptance thresholds were not met on both folds")
        if either_fold_draw_recall_worse:
            reject_reasons.append("draw recall worsened by more than 0.02 on at least one fold")
        if aggregate_draw_recall_worse:
            reject_reasons.append("aggregate draw recall worsened by more than 0.02")
        if log_loss_worsens_materially:
            reject_reasons.append("log_loss worsened materially on at least one fold")

        if candidate and not reject_reasons:
            verdict_notes.append(f"{h2h_model}: candidate improvement")
        else:
            verdict_notes.append(
                f"{h2h_model}: keep experimental/reject for now ({'; '.join(reject_reasons)})"
            )

    print("=== H2H Acceptance Verdict ===")
    for note in verdict_notes:
        print(f"- {note}")
    print("H2H is not accepted as core unless the fold-level validation rules pass.")


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    with conn.connect() as db_conn:
        for table_name in WATCHED_TABLES:
            exists = db_conn.execute(
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
            if exists:
                counts[table_name] = int(
                    db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
                )
            else:
                counts[table_name] = "MISSING"
    return counts


def assert_watched_counts_unchanged(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    if changed:
        raise RuntimeError(f"Watched table counts changed unexpectedly: {changed}")
    print("Watched table counts unchanged.")


def main() -> None:
    print("=== Tier 3 H2H Model Experiment ===")
    conn = get_db_connection()
    validate_historical_match_integrity(conn)

    before_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts before", before_counts)

    df = load_h2h_feature_data(conn)
    print_h2h_diagnostics(df)
    h2h_features = get_h2h_feature_columns()
    df = apply_h2h_threshold(df, h2h_features)
    validate_no_final_holdout_loaded(df)

    base_elo_features = get_base_elo_feature_columns(df)
    feature_sets = {
        "logistic_elo": base_elo_features,
        "logistic_elo_h2h": base_elo_features + h2h_features,
        "xgb_elo": base_elo_features,
        "xgb_elo_h2h": base_elo_features + h2h_features,
    }
    models = build_models()
    fold_results = [
        run_fold(df, fold_config, feature_sets, models)
        for fold_config in FOLDS
    ]
    results = aggregate_results(fold_results)
    print_comparison_and_verdict(results)

    after_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts after", after_counts)
    assert_watched_counts_unchanged(before_counts, after_counts)

    print(f"{FINAL_TEST_SEASON} was not loaded or evaluated.")
    print("No database writes occurred.")
    print("No model artifacts were saved.")
    print("No Streamlit, Tier 2 artifact, betting odds, manager, sentiment, injury, style, Poisson, deployment, or app work occurred.")


def _encode_targets(y: pd.Series, labels: list[str]) -> np.ndarray:
    if y.isna().any():
        raise ValueError(f"Target contains {int(y.isna().sum())} null value(s)")
    unknown_labels = sorted(set(y) - set(labels))
    if unknown_labels:
        raise ValueError(f"Unknown target labels: {unknown_labels}")
    label_to_index = {label: index for index, label in enumerate(labels)}
    return y.map(label_to_index).to_numpy(dtype=int)


def _multiclass_brier_score(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> float:
    y_one_hot = np.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[np.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def _normalize_probabilities(y_proba: np.ndarray) -> np.ndarray:
    row_sums = y_proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Predicted probability row with non-positive sum found")
    return y_proba / row_sums


def _align_predict_proba(model, X_valid: pd.DataFrame, labels: list[str]) -> np.ndarray:
    y_proba = model.predict_proba(X_valid)
    observed_classes = model.named_steps["model"].classes_
    aligned = np.zeros((y_proba.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = y_proba[:, source_index]
    return _normalize_probabilities(aligned)


def _confusion_counts(
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, int]]:
    encoded_labels = list(range(len(labels)))
    matrix = confusion_matrix(y_true_encoded, y_pred_encoded, labels=encoded_labels)
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(labels)
        }
        for true_index, true_label in enumerate(labels)
    }


def _print_confusion_counts(confusion_counts: dict[str, dict[str, int]]) -> None:
    print("Confusion counts (rows=true, columns=pred):")
    print("true\\pred    H    D    A")
    for true_label in CLASS_LABELS:
        row = confusion_counts[true_label]
        print(
            f"{true_label:>4}     "
            f"{row['H']:4d} {row['D']:4d} {row['A']:4d}"
        )


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name, count in counts.items():
        print(f"{table_name}: {count}")


if __name__ == "__main__":
    main()
