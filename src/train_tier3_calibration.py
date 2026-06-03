from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy
import pandas
from scipy.special import softmax
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
CALIBRATION_FOLDS = [
    {
        "fold": 1,
        "fit_seasons": ["2021-22"],
        "calibration_season": "2022-23",
        "validation_season": "2023-24",
    },
    {
        "fold": 2,
        "fit_seasons": ["2021-22", "2022-23"],
        "calibration_season": "2023-24",
        "validation_season": "2024-25",
    },
]
RANDOM_STATE = 42
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
TEMPERATURE_GRID = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
PRIOR_BLEND_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
DRAW_THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
DRAW_MARGIN_GRID = [0.00, 0.03, 0.06, 0.09, 0.12]

EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_SEASON_ROWS = 380
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

FORBIDDEN_POST_MATCH_ELO_COLUMNS = {
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
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
    *FORBIDDEN_POST_MATCH_ELO_COLUMNS,
}


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_elo_feature_data(conn) -> pandas.DataFrame:
    if FINAL_TEST_SEASON in DEV_SEASONS:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be loaded as development data")

    query = text(
        """
        SELECT *
        FROM match_features_v3_elo
        WHERE season_id = ANY(:dev_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pandas.read_sql(query, conn, params={"dev_seasons": DEV_SEASONS})

    errors: list[str] = []
    if len(df) != EXPECTED_DEV_ROW_COUNT:
        errors.append(f"expected {EXPECTED_DEV_ROW_COUNT} rows, found {len(df)}")
    if df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}"
        )
    if df[TARGET_COLUMN].isna().any():
        errors.append(f"null target count: {int(df[TARGET_COLUMN].isna().sum())}")

    seasons_present = sorted(df["season_id"].dropna().unique().tolist())
    if seasons_present != DEV_SEASONS:
        errors.append(f"development seasons {seasons_present} != {DEV_SEASONS}")

    unknown_results = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
    if unknown_results:
        errors.append(f"unknown result labels: {unknown_results}")

    if any(column in df.columns for column in FORBIDDEN_POST_MATCH_ELO_COLUMNS):
        forbidden = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & set(df.columns))
        errors.append(f"forbidden post-match Elo columns loaded: {forbidden}")

    if errors:
        raise ValueError("Elo development dataset validation failed: " + "; ".join(errors))

    df["match_date"] = pandas.to_datetime(df["match_date"])
    validate_no_final_holdout_loaded(df)

    print("=== Elo Development Dataset ===")
    print(f"Development dataframe row count: {len(df)}")
    for season_id, count in df.groupby("season_id").size().sort_index().items():
        print(f"- {season_id}: {int(count)} rows")
    print(f"{FINAL_TEST_SEASON} was not loaded into the calibration dataframe.")
    return df


def validate_no_final_holdout_loaded(df) -> None:
    if FINAL_TEST_SEASON in set(df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} was loaded into development data")
    print(f"PASS: {FINAL_TEST_SEASON} not loaded.")


def get_feature_columns(df) -> list[str]:
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

    forbidden_features = sorted(set(feature_columns) & EXCLUDED_FEATURE_COLUMNS)
    if forbidden_features:
        raise ValueError(f"Forbidden columns selected as features: {forbidden_features}")

    bad_pattern_features = [
        column
        for column in feature_columns
        if column.startswith("h2h_")
        or "poisson" in column.lower()
        or "odds" in column.lower()
    ]
    if bad_pattern_features:
        raise ValueError(f"Forbidden feature family selected: {bad_pattern_features}")

    print(f"Base + Elo feature count: {len(feature_columns)}")
    return feature_columns


def build_model_specs() -> dict:
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
        "logistic_elo": logistic_model,
        "xgb_elo": xgb_model,
    }


def fit_model(model_name, model_spec, X_train, y_train):
    del model_name
    model = clone(model_spec)
    model.fit(X_train, y_train)
    return model


def predict_probabilities(model, X, labels) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned)


def apply_temperature(probabilities, temperature) -> numpy.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    probabilities = _normalize_probabilities(probabilities)
    log_p = numpy.log(numpy.clip(probabilities, 1e-12, 1.0))
    return _normalize_probabilities(softmax(log_p / temperature, axis=1))


def apply_prior_blend(probabilities, class_prior, alpha) -> numpy.ndarray:
    if alpha < 0 or alpha > 1:
        raise ValueError("Prior blend alpha must be between 0 and 1")
    probabilities = _normalize_probabilities(probabilities)
    prior = numpy.asarray(class_prior, dtype=float)
    if prior.shape != (probabilities.shape[1],):
        raise ValueError("Class prior length does not match probability columns")
    prior = prior / prior.sum()
    blended = ((1.0 - alpha) * probabilities) + (alpha * prior.reshape(1, -1))
    return _normalize_probabilities(blended)


def calibrate_probabilities_on_season(probabilities, y_true, labels) -> dict:
    y_true_encoded = _encode_labels(y_true, labels)
    class_counts = numpy.bincount(y_true_encoded, minlength=len(labels)).astype(float)
    class_prior = class_counts / class_counts.sum()
    encoded_labels = list(range(len(labels)))

    best_config: dict[str, Any] | None = None
    for temperature in TEMPERATURE_GRID:
        temperature_scaled = apply_temperature(probabilities, temperature)
        for alpha in PRIOR_BLEND_GRID:
            calibrated = apply_prior_blend(temperature_scaled, class_prior, alpha)
            candidate_log_loss = float(
                log_loss(y_true_encoded, calibrated, labels=encoded_labels)
            )
            candidate_brier = _multiclass_brier_score(
                y_true_encoded,
                calibrated,
                labels,
            )
            if best_config is None or (
                candidate_log_loss,
                candidate_brier,
                abs(temperature - 1.0),
                alpha,
            ) < (
                best_config["calibration_log_loss"],
                best_config["calibration_brier_score"],
                abs(best_config["temperature"] - 1.0),
                best_config["prior_blend_alpha"],
            ):
                best_config = {
                    "temperature": temperature,
                    "prior_blend_alpha": alpha,
                    "class_prior": class_prior.tolist(),
                    "calibration_log_loss": candidate_log_loss,
                    "calibration_brier_score": candidate_brier,
                }

    if best_config is None:
        raise RuntimeError("No probability calibration config was selected")
    return best_config


def apply_probability_calibration(probabilities, calibration_config) -> numpy.ndarray:
    temperature_scaled = apply_temperature(
        probabilities,
        calibration_config["temperature"],
    )
    return apply_prior_blend(
        temperature_scaled,
        calibration_config["class_prior"],
        calibration_config["prior_blend_alpha"],
    )


def predict_argmax(probabilities, labels) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    return [labels[int(index)] for index in prediction_indexes]


def predict_with_draw_rule(probabilities, labels, draw_threshold, draw_margin) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    predictions = predict_argmax(probabilities, labels)
    draw_index = labels.index("D")
    non_draw_indexes = [index for index, label in enumerate(labels) if label != "D"]
    best_non_draw_probabilities = probabilities[:, non_draw_indexes].max(axis=1)
    draw_probabilities = probabilities[:, draw_index]
    draw_mask = (
        (draw_probabilities >= draw_threshold)
        & ((best_non_draw_probabilities - draw_probabilities) <= draw_margin)
    )
    return [
        "D" if draw_mask[row_index] else prediction
        for row_index, prediction in enumerate(predictions)
    ]


def tune_draw_rule(probabilities, y_true, labels) -> dict:
    argmax_predictions = predict_argmax(probabilities, labels)
    argmax_metrics = evaluate_predictions(y_true, argmax_predictions, probabilities, labels)

    best_config: dict[str, Any] | None = None
    for draw_threshold in DRAW_THRESHOLD_GRID:
        for draw_margin in DRAW_MARGIN_GRID:
            rule_predictions = predict_with_draw_rule(
                probabilities,
                labels,
                draw_threshold,
                draw_margin,
            )
            metrics = evaluate_predictions(y_true, rule_predictions, probabilities, labels)
            if metrics["accuracy"] < argmax_metrics["accuracy"] - 0.03:
                continue

            predicted_draws = metrics["predicted_class_distribution"]["D"]
            candidate_key = (
                metrics["draw_f1"],
                metrics["accuracy"],
                metrics["draw_precision"],
                -predicted_draws,
                -draw_threshold,
                -draw_margin,
            )
            if best_config is None or candidate_key > best_config["selection_key"]:
                best_config = {
                    "draw_threshold": draw_threshold,
                    "draw_margin": draw_margin,
                    "calibration_argmax_accuracy": argmax_metrics["accuracy"],
                    "calibration_argmax_draw_recall": argmax_metrics["draw_recall"],
                    "calibration_argmax_draw_precision": argmax_metrics["draw_precision"],
                    "calibration_argmax_draw_f1": argmax_metrics["draw_f1"],
                    "calibration_rule_accuracy": metrics["accuracy"],
                    "calibration_rule_draw_recall": metrics["draw_recall"],
                    "calibration_rule_draw_precision": metrics["draw_precision"],
                    "calibration_rule_draw_f1": metrics["draw_f1"],
                    "calibration_rule_predicted_draws": predicted_draws,
                    "selection_key": candidate_key,
                    "rule_available": True,
                }

    if best_config is None:
        best_config = {
            "draw_threshold": None,
            "draw_margin": None,
            "calibration_argmax_accuracy": argmax_metrics["accuracy"],
            "calibration_argmax_draw_recall": argmax_metrics["draw_recall"],
            "calibration_argmax_draw_precision": argmax_metrics["draw_precision"],
            "calibration_argmax_draw_f1": argmax_metrics["draw_f1"],
            "calibration_rule_accuracy": argmax_metrics["accuracy"],
            "calibration_rule_draw_recall": argmax_metrics["draw_recall"],
            "calibration_rule_draw_precision": argmax_metrics["draw_precision"],
            "calibration_rule_draw_f1": argmax_metrics["draw_f1"],
            "calibration_rule_predicted_draws": argmax_metrics[
                "predicted_class_distribution"
            ]["D"],
            "selection_key": None,
            "rule_available": False,
        }

    best_config.pop("selection_key", None)
    return best_config


def evaluate_predictions(y_true, y_pred, probabilities, labels) -> dict:
    y_true_encoded = _encode_labels(y_true, labels)
    y_pred_encoded = _encode_labels(y_pred, labels)
    probabilities = _normalize_probabilities(probabilities)
    encoded_labels = list(range(len(labels)))
    draw_index = labels.index("D")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=[draw_index],
        average=None,
        zero_division=0,
    )
    predicted_distribution = {
        label: int(numpy.sum(y_pred_encoded == index))
        for index, label in enumerate(labels)
    }
    mean_predicted_probability = {
        label: float(probabilities[:, index].mean())
        for index, label in enumerate(labels)
    }

    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(log_loss(y_true_encoded, probabilities, labels=encoded_labels)),
        "brier_score": _multiclass_brier_score(y_true_encoded, probabilities, labels),
        "draw_recall": float(recall[0]),
        "draw_precision": float(precision[0]),
        "draw_f1": float(f1[0]),
        "confusion_counts": _confusion_counts(y_true_encoded, y_pred_encoded, labels),
        "predicted_class_distribution": predicted_distribution,
        "mean_predicted_probability": mean_predicted_probability,
    }


def run_calibration_fold(df, fold_config, model_specs, feature_columns) -> list[dict]:
    fold = fold_config["fold"]
    fit_seasons = fold_config["fit_seasons"]
    calibration_season = fold_config["calibration_season"]
    validation_season = fold_config["validation_season"]

    fit_df = df.loc[df["season_id"].isin(fit_seasons)].copy()
    calibration_df = df.loc[df["season_id"] == calibration_season].copy()
    validation_df = df.loc[df["season_id"] == validation_season].copy()

    _validate_fold_window(
        fold,
        fit_df,
        calibration_df,
        validation_df,
        fit_seasons,
        calibration_season,
        validation_season,
    )

    X_fit = fit_df[feature_columns].copy()
    X_calibration = calibration_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()
    y_fit_encoded = _encode_labels(fit_df[TARGET_COLUMN], LABELS)
    y_calibration = calibration_df[TARGET_COLUMN].tolist()
    y_validation = validation_df[TARGET_COLUMN].tolist()

    print(f"=== Calibration Fold {fold} ===")
    print(f"Model fit seasons: {', '.join(fit_seasons)}")
    print(f"Calibration season: {calibration_season}")
    print(f"Validation season: {validation_season}")
    print(f"Fit rows: {len(fit_df)}")
    print(f"Calibration rows: {len(calibration_df)}")
    print(f"Validation rows: {len(validation_df)}")

    result_rows: list[dict[str, Any]] = []
    for base_model_name, model_spec in model_specs.items():
        model = fit_model(base_model_name, model_spec, X_fit, y_fit_encoded)
        calibration_probabilities = predict_probabilities(
            model,
            X_calibration,
            LABELS,
        )
        validation_probabilities = predict_probabilities(model, X_validation, LABELS)

        calibration_config = calibrate_probabilities_on_season(
            calibration_probabilities,
            y_calibration,
            LABELS,
        )
        calibrated_calibration_probabilities = apply_probability_calibration(
            calibration_probabilities,
            calibration_config,
        )
        calibrated_validation_probabilities = apply_probability_calibration(
            validation_probabilities,
            calibration_config,
        )
        draw_rule_config = tune_draw_rule(
            calibrated_calibration_probabilities,
            y_calibration,
            LABELS,
        )

        print(
            f"{base_model_name} calibration: "
            f"temperature={calibration_config['temperature']}, "
            f"prior_blend_alpha={calibration_config['prior_blend_alpha']}, "
            f"calibration_log_loss={calibration_config['calibration_log_loss']:.4f}, "
            f"calibration_brier={calibration_config['calibration_brier_score']:.4f}"
        )
        print(
            f"{base_model_name} draw rule: "
            f"threshold={draw_rule_config['draw_threshold']}, "
            f"margin={draw_rule_config['draw_margin']}, "
            f"calibration_draw_f1="
            f"{draw_rule_config['calibration_rule_draw_f1']:.4f}, "
            f"calibration_accuracy="
            f"{draw_rule_config['calibration_rule_accuracy']:.4f}"
        )

        model_results = [
            (
                f"{base_model_name}_uncalibrated",
                validation_probabilities,
                predict_argmax(validation_probabilities, LABELS),
                {},
            ),
            (
                f"{base_model_name}_calibrated",
                calibrated_validation_probabilities,
                predict_argmax(calibrated_validation_probabilities, LABELS),
                calibration_config,
            ),
            (
                f"{base_model_name}_calibrated_draw_rule",
                calibrated_validation_probabilities,
                _apply_draw_rule_or_argmax(
                    calibrated_validation_probabilities,
                    LABELS,
                    draw_rule_config,
                ),
                calibration_config,
            ),
        ]

        for model_name, probabilities, predictions, config in model_results:
            metrics = evaluate_predictions(
                y_validation,
                predictions,
                probabilities,
                LABELS,
            )
            _print_model_result(model_name, metrics)
            result_rows.append(
                {
                    "fold": fold,
                    "model_name": model_name,
                    "base_model_name": base_model_name,
                    "fit_seasons": ", ".join(fit_seasons),
                    "calibration_season": calibration_season,
                    "validation_season": validation_season,
                    "fit_rows": len(fit_df),
                    "calibration_rows": len(calibration_df),
                    "validation_rows": len(validation_df),
                    "temperature": config.get("temperature"),
                    "prior_blend_alpha": config.get("prior_blend_alpha"),
                    "class_prior": config.get("class_prior"),
                    "calibration_log_loss": config.get("calibration_log_loss"),
                    "calibration_brier_score": config.get(
                        "calibration_brier_score"
                    ),
                    "draw_threshold": draw_rule_config["draw_threshold"],
                    "draw_margin": draw_rule_config["draw_margin"],
                    "draw_rule_available": draw_rule_config["rule_available"],
                    "calibration_argmax_accuracy": draw_rule_config[
                        "calibration_argmax_accuracy"
                    ],
                    "calibration_argmax_draw_recall": draw_rule_config[
                        "calibration_argmax_draw_recall"
                    ],
                    "calibration_argmax_draw_precision": draw_rule_config[
                        "calibration_argmax_draw_precision"
                    ],
                    "calibration_argmax_draw_f1": draw_rule_config[
                        "calibration_argmax_draw_f1"
                    ],
                    "calibration_rule_accuracy": draw_rule_config[
                        "calibration_rule_accuracy"
                    ],
                    "calibration_rule_draw_recall": draw_rule_config[
                        "calibration_rule_draw_recall"
                    ],
                    "calibration_rule_draw_precision": draw_rule_config[
                        "calibration_rule_draw_precision"
                    ],
                    "calibration_rule_draw_f1": draw_rule_config[
                        "calibration_rule_draw_f1"
                    ],
                    "calibration_rule_predicted_draws": draw_rule_config[
                        "calibration_rule_predicted_draws"
                    ],
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "draw_recall": metrics["draw_recall"],
                    "draw_precision": metrics["draw_precision"],
                    "draw_f1": metrics["draw_f1"],
                    "confusion_counts": metrics["confusion_counts"],
                    "predicted_class_distribution": metrics[
                        "predicted_class_distribution"
                    ],
                    "mean_predicted_probability": metrics[
                        "mean_predicted_probability"
                    ],
                }
            )

    return result_rows


def aggregate_results(results) -> pandas.DataFrame:
    results_df = pandas.DataFrame(results)
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]

    print("=== Aggregate Calibration Metrics ===")
    for model_name, model_df in results_df.groupby("model_name", sort=False):
        print(model_name)
        for metric in metric_columns:
            mean_value = float(model_df[metric].mean())
            std_value = model_df[metric].std(ddof=1)
            if pandas.isna(std_value):
                std_value = 0.0
            print(f"- {metric}: mean={mean_value:.4f}, std={float(std_value):.4f}")
        mean_probs = _mean_probability_dict(model_df["mean_predicted_probability"])
        print(
            "- mean_predicted_probability: "
            + ", ".join(f"{label}={mean_probs[label]:.4f}" for label in LABELS)
        )

    return results_df


def print_comparison_and_verdict(results) -> None:
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]
    model_families = ["logistic_elo", "xgb_elo"]

    print("=== Calibration and Draw-Rule Comparison ===")
    for family in model_families:
        uncalibrated = f"{family}_uncalibrated"
        calibrated = f"{family}_calibrated"
        draw_rule = f"{family}_calibrated_draw_rule"

        print(f"{family}: calibrated - uncalibrated")
        calibration_fold_deltas: dict[int, dict[str, float]] = {}
        for fold in sorted(results["fold"].unique()):
            base_row = _single_result_row(results, fold, uncalibrated)
            calibrated_row = _single_result_row(results, fold, calibrated)
            deltas = {
                metric: float(calibrated_row[metric] - base_row[metric])
                for metric in metric_columns
            }
            calibration_fold_deltas[fold] = deltas
            print(f"Fold {fold}:")
            for metric in metric_columns:
                print(f"- {metric}: {deltas[metric]:+.4f}")

        base_agg = _aggregate_model_metrics(results, uncalibrated, metric_columns)
        calibrated_agg = _aggregate_model_metrics(results, calibrated, metric_columns)
        calibration_aggregate_delta = calibrated_agg - base_agg
        print("Aggregate:")
        for metric in metric_columns:
            print(f"- {metric}: {calibration_aggregate_delta[metric]:+.4f}")

        probability_candidate = (
            calibration_aggregate_delta["log_loss"] <= -0.005
            and calibration_aggregate_delta["brier_score"] <= -0.005
            and all(
                deltas["log_loss"] <= 0.005
                for deltas in calibration_fold_deltas.values()
            )
        )

        print(f"{family}: draw rule - calibrated")
        draw_rule_fold_deltas: dict[int, dict[str, float]] = {}
        too_many_draw_reasons: list[str] = []
        for fold in sorted(results["fold"].unique()):
            calibrated_row = _single_result_row(results, fold, calibrated)
            draw_row = _single_result_row(results, fold, draw_rule)
            deltas = {
                metric: float(draw_row[metric] - calibrated_row[metric])
                for metric in metric_columns
            }
            draw_rule_fold_deltas[fold] = deltas
            print(f"Fold {fold}:")
            for metric in metric_columns:
                print(f"- {metric}: {deltas[metric]:+.4f}")

            draw_count = draw_row["predicted_class_distribution"]["D"]
            draw_rate = draw_count / draw_row["validation_rows"]
            actual_draw_rate = _actual_draw_rate(results, fold)
            if (
                draw_rate > actual_draw_rate + 0.15
                and draw_row["draw_precision"] <= calibrated_row["draw_precision"]
            ):
                too_many_draw_reasons.append(
                    f"fold {fold} predicted draw rate {draw_rate:.2%} "
                    f"vs actual {actual_draw_rate:.2%}"
                )

        draw_rule_candidate = (
            all(deltas["draw_f1"] >= 0.03 for deltas in draw_rule_fold_deltas.values())
            and all(
                deltas["draw_recall"] > 0 for deltas in draw_rule_fold_deltas.values()
            )
            and all(
                deltas["accuracy"] >= -0.02
                for deltas in draw_rule_fold_deltas.values()
            )
            and all(
                abs(deltas["log_loss"]) < 1e-12
                and abs(deltas["brier_score"]) < 1e-12
                for deltas in draw_rule_fold_deltas.values()
            )
            and not too_many_draw_reasons
        )

        print(f"=== {family} Acceptance Verdict ===")
        if probability_candidate:
            print("- Probability calibration: core candidate")
        else:
            print("- Probability calibration: reject/keep experimental for now")

        if draw_rule_candidate:
            print("- Draw rule: experimental candidate")
        else:
            print("- Draw rule: reject/keep experimental for now")
            if too_many_draw_reasons:
                print("- Draw-rate caution: " + "; ".join(too_many_draw_reasons))

    print(f"{FINAL_TEST_SEASON} was not used for tuning, calibration, or metrics.")


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
    print("=== Tier 3 Elo Calibration and Draw Rule Experiment ===")
    conn = get_db_connection()
    validate_historical_match_integrity(conn)

    before_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts before", before_counts)

    df = load_elo_feature_data(conn)
    feature_columns = get_feature_columns(df)
    model_specs = build_model_specs()

    all_results: list[dict[str, Any]] = []
    for fold_config in CALIBRATION_FOLDS:
        all_results.extend(
            run_calibration_fold(df, fold_config, model_specs, feature_columns)
        )

    results_df = aggregate_results(all_results)
    print_comparison_and_verdict(results_df)

    after_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts after", after_counts)
    assert_watched_counts_unchanged(before_counts, after_counts)

    print(f"{FINAL_TEST_SEASON} was not loaded, evaluated, tuned, or calibrated.")
    print("No database writes occurred.")
    print("No model artifacts were saved.")
    print("No Streamlit, Tier 2 artifact, H2H, Poisson-feature, betting odds, manager, sentiment, injury, style, deployment, or app work occurred.")


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Predicted probability row with non-positive sum found")
    normalized = probabilities / row_sums
    if not numpy.all(numpy.isfinite(normalized)):
        raise ValueError("Non-finite probability value found")
    return normalized


def _encode_labels(values, labels: list[str]) -> numpy.ndarray:
    if isinstance(values, pandas.Series):
        raw_values = values.tolist()
    else:
        raw_values = list(values)

    if all(isinstance(value, (int, numpy.integer)) for value in raw_values):
        encoded = numpy.asarray(raw_values, dtype=int)
        unknown_indexes = sorted(set(encoded.tolist()) - set(range(len(labels))))
        if unknown_indexes:
            raise ValueError(f"Unknown encoded label indexes: {unknown_indexes}")
        return encoded

    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_labels = sorted(set(raw_values) - set(labels))
    if unknown_labels:
        raise ValueError(f"Unknown label values: {unknown_labels}")
    return numpy.asarray([label_to_index[value] for value in raw_values], dtype=int)


def _multiclass_brier_score(
    y_true_encoded: numpy.ndarray,
    probabilities: numpy.ndarray,
    labels: list[str],
) -> float:
    probabilities = _normalize_probabilities(probabilities)
    y_one_hot = numpy.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[numpy.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(numpy.mean(numpy.sum((probabilities - y_one_hot) ** 2, axis=1)))


def _confusion_counts(
    y_true_encoded: numpy.ndarray,
    y_pred_encoded: numpy.ndarray,
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=list(range(len(labels))),
    )
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
    for true_label in LABELS:
        row = confusion_counts[true_label]
        print(
            f"{true_label:>4}     "
            f"{row['H']:4d} {row['D']:4d} {row['A']:4d}"
        )


def _print_model_result(model_name: str, metrics: dict[str, Any]) -> None:
    distribution = metrics["predicted_class_distribution"]
    mean_probs = metrics["mean_predicted_probability"]
    print(
        f"{model_name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"brier_score={metrics['brier_score']:.4f}, "
        f"draw_recall={metrics['draw_recall']:.4f}, "
        f"draw_precision={metrics['draw_precision']:.4f}, "
        f"draw_f1={metrics['draw_f1']:.4f}"
    )
    print(
        "Predicted distribution: "
        + ", ".join(f"{label}={distribution[label]}" for label in LABELS)
    )
    print(
        "Mean predicted probabilities: "
        + ", ".join(f"{label}={mean_probs[label]:.4f}" for label in LABELS)
    )
    _print_confusion_counts(metrics["confusion_counts"])


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name, count in counts.items():
        print(f"{table_name}: {count}")


def _validate_fold_window(
    fold: int,
    fit_df: pandas.DataFrame,
    calibration_df: pandas.DataFrame,
    validation_df: pandas.DataFrame,
    fit_seasons: list[str],
    calibration_season: str,
    validation_season: str,
) -> None:
    if any(
        FINAL_TEST_SEASON in set(frame["season_id"])
        for frame in [fit_df, calibration_df, validation_df]
    ):
        raise ValueError(f"{FINAL_TEST_SEASON} entered calibration fold {fold}")
    if len(fit_df) != len(fit_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(fit_seasons) * EXPECTED_SEASON_ROWS} "
            f"fit rows, found {len(fit_df)}"
        )
    if len(calibration_df) != EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {EXPECTED_SEASON_ROWS} calibration rows, "
            f"found {len(calibration_df)}"
        )
    if len(validation_df) != EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {EXPECTED_SEASON_ROWS} validation rows, "
            f"found {len(validation_df)}"
        )

    if fit_df["match_date"].max() >= calibration_df["match_date"].min():
        raise ValueError(f"Fold {fold} fit/calibration date leakage")
    if calibration_df["match_date"].max() >= validation_df["match_date"].min():
        raise ValueError(f"Fold {fold} calibration/validation date leakage")

    print(
        f"Fold {fold} date windows: "
        f"fit max {fit_df['match_date'].max().date()} < "
        f"calibration min {calibration_df['match_date'].min().date()}, "
        f"calibration max {calibration_df['match_date'].max().date()} < "
        f"validation min {validation_df['match_date'].min().date()}"
    )
    print(
        f"Fold {fold} seasons verified: fit={', '.join(fit_seasons)}, "
        f"calibration={calibration_season}, validation={validation_season}"
    )


def _apply_draw_rule_or_argmax(probabilities, labels, draw_rule_config) -> list[str]:
    if not draw_rule_config["rule_available"]:
        return predict_argmax(probabilities, labels)
    return predict_with_draw_rule(
        probabilities,
        labels,
        draw_rule_config["draw_threshold"],
        draw_rule_config["draw_margin"],
    )


def _mean_probability_dict(series: pandas.Series) -> dict[str, float]:
    totals = {label: 0.0 for label in LABELS}
    count = len(series)
    for probability_dict in series:
        for label in LABELS:
            totals[label] += float(probability_dict[label])
    return {label: totals[label] / count for label in LABELS}


def _single_result_row(results: pandas.DataFrame, fold: int, model_name: str) -> pandas.Series:
    rows = results.loc[
        (results["fold"] == fold) & (results["model_name"] == model_name)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one result row for fold={fold}, model={model_name}")
    return rows.iloc[0]


def _aggregate_model_metrics(
    results: pandas.DataFrame,
    model_name: str,
    metric_columns: list[str],
) -> pandas.Series:
    rows = results.loc[results["model_name"] == model_name]
    if rows.empty:
        raise ValueError(f"Missing model results for {model_name}")
    return rows[metric_columns].mean()


def _actual_draw_rate(results: pandas.DataFrame, fold: int) -> float:
    row = _single_result_row(results, fold, results["model_name"].iloc[0])
    confusion = row["confusion_counts"]
    actual_draws = sum(confusion["D"].values())
    return actual_draws / row["validation_rows"]


if __name__ == "__main__":
    main()
