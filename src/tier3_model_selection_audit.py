from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy
import pandas
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
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
RANDOM_STATE = 42

TEMPERATURE_GRID = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
PRIOR_BLEND_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
DRAW_THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
DRAW_MARGIN_GRID = [0.00, 0.03, 0.06, 0.09, 0.12]

EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_SEASON_ROWS = 380

EXPANDING_FOLDS = [
    {
        "fold": 1,
        "train_seasons": ["2021-22", "2022-23"],
        "validation_season": "2023-24",
    },
    {
        "fold": 2,
        "train_seasons": ["2021-22", "2022-23", "2023-24"],
        "validation_season": "2024-25",
    },
]
CALIBRATED_FOLDS = [
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
EXPANDING_CANDIDATES = [
    ("logistic_base_expanding", "logistic", "base"),
    ("logistic_elo_expanding", "logistic", "elo"),
    ("xgb_base_expanding", "xgb", "base"),
    ("xgb_elo_expanding", "xgb", "elo"),
]
CALIBRATED_CANDIDATES = [
    ("logistic_elo", "logistic"),
    ("xgb_elo", "xgb"),
]
PROBABILITY_CANDIDATES = [
    "logistic_base_expanding",
    "logistic_elo_expanding",
    "xgb_base_expanding",
    "xgb_elo_expanding",
    "logistic_elo_calibrated",
    "xgb_elo_calibrated",
]

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


def load_feature_data(conn) -> pandas.DataFrame:
    if FINAL_TEST_SEASON in DEV_SEASONS:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be a development season")

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
    if df.empty:
        errors.append("development feature data is empty")
    if df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}"
        )
    if df[TARGET_COLUMN].isna().any():
        errors.append(f"null target count: {int(df[TARGET_COLUMN].isna().sum())}")

    seasons_present = sorted(df["season_id"].dropna().unique().tolist())
    if seasons_present != DEV_SEASONS:
        errors.append(f"development seasons {seasons_present} != {DEV_SEASONS}")

    season_counts = df.groupby("season_id").size().to_dict()
    bad_counts = {
        season_id: int(count)
        for season_id, count in season_counts.items()
        if int(count) != EXPECTED_SEASON_ROWS
    }
    if bad_counts:
        errors.append(f"bad development season row counts: {bad_counts}")

    unknown_results = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
    if unknown_results:
        errors.append(f"unknown result labels: {unknown_results}")

    forbidden_loaded = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & set(df.columns))
    if forbidden_loaded:
        errors.append(f"forbidden post-match Elo columns loaded: {forbidden_loaded}")

    forbidden_family_columns = [
        column
        for column in df.columns
        if column.startswith("h2h_")
        or "poisson" in column.lower()
        or "odds" in column.lower()
    ]
    if forbidden_family_columns:
        errors.append(f"forbidden feature-family columns present: {forbidden_family_columns}")

    if errors:
        raise ValueError("Tier 3 audit feature data validation failed: " + "; ".join(errors))

    df["match_date"] = pandas.to_datetime(df["match_date"])
    validate_no_final_holdout_loaded(df)

    print("=== Development Feature Data ===")
    print(f"Development dataframe row count: {len(df)}")
    for season_id, count in df.groupby("season_id").size().sort_index().items():
        print(f"- {season_id}: {int(count)} rows")
    print(f"{FINAL_TEST_SEASON} was not loaded anywhere in the audit dataframe.")
    return df


def validate_no_final_holdout_loaded(df) -> None:
    if FINAL_TEST_SEASON in set(df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} was loaded into audit data")
    print(f"PASS: {FINAL_TEST_SEASON} not loaded.")


def get_feature_sets(df) -> dict:
    base_features = [
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
    ]
    elo_features = [
        "home_elo_before",
        "away_elo_before",
        "elo_diff_before",
        "elo_diff_home_adjusted",
        "expected_home_score",
        "expected_away_score",
    ]
    feature_sets = {
        "base": base_features,
        "elo": base_features + elo_features,
    }

    for feature_set_name, columns in feature_sets.items():
        missing_columns = sorted(set(columns) - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Missing {feature_set_name} feature columns: {missing_columns}"
            )
        forbidden_features = sorted(set(columns) & EXCLUDED_FEATURE_COLUMNS)
        if forbidden_features:
            raise ValueError(
                f"Forbidden columns selected in {feature_set_name}: {forbidden_features}"
            )
        forbidden_family_features = [
            column
            for column in columns
            if column.startswith("h2h_")
            or "poisson" in column.lower()
            or "odds" in column.lower()
        ]
        if forbidden_family_features:
            raise ValueError(
                f"Forbidden feature family selected in {feature_set_name}: "
                f"{forbidden_family_features}"
            )

    print(f"Base feature count: {len(feature_sets['base'])}")
    print(f"Base + Elo feature count: {len(feature_sets['elo'])}")
    return feature_sets


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
        "logistic": logistic_model,
        "xgb": xgb_model,
    }


def run_expanding_window_fold(df, fold_config, model_specs, feature_sets) -> list[dict]:
    fold = fold_config["fold"]
    train_seasons = fold_config["train_seasons"]
    validation_season = fold_config["validation_season"]
    train_df = df.loc[df["season_id"].isin(train_seasons)].copy()
    validation_df = df.loc[df["season_id"] == validation_season].copy()

    _validate_expanding_fold(
        fold,
        train_df,
        validation_df,
        train_seasons,
        validation_season,
    )

    print(f"=== Expanding Fold {fold} ===")
    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Validation season: {validation_season}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    y_train = _encode_labels(train_df[TARGET_COLUMN], LABELS)
    y_validation = validation_df[TARGET_COLUMN].tolist()

    rows: list[dict[str, Any]] = []
    for model_name, model_key, feature_set_key in EXPANDING_CANDIDATES:
        model = clone(model_specs[model_key])
        features = feature_sets[feature_set_key]
        model.fit(train_df[features].copy(), y_train)
        probabilities = _predict_probabilities(model, validation_df[features].copy(), LABELS)
        predictions = _predict_argmax(probabilities, LABELS)
        metrics = evaluate_predictions(y_validation, predictions, probabilities, LABELS)
        _print_model_result(model_name, metrics)

        rows.append(
            {
                "fold": fold,
                "mode": "expanding",
                "model_name": model_name,
                "base_model_name": model_key,
                "feature_set": feature_set_key,
                "train_seasons": ", ".join(train_seasons),
                "fit_seasons": None,
                "calibration_season": None,
                "validation_season": validation_season,
                "train_rows": len(train_df),
                "fit_rows": None,
                "calibration_rows": None,
                "validation_rows": len(validation_df),
                "temperature": None,
                "prior_blend_alpha": None,
                "class_prior": None,
                "calibration_log_loss": None,
                "calibration_brier_score": None,
                "draw_threshold": None,
                "draw_margin": None,
                "draw_rule_available": False,
                "validation_used_for_tuning": False,
                **metrics,
            }
        )
    return rows


def run_calibrated_fold(df, fold_config, model_specs, feature_sets) -> list[dict]:
    fold = fold_config["fold"]
    fit_seasons = fold_config["fit_seasons"]
    calibration_season = fold_config["calibration_season"]
    validation_season = fold_config["validation_season"]

    fit_df = df.loc[df["season_id"].isin(fit_seasons)].copy()
    calibration_df = df.loc[df["season_id"] == calibration_season].copy()
    validation_df = df.loc[df["season_id"] == validation_season].copy()
    _validate_calibrated_fold(
        fold,
        fit_df,
        calibration_df,
        validation_df,
        fit_seasons,
        calibration_season,
        validation_season,
    )

    print(f"=== Calibrated Fold {fold} ===")
    print(f"Fit seasons: {', '.join(fit_seasons)}")
    print(f"Calibration season: {calibration_season}")
    print(f"Validation season: {validation_season}")
    print(f"Fit rows: {len(fit_df)}")
    print(f"Calibration rows: {len(calibration_df)}")
    print(f"Validation rows: {len(validation_df)}")

    features = feature_sets["elo"]
    y_fit = _encode_labels(fit_df[TARGET_COLUMN], LABELS)
    y_calibration = calibration_df[TARGET_COLUMN].tolist()
    y_validation = validation_df[TARGET_COLUMN].tolist()

    rows: list[dict[str, Any]] = []
    for base_model_name, model_key in CALIBRATED_CANDIDATES:
        model = clone(model_specs[model_key])
        model.fit(fit_df[features].copy(), y_fit)

        calibration_probabilities = _predict_probabilities(
            model,
            calibration_df[features].copy(),
            LABELS,
        )
        validation_probabilities = _predict_probabilities(
            model,
            validation_df[features].copy(),
            LABELS,
        )

        calibration_config = select_calibration_config(
            calibration_probabilities,
            y_calibration,
            LABELS,
        )
        calibrated_calibration_probabilities = _apply_calibration_config(
            calibration_probabilities,
            calibration_config,
        )
        calibrated_validation_probabilities = _apply_calibration_config(
            validation_probabilities,
            calibration_config,
        )
        draw_rule = select_draw_rule(
            calibrated_calibration_probabilities,
            y_calibration,
            LABELS,
        )

        print(
            f"{base_model_name} calibration selected on {calibration_season}: "
            f"temperature={calibration_config['temperature']}, "
            f"prior_blend_alpha={calibration_config['prior_blend_alpha']}, "
            f"calibration_log_loss={calibration_config['calibration_log_loss']:.4f}, "
            f"calibration_brier={calibration_config['calibration_brier_score']:.4f}"
        )
        print(
            f"{base_model_name} draw rule selected on {calibration_season}: "
            f"threshold={draw_rule['draw_threshold']}, "
            f"margin={draw_rule['draw_margin']}, "
            f"calibration_draw_f1={draw_rule['calibration_rule_draw_f1']:.4f}, "
            f"calibration_accuracy={draw_rule['calibration_rule_accuracy']:.4f}"
        )

        calibrated_predictions = _predict_argmax(calibrated_validation_probabilities, LABELS)
        draw_rule_predictions = _apply_draw_rule_or_argmax(
            calibrated_validation_probabilities,
            LABELS,
            draw_rule,
        )
        model_outputs = [
            (
                f"{base_model_name}_calibrated",
                calibrated_predictions,
                False,
            ),
            (
                f"{base_model_name}_calibrated_draw_rule",
                draw_rule_predictions,
                True,
            ),
        ]
        for model_name, predictions, is_draw_rule_model in model_outputs:
            metrics = evaluate_predictions(
                y_validation,
                predictions,
                calibrated_validation_probabilities,
                LABELS,
            )
            _print_model_result(model_name, metrics)
            rows.append(
                {
                    "fold": fold,
                    "mode": "calibrated_draw_rule"
                    if is_draw_rule_model
                    else "calibrated",
                    "model_name": model_name,
                    "base_model_name": base_model_name,
                    "feature_set": "elo",
                    "train_seasons": None,
                    "fit_seasons": ", ".join(fit_seasons),
                    "calibration_season": calibration_season,
                    "validation_season": validation_season,
                    "train_rows": None,
                    "fit_rows": len(fit_df),
                    "calibration_rows": len(calibration_df),
                    "validation_rows": len(validation_df),
                    "temperature": calibration_config["temperature"],
                    "prior_blend_alpha": calibration_config["prior_blend_alpha"],
                    "class_prior": calibration_config["class_prior"],
                    "calibration_log_loss": calibration_config[
                        "calibration_log_loss"
                    ],
                    "calibration_brier_score": calibration_config[
                        "calibration_brier_score"
                    ],
                    "draw_threshold": draw_rule["draw_threshold"],
                    "draw_margin": draw_rule["draw_margin"],
                    "draw_rule_available": draw_rule["rule_available"],
                    "calibration_argmax_accuracy": draw_rule[
                        "calibration_argmax_accuracy"
                    ],
                    "calibration_argmax_draw_recall": draw_rule[
                        "calibration_argmax_draw_recall"
                    ],
                    "calibration_argmax_draw_precision": draw_rule[
                        "calibration_argmax_draw_precision"
                    ],
                    "calibration_argmax_draw_f1": draw_rule[
                        "calibration_argmax_draw_f1"
                    ],
                    "calibration_rule_accuracy": draw_rule[
                        "calibration_rule_accuracy"
                    ],
                    "calibration_rule_draw_recall": draw_rule[
                        "calibration_rule_draw_recall"
                    ],
                    "calibration_rule_draw_precision": draw_rule[
                        "calibration_rule_draw_precision"
                    ],
                    "calibration_rule_draw_f1": draw_rule[
                        "calibration_rule_draw_f1"
                    ],
                    "calibration_rule_predicted_draws": draw_rule[
                        "calibration_rule_predicted_draws"
                    ],
                    "validation_used_for_tuning": False,
                    **metrics,
                }
            )
    return rows


def apply_temperature(probabilities, temperature) -> numpy.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    probabilities = _normalize_probabilities(probabilities)
    log_p = numpy.log(numpy.clip(probabilities, 1e-12, 1.0))
    scaled = log_p / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = numpy.exp(scaled)
    return _normalize_probabilities(exp_scaled)


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


def select_calibration_config(probabilities, y_true, labels) -> dict:
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
            candidate_key = (
                candidate_log_loss,
                candidate_brier,
                abs(temperature - 1.0),
                alpha,
            )
            if best_config is None or candidate_key < best_config["selection_key"]:
                best_config = {
                    "temperature": temperature,
                    "prior_blend_alpha": alpha,
                    "class_prior": class_prior.tolist(),
                    "calibration_log_loss": candidate_log_loss,
                    "calibration_brier_score": candidate_brier,
                    "selection_key": candidate_key,
                }

    if best_config is None:
        raise RuntimeError("No calibration config was selected")
    best_config.pop("selection_key")
    return best_config


def select_draw_rule(probabilities, y_true, labels) -> dict:
    argmax_predictions = _predict_argmax(probabilities, labels)
    argmax_metrics = evaluate_predictions(y_true, argmax_predictions, probabilities, labels)

    best_config: dict[str, Any] | None = None
    for draw_threshold in DRAW_THRESHOLD_GRID:
        for draw_margin in DRAW_MARGIN_GRID:
            predictions = predict_with_draw_rule(
                probabilities,
                labels,
                draw_threshold,
                draw_margin,
            )
            metrics = evaluate_predictions(y_true, predictions, probabilities, labels)
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
                    "calibration_argmax_draw_precision": argmax_metrics[
                        "draw_precision"
                    ],
                    "calibration_argmax_draw_f1": argmax_metrics["draw_f1"],
                    "calibration_rule_accuracy": metrics["accuracy"],
                    "calibration_rule_draw_recall": metrics["draw_recall"],
                    "calibration_rule_draw_precision": metrics["draw_precision"],
                    "calibration_rule_draw_f1": metrics["draw_f1"],
                    "calibration_rule_predicted_draws": predicted_draws,
                    "rule_available": True,
                    "selection_key": candidate_key,
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
            "rule_available": False,
            "selection_key": None,
        }

    best_config.pop("selection_key", None)
    return best_config


def predict_with_draw_rule(probabilities, labels, draw_threshold, draw_margin) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    predictions = _predict_argmax(probabilities, labels)
    if draw_threshold is None or draw_margin is None:
        return predictions

    draw_index = labels.index("D")
    non_draw_indexes = [index for index, label in enumerate(labels) if label != "D"]
    draw_probabilities = probabilities[:, draw_index]
    best_non_draw_probabilities = probabilities[:, non_draw_indexes].max(axis=1)
    draw_mask = (
        (draw_probabilities >= draw_threshold)
        & ((best_non_draw_probabilities - draw_probabilities) <= draw_margin)
    )
    return [
        "D" if draw_mask[row_index] else prediction
        for row_index, prediction in enumerate(predictions)
    ]


def evaluate_predictions(y_true, y_pred, probabilities, labels) -> dict:
    if labels != ["H", "D", "A"]:
        raise ValueError(f"Label order must be ['H', 'D', 'A'], got {labels}")

    probabilities = _normalize_probabilities(probabilities)
    _assert_probability_rows(probabilities)
    y_true_encoded = _encode_labels(y_true, labels)
    y_pred_encoded = _encode_labels(y_pred, labels)
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


def aggregate_results(results) -> pandas.DataFrame:
    results_df = _coerce_results_df(results)
    leaderboard = _build_leaderboard(results_df)
    print("=== Development Leaderboard ===")
    print(_format_leaderboard(leaderboard))
    return leaderboard


def print_grid_boundary_warnings(results) -> None:
    results_df = _coerce_results_df(results)
    calibrated_rows = results_df.loc[
        results_df["mode"].isin(["calibrated", "calibrated_draw_rule"])
    ].drop_duplicates(
        subset=[
            "fold",
            "base_model_name",
            "temperature",
            "prior_blend_alpha",
            "draw_threshold",
            "draw_margin",
        ]
    )

    print("=== Grid Boundary Warnings ===")
    warnings_found = False
    for row in calibrated_rows.itertuples(index=False):
        warning_parts: list[str] = []
        if row.temperature == min(TEMPERATURE_GRID):
            warning_parts.append(f"temperature at min {row.temperature}")
        if row.temperature == max(TEMPERATURE_GRID):
            warning_parts.append(f"temperature at max {row.temperature}")
        if row.prior_blend_alpha == min(PRIOR_BLEND_GRID):
            warning_parts.append(f"prior_blend_alpha at min {row.prior_blend_alpha}")
        if row.prior_blend_alpha == max(PRIOR_BLEND_GRID):
            warning_parts.append(f"prior_blend_alpha at max {row.prior_blend_alpha}")
        if row.draw_threshold == min(DRAW_THRESHOLD_GRID):
            warning_parts.append(f"draw_threshold at min {row.draw_threshold}")
        if row.draw_threshold == max(DRAW_THRESHOLD_GRID):
            warning_parts.append(f"draw_threshold at max {row.draw_threshold}")
        if row.draw_margin == min(DRAW_MARGIN_GRID):
            warning_parts.append(f"draw_margin at min {row.draw_margin}")
        if row.draw_margin == max(DRAW_MARGIN_GRID):
            warning_parts.append(f"draw_margin at max {row.draw_margin}")

        if warning_parts:
            warnings_found = True
            print(
                f"Fold {row.fold} {row.base_model_name}: "
                + "; ".join(warning_parts)
            )

    if not warnings_found:
        print("No selected calibration or draw-rule values were on grid boundaries.")


def print_leaderboard_and_verdict(results) -> None:
    results_df = _coerce_results_df(results)
    leaderboard = _build_leaderboard(results_df)

    print("=== Probability Champion ===")
    probability_leaderboard = leaderboard.loc[
        leaderboard["model_name"].isin(PROBABILITY_CANDIDATES)
    ].sort_values(
        ["log_loss", "brier_score", "accuracy"],
        ascending=[True, True, False],
    )
    champion = probability_leaderboard.iloc[0]
    print(
        f"Metric champion: {champion['model_name']} "
        f"(log_loss={champion['log_loss']:.4f}, "
        f"brier={champion['brier_score']:.4f}, "
        f"accuracy={champion['accuracy']:.4f})"
    )

    print("=== Calibrated vs Expanding Logistic Elo Gate ===")
    for calibrated_model in ["logistic_elo_calibrated", "xgb_elo_calibrated"]:
        verdict = _calibrated_gate_verdict(
            results_df,
            calibrated_model,
            "logistic_elo_expanding",
        )
        print(
            f"{calibrated_model}: "
            f"log_loss_delta={verdict['aggregate_log_loss_delta']:+.4f}, "
            f"brier_delta={verdict['aggregate_brier_delta']:+.4f}, "
            f"max_fold_log_loss_delta="
            f"{verdict['max_fold_log_loss_delta']:+.4f}, "
            f"gate={'PASS' if verdict['passes'] else 'FAIL'}"
        )

    print("=== Draw-Rule Verdict ===")
    for family in ["logistic_elo", "xgb_elo"]:
        base_model = f"{family}_calibrated"
        draw_model = f"{family}_calibrated_draw_rule"
        verdict = _draw_rule_verdict(results_df, base_model, draw_model)
        print(
            f"{draw_model}: "
            f"draw_f1_improved_both={verdict['draw_f1_improved_both']}, "
            f"draw_recall_improved_both={verdict['draw_recall_improved_both']}, "
            f"max_accuracy_drop={verdict['max_accuracy_drop']:.4f}, "
            f"verdict={'experimental candidate' if verdict['passes'] else 'reject/keep experimental'}"
        )


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
    if LABELS != ["H", "D", "A"]:
        raise RuntimeError(f"Label order changed unexpectedly: {LABELS}")

    print("=== Tier 3 Development Leaderboard and Calibration Audit ===")
    conn = get_db_connection()
    validate_historical_match_integrity(conn)

    before_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts before", before_counts)

    df = load_feature_data(conn)
    validate_no_final_holdout_loaded(df)
    feature_sets = get_feature_sets(df)
    model_specs = build_model_specs()

    result_rows: list[dict[str, Any]] = []
    for fold_config in EXPANDING_FOLDS:
        result_rows.extend(
            run_expanding_window_fold(df, fold_config, model_specs, feature_sets)
        )
    for fold_config in CALIBRATED_FOLDS:
        result_rows.extend(
            run_calibrated_fold(df, fold_config, model_specs, feature_sets)
        )

    results_df = pandas.DataFrame(result_rows)
    aggregate_results(results_df)
    print_grid_boundary_warnings(results_df)
    print_leaderboard_and_verdict(results_df)

    after_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts after", after_counts)
    assert_watched_counts_unchanged(before_counts, after_counts)

    print("PASS: probability rows were finite and summed to 1 within tolerance.")
    print("PASS: label order was exactly ['H', 'D', 'A'].")
    print("PASS: calibration settings were selected only on calibration seasons.")
    print("PASS: validation seasons were never used for tuning.")
    print(f"PASS: {FINAL_TEST_SEASON} was not loaded, tuned, calibrated, evaluated, or reported for model metrics.")
    print("No database writes occurred.")
    print("No model artifacts were saved.")
    print("No Streamlit, Tier 2 artifact, H2H, Poisson-feature, betting odds, manager, sentiment, injury, style, deployment, or app work occurred.")


def _coerce_results_df(results) -> pandas.DataFrame:
    if isinstance(results, pandas.DataFrame):
        return results.copy()
    return pandas.DataFrame(results)


def _build_leaderboard(results_df: pandas.DataFrame) -> pandas.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
    ]
    for model_name, model_df in results_df.groupby("model_name", sort=False):
        predicted_totals = _sum_distribution_dicts(
            model_df["predicted_class_distribution"]
        )
        mean_probs = _mean_probability_dicts(model_df["mean_predicted_probability"])
        row = {
            "model_name": model_name,
            "folds": int(model_df["fold"].nunique()),
            "accuracy": float(model_df["accuracy"].mean()),
            "log_loss": float(model_df["log_loss"].mean()),
            "brier_score": float(model_df["brier_score"].mean()),
            "draw_recall": float(model_df["draw_recall"].mean()),
            "draw_precision": float(model_df["draw_precision"].mean()),
            "draw_f1": float(model_df["draw_f1"].mean()),
            "pred_H": predicted_totals["H"],
            "pred_D": predicted_totals["D"],
            "pred_A": predicted_totals["A"],
            "mean_prob_H": mean_probs["H"],
            "mean_prob_D": mean_probs["D"],
            "mean_prob_A": mean_probs["A"],
        }
        for metric in metric_columns:
            std_value = model_df[metric].std(ddof=1)
            row[f"{metric}_std"] = 0.0 if pandas.isna(std_value) else float(std_value)
        rows.append(row)

    leaderboard = pandas.DataFrame(rows)
    return leaderboard.sort_values(
        ["log_loss", "brier_score", "accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _format_leaderboard(leaderboard: pandas.DataFrame) -> str:
    columns = [
        "model_name",
        "folds",
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "draw_precision",
        "draw_f1",
        "pred_H",
        "pred_D",
        "pred_A",
        "mean_prob_H",
        "mean_prob_D",
        "mean_prob_A",
    ]
    formatters = {
        column: "{:.4f}".format
        for column in [
            "accuracy",
            "log_loss",
            "brier_score",
            "draw_recall",
            "draw_precision",
            "draw_f1",
            "mean_prob_H",
            "mean_prob_D",
            "mean_prob_A",
        ]
    }
    return leaderboard[columns].to_string(index=False, formatters=formatters)


def _validate_expanding_fold(
    fold: int,
    train_df: pandas.DataFrame,
    validation_df: pandas.DataFrame,
    train_seasons: list[str],
    validation_season: str,
) -> None:
    if FINAL_TEST_SEASON in set(train_df["season_id"]) | set(validation_df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} entered expanding fold {fold}")
    if len(train_df) != len(train_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(train_seasons) * EXPECTED_SEASON_ROWS} "
            f"train rows, found {len(train_df)}"
        )
    if len(validation_df) != EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {EXPECTED_SEASON_ROWS} validation rows, "
            f"found {len(validation_df)}"
        )
    if train_df["match_date"].max() >= validation_df["match_date"].min():
        raise ValueError(f"Fold {fold} expanding date leakage")
    print(
        f"Expanding fold {fold} date check: "
        f"train max {train_df['match_date'].max().date()} < "
        f"validation min {validation_df['match_date'].min().date()}"
    )
    print(
        f"Expanding fold {fold} seasons verified: "
        f"train={', '.join(train_seasons)}, validation={validation_season}"
    )


def _validate_calibrated_fold(
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
        raise ValueError(f"{FINAL_TEST_SEASON} entered calibrated fold {fold}")
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
        f"Calibrated fold {fold} date check: "
        f"fit max {fit_df['match_date'].max().date()} < "
        f"calibration min {calibration_df['match_date'].min().date()}, "
        f"calibration max {calibration_df['match_date'].max().date()} < "
        f"validation min {validation_df['match_date'].min().date()}"
    )
    print(
        f"Calibrated fold {fold} seasons verified: "
        f"fit={', '.join(fit_seasons)}, calibration={calibration_season}, "
        f"validation={validation_season}"
    )


def _predict_probabilities(model, X, labels) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    aligned = _normalize_probabilities(aligned)
    _assert_probability_rows(aligned)
    return aligned


def _predict_argmax(probabilities, labels) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    indexes = numpy.argmax(probabilities, axis=1)
    return [labels[int(index)] for index in indexes]


def _apply_calibration_config(probabilities, calibration_config: dict) -> numpy.ndarray:
    temperature_scaled = apply_temperature(
        probabilities,
        calibration_config["temperature"],
    )
    return apply_prior_blend(
        temperature_scaled,
        calibration_config["class_prior"],
        calibration_config["prior_blend_alpha"],
    )


def _apply_draw_rule_or_argmax(probabilities, labels, draw_rule: dict) -> list[str]:
    if not draw_rule["rule_available"]:
        return _predict_argmax(probabilities, labels)
    return predict_with_draw_rule(
        probabilities,
        labels,
        draw_rule["draw_threshold"],
        draw_rule["draw_margin"],
    )


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("NaN or infinite probabilities found")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Probability row with non-positive sum found")
    normalized = probabilities / row_sums
    _assert_probability_rows(normalized)
    return normalized


def _assert_probability_rows(probabilities: numpy.ndarray) -> None:
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("NaN or infinite probabilities found")
    if not numpy.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Probability rows do not sum to 1")
    if numpy.any(probabilities < -1e-12):
        raise ValueError("Negative probabilities found")


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


def _sum_distribution_dicts(series: pandas.Series) -> dict[str, int]:
    totals = {label: 0 for label in LABELS}
    for distribution in series:
        for label in LABELS:
            totals[label] += int(distribution[label])
    return totals


def _mean_probability_dicts(series: pandas.Series) -> dict[str, float]:
    totals = {label: 0.0 for label in LABELS}
    count = len(series)
    for probabilities in series:
        for label in LABELS:
            totals[label] += float(probabilities[label])
    return {label: totals[label] / count for label in LABELS}


def _calibrated_gate_verdict(
    results_df: pandas.DataFrame,
    calibrated_model: str,
    baseline_model: str,
) -> dict[str, Any]:
    metric_columns = ["log_loss", "brier_score"]
    baseline_agg = results_df.loc[
        results_df["model_name"] == baseline_model,
        metric_columns,
    ].mean()
    calibrated_agg = results_df.loc[
        results_df["model_name"] == calibrated_model,
        metric_columns,
    ].mean()

    fold_deltas: list[float] = []
    for fold in sorted(results_df["fold"].unique()):
        baseline_row = _single_result_row(results_df, fold, baseline_model)
        calibrated_row = _single_result_row(results_df, fold, calibrated_model)
        fold_deltas.append(float(calibrated_row["log_loss"] - baseline_row["log_loss"]))

    log_loss_delta = float(calibrated_agg["log_loss"] - baseline_agg["log_loss"])
    brier_delta = float(calibrated_agg["brier_score"] - baseline_agg["brier_score"])
    max_fold_log_loss_delta = max(fold_deltas)
    return {
        "aggregate_log_loss_delta": log_loss_delta,
        "aggregate_brier_delta": brier_delta,
        "max_fold_log_loss_delta": max_fold_log_loss_delta,
        "passes": (
            log_loss_delta <= -0.003
            and brier_delta <= -0.003
            and max_fold_log_loss_delta <= 0.005
        ),
    }


def _draw_rule_verdict(
    results_df: pandas.DataFrame,
    calibrated_model: str,
    draw_rule_model: str,
) -> dict[str, Any]:
    fold_deltas: list[dict[str, float]] = []
    for fold in sorted(results_df["fold"].unique()):
        calibrated_row = _single_result_row(results_df, fold, calibrated_model)
        draw_row = _single_result_row(results_df, fold, draw_rule_model)
        fold_deltas.append(
            {
                "accuracy": float(draw_row["accuracy"] - calibrated_row["accuracy"]),
                "draw_recall": float(
                    draw_row["draw_recall"] - calibrated_row["draw_recall"]
                ),
                "draw_f1": float(draw_row["draw_f1"] - calibrated_row["draw_f1"]),
                "log_loss": float(draw_row["log_loss"] - calibrated_row["log_loss"]),
                "brier_score": float(
                    draw_row["brier_score"] - calibrated_row["brier_score"]
                ),
            }
        )
    max_accuracy_drop = max(max(0.0, -delta["accuracy"]) for delta in fold_deltas)
    return {
        "draw_f1_improved_both": all(delta["draw_f1"] > 0 for delta in fold_deltas),
        "draw_recall_improved_both": all(
            delta["draw_recall"] > 0 for delta in fold_deltas
        ),
        "max_accuracy_drop": max_accuracy_drop,
        "probability_metrics_unchanged": all(
            abs(delta["log_loss"]) < 1e-12 and abs(delta["brier_score"]) < 1e-12
            for delta in fold_deltas
        ),
        "passes": (
            all(delta["draw_f1"] > 0 for delta in fold_deltas)
            and all(delta["draw_recall"] > 0 for delta in fold_deltas)
            and max_accuracy_drop <= 0.02
            and all(
                abs(delta["log_loss"]) < 1e-12
                and abs(delta["brier_score"]) < 1e-12
                for delta in fold_deltas
            )
        ),
    }


def _single_result_row(
    results_df: pandas.DataFrame,
    fold: int,
    model_name: str,
) -> pandas.Series:
    rows = results_df.loc[
        (results_df["fold"] == fold) & (results_df["model_name"] == model_name)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for fold={fold}, model={model_name}")
    return rows.iloc[0]


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


def _print_confusion_counts(confusion_counts: dict[str, dict[str, int]]) -> None:
    print("Confusion counts (rows=true, columns=pred):")
    print("true\\pred    H    D    A")
    for true_label in LABELS:
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
