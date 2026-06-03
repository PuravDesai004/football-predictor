from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    recall_score,
)
from sqlalchemy import text

from data_pipeline import get_engine
from tier3_validation import (
    build_walk_forward_season_splits,
    get_ordered_seasons,
    split_development_and_final_test,
    validate_final_test_holdout,
    validate_historical_match_integrity,
    validate_walk_forward_splits,
)


CLASS_LABELS = ["H", "D", "A"]
FINAL_TEST_SEASON = "2025-26"
DEVELOPMENT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
MAX_GOALS = 6
MIN_LAMBDA = 0.05
MAX_LAMBDA = 5.0
PROBABILITY_TOLERANCE = 1e-9

ID_TARGET_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]

POISSON_FEATURE_COLUMNS = [
    "home_xg_home_last5",
    "home_xga_home_last5",
    "away_xg_away_last5",
    "away_xga_away_last5",
    "home_xg_overall_last5",
    "home_xga_overall_last5",
    "away_xg_overall_last5",
    "away_xga_overall_last5",
    "home_home_matches_last5",
    "away_away_matches_last5",
    "home_overall_matches_last5",
    "away_overall_matches_last5",
    "home_elo_before",
    "away_elo_before",
    "elo_diff_home_adjusted",
    "expected_home_score",
    "expected_away_score",
]

HOME_LAMBDA_COLUMNS = [
    "home_xg_home_last5",
    "home_xg_overall_last5",
    "away_xga_away_last5",
    "away_xga_overall_last5",
]

AWAY_LAMBDA_COLUMNS = [
    "away_xg_away_last5",
    "away_xg_overall_last5",
    "home_xga_home_last5",
    "home_xga_overall_last5",
]

HOME_XG_COLUMNS = ["home_xg_home_last5", "home_xg_overall_last5"]
AWAY_XG_COLUMNS = ["away_xg_away_last5", "away_xg_overall_last5"]
XGA_COLUMNS = [
    "home_xga_home_last5",
    "away_xga_away_last5",
    "home_xga_overall_last5",
    "away_xga_overall_last5",
]
XG_INPUT_COLUMNS = sorted(set(HOME_LAMBDA_COLUMNS + AWAY_LAMBDA_COLUMNS))

WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "players",
    "teams",
    "fixtures",
    "gameweeks",
    "understat_xg",
    "understat_team_history",
    "player_gameweek_history",
    "player_gameweek_features",
]

EXISTING_DEV_BASELINES = {
    "logistic_elo": {
        "accuracy": 0.5579,
        "log_loss": 0.9705,
        "brier_score": 0.5730,
        "draw_recall": 0.0534,
    },
    "xgb_elo": {
        "accuracy": 0.5382,
        "log_loss": 0.9984,
        "brier_score": 0.5901,
        "draw_recall": 0.0764,
    },
}

POISSON_AGGREGATE_METRICS: dict[str, float] | None = None


def _table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
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
        )


def _count_table_rows(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _verify_source_table_columns(engine) -> None:
    if not _table_exists(engine, "match_features_v3_elo"):
        raise RuntimeError("match_features_v3_elo table does not exist")

    with engine.connect() as conn:
        existing_columns = set(
            conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = CURRENT_SCHEMA()
                        AND table_name = 'match_features_v3_elo'
                    """
                )
            ).scalars()
        )

    required_columns = set(ID_TARGET_COLUMNS + POISSON_FEATURE_COLUMNS)
    missing_columns = sorted(required_columns - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_elo is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    print("PASS: match_features_v3_elo source schema verified.")


def load_poisson_dataset(engine, development_seasons: list[str]) -> pd.DataFrame:
    _verify_source_table_columns(engine)
    selected_columns = ID_TARGET_COLUMNS + POISSON_FEATURE_COLUMNS
    query = text(
        f"""
        SELECT {", ".join(selected_columns)}
        FROM match_features_v3_elo
        WHERE season_id = ANY(:development_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    dataset_df = pd.read_sql(
        query,
        engine,
        params={"development_seasons": development_seasons},
    )
    dataset_df["match_date"] = pd.to_datetime(dataset_df["match_date"])

    expected_count = 380 * len(development_seasons)
    errors: list[str] = []
    if len(dataset_df) != expected_count:
        errors.append(
            f"expected {expected_count} development rows, found {len(dataset_df)}"
        )
    if FINAL_TEST_SEASON in set(dataset_df["season_id"]):
        errors.append(f"{FINAL_TEST_SEASON} rows were loaded unexpectedly")
    if dataset_df["match_id"].duplicated().any():
        duplicate_count = int(dataset_df["match_id"].duplicated().sum())
        errors.append(f"duplicate match_id values found: {duplicate_count}")
    bad_results = sorted(set(dataset_df["result"]) - set(CLASS_LABELS))
    if bad_results:
        errors.append(f"unexpected result labels found: {bad_results}")

    season_counts = dataset_df.groupby("season_id").size().sort_index()
    print("=== Poisson Development Dataset ===")
    for season_id, row_count in season_counts.items():
        print(f"{season_id}: {int(row_count)} rows")

    if errors:
        raise ValueError("Poisson dataset validation failed: " + "; ".join(errors))

    print(
        "Poisson dataset validation passed: "
        f"{len(dataset_df)} development rows loaded from match_features_v3_elo."
    )
    print(f"{FINAL_TEST_SEASON} was not loaded for Poisson evaluation.")
    return dataset_df


def _mean_from_columns(df: pd.DataFrame, columns: list[str]) -> float | None:
    numeric_values = df[columns].apply(pd.to_numeric, errors="coerce")
    values = numeric_values.to_numpy(dtype=float)
    non_null_values = values[~np.isnan(values)]
    if non_null_values.size == 0:
        return None
    return float(np.mean(non_null_values))


def _fallback_for_column(column: str, params: dict[str, Any]) -> float:
    if "xga" in column:
        return float(params["league_avg_xga"])
    if column.startswith("home_xg"):
        return float(params["league_avg_home_xg"])
    if column.startswith("away_xg"):
        return float(params["league_avg_away_xg"])
    return float(
        (params["league_avg_home_goals"] + params["league_avg_away_goals"]) / 2.0
    )


def fit_fold_poisson_parameters(train_df: pd.DataFrame) -> dict[str, Any]:
    league_avg_home_goals = float(train_df["home_goals"].mean())
    league_avg_away_goals = float(train_df["away_goals"].mean())
    league_avg_home_xg = _mean_from_columns(train_df, HOME_XG_COLUMNS)
    league_avg_away_xg = _mean_from_columns(train_df, AWAY_XG_COLUMNS)
    league_avg_xga = _mean_from_columns(train_df, XGA_COLUMNS)

    if league_avg_home_xg is None:
        league_avg_home_xg = league_avg_home_goals
    if league_avg_away_xg is None:
        league_avg_away_xg = league_avg_away_goals
    if league_avg_xga is None:
        league_avg_xga = (league_avg_home_goals + league_avg_away_goals) / 2.0

    params: dict[str, Any] = {
        "league_avg_home_goals": league_avg_home_goals,
        "league_avg_away_goals": league_avg_away_goals,
        "league_avg_home_xg": float(league_avg_home_xg),
        "league_avg_away_xg": float(league_avg_away_xg),
        "league_avg_xga": float(league_avg_xga),
        "xg_medians": {},
    }

    median_values = train_df[XG_INPUT_COLUMNS].median(skipna=True).to_dict()
    for column in XG_INPUT_COLUMNS:
        median_value = median_values.get(column)
        if pd.isna(median_value):
            median_value = _fallback_for_column(column, params)
        params["xg_medians"][column] = float(median_value)

    return params


def _value_or_fallback(row: pd.Series, column: str, params: dict[str, Any]) -> float:
    value = row[column]
    if pd.isna(value):
        return float(params["xg_medians"][column])
    return float(value)


def estimate_lambdas(row: pd.Series, params: dict[str, Any]) -> tuple[float, float]:
    home_base = float(
        np.mean([_value_or_fallback(row, column, params) for column in HOME_LAMBDA_COLUMNS])
    )
    away_base = float(
        np.mean([_value_or_fallback(row, column, params) for column in AWAY_LAMBDA_COLUMNS])
    )

    elo_diff = 0.0 if pd.isna(row["elo_diff_home_adjusted"]) else float(row["elo_diff_home_adjusted"])
    elo_scale = elo_diff / 400.0
    home_lambda = home_base * math.exp(0.10 * elo_scale)
    away_lambda = away_base * math.exp(-0.10 * elo_scale)

    home_lambda = float(np.clip(home_lambda, MIN_LAMBDA, MAX_LAMBDA))
    away_lambda = float(np.clip(away_lambda, MIN_LAMBDA, MAX_LAMBDA))
    return home_lambda, away_lambda


def poisson_pmf(lambda_value: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    clipped_lambda = float(np.clip(lambda_value, MIN_LAMBDA, MAX_LAMBDA))
    probabilities = np.array(
        [
            math.exp(-clipped_lambda)
            * (clipped_lambda**goals)
            / math.factorial(goals)
            for goals in range(max_goals + 1)
        ],
        dtype=float,
    )
    total_probability = probabilities.sum()
    if total_probability <= 0:
        raise ValueError("Poisson PMF has non-positive total probability")
    return probabilities / total_probability


def scoreline_matrix(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    home_probabilities = poisson_pmf(home_lambda, max_goals=max_goals)
    away_probabilities = poisson_pmf(away_lambda, max_goals=max_goals)
    matrix = np.outer(home_probabilities, away_probabilities)
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    home_win_proba = float(np.tril(matrix, k=-1).sum())
    draw_proba = float(np.trace(matrix))
    away_win_proba = float(np.triu(matrix, k=1).sum())
    total = home_win_proba + draw_proba + away_win_proba
    return {
        "home_win_proba": home_win_proba / total,
        "draw_proba": draw_proba / total,
        "away_win_proba": away_win_proba / total,
    }


def _validate_probability_sanity(
    match_id: int,
    home_lambda: float,
    away_lambda: float,
    matrix: np.ndarray,
    probabilities: dict[str, float],
) -> None:
    errors: list[str] = []
    if not MIN_LAMBDA <= home_lambda <= MAX_LAMBDA:
        errors.append(f"home_lambda {home_lambda:.6f} outside allowed bounds")
    if not MIN_LAMBDA <= away_lambda <= MAX_LAMBDA:
        errors.append(f"away_lambda {away_lambda:.6f} outside allowed bounds")

    matrix_sum = float(matrix.sum())
    if not np.isclose(matrix_sum, 1.0, atol=PROBABILITY_TOLERANCE):
        errors.append(f"scoreline matrix sums to {matrix_sum:.12f}")

    probability_values = np.array(list(probabilities.values()), dtype=float)
    if np.any(probability_values < -PROBABILITY_TOLERANCE) or np.any(
        probability_values > 1.0 + PROBABILITY_TOLERANCE
    ):
        errors.append(f"outcome probabilities outside [0, 1]: {probabilities}")

    probability_sum = float(probability_values.sum())
    if not np.isclose(probability_sum, 1.0, atol=PROBABILITY_TOLERANCE):
        errors.append(f"outcome probabilities sum to {probability_sum:.12f}")

    if errors:
        raise ValueError(
            f"Probability sanity check failed for match_id {match_id}: "
            + "; ".join(errors)
        )


def most_likely_scorelines(matrix: np.ndarray, top_n: int = 5) -> list[dict[str, float]]:
    flat_indices = np.argsort(matrix.ravel())[::-1][:top_n]
    _, away_goal_count = matrix.shape
    scorelines: list[dict[str, float]] = []
    for flat_index in flat_indices:
        home_goals = int(flat_index // away_goal_count)
        away_goals = int(flat_index % away_goal_count)
        scorelines.append(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability": float(matrix[home_goals, away_goals]),
            }
        )
    return scorelines


def predict_poisson_for_fold(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    params = fit_fold_poisson_parameters(train_df)
    prediction_rows: list[dict[str, Any]] = []

    sorted_valid_df = valid_df.sort_values(["match_date", "match_id"]).reset_index(
        drop=True
    )
    for row in sorted_valid_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        home_lambda, away_lambda = estimate_lambdas(row_series, params)
        matrix = scoreline_matrix(home_lambda, away_lambda)
        probabilities = outcome_probabilities(matrix)
        _validate_probability_sanity(
            int(row.match_id),
            home_lambda,
            away_lambda,
            matrix,
            probabilities,
        )
        top_scorelines = most_likely_scorelines(matrix, top_n=5)

        class_probabilities = {
            "H": probabilities["home_win_proba"],
            "D": probabilities["draw_proba"],
            "A": probabilities["away_win_proba"],
        }
        predicted_result = max(CLASS_LABELS, key=lambda label: class_probabilities[label])
        top_scoreline = top_scorelines[0]

        prediction_rows.append(
            {
                "match_id": int(row.match_id),
                "season_id": row.season_id,
                "match_date": row.match_date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "actual_home_goals": int(row.home_goals),
                "actual_away_goals": int(row.away_goals),
                "result": row.result,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "home_win_proba": probabilities["home_win_proba"],
                "draw_proba": probabilities["draw_proba"],
                "away_win_proba": probabilities["away_win_proba"],
                "predicted_result": predicted_result,
                "predicted_home_goals": int(top_scoreline["home_goals"]),
                "predicted_away_goals": int(top_scoreline["away_goals"]),
                "top_scorelines": top_scorelines,
            }
        )

    return pd.DataFrame(prediction_rows)


def validate_prediction_frame_sanity(pred_df: pd.DataFrame) -> None:
    probability_columns = ["home_win_proba", "draw_proba", "away_win_proba"]
    probability_values = pred_df[probability_columns].to_numpy(dtype=float)
    row_sums = probability_values.sum(axis=1)

    errors: list[str] = []
    bad_sum_count = int(
        (~np.isclose(row_sums, 1.0, atol=PROBABILITY_TOLERANCE)).sum()
    )
    if bad_sum_count:
        errors.append(f"{bad_sum_count} prediction row(s) have probabilities not summing to 1")

    outside_probability_count = int(
        (
            (probability_values < -PROBABILITY_TOLERANCE)
            | (probability_values > 1.0 + PROBABILITY_TOLERANCE)
        ).sum()
    )
    if outside_probability_count:
        errors.append(
            f"{outside_probability_count} probability value(s) are outside [0, 1]"
        )

    bad_lambda_count = int(
        (
            ~pred_df["home_lambda"].between(MIN_LAMBDA, MAX_LAMBDA)
            | ~pred_df["away_lambda"].between(MIN_LAMBDA, MAX_LAMBDA)
        ).sum()
    )
    if bad_lambda_count:
        errors.append(f"{bad_lambda_count} row(s) have lambda values outside bounds")

    if errors:
        raise ValueError("Prediction sanity validation failed: " + "; ".join(errors))

    print("Probability sanity checks passed for prediction frame.")


def multiclass_brier_score(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> float:
    y_one_hot = np.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[np.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def _confusion_counts(
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, int]]:
    encoded_labels = list(range(len(labels)))
    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=encoded_labels,
    )
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(labels)
        }
        for true_index, true_label in enumerate(labels)
    }


def evaluate_poisson_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    y_true_encoded = pred_df["result"].map(label_to_index).to_numpy(dtype=int)
    y_pred_encoded = pred_df["predicted_result"].map(label_to_index).to_numpy(dtype=int)
    y_proba = pred_df[
        ["home_win_proba", "draw_proba", "away_win_proba"]
    ].to_numpy(dtype=float)
    y_proba = np.clip(y_proba, 1e-15, 1.0)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    draw_index = label_to_index["D"]
    actual_total_goals = pred_df["actual_home_goals"] + pred_df["actual_away_goals"]
    predicted_total_goals = (
        pred_df["predicted_home_goals"] + pred_df["predicted_away_goals"]
    )

    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(
            log_loss(
                y_true_encoded,
                y_proba,
                labels=list(range(len(CLASS_LABELS))),
            )
        ),
        "brier_score": multiclass_brier_score(
            y_true_encoded,
            y_proba,
            CLASS_LABELS,
        ),
        "draw_recall": float(
            recall_score(
                y_true_encoded,
                y_pred_encoded,
                labels=[draw_index],
                average=None,
                zero_division=0,
            )[0]
        ),
        "home_goals_mae": float(
            mean_absolute_error(
                pred_df["actual_home_goals"],
                pred_df["predicted_home_goals"],
            )
        ),
        "away_goals_mae": float(
            mean_absolute_error(
                pred_df["actual_away_goals"],
                pred_df["predicted_away_goals"],
            )
        ),
        "total_goals_mae": float(
            mean_absolute_error(actual_total_goals, predicted_total_goals)
        ),
        "exact_score_accuracy": float(
            (
                (pred_df["actual_home_goals"] == pred_df["predicted_home_goals"])
                & (pred_df["actual_away_goals"] == pred_df["predicted_away_goals"])
            ).mean()
        ),
        "confusion_counts": _confusion_counts(
            y_true_encoded,
            y_pred_encoded,
            CLASS_LABELS,
        ),
    }


def draw_probability_diagnostics(pred_df: pd.DataFrame) -> dict[str, float | int]:
    probability_columns = ["home_win_proba", "draw_proba", "away_win_proba"]
    proba_df = pred_df[probability_columns]
    top_probability_column = proba_df.idxmax(axis=1)

    return {
        "mean_draw_proba": float(pred_df["draw_proba"].mean()),
        "median_draw_proba": float(pred_df["draw_proba"].median()),
        "max_draw_proba": float(pred_df["draw_proba"].max()),
        "draw_highest_count": int((top_probability_column == "draw_proba").sum()),
        "draw_proba_ge_025_count": int((pred_df["draw_proba"] >= 0.25).sum()),
        "draw_proba_ge_030_count": int((pred_df["draw_proba"] >= 0.30).sum()),
        "actual_draw_rate": float((pred_df["result"] == "D").mean()),
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


def _print_fold_metrics(metrics: dict[str, Any]) -> None:
    print(
        "Poisson metrics: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"brier_score={metrics['brier_score']:.4f}, "
        f"draw_recall={metrics['draw_recall']:.4f}, "
        f"exact_score_accuracy={metrics['exact_score_accuracy']:.4f}, "
        f"home_goals_mae={metrics['home_goals_mae']:.4f}, "
        f"away_goals_mae={metrics['away_goals_mae']:.4f}, "
        f"total_goals_mae={metrics['total_goals_mae']:.4f}"
    )
    _print_confusion_counts(metrics["confusion_counts"])


def _print_draw_probability_diagnostics(
    diagnostics: dict[str, float | int],
    label: str,
) -> None:
    print(f"=== Draw Probability Diagnostics: {label} ===")
    print(f"- mean_draw_proba: {diagnostics['mean_draw_proba']:.4f}")
    print(f"- median_draw_proba: {diagnostics['median_draw_proba']:.4f}")
    print(f"- max_draw_proba: {diagnostics['max_draw_proba']:.4f}")
    print(f"- draw_highest_count: {diagnostics['draw_highest_count']}")
    print(f"- draw_proba_ge_025_count: {diagnostics['draw_proba_ge_025_count']}")
    print(f"- draw_proba_ge_030_count: {diagnostics['draw_proba_ge_030_count']}")
    print(f"- actual_draw_rate: {diagnostics['actual_draw_rate']:.4f}")


def _format_scorelines(scorelines: list[dict[str, float]]) -> str:
    return "; ".join(
        f"{scoreline['home_goals']}-{scoreline['away_goals']} "
        f"{scoreline['probability']:.3f}"
        for scoreline in scorelines
    )


def _print_sample_predictions(pred_df: pd.DataFrame, top_n: int = 5) -> None:
    print("Deterministic sample scoreline predictions:")
    sample_df = pred_df.sort_values(["match_date", "match_id"]).head(top_n)
    for row in sample_df.itertuples(index=False):
        match_date = pd.Timestamp(row.match_date).date()
        print(
            f"- match_id={row.match_id}, date={match_date}, "
            f"{row.home_team} vs {row.away_team}: "
            f"actual {row.actual_home_goals}-{row.actual_away_goals} ({row.result}), "
            f"predicted {row.predicted_home_goals}-{row.predicted_away_goals} "
            f"({row.predicted_result}), "
            f"home_lambda={row.home_lambda:.4f}, "
            f"away_lambda={row.away_lambda:.4f}, "
            f"home_win_proba={row.home_win_proba:.4f}, "
            f"draw_proba={row.draw_proba:.4f}, "
            f"away_win_proba={row.away_win_proba:.4f}, "
            f"top scorelines: {_format_scorelines(row.top_scorelines)}"
        )


def _aggregate_poisson_metrics(results_df: pd.DataFrame) -> dict[str, float]:
    metric_columns = [
        "accuracy",
        "log_loss",
        "brier_score",
        "draw_recall",
        "exact_score_accuracy",
        "home_goals_mae",
        "away_goals_mae",
        "total_goals_mae",
    ]
    aggregate_metrics: dict[str, float] = {}
    print("=== Poisson Aggregate Walk-Forward Metrics ===")
    for metric in metric_columns:
        mean_value = float(results_df[metric].mean())
        std_value = results_df[metric].std(ddof=1)
        if pd.isna(std_value):
            std_value = 0.0
        aggregate_metrics[metric] = mean_value
        aggregate_metrics[f"{metric}_std"] = float(std_value)
        print(f"- {metric}: mean={mean_value:.4f}, std={float(std_value):.4f}")
    return aggregate_metrics


def run_walk_forward_poisson(engine) -> pd.DataFrame:
    global POISSON_AGGREGATE_METRICS

    validate_historical_match_integrity(engine)
    ordered_seasons = get_ordered_seasons(engine)
    print(f"Ordered seasons: {', '.join(ordered_seasons)}")

    split_config = split_development_and_final_test(
        ordered_seasons,
        final_test_season=FINAL_TEST_SEASON,
    )
    development_seasons = split_config["development_seasons"]
    final_test_season = split_config["final_test_season"]
    if development_seasons != DEVELOPMENT_SEASONS:
        raise ValueError(
            "Unexpected development seasons. "
            f"Expected {DEVELOPMENT_SEASONS}, found {development_seasons}"
        )
    if final_test_season != FINAL_TEST_SEASON:
        raise ValueError(
            f"Expected final test season {FINAL_TEST_SEASON}, found {final_test_season}"
        )

    print(f"Development seasons: {', '.join(development_seasons)}")
    print(f"Reserved final test season: {final_test_season}")

    splits = build_walk_forward_season_splits(development_seasons)
    validate_walk_forward_splits(engine, splits)
    validate_final_test_holdout(engine, development_seasons, final_test_season)

    poisson_df = load_poisson_dataset(engine, development_seasons)
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split in splits:
        fold = split["fold"]
        train_seasons = split["train_seasons"]
        validation_seasons = split["validation_seasons"]
        train_df = poisson_df.loc[poisson_df["season_id"].isin(train_seasons)].copy()
        valid_df = poisson_df.loc[
            poisson_df["season_id"].isin(validation_seasons)
        ].copy()

        print(f"=== Poisson Fold {fold} ===")
        print(f"Train seasons: {', '.join(train_seasons)}")
        print(f"Validation seasons: {', '.join(validation_seasons)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(valid_df)}")

        if FINAL_TEST_SEASON in set(train_df["season_id"]) | set(valid_df["season_id"]):
            raise RuntimeError(f"{FINAL_TEST_SEASON} entered a Poisson CV fold")

        pred_df = predict_poisson_for_fold(train_df, valid_df)
        validate_prediction_frame_sanity(pred_df)
        metrics = evaluate_poisson_predictions(pred_df)
        draw_diagnostics = draw_probability_diagnostics(pred_df)
        _print_fold_metrics(metrics)
        _print_draw_probability_diagnostics(draw_diagnostics, f"Fold {fold}")
        _print_sample_predictions(pred_df, top_n=5)
        prediction_frames.append(pred_df.assign(fold=fold))

        result_rows.append(
            {
                "fold": fold,
                "train_seasons": ", ".join(train_seasons),
                "validation_seasons": ", ".join(validation_seasons),
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "draw_recall": metrics["draw_recall"],
                "exact_score_accuracy": metrics["exact_score_accuracy"],
                "home_goals_mae": metrics["home_goals_mae"],
                "away_goals_mae": metrics["away_goals_mae"],
                "total_goals_mae": metrics["total_goals_mae"],
                "mean_draw_proba": draw_diagnostics["mean_draw_proba"],
                "median_draw_proba": draw_diagnostics["median_draw_proba"],
                "max_draw_proba": draw_diagnostics["max_draw_proba"],
                "draw_highest_count": draw_diagnostics["draw_highest_count"],
                "draw_proba_ge_025_count": draw_diagnostics[
                    "draw_proba_ge_025_count"
                ],
                "draw_proba_ge_030_count": draw_diagnostics[
                    "draw_proba_ge_030_count"
                ],
                "actual_draw_rate": draw_diagnostics["actual_draw_rate"],
                "confusion_counts": metrics["confusion_counts"],
            }
        )

    results_df = pd.DataFrame(result_rows)
    POISSON_AGGREGATE_METRICS = _aggregate_poisson_metrics(results_df)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    aggregate_draw_diagnostics = draw_probability_diagnostics(all_predictions_df)
    _print_draw_probability_diagnostics(
        aggregate_draw_diagnostics,
        "Aggregate validation predictions",
    )
    print(f"{FINAL_TEST_SEASON} was not evaluated and remains reserved as final test.")
    return results_df


def compare_poisson_to_existing_dev_baselines() -> None:
    if POISSON_AGGREGATE_METRICS is None:
        raise RuntimeError("Poisson aggregate metrics are not available yet")

    print("=== Poisson vs Existing Development Baselines ===")
    print(
        "Poisson aggregate: "
        f"accuracy={POISSON_AGGREGATE_METRICS['accuracy']:.4f}, "
        f"log_loss={POISSON_AGGREGATE_METRICS['log_loss']:.4f}, "
        f"brier_score={POISSON_AGGREGATE_METRICS['brier_score']:.4f}, "
        f"draw_recall={POISSON_AGGREGATE_METRICS['draw_recall']:.4f}, "
        f"exact_score_accuracy={POISSON_AGGREGATE_METRICS['exact_score_accuracy']:.4f}, "
        f"home_goals_mae={POISSON_AGGREGATE_METRICS['home_goals_mae']:.4f}, "
        f"away_goals_mae={POISSON_AGGREGATE_METRICS['away_goals_mae']:.4f}, "
        f"total_goals_mae={POISSON_AGGREGATE_METRICS['total_goals_mae']:.4f}"
    )
    for model_name, baseline_metrics in EXISTING_DEV_BASELINES.items():
        print(
            f"{model_name}: "
            f"accuracy={baseline_metrics['accuracy']:.4f}, "
            f"log_loss={baseline_metrics['log_loss']:.4f}, "
            f"brier_score={baseline_metrics['brier_score']:.4f}, "
            f"draw_recall={baseline_metrics['draw_recall']:.4f}"
        )
        print(
            f"- deltas vs {model_name}: "
            f"accuracy={POISSON_AGGREGATE_METRICS['accuracy'] - baseline_metrics['accuracy']:+.4f}, "
            f"log_loss={POISSON_AGGREGATE_METRICS['log_loss'] - baseline_metrics['log_loss']:+.4f}, "
            f"brier_score={POISSON_AGGREGATE_METRICS['brier_score'] - baseline_metrics['brier_score']:+.4f}, "
            f"draw_recall={POISSON_AGGREGATE_METRICS['draw_recall'] - baseline_metrics['draw_recall']:+.4f}"
        )

    best_probability_baseline = min(
        EXISTING_DEV_BASELINES.values(),
        key=lambda metrics: (metrics["log_loss"], metrics["brier_score"]),
    )
    probability_metrics_improved = (
        POISSON_AGGREGATE_METRICS["log_loss"] < best_probability_baseline["log_loss"]
        and POISSON_AGGREGATE_METRICS["brier_score"]
        < best_probability_baseline["brier_score"]
    )
    if probability_metrics_improved:
        print(
            "Poisson improved both probability metrics versus the strongest listed "
            "development baseline; still treat this as validation evidence only."
        )
    else:
        print(
            "Poisson did not clearly improve probability metrics versus the existing "
            "development baselines."
        )
    print(
        "Interpretation: Poisson can be kept as a scoreline layer only if the "
        "scoreline diagnostics are useful."
    )
    print(
        "Poisson should not replace logistic_elo or xgb_elo for W/D/L unless "
        "probability metrics improve."
    )
    print(
        "Draw recall is 0, so do not use the hard-class Poisson result as the "
        "final W/D/L predictor yet."
    )


def capture_table_counts(engine, table_names: list[str]) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    for table_name in table_names:
        if _table_exists(engine, table_name):
            counts[table_name] = _count_table_rows(engine, table_name)
        else:
            counts[table_name] = "MISSING"
    return counts


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name in WATCHED_TABLES:
        print(f"{table_name}: {counts[table_name]}")


def _verify_table_counts_unchanged(
    before_counts: dict[str, int | str],
    after_counts: dict[str, int | str],
) -> None:
    changed_counts = {
        table_name: (before_counts.get(table_name), after_counts.get(table_name))
        for table_name in WATCHED_TABLES
        if before_counts.get(table_name) != after_counts.get(table_name)
    }
    if changed_counts:
        raise RuntimeError(f"Watched table counts changed unexpectedly: {changed_counts}")
    print("Watched table counts unchanged.")


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    validate_historical_match_integrity(engine)
    before_counts = capture_table_counts(engine, WATCHED_TABLES)
    _print_table_counts("Watched table counts before", before_counts)

    run_walk_forward_poisson(engine)
    compare_poisson_to_existing_dev_baselines()

    after_counts = capture_table_counts(engine, WATCHED_TABLES)
    _print_table_counts("Watched table counts after", after_counts)
    _verify_table_counts_unchanged(before_counts, after_counts)

    print(f"{FINAL_TEST_SEASON} was not evaluated and remains reserved as final test.")
    print("No model artifacts were saved.")
    print("No database writes occurred.")
    print("No model features were written back to Tier 3 or Tier 2.")


if __name__ == "__main__":
    main()
