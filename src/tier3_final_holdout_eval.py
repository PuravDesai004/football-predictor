from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy
import pandas
from sqlalchemy import text
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

from data_pipeline import get_engine


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE = "match_features_v3_elo"
FINAL_CANDIDATE_NAME = "logistic_elo_expanding"
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_HOLDOUT_SEASON = "2025-26"
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
RANDOM_STATE = 42
DOMINANT_CLASS_MAX_PROB = 0.50
THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]

EXPECTED_TRAIN_ROWS = 1520
EXPECTED_HOLDOUT_ROWS = 380
EXPECTED_FEATURE_COUNT = 32

BASE_ELO_FEATURE_COLUMNS = [
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

FORBIDDEN_EXACT_FEATURES = {
    "match_id",
    "season",
    "season_id",
    "match_date",
    "kickoff_time",
    "home_team",
    "away_team",
    TARGET_COLUMN,
    "home_goals",
    "away_goals",
    "home_win",
    "is_draw",
    "away_win",
    "created_at",
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}
FORBIDDEN_FEATURE_TOKENS = [
    "h2h",
    "style",
    "pressure",
    "poisson",
    "odds",
    "betting",
    "manager",
    "sentiment",
    "injury",
    "rivalry",
    "derby",
    "calibration",
]
WATCHED_TABLES = [
    "historical_matches",
    "historical_understat_xg",
    "match_features_v3_base",
    "elo_ratings_v3",
    "match_features_v3_elo",
    "standings_before_match_v3",
    "match_features_v3_pressure_experiment",
    "match_features_v3_style_experiment",
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


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def load_final_eval_data(conn) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    train_df = _load_seasons(conn, TRAIN_SEASONS)
    holdout_df = _load_seasons(conn, [FINAL_HOLDOUT_SEASON])

    train_df["match_date"] = pandas.to_datetime(train_df["match_date"])
    holdout_df["match_date"] = pandas.to_datetime(holdout_df["match_date"])

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} training rows, found {len(train_df)}")
    if len(holdout_df) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_HOLDOUT_ROWS} final holdout rows, found {len(holdout_df)}"
        )

    return train_df, holdout_df


def get_feature_columns(df) -> list[str]:
    feature_columns = BASE_ELO_FEATURE_COLUMNS.copy()
    missing_columns = sorted(set(feature_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing frozen candidate feature column(s): {missing_columns}")
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, found {len(feature_columns)}"
        )

    forbidden_exact = sorted(set(feature_columns) & FORBIDDEN_EXACT_FEATURES)
    forbidden_tokens = sorted(
        column
        for column in feature_columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    non_numeric_features = [
        column
        for column in feature_columns
        if column in df.columns and not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if forbidden_exact or forbidden_tokens or non_numeric_features:
        raise ValueError(
            "Frozen candidate feature audit failed: "
            f"forbidden_exact={forbidden_exact}, "
            f"forbidden_tokens={forbidden_tokens}, "
            f"non_numeric={non_numeric_features}"
        )

    return feature_columns


def validate_frozen_candidate_inputs(train_df, holdout_df, feature_columns) -> None:
    errors: list[str] = []

    train_seasons = sorted(train_df["season_id"].dropna().unique().tolist())
    holdout_seasons = sorted(holdout_df["season_id"].dropna().unique().tolist())
    if train_seasons != TRAIN_SEASONS:
        errors.append(f"training seasons {train_seasons} != {TRAIN_SEASONS}")
    if holdout_seasons != [FINAL_HOLDOUT_SEASON]:
        errors.append(f"holdout seasons {holdout_seasons} != {[FINAL_HOLDOUT_SEASON]}")
    if FINAL_HOLDOUT_SEASON in set(train_df["season_id"]):
        errors.append(f"{FINAL_HOLDOUT_SEASON} was loaded into training data")
    if set(TRAIN_SEASONS) & set(holdout_df["season_id"]):
        errors.append("training season appeared in holdout dataframe")
    if train_df["match_date"].max() >= holdout_df["match_date"].min():
        errors.append(
            "date leakage: max training date "
            f"{train_df['match_date'].max().date()} >= "
            f"min holdout date {holdout_df['match_date'].min().date()}"
        )

    for name, frame in [("training", train_df), ("holdout", holdout_df)]:
        missing_columns = sorted({TARGET_COLUMN, *feature_columns} - set(frame.columns))
        if missing_columns:
            errors.append(f"{name} missing columns: {missing_columns}")
        null_targets = int(frame[TARGET_COLUMN].isna().sum())
        if null_targets:
            errors.append(f"{name} null target count: {null_targets}")
        unknown_labels = sorted(set(frame[TARGET_COLUMN].dropna()) - set(LABELS))
        if unknown_labels:
            errors.append(f"{name} unknown result labels: {unknown_labels}")
        duplicate_match_ids = int(frame["match_id"].duplicated().sum())
        if duplicate_match_ids:
            errors.append(f"{name} duplicate match_id count: {duplicate_match_ids}")

    if errors:
        raise ValueError("Frozen final holdout input validation failed: " + "; ".join(errors))

    print("PASS: frozen candidate inputs validated.")
    print(f"Training rows: {len(train_df)}")
    print(f"Training seasons: {', '.join(train_seasons)}")
    print(f"Final holdout rows: {len(holdout_df)}")
    print(f"Final holdout season: {FINAL_HOLDOUT_SEASON}")
    print(
        f"Date check: train max {train_df['match_date'].max().date()} < "
        f"holdout min {holdout_df['match_date'].min().date()}"
    )
    print(f"Frozen candidate feature count: {len(feature_columns)}")


def build_logistic_elo_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def predict_probabilities(model, X, labels) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned)


def predict_argmax(probabilities, labels) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    return [labels[int(index)] for index in prediction_indexes]


def is_draw_second_highest(probability_row, labels) -> bool:
    probabilities = numpy.asarray(probability_row, dtype=float)
    if probabilities.shape != (len(labels),):
        raise ValueError("Probability row shape does not match labels")
    draw_index = labels.index("D")
    descending_indexes = numpy.argsort(-probabilities, kind="mergesort")
    return int(descending_indexes[1]) == draw_index


def apply_draw_overlay(probabilities, labels, draw_threshold) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    predictions = predict_argmax(probabilities, labels)
    draw_index = labels.index("D")

    overlay_predictions: list[str] = []
    for row_index, probability_row in enumerate(probabilities):
        draw_prob = float(probability_row[draw_index])
        max_prob = float(probability_row.max())
        should_change_to_draw = (
            draw_prob >= draw_threshold
            and is_draw_second_highest(probability_row, labels)
            and max_prob < DOMINANT_CLASS_MAX_PROB
        )
        overlay_predictions.append("D" if should_change_to_draw else predictions[row_index])
    return overlay_predictions


def select_draw_threshold_from_training(probabilities, y_true, labels) -> dict:
    probabilities = _normalize_probabilities(probabilities)
    argmax_predictions = predict_argmax(probabilities, labels)
    argmax_metrics = evaluate_predictions(y_true, argmax_predictions, probabilities, labels)

    best_config: dict[str, Any] | None = None
    for draw_threshold in THRESHOLD_GRID:
        overlay_predictions = apply_draw_overlay(probabilities, labels, draw_threshold)
        metrics = evaluate_predictions(y_true, overlay_predictions, probabilities, labels)
        labels_changed_to_draw = _count_labels_changed_to_draw(
            argmax_predictions,
            overlay_predictions,
        )
        if metrics["accuracy"] < argmax_metrics["accuracy"] - 0.01:
            continue

        selection_key = (
            metrics["draw_f1"],
            metrics["accuracy"],
            -labels_changed_to_draw,
            draw_threshold,
        )
        if best_config is None or selection_key > best_config["selection_key"]:
            best_config = {
                "selected_draw_threshold": draw_threshold,
                "selection_key": selection_key,
                "training_argmax_accuracy": argmax_metrics["accuracy"],
                "training_argmax_draw_f1": argmax_metrics["draw_f1"],
                "training_overlay_accuracy": metrics["accuracy"],
                "training_overlay_draw_f1": metrics["draw_f1"],
                "training_labels_changed_to_draw": labels_changed_to_draw,
            }

    if best_config is None:
        raise RuntimeError("No draw threshold candidate survived training accuracy gate")

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
    actual_distribution = {
        label: int(numpy.sum(y_true_encoded == index))
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
        "predicted_class_distribution": predicted_distribution,
        "actual_class_distribution": actual_distribution,
        "mean_predicted_probability": mean_predicted_probability,
        "confusion_counts": _confusion_counts(y_true_encoded, y_pred_encoded, labels),
    }


def print_final_holdout_report(results) -> None:
    print("=== Tier 3 Final Holdout Evaluation Report ===")
    print(f"Frozen candidate: {FINAL_CANDIDATE_NAME}")
    print(f"Source table: {SOURCE_TABLE}")
    print(f"Training seasons: {', '.join(results['training_seasons'])}")
    print(f"Training rows: {results['training_rows']}")
    print(f"Final holdout season: {FINAL_HOLDOUT_SEASON}")
    print(f"Final holdout rows: {results['holdout_rows']}")
    print(f"Feature count: {results['feature_count']}")
    print(
        f"Selected draw threshold from training only: "
        f"{results['selected_draw_threshold']:.2f}"
    )
    print(
        "Training threshold selection: "
        f"argmax_accuracy={results['threshold_config']['training_argmax_accuracy']:.4f}, "
        f"argmax_draw_f1={results['threshold_config']['training_argmax_draw_f1']:.4f}, "
        f"overlay_accuracy={results['threshold_config']['training_overlay_accuracy']:.4f}, "
        f"overlay_draw_f1={results['threshold_config']['training_overlay_draw_f1']:.4f}, "
        f"changed_to_draw={results['threshold_config']['training_labels_changed_to_draw']}"
    )

    print("Actual class distribution:")
    _print_distribution(results["argmax_metrics"]["actual_class_distribution"])

    print("Argmax final holdout metrics:")
    _print_metrics(results["argmax_metrics"], changed_to_draw=0)

    print("Draw overlay final holdout metrics:")
    _print_metrics(
        results["overlay_metrics"],
        changed_to_draw=results["labels_changed_to_draw"],
    )

    print(
        "Probability metric unchanged assertion: "
        f"{'PASS' if results['probability_metrics_unchanged'] else 'FAIL'}"
    )
    print("Watched table counts before/after:")
    for table_name in WATCHED_TABLES:
        before_count = results["before_counts"].get(table_name)
        after_count = results["after_counts"].get(table_name)
        print(f"- {table_name}: {before_count} -> {after_count}")


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


def assert_no_counts_changed(before, after) -> None:
    changed = {
        table_name: (before.get(table_name), after.get(table_name))
        for table_name in sorted(set(before) | set(after))
        if before.get(table_name) != after.get(table_name)
    }
    if changed:
        raise RuntimeError(f"Watched table counts changed unexpectedly: {changed}")
    print("PASS: watched table counts unchanged.")


def main() -> None:
    print("=== Tier 3 Phase 10B Final 2025-26 Holdout Evaluation ===")
    print("This evaluates the frozen candidate once on the reserved final holdout.")
    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)

    train_df, holdout_df = load_final_eval_data(conn)
    feature_columns = get_feature_columns(train_df)
    validate_frozen_candidate_inputs(train_df, holdout_df, feature_columns)

    model = build_logistic_elo_model()
    model.fit(
        train_df[feature_columns].copy(),
        _encode_labels(train_df[TARGET_COLUMN], LABELS),
    )

    training_probabilities = predict_probabilities(
        model,
        train_df[feature_columns].copy(),
        LABELS,
    )
    threshold_config = select_draw_threshold_from_training(
        training_probabilities,
        train_df[TARGET_COLUMN].tolist(),
        LABELS,
    )
    selected_draw_threshold = threshold_config["selected_draw_threshold"]

    holdout_probabilities = predict_probabilities(
        model,
        holdout_df[feature_columns].copy(),
        LABELS,
    )
    argmax_predictions = predict_argmax(holdout_probabilities, LABELS)
    overlay_predictions = apply_draw_overlay(
        holdout_probabilities,
        LABELS,
        selected_draw_threshold,
    )

    argmax_metrics = evaluate_predictions(
        holdout_df[TARGET_COLUMN].tolist(),
        argmax_predictions,
        holdout_probabilities,
        LABELS,
    )
    overlay_metrics = evaluate_predictions(
        holdout_df[TARGET_COLUMN].tolist(),
        overlay_predictions,
        holdout_probabilities,
        LABELS,
    )
    labels_changed_to_draw = _count_labels_changed_to_draw(
        argmax_predictions,
        overlay_predictions,
    )

    probability_metrics_unchanged = (
        argmax_metrics["log_loss"] == overlay_metrics["log_loss"]
        and argmax_metrics["brier_score"] == overlay_metrics["brier_score"]
    )
    if not probability_metrics_unchanged:
        raise RuntimeError("Overlay changed probability metrics unexpectedly")

    after_counts = capture_watched_table_counts(conn)
    assert_no_counts_changed(before_counts, after_counts)

    results = {
        "training_seasons": TRAIN_SEASONS,
        "training_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "feature_count": len(feature_columns),
        "selected_draw_threshold": selected_draw_threshold,
        "threshold_config": threshold_config,
        "argmax_metrics": argmax_metrics,
        "overlay_metrics": overlay_metrics,
        "labels_changed_to_draw": labels_changed_to_draw,
        "probability_metrics_unchanged": probability_metrics_unchanged,
        "before_counts": before_counts,
        "after_counts": after_counts,
    }
    print_final_holdout_report(results)

    print("PASS: final holdout evaluation used the frozen candidate only.")
    print("PASS: draw threshold was selected from training rows only.")
    print("PASS: draw overlay did not alter probabilities.")
    print("No model artifact was saved.")
    print("No database writes occurred.")
    print("No Streamlit, Tier 2 artifact, H2H, style, pressure, Poisson, calibration, odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


def _load_seasons(conn, seasons: list[str]) -> pandas.DataFrame:
    query = text(
        f"""
        SELECT *
        FROM {SOURCE_TABLE}
        WHERE season_id = ANY(:seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    return pandas.read_sql(query, conn, params={"seasons": seasons})


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    if probabilities.shape[1] != len(LABELS):
        raise ValueError("Probability column count does not match labels")
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("Non-finite probability value found")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Probability row with non-positive sum found")
    normalized = probabilities / row_sums
    if not numpy.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Probability rows do not sum to 1")
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


def _count_labels_changed_to_draw(
    baseline_predictions: list[str],
    overlay_predictions: list[str],
) -> int:
    return int(
        sum(
            baseline != "D" and overlay == "D"
            for baseline, overlay in zip(baseline_predictions, overlay_predictions)
        )
    )


def _print_metrics(metrics: dict[str, Any], changed_to_draw: int) -> None:
    print(f"- accuracy: {metrics['accuracy']:.4f}")
    print(f"- log_loss: {metrics['log_loss']:.4f}")
    print(f"- brier_score: {metrics['brier_score']:.4f}")
    print(f"- draw_recall: {metrics['draw_recall']:.4f}")
    print(f"- draw_precision: {metrics['draw_precision']:.4f}")
    print(f"- draw_f1: {metrics['draw_f1']:.4f}")
    print(f"- labels_changed_to_draw: {changed_to_draw}")
    print("- predicted_class_distribution:")
    _print_distribution(metrics["predicted_class_distribution"])
    print("- mean_predicted_probability:")
    for label in LABELS:
        print(f"  {label}: {metrics['mean_predicted_probability'][label]:.4f}")
    print("- confusion_counts:")
    _print_confusion_counts(metrics["confusion_counts"])


def _print_distribution(distribution: dict[str, int]) -> None:
    for label in LABELS:
        print(f"  {label}: {distribution[label]}")


def _print_confusion_counts(confusion_counts: dict[str, dict[str, int]]) -> None:
    print("  true\\pred    H    D    A")
    for true_label in LABELS:
        row = confusion_counts[true_label]
        print(
            f"  {true_label:>4}     "
            f"{row['H']:4d} {row['D']:4d} {row['A']:4d}"
        )


if __name__ == "__main__":
    main()
