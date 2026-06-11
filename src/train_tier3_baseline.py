from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_pipeline import get_engine
from tier3_validation import (
    build_walk_forward_season_splits,
    get_ordered_seasons,
    split_development_and_final_test,
    validate_final_test_holdout,
    validate_historical_match_integrity,
    validate_walk_forward_splits,
)


warnings.filterwarnings("ignore", category=FutureWarning)

CLASS_LABELS = ["H", "D", "A"]
FINAL_TEST_SEASON = "2025-26"
EXPECTED_DEVELOPMENT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
EXPECTED_SEASONS = [*EXPECTED_DEVELOPMENT_SEASONS, FINAL_TEST_SEASON]
EXPECTED_FEATURE_ROW_COUNT = 1900
EXPECTED_SPLITS = [
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

ID_COLUMNS = [
    "match_id",
    "season_id",
    "match_date",
    "home_team",
    "away_team",
    "result",
]


def _table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        return conn.execute(
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


def _query_scalar(engine, query: str, params: dict[str, Any] | None = None):
    with engine.connect() as conn:
        return conn.execute(text(query), params or {}).scalar_one()


def _query_mappings(
    engine,
    query: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(query), params or {}).mappings().all()
    return [dict(row) for row in rows]


def get_baseline_feature_columns() -> list[str]:
    return [
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


def _verify_feature_table(engine) -> None:
    if not _table_exists(engine, "match_features_v3_base"):
        raise RuntimeError("match_features_v3_base table does not exist")

    required_columns = set(ID_COLUMNS + get_baseline_feature_columns())
    rows = _query_mappings(
        engine,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = CURRENT_SCHEMA()
            AND table_name = 'match_features_v3_base'
        """,
    )
    existing_columns = {row["column_name"] for row in rows}
    missing_columns = sorted(required_columns - existing_columns)
    if missing_columns:
        raise RuntimeError(
            "match_features_v3_base is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )

    total_rows = _query_scalar(engine, "SELECT COUNT(*) FROM match_features_v3_base")
    if total_rows != EXPECTED_FEATURE_ROW_COUNT:
        raise RuntimeError(
            "match_features_v3_base expected "
            f"{EXPECTED_FEATURE_ROW_COUNT} rows, found {total_rows}"
        )

    season_rows = _query_mappings(
        engine,
        """
        SELECT season_id, COUNT(*) AS row_count
        FROM match_features_v3_base
        GROUP BY season_id
        ORDER BY season_id
        """,
    )
    seasons_present = [row["season_id"] for row in season_rows]
    if seasons_present != EXPECTED_SEASONS:
        raise RuntimeError(
            f"match_features_v3_base seasons {seasons_present} != {EXPECTED_SEASONS}"
        )

    bad_season_counts = {
        row["season_id"]: row["row_count"]
        for row in season_rows
        if row["row_count"] != 380
    }
    if bad_season_counts:
        raise RuntimeError(
            "match_features_v3_base expected 380 rows per season, found "
            f"{bad_season_counts}"
        )

    print(
        "match_features_v3_base prerequisite check passed: "
        f"{total_rows} rows across {', '.join(seasons_present)}"
    )


def _assert_expected_walk_forward_setup(
    ordered_seasons: list[str],
    development_seasons: list[str],
    final_test_season: str,
    splits: list[dict[str, Any]],
) -> None:
    if ordered_seasons != EXPECTED_SEASONS:
        raise RuntimeError(f"Ordered seasons {ordered_seasons} != {EXPECTED_SEASONS}")
    if development_seasons != EXPECTED_DEVELOPMENT_SEASONS:
        raise RuntimeError(
            f"Development seasons {development_seasons} != "
            f"{EXPECTED_DEVELOPMENT_SEASONS}"
        )
    if final_test_season != FINAL_TEST_SEASON:
        raise RuntimeError(
            f"Final test season {final_test_season} != {FINAL_TEST_SEASON}"
        )
    if splits != EXPECTED_SPLITS:
        raise RuntimeError(f"Walk-forward splits {splits} != {EXPECTED_SPLITS}")

    print("Walk-forward setup check passed.")
    print(f"Development seasons: {', '.join(development_seasons)}")
    print(f"Reserved final test season: {final_test_season}")


def load_tier3_feature_dataset(
    engine,
    development_seasons: list[str],
) -> pd.DataFrame:
    if FINAL_TEST_SEASON in development_seasons:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be a development season")

    selected_columns = ID_COLUMNS + get_baseline_feature_columns()
    query = text(
        f"""
        SELECT {", ".join(selected_columns)}
        FROM match_features_v3_base
        WHERE season_id = ANY(:development_seasons)
        ORDER BY match_date, match_id
        """
    )
    df = pd.read_sql(
        query,
        engine,
        params={"development_seasons": development_seasons},
    )

    errors: list[str] = []
    if df.empty:
        errors.append("development feature dataset is empty")
    if (df["season_id"] == FINAL_TEST_SEASON).any():
        errors.append(f"{FINAL_TEST_SEASON} rows were loaded into development data")
    if df["match_id"].duplicated().any():
        errors.append(
            f"duplicate match_id values found: {int(df['match_id'].duplicated().sum())}"
        )

    unknown_results = sorted(set(df["result"].dropna()) - set(CLASS_LABELS))
    if unknown_results:
        errors.append(f"unknown result labels found: {unknown_results}")
    if df["result"].isna().any():
        errors.append(f"null result labels found: {int(df['result'].isna().sum())}")

    missing_seasons = [
        season_id for season_id in development_seasons if season_id not in set(df["season_id"])
    ]
    if missing_seasons:
        errors.append(f"missing development season rows: {missing_seasons}")

    if errors:
        raise ValueError("Tier 3 feature dataset validation failed: " + "; ".join(errors))

    print("=== Development Feature Rows ===")
    season_counts = df.groupby("season_id").size().sort_index()
    for season_id, count in season_counts.items():
        print(f"{season_id}: {int(count)} rows")
    print(f"Loaded {len(df)} development rows from match_features_v3_base")
    print(f"{FINAL_TEST_SEASON} was not loaded.")

    return df


def encode_targets(y: pd.Series) -> tuple[np.ndarray, list[str]]:
    if y.isna().any():
        raise ValueError(f"Target contains {int(y.isna().sum())} null value(s)")

    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    unknown_labels = sorted(set(y) - set(CLASS_LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown target label(s): {unknown_labels}")

    encoded = y.map(label_to_index).to_numpy(dtype=int)
    return encoded, CLASS_LABELS.copy()


def multiclass_brier_score(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
) -> float:
    n_classes = len(labels)
    y_one_hot = np.zeros((len(y_true_encoded), n_classes), dtype=float)
    y_one_hot[np.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def _align_predict_proba(
    y_proba: np.ndarray,
    observed_classes: np.ndarray,
    class_labels: list[str],
) -> np.ndarray:
    aligned = np.zeros((y_proba.shape[0], len(class_labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = y_proba[:, source_index]
    return _normalize_probabilities(aligned)


def _normalize_probabilities(y_proba: np.ndarray) -> np.ndarray:
    row_sums = y_proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Predicted probability row with non-positive sum found")
    return y_proba / row_sums


def _confusion_counts(
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    class_labels: list[str],
) -> dict[str, dict[str, int]]:
    encoded_labels = list(range(len(class_labels)))
    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=encoded_labels,
    )
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(class_labels)
        }
        for true_index, true_label in enumerate(class_labels)
    }


def evaluate_predictions(
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    encoded_labels = list(range(len(class_labels)))
    draw_index = class_labels.index("D")

    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(log_loss(y_true_encoded, y_proba, labels=encoded_labels)),
        "brier_score": multiclass_brier_score(
            y_true_encoded,
            y_proba,
            class_labels,
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
        "confusion_counts": _confusion_counts(
            y_true_encoded,
            y_pred_encoded,
            class_labels,
        ),
    }


def train_majority_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    del X_train

    class_counts = np.bincount(y_train, minlength=len(class_labels)).astype(float)
    majority_class = int(np.argmax(class_counts))
    class_proba = class_counts / class_counts.sum()
    y_pred = np.full(len(X_valid), majority_class, dtype=int)
    y_proba = _normalize_probabilities(np.tile(class_proba, (len(X_valid), 1)))

    return evaluate_predictions(y_valid, y_pred, y_proba, class_labels)


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_proba = _align_predict_proba(
        model.predict_proba(X_valid),
        model.named_steps["model"].classes_,
        class_labels,
    )
    return evaluate_predictions(y_valid, y_pred, y_proba, class_labels)


def train_xgb_imputed_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    model = Pipeline(
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
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_proba = _align_predict_proba(
        model.predict_proba(X_valid),
        model.named_steps["model"].classes_,
        class_labels,
    )
    return evaluate_predictions(y_valid, y_pred, y_proba, class_labels)


def train_xgb_native_nan_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    X_train_float = X_train.astype(float)
    X_valid_float = X_valid.astype(float)
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train_float, y_train)
    y_pred = model.predict(X_valid_float)
    y_proba = _align_predict_proba(
        model.predict_proba(X_valid_float),
        model.classes_,
        class_labels,
    )
    return evaluate_predictions(y_valid, y_pred, y_proba, class_labels)


def _print_confusion_counts(confusion_counts: dict[str, dict[str, int]]) -> None:
    print("Confusion counts (rows=true, columns=pred):")
    print("true\\pred    H    D    A")
    for true_label in CLASS_LABELS:
        row = confusion_counts[true_label]
        print(
            f"{true_label:>4}     "
            f"{row['H']:4d} {row['D']:4d} {row['A']:4d}"
        )


def _print_model_result(model_name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{model_name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"brier_score={metrics['brier_score']:.4f}, "
        f"draw_recall={metrics['draw_recall']:.4f}"
    )
    _print_confusion_counts(metrics["confusion_counts"])


def _print_aggregate_results(results_df: pd.DataFrame) -> None:
    print("=== Aggregate Walk-Forward Metrics ===")
    metric_columns = ["accuracy", "log_loss", "brier_score", "draw_recall"]
    for model_name, model_df in results_df.groupby("model_name", sort=False):
        print(model_name)
        for metric in metric_columns:
            mean_value = model_df[metric].mean()
            std_value = model_df[metric].std(ddof=1)
            if pd.isna(std_value):
                std_value = 0.0
            print(f"- {metric}: mean={mean_value:.4f}, std={std_value:.4f}")


def compare_xgb_variants(results_df: pd.DataFrame) -> None:
    print("=== XGBoost Variant Comparison ===")
    metric_columns = ["accuracy", "log_loss", "brier_score", "draw_recall"]
    aggregate = results_df.groupby("model_name")[metric_columns].mean()
    required_models = ["xgb_imputed_baseline", "xgb_native_nan_baseline"]
    missing_models = [model_name for model_name in required_models if model_name not in aggregate.index]
    if missing_models:
        raise ValueError(f"Missing XGBoost result(s): {missing_models}")

    imputed = aggregate.loc["xgb_imputed_baseline"]
    native = aggregate.loc["xgb_native_nan_baseline"]
    print("Mean metrics:")
    print(
        "xgb_imputed_baseline: "
        f"accuracy={imputed['accuracy']:.4f}, "
        f"log_loss={imputed['log_loss']:.4f}, "
        f"brier_score={imputed['brier_score']:.4f}, "
        f"draw_recall={imputed['draw_recall']:.4f}"
    )
    print(
        "xgb_native_nan_baseline: "
        f"accuracy={native['accuracy']:.4f}, "
        f"log_loss={native['log_loss']:.4f}, "
        f"brier_score={native['brier_score']:.4f}, "
        f"draw_recall={native['draw_recall']:.4f}"
    )

    deltas = native - imputed
    print("Deltas (native_nan minus imputed):")
    print(f"- accuracy: {deltas['accuracy']:+.4f} (higher is better)")
    print(f"- draw_recall: {deltas['draw_recall']:+.4f} (higher is better)")
    print(f"- log_loss: {deltas['log_loss']:+.4f} (lower is better)")
    print(f"- brier_score: {deltas['brier_score']:+.4f} (lower is better)")

    native_probability_wins = int(native["log_loss"] < imputed["log_loss"]) + int(
        native["brier_score"] < imputed["brier_score"]
    )
    imputed_probability_wins = int(native["log_loss"] > imputed["log_loss"]) + int(
        native["brier_score"] > imputed["brier_score"]
    )
    native_accuracy_better = native["accuracy"] > imputed["accuracy"]

    if native_probability_wins == 2:
        recommendation = (
            "Native NaN handling is the more promising XGBoost cleanup candidate "
            "on the probability metrics; keep accuracy as a secondary signal."
        )
    elif imputed_probability_wins == 2:
        recommendation = (
            "The imputed XGBoost variant remains the more cautious choice on the "
            "probability metrics; do not switch based on accuracy alone."
        )
    elif native_accuracy_better:
        recommendation = (
            "The probability metrics are mixed, so treat native NaN handling as "
            "inconclusive even if accuracy is better."
        )
    else:
        recommendation = (
            "The probability metrics are mixed and accuracy does not clearly favor "
            "native NaN handling; keep the imputed version as the conservative baseline."
        )
    print(f"Cautious recommendation: {recommendation}")


def run_walk_forward_baseline(engine) -> pd.DataFrame:
    _verify_feature_table(engine)

    ordered_seasons = get_ordered_seasons(engine)
    split_config = split_development_and_final_test(ordered_seasons)
    development_seasons = split_config["development_seasons"]
    final_test_season = split_config["final_test_season"]
    splits = build_walk_forward_season_splits(development_seasons)

    _assert_expected_walk_forward_setup(
        ordered_seasons,
        development_seasons,
        final_test_season,
        splits,
    )
    validate_walk_forward_splits(engine, splits)
    validate_final_test_holdout(engine, development_seasons, final_test_season)

    dataset = load_tier3_feature_dataset(engine, development_seasons)
    feature_columns = get_baseline_feature_columns()
    print(f"Baseline feature count: {len(feature_columns)}")

    expected_fold_rows = {
        1: (760, 380),
        2: (1140, 380),
    }
    trainers = [
        ("majority_baseline", train_majority_baseline),
        ("logistic_baseline", train_logistic_baseline),
        ("xgb_imputed_baseline", train_xgb_imputed_baseline),
        ("xgb_native_nan_baseline", train_xgb_native_nan_baseline),
    ]

    result_rows: list[dict[str, Any]] = []
    for split in splits:
        fold = split["fold"]
        train_seasons = split["train_seasons"]
        validation_seasons = split["validation_seasons"]
        train_df = dataset.loc[dataset["season_id"].isin(train_seasons)].copy()
        valid_df = dataset.loc[dataset["season_id"].isin(validation_seasons)].copy()

        expected_train_rows, expected_valid_rows = expected_fold_rows[fold]
        if len(train_df) != expected_train_rows or len(valid_df) != expected_valid_rows:
            raise RuntimeError(
                f"Fold {fold} expected {expected_train_rows}/{expected_valid_rows} "
                f"train/validation rows, found {len(train_df)}/{len(valid_df)}"
            )

        X_train = train_df[feature_columns].copy()
        X_valid = valid_df[feature_columns].copy()
        y_train, class_labels = encode_targets(train_df["result"])
        y_valid, _ = encode_targets(valid_df["result"])

        print(f"=== Fold {fold} ===")
        print(f"Train seasons: {', '.join(train_seasons)}")
        print(f"Validation seasons: {', '.join(validation_seasons)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Validation rows: {len(valid_df)}")
        print("Class labels: " + ", ".join(class_labels))

        for model_name, trainer in trainers:
            metrics = trainer(X_train, y_train, X_valid, y_valid, class_labels)
            _print_model_result(model_name, metrics)
            result_rows.append(
                {
                    "fold": fold,
                    "model_name": model_name,
                    "train_seasons": ", ".join(train_seasons),
                    "validation_seasons": ", ".join(validation_seasons),
                    "train_rows": len(train_df),
                    "validation_rows": len(valid_df),
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "draw_recall": metrics["draw_recall"],
                    "confusion_counts": metrics["confusion_counts"],
                }
            )

    results_df = pd.DataFrame(result_rows)
    _print_aggregate_results(results_df)
    compare_xgb_variants(results_df)
    return results_df


def main() -> None:
    engine = get_engine()
    if engine is None:
        raise SystemExit("Could not connect to PostgreSQL.")

    print("=== Tier 3 Baseline Walk-Forward Evaluation ===")
    validate_historical_match_integrity(engine)
    results_df = run_walk_forward_baseline(engine)
    print(f"Evaluated {len(results_df)} fold/model result rows.")
    print(
        "2025-26 was not evaluated and remains reserved as final test; "
        "this was development-only walk-forward validation."
    )
    print("No model artifacts were saved.")
    print("No database writes occurred.")


if __name__ == "__main__":
    main()
