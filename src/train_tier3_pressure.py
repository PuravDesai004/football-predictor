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
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]

EXPECTED_FULL_ROW_COUNT = 1900
EXPECTED_DEV_ROW_COUNT = 1520
EXPECTED_SEASON_ROWS = 380
PRESSURE_TABLE = "match_features_v3_pressure_experiment"

CHAMPION_METRICS = {
    "accuracy": 0.5579,
    "log_loss": 0.9705,
    "brier_score": 0.5730,
    "draw_f1": 0.0893,
}

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

PRESSURE_FEATURE_COLUMNS = [
    "home_games_played_before",
    "away_games_played_before",
    "home_points_before",
    "away_points_before",
    "home_ppg_before",
    "away_ppg_before",
    "home_rank_before",
    "away_rank_before",
    "home_goal_diff_before",
    "away_goal_diff_before",
    "rank_diff_before",
    "points_diff_before",
    "ppg_diff_before",
    "goal_diff_table_diff_before",
    "home_title_pressure_before",
    "away_title_pressure_before",
    "home_top4_pressure_before",
    "away_top4_pressure_before",
    "home_top6_pressure_before",
    "away_top6_pressure_before",
    "home_relegation_pressure_before",
    "away_relegation_pressure_before",
    "home_pressure_index_before",
    "away_pressure_index_before",
    "match_pressure_index_before",
    "pressure_diff_before",
    "season_progress_before",
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

FORBIDDEN_POST_MATCH_ELO_COLUMNS = {
    "home_elo_after",
    "away_elo_after",
    "home_elo_delta",
    "away_elo_delta",
    "actual_home_score",
    "actual_away_score",
}

EXCLUDED_FEATURE_COLUMNS = {
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
    "created_at",
    "home_initialization",
    "away_initialization",
    *FORBIDDEN_POST_MATCH_ELO_COLUMNS,
}

FORBIDDEN_FEATURE_TOKENS = [
    "h2h",
    "style",
    "poisson",
    "odds",
    "manager",
    "sentiment",
    "injury",
    "rivalry",
    "derby",
    "cluster",
    "label",
]


def get_db_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Could not connect to PostgreSQL.")
    return engine


def _table_exists(conn, table_name: str) -> bool:
    with conn.connect() as db_conn:
        return db_conn.execute(
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


def _count_table_rows(conn, table_name: str) -> int:
    with conn.connect() as db_conn:
        return int(db_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _encode_labels(values, labels: list[str]) -> numpy.ndarray:
    raw_values = values.tolist() if isinstance(values, pandas.Series) else list(values)
    label_to_index = {label: index for index, label in enumerate(labels)}
    unknown_labels = sorted(set(raw_values) - set(labels))
    if unknown_labels:
        raise ValueError(f"Unknown target label(s): {unknown_labels}")
    return numpy.asarray([label_to_index[value] for value in raw_values], dtype=int)


def _normalize_probabilities(probabilities) -> numpy.ndarray:
    probabilities = numpy.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a 2D array")
    if not numpy.all(numpy.isfinite(probabilities)):
        raise ValueError("NaN or infinite predicted probabilities found")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if numpy.any(row_sums <= 0):
        raise ValueError("Predicted probability row with non-positive sum found")
    normalized = probabilities / row_sums
    if not numpy.allclose(normalized.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Predicted probability rows do not sum to 1")
    return normalized


def _align_predict_proba(model, X, labels: list[str]) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(X)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(labels)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned)


def _predict_labels_from_probabilities(probabilities, labels: list[str]) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    return [labels[int(index)] for index in numpy.argmax(probabilities, axis=1)]


def _multiclass_brier_score(y_true, probabilities, labels: list[str]) -> float:
    y_true_encoded = _encode_labels(y_true, labels)
    probabilities = _normalize_probabilities(probabilities)
    y_one_hot = numpy.zeros((len(y_true_encoded), len(labels)), dtype=float)
    y_one_hot[numpy.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(numpy.mean(numpy.sum((probabilities - y_one_hot) ** 2, axis=1)))


def _confusion_counts(y_true, y_pred, labels: list[str]) -> dict[str, dict[str, int]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(labels)
        }
        for true_index, true_label in enumerate(labels)
    }


def _predicted_distribution(y_pred, labels: list[str]) -> dict[str, int]:
    series = pandas.Series(y_pred)
    return {label: int((series == label).sum()) for label in labels}


def _mean_predicted_probability(probabilities, labels: list[str]) -> dict[str, float]:
    probabilities = _normalize_probabilities(probabilities)
    return {
        label: float(probabilities[:, index].mean())
        for index, label in enumerate(labels)
    }


def _validate_selected_features(df: pandas.DataFrame, features: list[str], label: str) -> None:
    missing_columns = sorted(set(features) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing {label} feature column(s): {missing_columns}")

    excluded_columns = sorted(set(features) & EXCLUDED_FEATURE_COLUMNS)
    if excluded_columns:
        raise ValueError(f"Excluded columns selected in {label}: {excluded_columns}")

    forbidden_columns = sorted(
        column
        for column in features
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if forbidden_columns:
        raise ValueError(f"Forbidden feature family selected in {label}: {forbidden_columns}")

    non_numeric_columns = [
        column for column in features if not pandas.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Non-numeric columns selected in {label}: {non_numeric_columns}")


def _print_table_counts(label: str, counts: dict[str, int | str]) -> None:
    print(f"=== {label} ===")
    for table_name in WATCHED_TABLES:
        print(f"{table_name}: {counts.get(table_name)}")


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
    _print_confusion_counts(metrics["confusion_counts"])


def load_pressure_feature_data(conn) -> pandas.DataFrame:
    if FINAL_TEST_SEASON in DEV_SEASONS:
        raise ValueError(f"{FINAL_TEST_SEASON} cannot be a development season")
    if not _table_exists(conn, PRESSURE_TABLE):
        raise RuntimeError(f"{PRESSURE_TABLE} table does not exist")

    full_count = _count_table_rows(conn, PRESSURE_TABLE)
    if full_count != EXPECTED_FULL_ROW_COUNT:
        raise RuntimeError(
            f"{PRESSURE_TABLE} expected {EXPECTED_FULL_ROW_COUNT} rows, found {full_count}"
        )

    query = text(
        f"""
        SELECT *
        FROM {PRESSURE_TABLE}
        WHERE season_id = ANY(:dev_seasons)
        ORDER BY match_date, kickoff_time, match_id
        """
    )
    df = pandas.read_sql(query, conn, params={"dev_seasons": DEV_SEASONS})
    df["match_date"] = pandas.to_datetime(df["match_date"])

    errors: list[str] = []
    if len(df) != EXPECTED_DEV_ROW_COUNT:
        errors.append(f"expected {EXPECTED_DEV_ROW_COUNT} development rows, found {len(df)}")
    if df.empty:
        errors.append("development modeling dataframe is empty")
    if df["match_id"].duplicated().any():
        errors.append(f"duplicate match_id count: {int(df['match_id'].duplicated().sum())}")

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

    if df[TARGET_COLUMN].isna().any():
        errors.append(f"null target count: {int(df[TARGET_COLUMN].isna().sum())}")
    unknown_targets = sorted(set(df[TARGET_COLUMN].dropna()) - set(LABELS))
    if unknown_targets:
        errors.append(f"unknown target labels: {unknown_targets}")

    forbidden_present = sorted(FORBIDDEN_POST_MATCH_ELO_COLUMNS & set(df.columns))
    if forbidden_present:
        errors.append(f"forbidden post-match Elo columns loaded: {forbidden_present}")

    forbidden_family_columns = sorted(
        column
        for column in df.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if forbidden_family_columns:
        errors.append(f"forbidden feature-family columns present: {forbidden_family_columns}")

    pressure_columns = get_pressure_feature_columns()
    missing_pressure = sorted(set(pressure_columns) - set(df.columns))
    if missing_pressure:
        errors.append(f"missing pressure feature columns: {missing_pressure}")

    if errors:
        raise ValueError("Pressure experiment data validation failed: " + "; ".join(errors))

    validate_no_final_holdout_loaded(df)
    print("=== Development Pressure Feature Data ===")
    print(f"Loaded modeling dataframe rows: {len(df)}")
    for season_id, count in df.groupby("season_id").size().sort_index().items():
        print(f"- {season_id}: {int(count)} rows")
    print(f"{FINAL_TEST_SEASON} was not loaded into the modeling dataframe.")
    return df


def validate_no_final_holdout_loaded(df) -> None:
    if FINAL_TEST_SEASON in set(df["season_id"].dropna()):
        raise ValueError(f"{FINAL_TEST_SEASON} was loaded into the modeling dataframe")
    print(f"PASS: {FINAL_TEST_SEASON} not loaded.")


def get_base_elo_feature_columns(df) -> list[str]:
    features = BASE_ELO_FEATURE_COLUMNS.copy()
    _validate_selected_features(df, features, "base Elo")
    return features


def get_pressure_feature_columns() -> list[str]:
    return PRESSURE_FEATURE_COLUMNS.copy()


def compute_pressure_coverage_diagnostics(df, pressure_features) -> None:
    print("=== Pressure Coverage Diagnostics ===")
    print("Pressure feature null counts:")
    total_nulls = df[pressure_features].isna().sum()
    for column, count in total_nulls.items():
        print(f"- {column}: {int(count)}")

    print("Pressure feature null counts by season:")
    season_nulls = df.groupby("season_id")[pressure_features].apply(
        lambda frame: frame.isna().sum()
    )
    print(season_nulls.to_string())

    print("Pressure feature null counts by fold:")
    for fold_config in FOLDS:
        fold = fold_config["fold"]
        fold_df = df.loc[
            df["season_id"].isin(
                fold_config["train_seasons"] + fold_config["validation_seasons"]
            )
        ]
        fold_nulls = fold_df[pressure_features].isna().sum()
        print(f"Fold {fold} rows: {len(fold_df)}")
        for column, count in fold_nulls.items():
            print(f"- {column}: {int(count)}")

    home_non_null = df["home_pressure_index_before"].notna()
    away_non_null = df["away_pressure_index_before"].notna()
    both_non_null = int((home_non_null & away_non_null).sum())
    only_one_non_null = int((home_non_null ^ away_non_null).sum())
    both_null = int((~home_non_null & ~away_non_null).sum())
    print(f"Rows where both pressure indexes are non-null: {both_non_null}")
    print(f"Rows where only one pressure index is non-null: {only_one_non_null}")
    print(f"Rows where both pressure indexes are null: {both_null}")
    print("Pressure coverage by season:")
    for season_id, season_df in df.groupby("season_id", sort=True):
        home_nn = season_df["home_pressure_index_before"].notna()
        away_nn = season_df["away_pressure_index_before"].notna()
        print(
            f"- {season_id}: both_non_null={int((home_nn & away_nn).sum())}, "
            f"only_one_non_null={int((home_nn ^ away_nn).sum())}, "
            f"both_null={int((~home_nn & ~away_nn).sum())}"
        )


def compute_redundancy_diagnostics(
    df,
    base_elo_features,
    pressure_features,
) -> pandas.DataFrame:
    rows: list[dict[str, Any]] = []
    for pressure_feature in pressure_features:
        for base_feature in base_elo_features:
            pair_df = df[[pressure_feature, base_feature]].dropna()
            if len(pair_df) < 2:
                continue
            if (
                pair_df[pressure_feature].std(ddof=0) == 0
                or pair_df[base_feature].std(ddof=0) == 0
            ):
                continue
            corr = pair_df[pressure_feature].corr(pair_df[base_feature], method="pearson")
            if pandas.isna(corr):
                continue
            rows.append(
                {
                    "pressure_feature": pressure_feature,
                    "base_elo_feature": base_feature,
                    "pearson_corr": float(corr),
                    "abs_pearson_corr": float(abs(corr)),
                    "likely_redundant": bool(abs(corr) >= 0.95),
                    "paired_rows": int(len(pair_df)),
                }
            )

    redundancy_df = pandas.DataFrame(
        rows,
        columns=[
            "pressure_feature",
            "base_elo_feature",
            "pearson_corr",
            "abs_pearson_corr",
            "likely_redundant",
            "paired_rows",
        ],
    )
    if redundancy_df.empty:
        print("No redundancy diagnostics available.")
        return redundancy_df

    redundancy_df = redundancy_df.sort_values(
        ["abs_pearson_corr", "pressure_feature", "base_elo_feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    print("=== Pressure Redundancy Diagnostics ===")
    print("Top 20 absolute Pearson correlations against base/Elo features:")
    print(
        redundancy_df.head(20).to_string(
            index=False,
            formatters={
                "pearson_corr": "{:.4f}".format,
                "abs_pearson_corr": "{:.4f}".format,
            },
        )
    )

    flagged = redundancy_df.loc[redundancy_df["likely_redundant"]]
    flagged_pressure_count = int(flagged["pressure_feature"].nunique())
    print(f"Pressure features with abs correlation >= 0.95: {flagged_pressure_count}")
    if flagged_pressure_count:
        for feature_name in sorted(flagged["pressure_feature"].unique()):
            max_corr = flagged.loc[
                flagged["pressure_feature"] == feature_name,
                "abs_pearson_corr",
            ].max()
            print(f"- {feature_name}: max abs corr {max_corr:.4f}")

    return redundancy_df


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


def evaluate_predictions(y_true, y_pred, y_prob, labels) -> dict:
    y_prob = _normalize_probabilities(y_prob)
    y_true_encoded = _encode_labels(y_true, labels)
    draw_precision, draw_recall, draw_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["D"],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(
            log_loss(y_true_encoded, y_prob, labels=list(range(len(labels))))
        ),
        "brier_score": _multiclass_brier_score(y_true, y_prob, labels),
        "draw_recall": float(draw_recall[0]),
        "draw_precision": float(draw_precision[0]),
        "draw_f1": float(draw_f1[0]),
        "predicted_class_distribution": _predicted_distribution(y_pred, labels),
        "mean_predicted_probability": _mean_predicted_probability(y_prob, labels),
        "confusion_counts": _confusion_counts(y_true, y_pred, labels),
    }


def run_fold(df, fold_config, feature_sets, models) -> list[dict]:
    fold = fold_config["fold"]
    train_seasons = fold_config["train_seasons"]
    validation_seasons = fold_config["validation_seasons"]

    train_df = df.loc[df["season_id"].isin(train_seasons)].copy()
    validation_df = df.loc[df["season_id"].isin(validation_seasons)].copy()

    if FINAL_TEST_SEASON in set(train_df["season_id"]) | set(validation_df["season_id"]):
        raise ValueError(f"{FINAL_TEST_SEASON} entered fold {fold}")
    if set(train_seasons) & set(validation_seasons):
        raise ValueError(f"Fold {fold} train/validation seasons overlap")
    if len(train_df) != len(train_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(train_seasons) * EXPECTED_SEASON_ROWS} "
            f"train rows, found {len(train_df)}"
        )
    if len(validation_df) != len(validation_seasons) * EXPECTED_SEASON_ROWS:
        raise ValueError(
            f"Fold {fold} expected {len(validation_seasons) * EXPECTED_SEASON_ROWS} "
            f"validation rows, found {len(validation_df)}"
        )
    if train_df["match_date"].max() >= validation_df["match_date"].min():
        raise ValueError(f"Fold {fold} has date leakage")

    print(f"=== Fold {fold} ===")
    print(f"Train seasons: {', '.join(train_seasons)}")
    print(f"Validation seasons: {', '.join(validation_seasons)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(
        f"Date check: train max {train_df['match_date'].max().date()} < "
        f"validation min {validation_df['match_date'].min().date()}"
    )

    y_train = _encode_labels(train_df[TARGET_COLUMN], LABELS)
    y_validation = validation_df[TARGET_COLUMN].tolist()

    model_runs = [
        ("logistic_elo", "logistic", "base_elo"),
        ("logistic_elo_pressure", "logistic", "base_elo_pressure"),
        ("xgb_elo", "xgb", "base_elo"),
        ("xgb_elo_pressure", "xgb", "base_elo_pressure"),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, model_key, feature_set_key in model_runs:
        model = clone(models[model_key])
        feature_columns = feature_sets[feature_set_key]
        model.fit(train_df[feature_columns].copy(), y_train)
        probabilities = _align_predict_proba(
            model,
            validation_df[feature_columns].copy(),
            LABELS,
        )
        predictions = _predict_labels_from_probabilities(probabilities, LABELS)
        metrics = evaluate_predictions(
            y_validation,
            predictions,
            probabilities,
            LABELS,
        )
        _print_model_result(model_name, metrics)
        rows.append(
            {
                "fold": fold,
                "model_name": model_name,
                "model_family": model_key,
                "feature_set": feature_set_key,
                "train_seasons": ", ".join(train_seasons),
                "validation_seasons": ", ".join(validation_seasons),
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                **metrics,
            }
        )
    return rows


def aggregate_results(results) -> pandas.DataFrame:
    results_df = pandas.DataFrame(results)
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
        row = {
            "model_name": model_name,
            "folds": int(model_df["fold"].nunique()),
        }
        for metric in metric_columns:
            row[metric] = float(model_df[metric].mean())
            std_value = model_df[metric].std(ddof=1)
            row[f"{metric}_std"] = 0.0 if pandas.isna(std_value) else float(std_value)

        distribution = {label: 0 for label in LABELS}
        for fold_distribution in model_df["predicted_class_distribution"]:
            for label in LABELS:
                distribution[label] += int(fold_distribution[label])
        for label in LABELS:
            row[f"pred_{label}"] = distribution[label]

        rows.append(row)

    aggregate_df = pandas.DataFrame(rows)
    aggregate_df = aggregate_df.sort_values(
        ["log_loss", "brier_score", "accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    print("=== Aggregate Metrics ===")
    print(
        aggregate_df[
            [
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
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
                "draw_recall": "{:.4f}".format,
                "draw_precision": "{:.4f}".format,
                "draw_f1": "{:.4f}".format,
            },
        )
    )
    return aggregate_df


def _result_row(results_df: pandas.DataFrame, fold: int, model_name: str) -> pandas.Series:
    rows = results_df.loc[
        (results_df["fold"] == fold) & (results_df["model_name"] == model_name)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for fold={fold}, model={model_name}")
    return rows.iloc[0]


def _aggregate_row(aggregate_df: pandas.DataFrame, model_name: str) -> pandas.Series:
    rows = aggregate_df.loc[aggregate_df["model_name"] == model_name]
    if len(rows) != 1:
        raise ValueError(f"Expected one aggregate row for {model_name}")
    return rows.iloc[0]


def _print_pair_deltas(
    results_df: pandas.DataFrame,
    aggregate_df: pandas.DataFrame,
    base_model: str,
    pressure_model: str,
) -> dict[str, Any]:
    base_agg = _aggregate_row(aggregate_df, base_model)
    pressure_agg = _aggregate_row(aggregate_df, pressure_model)
    aggregate_deltas = {
        metric: float(pressure_agg[metric] - base_agg[metric])
        for metric in [
            "accuracy",
            "log_loss",
            "brier_score",
            "draw_recall",
            "draw_precision",
            "draw_f1",
        ]
    }

    fold_log_loss_deltas: list[float] = []
    improved_folds = 0
    print(f"{pressure_model} vs {base_model}")
    print("Aggregate deltas (pressure minus Elo):")
    print(f"- accuracy: {aggregate_deltas['accuracy']:+.4f} (higher is better)")
    print(f"- log_loss: {aggregate_deltas['log_loss']:+.4f} (lower is better)")
    print(f"- brier_score: {aggregate_deltas['brier_score']:+.4f} (lower is better)")
    print(f"- draw_recall: {aggregate_deltas['draw_recall']:+.4f} (higher is better)")
    print(f"- draw_precision: {aggregate_deltas['draw_precision']:+.4f} (higher is better)")
    print(f"- draw_f1: {aggregate_deltas['draw_f1']:+.4f} (higher is better)")
    print("Fold log_loss deltas:")
    for fold in sorted(results_df["fold"].unique()):
        base_row = _result_row(results_df, int(fold), base_model)
        pressure_row = _result_row(results_df, int(fold), pressure_model)
        delta = float(pressure_row["log_loss"] - base_row["log_loss"])
        fold_log_loss_deltas.append(delta)
        if delta < 0:
            improved_folds += 1
        print(f"- Fold {int(fold)}: {delta:+.4f}")

    return {
        "aggregate_deltas": aggregate_deltas,
        "fold_log_loss_deltas": fold_log_loss_deltas,
        "improved_folds": improved_folds,
        "max_fold_log_loss_delta": max(fold_log_loss_deltas),
    }


def print_comparison_and_verdict(results, redundancy_df) -> None:
    results_df = pandas.DataFrame(results)
    aggregate_df = aggregate_results(results_df)

    print("=== Pressure vs Elo Deltas ===")
    logistic_deltas = _print_pair_deltas(
        results_df,
        aggregate_df,
        "logistic_elo",
        "logistic_elo_pressure",
    )
    xgb_deltas = _print_pair_deltas(
        results_df,
        aggregate_df,
        "xgb_elo",
        "xgb_elo_pressure",
    )

    flagged_pressure_count = 0
    if not redundancy_df.empty:
        flagged_pressure_count = int(
            redundancy_df.loc[
                redundancy_df["likely_redundant"],
                "pressure_feature",
            ].nunique()
        )
    mostly_redundant = flagged_pressure_count >= max(1, len(PRESSURE_FEATURE_COLUMNS) // 2)

    logistic_agg = logistic_deltas["aggregate_deltas"]
    logistic_acceptance = {
        "aggregate_log_loss_improved_at_least_0.003": logistic_agg["log_loss"] <= -0.003,
        "aggregate_brier_improved_or_matched_within_0.002": (
            logistic_agg["brier_score"] <= 0.002
        ),
        "no_fold_log_loss_regression_worse_than_0.005": (
            logistic_deltas["max_fold_log_loss_delta"] <= 0.005
        ),
        "accuracy_drop_not_more_than_0.01": logistic_agg["accuracy"] >= -0.01,
        "draw_f1_drop_not_more_than_0.02": logistic_agg["draw_f1"] >= -0.02,
        "log_loss_improved_in_both_folds": logistic_deltas["improved_folds"] == len(FOLDS),
        "pressure_not_mostly_redundant": not mostly_redundant,
    }
    can_promote = all(logistic_acceptance.values())

    xgb_agg = xgb_deltas["aggregate_deltas"]
    xgb_improved = (
        xgb_agg["log_loss"] < 0
        and xgb_agg["brier_score"] <= 0.002
        and xgb_deltas["improved_folds"] == len(FOLDS)
    )
    xgb_pressure_log_loss = float(
        _aggregate_row(aggregate_df, "xgb_elo_pressure")["log_loss"]
    )
    logistic_elo_log_loss = float(_aggregate_row(aggregate_df, "logistic_elo")["log_loss"])

    print("=== Champion Reference ===")
    print(
        "logistic_elo_expanding reference: "
        f"accuracy={CHAMPION_METRICS['accuracy']:.4f}, "
        f"log_loss={CHAMPION_METRICS['log_loss']:.4f}, "
        f"brier_score={CHAMPION_METRICS['brier_score']:.4f}, "
        f"draw_f1={CHAMPION_METRICS['draw_f1']:.4f}"
    )
    print("=== Acceptance Gate Checks ===")
    for check_name, passed in logistic_acceptance.items():
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    if can_promote:
        verdict = "PROMOTE_PRESSURE_TO_CANDIDATE"
        explanation = "logistic_elo_pressure passed all promotion gates."
    elif xgb_improved and xgb_pressure_log_loss >= logistic_elo_log_loss:
        verdict = "KEEP_PRESSURE_EXPERIMENTAL"
        explanation = (
            "xgb_elo_pressure improved its XGB baseline but did not beat logistic_elo."
        )
    elif mostly_redundant:
        verdict = "REJECT_PRESSURE_EXPERIMENT"
        explanation = "pressure features are mostly redundant with existing Elo/rolling form."
    elif logistic_agg["log_loss"] > 0:
        verdict = "REJECT_PRESSURE_EXPERIMENT"
        explanation = "logistic_elo_pressure worsened aggregate log_loss."
    elif logistic_deltas["improved_folds"] < len(FOLDS):
        verdict = "REJECT_PRESSURE_EXPERIMENT"
        explanation = "pressure log_loss improvement did not appear in both folds."
    elif logistic_agg["brier_score"] > 0.002:
        verdict = "REJECT_PRESSURE_EXPERIMENT"
        explanation = "pressure materially worsened aggregate Brier score."
    elif logistic_agg["draw_f1"] < -0.02:
        verdict = "REJECT_PRESSURE_EXPERIMENT"
        explanation = "pressure caused a large aggregate draw F1 loss."
    else:
        verdict = "KEEP_PRESSURE_EXPERIMENTAL"
        explanation = "pressure did not clear all promotion gates."

    print("=== Phase 8B Verdict ===")
    print(f"Verdict: {verdict}")
    print(f"Reason: {explanation}")
    print(
        "Reference champion: logistic_elo here is the expanding-window "
        "development comparison equivalent of logistic_elo_expanding."
    )


def capture_watched_table_counts(conn) -> dict:
    counts: dict[str, int | str] = {}
    for table_name in WATCHED_TABLES:
        if _table_exists(conn, table_name):
            counts[table_name] = _count_table_rows(conn, table_name)
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

    print("=== Tier 3 Phase 8B Pressure Experiment ===")
    conn = get_db_connection()
    validate_historical_match_integrity(conn)

    before_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts before experiment", before_counts)

    df = load_pressure_feature_data(conn)
    validate_no_final_holdout_loaded(df)
    base_elo_features = get_base_elo_feature_columns(df)
    pressure_features = get_pressure_feature_columns()
    _validate_selected_features(df, pressure_features, "pressure")
    feature_sets = {
        "base_elo": base_elo_features,
        "base_elo_pressure": [*base_elo_features, *pressure_features],
    }
    print(f"Base + Elo feature count: {len(feature_sets['base_elo'])}")
    print(f"Pressure feature count: {len(pressure_features)}")
    print(
        "Base + Elo + pressure feature count: "
        f"{len(feature_sets['base_elo_pressure'])}"
    )

    compute_pressure_coverage_diagnostics(df, pressure_features)
    redundancy_df = compute_redundancy_diagnostics(
        df,
        base_elo_features,
        pressure_features,
    )
    models = build_model_specs()

    result_rows: list[dict[str, Any]] = []
    for fold_config in FOLDS:
        result_rows.extend(run_fold(df, fold_config, feature_sets, models))

    print_comparison_and_verdict(result_rows, redundancy_df)

    after_counts = capture_watched_table_counts(conn)
    _print_table_counts("Watched table counts after experiment", after_counts)
    assert_watched_counts_unchanged(before_counts, after_counts)

    print(f"PASS: {FINAL_TEST_SEASON} was not loaded, tuned, evaluated, or reported for model metrics.")
    print("PASS: no random train/test split was used.")
    print("No database writes occurred.")
    print("No model artifacts were saved.")
    print("No Streamlit, Tier 2 artifact, H2H, style, Poisson-feature, betting odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


if __name__ == "__main__":
    main()
