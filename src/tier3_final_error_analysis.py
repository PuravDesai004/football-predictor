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
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_HOLDOUT_SEASON = "2025-26"
TARGET_COLUMN = "result"
LABELS = ["H", "D", "A"]
RANDOM_STATE = 42

REPORT_PATH = PROJECT_ROOT / "docs" / "tier3_final_error_analysis.md"
EXPECTED_TRAIN_ROWS = 1520
EXPECTED_HOLDOUT_ROWS = 380
EXPECTED_FEATURE_COUNT = 32
DOMINANT_CLASS_MAX_PROB = 0.50
THRESHOLD_GRID = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]

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


def load_analysis_data(conn) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    train_df = _load_seasons(conn, TRAIN_SEASONS)
    holdout_df = _load_seasons(conn, [FINAL_HOLDOUT_SEASON])

    train_df["match_date"] = pandas.to_datetime(train_df["match_date"])
    holdout_df["match_date"] = pandas.to_datetime(holdout_df["match_date"])

    errors: list[str] = []
    if len(train_df) != EXPECTED_TRAIN_ROWS:
        errors.append(f"expected {EXPECTED_TRAIN_ROWS} training rows, found {len(train_df)}")
    if len(holdout_df) != EXPECTED_HOLDOUT_ROWS:
        errors.append(
            f"expected {EXPECTED_HOLDOUT_ROWS} holdout rows, found {len(holdout_df)}"
        )
    train_seasons = sorted(train_df["season_id"].dropna().unique().tolist())
    holdout_seasons = sorted(holdout_df["season_id"].dropna().unique().tolist())
    if train_seasons != TRAIN_SEASONS:
        errors.append(f"training seasons {train_seasons} != {TRAIN_SEASONS}")
    if holdout_seasons != [FINAL_HOLDOUT_SEASON]:
        errors.append(f"holdout seasons {holdout_seasons} != {[FINAL_HOLDOUT_SEASON]}")
    if FINAL_HOLDOUT_SEASON in set(train_df["season_id"]):
        errors.append(f"{FINAL_HOLDOUT_SEASON} appeared in training data")
    if train_df["match_date"].max() >= holdout_df["match_date"].min():
        errors.append(
            "date leakage: max training date "
            f"{train_df['match_date'].max().date()} >= "
            f"min holdout date {holdout_df['match_date'].min().date()}"
        )
    if errors:
        raise ValueError("Analysis data validation failed: " + "; ".join(errors))

    return train_df, holdout_df


def get_feature_columns(df) -> list[str]:
    feature_columns = BASE_ELO_FEATURE_COLUMNS.copy()
    missing_columns = sorted(set(feature_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing frozen feature column(s): {missing_columns}")
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
            "Frozen feature audit failed: "
            f"forbidden_exact={forbidden_exact}, "
            f"forbidden_tokens={forbidden_tokens}, "
            f"non_numeric={non_numeric_features}"
        )

    return feature_columns


def reproduce_final_predictions(
    train_df,
    holdout_df,
    feature_columns,
) -> pandas.DataFrame:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(
        train_df[feature_columns].copy(),
        _encode_labels(train_df[TARGET_COLUMN], LABELS),
    )

    train_probabilities = _predict_probabilities(model, train_df[feature_columns].copy())
    threshold_config = _select_draw_threshold_from_training(
        train_probabilities,
        train_df[TARGET_COLUMN].tolist(),
    )
    selected_draw_threshold = float(threshold_config["selected_draw_threshold"])

    holdout_probabilities = _predict_probabilities(model, holdout_df[feature_columns].copy())
    argmax_predictions = _predict_argmax(holdout_probabilities)
    overlay_predictions = _apply_draw_overlay(
        holdout_probabilities,
        selected_draw_threshold,
    )

    pred_df = holdout_df.copy().reset_index(drop=True)
    pred_df["match_order"] = numpy.arange(1, len(pred_df) + 1)
    pred_df["p_home"] = holdout_probabilities[:, LABELS.index("H")]
    pred_df["p_draw"] = holdout_probabilities[:, LABELS.index("D")]
    pred_df["p_away"] = holdout_probabilities[:, LABELS.index("A")]
    pred_df["max_probability"] = holdout_probabilities.max(axis=1)
    pred_df["argmax_prediction"] = argmax_predictions
    pred_df["overlay_prediction"] = overlay_predictions
    pred_df["argmax_correct"] = pred_df["argmax_prediction"] == pred_df[TARGET_COLUMN]
    pred_df["overlay_correct"] = pred_df["overlay_prediction"] == pred_df[TARGET_COLUMN]
    pred_df["individual_log_loss"] = _individual_log_loss(
        pred_df[TARGET_COLUMN].tolist(),
        holdout_probabilities,
    )
    pred_df["final_score"] = pred_df.apply(_format_score, axis=1)
    pred_df.attrs["selected_draw_threshold"] = selected_draw_threshold
    pred_df.attrs["threshold_config"] = threshold_config
    pred_df.attrs["feature_count"] = len(feature_columns)
    return pred_df


def analyze_confusion(pred_df) -> dict:
    return {
        "argmax": _evaluate_mode(pred_df, "argmax_prediction"),
        "overlay": _evaluate_mode(pred_df, "overlay_prediction"),
        "selected_draw_threshold": pred_df.attrs["selected_draw_threshold"],
        "feature_count": pred_df.attrs["feature_count"],
    }


def analyze_draw_failures(pred_df) -> dict:
    actual_draws = pred_df[TARGET_COLUMN] == "D"
    draw_bins = _bin_table(
        pred_df,
        value_column="p_draw",
        bins=[0.00, 0.15, 0.20, 0.25, 0.30, 0.35, numpy.inf],
        labels=["0.00-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.30", "0.30-0.35", "0.35+"],
        extra_columns={
            "actual_draw_count": lambda frame: int((frame[TARGET_COLUMN] == "D").sum()),
            "actual_draw_rate": lambda frame: _safe_mean(frame[TARGET_COLUMN] == "D"),
            "argmax_draw_count": lambda frame: int((frame["argmax_prediction"] == "D").sum()),
            "overlay_draw_count": lambda frame: int(
                (frame["overlay_prediction"] == "D").sum()
            ),
        },
    )
    return {
        "actual_draw_count": int(actual_draws.sum()),
        "argmax_predicted_draw_count": int((pred_df["argmax_prediction"] == "D").sum()),
        "overlay_predicted_draw_count": int((pred_df["overlay_prediction"] == "D").sum()),
        "argmax_draw_miss_count": int(
            ((pred_df[TARGET_COLUMN] == "D") & (pred_df["argmax_prediction"] != "D")).sum()
        ),
        "overlay_draw_miss_count": int(
            ((pred_df[TARGET_COLUMN] == "D") & (pred_df["overlay_prediction"] != "D")).sum()
        ),
        "average_p_draw_for_actual_draws": float(pred_df.loc[actual_draws, "p_draw"].mean()),
        "average_p_draw_for_non_draws": float(pred_df.loc[~actual_draws, "p_draw"].mean()),
        "p_draw_bins": draw_bins,
    }


def analyze_confidence(pred_df) -> dict:
    confidence_bins = _bin_table(
        pred_df,
        value_column="max_probability",
        bins=[0.00, 0.40, 0.50, 0.60, 0.70, numpy.inf],
        labels=["0.00-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70+"],
        extra_columns={
            "accuracy": lambda frame: _safe_mean(frame["argmax_correct"]),
            "mean_log_loss": lambda frame: float(frame["individual_log_loss"].mean()),
            "total_log_loss": lambda frame: float(frame["individual_log_loss"].sum()),
            "wrong_count": lambda frame: int((~frame["argmax_correct"]).sum()),
        },
    )
    total_log_loss = float(pred_df["individual_log_loss"].sum())
    if total_log_loss > 0:
        confidence_bins["log_loss_share"] = (
            confidence_bins["total_log_loss"] / total_log_loss
        )
    else:
        confidence_bins["log_loss_share"] = 0.0

    high_conf_wrong = pred_df.loc[
        (pred_df["max_probability"] >= 0.60) & (~pred_df["argmax_correct"])
    ]
    return {
        "confidence_bins": confidence_bins,
        "high_confidence_wrong_count": int(len(high_conf_wrong)),
        "high_confidence_wrong_rate": float(len(high_conf_wrong) / len(pred_df)),
    }


def analyze_home_away_bias(pred_df) -> dict:
    actual_distribution = _distribution(pred_df[TARGET_COLUMN])
    argmax_distribution = _distribution(pred_df["argmax_prediction"])
    overlay_distribution = _distribution(pred_df["overlay_prediction"])
    prediction_accuracy = []
    for label in LABELS:
        argmax_subset = pred_df.loc[pred_df["argmax_prediction"] == label]
        overlay_subset = pred_df.loc[pred_df["overlay_prediction"] == label]
        prediction_accuracy.append(
            {
                "prediction": label,
                "argmax_count": int(len(argmax_subset)),
                "argmax_accuracy_when_predicted": _safe_mean(
                    argmax_subset["argmax_correct"]
                ),
                "overlay_count": int(len(overlay_subset)),
                "overlay_accuracy_when_predicted": _safe_mean(
                    overlay_subset["overlay_correct"]
                ),
            }
        )

    actual_home_rate = actual_distribution["H"] / len(pred_df)
    argmax_home_rate = argmax_distribution["H"] / len(pred_df)
    overlay_home_rate = overlay_distribution["H"] / len(pred_df)
    return {
        "actual_distribution": actual_distribution,
        "argmax_distribution": argmax_distribution,
        "overlay_distribution": overlay_distribution,
        "prediction_accuracy": pandas.DataFrame(prediction_accuracy),
        "argmax_home_overprediction_count": int(
            argmax_distribution["H"] - actual_distribution["H"]
        ),
        "overlay_home_overprediction_count": int(
            overlay_distribution["H"] - actual_distribution["H"]
        ),
        "argmax_home_rate_gap": float(argmax_home_rate - actual_home_rate),
        "overlay_home_rate_gap": float(overlay_home_rate - actual_home_rate),
        "argmax_overpredicted_home_wins": bool(argmax_home_rate > actual_home_rate),
        "overlay_overpredicted_home_wins": bool(overlay_home_rate > actual_home_rate),
    }


def analyze_elo_gap(pred_df) -> dict:
    elo_bins = _bin_table(
        pred_df,
        value_column="elo_diff_home_adjusted",
        bins=[-numpy.inf, -100.0, -25.0, 25.0, 100.0, numpy.inf],
        labels=[
            "strong away edge",
            "slight away edge",
            "balanced",
            "slight home edge",
            "strong home edge",
        ],
        extra_columns={
            "accuracy": lambda frame: _safe_mean(frame["argmax_correct"]),
            "log_loss": lambda frame: float(
                log_loss(
                    _encode_labels(frame[TARGET_COLUMN], LABELS),
                    frame[["p_home", "p_draw", "p_away"]].to_numpy(),
                    labels=list(range(len(LABELS))),
                )
            ),
            "actual_home_rate": lambda frame: _safe_mean(frame[TARGET_COLUMN] == "H"),
            "argmax_home_rate": lambda frame: _safe_mean(
                frame["argmax_prediction"] == "H"
            ),
        },
    )
    big_favorites = pred_df.loc[pred_df["elo_diff_home_adjusted"].abs() >= 100.0]
    return {
        "elo_bins": elo_bins,
        "big_favorite_match_count": int(len(big_favorites)),
        "big_favorite_wrong_count": int((~big_favorites["argmax_correct"]).sum()),
        "big_favorite_wrong_rate": _safe_mean(~big_favorites["argmax_correct"]),
    }


def analyze_season_stage(pred_df) -> dict:
    staged = pred_df.copy().sort_values(["match_date", "kickoff_time", "match_id"])
    n_rows = len(staged)
    positions = numpy.arange(n_rows)
    staged["season_stage"] = numpy.where(
        positions < n_rows / 3,
        "early",
        numpy.where(positions < 2 * n_rows / 3, "middle", "late"),
    )

    rows = []
    for stage in ["early", "middle", "late"]:
        frame = staged.loc[staged["season_stage"] == stage]
        probabilities = frame[["p_home", "p_draw", "p_away"]].to_numpy()
        metrics = _evaluate_mode(frame, "argmax_prediction")
        rows.append(
            {
                "stage": stage,
                "count": int(len(frame)),
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "draw_f1": metrics["class_metrics"].loc[
                    metrics["class_metrics"]["class"] == "D", "f1"
                ].iloc[0],
                "actual_draw_rate": _safe_mean(frame[TARGET_COLUMN] == "D"),
                "argmax_draw_rate": _safe_mean(frame["argmax_prediction"] == "D"),
                "overlay_draw_rate": _safe_mean(frame["overlay_prediction"] == "D"),
                "mean_p_draw": float(probabilities[:, LABELS.index("D")].mean()),
            }
        )
    return {"stage_metrics": pandas.DataFrame(rows)}


def analyze_team_errors(pred_df) -> dict:
    teams = sorted(set(pred_df["home_team"]) | set(pred_df["away_team"]))
    rows = []
    for team in teams:
        involved = pred_df.loc[
            (pred_df["home_team"] == team) | (pred_df["away_team"] == team)
        ]
        draw_misses = involved.loc[
            (involved[TARGET_COLUMN] == "D") & (involved["argmax_prediction"] != "D")
        ]
        rows.append(
            {
                "team": team,
                "matches": int(len(involved)),
                "argmax_accuracy": _safe_mean(involved["argmax_correct"]),
                "overlay_accuracy": _safe_mean(involved["overlay_correct"]),
                "draw_miss_count": int(len(draw_misses)),
                "mean_log_loss": float(involved["individual_log_loss"].mean()),
            }
        )

    team_df = pandas.DataFrame(rows).sort_values(
        ["argmax_accuracy", "mean_log_loss", "team"],
        ascending=[True, False, True],
    )

    overpredicted_winners = {team: 0 for team in teams}
    for row in pred_df.itertuples(index=False):
        predicted_winner = _winner_from_result(
            row.argmax_prediction,
            row.home_team,
            row.away_team,
        )
        actual_winner = _winner_from_result(row.result, row.home_team, row.away_team)
        if predicted_winner is not None and predicted_winner != actual_winner:
            overpredicted_winners[predicted_winner] += 1

    overpredicted_df = pandas.DataFrame(
        [
            {"team": team, "overpredicted_as_winner_count": count}
            for team, count in overpredicted_winners.items()
        ]
    ).sort_values(["overpredicted_as_winner_count", "team"], ascending=[False, True])

    return {
        "team_accuracy": team_df.sort_values("team").reset_index(drop=True),
        "worst_10": team_df.head(10).reset_index(drop=True),
        "best_10": team_df.sort_values(
            ["argmax_accuracy", "mean_log_loss", "team"],
            ascending=[False, True, True],
        )
        .head(10)
        .reset_index(drop=True),
        "draw_miss_teams": team_df.sort_values(
            ["draw_miss_count", "team"], ascending=[False, True]
        )
        .head(10)
        .reset_index(drop=True),
        "overpredicted_winners": overpredicted_df.head(10).reset_index(drop=True),
    }


def analyze_promoted_teams(conn, pred_df) -> dict:
    query = text(
        """
        SELECT DISTINCT team
        FROM (
            SELECT home_team AS team, home_initialization AS initialization
            FROM elo_ratings_v3
            WHERE season_id = :season_id
            UNION ALL
            SELECT away_team AS team, away_initialization AS initialization
            FROM elo_ratings_v3
            WHERE season_id = :season_id
        ) initialization_rows
        WHERE initialization = 'promoted_or_returning'
        ORDER BY team
        """
    )
    with conn.connect() as db_conn:
        promoted_teams = [
            row["team"]
            for row in db_conn.execute(
                query,
                {"season_id": FINAL_HOLDOUT_SEASON},
            ).mappings().all()
        ]

    frame = pred_df.copy()
    frame["involves_promoted_or_returning"] = (
        frame["home_team"].isin(promoted_teams) | frame["away_team"].isin(promoted_teams)
    )
    rows = []
    for label, subset in [
        ("involving promoted_or_returning", frame.loc[frame["involves_promoted_or_returning"]]),
        ("not involving promoted_or_returning", frame.loc[~frame["involves_promoted_or_returning"]]),
    ]:
        if subset.empty:
            rows.append(
                {
                    "segment": label,
                    "count": 0,
                    "accuracy": numpy.nan,
                    "log_loss": numpy.nan,
                    "brier_score": numpy.nan,
                    "draw_rate": numpy.nan,
                }
            )
            continue
        rows.append(
            {
                "segment": label,
                "count": int(len(subset)),
                "accuracy": _safe_mean(subset["argmax_correct"]),
                "log_loss": float(
                    log_loss(
                        _encode_labels(subset[TARGET_COLUMN], LABELS),
                        subset[["p_home", "p_draw", "p_away"]].to_numpy(),
                        labels=list(range(len(LABELS))),
                    )
                ),
                "brier_score": _multiclass_brier_score(
                    _encode_labels(subset[TARGET_COLUMN], LABELS),
                    subset[["p_home", "p_draw", "p_away"]].to_numpy(),
                ),
                "draw_rate": _safe_mean(subset[TARGET_COLUMN] == "D"),
            }
        )
    return {
        "promoted_or_returning_teams": promoted_teams,
        "comparison": pandas.DataFrame(rows),
    }


def analyze_biggest_logloss_errors(pred_df) -> pandas.DataFrame:
    columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        TARGET_COLUMN,
        "argmax_prediction",
        "p_home",
        "p_draw",
        "p_away",
        "overlay_prediction",
        "final_score",
        "individual_log_loss",
    ]
    display_df = pred_df.sort_values(
        "individual_log_loss",
        ascending=False,
    )[columns].head(20)
    display_df = display_df.rename(
        columns={
            TARGET_COLUMN: "actual_result",
            "argmax_prediction": "predicted_argmax",
            "p_home": "P(H)",
            "p_draw": "P(D)",
            "p_away": "P(A)",
            "overlay_prediction": "overlay_prediction",
        }
    )
    display_df["match_date"] = pandas.to_datetime(display_df["match_date"]).dt.date
    return display_df.reset_index(drop=True)


def write_markdown_report(analysis_results) -> None:
    confusion = analysis_results["confusion"]
    draw = analysis_results["draw_failures"]
    confidence = analysis_results["confidence"]
    bias = analysis_results["home_away_bias"]
    elo_gap = analysis_results["elo_gap"]
    stages = analysis_results["season_stage"]
    team_errors = analysis_results["team_errors"]
    promoted = analysis_results["promoted_teams"]
    biggest_errors = analysis_results["biggest_logloss_errors"]
    before_counts = analysis_results["before_counts"]
    after_counts = analysis_results["after_counts"]

    diagnosis = _build_summary_diagnosis(analysis_results)
    lines: list[str] = [
        "# Tier 3 Final Holdout Error Analysis",
        "",
        "## Status",
        "",
        "Phase 11A is post-final error analysis only. It inspects the already-opened `2025-26` final holdout to explain the frozen model result. It does not tune features, tune hyperparameters, tune draw thresholds, run competing models, save artifacts, or write to the database.",
        "",
        "## Reproduced Frozen Setup",
        "",
        f"- Source table: `{SOURCE_TABLE}`",
        f"- Training seasons: {', '.join(f'`{season}`' for season in TRAIN_SEASONS)}",
        f"- Holdout season: `{FINAL_HOLDOUT_SEASON}`",
        f"- Feature count: {confusion['feature_count']}",
        f"- Draw threshold selected from training only: {confusion['selected_draw_threshold']:.2f}",
        "- Model path: frozen `logistic_elo_expanding` only",
        "- Official performance is not changed by this analysis",
        "",
        "## Confusion Matrix",
        "",
        "Argmax confusion matrix:",
        "",
        _df_to_markdown(_confusion_to_df(confusion["argmax"]["confusion_counts"])),
        "",
        "Draw overlay confusion matrix:",
        "",
        _df_to_markdown(_confusion_to_df(confusion["overlay"]["confusion_counts"])),
        "",
        "Argmax class metrics:",
        "",
        _df_to_markdown(confusion["argmax"]["class_metrics"]),
        "",
        "Draw overlay class metrics:",
        "",
        _df_to_markdown(confusion["overlay"]["class_metrics"]),
        "",
        "## Draw Failure Analysis",
        "",
        f"- Actual draw count: {draw['actual_draw_count']}",
        f"- Argmax predicted draw count: {draw['argmax_predicted_draw_count']}",
        f"- Draw overlay predicted draw count: {draw['overlay_predicted_draw_count']}",
        f"- Argmax draw miss count: {draw['argmax_draw_miss_count']}",
        f"- Draw overlay draw miss count: {draw['overlay_draw_miss_count']}",
        f"- Average `P(D)` for actual draws: {draw['average_p_draw_for_actual_draws']:.4f}",
        f"- Average `P(D)` for non-draws: {draw['average_p_draw_for_non_draws']:.4f}",
        "",
        "`P(D)` distribution:",
        "",
        _df_to_markdown(draw["p_draw_bins"]),
        "",
        "## Confidence Analysis",
        "",
        _df_to_markdown(confidence["confidence_bins"]),
        "",
        f"- High-confidence wrong predictions with max probability >= 0.60: {confidence['high_confidence_wrong_count']}",
        f"- High-confidence wrong prediction rate: {confidence['high_confidence_wrong_rate']:.4f}",
        "",
        "## Home/Away Bias",
        "",
        "Actual distribution:",
        "",
        _dict_table(bias["actual_distribution"], "result", "count"),
        "",
        "Argmax predicted distribution:",
        "",
        _dict_table(bias["argmax_distribution"], "prediction", "count"),
        "",
        "Draw overlay predicted distribution:",
        "",
        _dict_table(bias["overlay_distribution"], "prediction", "count"),
        "",
        "Accuracy by predicted label:",
        "",
        _df_to_markdown(bias["prediction_accuracy"]),
        "",
        f"- Argmax home overprediction count: {bias['argmax_home_overprediction_count']}",
        f"- Argmax home-rate gap: {bias['argmax_home_rate_gap']:.4f}",
        f"- Overlay home overprediction count: {bias['overlay_home_overprediction_count']}",
        f"- Overlay home-rate gap: {bias['overlay_home_rate_gap']:.4f}",
        "",
        "## Elo Gap Analysis",
        "",
        _df_to_markdown(elo_gap["elo_bins"]),
        "",
        f"- Big Elo favorite match count: {elo_gap['big_favorite_match_count']}",
        f"- Big Elo favorite wrong count: {elo_gap['big_favorite_wrong_count']}",
        f"- Big Elo favorite wrong rate: {elo_gap['big_favorite_wrong_rate']:.4f}",
        "",
        "## Season-Stage Analysis",
        "",
        _df_to_markdown(stages["stage_metrics"]),
        "",
        "## Team-Level Error Analysis",
        "",
        "Worst 10 teams by argmax prediction accuracy:",
        "",
        _df_to_markdown(team_errors["worst_10"]),
        "",
        "Best 10 teams by argmax prediction accuracy:",
        "",
        _df_to_markdown(team_errors["best_10"]),
        "",
        "Teams with the most draw misses:",
        "",
        _df_to_markdown(team_errors["draw_miss_teams"]),
        "",
        "Teams most often overpredicted as winners:",
        "",
        _df_to_markdown(team_errors["overpredicted_winners"]),
        "",
        "## Promoted/Returning Team Analysis",
        "",
        "Promoted or returning teams from `elo_ratings_v3`: "
        + (
            ", ".join(f"`{team}`" for team in promoted["promoted_or_returning_teams"])
            if promoted["promoted_or_returning_teams"]
            else "none"
        ),
        "",
        _df_to_markdown(promoted["comparison"]),
        "",
        "## Biggest Log-Loss Errors",
        "",
        _df_to_markdown(biggest_errors),
        "",
        "## Watched Table Counts",
        "",
        _df_to_markdown(_counts_to_df(before_counts, after_counts)),
        "",
        "## Summary Diagnosis",
        "",
        diagnosis,
        "",
        "## Guardrails",
        "",
        "- Do not claim this analysis improves official Tier 3 performance.",
        "- Do not tune on `2025-26`.",
        "- Any improved future model needs a new untouched future holdout.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis report: {REPORT_PATH}")


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
    print("=== Tier 3 Phase 11A Final Holdout Error Analysis ===")
    print("Analysis only: no tuning, no competing models, no DB writes, no artifacts.")
    conn = get_db_connection()
    before_counts = capture_watched_table_counts(conn)

    train_df, holdout_df = load_analysis_data(conn)
    feature_columns = get_feature_columns(train_df)
    pred_df = reproduce_final_predictions(train_df, holdout_df, feature_columns)

    analysis_results = {
        "confusion": analyze_confusion(pred_df),
        "draw_failures": analyze_draw_failures(pred_df),
        "confidence": analyze_confidence(pred_df),
        "home_away_bias": analyze_home_away_bias(pred_df),
        "elo_gap": analyze_elo_gap(pred_df),
        "season_stage": analyze_season_stage(pred_df),
        "team_errors": analyze_team_errors(pred_df),
        "promoted_teams": analyze_promoted_teams(conn, pred_df),
        "biggest_logloss_errors": analyze_biggest_logloss_errors(pred_df),
        "before_counts": before_counts,
    }

    after_counts = capture_watched_table_counts(conn)
    analysis_results["after_counts"] = after_counts
    assert_no_counts_changed(before_counts, after_counts)
    write_markdown_report(analysis_results)
    _print_console_summary(analysis_results)

    print("PASS: reproduced frozen final evaluation in memory only.")
    print("PASS: no thresholds, features, or hyperparameters were tuned.")
    print("No model artifact was saved.")
    print("No database writes occurred.")
    print("No Streamlit, Tier 2 artifact, H2H, style, pressure, Poisson, odds, manager, sentiment, injury, rivalry, derby, deployment, or app work occurred.")


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


def _predict_probabilities(model, x_values) -> numpy.ndarray:
    raw_probabilities = model.predict_proba(x_values)
    observed_classes = model.named_steps["model"].classes_
    aligned = numpy.zeros((raw_probabilities.shape[0], len(LABELS)), dtype=float)
    for source_index, class_index in enumerate(observed_classes):
        aligned[:, int(class_index)] = raw_probabilities[:, source_index]
    return _normalize_probabilities(aligned)


def _predict_argmax(probabilities) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    prediction_indexes = numpy.argmax(probabilities, axis=1)
    return [LABELS[int(index)] for index in prediction_indexes]


def _apply_draw_overlay(probabilities, draw_threshold: float) -> list[str]:
    probabilities = _normalize_probabilities(probabilities)
    predictions = _predict_argmax(probabilities)
    draw_index = LABELS.index("D")
    overlay_predictions: list[str] = []
    for row_index, probability_row in enumerate(probabilities):
        draw_prob = float(probability_row[draw_index])
        max_prob = float(probability_row.max())
        should_change_to_draw = (
            draw_prob >= draw_threshold
            and _is_draw_second_highest(probability_row)
            and max_prob < DOMINANT_CLASS_MAX_PROB
        )
        overlay_predictions.append("D" if should_change_to_draw else predictions[row_index])
    return overlay_predictions


def _is_draw_second_highest(probability_row) -> bool:
    probabilities = numpy.asarray(probability_row, dtype=float)
    draw_index = LABELS.index("D")
    descending_indexes = numpy.argsort(-probabilities, kind="mergesort")
    return int(descending_indexes[1]) == draw_index


def _select_draw_threshold_from_training(probabilities, y_true) -> dict:
    probabilities = _normalize_probabilities(probabilities)
    argmax_predictions = _predict_argmax(probabilities)
    argmax_metrics = _evaluate_arrays(y_true, argmax_predictions, probabilities)

    best_config: dict[str, Any] | None = None
    for draw_threshold in THRESHOLD_GRID:
        overlay_predictions = _apply_draw_overlay(probabilities, draw_threshold)
        metrics = _evaluate_arrays(y_true, overlay_predictions, probabilities)
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


def _evaluate_mode(pred_df, prediction_column: str) -> dict:
    probabilities = pred_df[["p_home", "p_draw", "p_away"]].to_numpy()
    y_true = pred_df[TARGET_COLUMN].tolist()
    y_pred = pred_df[prediction_column].tolist()
    metrics = _evaluate_arrays(y_true, y_pred, probabilities)
    metrics["confusion_counts"] = _confusion_counts(y_true, y_pred)
    metrics["class_metrics"] = _class_metrics(y_true, y_pred)
    return metrics


def _evaluate_arrays(y_true, y_pred, probabilities) -> dict:
    y_true_encoded = _encode_labels(y_true, LABELS)
    y_pred_encoded = _encode_labels(y_pred, LABELS)
    probabilities = _normalize_probabilities(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=[LABELS.index("D")],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
        "log_loss": float(
            log_loss(y_true_encoded, probabilities, labels=list(range(len(LABELS))))
        ),
        "brier_score": _multiclass_brier_score(y_true_encoded, probabilities),
        "draw_recall": float(recall[0]),
        "draw_precision": float(precision[0]),
        "draw_f1": float(f1[0]),
    }


def _class_metrics(y_true, y_pred) -> pandas.DataFrame:
    y_true_encoded = _encode_labels(y_true, LABELS)
    y_pred_encoded = _encode_labels(y_pred, LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    return pandas.DataFrame(
        [
            {
                "class": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        ]
    )


def _confusion_counts(y_true, y_pred) -> dict[str, dict[str, int]]:
    y_true_encoded = _encode_labels(y_true, LABELS)
    y_pred_encoded = _encode_labels(y_pred, LABELS)
    matrix = confusion_matrix(
        y_true_encoded,
        y_pred_encoded,
        labels=list(range(len(LABELS))),
    )
    return {
        true_label: {
            pred_label: int(matrix[true_index, pred_index])
            for pred_index, pred_label in enumerate(LABELS)
        }
        for true_index, true_label in enumerate(LABELS)
    }


def _confusion_to_df(confusion_counts: dict[str, dict[str, int]]) -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                "actual": true_label,
                "pred_H": confusion_counts[true_label]["H"],
                "pred_D": confusion_counts[true_label]["D"],
                "pred_A": confusion_counts[true_label]["A"],
            }
            for true_label in LABELS
        ]
    )


def _bin_table(
    df: pandas.DataFrame,
    value_column: str,
    bins: list[float],
    labels: list[str],
    extra_columns: dict[str, Any],
) -> pandas.DataFrame:
    working = df.copy()
    working["bin"] = pandas.cut(
        working[value_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows = []
    for label in labels:
        frame = working.loc[working["bin"] == label]
        row: dict[str, Any] = {"bin": label, "count": int(len(frame))}
        for column_name, func in extra_columns.items():
            row[column_name] = numpy.nan if frame.empty else func(frame)
        rows.append(row)
    return pandas.DataFrame(rows)


def _individual_log_loss(y_true, probabilities) -> numpy.ndarray:
    probabilities = numpy.clip(_normalize_probabilities(probabilities), 1e-15, 1.0)
    y_true_encoded = _encode_labels(y_true, LABELS)
    return -numpy.log(probabilities[numpy.arange(len(y_true_encoded)), y_true_encoded])


def _multiclass_brier_score(y_true_encoded, probabilities) -> float:
    probabilities = _normalize_probabilities(probabilities)
    y_one_hot = numpy.zeros((len(y_true_encoded), len(LABELS)), dtype=float)
    y_one_hot[numpy.arange(len(y_true_encoded)), y_true_encoded] = 1.0
    return float(numpy.mean(numpy.sum((probabilities - y_one_hot) ** 2, axis=1)))


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


def _distribution(values) -> dict[str, int]:
    series = pandas.Series(values)
    return {label: int((series == label).sum()) for label in LABELS}


def _safe_mean(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pandas.Series(values).mean())


def _format_score(row) -> str:
    if "home_goals" not in row.index or "away_goals" not in row.index:
        return ""
    if pandas.isna(row["home_goals"]) or pandas.isna(row["away_goals"]):
        return ""
    return f"{int(row['home_goals'])}-{int(row['away_goals'])}"


def _winner_from_result(result: str, home_team: str, away_team: str) -> str | None:
    if result == "H":
        return home_team
    if result == "A":
        return away_team
    return None


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


def _counts_to_df(before_counts: dict, after_counts: dict) -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                "table": table_name,
                "before": before_counts.get(table_name),
                "after": after_counts.get(table_name),
                "changed": before_counts.get(table_name) != after_counts.get(table_name),
            }
            for table_name in WATCHED_TABLES
        ]
    )


def _dict_table(values: dict[str, Any], key_name: str, value_name: str) -> str:
    return _df_to_markdown(
        pandas.DataFrame(
            [{key_name: key, value_name: value} for key, value in values.items()]
        )
    )


def _df_to_markdown(df: pandas.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    rows = [[_format_markdown_value(row[column]) for column in columns] for _, row in df.iterrows()]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_markdown_value(value: Any) -> str:
    if pandas.isna(value):
        return ""
    if isinstance(value, (float, numpy.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, (int, numpy.integer)):
        return str(int(value))
    return str(value)


def _build_summary_diagnosis(analysis_results) -> str:
    confusion = analysis_results["confusion"]
    draw = analysis_results["draw_failures"]
    confidence = analysis_results["confidence"]
    bias = analysis_results["home_away_bias"]
    elo_gap = analysis_results["elo_gap"]
    promoted = analysis_results["promoted_teams"]
    stage_df = analysis_results["season_stage"]["stage_metrics"]

    argmax_accuracy = confusion["argmax"]["accuracy"]
    argmax_log_loss = confusion["argmax"]["log_loss"]
    draw_miss_rate = draw["argmax_draw_miss_count"] / max(draw["actual_draw_count"], 1)
    home_gap = bias["argmax_home_rate_gap"]
    high_conf_wrong = confidence["high_confidence_wrong_count"]
    worst_stage = stage_df.sort_values("accuracy").iloc[0]
    promoted_comparison = promoted["comparison"]
    promoted_row = promoted_comparison.loc[
        promoted_comparison["segment"] == "involving promoted_or_returning"
    ]
    promoted_text = "No promoted_or_returning teams were identified."
    if not promoted_row.empty:
        row = promoted_row.iloc[0]
        promoted_text = (
            "Matches involving promoted_or_returning teams had "
            f"{int(row['count'])} rows, {row['accuracy']:.4f} accuracy, and "
            f"{row['log_loss']:.4f} log loss."
        )

    return "\n".join(
        [
            f"- Reproduced argmax accuracy was {argmax_accuracy:.4f} with log_loss {argmax_log_loss:.4f}.",
            f"- Draw underprediction is the main issue: {draw['actual_draw_count']} actual draws, {draw['argmax_predicted_draw_count']} argmax predicted draws, and {draw['argmax_draw_miss_count']} argmax draw misses ({draw_miss_rate:.4f} of actual draws).",
            f"- Home bias is severe: argmax predicted {bias['argmax_distribution']['H']} home wins against {bias['actual_distribution']['H']} actual home wins, a home-rate gap of {home_gap:.4f}.",
            f"- Confidence was not reliable enough in the top bins: {high_conf_wrong} wrong predictions had max probability >= 0.60.",
            f"- Big Elo favorites were not safe: {elo_gap['big_favorite_wrong_count']} of {elo_gap['big_favorite_match_count']} matches with absolute adjusted Elo gap >= 100 were wrong.",
            f"- The weakest season stage by accuracy was {worst_stage['stage']} at {worst_stage['accuracy']:.4f}.",
            f"- {promoted_text}",
            "- Future work should focus on more seasons, better pre-match draw modeling, calibration/decision-rule research on development data only, and reliable dated team-availability sources. Any improved model needs a new untouched future holdout.",
        ]
    )


def _print_console_summary(analysis_results) -> None:
    confusion = analysis_results["confusion"]
    draw = analysis_results["draw_failures"]
    confidence = analysis_results["confidence"]
    bias = analysis_results["home_away_bias"]
    promoted = analysis_results["promoted_teams"]
    biggest = analysis_results["biggest_logloss_errors"]

    print("=== Phase 11A Error Analysis Summary ===")
    print(
        f"Argmax: accuracy={confusion['argmax']['accuracy']:.4f}, "
        f"log_loss={confusion['argmax']['log_loss']:.4f}, "
        f"brier={confusion['argmax']['brier_score']:.4f}"
    )
    print(
        f"Overlay: accuracy={confusion['overlay']['accuracy']:.4f}, "
        f"log_loss={confusion['overlay']['log_loss']:.4f}, "
        f"brier={confusion['overlay']['brier_score']:.4f}"
    )
    print(
        "Draws: "
        f"actual={draw['actual_draw_count']}, "
        f"argmax_predicted={draw['argmax_predicted_draw_count']}, "
        f"overlay_predicted={draw['overlay_predicted_draw_count']}, "
        f"argmax_missed={draw['argmax_draw_miss_count']}"
    )
    print(
        "Home bias: "
        f"actual_H={bias['actual_distribution']['H']}, "
        f"argmax_pred_H={bias['argmax_distribution']['H']}, "
        f"home_rate_gap={bias['argmax_home_rate_gap']:.4f}"
    )
    print(
        "Confidence: "
        f"high_conf_wrong_count={confidence['high_confidence_wrong_count']} "
        "(max probability >= 0.60)"
    )
    print(
        "Promoted/returning teams: "
        + (
            ", ".join(promoted["promoted_or_returning_teams"])
            if promoted["promoted_or_returning_teams"]
            else "none"
        )
    )
    print("Top log-loss error:")
    top = biggest.iloc[0]
    print(
        f"- {top['match_date']} {top['home_team']} vs {top['away_team']} "
        f"{top['final_score']}: actual={top['actual_result']}, "
        f"argmax={top['predicted_argmax']}, "
        f"P(H)={top['P(H)']:.4f}, P(D)={top['P(D)']:.4f}, "
        f"P(A)={top['P(A)']:.4f}, log_loss={top['individual_log_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
