import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# WHY TIME-BASED SPLIT:
# Football data is time-ordered. Using random split would let the model
# train on GW30 matches and predict GW20 by using future data to predict
# the past. We always train on earlier gameweeks and test on later ones.
#
# WHY H2H FEATURES REMOVED:
# With only one season of data, H2H stats are calculated from the same
# matches being predicted. That is data leakage. H2H will be added back
# properly in Tier 3 when multiple historical seasons are available.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
warnings.filterwarnings("ignore")

from src.model_validation import (
    evaluate_timeseries_cv,
    make_gameweek_split,
    sort_by_time,
    validate_no_forbidden_features,
    validate_required_features,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FEATURES_PATH = MODELS_DIR / "model_features.json"
TIME_SERIES_CV_FOLDS = 5

BASE_FEATURES_12 = [
    "home_form_scored",
    "home_form_conceded",
    "home_clean_sheet_rate",
    "away_form_scored",
    "away_form_conceded",
    "away_clean_sheet_rate",
    "home_fdr",
    "away_fdr",
    "strength_overall_home",
    "strength_overall_away",
    "home_team_away_str",
    "away_team_home_str",
]

XG_FEATURES_4 = [
    "home_xg_last5",
    "home_xga_last5",
    "away_xg_last5",
    "away_xga_last5",
]

MODEL_FEATURES_16 = BASE_FEATURES_12 + XG_FEATURES_4

STYLE_FEATURES_4 = [
    "home_style_cluster",
    "away_style_cluster",
    "home_style_matches_last5",
    "away_style_matches_last5",
]

MODEL_FEATURES_STYLE = MODEL_FEATURES_16 + STYLE_FEATURES_4

FORBIDDEN_LEAKAGE_COLUMNS = [
    "h2h_matches",
    "h2h_home_win_rate",
    "h2h_avg_home_goals",
    "h2h_avg_away_goals",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
]


# Creates a fresh Logistic Regression model for each evaluation run.
def make_logistic_model():
    return LogisticRegression(max_iter=1000, random_state=42)


# Creates a fresh XGBoost classifier for each evaluation run.
def make_xgb_classifier():
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )


# Creates a fresh XGBoost regressor for goal prediction.
def make_xgb_regressor():
    return XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )


# Prepares match_features for all model comparison groups.
def prepare_features(df):
    try:
        print("=== Pre-flight feature checks ===")

        feature_groups = {
            "Baseline": BASE_FEATURES_12,
            "xG": MODEL_FEATURES_16,
            "xG + style": MODEL_FEATURES_STYLE,
        }
        selected_features = MODEL_FEATURES_STYLE
        validate_no_forbidden_features(selected_features, FORBIDDEN_LEAKAGE_COLUMNS)

        for group_name, feature_list in feature_groups.items():
            validate_required_features(df, feature_list)

        required_target_columns = [
            "result",
            "home_goals",
            "away_goals",
            "gameweek",
            "fixture_id",
        ]
        validate_required_features(df, required_target_columns)

        df = df.dropna(subset=["home_form_scored", "away_form_scored"]).copy()
        df = sort_by_time(df)
        df[MODEL_FEATURES_STYLE] = df[MODEL_FEATURES_STYLE].fillna(0)
        train_df, test_df = make_gameweek_split(df)

        print(f"Baseline feature count: {len(BASE_FEATURES_12)}")
        print(f"xG feature count: {len(MODEL_FEATURES_16)}")
        print(f"xG + style feature count: {len(MODEL_FEATURES_STYLE)}")
        print(f"Class distribution:\n{df['result'].value_counts()}")

        return train_df, test_df
    except Exception as error:
        print(f"Error preparing features: {error}")
        return None, None


# Trains and evaluates fresh Logistic Regression and XGBoost classifiers for a feature set.
def train_classifier_variant(train_df, test_df, feature_cols, label):
    try:
        x_train = train_df[feature_cols].copy()
        x_test = test_df[feature_cols].copy()

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["result"])
        unseen_test_labels = sorted(set(test_df["result"]) - set(label_encoder.classes_))
        if unseen_test_labels:
            raise ValueError(
                "Test set contains labels absent from training set: "
                f"{unseen_test_labels}"
            )
        y_train = pd.Series(label_encoder.transform(train_df["result"]))
        y_test = pd.Series(label_encoder.transform(test_df["result"]))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        lr_model = make_logistic_model()
        lr_model.fit(x_train_scaled, y_train)

        xgb_model = make_xgb_classifier()
        xgb_model.fit(x_train, y_train)

        lr_cv_mean, lr_cv_std, _ = evaluate_timeseries_cv(
            lambda: make_pipeline(StandardScaler(), make_logistic_model()),
            x_train,
            y_train,
            scorer_name="accuracy",
            n_splits=TIME_SERIES_CV_FOLDS,
        )
        xgb_cv_mean, xgb_cv_std, _ = evaluate_timeseries_cv(
            make_xgb_classifier,
            x_train,
            y_train,
            scorer_name="accuracy",
            n_splits=TIME_SERIES_CV_FOLDS,
        )

        lr_test_pred = lr_model.predict(x_test_scaled)
        xgb_test_pred = xgb_model.predict(x_test)
        lr_test_acc = accuracy_score(y_test, lr_test_pred)
        xgb_test_acc = accuracy_score(y_test, xgb_test_pred)

        print(f"Logistic Regression test accuracy for {label}: {lr_test_acc:.3f}")
        print(
            f"Logistic Regression TimeSeries CV for {label}: "
            f"{lr_cv_mean:.3f} +/- {lr_cv_std:.3f}"
        )
        print(f"XGBoost Classifier test accuracy for {label}: {xgb_test_acc:.3f}")
        print(
            f"XGBoost Classifier TimeSeries CV for {label}: "
            f"{xgb_cv_mean:.3f} +/- {xgb_cv_std:.3f}"
        )

        return {
            "lr_model": lr_model,
            "xgb_model": xgb_model,
            "label_encoder": label_encoder,
            "scaler": scaler,
            "x_test": x_test,
            "y_test": y_test,
            "lr_test_acc": lr_test_acc,
            "xgb_test_acc": xgb_test_acc,
            "lr_cv_mean": lr_cv_mean,
            "lr_cv_std": lr_cv_std,
            "xgb_cv_mean": xgb_cv_mean,
            "xgb_cv_std": xgb_cv_std,
        }
    except Exception as error:
        print(f"Error training classifier variant {label}: {error}")
        return None


# Trains fresh XGBoost regressors using the same complete-gameweek split as classifiers.
def train_score_predictor(train_df, test_df, feature_cols, label):
    try:
        x_train = train_df[feature_cols].copy()
        x_test = test_df[feature_cols].copy()
        y_home_train = train_df["home_goals"].reset_index(drop=True)
        y_home_test = test_df["home_goals"].reset_index(drop=True)
        y_away_train = train_df["away_goals"].reset_index(drop=True)
        y_away_test = test_df["away_goals"].reset_index(drop=True)

        home_model = make_xgb_regressor()
        home_model.fit(x_train, y_home_train)

        away_model = make_xgb_regressor()
        away_model.fit(x_train, y_away_train)

        home_pred = home_model.predict(x_test)
        away_pred = away_model.predict(x_test)
        home_pred = np.clip(np.round(home_pred), 0, None)
        away_pred = np.clip(np.round(away_pred), 0, None)

        home_mae = mean_absolute_error(y_home_test, home_pred)
        home_r2 = r2_score(y_home_test, home_pred)
        away_mae = mean_absolute_error(y_away_test, away_pred)
        away_r2 = r2_score(y_away_test, away_pred)

        print(f"Home Goals MAE/R2 with {label}: {home_mae:.3f}, {home_r2:.3f}")
        print(f"Away Goals MAE/R2 with {label}: {away_mae:.3f}, {away_r2:.3f}")

        return {
            "home_model": home_model,
            "away_model": away_model,
            "home_mae": home_mae,
            "home_r2": home_r2,
            "away_mae": away_mae,
            "away_r2": away_r2,
        }
    except Exception as error:
        print(f"Error training score predictor for {label}: {error}")
        return None


# Applies a conservative rule to decide whether style features are stable enough to keep.
def decide_style_features(xg_results, style_results, xg_score_results, style_score_results):
    print("=== Style feature decision ===")

    xgb_holdout_delta = style_results["xgb_test_acc"] - xg_results["xgb_test_acc"]
    xgb_cv_delta = style_results["xgb_cv_mean"] - xg_results["xgb_cv_mean"]
    home_r2_delta = style_score_results["home_r2"] - xg_score_results["home_r2"]
    away_r2_delta = style_score_results["away_r2"] - xg_score_results["away_r2"]

    print(f"XGBoost holdout delta: {xgb_holdout_delta:.3f}")
    print(f"XGBoost TimeSeries CV delta: {xgb_cv_delta:.3f}")
    print(f"Home goal R2 delta: {home_r2_delta:.3f}")
    print(f"Away goal R2 delta: {away_r2_delta:.3f}")

    cv_dropped_lot = xgb_cv_delta < -0.03
    meaningful_improvement = max(xgb_holdout_delta, xgb_cv_delta) >= 0.01
    harmful_goal_r2 = home_r2_delta < -0.05 or away_r2_delta < -0.05

    if harmful_goal_r2:
        print(
            "DO NOT KEEP STYLE FEATURES: no meaningful improvement or unstable metrics."
        )
        print("Reason: goal R2 worsened by more than 0.05.")
        return False

    if xgb_holdout_delta >= 0.01 and cv_dropped_lot:
        print(
            "DO NOT KEEP STYLE FEATURES: no meaningful improvement or unstable metrics."
        )
        print("Reason: holdout improved but TimeSeries CV dropped too much.")
        return False

    if not meaningful_improvement:
        print(
            "DO NOT KEEP STYLE FEATURES: no meaningful improvement or unstable metrics."
        )
        print("Reason: improvement was less than 0.01 accuracy.")
        return False

    print(
        "KEEP STYLE FEATURES: improved XGBoost TimeSeries CV or holdout accuracy "
        "without worsening goal R2 badly."
    )
    return True


# Saves the selected final model set and the exact feature order used by training.
def save_final_models(classifier_results, score_results, feature_cols, label):
    try:
        print(f"=== Saving final {len(feature_cols)}-feature models ===")

        joblib.dump(classifier_results["xgb_model"], MODELS_DIR / "xgb_classifier.pkl")
        joblib.dump(classifier_results["lr_model"], MODELS_DIR / "logistic_classifier.pkl")
        joblib.dump(classifier_results["scaler"], MODELS_DIR / "scaler.pkl")
        joblib.dump(classifier_results["label_encoder"], MODELS_DIR / "label_encoder.pkl")
        joblib.dump(score_results["home_model"], MODELS_DIR / "xgb_home_goals.pkl")
        joblib.dump(score_results["away_model"], MODELS_DIR / "xgb_away_goals.pkl")

        with MODEL_FEATURES_PATH.open("w", encoding="utf-8") as file:
            json.dump(feature_cols, file, indent=2)

        print(f"Saved final model set: {label}")
        print("Saved xgb_classifier.pkl")
        print("Saved logistic_classifier.pkl")
        print("Saved scaler.pkl")
        print("Saved label_encoder.pkl")
        print("Saved xgb_home_goals.pkl")
        print("Saved xgb_away_goals.pkl")
        print("Saved model_features.json")
    except Exception as error:
        print(f"Error saving final models: {error}")


# Prints the currently saved model feature count for sanity.
def print_saved_feature_count():
    try:
        if not MODEL_FEATURES_PATH.exists():
            print("Saved model_features.json feature count: missing")
            return

        with MODEL_FEATURES_PATH.open("r", encoding="utf-8") as file:
            saved_features = json.load(file)

        print(f"Saved model_features.json feature count: {len(saved_features)}")
    except Exception as error:
        print(f"Could not read saved model_features.json: {error}")


if __name__ == "__main__":
    try:
        from src.data_pipeline import get_engine
        from src.feature_engineering import load_match_features

        engine = get_engine()
        df = load_match_features(engine)

        if df is None or len(df) == 0:
            print("Error: No match features found. Run feature engineering first.")
            sys.exit()

        train_df, test_df = prepare_features(df)

        if train_df is None or test_df is None:
            print("Error: Could not prepare model features.")
            sys.exit()

        print("=== Baseline 12-feature model ===")
        baseline_results = train_classifier_variant(
            train_df,
            test_df,
            BASE_FEATURES_12,
            "baseline 12",
        )

        print("=== xG 16-feature model ===")
        xg_results = train_classifier_variant(
            train_df,
            test_df,
            MODEL_FEATURES_16,
            "xG 16",
        )

        print("=== xG + style 20-feature model ===")
        style_results = train_classifier_variant(
            train_df,
            test_df,
            MODEL_FEATURES_STYLE,
            "xG + style 20",
        )

        print("=== Goal regression with xG 16 ===")
        xg_score_results = train_score_predictor(
            train_df,
            test_df,
            MODEL_FEATURES_16,
            "xG 16",
        )

        print("=== Goal regression with xG + style 20 ===")
        style_score_results = train_score_predictor(
            train_df,
            test_df,
            MODEL_FEATURES_STYLE,
            "xG + style 20",
        )

        if (
            baseline_results is None
            or xg_results is None
            or style_results is None
            or xg_score_results is None
            or style_score_results is None
        ):
            print("Error: One or more model comparisons failed.")
            sys.exit()

        keep_style = decide_style_features(
            xg_results,
            style_results,
            xg_score_results,
            style_score_results,
        )

        if keep_style:
            save_final_models(
                style_results,
                style_score_results,
                MODEL_FEATURES_STYLE,
                "xG + style 20",
            )
        else:
            print("Keeping existing 16-feature xG saved models unchanged.")

        print_saved_feature_count()
        print("\nStyle comparison complete.")
    except Exception as error:
        print(f"Error running training comparison pipeline: {error}")
