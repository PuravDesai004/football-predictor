import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier, XGBRegressor


# WHY TIME-BASED SPLIT:
# Football data is time-ordered. Using random split would let the model
# train on GW30 matches and predict GW20 — using future data to predict
# the past. We always train on earlier gameweeks and test on later ones.
#
# WHY H2H FEATURES REMOVED:
# With only one season of data, H2H stats are calculated from the same
# matches being predicted — this is data leakage. H2H will be added back
# properly in Tier 3 when multiple seasons of historical data are available.


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Prepares the match_features DataFrame for classification and regression modeling.
def prepare_features(df):
    try:
        df = df.dropna(subset=["home_form_scored"])
        df = df.dropna(subset=["away_form_scored"])
        df = df.sort_values("gameweek").reset_index(drop=True)

        feature_cols = [
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

        df[feature_cols] = df[feature_cols].fillna(0)

        x = df[feature_cols]
        y_result = df["result"]
        y_home_goals = df["home_goals"]
        y_away_goals = df["away_goals"]
        gameweeks = df["gameweek"]

        print(f"Features shape after cleaning: {x.shape}")
        print(f"Class distribution:\n{y_result.value_counts()}")

        return x, y_result, y_home_goals, y_away_goals, gameweeks, feature_cols
    except Exception as error:
        print(f"Error preparing features: {error}")
        return None, None, None, None, None, None


# Trains Logistic Regression and XGBoost classifiers and saves the best classifier assets.
def train_classifier(X, y, gameweeks):
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        split_index = int(len(X) * 0.8)
        x_train = X.iloc[:split_index]
        x_test = X.iloc[split_index:]
        y_train = y_encoded[:split_index]
        y_test = y_encoded[split_index:]
        train_gameweeks = gameweeks.iloc[:split_index]
        test_gameweeks = gameweeks.iloc[split_index:]

        print(
            f"Train: {len(x_train)} matches | "
            f"GW{train_gameweeks.min()} to GW{train_gameweeks.max()}"
        )
        print(
            f"Test:  {len(x_test)} matches | "
            f"GW{test_gameweeks.min()} to GW{test_gameweeks.max()}"
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(x_train_scaled, y_train)

        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        xgb_model.fit(x_train, y_train)

        lr_cv = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            x_train_scaled,
            y_train,
            cv=5,
            scoring="accuracy",
        )
        xgb_cv = cross_val_score(
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            x_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        lr_test_pred = lr_model.predict(x_test_scaled)
        xgb_test_pred = xgb_model.predict(x_test)
        lr_test_acc = accuracy_score(y_test, lr_test_pred)
        xgb_test_acc = accuracy_score(y_test, xgb_test_pred)

        print("\n── CLASSIFIER RESULTS ──────────────────")
        print(
            f"Logistic Regression  | CV Accuracy: {lr_cv.mean():.3f} ± {lr_cv.std():.3f} | Test: {lr_test_acc:.3f}"
        )
        print(
            f"XGBoost Classifier   | CV Accuracy: {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f} | Test: {xgb_test_acc:.3f}"
        )

        print(
            classification_report(
                label_encoder.inverse_transform(y_test),
                label_encoder.inverse_transform(xgb_test_pred),
            )
        )

        joblib.dump(xgb_model, MODELS_DIR / "xgb_classifier.pkl")
        joblib.dump(lr_model, MODELS_DIR / "logistic_classifier.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        print("Logistic Regression also saved as logistic_classifier.pkl")
        joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
        print("Models saved to models/saved/")

        return lr_model, xgb_model, label_encoder, scaler, x_test, y_test
    except Exception as error:
        print(f"Error training classifier: {error}")
        return None, None, None, None, None, None


# Trains XGBoost regressors to predict exact home and away goals.
def train_score_predictor(X, y_home, y_away):
    try:
        split_index = int(len(X) * 0.8)
        x_train = X.iloc[:split_index]
        x_test = X.iloc[split_index:]
        y_home_train = y_home.iloc[:split_index]
        y_home_test = y_home.iloc[split_index:]
        y_away_train = y_away.iloc[:split_index]
        y_away_test = y_away.iloc[split_index:]

        home_model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        home_model.fit(x_train, y_home_train)

        away_model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        away_model.fit(x_train, y_away_train)

        home_pred = home_model.predict(x_test)
        away_pred = away_model.predict(x_test)
        home_pred = np.clip(np.round(home_pred), 0, None)
        away_pred = np.clip(np.round(away_pred), 0, None)

        home_mae = mean_absolute_error(y_home_test, home_pred)
        home_r2 = r2_score(y_home_test, home_pred)
        away_mae = mean_absolute_error(y_away_test, away_pred)
        away_r2 = r2_score(y_away_test, away_pred)

        print("\n── SCORE PREDICTOR RESULTS ─────────────")
        print(f"Home Goals | MAE: {home_mae:.3f} | R2: {home_r2:.3f}")
        print(f"Away Goals | MAE: {away_mae:.3f} | R2: {away_r2:.3f}")

        print("\nSample predictions vs actual:")
        print("Predicted | Actual")
        for i in range(5):
            print(
                f"  {home_pred[i]:.0f} - {away_pred[i]:.0f}  |  {y_home_test.iloc[i]} - {y_away_test.iloc[i]}"
            )

        joblib.dump(home_model, MODELS_DIR / "xgb_home_goals.pkl")
        joblib.dump(away_model, MODELS_DIR / "xgb_away_goals.pkl")
        print("Score predictor models saved.")

        return home_model, away_model
    except Exception as error:
        print(f"Error training score predictor: {error}")
        return None, None


# Prints Logistic Regression test performance using the saved scaler and label encoder.
def evaluate_logistic(lr_model, scaler, X_test, y_test, label_encoder):
    try:
        X_test_scaled = scaler.transform(X_test)
        lr_preds = lr_model.predict(X_test_scaled)
        lr_preds_labels = label_encoder.inverse_transform(lr_preds)
        y_test_labels = label_encoder.inverse_transform(y_test)

        print("\n── LOGISTIC REGRESSION REPORT ──────────")
        print(classification_report(y_test_labels, lr_preds_labels))
        matrix_labels = list(label_encoder.classes_)
        print(f"Confusion Matrix labels: {matrix_labels}")
        print(confusion_matrix(y_test_labels, lr_preds_labels, labels=matrix_labels))
    except Exception as error:
        print(f"Error evaluating Logistic Regression: {error}")


# Given two team names, fetches latest database stats and predicts result probabilities and score.
def predict_match(
    home_team_name,
    away_team_name,
    engine,
    lr_model,
    scaler,
    label_encoder,
    home_model,
    away_model,
    feature_cols,
):
    try:
        home_strength_query = text(
            """
            SELECT team, strength_overall_home, strength_overall_away
            FROM player_fpl_features
            WHERE team_name = :team_name
            LIMIT 1
            """
        )
        away_strength_query = text(
            """
            SELECT team, strength_overall_home, strength_overall_away
            FROM player_fpl_features
            WHERE team_name = :team_name
            LIMIT 1
            """
        )

        home_strength = pd.read_sql(
            home_strength_query, engine, params={"team_name": home_team_name}
        )
        away_strength = pd.read_sql(
            away_strength_query, engine, params={"team_name": away_team_name}
        )

        if home_strength.empty:
            print(f"Team not found: {home_team_name}")
            return

        if away_strength.empty:
            print(f"Team not found: {away_team_name}")
            return

        home_form_query = text(
            """
            SELECT home_form_scored, home_form_conceded, home_clean_sheet_rate,
                   home_fdr
            FROM match_features
            WHERE home_team_name = :team_name
            ORDER BY gameweek DESC LIMIT 1
            """
        )
        away_form_query = text(
            """
            SELECT away_form_scored, away_form_conceded, away_clean_sheet_rate,
                   away_fdr
            FROM match_features
            WHERE away_team_name = :team_name
            ORDER BY gameweek DESC LIMIT 1
            """
        )

        home_form = pd.read_sql(home_form_query, engine, params={"team_name": home_team_name})
        away_form = pd.read_sql(away_form_query, engine, params={"team_name": away_team_name})

        if home_form.empty:
            print(f"Team not found: {home_team_name}")
            return

        if away_form.empty:
            print(f"Team not found: {away_team_name}")
            return

        def get_value(df, column, default=0):
            value = df.iloc[0][column]
            if pd.isna(value):
                return default
            return value

        feature_values = {
            "home_form_scored": get_value(home_form, "home_form_scored"),
            "home_form_conceded": get_value(home_form, "home_form_conceded"),
            "home_clean_sheet_rate": get_value(home_form, "home_clean_sheet_rate"),
            "away_form_scored": get_value(away_form, "away_form_scored"),
            "away_form_conceded": get_value(away_form, "away_form_conceded"),
            "away_clean_sheet_rate": get_value(away_form, "away_clean_sheet_rate"),
            "home_fdr": get_value(home_form, "home_fdr", 3),
            "away_fdr": get_value(away_form, "away_fdr", 3),
            "strength_overall_home": get_value(home_strength, "strength_overall_home"),
            "strength_overall_away": get_value(away_strength, "strength_overall_away"),
            "home_team_away_str": get_value(home_strength, "strength_overall_away"),
            "away_team_home_str": get_value(away_strength, "strength_overall_home"),
        }

        features = pd.DataFrame([feature_values])
        features = features[feature_cols].fillna(0)
        features_scaled = scaler.transform(features)

        probabilities = lr_model.predict_proba(features_scaled)[0]
        probability_by_label = dict(zip(label_encoder.classes_, probabilities))
        home_prob = probability_by_label.get("H", 0)
        draw_prob = probability_by_label.get("D", 0)
        away_prob = probability_by_label.get("A", 0)

        home_goals = int(np.clip(np.round(home_model.predict(features)[0]), 0, None))
        away_goals = int(np.clip(np.round(away_model.predict(features)[0]), 0, None))

        result_probabilities = {"H": home_prob, "D": draw_prob, "A": away_prob}
        predicted_result = max(result_probabilities, key=result_probabilities.get)

        # Keep the displayed score consistent with the classifier's most likely result.
        if predicted_result == "H" and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif predicted_result == "A" and away_goals <= home_goals:
            away_goals = home_goals + 1
        elif predicted_result == "D":
            average_goals = int(round((home_goals + away_goals) / 2))
            home_goals = average_goals
            away_goals = average_goals

        if predicted_result == "H":
            result_text = f"{home_team_name} win"
        elif predicted_result == "A":
            result_text = f"{away_team_name} win"
        else:
            result_text = "Draw"

        print(f"\n── MATCH PREDICTION ─────────────────────")
        print(f"  {home_team_name} vs {away_team_name}")
        print(f"  Win Probabilities:")
        print(f"    {home_team_name} Win : {home_prob:.1%}")
        print(f"    Draw              : {draw_prob:.1%}")
        print(f"    {away_team_name} Win : {away_prob:.1%}")
        print(f"  Predicted Result: {result_text}")
        print(
            f"  Predicted Score: {home_team_name} {home_goals:.0f} - {away_goals:.0f} {away_team_name}"
        )
    except Exception as error:
        print(f"Error predicting match: {error}")


# Prints XGBoost feature importances to show which engineered features are useful.
def show_feature_importance(model, feature_cols):
    try:
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=False)

        print("\n── FEATURE IMPORTANCE (XGBoost Classifier) ──")
        for feat, score in importance.items():
            bar = "█" * int(score * 100)
            print(f"  {feat:<35} {score:.4f}  {bar}")
    except Exception as error:
        print(f"Error showing feature importance: {error}")


if __name__ == "__main__":
    try:
        from src.data_pipeline import get_engine
        from src.feature_engineering import load_match_features

        engine = get_engine()
        df = load_match_features(engine)

        if df is None or len(df) == 0:
            print("Error: No match features found. Run feature engineering first.")
            sys.exit()

        X, y_result, y_home_goals, y_away_goals, gameweeks, FEATURE_COLS = prepare_features(df)

        if X is None:
            print("Error: Could not prepare model features.")
            sys.exit()

        lr_classifier, xgb_classifier, label_encoder, scaler, X_test, y_test = train_classifier(
            X, y_result, gameweeks
        )

        if lr_classifier is not None and scaler is not None and label_encoder is not None:
            evaluate_logistic(lr_classifier, scaler, X_test, y_test, label_encoder)

        home_model, away_model = train_score_predictor(X, y_home_goals, y_away_goals)

        if xgb_classifier is not None:
            show_feature_importance(xgb_classifier, FEATURE_COLS)

        if (
            lr_classifier is not None
            and scaler is not None
            and label_encoder is not None
            and home_model is not None
            and away_model is not None
        ):
            predict_match(
                "Arsenal",
                "Chelsea",
                engine,
                lr_classifier,
                scaler,
                label_encoder,
                home_model,
                away_model,
                FEATURE_COLS,
            )
            predict_match(
                "Liverpool",
                "Man City",
                engine,
                lr_classifier,
                scaler,
                label_encoder,
                home_model,
                away_model,
                FEATURE_COLS,
            )

        print("Day 5 complete.")
    except Exception as error:
        print(f"Error running training pipeline: {error}")
