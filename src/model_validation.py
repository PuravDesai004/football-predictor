import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# Sorts match rows by the canonical time order used by all model experiments.
def sort_by_time(df):
    required_columns = ["gameweek", "fixture_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required time columns: {missing_columns}")

    sorted_df = df.sort_values(["gameweek", "fixture_id"]).reset_index(drop=True)
    print("Data sorted by: gameweek, fixture_id")
    return sorted_df


# Builds a complete-gameweek train/test split with no gameweek leakage.
def make_gameweek_split(df, train_fraction=0.8):
    if "gameweek" not in df.columns:
        raise ValueError("Missing required column: gameweek")

    unique_gameweeks = sorted(df["gameweek"].dropna().unique())
    split_index = int(len(unique_gameweeks) * train_fraction)

    if split_index <= 0 or split_index >= len(unique_gameweeks):
        raise ValueError("Not enough gameweeks to create a time-safe split.")

    train_gameweeks = unique_gameweeks[:split_index]
    test_gameweeks = unique_gameweeks[split_index:]
    overlap = sorted(set(train_gameweeks).intersection(set(test_gameweeks)))

    if overlap:
        raise ValueError(f"Gameweek overlap found: {overlap}")

    train_df = df[df["gameweek"].isin(train_gameweeks)].reset_index(drop=True)
    test_df = df[df["gameweek"].isin(test_gameweeks)].reset_index(drop=True)

    print("=== Time-safe split check ===")
    print(f"Training rows: {len(train_df)}")
    print(f"Train gameweeks: GW{int(min(train_gameweeks))} to GW{int(max(train_gameweeks))}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Test gameweeks: GW{int(min(test_gameweeks))} to GW{int(max(test_gameweeks))}")
    print("Gameweek overlap: none")

    return train_df, test_df


# Returns the standard time-series cross-validator used by model experiments.
def get_time_series_cv(n_splits=5):
    print(f"TimeSeries CV folds: {n_splits}")
    return TimeSeriesSplit(n_splits=n_splits)


# Evaluates a fresh model per TimeSeriesSplit fold without shuffled CV.
def evaluate_timeseries_cv(model_factory, X, y, scorer_name="accuracy", n_splits=5):
    supported_scorers = ["accuracy", "neg_mean_absolute_error", "r2"]
    if scorer_name not in supported_scorers:
        raise ValueError(f"Unsupported scorer_name: {scorer_name}")

    splitter = get_time_series_cv(n_splits=n_splits)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    scores = []

    for train_idx, test_idx in splitter.split(X):
        model = model_factory()
        x_train = X.iloc[train_idx].reset_index(drop=True)
        x_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        if scorer_name == "accuracy":
            score = accuracy_score(y_test, predictions)
        elif scorer_name == "neg_mean_absolute_error":
            score = -mean_absolute_error(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

        scores.append(score)

    scores = np.array(scores)
    return scores.mean(), scores.std(), scores


# Blocks known leakage columns from accidentally entering model features.
def validate_no_forbidden_features(feature_names, forbidden_features):
    forbidden_selected = [
        feature for feature in feature_names if feature in forbidden_features
    ]
    if forbidden_selected:
        raise ValueError(f"Forbidden leakage columns selected: {forbidden_selected}")

    print("Forbidden feature check passed")


# Confirms every required feature exists before training or prediction.
def validate_required_features(df, feature_names):
    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    print(f"Required feature check passed for {len(feature_names)} features")
