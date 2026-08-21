import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# Sorts match rows by the canonical time order used by all model experiments.
def sort_by_time(df):
    required_columns = ["gameweek", "fixture_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required time columns: {missing_columns}")

    sort_columns = (["season"] if "season" in df.columns else []) + ["gameweek", "fixture_id"]
    sorted_df = df.sort_values(sort_columns).reset_index(drop=True)
    print(f"Data sorted by: {', '.join(sort_columns)}")
    return sorted_df


# Builds a complete-gameweek train/test split with no gameweek leakage.
def make_gameweek_split(df, train_fraction=0.8):
    if "gameweek" not in df.columns:
        raise ValueError("Missing required column: gameweek")

    period_columns = (["season"] if "season" in df.columns else []) + ["gameweek"]
    periods = (
        df[period_columns]
        .dropna()
        .drop_duplicates()
        .sort_values(period_columns)
        .reset_index(drop=True)
    )
    split_index = int(len(periods) * train_fraction)

    if split_index <= 0 or split_index >= len(periods):
        raise ValueError("Not enough chronological periods to create a time-safe split.")

    train_periods = set(map(tuple, periods.iloc[:split_index].to_numpy()))
    test_periods = set(map(tuple, periods.iloc[split_index:].to_numpy()))
    overlap = sorted(train_periods.intersection(test_periods))

    if overlap:
        raise ValueError(f"Gameweek overlap found: {overlap}")

    row_periods = list(map(tuple, df[period_columns].to_numpy()))
    train_df = df.loc[[period in train_periods for period in row_periods]].reset_index(drop=True)
    test_df = df.loc[[period in test_periods for period in row_periods]].reset_index(drop=True)

    print("=== Time-safe split check ===")
    print(f"Training rows: {len(train_df)}")
    print(f"Train periods: {periods.iloc[0].to_dict()} to {periods.iloc[split_index - 1].to_dict()}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Test periods: {periods.iloc[split_index].to_dict()} to {periods.iloc[-1].to_dict()}")
    print(f"Chronological period overlap: {'none' if not overlap else overlap}")

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
