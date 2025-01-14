from collections import defaultdict, deque
from river.drift import ADWIN
from config.config import adaptive_window_min_size, adaptive_window_max_size, initial_window_size, max_top_n_features, \
    temporal_decay_rate
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# --- GLOBAL VARIABLES ---
directly_follows_matrix = defaultdict(lambda: defaultdict(int))  # Directly follows relationships
footprint_matrix = defaultdict(lambda: defaultdict(float))  # Footprint matrix with stability/forgetting
drift_detector = defaultdict(ADWIN)  # Drift detector for each feature


def configure_window_sizes():
    """
    Configures the initial size for sliding windows globally.
    """
    global feature_window_sizes, feature_importance_windows
    feature_window_sizes = defaultdict(lambda: initial_window_size)
    feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))


def ensemble_feature_scoring(event, activity_label):
    """
    Score features using multiple scoring methods and ensemble the results.

    Parameters:
        event (dict): The event being processed.
        activity_label (str): The activity label for which features are scored.

    Returns:
        dict: A dictionary containing ensemble scores for all features.
    """
    # Exclude the activity label and handle missing values
    features = {k: v for k, v in event.items() if k != activity_label and not pd.isna(v)}

    # Separate numeric and categorical features
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    categorical_features = {k: v for k, v in features.items() if isinstance(v, str)}

    # Convert to arrays
    numeric_array = np.array(list(numeric_features.values())).reshape(1, -1) if numeric_features else np.empty((0, 0))
    categorical_array = np.array(list(categorical_features.values())).reshape(1,
                                                                              -1) if categorical_features else np.empty(
        (0, 0))

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_categorical = encoder.fit_transform(categorical_array) if categorical_features else np.empty((0, 0))

    # Align dimensions for numeric and categorical arrays
    if numeric_array.shape[0] != encoded_categorical.shape[0]:
        max_rows = max(numeric_array.shape[0], encoded_categorical.shape[0])
        numeric_array = np.resize(numeric_array, (max_rows, numeric_array.shape[1]))
        encoded_categorical = np.resize(encoded_categorical, (max_rows, encoded_categorical.shape[1]))

    # Combine numeric and categorical into a feature matrix
    feature_matrix = np.hstack(
        [numeric_array, encoded_categorical]) if numeric_array.size > 0 or encoded_categorical.size > 0 else np.empty(
        (0, 0))

    # Prepare feature keys and labels for scoring
    feature_keys = list(numeric_features.keys()) + list(categorical_features.keys())
    target = [activity_label] * feature_matrix.shape[0]

    # Compute scores using multiple methods
    scores = {"mutual_information": {}, "f_statistic": {}}
    if feature_matrix.size > 0:  # Only compute scores if there are valid features
        valid_features = [
            i for i in range(feature_matrix.shape[1]) if np.nanstd(feature_matrix[:, i]) > 0
        ]  # Filter out zero-variance features

        for i in valid_features:
            try:
                scores["mutual_information"][feature_keys[i]] = mutual_info_classif(
                    feature_matrix[:, [i]], target, discrete_features=False
                )[0]
                scores["f_statistic"][feature_keys[i]] = f_classif(
                    feature_matrix[:, [i]], target
                )[0][0]
            except ValueError:  # Gracefully handle computation errors
                scores["mutual_information"][feature_keys[i]] = 0
                scores["f_statistic"][feature_keys[i]] = 0

    # Aggregate scores into an ensemble score
    ensemble_scores = {key: scores["mutual_information"].get(key, 0) + scores["f_statistic"].get(key, 0)
                       for key in feature_keys}

    return ensemble_scores


def temporal_weighting(score, event_time, current_time, decay_rate=temporal_decay_rate):
    """
    Apply temporal decay to feature scores based on time difference.
    """
    time_diff = (current_time - event_time).total_seconds()
    return score * np.exp(-decay_rate * time_diff)


def adaptive_threshold(feature_scores):
    """
    Dynamically adjust the number of top features based on score variability.

    Parameters:
        feature_scores (dict): Feature scores.

    Returns:
        int: Number of top features to select.
    """

    scores = np.array(list(feature_scores.values()))
    if len(scores) == 0:
        return max_top_n_features  # Default to a minimum threshold when no scores are available

    variability = np.nanstd(scores)  # Use nanstd to handle NaN values gracefully
    variability_threshold = 0.1  # Define a default threshold for variability
    return max(max_top_n_features,
               int(np.nan_to_num(variability) / variability_threshold))  # Handle NaN with nan_to_num


def compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Compute scores for individual features based on their associated perspectives.
    """
    feature_scores = defaultdict(float)

    # Control-Flow Perspective
    if previous_event is not None and activity_column in previous_event:
        prev_activity = previous_event[activity_column]
        curr_activity = event[activity_column]
        print(f"Control-Flow Debug: Previous Activity: {prev_activity}, Current Activity: {curr_activity}")
        if prev_activity != curr_activity:
            feature_scores[activity_column] += 1.0  # Weight for activity transitions
            print(f"Control-Flow Contribution for '{activity_column}': {feature_scores[activity_column]}")

    # Temporal Perspective
    temporal_features = ['hour', 'day_of_week', 'season']
    for feature in temporal_features:
        if feature in event:
            feature_scores[feature] += 0.3
    print(f"Temporal Perspective Scores: {feature_scores}")

    # Resource Perspective
    if resource_column in event:
        feature_scores[resource_column] += 0.5
    print(f"Resource Perspective Scores: {feature_scores}")

    # Data Perspective
    for column in data_columns:
        if column in event:
            feature_scores[column] += 0.6
    print(f"Data Perspective Scores: {feature_scores}")

    print(f"Final Feature Scores for Event: {dict(feature_scores)}")  # Log all feature scores
    return feature_scores




def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0
    return drift_detector[feature].update(avg_score)


def select_features(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Compute feature scores, detect drift, and select top dataset features.
    """
    # Compute feature scores
    feature_scores = compute_feature_scores(
        event, previous_event, activity_column, timestamp_column, resource_column, data_columns
    )

    # Ensemble Scoring
    ensemble_scores = ensemble_feature_scoring(event, event[activity_column])
    for feature, score in ensemble_scores.items():
        feature_scores[feature] += score

    # Temporal Weighting
    current_time = event[timestamp_column]
    for feature in feature_scores:
        feature_scores[feature] = temporal_weighting(feature_scores[feature], event[timestamp_column], current_time)

    # Detect drift and adjust windows
    for feature, score in feature_scores.items():
        if not feature_importance_windows[feature]:
            feature_importance_windows[feature].append(0)
        drift_detected = detect_drift(feature, list(feature_importance_windows[feature]))
        feature_importance_windows[feature] = adjust_window_size(feature, drift_detected)
        feature_importance_windows[feature].append(score)

    # Aggregate scores
    aggregated_scores = {
        feature: (sum(window) / len(window)) if len(window) > 0 else 0
        for feature, window in feature_importance_windows.items()
    }

    # Adjust thresholds for dynamic feature selection
    top_n = adaptive_threshold(feature_scores)

    # Select top N features
    top_features = sorted(aggregated_scores, key=aggregated_scores.get, reverse=True)[:top_n]

    # Return the selected features
    return top_features


def adjust_window_size(feature, drift_detected):
    """
    Adjust the sliding window size dynamically based on drift detection.
    """
    current_size = feature_window_sizes[feature]
    new_size = max(adaptive_window_min_size, current_size // 2) if drift_detected else min(adaptive_window_max_size,
                                                                                           current_size + 10)
    feature_window_sizes[feature] = new_size
    return deque(maxlen=new_size)
