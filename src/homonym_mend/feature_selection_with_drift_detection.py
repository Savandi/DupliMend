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

feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
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
    try:
        # Convert string timestamps to datetime objects
        if isinstance(event_time, str):
            event_time = pd.to_datetime(event_time)
        if isinstance(current_time, str):
            current_time = pd.to_datetime(current_time)

        time_diff = (current_time - event_time).total_seconds()
        return score * np.exp(-decay_rate * time_diff)
    except Exception as e:
        print(f"Warning in temporal_weighting: {e}")
        return score


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
    Compute dynamic feature scores
    """
    feature_scores = defaultdict(float)

    # Control-Flow Perspective - Dynamic scoring based on transitions
    if previous_event is not None and activity_column in previous_event:
        prev_activity = previous_event[activity_column]
        curr_activity = event[activity_column]

        if prev_activity != curr_activity:
            # Base weight for activity transitions
            feature_scores[activity_column] += 1.0

            # Dynamically score contextual transitions
            for column in data_columns:
                if column in event and column in previous_event:
                    if event[column] != previous_event[column]:
                        # Higher weight for contextual changes during activity transitions
                        feature_scores[column] += 1.2

    # Resource Perspective
    if resource_column in event:
        feature_scores[resource_column] += 0.8

    # Data Perspective - Dynamic importance calculation
    for column in data_columns:
        if column in event:
            # Base score for all data columns
            base_score = 0.6

            # Enhance score based on value type and patterns
            try:
                if isinstance(event[column], (int, float)):
                    # Numerical columns get slight boost for variance detection
                    feature_scores[column] += base_score * 1.1
                elif isinstance(event[column], str):
                    # Categorical columns base score
                    feature_scores[column] += base_score

                    # Boost score if the column has high cardinality in recent events
                    if previous_event and column in previous_event:
                        if event[column] != previous_event[column]:
                            feature_scores[column] *= 1.15
            except Exception:
                # Fallback to base score if any error in processing
                feature_scores[column] += base_score

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
    Dynamic feature selection
    """
    # Get basic feature scores
    feature_scores = compute_feature_scores(
        event, previous_event, activity_column, timestamp_column, resource_column, data_columns
    )

    # Add ensemble scores
    ensemble_scores = ensemble_feature_scoring(event, event[activity_column])
    for feature, score in ensemble_scores.items():
        feature_scores[feature] += score

    # Apply temporal weighting
    current_time = event[timestamp_column]
    for feature in feature_scores:
        feature_scores[feature] = temporal_weighting(
            feature_scores[feature],
            event[timestamp_column],
            current_time
        )

    # Detect drift and adjust windows dynamically
    for feature, score in feature_scores.items():
        if not feature_importance_windows[feature]:
            feature_importance_windows[feature].append(0)

        # Check for drift
        drift_detected = detect_drift(feature, list(feature_importance_windows[feature]))
        feature_importance_windows[feature] = adjust_window_size(feature, drift_detected)
        feature_importance_windows[feature].append(score)

    # Aggregate scores considering drift and temporal aspects
    aggregated_scores = {}
    for feature, window in feature_importance_windows.items():
        if len(window) > 0:
            # Weight recent scores more heavily
            weights = np.exp(-0.1 * np.arange(len(window)))
            weighted_scores = np.multiply(window, weights[::-1])
            aggregated_scores[feature] = np.sum(weighted_scores) / np.sum(weights)
        else:
            aggregated_scores[feature] = 0

    # Dynamic threshold based on score distribution
    top_n = adaptive_threshold(feature_scores)

    # Select top features while avoiding redundancy
    selected_features = []
    sorted_features = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    for feature, score in sorted_features:
        if len(selected_features) < top_n:
            selected_features.append(feature)

    return selected_features


def adjust_window_size(feature, drift_detected):
    """
    Adjust the sliding window size dynamically based on drift detection.
    """
    current_size = feature_window_sizes[feature]
    new_size = max(adaptive_window_min_size, current_size // 2) if drift_detected else min(adaptive_window_max_size,
                                                                                           current_size + 10)
    feature_window_sizes[feature] = new_size
    return deque(maxlen=new_size)
