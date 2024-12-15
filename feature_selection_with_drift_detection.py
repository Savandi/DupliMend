from collections import defaultdict, deque
from river.drift import ADWIN
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from config import adaptive_window_min_size, adaptive_window_max_size, initial_window_size


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


def ensemble_feature_scoring(features, target):
    """
    Combine multiple feature selection methods (e.g., correlation, mutual information).
    """
    scores = {}
    scores["mutual_information"] = {
        f: mutual_info_classif(features[[f]], target, discrete_features='auto')[0] for f in features
    }
    return {f: scores["mutual_information"][f] for f in features}


def temporal_weighting(score, event_time, current_time, decay_rate=0.01):
    """
    Apply temporal decay to feature scores based on time difference.
    """
    time_diff = (current_time - event_time).total_seconds()
    return score * np.exp(-decay_rate * time_diff)


def adaptive_threshold(feature_scores, variability_threshold=0.1):
    """
    Dynamically adjust the threshold for selecting top N features.
    """
    variability = np.std(list(feature_scores.values()))
    return max(3, int(variability / variability_threshold))


def compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Compute scores for individual features based on their associated perspectives.
    """
    feature_scores = defaultdict(float)

    # Control-Flow Perspective
    if previous_event is not None:
        feature_scores[activity_column] += 1  # Activity transitions contribute to control-flow

    # Temporal Perspective
    temporal_features = ['hour', 'day_of_week', 'season']
    for feature in temporal_features:
        if feature in event:
            feature_scores[feature] += 0.3

    # Resource Perspective
    if resource_column in event:
        feature_scores[resource_column] += 0.5

    # Data Perspective
    for column in data_columns:
        if column in event:
            feature_scores[column] += 0.2

    return feature_scores


def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0
    return drift_detector[feature].update(avg_score)


def select_features(event, previous_event, activity_column, timestamp_column, resource_column, data_columns, top_n=3):
    """
    Compute feature scores, detect drift, and select top dataset features.
    """
    feature_scores = compute_feature_scores(
        event, previous_event, activity_column, timestamp_column, resource_column, data_columns
    )

    # Ensemble Scoring
    ensemble_scores = ensemble_feature_scoring(event, event[activity_column])
    for feature, score in ensemble_scores.items():
        feature_scores[feature] += score

    # Adjust thresholds for dynamic feature selection
    top_n = adaptive_threshold(feature_scores)

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

    # Select top N features
    top_features = sorted(aggregated_scores, key=aggregated_scores.get, reverse=True)[:top_n]
    print(f"Activity: {event[activity_column]}, Top Features: {top_features}")
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
