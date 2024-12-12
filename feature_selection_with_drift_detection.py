from collections import defaultdict, deque
from river.drift import ADWIN
import numpy as np
from config import adaptive_window_min_size, adaptive_window_max_size, initial_window_size

# --- GLOBAL VARIABLES ---
directly_follows_matrix = defaultdict(lambda: defaultdict(int))  # Directly follows relationships
footprint_matrix = defaultdict(lambda: defaultdict(float))  # Footprint matrix with stability/forgetting

# Drift detectors for each feature
drift_detector = defaultdict(ADWIN)

def configure_window_sizes():
    """
    Configures the initial size for sliding windows globally.
    """
    global feature_window_sizes, feature_importance_windows
    feature_window_sizes = defaultdict(lambda: initial_window_size)
    feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))


def select_features(event, previous_event, activity_column, timestamp_column, resource_column, case_id_column, data_columns, top_n=3):
    """
    Compute feature scores, detect drift, and select top dataset features.
    """
    feature_scores = compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns)

    # Detect drift for each feature
    for feature, score in feature_scores.items():
        if not feature_importance_windows[feature]:
            feature_importance_windows[feature].append(0)  # Initialize with default value
        drift_detected = detect_drift(feature, list(feature_importance_windows[feature]))
        feature_importance_windows[feature] = adjust_window_size(feature, drift_detected)
        feature_importance_windows[feature].append(score)

    # Aggregate scores for each feature
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

    Parameters:
        feature (str): The feature being processed.
        drift_detected (bool): Whether drift was detected for the feature.

    Returns:
        deque: A deque with the adjusted maximum length.
    """
    current_size = feature_window_sizes[feature]
    new_size = max(adaptive_window_min_size, current_size // 2) if drift_detected else min(adaptive_window_max_size, current_size + 10)
    feature_window_sizes[feature] = new_size
    return deque(maxlen=new_size)


def compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Compute scores for individual features based on their associated perspectives.
    """
    feature_scores = defaultdict(float)

    # Control-Flow Perspective
    if previous_event is not None:
        feature_scores[activity_column] += 1  # Activity transitions contribute to control-flow

    # Temporal Perspective
    temporal_features = ['hour', 'day_of_week', 'season']  # Derived temporal features
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


# --- DRIFT DETECTION ---
def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores.
    """
    # Ensure the drift detector exists for the feature
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()  # Initialize an ADWIN detector for the feature

    # Avoid calculating drift on empty feature scores
    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0

    # Update the drift detector and return whether drift was detected
    return drift_detector[feature].update(avg_score)
