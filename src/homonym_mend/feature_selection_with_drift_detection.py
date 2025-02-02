from collections import defaultdict, deque
from river.drift import ADWIN
from config.config import (
    adaptive_window_min_size, adaptive_window_max_size, initial_window_size,
    max_top_n_features, temporal_decay_rate, case_id_column
)
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from src.utils.directly_follows_graph import DirectlyFollowsGraph
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_history

# --- GLOBAL VARIABLES ---
directly_follows_graph = DirectlyFollowsGraph()
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
feature_relevance_tracker = defaultdict(float)
drift_detector = defaultdict(ADWIN)
# Stores the last event per CaseID
previous_events = {}
def configure_window_sizes():
    """Configures initial sliding window sizes."""
    global feature_window_sizes, feature_importance_windows
    feature_window_sizes = defaultdict(lambda: initial_window_size)
    feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))

from collections import defaultdict
import numpy as np
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_history
from src.utils.directly_follows_graph import DirectlyFollowsGraph

def compute_feature_scores(event, activity_column, timestamp_column, resource_column, case_id_column, data_columns):
    """
    Compute dynamic feature scores incorporating:
    - Control-flow tracking (per CaseID)
    - Feature variation across similar activity labels
    - Resource and time-based weighting
    """
    global previous_events
    feature_scores = defaultdict(float)
    case_id = event[case_id_column]
    activity_label = event[activity_column]

    ## --- 1. Compare Against All Past Events of the Same Activity Label (Homonym Detection) ---
    if activity_label in activity_feature_history and len(activity_feature_history[activity_label]) > 0:
        previous_vectors = np.array(activity_feature_history[activity_label])

        # Compute mean vector of all past events with this activity label
        mean_vector = np.mean(previous_vectors, axis=0)

        # Compute standard deviation for normalization
        std_vector = np.std(previous_vectors, axis=0)
        std_vector[std_vector == 0] = 1  # Avoid division by zero

        # Compute feature importance based on deviation from the mean
        new_feature_vector = np.array([event[column] for column in data_columns if column in event])
        deviations = np.abs(new_feature_vector - mean_vector) / std_vector

        # Apply weight scaling for high deviation features
        for i, column in enumerate(data_columns):
            if isinstance(feature_scores[column], (int, float)):  # Ensure it's a valid number
                feature_scores[column] += deviations[i] * 1.5

    ## --- 2. Control-Flow Perspective (Highest Weight) ---
    prev_event = previous_events.get(case_id)  # Get previous event within the same CaseID
    if prev_event and activity_column in prev_event:
        prev_activity = prev_event[activity_column]
        curr_activity = event[activity_column]

        if prev_activity != curr_activity:
            feature_scores[activity_column] += 2.5  # Prioritize process control-flow

            # Track transition frequency in Directly Follows Graph (CaseID-specific)
            directly_follows_graph.add_transition(prev_activity, curr_activity)

            # Contextual reinforcement: check frequent transitions
            if directly_follows_graph.get_global_frequency(prev_activity, curr_activity) > 5:
                feature_scores[activity_column] += 1.5  # Further weight for recurring transitions

    ## --- 3. Resource Perspective: Score role/resource-based context ---
    if resource_column in event:
        feature_scores[resource_column] += 0.8

    ## --- 4. Data Perspective ---
    if prev_event:
        for column in data_columns:
            if column in event and column in prev_event:
                base_score = 0.6
                try:
                    if isinstance(event[column], (int, float)):
                        # Weight continuous features higher if variance is detected
                        feature_scores[column] += base_score * (1.2 if np.var([prev_event[column], event[column]]) > 0.05 else 1.0)
                    elif isinstance(event[column], str):
                        # Score categorical shifts dynamically
                        feature_scores[column] += base_score * (1.15 if prev_event[column] != event[column] else 1.0)
                except Exception:
                    feature_scores[column] += base_score  # Fallback handling

    ## --- 5. Time Perspective ---
    time_features = [col for col in event.keys() if col in [
        "hour_bin", "day_period", "day_of_week", "is_weekend", "week_of_month", "season", "month"
    ]]

    for time_feature in time_features:
        if time_feature in event and prev_event and time_feature in prev_event:
            if event[time_feature] != prev_event[time_feature]:
                feature_scores[time_feature] += 0.9  # Weight changes in time categories
            else:
                feature_scores[time_feature] += 0.5  # Maintain some relevance even without change

    ## Update previous event tracker for this CaseID
    previous_events[case_id] = event

    return feature_scores


def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores and update selection strategies.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0
    return drift_detector[feature].update(avg_score)

def select_features(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Fast Online Feature Selection with Adaptive Weighting.
    """
    # Compute base feature scores
    feature_scores = compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns)

    # Ensemble Scoring (Mutual Information + Process Importance)
    ensemble_scores = feature_scores.copy()
    feature_keys = list(feature_scores.keys())

    # Apply decay weighting to prioritize recent context
    current_time = event[timestamp_column]
    for feature in feature_scores:
        feature_scores[feature] *= np.exp(-temporal_decay_rate * (current_time - previous_event[timestamp_column]).total_seconds() if previous_event else 1)

    # Adjust Window Size Based on Drift
    for feature, score in feature_scores.items():
        if not feature_importance_windows[feature]:
            feature_importance_windows[feature].append(0)

        drift_detected = detect_drift(feature, list(feature_importance_windows[feature]))
        feature_importance_windows[feature] = adjust_window_size(feature, drift_detected)
        feature_importance_windows[feature].append(score)

    # Aggregate Scores with Contextual Weighting
    aggregated_scores = {}
    for feature, window in feature_importance_windows.items():
        weights = np.exp(-0.1 * np.arange(len(window)))  # More recent scores have higher weight
        weighted_scores = np.multiply(window, weights[::-1])
        aggregated_scores[feature] = np.sum(weighted_scores) / np.sum(weights) if np.sum(weights) > 0 else 0

    # Dynamic thresholding for selecting top features
    top_n = adaptive_threshold(feature_scores)
    selected_features = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [feature for feature, _ in selected_features]

def adjust_window_size(feature, drift_detected):
    """
    Dynamically adjust the sliding window size based on drift detection.
    """
    current_size = feature_window_sizes[feature]
    new_size = max(adaptive_window_min_size, current_size // 2) if drift_detected else min(adaptive_window_max_size, current_size + 10)
    feature_window_sizes[feature] = new_size
    return deque(maxlen=new_size)

def adaptive_threshold(feature_scores):
    """
    Dynamically adjust top feature count based on variability.
    """
    scores = np.array(list(feature_scores.values()))
    if len(scores) == 0:
        return max_top_n_features

    variability = np.std(scores)
    return max(max_top_n_features, int(variability / 0.1))  # Adjust dynamically based on variance
