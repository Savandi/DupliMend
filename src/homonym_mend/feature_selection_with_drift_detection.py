from collections import defaultdict, deque
from river.drift import ADWIN
from config.config import (
    adaptive_window_min_size, adaptive_window_max_size, initial_window_size,
    max_top_n_features, temporal_decay_rate
)
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from src.utils.directly_follows_graph import DirectlyFollowsGraph

# --- GLOBAL VARIABLES ---
directly_follows_graph = DirectlyFollowsGraph()
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
feature_relevance_tracker = defaultdict(float)
drift_detector = defaultdict(ADWIN)

def configure_window_sizes():
    """Configures initial sliding window sizes."""
    global feature_window_sizes, feature_importance_windows
    feature_window_sizes = defaultdict(lambda: initial_window_size)
    feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))

def compute_feature_scores(event, previous_event, activity_column, timestamp_column, resource_column, data_columns):
    """
    Compute dynamic feature scores incorporating control-flow tracking.
    """
    feature_scores = defaultdict(float)

    # Control-Flow Perspective: Adaptively reinforce process-based context
    if previous_event is not None and activity_column in previous_event:
        prev_activity = previous_event[activity_column]
        curr_activity = event[activity_column]

        if prev_activity != curr_activity:
            # Boost control-flow importance
            feature_scores[activity_column] += 2.0  # Higher base weight for process transitions
            directly_follows_graph.add_transition(prev_activity, curr_activity)

            # Contextual reinforcement: check frequent transitions
            if directly_follows_graph.get_frequency(prev_activity, curr_activity) > 5:
                feature_scores[activity_column] += 1.5  # Further weight for recurring transitions

    # Resource Perspective: Score role/resource-based context
    if resource_column in event:
        feature_scores[resource_column] += 0.8

    # Data Perspective: Introduce streaming feature relevance tracking
    for column in data_columns:
        if column in event:
            base_score = 0.6

            try:
                if isinstance(event[column], (int, float)):
                    # Weight continuous features higher if variance is detected
                    feature_scores[column] += base_score * (1.2 if np.var([previous_event[column], event[column]]) > 0.05 else 1.0)
                elif isinstance(event[column], str):
                    # Score categorical shifts dynamically
                    feature_scores[column] += base_score * (1.15 if previous_event and event[column] != previous_event[column] else 1.0)
            except Exception:
                feature_scores[column] += base_score  # Fallback handling

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
