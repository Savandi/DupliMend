from collections import defaultdict, deque
import numpy as np
from river.drift import ADWIN
from config import adaptive_window_min_size, adaptive_window_max_size, initial_window_size

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(lambda: deque(maxlen=initial_window_size))  # Dynamic feature vectors per feature
feature_window_sizes = defaultdict(lambda: initial_window_size)  # Adaptive window sizes
adaptive_bin_models = defaultdict(lambda: ADWIN())  # Drift-aware bin models for feature scoring

# --- DYNAMIC FEATURE VECTOR CONSTRUCTION ---
def construct_dynamic_feature_vector(event, top_features, timestamp_column, decay_rate=0.01):
    """
    Construct a dynamic feature vector for the current event.

    Parameters:
        event (dict): The current event being processed.
        top_features (list): Top features selected in Step 2.
        timestamp_column (str): The column representing the timestamp.
        decay_rate (float): The rate of temporal decay for older features.

    Returns:
        dict: A dynamic feature vector for the event.
    """
    current_time = event[timestamp_column]
    dynamic_vector = {}

    for feature in top_features:
        value = event.get(feature, None)
        if value is not None:
            if feature in feature_vectors:
                past_values = np.array([v[1] for v in feature_vectors[feature]])
                time_differences = np.array([
                    (current_time - v[0]).total_seconds() for v in feature_vectors[feature]
                ])
                weights = np.exp(-decay_rate * time_differences)
                dynamic_value = np.sum(past_values * weights) / np.sum(weights)
            else:
                dynamic_value = value

            # Update the feature vector with current event data
            feature_vectors[feature].append((current_time, value))
            dynamic_vector[feature] = dynamic_value

    return dynamic_vector

# --- UNIFIED RECENCY-SENSITIVE MECHANISM ---
def integrate_with_unified_mechanism(dynamic_vector, sliding_windows, decay_rate=0.01):
    """
    Integrate the constructed dynamic feature vector with the unified recency-sensitive mechanism.

    Parameters:
        dynamic_vector (dict): The dynamic feature vector.
        sliding_windows (dict): Adaptive sliding windows.
        decay_rate (float): Temporal decay rate.

    Returns:
        dict: Unified feature vector.
    """
    unified_vector = {}

    for feature, value in dynamic_vector.items():
        if feature in sliding_windows:
            sliding_window = sliding_windows[feature]
            sliding_window.append(value)

            # Compute recency-sensitive value
            time_weights = np.exp(-decay_rate * np.arange(len(sliding_window)))
            weighted_sum = np.dot(sliding_window, time_weights)
            unified_vector[feature] = weighted_sum / time_weights.sum()
        else:
            unified_vector[feature] = value

    return unified_vector

# --- ADAPTIVE SLIDING WINDOW ADJUSTMENT ---
def adjust_window_size_for_vector(feature, drift_detected):
    """
    Adjust the sliding window size dynamically for feature vectors.

    Parameters:
        feature (str): Feature name.
        drift_detected (bool): Whether drift was detected for the feature.

    Returns:
        deque: Updated sliding window.
    """
    current_size = feature_window_sizes[feature]
    new_size = max(adaptive_window_min_size, current_size // 2) if drift_detected else min(adaptive_window_max_size,
                                                                                           current_size + 10)
    feature_window_sizes[feature] = new_size
    return deque(maxlen=new_size)

# --- DRIFT DETECTION FOR FEATURES ---
def detect_feature_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores.

    Parameters:
        feature (str): Feature name.
        feature_scores (list): Scores for the feature.

    Returns:
        bool: True if drift is detected, False otherwise.
    """
    if feature not in adaptive_bin_models:
        adaptive_bin_models[feature] = ADWIN()

    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0
    return adaptive_bin_models[feature].update(avg_score)

# --- MAIN PROCESSING FUNCTION ---
def process_event(event, top_features, timestamp_column, sliding_windows):
    """
    Process an event to construct and integrate the dynamic feature vector.

    Parameters:
        event (dict): The current event being processed.
        top_features (list): Top features selected in Step 2.
        timestamp_column (str): The column representing the timestamp.
        sliding_windows (dict): Adaptive sliding windows from Step 2.

    Returns:
        dict: Unified feature vector for the event.
    """
    # Step 1: Construct the dynamic feature vector
    dynamic_vector = construct_dynamic_feature_vector(event, top_features, timestamp_column)

    # Step 2: Integrate with the unified recency-sensitive mechanism
    unified_vector = integrate_with_unified_mechanism(dynamic_vector, sliding_windows)

    return unified_vector
