from collections import deque

import pandas as pd
from river.drift import ADWIN
from config.config import ( initial_window_size,
    max_top_n_features, temporal_decay_rate, case_id_column, frequency_decay_threshold,
    decay_after_events, removal_threshold_events
)
from src.homonym_mend.dynamic_binning_and_categorization import time_distribution, extract_temporal_features, \
    update_time_distribution
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import directly_follows_graph
from collections import defaultdict
import numpy as np
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_history, activity_feature_metadata

# Initialize feature tracking structures
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
feature_relevance_tracker = defaultdict(float)
drift_detector = defaultdict(ADWIN)
day_encoder = CustomLabelEncoder()

# Stores the last event per CaseID
previous_events = {}
feature_last_seen_event = {}  # Track last-seen event count for each feature
feature_accumulations = defaultdict(dict)  # Store feature accumulations per case (for adaptive forgetting)


def forget_old_cases(activity_column, global_event_counter):
    """
    Remove inactive cases from `previous_events` and related accumulations based on event count decay.
    """
    for case_id in list(previous_events.keys()):
        events_since_last_seen = global_event_counter - previous_events[case_id].get("last_seen_event", global_event_counter)

        if "frequency" not in previous_events[case_id]:
            previous_events[case_id]["frequency"] = 1  # Initialize frequency
        if "frequency" in previous_events[case_id]:
            previous_events[case_id]["frequency"] *= np.exp(-events_since_last_seen / decay_after_events)

        if previous_events[case_id]["frequency"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
            # Deduct transition counts related to this case
            directly_follows_graph.remove_case_transitions(case_id)

            # Remove case-specific feature tracking
            if case_id in feature_accumulations:
                del feature_accumulations[case_id]

            # Remove case-specific feature history from activity_feature_history
            if case_id in previous_events:
                activity_label = previous_events[case_id][activity_column]
                if activity_label in activity_feature_history:
                    activity_feature_history[activity_label] = [
                        vector for vector in activity_feature_history[activity_label]
                        if vector[case_id_column] != case_id
                    ]

            del previous_events[case_id]

def compute_time_feature_score(activity, time_features):
    """
    Computes the feature score for time attributes based on incremental tracking.
    """
    hour_bin = time_features["hour_bin"]
    day_of_week = time_features["day_of_week"]
    month = time_features["month"]

    # Retrieve past occurrences for this time configuration
    past_occurrences = time_distribution[activity][hour_bin][day_of_week].get(month, 0)

    # Apply inverse frequency scoring (rarer occurrences get higher weight)
    score = 1 / (past_occurrences + 1)

    return score

def compute_feature_scores(event, activity_column, timestamp_column, resource_column, case_id_column, data_columns, global_event_counter):
    """
    Compute dynamic feature scores incorporating:
    - Control-flow tracking (per CaseID)
    - Feature variation across similar activity labels
    - Resource and time-based weighting
    - Event-driven forgetting
    """
    global previous_events, feature_last_seen_event
    feature_scores = defaultdict(float)
    case_id = event[case_id_column]
    activity_label = event[activity_column]

    # Forget old cases before processing new events
    forget_old_cases(activity_column, global_event_counter)

    ## --- 1. Compare Against All Past Events of the Same Activity Label (Homonym Detection) ---
    if activity_label in activity_feature_history and len(activity_feature_history[activity_label]) > 0:
        previous_vectors = np.array(activity_feature_history[activity_label])

        # Compute mean vector of all past events with this activity label
        mean_vector = np.mean(previous_vectors, axis=0)

        # Compute standard deviation for normalization
        std_vector = np.std(previous_vectors, axis=0)
        std_vector[std_vector == 0] = 1  # Avoid division by zero

        # Compute feature importance based on deviation from the mean
        new_feature_vector = np.array([
            float(event[column]) if column in event and isinstance(event[column], (int, float)) else 0.0
            for column in data_columns
        ], dtype=np.float64)  # ✅ Ensure numerical consistency before performing NumPy operations

        deviations = np.abs(new_feature_vector - mean_vector) / std_vector

        # Apply weight scaling for high deviation features with event-based decay
        for i, column in enumerate(data_columns):
            vector_tuple = tuple(new_feature_vector)

            if activity_label in activity_feature_metadata and vector_tuple in activity_feature_metadata[activity_label]:
                last_seen_event = activity_feature_metadata[activity_label][vector_tuple].get("last_seen_event", global_event_counter)
            else:
                last_seen_event = global_event_counter  # Default if new

            events_since_last_seen = global_event_counter - last_seen_event

            # Apply exponential decay to smooth feature importance reduction
            feature_scores[column] *= np.exp(-events_since_last_seen / (decay_after_events * 2))
            feature_scores[column] += deviations[i] * 1.5

            # Update last seen event counter
            feature_last_seen_event[column] = global_event_counter

    ## --- 2. Control-Flow Perspective (Highest Weight) ---
    prev_event = previous_events.get(case_id)  # Get previous event within the same CaseID
    if prev_event and activity_column in prev_event:
        prev_activity = prev_event[activity_column]
        curr_activity = event[activity_column]

        if prev_activity != curr_activity:
            feature_scores[activity_column] += 2.5  # Prioritize process control-flow

            # Track transition frequency in Directly Follows Graph (Global Tracking)
            directly_follows_graph.add_transition(case_id, prev_activity, curr_activity, global_event_counter)

            # Contextual reinforcement: check frequent transitions
            if directly_follows_graph.get_global_frequency(prev_activity, curr_activity) > 5:
                feature_scores[activity_column] += 1.5  # Further weight for recurring transitions

    ## --- 3. Resource Perspective: Score role/resource-based context ---
    if resource_column in event:
        feature_scores[resource_column] += 0.8

    ## --- 4. Time Perspective (NEW ADDITION) ---
    if isinstance(event[timestamp_column], str):
        event[timestamp_column] = pd.to_datetime(event[timestamp_column], errors='coerce')

    if pd.isna(event[timestamp_column]):
        return {}  # Skip processing if timestamp is invalid

    time_features = extract_temporal_features(event[timestamp_column])

    # Encode categorical features before using them in calculations
    if "day_of_week" in time_features:
        time_features["day_of_week"] = day_encoder.transform(time_features["day_of_week"])  # Encode day as numerical

    # Compute time-based feature score
    feature_scores["time_score"] = compute_time_feature_score(activity_label, time_features)

    ## --- 5. Data Perspective ---
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


def select_features(event, previous_event, activity_column, timestamp_column, resource_column, data_columns, global_event_counter):
    """
    Fast Online Feature Selection with Adaptive Weighting.
    """
    # Compute base feature scores
    feature_scores = compute_feature_scores(
        event, activity_column, timestamp_column, resource_column, case_id_column, data_columns, global_event_counter
    )

    # Apply decay weighting to prioritize recent context
    current_time = event[timestamp_column]
    for feature in feature_scores:
        feature_scores[feature] *= np.exp(-temporal_decay_rate * (current_time - previous_event[timestamp_column]).total_seconds() if previous_event else 1)

    # Dynamic thresholding for selecting top features
    top_n = adaptive_threshold(feature_scores)
    selected_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [feature for feature, _ in selected_features]


def adaptive_threshold(feature_scores):
    """
    Dynamically adjust top feature count based on variability.
    """
    scores = np.array(list(feature_scores.values()))
    if len(scores) == 0:
        return max_top_n_features

    variability = np.std(scores)
    return max(max_top_n_features, int(variability / 0.1))  # Adjust dynamically based on variance
