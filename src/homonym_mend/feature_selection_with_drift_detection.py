from collections import deque

import pandas as pd
from river.drift import ADWIN
from config.config import (initial_window_size,
                           max_top_n_features, temporal_decay_rate, case_id_column, frequency_decay_threshold,
                           decay_after_events, removal_threshold_events, previousEvents
                           )
from src.homonym_mend.dynamic_binning_and_categorization import time_distribution
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import directly_follows_graph, resource_usage_history, activity_feature_history, \
    feature_weights
from collections import defaultdict
import numpy as np
from src.utils.global_state import previous_events

# Initialize feature tracking structures
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
feature_relevance_tracker = defaultdict(float)
day_encoder = CustomLabelEncoder()
hour_bin_encoder = CustomLabelEncoder()
month_encoder = CustomLabelEncoder()
drift_detector = defaultdict(ADWIN)
feature_last_seen_event = {}  # Track last-seen event count for each feature
feature_accumulations = defaultdict(dict)  # Store feature accumulations per case (for adaptive forgetting)

def forget_old_cases(control_flow_column, global_event_counter):
    """
    Remove inactive cases from `previous_events` and related accumulations based on event count decay.
    Ensures that expired cases no longer influence historical tracking.

    Removes:
    - Control-Flow historical records (previous activities per case)
    - Resource usage history (past occurrences of resources)
    - Time distribution history (past occurrences of timestamps)
    - Event data history (past feature vectors)
    """
    for case_id in list(previous_events.keys()):
        previous_events_metadata = previous_events.get(case_id, deque(maxlen=previousEvents))

        # Ensure `previous_events_metadata` exists and has data
        if isinstance(previous_events_metadata, deque) and len(previous_events_metadata) > 0:
            last_event = previous_events_metadata[-1]

            # Convert last_event to an integer if it's a string
            if isinstance(last_event, str):
                try:
                    last_event = int(last_event)  # Convert string to integer
                except ValueError:
                    print(f"[ERROR] Non-numeric last event for case {case_id}: {last_event} - Using fallback.")
                    last_event = global_event_counter

            # Ensure it's numeric before using it
            events_since_last_seen = last_event if isinstance(last_event, (int, float)) else global_event_counter
        else:
            events_since_last_seen = global_event_counter  # Default fallback

        # Initialize frequency tracking if missing
        if case_id not in previous_events:
            previous_events[case_id] = {"frequency": 1}

        if "frequency" in previous_events[case_id]:
            previous_events[case_id]["frequency"] *= np.exp(-events_since_last_seen / decay_after_events)

        # Remove cases based on frequency decay
        if previous_events[case_id]["frequency"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
            activity_label = previous_events[case_id].get(control_flow_column, None)

            ## --- 1️⃣ Remove Control-Flow Historical Data ---
            if case_id in previous_events:
                del previous_events[case_id]  # Remove last three activities tracking

            if case_id in directly_follows_graph.case_transitions:
                directly_follows_graph.remove_case_transitions(case_id)  # Remove case transitions

            ## --- 2️⃣ Remove Resource Perspective Historical Data ---
            if activity_label and activity_label in resource_usage_history:
                del resource_usage_history[activity_label]  # Forget resource usage tracking for expired cases

            ## --- 3️⃣ Remove Time Perspective Historical Data ---
            if activity_label and activity_label in time_distribution:
                del time_distribution[activity_label]  # Forget past time occurrences for expired cases

            ## --- 4️⃣ Remove Data Column (Event Attribute) Historical Data ---
            if activity_label and activity_label in activity_feature_history:
                activity_feature_history[activity_label] = [
                    vector for vector in activity_feature_history[activity_label]
                    if vector.get(case_id_column) != case_id
                ]

            if activity_label and activity_label in feature_last_seen_event:
                del feature_last_seen_event[activity_label]  # Forget last seen tracking for expired features

            # Remove case-specific feature tracking
            if case_id in feature_accumulations:
                del feature_accumulations[case_id]


def compute_time_feature_score(activity, time_features):
    """
    Computes the feature score for time attributes based on incremental tracking and drift adaptation.
    """
    # Retrieve encoded time feature values
    hour_bin = time_features.get("hour_bin", 0)
    day_of_week = time_features.get("day_of_week", 0)
    month = time_features.get("month", 0)
    is_weekend = time_features.get("is_weekend", 0)

    # Retrieve past occurrences for this time configuration
    past_occurrences = time_distribution.get(activity, {}).get(hour_bin, {}).get(day_of_week, {}).get(month, 0)

    # Compute inverse frequency score (rarer occurrences get higher weight)
    base_score = 1 / (past_occurrences + 1)

    # Dynamically adjust feature weights based on drift detection
    weight_hour = feature_weights["hour_bin"]
    weight_day = feature_weights["day_of_week"]
    weight_month = feature_weights["month"]
    weight_weekend = feature_weights["is_weekend"]

    # Compute final score using dynamically updated weights
    score = (base_score * weight_hour * hour_bin) + \
            (base_score * weight_day * day_of_week) + \
            (base_score * weight_month * month) + \
            (base_score * weight_weekend * is_weekend)

    return score

def update_feature_weights(feature, new_score):
    """
    Updates feature weights dynamically based on detected drift and historical frequency.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    drift_detected = drift_detector[feature].update(new_score)

    # Increase weight if drift is detected (resource importance is changing)
    if drift_detected:
        feature_weights[feature] *= 1.1
    else:
        feature_weights[feature] *= 0.99

    # Normalize feature weights between reasonable limits
    feature_weights[feature] = min(max(feature_weights[feature], 0.5), 2.0)

def compute_feature_scores(event, event_id_column, control_flow_column, timestamp_column, resource_column, case_id_column, data_columns, global_event_counter):
    """
    Compute feature scores incorporating control-flow, resource, and time-based factors.
    """
    global feature_last_seen_event
    feature_scores = defaultdict(float)
    case_id = event[case_id_column]
    activity_label = event[control_flow_column]

    # Ensure timestamp is numeric (convert Timestamp to float safely)
    current_time = event[timestamp_column]

    if isinstance(current_time, pd.Timestamp):
        current_time = current_time.timestamp()  # Convert to float
    elif isinstance(current_time, str):
        try:
            current_time = pd.to_datetime(current_time, errors='coerce').timestamp()
        except Exception as e:
            print(f"[ERROR] Failed to convert timestamp for Event {event[event_id_column]}: {e}")
            return {}  # Skip processing if timestamp conversion fails

    if pd.isna(current_time) or current_time is None:
        print(f"[ERROR] Invalid Timestamp detected for Event {event[event_id_column]} - Skipping event.")
        return {}  # Skip this event if timestamp is still NaT

    # Forget old cases before processing new events
    forget_old_cases(control_flow_column, global_event_counter)

    ## --- 1. Compare Against All Past Events of the Same Activity Label (Homonym Detection) ---
    if activity_label in activity_feature_history and len(activity_feature_history[activity_label]) > 0:
        previous_vectors = np.array(activity_feature_history[activity_label])

        # Compute mean and standard deviation for normalization
        mean_vector = np.mean(previous_vectors, axis=0)
        std_vector = np.std(previous_vectors, axis=0)
        std_vector[std_vector == 0] = 1  # Avoid division by zero

        # Get selected features for this event
        selected_features = [column for column in data_columns if column != control_flow_column]

        # Compute feature deviations only for selected features
        new_feature_vector = np.array([
            float(event[column]) if column in event and isinstance(event[column], (int, float)) else 0.0
            for column in selected_features
        ], dtype=np.float64)

        if not isinstance(new_feature_vector, np.ndarray) or new_feature_vector.dtype != np.float64:
            print(f"[ERROR] Invalid feature vector for Event {event[event_id_column]}: {new_feature_vector}")

        try:
            deviations = np.abs(new_feature_vector - mean_vector) / std_vector
        except Exception as e:
            print(f"[ERROR] Failed to compute deviations for Event {event[event_id_column]}: {e}")
            deviations = np.zeros_like(new_feature_vector)  # Fallback

        features_to_update = {}
        vector_tuple = tuple(new_feature_vector)
        # Apply weight scaling for high deviation features with adaptive weighting
        previous_events_metadata = previous_events.get(case_id, deque(maxlen=previousEvents))

        for i, column in enumerate(selected_features):

            if isinstance(previous_events_metadata, deque) and len(previous_events_metadata) > 0:
                last_event = previous_events_metadata[-1]

                if isinstance(last_event, (int, float)):  # Ensure it's a valid numeric event counter
                    events_since_last_seen = last_event
                else:
                    print(f"[ERROR] Invalid last event type for case {case_id}: {type(last_event)}")
                    events_since_last_seen = global_event_counter
            else:
                events_since_last_seen = global_event_counter  # Default fallback

            # Compute adaptive feature weight
            feature_scores[column] += feature_weights[column] * deviations[i]

            # Store for batch updating later
            features_to_update[column] = deviations[i]

            # Update last seen event counter
            feature_last_seen_event[column] = global_event_counter

        # Perform batch weight update after scoring all features
        for feature, deviation in features_to_update.items():
            update_feature_weights(feature, deviation)

    ## --- 2. Control-Flow Perspective: Track Last Three Activities ---
    previous_activities = list(previous_events[case_id])  # Get the last three activities (if available)
    curr_activity = event[control_flow_column]

    # Ensure we always have three previous activities, filling gaps with "UNKNOWN"
    while len(previous_activities) < previousEvents:
        previous_activities.insert(0, "UNKNOWN")

    feature_scores["prev_activity_1"] = previous_activities[-3] if len(previous_activities) >= 3 else "UNKNOWN"
    feature_scores["prev_activity_2"] = previous_activities[-2] if len(previous_activities) >= 2 else "UNKNOWN"
    feature_scores["prev_activity_3"] = previous_activities[-1] if len(previous_activities) >= 1 else "UNKNOWN"

    directly_follows_graph.add_transition(case_id, previous_activities[-previousEvents:], curr_activity,
                                          global_event_counter)

    # Update the stored previous activities for this case
    previous_events[case_id].append(curr_activity)

    ## --- 3. Resource Perspective: Score role/resource-based context ---
    if resource_column in event:
        resource = event[resource_column]

        # Retrieve historical usage count for this resource in the given activity
        past_usage = resource_usage_history[activity_label][resource]

        # Compute inverse frequency score (rarer resources get higher weight)
        frequency_score = 1 / (past_usage + 1)

        # Assign score dynamically based on adaptive feature weight and past occurrence frequency
        feature_scores[resource_column] += feature_weights[resource_column] * frequency_score

        # Update resource usage count incrementally
        resource_usage_history[activity_label][resource] += 1

        # Update resource feature weight dynamically
        update_feature_weights(resource_column, frequency_score)

    ## --- 4. Time Perspective ---
    time_features = {key: event[key] for key in event.keys() if key.endswith("_bin")}

    # Encode categorical time features before using them in calculations
    if "day_of_week" in time_features:
        time_features["day_of_week"] = day_encoder.transform([time_features["day_of_week"]])[0]

    if "hour_bin" in time_features:
        time_features["hour_bin"] = hour_bin_encoder.transform([time_features["hour_bin"]])[0]

    if "month" in time_features:
        time_features["month"] = month_encoder.transform([time_features["month"]])[0]

    if "is_weekend" in time_features:
        time_features["is_weekend"] = 1 if time_features["is_weekend"] else 0

    # Compute time-based feature score with adaptive weights
    feature_scores["time_score"] = (
        feature_weights["hour_bin"] * time_features["hour_bin"] +
        feature_weights["day_of_week"] * time_features["day_of_week"] +
        feature_weights["month"] * time_features["month"] +
        feature_weights["is_weekend"] * time_features["is_weekend"]
    )

    # Update time-related feature weights
    update_feature_weights("hour_bin", time_features["hour_bin"])
    update_feature_weights("day_of_week", time_features["day_of_week"])
    update_feature_weights("month", time_features["month"])
    update_feature_weights("is_weekend", time_features["is_weekend"])

    ## --- 5. Data Perspective ---
    prev_event = previous_events.get(case_id)
    if prev_event:
        for column in data_columns:
            if column == control_flow_column:
                continue  # Ensure control_flow_column is NOT used as a feature

            if column in event and column in prev_event:
                base_score = feature_weights[column] * 0.6
                try:
                    if isinstance(event[column], (int, float)):
                        feature_scores[column] += base_score * (
                            1.2 if np.var([prev_event[column], event[column]]) > 0.05 else 1.0
                        )
                    elif isinstance(event[column], str):
                        feature_scores[column] += base_score * (
                            1.15 if prev_event[column] != event[column] else 1.0
                        )
                except Exception:
                    feature_scores[column] += base_score  # Fallback handling

                # Update feature weights
                update_feature_weights(column, feature_scores[column])

    ## Update previous events tracker for this CaseID
    previous_events[case_id] = event

    for i in range(1, previousEvents + 1):
        feature_scores[f"prev_activity_{i}"] = 0  # Assign 0 to prevent impact on feature importance

    return feature_scores

def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores and update feature weights dynamically.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    avg_score = np.mean(feature_scores) if len(feature_scores) > 0 else 0
    drift_detected = drift_detector[feature].update(avg_score)

    # If drift is detected, update feature weight
    if drift_detected:
        feature_weights[feature] = min(2.0, feature_weights[feature] * 1.1)  # Increase weight
    else:
        feature_weights[feature] = max(0.5, feature_weights[feature] * 0.9)  # Decrease weight slowly

    return drift_detected


def select_features(event, event_id_column, control_flow_column, timestamp_column, resource_column, data_columns,
                    global_event_counter):
    """
    Fast Online Feature Selection with Adaptive Weighting.
    """
    # Compute base feature scores
    feature_scores = compute_feature_scores(
        event, event_id_column, control_flow_column, timestamp_column, resource_column, case_id_column, data_columns,
        global_event_counter
    )

    # Retrieve the previous event from `previous_events`
    case_id = event[case_id_column]
    previous_event = previous_events.get(case_id, None)  # Get last seen event

    # Apply decay weighting to prioritize recent context
    current_time = event[timestamp_column]
    if isinstance(current_time, pd.Timestamp):
        current_time = current_time.timestamp()

    if previous_event:
        prev_time = previous_event[timestamp_column] if timestamp_column in previous_event else None
        if isinstance(prev_time, pd.Timestamp):
            prev_time = prev_time.timestamp()

        if prev_time is not None:
            for feature in feature_scores:
                feature_scores[feature] *= np.exp(-temporal_decay_rate * (current_time - prev_time))

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
