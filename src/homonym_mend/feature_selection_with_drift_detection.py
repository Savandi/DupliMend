import traceback
from river.drift import ADWIN
from config.config import (initial_window_size,
                           max_top_n_features, temporal_decay_rate, case_id_column, frequency_decay_threshold,
                           decay_after_events, removal_threshold_events, previousEvents
                           )
from src.homonym_mend.dynamic_binning_and_categorization import time_distribution
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import *
from collections import defaultdict
import numpy as np
from src.utils.global_state import previous_events

# Initialize feature tracking structures
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
day_encoder = CustomLabelEncoder()
hour_bin_encoder = CustomLabelEncoder()
month_encoder = CustomLabelEncoder()
drift_detector = defaultdict(ADWIN)
feature_last_seen_event = {}  # Track last-seen event count for each feature
feature_accumulations = defaultdict(dict)  # Store feature accumulations per case (for adaptive forgetting)

def forget_old_cases(activity_label):
    """
    Remove inactive cases and outdated feature tracking using event-driven forgetting.
    Ensures that expired cases no longer influence historical tracking.

    This removes:
    - Control-Flow records (previous activities per case)
    - Resource usage history (past occurrences of resources)
    - Time distribution history (past timestamps)
    - Feature tracking (event data history per activity)

    Uses `activity_event_counters` for adaptive forgetting instead of a global counter.
    """

    # Get event count for the activity label
    activity_event_count = activity_event_counters.get(activity_label, 0)

    for case_id in list(previous_events.keys()):
        prev_events_data = previous_events.get(case_id, deque(maxlen=previousEvents))
        if not prev_events_data:
            continue

        last_event = prev_events_data[-1] if prev_events_data else None

        if isinstance(last_event, dict) and case_id_column in last_event:
            last_event_counter = activity_event_count
        elif isinstance(last_event, (int, float)):
            last_event_counter = last_event
        else:
            last_event_counter = activity_event_count

        events_since_last_seen = activity_event_count - last_event_counter
        frequency = np.exp(-events_since_last_seen / max(decay_after_events, 1))

        if frequency < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
            del previous_events[case_id]
            directly_follows_graph.remove_case_transitions(case_id)

            if activity_label in resource_usage_history:
                del resource_usage_history[activity_label]
            if activity_label in time_distribution:
                del time_distribution[activity_label]
            if activity_label in activity_feature_history:
                activity_feature_history[activity_label] = [
                    v for v in activity_feature_history[activity_label] if v.get(case_id_column) != case_id
                ]
            if activity_label in feature_last_seen_event:
                del feature_last_seen_event[activity_label]
            if case_id in feature_accumulations:
                del feature_accumulations[case_id]

            print(f"[DEBUG] Removed expired case {case_id} from tracking")




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

def update_feature_weights(activity_label, feature, new_score):
    """
    Dynamically updates feature weights based on drift and event count decay.
    """

    # Ensure feature exists in the weights dictionary
    if feature not in feature_weights:
        feature_weights[feature] = 1.0

    # Ensure activity_label is tracked correctly
    activity_event_count = activity_event_counters.get(activity_label, 0)
    last_seen_event = feature_last_seen_event.get(feature, activity_event_count)

    events_since_last_seen = activity_event_count - last_seen_event
    decay_factor = np.exp(-events_since_last_seen / max(decay_after_events, 1))

    # Apply adaptive weight update
    feature_weights[feature] = (feature_weights[feature] * decay_factor) + (new_score * (1 - decay_factor))

    # Update last seen event counter
    feature_last_seen_event[feature] = activity_event_count

    print(f"[DEBUG] Updated feature weight for {feature}: {feature_weights[feature]} (Decay Factor: {decay_factor})")


def compute_feature_scores(event, event_id_column, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
    """
    Computes feature scores dynamically, applying adaptive forgetting and temporal decay.
    """
    feature_scores = defaultdict(float)
    case_id = event[case_id_column]
    activity_label = event.get(control_flow_column, "UNKNOWN")

    # ✅ Track per-activity event counts adaptively
    activity_event_counters[activity_label] += 1

    # ✅ Forget old cases based on adaptive decay
    forget_old_cases(activity_label)

    # ✅ Retrieve previous events safely
    previous_activities = []
    prev_events_data = previous_events.get(case_id, deque(maxlen=previousEvents))

    if isinstance(prev_events_data, deque):
        previous_activities = [
            prev_event.get(control_flow_column, "UNKNOWN") if isinstance(prev_event, dict) else prev_event
            for prev_event in prev_events_data
        ]

    # Ensure we always maintain `previousEvents` activities
    previous_activities = (["UNKNOWN"] * previousEvents + previous_activities)[-previousEvents:]

    # ✅ Process timestamp
    current_time = pd.to_datetime(event[timestamp_column], errors='coerce', utc=True)
    prev_time = pd.to_datetime(prev_events_data[-1][timestamp_column], errors='coerce', utc=True) if prev_events_data else None

    # ✅ Apply temporal decay based on previous timestamps
    if prev_time is not None and isinstance(prev_time, pd.Timestamp):
        time_diff = (current_time - prev_time).total_seconds()
        for feature in feature_scores:
            if isinstance(feature_scores[feature], (int, float)):
                feature_scores[feature] *= np.exp(-temporal_decay_rate * time_diff)

    ## --- 2️⃣ Control-Flow Perspective: Track Previous Activities ---
    for i in range(previousEvents):
        feature_name = f"prev_activity_{i+1}"
        feature_scores[feature_name] = 0.0  # ✅ Ensure numeric value for previous activities

    # ✅ Update Directly-Follows Graph
    try:
        prev_activities_tuple = tuple(previous_activities)
        directly_follows_graph.add_transition(case_id, prev_activities_tuple, activity_label, global_event_counter)
    except Exception as e:
        print(f"[WARNING] Failed to update directly follows graph: {e}")

    previous_events.setdefault(case_id, deque(maxlen=previousEvents)).append(event)

    ## --- 3️⃣ Resource Perspective ---
    if resource_column in event:
        resource = event[resource_column]
        past_usage = resource_usage_history[activity_label].get(resource, 0)
        frequency_score = 1 / (past_usage + 1)
        feature_scores[resource_column] += feature_weights.get(resource_column, 1.0) * frequency_score
        resource_usage_history[activity_label][resource] += 1
        update_feature_weights(activity_label, resource_column, frequency_score)

    ## --- 4️⃣ Time Perspective ---
    time_features = {}
    time_encoding_map = {
        "hour_bin": hour_bin_encoder,
        "day_of_week": day_encoder,
        "month": month_encoder
    }

    for time_feature, encoder in time_encoding_map.items():
        if time_feature in event:
            try:
                encoded_value = encoder.transform([str(event[time_feature])])[0]
                time_features[time_feature] = float(encoded_value)  # ✅ Ensure numeric type
            except Exception as e:
                print(f"[WARNING] Failed to encode {time_feature}: {e}")
                time_features[time_feature] = 0.0

    time_features["is_weekend"] = 1.0 if event.get("is_weekend", False) else 0.0  # ✅ Ensure numeric type

    # ✅ Compute weighted time score
    feature_scores["time_score"] = sum(
        feature_weights.get(feat, 1.0) * time_features.get(feat, 0.0)
        for feat in time_features
    )

    for feature_name in time_features:
        update_feature_weights(activity_label, feature_name, time_features[feature_name])

    ## --- 5️⃣ Data Perspective: Compare Current Event with Previous Events for Same Case ---
    prev_event = previous_events.get(case_id)
    if prev_event:
        for column in data_columns:
            if column == control_flow_column:
                continue

            if column in event and column in prev_event:
                base_score = feature_weights.get(column, 1.0) * 0.6
                try:
                    if isinstance(event[column], (int, float)):
                        feature_scores[column] += base_score * (
                            1.2 if np.var([prev_event[column], event[column]]) > 0.05 else 1.0
                        )
                    if isinstance(event[column], str):
                        encoded_value = hash(event[column]) % 1000  # Convert string to a numeric representation
                        prev_encoded_value = hash(prev_event[column]) % 1000 if column in prev_event else 0

                        difference = abs(encoded_value - prev_encoded_value) / 1000.0  # Normalize difference
                        feature_scores[column] += base_score * (1.15 if difference > 0.1 else 1.0)
                except Exception:
                    feature_scores[column] += base_score  # ✅ Fallback for robustness

                update_feature_weights(activity_label, column, feature_scores[column])

    # ✅ Convert all scores to floats for consistency & fix non-numeric values
    feature_scores = {key: float(value) if isinstance(value, (int, float)) else 0.0 for key, value in feature_scores.items()}

    return feature_scores


def detect_drift(feature, feature_scores):
    """
    Uses ADWIN to detect drift in feature scores and dynamically adjust feature weights.
    """
    if feature not in drift_detector:
        drift_detector[feature] = ADWIN()

    avg_score = np.mean(list(feature_scores.values())) if feature_scores else 0
    drift_detected = drift_detector[feature].update(avg_score)

    if drift_detected:
        feature_weights[feature] = min(2.0, feature_weights[feature] * 1.1)
    else:
        feature_weights[feature] = max(0.5, feature_weights[feature] * 0.9)

    return drift_detected



def select_features(event, event_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
    """
    Selects the most relevant features dynamically, ensuring adaptive forgetting is applied.
    """
    try:
        feature_scores = compute_feature_scores(event, event_id_column, case_id_column, control_flow_column,
                                                timestamp_column, resource_column, data_columns, global_event_counter)
        top_n = int(min(adaptive_threshold(feature_scores), max_top_n_features))

        valid_scores = {k: v for k, v in feature_scores.items() if isinstance(v, (int, float))}
        selected_features = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [feature for feature, _ in selected_features]

    except Exception as e:
        print(f"[ERROR] Failed to select features: {str(e)}")
        traceback.print_exc()
        return []



def adaptive_threshold(feature_scores):
    """
    Compute an adaptive threshold for feature selection based on the variability in feature scores.
    """
    # ✅ Convert scores to a numeric array, ignoring non-numeric values
    numeric_scores = [score for score in feature_scores.values() if isinstance(score, (int, float))]

    if not numeric_scores:  # ✅ Avoid empty lists
        return 1.0  # Default threshold if no valid scores exist

    variability = np.std(numeric_scores)  # ✅ Now only operates on numeric values

    # Compute the adaptive threshold using variability
    adaptive_threshold = min(max(variability / 10.0, 0.1), 1.5)

    return adaptive_threshold

# def compute_feature_scores(event, event_id_column, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
#     feature_scores = defaultdict(float)
#     case_id = event[case_id_column]
#     if control_flow_column in event:
#         activity_label = event[control_flow_column]
#     else:
#         print(f"[ERROR] Missing control flow column {control_flow_column} in event: {event}")
#         return feature_scores
#
#     # Initialize previous activities list
#     previous_activities = []
#     prev_event = None
#
#     # Get previous event data safely
#     prev_events_data = previous_events.get(case_id, deque(maxlen=previousEvents))
#     if isinstance(prev_events_data, deque) and len(prev_events_data) > 0:
#         for prev_event in prev_events_data:
#             if isinstance(prev_event, dict) and control_flow_column in prev_event:
#                 previous_activities.append(prev_event[control_flow_column])
#             elif isinstance(prev_event, str):
#                 previous_activities.append(prev_event)
#
#     # Ensure we always have three previous activities
#     while len(previous_activities) < previousEvents:
#         previous_activities.insert(0, "UNKNOWN")
#     previous_activities = previous_activities[-previousEvents:]
#
#     # Process timestamp
#     current_time = event[timestamp_column]
#     if not isinstance(current_time, pd.Timestamp):
#         try:
#             current_time = pd.to_datetime(current_time)
#         except Exception as e:
#             print(f"[ERROR] Invalid timestamp for Event {event[event_id_column]}: {e}")
#             return {}
#
#     # Apply temporal decay
#     if prev_event and timestamp_column in prev_event:
#         try:
#             prev_time = prev_event[timestamp_column]
#             if isinstance(prev_time, pd.Timestamp):
#                 time_diff = (current_time - prev_time).total_seconds()
#                 for feature in feature_scores:
#                     if isinstance(feature_scores[feature], (int, float)):
#                         feature_scores[feature] *= np.exp(-temporal_decay_rate * time_diff)
#         except Exception as e:
#             print(f"[WARNING] Failed to apply temporal decay: {e}")
#
#     # Forget old cases before processing new events
#     forget_old_cases(global_event_counter, activity_label)
#
#     ## --- 1. Compare Against All Past Events of the Same Activity Label (Homonym Detection) ---
#     if activity_label in activity_feature_history and len(activity_feature_history[activity_label]) > 0:
#         previous_vectors = np.array(activity_feature_history[activity_label])
#
#         # Compute mean and standard deviation for normalization
#         mean_vector = np.mean(previous_vectors, axis=0)
#         std_vector = np.std(previous_vectors, axis=0)
#         std_vector[std_vector == 0] = 1  # Avoid division by zero
#
#         selected_features = [column for column in data_columns if column != control_flow_column]
#
#         new_feature_vector = np.array([
#             float(event[column]) if column in event and isinstance(event[column], (int, float)) else 0.0
#             for column in selected_features
#         ], dtype=np.float64)
#
#         if not isinstance(new_feature_vector, np.ndarray) or new_feature_vector.dtype != np.float64:
#             print(f"[ERROR] Invalid feature vector for Event {event[event_id_column]}: {new_feature_vector}")
#
#         try:
#             deviations = np.abs(new_feature_vector - mean_vector) / std_vector
#         except Exception as e:
#             print(f"[ERROR] Failed to compute deviations for Event {event[event_id_column]}: {e}")
#             deviations = np.zeros_like(new_feature_vector)
#
#         features_to_update = {}
#
#         for i, column in enumerate(selected_features):
#             feature_scores[column] += feature_weights.get(column, 1.0) * deviations[i]
#             features_to_update[column] = deviations[i]
#             feature_last_seen_event[column] = global_event_counter
#
#         # Batch update feature weights
#         for feature, deviation in features_to_update.items():
#             update_feature_weights(activity_label, feature, deviation)  # ✅ Pass activity_label
#
#     ## --- 2. Control-Flow Perspective ---
#     curr_activity = activity_label
#
#     for i in range(previousEvents):
#         feature_name = f"prev_activity_{i+1}"
#         feature_scores[feature_name] = previous_activities[i] if i < len(previous_activities) else "UNKNOWN"
#
#     try:
#         prev_activities_tuple = tuple(previous_activities)
#         directly_follows_graph.add_transition(case_id, prev_activities_tuple, curr_activity, global_event_counter)
#     except Exception as e:
#         print(f"[WARNING] Failed to update directly follows graph: {e}")
#
#     if case_id not in previous_events:
#         previous_events[case_id] = deque(maxlen=previousEvents)
#
#     previous_events[case_id].append(event)
#
#     print(f"Resource Perspective")
#     ## --- 3. Resource Perspective ---
#     if resource_column in event:
#         resource = event[resource_column]
#         past_usage = resource_usage_history[activity_label].get(resource, 0)
#         frequency_score = 1 / (past_usage + 1)
#         feature_scores[resource_column] += feature_weights.get(resource_column, 1.0) * frequency_score
#         resource_usage_history[activity_label][resource] += 1
#         update_feature_weights(activity_label, resource_column, frequency_score)  # Fixed
#
#     print(f"Time Perspective")
#     ## --- 4. Time Perspective ---
#     time_features = {}
#
#     if "hour_bin" in event:
#         try:
#             time_features["hour_bin"] = hour_bin_encoder.transform([event["hour_bin"]])[0]
#         except Exception as e:
#             print(f"[WARNING] Failed to encode hour_bin: {e}")
#             time_features["hour_bin"] = 0
#
#     if "day_of_week" in event:
#         try:
#             time_features["day_of_week"] = day_encoder.transform([str(event["day_of_week"])])[0]
#         except Exception as e:
#             print(f"[WARNING] Failed to encode day_of_week: {e}")
#             time_features["day_of_week"] = 0
#
#     if "month" in event:
#         try:
#             time_features["month"] = month_encoder.transform([str(event["month"])])[0]
#         except Exception as e:
#             print(f"[WARNING] Failed to encode month: {e}")
#             time_features["month"] = 0
#
#     time_features["is_weekend"] = 1 if event.get("is_weekend", False) else 0
#
#     feature_scores["time_score"] = (
#         feature_weights.get("hour_bin", 1.0) * time_features.get("hour_bin", 0) +
#         feature_weights.get("day_of_week", 1.0) * time_features.get("day_of_week", 0) +
#         feature_weights.get("month", 1.0) * time_features.get("month", 0) +
#         feature_weights.get("is_weekend", 1.0) * time_features.get("is_weekend", 0)
#     )
#
#     for feature_name in ["hour_bin", "day_of_week", "month", "is_weekend"]:
#         if feature_name in time_features:
#             update_feature_weights(activity_label, feature_name, time_features[feature_name])  # ✅ Fixed
#
#     print(f"Data Perspective")
#     ## --- 5. Data Perspective ---
#     prev_event = previous_events.get(case_id)
#     if prev_event:
#         for column in data_columns:
#             if column == control_flow_column:
#                 continue
#
#             if column in event and column in prev_event:
#                 base_score = feature_weights.get(column, 1.0) * 0.6
#                 try:
#                     if isinstance(event[column], (int, float)):
#                         feature_scores[column] += base_score * (
#                             1.2 if np.var([prev_event[column], event[column]]) > 0.05 else 1.0
#                         )
#                     elif isinstance(event[column], str):
#                         feature_scores[column] += base_score * (
#                             1.15 if prev_event[column] != event[column] else 1.0
#                         )
#                 except Exception:
#                     feature_scores[column] += base_score  # Fallback
#
#                 update_feature_weights(activity_label, column, feature_scores[column])  # ✅ Fixed
#
#     feature_scores = {key: float(value) if isinstance(value, (int, float)) else 0.0 for key, value in
#                       feature_scores.items()}
#
#     print(f"Out of Feature Scores")
#     return feature_scores