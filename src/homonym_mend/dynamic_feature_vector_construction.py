from collections import defaultdict, deque
import traceback
import numpy as np
import logging
from config.config import temporal_decay_rate, lossy_counting_budget, frequency_decay_threshold, decay_after_events, \
    removal_threshold_events, case_id_column, previousEvents
from src.utils.global_state import dbstream_clusters  # ✅ Import from global state
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import activity_feature_metadata, activity_feature_history, previous_events
from src.utils.logging_utils import log_traceability

# --- GLOBAL VARIABLES ---

audit_log = []
encoders = defaultdict(CustomLabelEncoder)  # Automatically initializes a CustomLabelEncoder for each feature

# --- FUNCTION DEFINITIONS ---
def apply_temporal_decay(value, time_difference):
    """
    Apply temporal decay to a value based on time difference.
    """
    return value * np.exp(-temporal_decay_rate * time_difference)


def encode_categorical_feature(feature, value):
    """
    Encode a categorical feature dynamically, adding unseen values.
    """
    try:
        return encoders[feature].transform(value)  # No need for manual initialization
    except Exception as e:
        log_traceability("error", "Feature Encoding", {"feature": feature, "value": value, "error": str(e)})
        return -1  # Fallback for encoding errors


def normalize_feature_vector(vector):
    """
    Normalize a numerical feature vector using min-max scaling.
    """
    if len(vector) == 0:
        return np.array(vector)

    min_val = np.min(vector)
    max_val = np.max(vector)

    if max_val - min_val == 0:
        return np.zeros_like(vector)  # Avoid division by zero

    return (vector - min_val) / (max_val - min_val)



def forget_old_feature_vectors(activity_label):
    """
    Forget feature vectors that have not been observed recently, based on decayed frequency and event count.
    Ensures metadata is initialized before access and removes only outdated entries without clearing the entire structure.
    """
    try:
        # Ensure metadata is initialized before checking
        if activity_label not in activity_feature_metadata:
            activity_feature_metadata[activity_label] = {}  # Initialize metadata
            log_traceability("warning", "Feature Metadata", {"activity_label": activity_label, "warning": ""
                                                                                                          ", initialized now."})
            return  # Nothing to forget if it's newly initialized

        # Sort feature vectors based on frequency and last seen event
        sorted_vectors = sorted(
            activity_feature_metadata[activity_label].items(),
            key=lambda x: (x[1]["frequency"], x[1]["last_seen_event"])
        )

        # Debugging: Print metadata before cleanup
        print(f"[DEBUG] Before cleanup: {activity_label} metadata keys: {list(activity_feature_metadata[activity_label].keys())}")

        # Identify outdated feature vectors
        outdated_keys = []
        for vector_tuple, metadata in sorted_vectors:
            last_seen = metadata["last_seen_event"]
            activity_event_count = dbstream_clusters[activity_label].activity_event_counters.get(activity_label, 0)
            events_since_last_seen = activity_event_count - last_seen

            # Apply exponential decay on frequency
            metadata["frequency"] *= np.exp(-events_since_last_seen / max(decay_after_events, 1))

            # Mark outdated vectors for removal
            if metadata["frequency"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
                outdated_keys.append(vector_tuple)

        # Remove outdated feature vectors
        for key in outdated_keys:
            del activity_feature_metadata[activity_label][key]
            if key in activity_feature_history.get(activity_label, []):
                activity_feature_history[activity_label].remove(list(key))

        # Debugging: Print remaining metadata after cleanup
        print(f"[DEBUG] After cleanup, remaining metadata keys for {activity_label}: {list(activity_feature_metadata[activity_label].keys())}")

        # Stop if we are within the lossy counting budget
        if len(activity_feature_metadata[activity_label]) <= lossy_counting_budget:
            return

    except Exception as e:
        log_traceability("error", "Feature Forgetting", {"error": str(e)})
        traceback.print_exc()


def infer_feature_types(event_sample):
    """
    Infer feature types (categorical/numeric) dynamically from a sample event.
    :param event_sample: A dictionary of sample event attributes.
    :return: A dictionary mapping feature names to their types.
    """
    feature_types = {}

    for feature, value in event_sample.items():
        if isinstance(value, (int, float)):  # Numeric features
            feature_types[feature] = "numeric"
        else:  # Categorical features (default)
            feature_types[feature] = "categorical"

    log_traceability("debug", "Feature Type Inference", {"feature_types": feature_types})
    return feature_types


def construct_feature_vector(event, feature_metadata):
    """
    Construct a dynamic feature vector from an event based on feature metadata.
    :param event: The event dictionary containing attributes.
    :param feature_metadata: A dictionary containing feature types (categorical, numeric, etc.).
    :return: A list representing the feature vector.
    """
    feature_vector = []

    for feature, feature_type in feature_metadata.items():
        if feature in event:
            value = event[feature]

            if feature_type == "categorical":
                encoded_value = encode_categorical_feature(feature, value)  # Use encoding for consistency
                feature_vector.append(encoded_value)

            elif feature_type == "numeric":
                feature_vector.append(float(value))  # Ensure numeric conversion for consistency

    log_traceability("debug", "Feature Vector Construction", {"feature_vector": feature_vector})
    return feature_vector


def process_event(event, top_features, global_event_counter):
    """Process an event to construct and analyze dynamic feature vectors."""
    try:
        activity_label = event.get("Activity")
        if not activity_label:
            print(f"[ERROR] Missing activity label for event")
            return None

        # Apply memory management before processing
        forget_old_feature_vectors(activity_label)

        # Get the case ID for previous activities lookup
        case_id = event.get(case_id_column)

        # Get previous activities safely
        prev_events_data = previous_events.get(case_id, deque(maxlen=previousEvents))
        previous_activities = []

        # Extract activities from previous events
        if isinstance(prev_events_data, deque):
            for prev_event in prev_events_data:
                if isinstance(prev_event, dict):
                    activity = prev_event.get("Activity")
                    if activity:
                        previous_activities.append(str(activity))
                elif isinstance(prev_event, str):
                    previous_activities.append(str(prev_event))

        # Fill missing previous activities with "UNKNOWN"
        while len(previous_activities) < previousEvents:
            previous_activities.insert(0, "UNKNOWN")

        # Take only the last previousEvents activities
        previous_activities = previous_activities[-previousEvents:]

        # Infer feature types dynamically
        feature_metadata = infer_feature_types(event)

        # Construct feature vector using inferred types
        new_vector = construct_feature_vector(event, feature_metadata)

        # Ensure all values are numeric
        new_vector = [float(x) if x is not None else 0.0 for x in new_vector]

        # Normalize the vector
        normalized_vector = normalize_feature_vector(np.array(new_vector))
        normalized_vector = np.array(normalized_vector, dtype=np.float64)

        # After creating normalized_vector, update metadata
        vector_tuple = tuple(normalized_vector.tolist())

        if activity_label not in activity_feature_metadata:
            log_traceability("debug", "Feature Metadata Init", {
                "activity_label": activity_label,
                "reason": "First occurrence, initializing."
            })
            activity_feature_metadata[activity_label] = {}  # Initialize metadata storage
        else:
            log_traceability("debug", "Feature Metadata Access", {
                "activity_label": activity_label,
                "existing_keys": list(activity_feature_metadata[activity_label].keys())
            })

        if vector_tuple not in activity_feature_metadata[activity_label]:
            activity_feature_metadata[activity_label][vector_tuple] = {
                "frequency": 1,
                "last_seen_event": global_event_counter
            }
        else:
            metadata = activity_feature_metadata[activity_label][vector_tuple]
            metadata["frequency"] = metadata.get("frequency", 0) + 1
            metadata["last_seen_event"] = global_event_counter

        # ✅ ADD DEBUG LOGGING TO CHECK METADATA UPDATE
        print(f"[DEBUG] Updating metadata for {activity_label}: {activity_feature_metadata[activity_label]}")

        return {
            "activity_label": activity_label,
            "new_vector": normalized_vector,  # ✅ Ensure consistent NumPy format
            "top_features": top_features
        }

    except Exception as e:
        print(f"[ERROR] Failed to process event: {str(e)}")
        traceback.print_exc()
        return None


