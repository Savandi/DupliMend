from collections import defaultdict
import numpy as np
import logging
from config.config import temporal_decay_rate, lossy_counting_budget, frequency_decay_threshold, decay_after_events, \
    removal_threshold_events, case_id_column, previousEvents
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import activity_feature_metadata, activity_feature_history, previous_events
from src.utils.logging_utils import log_traceability

# --- GLOBAL VARIABLES ---

audit_log = []
encoders = defaultdict(CustomLabelEncoder)  # Automatically initializes a CustomLabelEncoder for each feature

# --- LOGGING CONFIGURATION ---
try:
    logging.basicConfig(
        filename="../../traceability_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
except PermissionError:
    logging.basicConfig(
        filename="traceability_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    print("Permission denied for logging to '../../traceability_log.txt'. Using local log file instead.")


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
        logging.error(f"Failed to encode feature '{feature}' with value '{value}': {str(e)}")
        return -1  # Fallback for encoding errors


def normalize_scores(feature_scores):
    """
    Normalize feature scores using min-max normalization.
    """
    values = np.array(list(feature_scores.values()))
    min_val, max_val = values.min(), values.max()

    if max_val - min_val == 0:
        return feature_scores  # Avoid division by zero

    for key in feature_scores:
        feature_scores[key] = (feature_scores[key] - min_val) / (max_val - min_val)

    return feature_scores

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


def forget_old_feature_vectors(global_event_counter):
    """
    Forget feature vectors that have not been observed recently, based on decayed frequency and event count.
    """
    for activity_label in list(activity_feature_metadata.keys()):
        sorted_vectors = sorted(
            activity_feature_metadata[activity_label].items(),
            key=lambda x: (x[1]["frequency"], x[1]["last_seen_event"])
        )

        for vector_tuple, metadata in sorted_vectors:
            events_since_last_seen = global_event_counter - metadata.get("last_seen_event", global_event_counter)
            metadata["frequency"] *= np.exp(-events_since_last_seen / decay_after_events)

            if metadata["frequency"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
                del activity_feature_metadata[activity_label][vector_tuple]
                activity_feature_history[activity_label].remove(list(vector_tuple))

            if len(activity_feature_metadata[activity_label]) <= lossy_counting_budget:
                break


def process_event(event, top_features, global_event_counter):
    """
    Process an event to construct and analyze dynamic feature vectors.
    Ensures:
    - `previousEvents` activities are always included in the vector.
    - Missing previous activities are set to "UNKNOWN".
    - Feature selection does not remove `previousEvents`.
    """
    activity_label = event["Activity"]

    # Apply memory management before processing
    forget_old_feature_vectors(global_event_counter)

    new_vector = {}

    # Ensure previousEvents activities are always included
    previous_activities = list(previous_events[event[case_id_column]])

    while len(previous_activities) < previousEvents:
        previous_activities.insert(0, "UNKNOWN")  # Fill missing slots

    for i in range(1, previousEvents + 1):
        new_vector[f"prev_activity_{i}"] = previous_activities[-i]  # Always included in the vector

    # Include only top selected features from feature selection
    for feature in top_features:
        if feature not in new_vector:  # Avoid duplication
            value = event.get(feature, "UNKNOWN")
            if isinstance(value, str):  # Categorical feature
                encoded_value = encode_categorical_feature(feature, value)
            else:  # Numerical feature
                encoded_value = float(value)
            new_vector[feature] = encoded_value

    # Convert dictionary to ordered list for vector representation
    vector_values = list(new_vector.values())

    # Normalize numerical values
    vector_values = normalize_feature_vector(np.array(vector_values).flatten())

    # Round to prevent floating-point precision errors
    rounded_vector = np.round(vector_values, decimals=5)
    # Ensure rounded_vector is always a 1D array
    rounded_vector = np.atleast_1d(rounded_vector)  # Converts scalar to array if needed

    if rounded_vector.size == 0:
        rounded_vector = np.array([0])

    rounded_vector_tuple = tuple(rounded_vector.tolist())  # Ensures tuple conversion

    if rounded_vector_tuple in activity_feature_metadata[activity_label]:
        # Update existing metadata
        activity_feature_metadata[activity_label][rounded_vector_tuple]["frequency"] += 1
        activity_feature_metadata[activity_label][rounded_vector_tuple]["recency"] = global_event_counter
        activity_feature_metadata[activity_label][rounded_vector_tuple]["last_seen_event"] = global_event_counter
    else:
        # Store new vector metadata without storing the vector itself
        activity_feature_metadata[activity_label][rounded_vector_tuple] = {
            "frequency": 1,
            "last_seen_event": global_event_counter
        }

    # Log the event's new vector
    log_traceability("new_vector", activity_label, {
        "vector": rounded_vector,
        "features": top_features,
        "metadata": activity_feature_metadata[activity_label]
    })

    return {"activity_label": activity_label, "new_vector": rounded_vector}



