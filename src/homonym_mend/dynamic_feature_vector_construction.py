from collections import defaultdict
import numpy as np
import logging
from config.config import temporal_decay_rate, lossy_counting_budget, frequency_decay_threshold, decay_after_events, \
    removal_threshold_events
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.logging_utils import log_traceability

# --- GLOBAL VARIABLES ---

audit_log = []
encoders = defaultdict(CustomLabelEncoder)  # Automatically initializes a CustomLabelEncoder for each feature
activity_feature_history = defaultdict(list)
activity_feature_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": None}))
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
    """
    activity_label = event["Activity"]

    # Apply memory management before processing
    forget_old_feature_vectors(global_event_counter)

    new_vector = []
    for feature in top_features:
        value = event.get(feature)

        # Handle missing values
        if value is None:
            log_traceability("skip", activity_label, f"Missing value for feature '{feature}'")
            value = "MISSING"  # Default missing value for categorical features

        # Encode categorical features
        if isinstance(value, str) and value != "MISSING":  # Categorical feature
            encoded_value = encoders[feature].transform([value])[0]  # Ensure encoding returns a valid format
        elif isinstance(value, (int, float)):
            encoded_value = float(value)  # Ensure it's treated as a float
        else:
            encoded_value = 0.0  # Default numerical value for unexpected cases

        new_vector.append(encoded_value)

    # Ensure feature vector has the correct dimension
    if not new_vector or len(new_vector) != len(top_features):
        log_traceability("dimension_mismatch", activity_label, f"Feature vector has incorrect length.")
        return None

    # Normalize numerical values
    new_vector = normalize_feature_vector(np.array(new_vector).flatten())

    # Apply rounding to prevent floating-point precision errors
    rounded_vector = np.round(np.array(new_vector), decimals=5)  # Round to 5 decimal places

    # Compare against existing vectors for this activity label
    found_match = False
    for existing_vector in activity_feature_history[activity_label]:
        if np.array_equal(np.round(existing_vector, decimals=5), rounded_vector):  # Round before comparing
            activity_feature_metadata[activity_label][tuple(existing_vector)]["frequency"] += 1
            activity_feature_metadata[activity_label][tuple(existing_vector)]["recency"] = global_event_counter
            found_match = True
            break

    if not found_match:
        # Store new unique feature vector
        rounded_vector_tuple = tuple(rounded_vector) if isinstance(rounded_vector, (list, np.ndarray)) else (
        rounded_vector,)
        activity_feature_history[activity_label].append(rounded_vector)
        activity_feature_metadata[activity_label][rounded_vector_tuple] = {
            "frequency": 1,
            "recency": global_event_counter,
            "last_seen_event": global_event_counter  # ✅ Ensure last_seen_event is stored
        }

    # Log the event's new vector
    log_traceability("new_vector", activity_label, {
        "vector": rounded_vector,
        "features": top_features,
        "metadata": activity_feature_metadata[activity_label]
    })

    return {"activity_label": activity_label, "new_vector": rounded_vector}

