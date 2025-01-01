from collections import defaultdict
from datetime import datetime
import numpy as np
import logging
from config.config import temporal_decay_rate
from src.utils.custom_label_encoder import CustomLabelEncoder


# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))
event_counter = defaultdict(int)  # Track events per activity
audit_log = []
encoders = defaultdict()  # Encoders for categorical features

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
    if feature not in encoders:
        encoders[feature] = CustomLabelEncoder()
    encoder = encoders[feature]
    try:
        encoded_value = encoder.transform(value)
    except Exception as e:
        logging.error(f"Failed to encode feature '{feature}' with value '{value}': {str(e)}")
        encoded_value = -1  # Fallback for encoding errors
    return encoded_value

def log_traceability(action, activity_label, details):
    """
    Log traceability and auditability details.
    """
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "action": action, "activity_label": activity_label, "details": details}
    audit_log.append(entry)
    logging.info(f"{action.upper()} - {activity_label}: {details}")

def process_event(event, top_features, timestamp_column):
    """
    Process an event to construct and analyze dynamic feature vectors.
    """
    activity_label = event["Activity"]
    event_counter[activity_label] += 1

    new_vector = []
    for feature in top_features:
        value = event.get(feature)
        if value is None:
            log_traceability("skip", activity_label, f"Missing value for feature '{feature}'")
            continue

        encoded_value = encode_categorical_feature(feature, value)
        new_vector.append(encoded_value)

    if not new_vector:
        log_traceability("empty_vector", activity_label, "Skipped processing due to empty vector.")
        return None

    # Ensure consistent dimensionality
    new_vector = np.array(new_vector).flatten()
    if new_vector.shape[0] != len(top_features):
        log_traceability(
            "dimension_mismatch", activity_label,
            f"Vector dimensions {new_vector.shape[0]} do not match expected {len(top_features)}"
        )
        return None

    log_traceability("new_vector", activity_label, new_vector)

    # Return relevant data for the next step
    return {"activity_label": activity_label, "new_vector": new_vector}
