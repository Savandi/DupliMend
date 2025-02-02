from collections import defaultdict
from datetime import datetime
import numpy as np
import logging
from config.config import temporal_decay_rate
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.logging_utils import log_traceability

# --- GLOBAL VARIABLES ---
event_counter = defaultdict(int)  # Track events per activity
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


def normalize_feature_vector(vector):
    """
    Normalize a numerical feature vector using mean and standard deviation.
    """
    if len(vector) == 0:
        return np.array(vector)

    mean = np.mean(vector)
    std = np.std(vector)

    if std == 0:
        return np.zeros_like(vector)

    return (vector - mean) / std


def process_event(event, top_features):
    """
    Process an event to construct and analyze dynamic feature vectors.
    """
    activity_label = event["Activity"]

    new_vector = []
    for feature in top_features:
        value = event.get(feature)

        # Handle missing values
        if value is None:
            log_traceability("skip", activity_label, f"Missing value for feature '{feature}'")
            value = "MISSING"  # Default missing value for categorical features

        # Encode categorical features
        if isinstance(value, str):  # Categorical feature
            encoded_value = encoders[feature].transform(value)
        else:  # Numerical feature
            encoded_value = float(value)  # Ensure it's treated as a float

        new_vector.append(encoded_value)

    # Ensure feature vector has the correct dimension
    if not new_vector or len(new_vector) != len(top_features):
        log_traceability("dimension_mismatch", activity_label, f"Feature vector has incorrect length.")
        return None

    # Normalize numerical values
    new_vector = normalize_feature_vector(np.array(new_vector).flatten())

    # Compare against existing vectors for this activity label
    found_match = False
    for existing_vector in activity_feature_history[activity_label]:
        if np.array_equal(existing_vector, new_vector):  # Check if the vector is an exact match
            activity_feature_metadata[activity_label][tuple(existing_vector)]["frequency"] += 1
            activity_feature_metadata[activity_label][tuple(existing_vector)]["recency"] = datetime.now()
            found_match = True
            break

    if not found_match:
        # Store new unique feature vector
        activity_feature_history[activity_label].append(new_vector)
        activity_feature_metadata[activity_label][tuple(new_vector)] = {"frequency": 1, "recency": datetime.now()}

    # Log the event's new vector
    log_traceability("new_vector", activity_label, {
        "vector": new_vector,
        "features": top_features,
        "metadata": activity_feature_metadata[activity_label]
    })

    return {"activity_label": activity_label, "new_vector": new_vector}