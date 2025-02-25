from collections import defaultdict, deque
import traceback
import numpy as np
import logging
from config.config import temporal_decay_rate, lossy_counting_budget, frequency_decay_threshold, decay_after_events, \
    removal_threshold_events, case_id_column, previousEvents
from src.utils.global_state import dbstream_clusters  # ✅ Import from global state
from src.utils.custom_label_encoder import CustomLabelEncoder
from src.utils.global_state import activity_feature_metadata, activity_feature_history, previous_events

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


def forget_old_feature_vectors(activity_label):
    """
    Forget feature vectors that have not been observed recently, based on decayed frequency and event count.
    """
    try:
        if activity_label not in activity_feature_metadata:
            return  # No features to forget

        sorted_vectors = sorted(
            activity_feature_metadata[activity_label].items(),
            key=lambda x: (x[1]["frequency"], x[1]["last_seen_event"])
        )

        for vector_tuple, metadata in sorted_vectors:
            last_seen = metadata["last_seen_event"]
            activity_event_count = dbstream_clusters[activity_label].activity_event_counters[
                activity_label]  # Per-activity counter

            events_since_last_seen = activity_event_count - last_seen  # Per-activity based event count
            metadata["frequency"] *= np.exp(-events_since_last_seen / decay_after_events)  # Adaptive forgetting

            # Remove if frequency drops below threshold
            if metadata["frequency"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
                try:
                    del activity_feature_metadata[activity_label][vector_tuple]
                    activity_feature_history[activity_label].remove(list(vector_tuple))
                except (KeyError, ValueError) as e:
                    print(f"[WARNING] Error removing vector: {e}")

            # Stop if we're within budget
            if len(activity_feature_metadata[activity_label]) <= lossy_counting_budget:
                break

    except Exception as e:
        print(f"[ERROR] Error in forget_old_feature_vectors: {e}")
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

    logging.debug(f"Inferred Feature Types: {feature_types}")
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
                feature_vector.append(value)  # Directly use categorical value

            elif feature_type == "numeric":
                feature_vector.append(value)  # ✅ No re-binning, assume value is already categorized

    logging.debug(f"Constructed Feature Vector: {feature_vector}")
    return feature_vector


def process_event(event, top_features, global_event_counter):
    """Process an event to construct and analyze dynamic feature vectors."""
    try:
        activity_label = event.get("Activity")
        if not activity_label:
            print(f"[ERROR] Missing activity label for event")
            return None

        # Apply memory management before processing
        forget_old_feature_vectors(global_event_counter)

        # Initialize vector with numeric values
        new_vector = []

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

        # Add previous activities to the vector using integer encoding
        for prev_activity in previous_activities:
            try:
                # Convert to list for CustomLabelEncoder
                encoded_value = encoders["previous_activity"].transform([str(prev_activity)])[0]
                new_vector.append(float(encoded_value))
            except Exception as e:
                print(f"[WARNING] Failed to encode previous activity {prev_activity}: {e}")
                new_vector.append(0.0)

        # Process selected features
        for feature in top_features:
            if feature.startswith("prev_activity"):
                continue  # Skip as we've already handled previous activities
                
            value = event.get(feature)
            if value is None:
                new_vector.append(0.0)
                continue

            try:
                if isinstance(value, str):
                    # Convert to list and ensure string type for CustomLabelEncoder
                    encoded_value = encoders[feature].transform([str(value)])[0]
                    new_vector.append(float(encoded_value))
                elif isinstance(value, (int, float)):
                    new_vector.append(float(value))
                else:
                    print(f"[WARNING] Unexpected type for feature {feature}: {type(value)}")
                    new_vector.append(0.0)
            except Exception as e:
                print(f"[ERROR] Failed to process feature {feature}: {e}")
                new_vector.append(0.0)

        # Ensure all values are numeric
        new_vector = [float(x) if x is not None else 0.0 for x in new_vector]
        
        # Normalize the vector
        normalized_vector = normalize_feature_vector(np.array(new_vector))
        normalized_vector = np.array(normalized_vector, dtype=np.float64)

        # After creating normalized_vector, update metadata
        vector_tuple = tuple(normalized_vector.tolist())
        if activity_label not in activity_feature_metadata:
            activity_feature_metadata[activity_label] = {}
        
        if vector_tuple not in activity_feature_metadata[activity_label]:
            activity_feature_metadata[activity_label][vector_tuple] = {
                "frequency": 1,
                "last_seen_event": global_event_counter
            }
        else:
            metadata = activity_feature_metadata[activity_label][vector_tuple]
            metadata["frequency"] = metadata.get("frequency", 0) + 1
            metadata["last_seen_event"] = global_event_counter

        return {
            "activity_label": activity_label,
            "new_vector": normalized_vector,  # ✅ Ensure consistent NumPy format
            "top_features": top_features
        }

    except Exception as e:
        print(f"[ERROR] Failed to process event: {str(e)}")
        traceback.print_exc()
        return None


