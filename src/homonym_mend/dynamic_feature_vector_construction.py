from collections import defaultdict, deque
import traceback
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
    try:
        for activity_label in list(activity_feature_metadata.keys()):
            # Sort vectors based on frequency and last seen event, with safe access
            sorted_vectors = sorted(
                activity_feature_metadata[activity_label].items(),
                key=lambda x: (
                    x[1].get("frequency", 0),
                    x[1].get("last_seen_event", 0)
                )
            )

            for vector_tuple, metadata in sorted_vectors:
                # Safe access to last_seen_event with default to current counter
                last_seen = metadata.get("last_seen_event", global_event_counter)
                events_since_last_seen = global_event_counter - last_seen

                # Update frequency with decay
                current_frequency = metadata.get("frequency", 0)
                new_frequency = current_frequency * np.exp(-events_since_last_seen / decay_after_events)
                metadata["frequency"] = new_frequency
                
                # Update last_seen_event
                metadata["last_seen_event"] = global_event_counter

                # Remove if frequency drops below threshold
                if new_frequency < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
                    try:
                        del activity_feature_metadata[activity_label][vector_tuple]
                        if vector_tuple in activity_feature_history[activity_label]:
                            activity_feature_history[activity_label].remove(list(vector_tuple))
                    except (KeyError, ValueError) as e:
                        print(f"[WARNING] Error removing vector: {e}")

                # Stop if we're within budget
                if len(activity_feature_metadata[activity_label]) <= lossy_counting_budget:
                    break

    except Exception as e:
        print(f"[ERROR] Error in forget_old_feature_vectors: {e}")
        traceback.print_exc()


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
            "new_vector": normalized_vector.tolist()
        }

    except Exception as e:
        print(f"[ERROR] Failed to process event: {str(e)}")
        traceback.print_exc()
        return None


