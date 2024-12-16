from collections import defaultdict
from datetime import datetime
import numpy as np
import logging
from config import (
    splitting_threshold,
    merging_threshold,
    forgetting_threshold,
    positional_penalty_alpha,
    temporal_decay_rate,
    dbstream_params
)
from custom_label_encoder import CustomLabelEncoder
from dbstream import DBStream
from homonym_detection import compute_contextual_weighted_similarity

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))
event_counter = defaultdict(int)  # Track events per activity
audit_log = []
streaming_dbstream_models = defaultdict(lambda: DBStream(**dbstream_params))
encoders = defaultdict()  # Encoders for categorical features

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    filename="traceability_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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
    encoded_value = encoder.transform(value)
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
    event_counter[activity_label] += 1  # Increment event counter

    # Construct a dynamic feature vector for the event
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
        return "no_change", activity_label

    log_traceability("new_vector", activity_label, new_vector)

    # Pass new vector to DBStream model
    dbstream_model = streaming_dbstream_models[activity_label]
    dbstream_model.learn_one(new_vector)

    # Analyze the DBStream micro-clusters
    micro_clusters = dbstream_model.get_microclusters()
    log_traceability("dbstream_micro_clusters", activity_label, f"{micro_clusters}")

    # Analyze splits and merges
    return analyze_splits_and_merges(activity_label, micro_clusters)


def aggregate_vectors(cluster):
    """
    Compute an aggregated vector for a cluster of vectors.
    """
    return np.mean(cluster, axis=0)


def analyze_splits_and_merges(activity_label, clusters):
    """
    Analyze DBStream micro-clusters to detect splits or merges.
    """
    if len(clusters) <= 1:
        log_traceability("no_change", activity_label, "Insufficient clusters for analysis.")
        return "no_change", activity_label

    # Analyze splits
    micro_cluster_vectors = [cluster["centroid"] for cluster in clusters]
    similarity_matrix = np.zeros((len(micro_cluster_vectors), len(micro_cluster_vectors)))

    for i in range(len(micro_cluster_vectors)):
        for j in range(len(micro_cluster_vectors)):
            if i != j:
                similarity_matrix[i][j] = compute_contextual_weighted_similarity(
                    micro_cluster_vectors[i], micro_cluster_vectors[j], [1.0] * len(micro_cluster_vectors[0]), [1.0] * len(micro_cluster_vectors[0])
                )

    # Split clusters if similarity falls below threshold
    cluster_labels = [
        f"Cluster_{i}" for i, cluster in enumerate(clusters)
        if np.any(similarity_matrix[i] < splitting_threshold)
    ]
    if len(cluster_labels) > 1:
        log_traceability("split", activity_label, {"clusters": cluster_labels})
        return "split", cluster_labels

    # Merge clusters if similarity exceeds threshold
    aggregated_vector = aggregate_vectors(micro_cluster_vectors)
    merged_similarity = compute_contextual_weighted_similarity(
        aggregated_vector, aggregated_vector, [1.0] * len(aggregated_vector), [1.0] * len(aggregated_vector)
    )
    if merged_similarity > merging_threshold:
        log_traceability("merge", activity_label, {"merged_to": activity_label})
        return "merge", activity_label

    log_traceability("no_change", activity_label, "No significant change detected.")
    return "no_change", activity_label
