from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from datetime import datetime
from dbstream import DBStream
from config import (
    splitting_threshold,
    merging_threshold,
    temporal_decay_rate,
    forgetting_threshold,
    positional_penalty_alpha,
    dbstream_params,
)

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)  # Feature vectors per activity label
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))
audit_log = []  # Traceability log
dbstream_clusters = defaultdict(lambda: DBStream(**dbstream_params))  # One DBStream instance per activity label

# Configure logging
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


def compute_contextual_weighted_similarity(v1, v2, w1, w2, alpha=positional_penalty_alpha):
    """
    Compute contextual weighted similarity for two feature vectors.
    """
    n, m = len(v1), len(v2)
    length_penalty = min(n, m) / max(n, m)
    normalization_factor = max(sum(w1), sum(w2))

    weighted_sum = 0
    for i in range(n):
        for j in range(m):
            sim = 1 - abs(v1[i] - v2[j])  # Numerical similarity for encoded features
            positional_penalty = 1 if i == j else alpha
            weight = (w1[i] + w2[j]) / 2
            weighted_sum += sim * positional_penalty * weight

    similarity = (weighted_sum / normalization_factor) * length_penalty
    return similarity


def log_traceability(action, activity_label, details):
    """
    Log traceability and auditability details.
    """
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "action": action, "activity_label": activity_label, "details": details}
    audit_log.append(entry)
    logging.info(f"{action.upper()} - {activity_label}: {details}")


def analyze_splits_and_merges(activity_label, dbstream_instance):
    """
    Analyze DBStream clusters to detect splits or merges.
    """
    micro_clusters = dbstream_instance.get_micro_clusters()
    if len(micro_clusters) <= 1:
        log_traceability("no_change", activity_label, "Insufficient clusters for analysis.")
        return "no_change", activity_label

    # Compute similarity between micro-clusters
    cluster_vectors = [cluster["centroid"] for cluster in micro_clusters]
    similarity_matrix = np.zeros((len(cluster_vectors), len(cluster_vectors)))

    for i in range(len(cluster_vectors)):
        for j in range(i, len(cluster_vectors)):
            similarity = compute_contextual_weighted_similarity(
                cluster_vectors[i], cluster_vectors[j], [1.0] * len(cluster_vectors[0]), [1.0] * len(cluster_vectors[0])
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    # Detect splits
    cluster_labels = [
        f"Cluster_{i}" for i, cluster in enumerate(micro_clusters)
        if np.any(similarity_matrix[i] < splitting_threshold)
    ]
    if len(cluster_labels) > 1:
        log_traceability("split", activity_label, {"clusters": cluster_labels})
        return "split", cluster_labels

    # Detect merges
    aggregated_vector = np.mean(cluster_vectors, axis=0)
    merged_similarity = compute_contextual_weighted_similarity(
        aggregated_vector, aggregated_vector, [1.0] * len(aggregated_vector), [1.0] * len(aggregated_vector)
    )
    if merged_similarity > merging_threshold:
        log_traceability("merge", activity_label, {"merged_to": activity_label})
        return "merge", activity_label

    log_traceability("no_change", activity_label, "No significant change detected.")
    return "no_change", activity_label


def process_event(event, top_features):
    """
    Process an event to construct and analyze feature vectors.
    """
    activity_label = event["Activity"]

    # Construct feature vector
    new_vector = []
    for feature in top_features:
        value = event.get(feature, 0)  # Assume 0 for missing features
        if isinstance(value, (int, float)):  # Encoded features must be numeric
            new_vector.append(value)
        else:
            log_traceability("skip_feature", activity_label, f"Skipping non-numeric feature '{feature}' with value '{value}'")

    log_traceability("new_vector", activity_label, new_vector)

    # Update DBStream clusters
    dbstream_instance = dbstream_clusters[activity_label]
    cluster_id = dbstream_instance.partial_fit(new_vector)

    log_traceability("update_clusters", activity_label, {"new_vector": new_vector, "cluster_id": cluster_id})

    # Analyze clusters for splits and merges
    return analyze_splits_and_merges(activity_label, dbstream_instance)
