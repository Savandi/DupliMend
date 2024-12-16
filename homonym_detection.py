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


def log_merge_or_split(action, clusters_involved, details=None):
    """
    Logs merge or split actions for traceability.
    """
    log_traceability(
        action, "Cluster Analysis",
        {
            "clusters_involved": clusters_involved,
            "details": details or "N/A"
        }
    )
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
            similarity_matrix[i][j] = compute_contextual_weighted_similarity(
                cluster_vectors[i], cluster_vectors[j], [1.0] * len(cluster_vectors[0]), [1.0] * len(cluster_vectors[0])
            )
            similarity_matrix[j][i] = similarity_matrix[i][j]

    # Detect splits
    split_clusters = [
        f"Cluster_{i}" for i, cluster in enumerate(micro_clusters)
        if np.any(similarity_matrix[i] < splitting_threshold)
    ]
    if len(split_clusters) > 1:
        log_merge_or_split("split", split_clusters, {"similarity_matrix": similarity_matrix.tolist()})
        return "split", split_clusters

    # Detect merges
    merged_clusters = []
    for i, centroid in enumerate(cluster_vectors):
        for j, other_centroid in enumerate(cluster_vectors):
            if i != j and similarity_matrix[i][j] > merging_threshold:
                merged_clusters.append((i, j))

    if merged_clusters:
        log_merge_or_split("merge", merged_clusters, {"similarity_matrix": similarity_matrix.tolist()})
        return "merge", merged_clusters

    log_traceability("no_change", activity_label, "No significant change detected.")
    return "no_change", activity_label

def process_event(event_data):
    """
    Analyze feature vectors for splits and merges using DBStream.

    Parameters:
        event_data (dict): Data from the previous step, including the vector and DBStream model.

    Returns:
        tuple: Result of split/merge analysis.
    """
    activity_label = event_data["activity_label"]
    new_vector = event_data["new_vector"]
    dbstream_instance = event_data["dbstream_model"]

    # Log the incoming vector for traceability
    log_traceability("analyze_vector", activity_label, {"new_vector": new_vector})

    # Analyze DBStream clusters
    cluster_id = dbstream_instance.partial_fit(new_vector)
    log_traceability("update_clusters", activity_label, {"new_vector": new_vector, "cluster_id": cluster_id})

    # Detect splits and merges
    return analyze_splits_and_merges(activity_label, dbstream_instance)


def log_cluster_summary(dbstream_instance):
    """
    Log a periodic summary of cluster dynamics.
    """
    micro_clusters = dbstream_instance.get_micro_clusters()
    event_count = sum(cluster["weight"] for cluster in micro_clusters)
    active_clusters = len(micro_clusters)
    avg_weight = event_count / active_clusters if active_clusters > 0 else 0

    log_traceability("cluster_summary", "Periodic Update", {
        "total_clusters": active_clusters,
        "average_weight": avg_weight,
        "event_count": event_count
    })