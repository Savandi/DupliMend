import logging
from src.utils.logging_utils import log_traceability
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from src.homonym_mend.dbstream import DBStream
from config.config import (
    splitting_threshold,
    merging_threshold,
    temporal_decay_rate,
    forgetting_threshold,
    positional_penalty_alpha,
    dbstream_params,
    grace_period_events,
    adaptive_threshold_variability,
)

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))
audit_log = []
dbstream_clusters = defaultdict(lambda: DBStream(dbstream_params))
event_counter = defaultdict(int)  # Track the number of processed events per activity label

adaptive_split_threshold = splitting_threshold
adaptive_merge_threshold = merging_threshold

cluster_last_updated = defaultdict(lambda: datetime.min)
cluster_grace_period = timedelta(seconds=5)

# Configure logging
logging.basicConfig(
    filename="../../traceability_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- FUNCTION DEFINITIONS ---
def adjust_thresholds(recent_variability):
    """
    Dynamically adjust splitting and merging thresholds based on recent variability.
    """
    global adaptive_split_threshold, adaptive_merge_threshold

    if recent_variability > adaptive_threshold_variability:
        adaptive_split_threshold = min(1.0, adaptive_split_threshold + 0.05)
        adaptive_merge_threshold = max(0.5, adaptive_merge_threshold - 0.05)
    else:
        adaptive_split_threshold = max(0.6, adaptive_split_threshold - 0.05)
        adaptive_merge_threshold = min(0.9, adaptive_merge_threshold + 0.05)

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

def log_merge_or_split(action, clusters_involved, details=None):
    """
    Logs merge or split actions for traceability.
    """
    log_traceability(
        action, "Cluster Analysis",
        {
            "clusters_involved": clusters_involved,
            "details": details or "N/A",
        },
    )

def aggregate_vectors(cluster_vectors):
    """
    Compute an aggregated vector (mean centroid) for a set of cluster vectors.
    """
    return np.mean(cluster_vectors, axis=0)

def analyze_splits_and_merges(activity_label, dbstream_instance):
    """
    Analyze DBStream clusters to detect splits or merges for a specific activity.
    """
    global cluster_last_updated, adaptive_split_threshold

    micro_clusters = dbstream_instance.get_micro_clusters()
    recent_variability = dbstream_instance.compute_cluster_variability()
    adjust_thresholds(recent_variability)

    if len(micro_clusters) <= 1:
        return "no_change", activity_label

    # Compute similarity between micro-clusters
    cluster_vectors = [cluster["centroid"] for cluster in micro_clusters]
    similarity_matrix = np.zeros((len(cluster_vectors), len(cluster_vectors)))

    for i in range(len(cluster_vectors)):
        for j in range(i, len(cluster_vectors)):
            similarity_matrix[i][j] = 1 - np.linalg.norm(cluster_vectors[i] - cluster_vectors[j])
            similarity_matrix[j][i] = similarity_matrix[i][j]

    # Detect splits
    split_clusters = []
    for i, cluster in enumerate(micro_clusters):
        if (datetime.now() - cluster_last_updated[i]) < cluster_grace_period:
            continue  # Skip recently updated clusters

        if np.any(similarity_matrix[i] < adaptive_split_threshold):
            split_clusters.append(f"Cluster_{i}")

    if split_clusters:
        return "split", split_clusters

    # Detect merges using aggregated vector
    aggregated_vector = np.mean(cluster_vectors, axis=0)
    merged_clusters = []
    for i, centroid in enumerate(cluster_vectors):
        similarity = 1 - np.linalg.norm(centroid - aggregated_vector)
        if similarity > adaptive_merge_threshold:
            merged_clusters.append(i)

    if merged_clusters:
        return "merge", merged_clusters

    return "no_change", activity_label
def process_event(event_data):
    """
    Process an incoming event and analyze splits or merges.
    """
    activity_label = event_data["activity_label"]
    new_vector = event_data["new_vector"]
    # Check if this is a new activity_label being processed
    if activity_label not in dbstream_clusters:
        log_traceability("new_activity_label", activity_label, "Initialized a new cluster group")

    dbstream_instance = dbstream_clusters[activity_label]
    log_traceability("incoming_vector", activity_label, {"vector": new_vector})

    # Update DBStream with the new vector
    cluster_id = dbstream_instance.partial_fit(new_vector)
    log_traceability("cluster_update", activity_label, {
        "new_vector": new_vector,
        "cluster_id": cluster_id,
        "micro_clusters": dbstream_instance.get_micro_clusters()
    })
    cluster_last_updated[cluster_id] = datetime.now()

    # Analyze for splits or merges
    result = analyze_splits_and_merges(activity_label, dbstream_instance)
    log_traceability("split_merge_result", activity_label, {"result": result})
    return result
def handle_temporal_decay(activity_label):
    """
    Apply temporal decay to clusters for a specific activity.
    """
    metadata = vector_metadata[activity_label]
    current_time = datetime.now()

    for vector, data in list(metadata.items()):
        if data["frequency"] < grace_period_events:
            continue  # Skip decay for vectors within the grace period

        time_diff = (current_time - data["recency"]).total_seconds()
        decayed_frequency = apply_temporal_decay(data["frequency"], time_diff)

        if decayed_frequency < forgetting_threshold:
            del metadata[vector]
        else:
            metadata[vector]["frequency"] = decayed_frequency

def log_cluster_summary(dbstream_instance):
    """
    Log a periodic summary of cluster dynamics.
    """
    if not isinstance(dbstream_instance, DBStream):
        log_traceability("error", "log_cluster_summary", "Provided instance is not a DBStream object")
        return

    micro_clusters = dbstream_instance.get_micro_clusters()
    event_count = sum(cluster.get("weight", 0) for cluster in micro_clusters)
    active_clusters = len(micro_clusters)
    avg_weight = event_count / active_clusters if active_clusters > 0 else 0

    log_traceability("cluster_summary", "Periodic Update", {
        "total_clusters": active_clusters,
        "average_weight": avg_weight,
        "event_count": event_count
    })
