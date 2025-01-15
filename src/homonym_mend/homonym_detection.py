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
    adaptive_threshold_min_variability, similarity_penalty, min_cluster_size)
from src.utils.similarity_utils import compute_contextual_weighted_similarity

# --- GLOBAL VARIABLES ---
cluster_grace_period = timedelta(seconds=grace_period_events)
feature_vectors = defaultdict(list)
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))
audit_log = []
dbstream_clusters = defaultdict(lambda: DBStream(dbstream_params))
event_counter = defaultdict(int)  # Track the number of processed events per activity label
adaptive_split_threshold = splitting_threshold
adaptive_merge_threshold = merging_threshold
cluster_last_updated = defaultdict(lambda: datetime.min)

# # Configure logging
# logging.basicConfig(
#     filename="../../traceability_log.txt",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# --- FUNCTION DEFINITIONS ---
def adaptive_threshold_variability(feature_vectors):
    """
    Compute variability factor based on feature vectors' dispersion.
    Higher variability leads to less aggressive splitting/merging thresholds.
    """
    if len(feature_vectors) <= 1:
        return adaptive_threshold_min_variability  # Use stricter lower bound

    # Compute pairwise distances between feature vectors
    distances = []
    for i in range(len(feature_vectors)):
        for j in range(i + 1, len(feature_vectors)):
            distances.append(np.linalg.norm(np.array(feature_vectors[i]) - np.array(feature_vectors[j])))

    if not distances:
        return adaptive_threshold_min_variability  # Use stricter lower bound

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Normalize the variability (mean +/- std range)
    variability_factor = min(max((mean_distance + std_distance) / 10.0, adaptive_threshold_min_variability), 1.5)
    return variability_factor

def adjust_thresholds(recent_variability):
    """
    Dynamically adjust splitting and merging thresholds based on recent variability.
    """
    global adaptive_split_threshold, adaptive_merge_threshold

    min_variability = adaptive_threshold_min_variability  # Use the configured lower bound

    if recent_variability > min_variability:
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


def find_similarity_clusters(similarity_matrix, high_threshold=0.8, low_threshold=0.2):
    n = len(similarity_matrix)
    clusters = []
    used = set()

    # First find highly similar groups
    for i in range(n):
        if i in used:
            continue

        cluster = {i}
        for j in range(i + 1, n):
            if j in used:
                continue
            # If highly similar to i and not very dissimilar to existing cluster members
            if (similarity_matrix[i][j] > high_threshold and
                    all(similarity_matrix[k][j] > low_threshold for k in cluster)):
                cluster.add(j)

        if len(cluster) > 1:  # Only consider groups of 2 or more
            clusters.append(cluster)
            used.update(cluster)

    return clusters


def compute_similarity_matrix(cluster_vectors):
    """
    Compute pairwise similarity matrix between all cluster vectors.

    Args:
        cluster_vectors (list): List of feature vectors from clusters

    Returns:
        numpy.ndarray: Matrix of pairwise similarities
    """
    n_clusters = len(cluster_vectors)
    similarity_matrix = np.zeros((n_clusters, n_clusters))
    feature_weights = compute_feature_weights(cluster_vectors)

    for i in range(n_clusters):
        for j in range(i, n_clusters):
            similarity = compute_contextual_weighted_similarity(
                cluster_vectors[i],
                cluster_vectors[j],
                feature_weights,
                feature_weights,
                alpha=similarity_penalty
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Matrix is symmetric

    return similarity_matrix
def compute_feature_weights(cluster_vectors):
    """
    Compute feature importance weights based on variance across clusters.
    """
    if not cluster_vectors or len(cluster_vectors[0]) == 0:
        return np.ones(1)

    vectors_array = np.array(cluster_vectors)
    variances = np.var(vectors_array, axis=0)

    # Normalize variances to get weights
    weights = variances / np.sum(variances) if np.sum(variances) > 0 else np.ones_like(variances)

    # Ensure weights don't get too small
    weights = np.maximum(weights, 0.1)
    weights = weights / np.sum(weights)

    return weights




def analyze_splits_and_merges(activity_label, dbstream_instance):
    """
    Enhanced split/merge analysis with improved sensitivity to contextual differences.
    """
    micro_clusters = dbstream_instance.get_micro_clusters()

    if len(micro_clusters) <= 1:
        log_traceability("no_change", activity_label, "No clusters available.")
        return "no_change", activity_label

    # Compute variability with increased sensitivity
    feature_vectors = [cluster["centroid"] for cluster in micro_clusters]
    variability = adaptive_threshold_variability(feature_vectors)

    # More aggressive splitting, more conservative merging
    dynamic_splitting_threshold = min(splitting_threshold * variability, 0.4)  # Lower threshold for easier splits
    dynamic_merging_threshold = min(max(merging_threshold * variability, 0.9), 0.98)  # Higher threshold for harder merges

    log_traceability("variability_and_thresholds", activity_label, {
        "variability_factor": variability,
        "dynamic_splitting_threshold": dynamic_splitting_threshold,
        "dynamic_merging_threshold": dynamic_merging_threshold,
    })

    # Get cluster vectors
    cluster_vectors = [cluster["centroid"] for cluster in micro_clusters]
    similarity_matrix = np.zeros((len(cluster_vectors), len(cluster_vectors)))
    feature_weights = compute_feature_weights(cluster_vectors)

    # Compute pairwise similarities and analyze distribution
    similarities = []
    for i in range(len(cluster_vectors)):
        for j in range(i, len(cluster_vectors)):
            similarity = compute_contextual_weighted_similarity(
                cluster_vectors[i],
                cluster_vectors[j],
                feature_weights,
                feature_weights,
                alpha=similarity_penalty
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
            if i != j:
                similarities.append(similarity)

    # Analyze similarity distribution
    if similarities:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        log_traceability("similarity_stats", activity_label, {
            "mean": mean_sim,
            "std": std_sim
        })

    logging.info(f"Similarity matrix for {activity_label}: {similarity_matrix}")

    # Check for splits using both threshold and distribution
    dissimilar_pairs = []
    for i in range(len(cluster_vectors)):
        for j in range(i+1, len(cluster_vectors)):
            # Split if either below threshold or significantly below mean
            if (similarity_matrix[i][j] < dynamic_splitting_threshold or
                (similarities and similarity_matrix[i][j] < (mean_sim - std_sim))):
                dissimilar_pairs.append((i, j))

    if dissimilar_pairs:
        unique_clusters = set([x for pair in dissimilar_pairs for x in pair])
        if len(unique_clusters) >= 2:
            log_traceability("split_decision", activity_label, {
                "reason": "Found dissimilar clusters",
                "pairs": dissimilar_pairs,
                "unique_clusters": list(unique_clusters)
            })
            return "split", list(unique_clusters)

    # Only check for merges if no splits and similarities are very high
    potential_merges = []
    for i in range(len(cluster_vectors)):
        # Count how many very similar neighbors each cluster has
        similar_neighbors = sum(1 for j in range(len(cluster_vectors))
                              if i != j and similarity_matrix[i][j] > dynamic_merging_threshold)
        # Only consider for merging if it has exactly one very similar neighbor
        if similar_neighbors == 1:
            potential_merges.append(i)

    # Only merge pairs of very similar clusters
    if len(potential_merges) == 2:
        if similarity_matrix[potential_merges[0]][potential_merges[1]] > dynamic_merging_threshold:
            log_traceability("merge_decision", activity_label, {
                "reason": "Found very similar pair",
                "clusters": potential_merges,
                "similarity": similarity_matrix[potential_merges[0]][potential_merges[1]]
            })
            return "merge", potential_merges

    return "no_change", activity_label

    # Get stable clusters but don't return if some are unstable
    now = datetime.now()
    stable_clusters = [
        cluster for cluster_id, cluster in enumerate(micro_clusters)
        if (now - cluster_last_updated[cluster_id]).total_seconds() > (
                    cluster_grace_period.total_seconds() / 8)  # Further reduced stability requirement
           and cluster.get("age", 0) > min_cluster_size/2  # Halved age requirement
    ]

    # Proceed with analysis even with just one stable cluster
    cluster_vectors = [cluster["centroid"] for cluster in stable_clusters]
    if not cluster_vectors:
        cluster_vectors = [cluster["centroid"] for cluster in micro_clusters]  # Fall back to all clusters

    similarity_matrix = np.zeros((len(cluster_vectors), len(cluster_vectors)))
    feature_weights = compute_feature_weights(cluster_vectors)

    # More aggressive split detection
    split_clusters = []
    for i, cluster in enumerate(cluster_vectors):
        # Check for any dissimilar pairs
        dissimilar_count = np.sum(similarity_matrix[i] < dynamic_splitting_threshold)
        if dissimilar_count > 0:  # Changed from >= 1 to > 0 for more sensitivity
            split_clusters.append(f"Cluster_{i}")

    # Changed split condition to be more lenient
    if len(split_clusters) > 0:  # Changed from > 1 to > 0
        log_merge_or_split("split", split_clusters, {
            "similarity_matrix": similarity_matrix.tolist(),
            "threshold": dynamic_splitting_threshold
        })
        return "split", split_clusters

    # Enhanced merge detection with lower threshold
    aggregated_vector = aggregate_vectors(cluster_vectors)
    merged_clusters = []

    for i, centroid in enumerate(cluster_vectors):
        similarity = compute_contextual_weighted_similarity(
            centroid,
            aggregated_vector,
            feature_weights,
            feature_weights,
            alpha=0.5  # Reduced from 0.7 for more aggressive merging
        )
        if similarity > dynamic_merging_threshold:
            merged_clusters.append(i)

    if len(potential_merges) > 0 and len(potential_merges) <= 2:
        return "merge", potential_merges

    log_traceability("no_change", activity_label, "No significant change detected.")
    return "no_change", activity_label




def normalize_vector(vector):
    """
    Normalize a numeric vector to have mean 0 and standard deviation 1.
    """
    if np.std(vector) == 0:  # Avoid division by zero
        return vector
    return (vector - np.mean(vector)) / np.std(vector)

def process_event(event_data):
    """
    Process an incoming event and analyze splits or merges.
    """
    activity_label = event_data["activity_label"]
    new_vector = event_data["new_vector"]

    # Normalize the vector
    normalized_vector = normalize_vector(new_vector)

    log_traceability("vector_normalization", activity_label, {
        "raw_vector": new_vector,
        "normalized_vector": normalized_vector
    })
    # Check if this is a new activity_label being processed
    if activity_label not in dbstream_clusters:
        log_traceability("new_activity_label", activity_label, "Initialized a new cluster group")

    dbstream_instance = dbstream_clusters[activity_label]
    log_traceability("incoming_vector", activity_label, {"vector": normalized_vector})

    # Update DBStream with the normalized vector
    cluster_id = dbstream_instance.partial_fit(normalized_vector)
    log_traceability("cluster_update", activity_label, {
        "new_vector": normalized_vector,
        "cluster_id": cluster_id,
        "micro_clusters": dbstream_instance.get_micro_clusters()
    })
    cluster_last_updated[cluster_id] = datetime.now()

    # Analyze for splits or merges
    feature_vectors = [cluster["centroid"] for cluster in dbstream_instance.get_micro_clusters()]
    variability = adaptive_threshold_variability(feature_vectors)
    adjust_thresholds(variability)  # Adjust thresholds dynamically

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
        if data["frequency"] < grace_period_events:  # Reference global config
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
