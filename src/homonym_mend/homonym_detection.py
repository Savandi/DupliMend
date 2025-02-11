from src.homonym_mend.feature_selection_with_drift_detection import update_feature_weights
from src.utils.logging_utils import log_traceability
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from config.config import (
    splitting_threshold,
    merging_threshold,
    grace_period_events,
    adaptive_threshold_min_variability, similarity_penalty, decay_after_events, previousEvents)
from src.utils.similarity_utils import compute_contextual_weighted_similarity
from src.utils.global_state import dbstream_clusters  # ✅ Import from global state

# --- GLOBAL VARIABLES ---
cluster_grace_period = timedelta(seconds=grace_period_events)
feature_vectors = defaultdict(list)
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "last_seen_event": 0}))
audit_log = []
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
def adaptive_threshold_variability(cluster_feature_vectors):
    """
    Compute variability factor based on feature vectors' dispersion.
    Higher variability leads to less aggressive splitting/merging thresholds.
    """
    if len(cluster_feature_vectors) <= 1:
        return adaptive_threshold_min_variability  # Use stricter lower bound

    # Compute pairwise distances between feature vectors
    distances = []
    for i in range(len(cluster_feature_vectors)):
        for j in range(i + 1, len(cluster_feature_vectors)):
            distances.append(
                np.linalg.norm(np.array(cluster_feature_vectors[i]) - np.array(cluster_feature_vectors[j])))

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


def compute_similarity(cluster_vectors, activity_label, feature_weights=None):
    """
    Compute pairwise similarity matrix between all cluster vectors.
    """
    if feature_weights is None:
        feature_weights = compute_feature_weights(cluster_vectors)  # Use computed feature weights

    n_clusters = len(cluster_vectors)
    similarity_matrix = np.zeros((n_clusters, n_clusters))

    # Dynamically extract previousEvents features
    previous_event_features = [f"prev_activity_{i}" for i in range(1, previousEvents + 1)]

    # Adjust feature weights dynamically for previousEvents
    for feature in previous_event_features:
        if feature not in feature_weights:
            feature_weights[feature] = 1.0  # Initialize with default weight

        # Use adaptive weight update for previousEvents features
        update_feature_weights(activity_label, feature, feature_weights.get(feature, 1.0))

    # Compute similarity with dynamically weighted previousEvents features
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
            similarity_matrix[j][i] = similarity  # Ensure matrix is symmetric

    return similarity_matrix


def analyze_splits_and_merges(activity_label, dbstream_instance, feature_vector):
    """
    Enhanced split/merge analysis with improved sensitivity to contextual differences.
    :param activity_label: The activity label being analyzed.
    :param dbstream_instance: The DBStream instance managing clusters for the activity.
    :param feature_vector: The feature vector to be clustered.
    :return: Tuple containing the split/merge result and a representative cluster_id.
    """
    # ✅ Ensure the feature vector is clustered before analysis
    cluster_id = dbstream_instance.partial_fit(feature_vector, activity_label)

    micro_clusters = dbstream_instance.get_micro_clusters()
    print(f"[DEBUG] Micro-Clusters Found for {activity_label}: {len(micro_clusters)}")

    if len(micro_clusters) <= 1:
        log_traceability("no_change", activity_label, "No clusters available.")
        return "no_change", cluster_id  # ✅ Return assigned cluster

    # ✅ Compute variability and adjust thresholds dynamically
    cluster_feature_vectors = [cluster["centroid"] for cluster in micro_clusters]
    variability = adaptive_threshold_variability(cluster_feature_vectors)

    global adaptive_split_threshold, adaptive_merge_threshold

    min_variability = adaptive_threshold_min_variability

    if variability > min_variability:
        adaptive_split_threshold = min(1.0, adaptive_split_threshold + 0.05)
        adaptive_merge_threshold = max(0.5, adaptive_merge_threshold - 0.05)
    else:
        adaptive_split_threshold = max(0.6, adaptive_split_threshold - 0.05)
        adaptive_merge_threshold = min(0.9, adaptive_merge_threshold + 0.05)

    log_traceability("variability_and_thresholds", activity_label, {
        "variability_factor": variability,
        "dynamic_splitting_threshold": adaptive_split_threshold,
        "dynamic_merging_threshold": adaptive_merge_threshold,
    })

    similarity_matrix = compute_similarity(cluster_feature_vectors, activity_label)
    print(f"[DEBUG] Similarity Matrix for {activity_label}:")
    print(similarity_matrix)

    # Analyze similarity distribution
    dissimilar_pairs = []
    for i in range(len(cluster_feature_vectors)):
        for j in range(i + 1, len(cluster_feature_vectors)):
            sim_score = similarity_matrix[i][j]
            if sim_score < adaptive_split_threshold:
                dissimilar_pairs.append((i, j))

    if dissimilar_pairs:
        unique_clusters = set([x for pair in dissimilar_pairs for x in pair])
        log_traceability("split_decision", activity_label, {
            "reason": "Found dissimilar clusters",
            "pairs": dissimilar_pairs,
            "unique_clusters": list(unique_clusters)
        })
        return "split", cluster_id  # ✅ Return updated cluster ID

    # Check for merges if no splits
    potential_merges = []
    for i in range(len(cluster_feature_vectors)):
        similar_neighbors = sum(1 for j in range(len(cluster_feature_vectors))
                                if i != j and similarity_matrix[i][j] > adaptive_merge_threshold)
        if similar_neighbors == 1:
            potential_merges.append(i)

    if len(potential_merges) == 2:
        if similarity_matrix[potential_merges[0]][potential_merges[1]] > adaptive_merge_threshold:
            log_traceability("merge_decision", activity_label, {
                "reason": "Found very similar pair",
                "clusters": potential_merges,
                "similarity": similarity_matrix[potential_merges[0]][potential_merges[1]]
            })
            return "merge", cluster_id  # ✅ Return merged cluster ID

    print(f"[DEBUG] No split/merge detected for {activity_label}")
    return "no_change", cluster_id  # ✅ Return default cluster ID


def compute_feature_weights(cluster_vectors):
    """
    Compute feature importance weights based on variance across clusters.
    :param cluster_vectors: List of feature vectors from clusters.
    :return: Normalized feature weights.
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


# def aggregate_vectors(cluster_vectors):
#     """
#     Compute an aggregated vector (mean centroid) for a set of cluster vectors.
#     """
#     return np.mean(cluster_vectors, axis=0)
#
#
# def find_similarity_clusters(similarity_matrix, high_threshold=0.8, low_threshold=0.2):
#     n = len(similarity_matrix)
#     clusters = []
#     used = set()
#
#     # First find highly similar groups
#     for i in range(n):
#         if i in used:
#             continue
#
#         cluster = {i}
#         for j in range(i + 1, n):
#             if j in used:
#                 continue
#             # If highly similar to i and not very dissimilar to existing cluster members
#             if (similarity_matrix[i][j] > high_threshold and
#                     all(similarity_matrix[k][j] > low_threshold for k in cluster)):
#                 cluster.add(j)
#
#         if len(cluster) > 1:  # Only consider groups of 2 or more
#             clusters.append(cluster)
#             used.update(cluster)
#
#     return clusters


# def log_merge_or_split(action, clusters_involved, details=None):
#     """
#     Logs merge or split actions for traceability.
#     """
#     log_traceability(
#         action, "Cluster Analysis",
#         {
#             "clusters_involved": clusters_involved,
#             "details": details or "N/A",
#         },
#     )

# def apply_temporal_decay(value, time_difference):
#     """
#     Apply temporal decay to feature frequency to prevent over-aggressive drops.
#     """
#     return value * np.exp(-temporal_decay_rate * min(time_difference, decay_after_events * 2))

# def normalize_vector(vector):
#     """
#     Normalize a numeric vector to have mean 0 and standard deviation 1.
#     :param vector: Numeric feature vector.
#     :return: Normalized vector.
#     """
#     if np.std(vector) == 0:  # Avoid division by zero
#         return vector
#     return (vector - np.mean(vector)) / np.std(vector)


# def process_event(event_data, global_event_counter):
#     """
#     Process an incoming event, assign a cluster, and analyze splits or merges.
#     """
#     activity_label = event_data["activity_label"]
#     new_vector = event_data["new_vector"]
#
#     # Normalize the feature vector before clustering
#     normalized_vector = normalize_vector(new_vector)
#     log_traceability("vector_normalization", activity_label, {
#         "raw_vector": new_vector,
#         "normalized_vector": normalized_vector
#     })
#
#     # Ensure a DBStream instance exists for this activity label
#     if activity_label not in dbstream_clusters:
#         dbstream_clusters[activity_label] = DBStream()
#         log_traceability("new_activity_label", activity_label, "Initialized a new cluster group")
#
#     dbstream_instance = dbstream_clusters[activity_label]
#
#     # Process feature vector through DBStream without passing `global_event_counter`
#     cluster_id = dbstream_instance.partial_fit(normalized_vector, activity_label)
#
#     # Store and track cluster assignments in activity_feature_metadata
#     vector_tuple = tuple(normalized_vector)
#     if vector_tuple in activity_feature_metadata[activity_label]:
#         activity_feature_metadata[activity_label][vector_tuple]["frequency"] += 1
#         activity_feature_metadata[activity_label][vector_tuple]["last_seen_event"] = \
#         dbstream_instance.activity_event_counters[activity_label]  # Update based on per-activity event count
#         activity_feature_metadata[activity_label]["cluster"] = cluster_id
#     else:
#         vector_metadata[activity_label][str(vector_tuple)] = {  # ✅ Ensure valid dictionary key
#             "frequency": 1,
#             "last_seen_event": dbstream_instance.activity_event_counters[activity_label]  # ✅ Keep as int
#         }
#
#     log_traceability("cluster_update", activity_label, {
#         "new_vector": normalized_vector,
#         "cluster_id": cluster_id,
#         "micro_clusters": dbstream_instance.get_micro_clusters()
#     })
#
#     # Analyze splits or merges
#     result, updated_cluster_id = analyze_splits_and_merges(activity_label, dbstream_instance)
#
#     log_traceability("split_merge_result", activity_label, {"result": result, "cluster_id": updated_cluster_id})
#
#     # Apply temporal decay after processing the event
#     handle_temporal_decay(activity_label, dbstream_instance.activity_event_counters[activity_label])
#
#     # Remove old clusters based on per-activity event count
#     dbstream_instance.forget_old_clusters(activity_label)
#
#     return result, updated_cluster_id


# def handle_temporal_decay(activity_label, current_activity_event_count):
#     """
#     Apply temporal decay to clusters for a specific activity.
#     """
#     metadata = vector_metadata.get(activity_label, {})
#
#     for vector in list(metadata.keys()):
#         time_diff = current_activity_event_count - metadata[vector]["last_seen_event"]
#         decayed_frequency = apply_temporal_decay(metadata[vector]["frequency"], time_diff)
#
#         if decayed_frequency < forgetting_threshold:
#             del metadata[vector]
#         else:
#             metadata[vector]["frequency"] = decayed_frequency

#
# def log_cluster_summary(dbstream_instance):
#     """
#     Log a periodic summary of cluster dynamics.
#     """
#     if not isinstance(dbstream_instance, DBStream):
#         log_traceability("error", "log_cluster_summary", "Provided instance is not a DBStream object")
#         return
#
#     micro_clusters = dbstream_instance.get_micro_clusters()
#     event_count = sum(cluster.get("weight", 0) for cluster in micro_clusters)
#     active_clusters = len(micro_clusters)
#     avg_weight = event_count / active_clusters if active_clusters > 0 else 0
#
#     log_traceability("cluster_summary", "Periodic Update", {
#         "total_clusters": active_clusters,
#         "average_weight": avg_weight,
#         "event_count": event_count
#     })
