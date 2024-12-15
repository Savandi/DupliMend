from collections import defaultdict
from river.drift import ADWIN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
import logging
from datetime import datetime, timedelta
from config import splitting_threshold, merging_threshold, temporal_decay_rate, forgetting_threshold, grace_period_duration

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)  # Store distinct feature vectors per activity label
aggregated_vectors = defaultdict(list)  # Aggregated vectors per activity label for merging analysis
vector_metadata = defaultdict(lambda: defaultdict(dict))  # Track recency, frequency, and creation time
audit_log = []  # List to track splits/merges for auditability

# --- FUNCTION DEFINITIONS ---

def apply_temporal_decay(value, time_difference):
    return value * np.exp(-temporal_decay_rate * time_difference)

def compute_similarity(vector1, vector2):
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

def compute_similarity_matrix(vectors):
    return cosine_similarity(np.array(vectors))

def aggregate_vector(cluster):
    return np.mean(cluster, axis=0)

def log_split_merge_change(change_type, activity_label, details):
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "change_type": change_type, "activity_label": activity_label, "details": details}
    audit_log.append(entry)
    logging.info(f"{change_type.upper()} - {activity_label}: {details}")

def analyze_feature_vector_group(activity_label):
    if len(feature_vectors[activity_label]) <= 1:
        print(f"No change detected for activity: {activity_label}")
        return "no_change", activity_label

    print(f"Analyzing feature vectors for activity label '{activity_label}': {feature_vectors[activity_label]}")
    similarity_matrix = compute_similarity_matrix(feature_vectors[activity_label])
    print(f"Similarity Matrix for '{activity_label}':\n{similarity_matrix}")

    clustering = DBSCAN(eps=1 - splitting_threshold, min_samples=2, metric="precomputed").fit(1 - similarity_matrix)
    labels = clustering.labels_
    unique_clusters = set(labels) - {-1}

    if len(unique_clusters) > 1:
        split_details = {f"{activity_label}_{i}": [feature_vectors[activity_label][j] for j in range(len(labels)) if labels[j] == i] for i in unique_clusters}
        log_split_merge_change("split", activity_label, split_details)
        print(f"Split detected for '{activity_label}': {split_details}")
        return "split", split_details

    aggregated_vectors[activity_label] = aggregate_vector(feature_vectors[activity_label])
    print(f"Aggregated Vector for '{activity_label}': {aggregated_vectors[activity_label]}")
    aggregated_similarity = compute_similarity_matrix([aggregated_vectors[activity_label]])
    if aggregated_similarity[0][0] > merging_threshold:
        log_split_merge_change("merge", activity_label, {"merged_to": activity_label})
        print(f"Merging detected for '{activity_label}'")
        return "merge", activity_label

    print(f"No change for activity label '{activity_label}' after analysis.")
    return "no_change", activity_label

def process_event(event, top_features, timestamp_column):
    activity_label = event["Activity"]
    timestamp = event[timestamp_column]
    new_vector = [event.get(feature, 0) for feature in top_features]

    match_found = False
    for i, existing_vector in enumerate(feature_vectors[activity_label]):
        similarity = compute_similarity(existing_vector, new_vector)
        if similarity >= merging_threshold:
            vector_metadata[activity_label][i]["frequency"] += 1
            vector_metadata[activity_label][i]["recency"] = timestamp
            match_found = True
            break

    if not match_found:
        feature_vectors[activity_label].append(new_vector)
        vector_metadata[activity_label][len(feature_vectors[activity_label]) - 1] = {
            "frequency": 1,
            "recency": timestamp,
            "created": timestamp,
        }

    for i in range(len(feature_vectors[activity_label]) - 1, -1, -1):
        vector_info = vector_metadata[activity_label][i]
        time_since_last_update = (datetime.now() - vector_info["recency"]).total_seconds()
        if datetime.now() - vector_info["created"] < timedelta(seconds=grace_period_duration):
            continue
        vector_info["frequency"] *= apply_temporal_decay(1.0, time_since_last_update)
        if vector_info["frequency"] < forgetting_threshold:
            feature_vectors[activity_label].pop(i)
            vector_metadata[activity_label].pop(i)

    return analyze_feature_vector_group(activity_label)
