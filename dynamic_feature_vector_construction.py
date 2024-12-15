from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging
from datetime import datetime
from config import (
    splitting_threshold,
    merging_threshold,
    temporal_decay_rate,
    forgetting_threshold
)

# --- GLOBAL VARIABLES ---
feature_vectors = defaultdict(list)  # Store distinct feature vectors per activity label
aggregated_vectors = defaultdict(list)  # Aggregated vectors per activity label for merging analysis
vector_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": datetime.now()}))  # Track recency and frequency of vectors
audit_log = []  # List to track splits/merges for auditability
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # One-hot encoder for categorical features

# Configure logging
logging.basicConfig(
    filename="traceability_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- FUNCTION DEFINITIONS ---

def apply_temporal_decay(value, time_difference, decay_rate=temporal_decay_rate):
    """
    Apply temporal decay to a value based on time difference.
    """
    return value * np.exp(-decay_rate * time_difference)

def compute_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors.
    """
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]

def compute_similarity_matrix(vectors):
    """
    Compute a similarity matrix for a list of vectors.
    """
    return cosine_similarity(np.array(vectors))

def aggregate_vector(cluster):
    """
    Compute an aggregated vector (e.g., weighted average) for a cluster of vectors.
    """
    return np.mean(cluster, axis=0)

def log_traceability(action, activity_label, details):
    """
    Log actions for traceability and auditability.
    """
    timestamp = datetime.now().isoformat()
    entry = {"timestamp": timestamp, "action": action, "activity_label": activity_label, "details": details}
    audit_log.append(entry)
    logging.info(f"{action.upper()} - {activity_label}: {details}")

def encode_categorical_features(event, top_features):
    """
    Encode categorical features using one-hot encoding.

    Parameters:
        event (dict): The event being processed.
        top_features (list): The selected top features.

    Returns:
        list: A list of encoded feature values.
    """
    categorical_features = [
        value for feature, value in event.items()
        if feature in top_features and isinstance(value, str)
    ]
    if categorical_features:
        categorical_array = np.array(categorical_features).reshape(1, -1)
        encoded_array = encoder.fit_transform(categorical_array)
        return encoded_array.flatten().tolist()
    return []

def analyze_feature_vector_group(activity_label):
    """
    Analyze feature vector group to detect splits or merges.
    """
    if len(feature_vectors[activity_label]) <= 1:
        log_traceability("no_change", activity_label, "Insufficient vectors for analysis.")
        print(f"No change detected for activity '{activity_label}' (Not enough vectors).")
        return "no_change", activity_label

    print(f"Analyzing feature vectors for activity label '{activity_label}': {feature_vectors[activity_label]}")

    # Step 1: Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(feature_vectors[activity_label])
    log_traceability("similarity_matrix", activity_label, similarity_matrix.tolist())
    print(f"Similarity Matrix for '{activity_label}':\n{similarity_matrix}")

    # Step 2: Clustering for splitting
    clustering = DBSCAN(eps=1 - splitting_threshold, min_samples=2, metric="precomputed").fit(1 - similarity_matrix)
    labels = clustering.labels_
    unique_clusters = set(labels) - {-1}  # Exclude noise

    if len(unique_clusters) > 1:
        # Splitting detected
        split_details = {f"{activity_label}_cluster_{i}": [feature_vectors[activity_label][j] for j in range(len(labels)) if labels[j] == i] for i in unique_clusters}
        log_traceability("split", activity_label, split_details)
        print(f"Split detected for '{activity_label}': {split_details}")
        return "split", split_details

    # Step 3: Check for merging
    aggregated_vectors[activity_label] = aggregate_vector(feature_vectors[activity_label])
    print(f"Aggregated Vector for '{activity_label}': {aggregated_vectors[activity_label]}")
    aggregated_similarity = compute_similarity_matrix([aggregated_vectors[activity_label]])
    if aggregated_similarity[0][0] > merging_threshold:
        log_traceability("merge", activity_label, {"merged_to": activity_label})
        print(f"Merging detected for '{activity_label}'.")
        return "merge", activity_label

    log_traceability("no_change", activity_label, "No significant change detected.")
    print(f"No change for activity label '{activity_label}' after analysis.")
    return "no_change", activity_label

def process_event(event, top_features, timestamp_column):
    """
    Process an event to construct and analyze dynamic feature vectors.
    """
    activity_label = event["Activity"]
    timestamp = event[timestamp_column]

    print(f"Processing event for activity label '{activity_label}': {event}")

    # Construct a dynamic feature vector for the event
    new_vector = []

    # Handle numeric features
    for feature in top_features:
        value = event.get(feature)
        if value is not None and isinstance(value, (int, float)):
            recency_weight = apply_temporal_decay(1.0, (datetime.now() - timestamp).total_seconds())
            new_vector.append(value * recency_weight)

    # Handle categorical features
    encoded_features = encode_categorical_features(event, top_features)
    new_vector.extend(encoded_features)

    print(f"Constructed new vector for '{activity_label}': {new_vector}")
    log_traceability("new_vector", activity_label, new_vector)

    # Match or create vector in the group
    match_found = False
    for i, existing_vector in enumerate(feature_vectors[activity_label]):
        similarity = compute_similarity(existing_vector, new_vector)
        if similarity >= merging_threshold:
            vector_metadata[activity_label][i]["frequency"] += 1
            vector_metadata[activity_label][i]["recency"] = timestamp
            match_found = True
            print(f"Matched with existing vector for '{activity_label}': {existing_vector}")
            log_traceability("match", activity_label, f"Matched existing vector: {existing_vector}")
            break

    if not match_found:
        feature_vectors[activity_label].append(new_vector)
        vector_metadata[activity_label][len(feature_vectors[activity_label]) - 1] = {"frequency": 1, "recency": timestamp}
        print(f"New vector added for activity '{activity_label}': {new_vector}")
        log_traceability("add_vector", activity_label, new_vector)

    # Decay inactive vectors and remove fully decayed ones
    for i in list(range(len(feature_vectors[activity_label]))):  # Convert to list to allow removal during iteration
        vector_info = vector_metadata[activity_label][i]
        time_since_last_update = (datetime.now() - vector_info["recency"]).total_seconds()
        vector_info["frequency"] *= apply_temporal_decay(1.0, time_since_last_update)
        if vector_info["frequency"] < forgetting_threshold:
            print(f"Removing fully decayed vector for '{activity_label}': {feature_vectors[activity_label][i]}")
            log_traceability("remove_vector", activity_label, f"Fully decayed vector: {feature_vectors[activity_label][i]}")
            feature_vectors[activity_label].pop(i)
            vector_metadata[activity_label].pop(i)

    # Analyze vector group for splitting/merging
    return analyze_feature_vector_group(activity_label)
