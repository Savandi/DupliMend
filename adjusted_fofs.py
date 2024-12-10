from collections import defaultdict, deque
from sklearn.cluster import KMeans
from river.drift import ADWIN
import pandas as pd
import numpy as np
import time

# --- PARAMETERS ---
top_n_features = 3  # Number of top features to select
forgetting_factor = 0.9  # Forgetting factor for stability in control-flow
adaptive_window_min_size = 50  # Minimum adaptive sliding window size
adaptive_window_max_size = 200  # Maximum adaptive sliding window size
initial_window_size = 100  # Initial sliding window size for all features

# --- GLOBAL VARIABLES ---
directly_follows_matrix = defaultdict(lambda: defaultdict(int))  # Directly follows relationships
footprint_matrix = defaultdict(lambda: defaultdict(float))  # Footprint matrix with stability/forgetting
control_flow_drift = ADWIN()  # Drift detection for control-flow perspective

# Drift detectors for other perspectives
resource_drift = ADWIN()
time_drift = ADWIN()
data_drift = ADWIN()

# Adaptive sliding windows for feature importance
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))
feature_window_sizes = defaultdict(lambda: initial_window_size)  # Store adaptive window sizes for each feature

# Perspective scores storage
perspective_scores = defaultdict(lambda: {"control_flow": 0, "resource": 0, "time": 0, "data": 0})


# --- CONTROL-FLOW FUNCTIONS ---
def update_control_flow_matrices(previous_activity, current_activity):
    """
    Update directly follows matrix and footprint matrix with forgetting factor.
    """
    if previous_activity:
        directly_follows_matrix[previous_activity][current_activity] += 1
        footprint_matrix[previous_activity][current_activity] = (
            forgetting_factor * footprint_matrix[previous_activity][current_activity] + 1
        )

    for a1 in footprint_matrix:
        for a2 in footprint_matrix[a1]:
            footprint_matrix[a1][a2] *= forgetting_factor


def compute_control_flow_score(activity):
    """
    Compute the control-flow score for an activity based on the footprint matrix.
    """
    if activity not in footprint_matrix:
        return 0
    total = sum(footprint_matrix[activity].values())
    if total == 0:
        return 0
    probabilities = [v / total for v in footprint_matrix[activity].values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


# --- ADAPTIVE SLIDING WINDOW FUNCTIONS ---
def adjust_window_size(feature, drift_detected):
    """
    Dynamically adjust sliding window size based on drift detection.
    """
    current_size = feature_window_sizes[feature]
    if drift_detected:
        new_size = max(adaptive_window_min_size, current_size // 2)  # Halve the window size
    else:
        new_size = min(adaptive_window_max_size, current_size + 10)  # Gradually increase size
    feature_window_sizes[feature] = new_size
    print(f"Adjusting window size for feature '{feature}' to: {new_size}")
    return deque(maxlen=new_size)


# --- OTHER PERSPECTIVES ---
def compute_perspective_scores(event, previous_event):
    """
    Compute scores for all perspectives based on the current and previous events.
    """
    scores = {"control_flow": 0, "resource": 0, "time": 0, "data": 0}

    # Control-Flow Perspective
    if previous_event and 'activity' in previous_event and 'activity' in event:
        update_control_flow_matrices(previous_event['activity'], event['activity'])
        scores["control_flow"] = compute_control_flow_score(event['activity'])

    # Resource Perspective
    if 'resource' in event:
        scores["resource"] += 0.5

    # Time Perspective
    if 'timestamp' in event:
        scores["time"] += 0.3

    # Data/Attributes Perspective
    for feature, value in event.items():
        if feature not in ['timestamp', 'resource', 'activity']:
            scores["data"] += 0.2

    return scores


# --- DRIFT DETECTION ---
def detect_drift(feature, feature_scores):
    """
    Use ADWIN to detect drift in feature scores.
    """
    drift_detector = {"control_flow": control_flow_drift, "resource": resource_drift, "time": time_drift, "data": data_drift}
    avg_score = np.mean(feature_scores)
    drift_detector[feature].update(avg_score)
    return drift_detector[feature].detected_change()


# --- FEATURE SELECTION ---
def dynamic_clustering(features):
    """
    Perform dynamic clustering on features to reduce redundancy.
    """
    if len(features) < 2:
        return features  # No clustering needed for small sets
    kmeans = KMeans(n_clusters=min(len(features), 5)).fit(np.array(features).reshape(-1, 1))
    return [features[idx] for idx in kmeans.cluster_centers_.argsort(axis=0).flatten()]


def select_features(event, previous_event, top_n=top_n_features):
    """
    Compute perspective scores, detect drift, and select top features.
    """
    scores = compute_perspective_scores(event, previous_event)

    # Update sliding windows and detect drift
    for feature, score in scores.items():
        drift_detected = detect_drift(feature, list(feature_importance_windows[feature]))
        feature_importance_windows[feature] = adjust_window_size(feature, drift_detected)  # Adjust window size
        feature_importance_windows[feature].append(score)

    # Aggregate scores from adaptive sliding windows
    aggregated_scores = {
        feature: sum(window) / len(window) for feature, window in feature_importance_windows.items()
    }

    # Apply dynamic clustering
    clustered_features = dynamic_clustering(list(aggregated_scores.keys()))

    # Select top N features
    top_features = sorted(clustered_features, key=lambda x: aggregated_scores[x], reverse=True)[:top_n]
    return top_features


# --- STREAMING FUNCTION ---
def stream_event_log_with_feature_selection(df):
    """
    Process each event in the stream for feature selection.
    """
    previous_event = None
    for _, event in df.iterrows():
        top_features = select_features(event, previous_event)
        print(f"Event: {event}")
        print(f"Top Features: {top_features}")
        previous_event = event
        time.sleep(0.1)  # Simulate streaming delay


# --- MAIN SCRIPT ---
# Load and prepare event log
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
df_event_log['timestamp'] = pd.to_datetime(df_event_log['timestamp'])  # Convert timestamp to datetime
df_event_log = df_event_log.sort_values(by='timestamp')  # Sort by timestamp

# Process each event in the stream
stream_event_log_with_feature_selection(df_event_log)
