# --- Perspective Column Mapping ---
import numpy as np

control_flow_column = 'Activity'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
case_id_column = 'CaseID'
event_id_column = 'EventID'

# --- Data Columns ---
data_columns = []

# --- Discretization and Binning Parameters ---
features_to_discretize = ['NumericFeature_1', 'NumericFeature_2', 'NumericFeature_3']
quantiles = [0.25, 0.5, 0.75]
sliding_window_size = 150
bin_density_threshold = 10
cluster_grace_period_seconds = 25
drift_threshold = 0.05  # For ADWIN drift detection in binning
grace_period_events = 10  # Number of events to delay decay for new vectors

# --- Feature Selection Parameters ---
top_n_features = 3
forgetting_factor = 0.9
adaptive_window_min_size = 50
adaptive_window_max_size = 200
initial_window_size = 100
temporal_decay_rate = 0.001
forgetting_threshold = 0.001
positional_penalty_alpha = 0.8

# --- Clustering and Drift Detection Parameters ---
dbstream_params = {
    "clustering_threshold": 0.95,
    "fading_factor": 0.02,
    "cleanup_interval": 2,
    "intersection_factor": 0.3
}

# --- Splitting and Merging Parameters ---

splitting_threshold = 0.70
merging_threshold = 0.60
adaptive_threshold_min_variability = 0.8

# --- Logging Parameters ---
log_frequency = 10

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