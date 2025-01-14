# --- Perspective Column Mapping ---
import numpy as np

control_flow_column = 'Activity'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
case_id_column = 'CaseID'
event_id_column = 'EventID'

# --- Data Columns ---
data_columns = []

# --- Enhanced Discretization and Binning Parameters ---
features_to_discretize = ['NumericFeature_1', 'NumericFeature_2', 'NumericFeature_3']
quantiles = [0.25, 0.5, 0.75]  # Base quantile points
sliding_window_size = 150
bin_density_threshold = 10

# New binning parameters
density_estimation_points = 100
min_bin_width = 0.005       # Finer-grained binning
decay_factor = 0.9          # Slightly faster decay
initial_bins = 15           # More initial bins
drift_threshold = 0.03      # More sensitive to drift
merge_threshold = 0.5       # Less aggressive bin merging
# Enhanced temporal binning
temporal_granularity = {
    'hour_bins': ['Early_Morning', 'Morning', 'Afternoon', 'Late_Afternoon', 'Night'],
    'day_periods': ['Weekend', 'Weekday'],
    'seasons': ['Winter', 'Spring', 'Summer', 'Fall']
}

# --- Feature Selection Parameters ---
max_top_n_features = 7       # Include more features for better discrimination
forgetting_factor = 0.8      # Forget old patterns faster
temporal_decay_rate = 0.005  # Faster temporal decay
forgetting_threshold = 0.0005  # Lower threshold to maintain more information
positional_penalty_alpha = 0.3  # Stronger positional penalty
adaptive_window_min_size = 50
adaptive_window_max_size = 200
initial_window_size = 100
lossy_counting_threshold = 0.01
lossy_counting_error_rate = 0.01

# --- Clustering and Drift Detection Parameters ---
dbstream_params = {
    "clustering_threshold": 0.45,  # Lower threshold to detect more subtle differences
    "fading_factor": 0.2,    # Increase fading to adapt faster to changes
    "cleanup_interval": 1,
    "intersection_factor": 0.15  # Lower to allow more splits
}

# --- Splitting and Merging Parameters ---
splitting_threshold = 0.15    # Even more aggressive splitting
merging_threshold = 0.85     # Much more conservative merging
min_cluster_size = 2         # Allow smaller clusters
grace_period_events = 3      # Shorter grace period
similarity_penalty = 0.3     # Stronger penalty for dissimilarity
adaptive_threshold_min_variability = 0.6  # Lower minimum variability

# --- Logging Parameters ---
log_frequency = 10

def adaptive_threshold_variability(feature_vectors):
    """
    Compute variability factor based on feature vectors' dispersion.
    Higher variability leads to less aggressive splitting/merging thresholds.
    """
    if len(feature_vectors) <= 1:
        return adaptive_threshold_min_variability

    distances = []
    for i in range(len(feature_vectors)):
        for j in range(i + 1, len(feature_vectors)):
            distances.append(np.linalg.norm(np.array(feature_vectors[i]) - np.array(feature_vectors[j])))

    if not distances:
        return adaptive_threshold_min_variability

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    variability_factor = min(max((mean_distance + std_distance) / 10.0, adaptive_threshold_min_variability), 1.5)
    return variability_factor