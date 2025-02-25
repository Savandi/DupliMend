from collections import defaultdict, deque
import numpy as np

# --- Perspective Column Mapping ---
control_flow_column = 'Activity'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
case_id_column = 'CaseID'
event_id_column = 'EventID'

# --- Enhanced Discretization and Binning Parameters ---
features_to_discretize = ['ProcessingTime', 'PriorityLevel', 'FileSize']
quantiles = [0.25, 0.5, 0.75]  # Base quantile points
sliding_window_size = 2000
bin_density_threshold = 10

# New binning parameters
density_estimation_points = 100
min_bin_width = 0.005       # Finer-grained binning
decay_factor = 0.9          # Slightly faster decay
initial_bins = 15           # More initial bins
drift_threshold = 0.03      # More sensitive to drift
merge_threshold = 0.6    # Less aggressive bin merging
# Enhanced temporal binning
time_feature_columns = ["hour_bin", "day_of_week", "is_weekend", "week_of_month", "season", "month"]

# --- Feature Selection Parameters ---
max_top_n_features = 7       # Include more features for better discrimination
forgetting_factor = 0.8      # Forget old patterns faster
temporal_decay_rate = 0.05  # Faster temporal decay
forgetting_threshold = 0.0005  # Lower threshold to maintain more information
positional_penalty_alpha = 0.3  # Stronger positional penalty
adaptive_window_min_size = 50
adaptive_window_max_size = 200
initial_window_size = 100
lossy_counting_threshold = 0.01
lossy_counting_error_rate = 0.01
previousEvents = 3

# --- Clustering and Drift Detection Parameters ---
dbstream_params = {
    "clustering_threshold": 0.35,
    "fading_factor": 0.05,
    "cleanup_interval": 2,
    "split_threshold": 0.15,
    "merge_threshold": 0.6,
    "eps": 0.02,
    "beta": 0.15,
    "lambda": 0.001
}
lossy_counting_budget = 500  # Maximum number of stored feature vectors before forgetting kicks in
frequency_decay_threshold = 0.0001  # Minimum frequency for retaining old clusters
decay_after_events = 200  # Number of events before applying decay
removal_threshold_events = decay_after_events * 2


# Homonym Handling
enable_homonym_debugging = True  # Enable detailed logging for homonym detection


# Add or update thresholds for similarity
similarity_threshold_high = 0.8  # High similarity (e.g., for almost identical events)
similarity_threshold_low = 0.4   # Low similarity (e.g., for loose grouping)

# --- Splitting and Merging Parameters ---
splitting_threshold = 0.15  # Increase to allow meaningful splits
merging_threshold = 0.80  # Reduce to permit more merges
adaptive_threshold_min_variability = 0.2  # Increase for better variability tracking
min_cluster_size = 1         # Allow any size clusters
grace_period_events = 1      # Minimal grace period
similarity_penalty = 0.05    # Maximum penalty for dissimilarity

dynamic_threshold_enabled = True
default_similarity_threshold = 0.5
threshold_adjustment_factor = 0.05

# --- Logging Parameters ---
log_frequency = 10

# Forgetting Parameters
decay_after_events = 10
lossy_counting_budget = 500

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

# Define window configurations globally
feature_window_sizes = defaultdict(lambda: initial_window_size)
feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))

def configure_window_sizes():
    """Configures initial sliding window sizes."""
    global feature_window_sizes, feature_importance_windows
    feature_window_sizes = defaultdict(lambda: initial_window_size)
    feature_importance_windows = defaultdict(lambda: deque(maxlen=initial_window_size))