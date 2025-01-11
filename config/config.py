# --- Perspective Column Mapping ---
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
drift_threshold = 0.05
grace_period_events = 100
cluster_grace_period_seconds = 25

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
    "cleanup_interval": 100,
    "intersection_factor": 0.3
}

# --- Splitting and Merging Parameters ---
splitting_threshold = 0.85
merging_threshold = 0.85
adaptive_threshold_min_variability = 0.8

# --- Logging Parameters ---
log_frequency = 10
