# --- Perspective Column Mapping ---
control_flow_column = 'Activity'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
case_id_column = 'CaseID'  # Define CaseID explicitly
event_id_column = 'EventID'  # Add EventID column for explicit exclusion

# --- Data Columns ---
data_columns = []  # To be auto-detected

# --- Discretization and Binning Parameters ---
features_to_discretize = ['NumericFeature_1', 'NumericFeature_2', 'NumericFeature_3']
quantiles = [0.25, 0.5, 0.75]
sliding_window_size = 150
bin_density_threshold = 10
drift_threshold = 0.05  # For ADWIN drift detection in binning
grace_period_events = 50  # Number of events to delay decay for new vectors

# --- Feature Selection Parameters ---
top_n_features = 3
forgetting_factor = 0.9
adaptive_window_min_size = 50
adaptive_window_max_size = 200
initial_window_size = 100
temporal_decay_rate = 0.001  # Reduced from 0.01
forgetting_threshold = 0.001  # Increased from 0.0001
positional_penalty_alpha = 0.8  # Positional penalty for misaligned features

# --- Clustering and Drift Detection Parameters ---
dbstream_params = {
    "clustering_threshold": 0.9,
    "fading_factor": 0.02,
    "cleanup_interval": 2,
    "intersection_factor": 0.3,
    "grace_period_events": 50
}

# --- Splitting and Merging Parameters ---
splitting_threshold = 0.70
merging_threshold = 0.80

# --- Logging Parameters ---
log_frequency = 10
