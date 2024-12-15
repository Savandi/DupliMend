# Perspective Column Mapping
control_flow_column = 'Activity'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
case_id_column = 'CaseID'
data_columns = []  # To be auto-detected

# Discretization and Binning Parameters
features_to_discretize = ['age', 'heart_rate', timestamp_column]
quantiles = [0.25, 0.5, 0.75]
sliding_window_size = 100
bin_density_threshold = 10

# Feature Selection Parameters
top_n_features = 3
forgetting_factor = 0.9
adaptive_window_min_size = 50
adaptive_window_max_size = 200
initial_window_size = 100
temporal_decay_rate = 0.01

# Clustering and Drift Detection Parameters
dbstream_params = {
    "clustering_threshold": 1.0,
    "fading_factor": 0.01,
    "cleanup_interval": 2,
    "intersection_factor": 0.3,
    "minimum_weight": 1.0,
}
