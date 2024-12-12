from dynamic_binning_and_categorization import stream_event_log
from feature_selection_with_drift_detection import select_features, configure_window_sizes
from config import *
import pandas as pd
import time

# Configure sliding window sizes
configure_window_sizes()

# --- MAIN SCRIPT ---
# Load and prepare the event log
df_event_log = pd.read_csv('C:/Users/Kalukapu/Documents/Mine Log Abstract 2.csv', encoding='ISO-8859-1')

# Auto-detect data_columns
excluded_columns = {control_flow_column, timestamp_column, resource_column, case_id_column}
data_columns = [col for col in df_event_log.columns if col not in excluded_columns]

df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])  # Convert timestamp to datetime
df_event_log = df_event_log.sort_values(by=timestamp_column)  # Sort by timestamp

# Streaming and processing events
previous_event = None
for event in stream_event_log(
    df_event_log,
    timestamp_column=timestamp_column,
    control_flow_column=control_flow_column,
    resource_column=resource_column,
    case_id_column=case_id_column,
    data_columns=data_columns,
    features_to_discretize=features_to_discretize,
    quantiles=quantiles,
    sliding_window_size=sliding_window_size,
    bin_density_threshold=bin_density_threshold,
    dbstream_params=dbstream_params
):
    print("Dynamic Binning and Categorization Processed Event:", event)

    # Pass Step 1 output to feature selection and clustering
    top_features = select_features(
        event,
        previous_event,
        activity_column=control_flow_column,
        timestamp_column=timestamp_column,
        resource_column=resource_column,
        data_columns=data_columns,
        top_n=top_n_features
    )
    print(f"Activity: {event[control_flow_column]}, Top Features: {top_features}")
    previous_event = event

    time.sleep(0.1)  # Simulate streaming delay
