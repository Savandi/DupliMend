from collections import defaultdict, deque
from dynamic_feature_vector_construction import process_event
from dynamic_binning_and_categorization import stream_event_log
from feature_selection_with_drift_detection import select_features, configure_window_sizes
from config import *
import pandas as pd
import time

# Configure sliding window sizes
configure_window_sizes()

# Load and prepare the event log
df_event_log = pd.read_csv('C:/Users/Kalukapu/Documents/Mine Log Abstract 2.csv', encoding='ISO-8859-1')

# Auto-detect data_columns
excluded_columns = {control_flow_column, timestamp_column, resource_column, case_id_column}
data_columns = [col for col in df_event_log.columns if col not in excluded_columns]

df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])
df_event_log = df_event_log.sort_values(by=timestamp_column)

# Streaming and processing events
processed_event_ids = set()
sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))

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
    event_id = event.get("EventID")
    if event_id in processed_event_ids:
        continue

    print(f"Processing Event ID: {event_id}")
    top_features = select_features(event, None, control_flow_column, timestamp_column, resource_column, data_columns)

    print(f"Activity: {event[control_flow_column]}, Top Features: {top_features}")
    result = process_event(event, top_features, timestamp_column)
    print(f"Change Detected: {result}")

    processed_event_ids.add(event_id)
    time.sleep(0.1)
