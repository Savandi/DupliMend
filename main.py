from collections import defaultdict, deque
from dbstream import DBStream
from dynamic_feature_vector_construction import process_event
from dynamic_binning_and_categorization import stream_event_log
from feature_selection_with_drift_detection import select_features, configure_window_sizes
from config import *
import pandas as pd
import time
from homonym_detection import handle_temporal_decay, log_cluster_summary

# Configure sliding window sizes
configure_window_sizes()

# Initialize DBStream instances
dbstream_instances = defaultdict(lambda: DBStream(**dbstream_params))

# Load and prepare the event log
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')

# Auto-detect data_columns
excluded_columns = {control_flow_column, timestamp_column, resource_column, case_id_column}
data_columns = [col for col in df_event_log.columns if col not in excluded_columns]

df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])
df_event_log = df_event_log.sort_values(by=timestamp_column)

# Streaming and processing events
processed_event_ids = set()
sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))

for event in stream_event_log(
    df=df_event_log,
    timestamp_column=timestamp_column,
    control_flow_column=control_flow_column,
    resource_column=resource_column,
    case_id_column=case_id_column,
    data_columns=data_columns,
    features_to_discretize=features_to_discretize,
    sliding_window_size=sliding_window_size,
    bin_density_threshold=bin_density_threshold,
    quantiles=quantiles,
    delay=1
):
    event_id = event.get("EventID")
    if event_id in processed_event_ids:
        continue

    activity_label = event[control_flow_column]
    print(f"Processing Event ID: {event_id}, Activity: {activity_label}")

    top_features = select_features(
        event,
        None,
        control_flow_column,
        timestamp_column,
        resource_column,
        data_columns,
        top_n=top_n_features
    )
    print(f"Top Features: {top_features}")

    # Process event
    result = process_event(event, top_features, timestamp_column)
    print(f"Change Detected: {result}")
    # Periodically log cluster summaries and handle temporal decay
    if isinstance(event_id, int) and event_id % log_frequency == 0:
        log_cluster_summary(dbstream_instances[activity_label])
        handle_temporal_decay(activity_label)

    processed_event_ids.add(event_id)
    time.sleep(0.1)
