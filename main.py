from collections import defaultdict, deque
from src.homonym_mend.dynamic_feature_vector_construction import process_event
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log
from src.homonym_mend.feature_selection_with_drift_detection import select_features, configure_window_sizes
from src.homonym_mend.homonym_detection import handle_temporal_decay, log_cluster_summary
from config.config import *
import pandas as pd
import time
import logging

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
        log_cluster_summary(activity_label)  # Log clusters (handled by homonym_detection)
        handle_temporal_decay(activity_label)  # Apply decay (handled by homonym_detection)

    processed_event_ids.add(event_id)
    logging.basicConfig(
        filename="../../traceability_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    time.sleep(0.1)
