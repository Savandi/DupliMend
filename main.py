from dynamic_binning_and_categorization import stream_event_log
from feature_selection_with_drift_detection import select_features, configure_window_sizes
from dynamic_feature_vector_construction import process_event
from config import *
import pandas as pd
import time
from collections import defaultdict, deque

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

# Sliding windows for Unified Recency-Sensitive Mechanism
sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))  # Unified sliding windows

# Track processed EventIDs to ensure unique processing
processed_event_ids = set()
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
    event_id = event.get("EventID")
    if event_id in processed_event_ids:
        continue  # Skip already processed events

    print("Dynamic Binning and Categorization Processed Event:", event)

    # Step 2: Feature Selection
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

    # Step 3: Dynamic Feature Vector Construction
    unified_vector = process_event(
        event, top_features, timestamp_column, sliding_windows, decay_rate=temporal_decay_rate
    )
    print(f"Dynamic Feature Vector: {unified_vector}")

    processed_event_ids.add(event_id)  # Mark this event as processed
    previous_event = event  # Update the previous event
    time.sleep(0.1)  # Simulate streaming delay
