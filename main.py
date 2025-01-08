from collections import defaultdict, deque
from src.homonym_mend.dynamic_feature_vector_construction import process_event as construct_feature_vector
from src.homonym_mend.homonym_detection import process_event as analyze_split_merge, dbstream_clusters
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log
from src.homonym_mend.feature_selection_with_drift_detection import select_features, configure_window_sizes
from src.homonym_mend.homonym_detection import handle_temporal_decay, log_cluster_summary
from config.config import *
import pandas as pd
import time
import logging

from src.utils.logging_utils import log_traceability

from config.config import (
    event_id_column, case_id_column, features_to_discretize, data_columns
)
# Configure sliding window sizes
configure_window_sizes()
# Generate synthetic data
#synthetic_data = generate_synthetic_dataset(num_clusters=3, num_samples_per_cluster=50, overlap=True)

#print(synthetic_data.head())

# Ensure required columns are formatted properly
#synthetic_data = synthetic_data.rename(columns={
#     'Feature_1': 'NumericFeature_1',
#     'Feature_2': 'NumericFeature_2',
#     'Feature_3': 'NumericFeature_3'
# })

# Set the dataframe as the event log
#df_event_log = synthetic_data
df_event_log = pd.read_csv('./src/homonym_mend/updated_combined_synthetic_log.csv', encoding='ISO-8859-1')
# Load and prepare the event log
# df_event_log = pd.read_csv('C:/Users/Kalukapu/Documents/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
#df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
# Auto-detect data_columns
excluded_columns = {control_flow_column, timestamp_column, resource_column, case_id_column, event_id_column}
data_columns = [col for col in df_event_log.columns if col not in excluded_columns]
df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])
df_event_log = df_event_log.sort_values(by=timestamp_column)


# Streaming and processing events
print(f"Total events in dataset: {len(df_event_log)}")
print(f"Total unique Event IDs: {len(df_event_log['EventID'].unique())}")
processed_event_ids = set()

sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))

for iteration, event in enumerate(stream_event_log(
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
), start=1):
    event_id = event.get("EventID")
    if event_id in processed_event_ids:
        continue

    activity_label = event[control_flow_column]
    print(f"Processing Event: {iteration}, Activity: {activity_label}", flush=True)

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

    # Step 1: Construct feature vector
    feature_vector_data = construct_feature_vector(event, top_features, timestamp_column)
    if not feature_vector_data:
        print("Feature vector construction failed. Skipping event.")
        continue
    # Log the constructed feature vector
    print(f"Constructed Feature Vector: {feature_vector_data['new_vector']}")

    # Step 2: Analyze splits and merges
    split_merge_result = analyze_split_merge(feature_vector_data)
    print(f"Split/Merge Analysis Result: {split_merge_result}")

    # Log the split and merge analysis result
    log_traceability("split_merge_analysis", feature_vector_data["activity_label"], {"result": split_merge_result})

    # Periodically log cluster summaries and handle temporal decay
    if isinstance(event_id, int) and event_id % log_frequency == 0:
        log_cluster_summary(dbstream_clusters[activity_label])
        handle_temporal_decay(activity_label)

    processed_event_ids.add(event_id)
    time.sleep(0.1)
