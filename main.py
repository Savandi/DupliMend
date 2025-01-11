import time
from collections import defaultdict, deque

import pandas as pd

from config.config import *
from config.config import (
    event_id_column, case_id_column, features_to_discretize
)
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log
from src.homonym_mend.dynamic_feature_vector_construction import process_event as construct_feature_vector
from src.homonym_mend.feature_selection_with_drift_detection import select_features, configure_window_sizes
from src.homonym_mend.homonym_detection import handle_temporal_decay, log_cluster_summary
<<<<<<< Updated upstream
from config.config import *
import pandas as pd
import time
import logging
logging.basicConfig(
    filename="../../logs/traceability_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
=======
from src.homonym_mend.homonym_detection import process_event as analyze_split_merge, dbstream_clusters
>>>>>>> Stashed changes
from src.utils.logging_utils import log_traceability

# Configure sliding window sizes
configure_window_sizes()

# Set the dataframe as the event log
<<<<<<< Updated upstream
#df_event_log = synthetic_data
df_event_log = pd.read_csv('./src/homonym_mend/updated_combined_synthetic_log.csv', encoding='ISO-8859-1')
# Load and prepare the event log
# df_event_log = pd.read_csv('C:/Users/Kalukapu/Documents/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
#df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
=======
df_event_log = pd.read_csv('./src/homonym_mend/synthetic_log_with_homonyms.csv', encoding='ISO-8859-1')

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
for iteration, event in enumerate(stream_event_log(
        df=df_event_log,
=======
event_counter = 1

for _, event in df_event_log.iterrows():
    event_dict = event.to_dict()
    processed_event = stream_event_log(
        event_dict=event_dict,
>>>>>>> Stashed changes
        timestamp_column=timestamp_column,
        control_flow_column=control_flow_column,
        resource_column=resource_column,
        case_id_column=case_id_column,
        event_id_column=event_id_column,
        data_columns=data_columns,
        features_to_discretize=features_to_discretize,
        sliding_window_size=sliding_window_size,
        bin_density_threshold=bin_density_threshold,
<<<<<<< Updated upstream
        quantiles=quantiles,
        delay=1
), start=1):
    event_id = event.get("EventID")
=======
        quantiles=quantiles
    )
    event_id = event.get(event_id_column)
    if event_id is None:
        print("Warning: Event ID is None. Skipping event.")
        continue

>>>>>>> Stashed changes
    if event_id in processed_event_ids:
        print(f"Skipping already processed event ID: {event_id}")
        continue

    activity_label = event[control_flow_column]
<<<<<<< Updated upstream
    print(f"Processing Event: {iteration}, Activity: {activity_label}", flush=True)
=======
    print(f"Processing Event {event_counter}, Event ID: {event_id}, Activity: {activity_label}", flush=True)
    event_counter += 1
>>>>>>> Stashed changes

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
    print(f"Added Event ID {event_id} to processed_event_ids")
    time.sleep(0.1)