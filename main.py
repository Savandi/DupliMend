import time
from collections import defaultdict, deque
import pandas as pd

from config.config import *
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log, extract_temporal_features, \
    EnhancedAdaptiveBinning
from src.homonym_mend.dynamic_feature_vector_construction import process_event as construct_feature_vector
from src.homonym_mend.feature_selection_with_drift_detection import (
    select_features, configure_window_sizes, compute_feature_scores
)
from src.homonym_mend.homonym_detection import (
    handle_temporal_decay, log_cluster_summary, process_event as analyze_split_merge,
    dbstream_clusters
)
from src.utils.directly_follows_graph import DirectlyFollowsGraph
from src.utils.logging_utils import log_traceability


def initialize_binning_models():
    """Initialize the enhanced binning models with configuration parameters."""
    return {
        feature: EnhancedAdaptiveBinning(
            initial_bins=initial_bins,
            bin_density_threshold=bin_density_threshold,
            drift_threshold=drift_threshold,
            decay_factor=decay_factor,
            min_bin_width=min_bin_width,
            quantile_points=quantiles
        )
        for feature in features_to_discretize
    }


def main():
    # Initialize Directly Follows Graph
    directly_follows_graph = DirectlyFollowsGraph()

    # Configure sliding window sizes
    configure_window_sizes()

    # Load and prepare event log
    df_event_log = pd.read_csv('./src/homonym_mend/synthetic_log_with_homonyms.csv', encoding='ISO-8859-1')

    def is_valid_timestamp(ts):
        try:
            pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S', errors='raise')
            return True
        except Exception:
            return False

    df_event_log = df_event_log[~df_event_log[timestamp_column].apply(is_valid_timestamp)]

    # Auto-detect data_columns
    excluded_columns = {control_flow_column, timestamp_column, resource_column, case_id_column, event_id_column}
    data_columns = [col for col in df_event_log.columns if col not in excluded_columns]
    print(f"Data columns used: {data_columns}")

    # Convert timestamp and sort
    df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])
    df_event_log = df_event_log.sort_values(by=timestamp_column)

    # Initialize enhanced binning models
    binning_models = initialize_binning_models()

    # Streaming and processing events
    print(f"Total events in dataset: {len(df_event_log)}")
    print(f"Total unique Event IDs: {len(df_event_log['EventID'].unique())}")

    processed_event_ids = set()
    sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))
    event_counter = 1
    previous_event = None

    for _, event in df_event_log.iterrows():
        try:
            # Extract temporal features with enhanced granularity
            temporal_features = extract_temporal_features(event[timestamp_column])
            event.update(temporal_features)

            # Convert event to dictionary for processing
            event_dict = event.to_dict()

            # Process event with enhanced binning
            processed_event = stream_event_log(
                event_dict=event_dict,
                timestamp_column=timestamp_column,
                control_flow_column=control_flow_column,
                resource_column=resource_column,
                case_id_column=case_id_column,
                event_id_column=event_id_column,
                data_columns=data_columns,
                features_to_discretize=features_to_discretize,
                sliding_window_size=sliding_window_size,
                bin_density_threshold=bin_density_threshold,
                quantiles=quantiles,
                decay_factor=decay_factor
            )

            # Compute feature scores, including adaptive binning updates
            feature_scores = compute_feature_scores(
                event=processed_event,
                previous_event=previous_event,
                activity_column=control_flow_column,
                timestamp_column=timestamp_column,
                resource_column=resource_column,
                data_columns=data_columns + list(temporal_features.keys())
            )

            # Log scores for debugging
            print(f"Feature Scores for Event ID {processed_event[event_id_column]}: {feature_scores}")

            # Control-flow debug
            if previous_event:
                print(
                    f"Control-Flow Debug: Previous Activity: {previous_event[control_flow_column]}, "
                    f"Current Activity: {processed_event[control_flow_column]}"
                )
                # Update the directly follows graph with the transition
                from_activity = previous_event[control_flow_column]
                to_activity = processed_event[control_flow_column]
                directly_follows_graph.add_transition(from_activity, to_activity)

            # Update previous_event
            previous_event = processed_event

            event_id = event.get(event_id_column)
            if event_id is None:
                print("Warning: Event ID is None. Skipping event.")
                continue

            if event_id in processed_event_ids:
                print(f"Skipping already processed event ID: {event_id}")
                continue

            activity_label = event.get(control_flow_column)
            if activity_label is None:
                print(f"Warning: Activity label is None for Event ID: {event_id}. Skipping event.")
                continue

            print(f"Processing Event {event_counter}, Event ID: {event_id}, Activity: {activity_label}", flush=True)
            event_counter += 1

            top_features = select_features(
                event,
                None,
                control_flow_column,
                timestamp_column,
                resource_column,
                data_columns
            )
            print(f"Top Features: {top_features}")

            # Process the event to construct feature vector
            feature_vector_data = construct_feature_vector(event, top_features, directly_follows_graph)
            if feature_vector_data is None:
                print(f"Warning: Feature vector construction failed for Event ID: {event_id}. Skipping event.")
                continue

            # Analyze splits and merges
            split_merge_result = analyze_split_merge(feature_vector_data)
            print(f"Split/Merge Result for Event ID {event_id}: {split_merge_result}")

            processed_event_ids.add(event_id)
            print(f"Added Event ID {event_id} to processed_event_ids")
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing event ID {event.get(event_id_column)}: {e}")
            log_traceability("error", "Event Processing", {
                "event_id": event.get(event_id_column),
                "error": str(e)
            })


if __name__ == "__main__":
    main()