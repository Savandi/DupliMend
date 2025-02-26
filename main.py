import sys
from src.homonym_mend.dbstream import DBStream
from src.utils.global_state import dbstream_clusters
from src.homonym_mend.homonym_detection import analyze_splits_and_merges
from src.utils.global_state import extract_temporal_features
print(f"Start of main, called from: {sys.argv[0]}")
import time
import pandas as pd
from config.config import *
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log, \
    EnhancedAdaptiveBinning
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_metadata, process_event
from src.homonym_mend.label_refinement import LabelRefiner
from src.utils.logging_utils import log_traceability

def is_valid_timestamp(ts):
    """
    Attempts to parse the timestamp using multiple formats to handle variations.
    Returns True if the timestamp is valid, otherwise False.
    """
    timestamp_formats = [
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %I:%M %p", "%Y/%m/%d %H:%M:%S",
        "%m-%d-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%a %b %d %H:%M:%S %Y"
    ]

    if isinstance(ts, str):
        for fmt in timestamp_formats:
            try:
                pd.to_datetime(ts, format=fmt)
                return True
            except ValueError:
                continue
    return False

# Lazy Import Functions to Prevent Circular Import Issues
def get_feature_scores(event, event_id_column, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
    """
    Lazily import compute_feature_scores to avoid circular import issues.
    """
    try:
        from src.homonym_mend.feature_selection_with_drift_detection import compute_feature_scores  # Lazy import
        return compute_feature_scores(event, event_id_column, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter)
    except ImportError as e:
        print(f"[ERROR] Circular import detected in feature selection: {e}")
        return {}

def get_top_features(event, event_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
    from src.homonym_mend.feature_selection_with_drift_detection import select_features  # ✅ Lazy import
    return select_features(event, event_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter)

def construct_feature_vector(event, top_features, global_event_counter):
    """
    Lazily import process_event to avoid circular import issues.
    """
    from src.homonym_mend.dynamic_feature_vector_construction import process_event  # ✅ Lazy import
    return process_event(event, top_features, global_event_counter)

# def get_homonym_analysis():
#     """
#     Lazily import analyze_splits_and_merges to avoid circular import issues.
#     """
#     from src.homonym_mend.homonym_detection import analyze_splits_and_merges  # ✅ Lazy import
#     return analyze_splits_and_merges  # ✅ Remove dbstream_clusters

def get_or_create_dbstream(activity_label):
    """
    Retrieve existing DBStream instance or create a new one.
    """
    if activity_label not in dbstream_clusters:
        from src.homonym_mend.dbstream import DBStream  # Lazy import to prevent circular import issues
        dbstream_clusters[activity_label] = DBStream()
        log_traceability("dbstream_create", activity_label, "Created new DBStream instance")

    return dbstream_clusters[activity_label]




def main():
    global_event_counter = 0

    # Initialize window sizes
    configure_window_sizes()

    input_log_path = './src/homonym_mend/synthetic_log_with_homonyms.csv'

    # Load and prepare event log
    df_event_log = pd.read_csv(input_log_path, encoding='ISO-8859-1')

    # Convert timestamps for sorting
    df_event_log[timestamp_column] = pd.to_datetime(
        df_event_log[timestamp_column], format="%Y-%m-%dT%H:%M:%S.%f", errors='coerce', utc=True
    )

    # Auto-detect data columns
    excluded_columns = {control_flow_column, timestamp_column, 'original_timestamp',
                        resource_column, case_id_column, event_id_column}
    data_columns = [col for col in df_event_log.columns if col not in excluded_columns]

    print(f"Data columns used: {data_columns}")

    # Extract column names dynamically
    input_columns = list(df_event_log.columns) + ["refined_activity"]

    # Sort events by timestamp
    df_event_log = df_event_log.sort_values(by=timestamp_column)

    # Initialize binning models
    binning_models = {
        feature: EnhancedAdaptiveBinning(
            initial_bins=20, bin_density_threshold=5, drift_threshold=0.02,
            decay_factor=0.85, min_bin_width=0.001, quantile_points=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        )
        for feature in features_to_discretize
    }

    # Initialize LabelRefiner
    refined_log_path = "./refined_log.csv"
    label_refiner = LabelRefiner(refined_log_path, input_columns)

    # Print initial log statistics
    print("\n=== Initial Log Statistics ===")
    print(f"Total events: {len(df_event_log)}")
    print(f"Total unique Event IDs: {len(df_event_log['EventID'].unique())}")
    print(f"Unique activities: {df_event_log['Activity'].unique()}")
    print(f"Activity frequencies:\n{df_event_log['Activity'].value_counts()}")

    processed_event_ids = set()

    # Process events one by one
    for _, event in df_event_log.iterrows():
        try:
            global_event_counter += 1
            print(f"\nStart Processing Event {global_event_counter}")

            # Extract temporal features
            temporal_features = extract_temporal_features(event[timestamp_column])
            if not temporal_features:
                print(f"[ERROR] Failed to extract temporal features for Event {event[event_id_column]}")
                continue

            # Update event with temporal features
            event_dict = event.to_dict()
            event_dict.update(temporal_features)

            print(f"[DEBUG] Event with temporal features: {event_dict}")

            # Process the event through binning model
            processed_event = stream_event_log(
                event_dict, timestamp_column, control_flow_column, resource_column,
                case_id_column, event_id_column, data_columns, features_to_discretize, binning_models
            )

            processed_event_copy = processed_event.copy()

            print(f"Start Feature Selection and Importance Analysis for {global_event_counter}")
            feature_scores = get_feature_scores(
                processed_event_copy, event_id_column, case_id_column, control_flow_column,
                timestamp_column, resource_column, data_columns, global_event_counter
            )
            print(f"Feature Scores for Event {event[event_id_column]}: {feature_scores}")

            # Check if event was already processed
            event_id = event.get(event_id_column)
            if event_id in processed_event_ids:
                print(f"Skipping already processed event ID: {event_id}")
                continue

            activity_label = event.get(control_flow_column)
            print(f"Processing Event {global_event_counter}, Event ID: {event_id}, Activity: {activity_label}")

            print(f"Start Getting Top Features for {global_event_counter}")
            top_features = get_top_features(
                processed_event_copy, event_id_column, control_flow_column, timestamp_column,
                resource_column, data_columns, global_event_counter
            )

            print(f"Start Dynamic Feature Vector Construction for {global_event_counter}")
            feature_vector_data = process_event(event_dict, top_features, global_event_counter)
            if feature_vector_data is None:
                print(f"[ERROR] Feature vector missing for activity: {activity_label}")
                continue

            print(f"[DEBUG] Feature vector for {activity_label}: {feature_vector_data['new_vector']}")

            # Perform homonym detection
            print(f"Start Homonym Detection and Online Clustering with Dbstream for {global_event_counter}")

            # Retrieve or create DBStream instance
            if activity_label not in dbstream_clusters:
                dbstream_clusters[activity_label] = DBStream()  # Initialize only once

            print(f"[DEBUG] Current DBStream instances: {list(dbstream_clusters.keys())}")

            dbstream_instance = get_or_create_dbstream(activity_label)

            # Assign cluster ID for the feature vector
            assigned_cluster_id = dbstream_instance.partial_fit(feature_vector_data["new_vector"], activity_label)
            print(f"[DEBUG] Assigned Cluster ID for {activity_label}: {assigned_cluster_id}")
            # Analyze splits and merges
            split_merge_result, cluster_id = analyze_splits_and_merges(
                activity_label, dbstream_instance, feature_vector_data["new_vector"], assigned_cluster_id
            )

            # Refine activity label
            print(f"Start Label Refinement for {global_event_counter}")
            refined_activity = label_refiner.refine_label(activity_label, cluster_id, dbstream_instance)
            event_dict["refined_activity"] = refined_activity
            label_refiner.append_event_to_csv(event_dict)

            # Update feature metadata
            print(f"Converting feature vector to immutable tuple")
            feature_vector_tuple = tuple(feature_vector_data["new_vector"])
            activity_feature_metadata[activity_label][feature_vector_tuple]["refined_label"] = refined_activity

            processed_event_ids.add(event_id)
            print(f"Added Event ID {event_id} to processed_event_ids")
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing event {event.get(event_id_column, 'Unknown')}: {e}")
            log_traceability("error", "Event Processing", {
                "event_id": event.get(event_id_column, 'Unknown'),
                "error": str(e)
            })

    print(f"Refined stream has been saved to {refined_log_path}")


if __name__ == "__main__":
    main()