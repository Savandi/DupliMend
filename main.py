import time
from collections import defaultdict, deque
import pandas as pd
from config.config import *
from src.homonym_mend.dynamic_binning_and_categorization import stream_event_log, extract_temporal_features, \
    EnhancedAdaptiveBinning
from src.homonym_mend.homonym_detection import analyze_splits_and_merges, dbstream_clusters
from src.homonym_mend.label_refinement import LabelRefiner
from src.utils.logging_utils import log_traceability

def initialize_window_sizes():
    """
    Lazily import configure_window_sizes to avoid circular import issues.
    """
    from src.homonym_mend.feature_selection_with_drift_detection import configure_window_sizes  # Lazy import
    configure_window_sizes()

def  get_feature_scores(event, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter):
    """
    Lazily import compute_feature_scores to avoid circular import issues.
    """
    from src.homonym_mend.feature_selection_with_drift_detection import compute_feature_scores  # Lazy import
    return compute_feature_scores(event, case_id_column, control_flow_column, timestamp_column, resource_column, data_columns, global_event_counter)

def get_top_features(event, control_flow_column, timestamp_column, resource_column, data_columns):
    """
    Lazily import select_features to avoid circular import issues.
    """
    from src.homonym_mend.feature_selection_with_drift_detection import select_features  # Lazy import
    return select_features(event, None, control_flow_column, timestamp_column, resource_column, data_columns)

def construct_feature_vector(event, top_features):
    """
    Lazily import process_event to avoid circular import issues.
    """
    from src.homonym_mend.dynamic_feature_vector_construction import process_event  # Lazy import
    return process_event(event, top_features)

def initialize_binning_models():
    """Initialize the enhanced binning models with refined parameters."""
    return {
        feature: EnhancedAdaptiveBinning(
            initial_bins=20,  # More bins to distribute values better
            bin_density_threshold=10,  # Higher threshold to prevent aggressive merging
            drift_threshold=0.02,  # Slightly more sensitive to drift
            decay_factor=0.85,  # Adjust decay to retain history better
            min_bin_width=0.005,  # Prevent small bins
            quantile_points=[0.1, 0.3, 0.5, 0.7, 0.9]  # Adjusted quantile points
        )
        for feature in features_to_discretize
    }

# Define global_event_counter in main.py instead of global_state.py
global_event_counter = 0

# Configure sliding window sizes
initialize_window_sizes()

input_log_path = './src/homonym_mend/synthetic_log_with_homonyms.csv'
# Load and prepare event log
df_event_log = pd.read_csv(input_log_path, encoding='ISO-8859-1')

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

# Extract column names dynamically and store globally
input_columns = list(df_event_log.columns)

# Ensure "refined_activity" is included in output
input_columns.append("refined_activity")

# Convert timestamp and sort
df_event_log[timestamp_column] = pd.to_datetime(df_event_log[timestamp_column])
df_event_log = df_event_log.sort_values(by=timestamp_column)
df_event_log = df_event_log.head(50)

# Initialize enhanced binning models
binning_models = initialize_binning_models()

# Initialize LabelRefiner
refined_log_path = f"./refined_log.csv"
label_refiner = LabelRefiner(refined_log_path, input_columns)

# Streaming and processing events
print("\n=== Initial Log Statistics ===")
print(f"Total events: {len(df_event_log)}")
print(f"Total unique Event IDs: {len(df_event_log['EventID'].unique())}")
print(f"Unique activities: {df_event_log['Activity'].unique()}")
activity_counts = df_event_log['Activity'].value_counts()
print(f"Activity frequencies:\n{activity_counts}")

processed_event_ids = set()
sliding_windows = defaultdict(lambda: deque(maxlen=sliding_window_size))

previous_event = None

for _, event in df_event_log.iterrows():
    try:
        global_event_counter += 1
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
            binning_models=binning_models  # <-- Pass the initialized binning models
        )

        # Compute feature scores, including adaptive binning updates
        feature_scores = get_feature_scores(
            event=processed_event,
            case_id_column=case_id_column,
            control_flow_column=control_flow_column,
            timestamp_column=timestamp_column,
            resource_column=resource_column,
            data_columns=data_columns + list(temporal_features.keys()),
            global_event_counter=global_event_counter
        )

        # Log scores for debugging
        print(f"Feature Scores for Event ID {processed_event[event_id_column]}: {feature_scores}")

        # Control-flow debug
        if previous_event:
            print(
                f"Control-Flow Debug: Previous Activity: {previous_event[control_flow_column]}, "
                f"Current Activity: {processed_event[control_flow_column]}"
            )

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

        print(f"Processing Event {global_event_counter}, Event ID: {event_id}, Activity: {activity_label}", flush=True)

        top_features = get_top_features(event, control_flow_column, timestamp_column, resource_column, data_columns)

        print(f"Top Features: {top_features}")

        # Process the event to construct feature vector
        feature_vector_data = construct_feature_vector(event, top_features)
        if feature_vector_data is None:
            print(f"Warning: Feature vector construction failed for Event ID: {event_id}. Skipping event.")
            continue

        # Retrieve the correct DBStream instance for the activity label
        activity_label = feature_vector_data["activity_label"]
        dbstream_instance = dbstream_clusters.get(activity_label, None)

        if dbstream_instance:
            split_merge_result, cluster_id = analyze_splits_and_merges(activity_label, dbstream_instance)
        else:
            split_merge_result, cluster_id = "no_change", 0  # Default behavior if no DBStream instance exists

        print(f"Split/Merge Result for Event ID {event_id}: {split_merge_result}")

        # Refine the activity label based on clustering
        refined_activity = label_refiner.refine_label(activity_label, cluster_id)
        event["refined_activity"] = refined_activity

        # Append the refined event to the output log
        label_refiner.append_event_to_csv(event)

        # Store refined activity mapping for future events
        activity_feature_metadata[activity_label][tuple(feature_vector_data["new_vector"])][
            "refined_label"] = refined_activity

        processed_event_ids.add(event_id)
        print(f"Added Event ID {event_id} to processed_event_ids")
        time.sleep(0.1)

    except Exception as e:
        print(f"Error processing event ID {event.get(event_id_column)}: {e}")
        log_traceability("error", "Event Processing", {
            "event_id": event.get(event_id_column),
            "error": str(e)
        })

# Save the complete refined stream after processing all events
print(f"Refined stream has been saved to {refined_log_path}")