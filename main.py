from dynamic_binning_and_categorization import stream_event_log
from feature_selection_with_drift_detection import select_features
import pandas as pd
import time

# --- MAIN SCRIPT ---
# Load and prepare event log
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
df_event_log['timestamp'] = pd.to_datetime(df_event_log['timestamp'])  # Convert timestamp to datetime
df_event_log = df_event_log.sort_values(by='timestamp')  # Sort by timestamp

# Streaming and processing events
previous_event = None  # Track the previous event for feature selection
for event in stream_event_log(df_event_log):  # Process events with dynamic binning and categorization
    print("Dynamic Binning and Categorization Processed Event:", event)  # Output Step 1 results

    # Pass Step 1 output to feature selection and clustering
    top_features = select_features(event, previous_event)  # Select features with drift detection
    print("Feature Selection and Drift Detection - Top Features:", top_features)  # Output Step 2 results

    # Update the previous event for Step 2
    previous_event = event

    # Optional delay to simulate real-time streaming
    time.sleep(0.1)
