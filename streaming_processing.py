from tdigest import TDigest
import pandas as pd
import time

# Initialize T-digest for quantile estimation
quantile_digest = TDigest()
quantiles = [0.25, 0.5, 0.75]  # Define quantile thresholds

# Define the features to be discretized
features_to_discretize = ['age', 'heart_rate']  # Specify the features you want to discretize


# Function to apply quantile binning
def quantile_binning(value):
    quantile_digest.update(value)  # Update t-digest with new value
    bins = [quantile_digest.percentile(q * 100) for q in quantiles]

    for i, bin_threshold in enumerate(bins):
        if value < bin_threshold:
            return f"Quantile_Bin_{i}"
    return f"Quantile_Bin_{len(bins)}"


# Load and prepare event log (assuming sorted by timestamp)
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
df_event_log['timestamp'] = pd.to_datetime(df_event_log['timestamp'])
df_event_log = df_event_log.sort_values(by='timestamp')


# Process each event and apply quantile binning on specified features
def process_event(event):
    for feature in features_to_discretize:
        if feature in event:
            event[f'{feature}_quantile_bin'] = quantile_binning(event[feature])
    return event


# Streaming function to process each event
def stream_event_log(df, delay=1):
    for _, event in df.iterrows():
        event = process_event(event)
        yield event
        time.sleep(delay)


# Process each event in the stream
for event in stream_event_log(df_event_log):
    print("Processed event with quantile-based bins:\n", event)
    # Add any additional processing here
