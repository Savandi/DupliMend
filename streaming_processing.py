from collections import defaultdict, deque
from tdigest import TDigest
import pandas as pd
import time

# Initialize constants and structures
quantiles = [0.25, 0.5, 0.75]  # Define quantile thresholds
features_to_discretize = ['age', 'heart_rate']  # Specify the features to discretize
sliding_window = deque(maxlen=100)  # Sliding window to maintain recent values


# Class for Decaying T-Digest
class DecayingTDigest(TDigest):
    def __init__(self, decay_factor=0.9):
        super().__init__()
        self.decay_factor = decay_factor

    def update_with_decay(self, value):
        # Apply decay to all centroids
        for centroid in self.C:
            centroid.mean *= self.decay_factor
            centroid.count *= self.decay_factor
        self.update(value)


# Initialize Decaying T-Digest
decaying_digest = DecayingTDigest()


# Adaptive quantile binning with density-sensitive adjustments
def adaptive_quantile_binning(value, bin_density_threshold=10):
    # Update Decaying T-Digest and get quantile bins
    decaying_digest.update_with_decay(value)
    bins = [decaying_digest.percentile(q * 100) for q in quantiles]
    density = [0] * len(bins)  # Track density for each bin

    # Assign value to a bin and adjust bins adaptively
    for i, bin_threshold in enumerate(bins):
        if value < bin_threshold:
            density[i] += 1
            # Split bin if density exceeds the threshold
            if density[i] > bin_density_threshold:
                bins.insert(i + 1, (bins[i] + bins[i + 1]) / 2)  # Add a midpoint bin
            return f"Quantile_Bin_{i}"

    # Merge sparse bins if density is too low
    sparse_bins = [i for i, d in enumerate(density) if d < bin_density_threshold / 2]
    for i in sparse_bins:
        if i + 1 < len(bins):
            bins[i] = (bins[i] + bins[i + 1]) / 2  # Merge bins
            bins.pop(i + 1)  # Remove the next bin
    return f"Quantile_Bin_{len(bins)}"


# Sliding window binning with recency sensitivity
def sliding_window_binning(value):
    # Add value to sliding window
    sliding_window.append(value)
    sorted_window = sorted(sliding_window)  # Sort the sliding window
    bins = [sorted_window[int(len(sorted_window) * q)] for q in quantiles]  # Calculate quantiles

    # Assign value to a bin
    for i, bin_threshold in enumerate(bins):
        if value < bin_threshold:
            return f"Quantile_Bin_{i}"
    return f"Quantile_Bin_{len(bins)}"


# Process event with both enhancements
def process_event(event):
    for feature in features_to_discretize:
        if feature in event:
            # Apply adaptive quantile binning with sliding window
            event[f'{feature}_quantile_bin'] = adaptive_quantile_binning(event[feature])
    return event


# Streaming function to process events
def stream_event_log(df, delay=1):
    for _, event in df.iterrows():
        event = process_event(event)  # Process the event
        yield event
        time.sleep(delay)  # Simulate streaming delay


# Load and prepare event log
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
df_event_log['Timestamp'] = pd.to_datetime(df_event_log['Timestamp'])  # Convert timestamp to datetime
df_event_log = df_event_log.sort_values(by='Timestamp')  # Sort by timestamp


# Process each event in the stream
for event in stream_event_log(df_event_log):
    print("Processed event with adaptive quantile bins and sliding window:\n", event)
