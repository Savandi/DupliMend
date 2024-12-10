from collections import defaultdict, deque
from tdigest import TDigest
from river.drift import ADWIN
import pandas as pd
import time

# --- PARAMETERS ---
quantiles = [0.25, 0.5, 0.75]  # Define quantile thresholds
features_to_discretize = ['age', 'heart_rate']  # Specify the features to discretize
adaptive_window_min_size = 50  # Minimum adaptive sliding window size
adaptive_window_max_size = 200  # Maximum adaptive sliding window size
initial_window_size = 100  # Initial sliding window size for each feature
bin_density_threshold = 10  # Threshold for adjusting bin density

# --- GLOBAL VARIABLES ---
sliding_windows = defaultdict(lambda: deque(maxlen=initial_window_size))  # Adaptive sliding windows for features
window_sizes = defaultdict(lambda: initial_window_size)  # Store adaptive window sizes for each feature
adwin_detectors = defaultdict(ADWIN)  # ADWIN drift detectors for each feature


# --- CLASS FOR DECAYING T-DIGEST ---
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


# --- ADAPTIVE SLIDING WINDOW FUNCTIONS ---
def adjust_window_size(feature, drift_detected):
    """
    Dynamically adjust sliding window size based on drift detection.
    """
    current_size = window_sizes[feature]
    if drift_detected:
        new_size = max(adaptive_window_min_size, current_size // 2)  # Halve the window size
    else:
        new_size = min(adaptive_window_max_size, current_size + 10)  # Gradually increase size
    window_sizes[feature] = new_size
    print(f"Adjusting window size for feature '{feature}' to: {new_size}")
    return deque(maxlen=new_size)


# --- DISCRETIZATION FUNCTIONS ---
def detect_drift(feature, sliding_window_values):
    """
    Detect drift in feature values using ADWIN.
    """
    adwin = adwin_detectors[feature]
    for value in sliding_window_values:
        adwin.update(value)
    return adwin.detected_change()


def adaptive_quantile_binning(value, feature):
    """
    Perform adaptive quantile binning with dynamic adjustments for density.
    """
    # Update Decaying T-Digest and sliding window
    decaying_digest.update_with_decay(value)
    sliding_windows[feature].append(value)

    # Detect drift and adjust window size if needed
    drift_detected = detect_drift(feature, sliding_windows[feature])
    sliding_windows[feature] = adjust_window_size(feature, drift_detected)

    # Compute quantile bins
    bins = [decaying_digest.percentile(q * 100) for q in quantiles]
    density = [0] * len(bins)

    # Assign value to a bin and adjust bins adaptively
    for i, bin_threshold in enumerate(bins):
        if value < bin_threshold:
            density[i] += 1
            if density[i] > bin_density_threshold:
                bins.insert(i + 1, (bins[i] + bins[i + 1]) / 2)  # Split bin
            return f"Quantile_Bin_{i}"

    # Merge sparse bins if density is too low
    sparse_bins = [i for i, d in enumerate(density) if d < bin_density_threshold / 2]
    for i in sparse_bins:
        if i + 1 < len(bins):
            bins[i] = (bins[i] + bins[i + 1]) / 2  # Merge bins
            bins.pop(i + 1)  # Remove the next bin
    return f"Quantile_Bin_{len(bins)}"


# --- EVENT PROCESSING ---
def process_event(event):
    """
    Process an event for adaptive discretization.
    """
    for feature in features_to_discretize:
        if feature in event:
            event[f'{feature}_quantile_bin'] = adaptive_quantile_binning(event[feature], feature)
    return event


# --- STREAMING FUNCTION ---
def stream_event_log(df, delay=1):
    """
    Stream and process events with adaptive discretization.
    """
    for _, event in df.iterrows():
        event = process_event(event)  # Process the event
        print("Processed event with adaptive quantile bins:\n", event)
        time.sleep(delay)  # Simulate streaming delay


# --- MAIN SCRIPT ---
# Load and prepare event log
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')
df_event_log['Timestamp'] = pd.to_datetime(df_event_log['Timestamp'])  # Convert timestamp to datetime
df_event_log = df_event_log.sort_values(by='Timestamp')  # Sort by timestamp

# Process each event in the stream
stream_event_log(df_event_log)
