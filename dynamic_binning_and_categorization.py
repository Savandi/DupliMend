from collections import defaultdict, deque
from river.cluster import DBSTREAM
from tdigest import TDigest
import time

# --- PARAMETERS ---
quantiles = [0.25, 0.5, 0.75]  # Define quantile thresholds
sliding_window_size = 100  # Sliding window for recency-sensitive adjustments
bin_density_threshold = 10  # Density threshold for hybrid binning
features_to_discretize = ['age', 'heart_rate', 'timestamp']  # Features for dynamic binning

# --- GLOBAL VARIABLES ---
sliding_window = defaultdict(lambda: deque(maxlen=sliding_window_size))  # Sliding windows for all features
feature_tdigest = defaultdict(TDigest)  # T-digest structures for numerical features
streaming_dbstream_models = {}  # DBSTREAM models for clustering


# --- HELPER FUNCTIONS ---
def extract_temporal_features(timestamp):
    """
    Dynamically extract temporal features (hour, day of the week, dynamic season) from a timestamp.
    """
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month

    # Dynamic categorization for seasons using clustering
    if 'season' not in streaming_dbstream_models:
        streaming_dbstream_models['season'] = DBSTREAM(epsilon=0.5, mu=5)

    season_model = streaming_dbstream_models['season']
    season_cluster = season_model.predict_one([[month]]) or 0  # Predict cluster for the current month
    season_label = f"Season_Cluster_{season_cluster}"

    return {'hour': hour, 'day_of_week': day_of_week, 'season': season_label}


def hybrid_binning(value, feature):
    """
    Apply hybrid binning (quantile-based + density-sensitive) for numerical features.
    """
    # Update T-digest for the feature
    feature_tdigest[feature].update(value)
    bins = [feature_tdigest[feature].percentile(q * 100) for q in quantiles]

    # Assign value to a bin based on density
    for i, bin_threshold in enumerate(bins):
        if value < bin_threshold:
            sliding_window[feature].append(value)

            # Density-sensitive adjustments
            if len(sliding_window[feature]) > bin_density_threshold:
                bins.insert(i + 1, (bins[i] + bins[i + 1]) / 2)  # Split dense bins
            return f"{feature}_Quantile_Bin_{i}"

    return f"{feature}_Quantile_Bin_{len(bins)}"


def dynamic_clustering(value, feature):
    """
    Perform incremental clustering for categorical or fine-grained numerical features using DBSTREAM.
    """
    if feature not in streaming_dbstream_models:
        streaming_dbstream_models[feature] = DBSTREAM(epsilon=0.5, mu=5)

    # Update the DBSTREAM model
    dbstream = streaming_dbstream_models[feature]
    value_reshaped = [[value]] if isinstance(value, (int, float)) else [[hash(value)]]
    dbstream.learn_one(value_reshaped[0])

    # Assign the value to a cluster
    cluster_label = dbstream.predict_one(value_reshaped[0])
    return f"{feature}_Cluster_{cluster_label}"


def process_event(event):
    """
    Process each event and dynamically bin features.
    """
    for feature in features_to_discretize:
        if feature in event:
            if feature == 'timestamp':
                # Extract temporal features and create bins dynamically
                temporal_features = extract_temporal_features(event[feature])
                for temp_feature, temp_value in temporal_features.items():
                    event[f'{feature}_{temp_feature}_bin'] = dynamic_clustering(temp_value, temp_feature)
            else:
                # Apply hybrid binning for numerical features
                event[f'{feature}_bin'] = hybrid_binning(event[feature], feature)
    return event


def stream_event_log(df, delay=1):
    """
    Streaming function to process events.
    """
    for _, event in df.iterrows():
        event = process_event(event)  # Process each event
        yield event
        time.sleep(delay)  # Simulate streaming delay
