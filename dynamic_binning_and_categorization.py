from collections import defaultdict, deque
from river.cluster import DBSTREAM
from tdigest import TDigest
import numpy as np
from river.drift import ADWIN
import time


class AdaptiveBinning:
    def __init__(self, initial_bins=10, drift_threshold=0.05):
        self.drift_detector = ADWIN(delta=drift_threshold)
        self.bins = np.linspace(0, 1, initial_bins + 1)  # Uniform bins by default
        self.data_distribution = TDigest()  # Track multimodal distributions

    def update_bins(self, new_value):
        self.data_distribution.update(new_value)
        if self.drift_detector.update(new_value):  # Detect drift in data distribution
            self.recalculate_bins()

    def recalculate_bins(self):
        self.bins = [self.data_distribution.percentile(q * 100) for q in np.linspace(0, 1, len(self.bins))]

    def assign_bin(self, value):
        for i, bin_threshold in enumerate(self.bins[:-1]):
            if value < self.bins[i + 1]:
                return f"Bin_{i}"
        return f"Bin_{len(self.bins) - 1}"


def stream_event_log(
    df, timestamp_column, control_flow_column, resource_column, case_id_column,
    data_columns, features_to_discretize, quantiles, sliding_window_size,
    bin_density_threshold, dbstream_params, delay=1
):
    sliding_window = defaultdict(lambda: deque(maxlen=sliding_window_size))
    adaptive_bin_models = defaultdict(lambda: AdaptiveBinning(initial_bins=10, drift_threshold=0.05))
    streaming_dbstream_models = defaultdict(
        lambda: DBSTREAM(
            clustering_threshold=dbstream_params["clustering_threshold"],
            fading_factor=dbstream_params["fading_factor"],
            cleanup_interval=dbstream_params["cleanup_interval"],
            intersection_factor=dbstream_params["intersection_factor"],
            minimum_weight=dbstream_params["minimum_weight"]
        )
    )

    def extract_temporal_features(timestamp):
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'season': "Winter" if timestamp.month in [12, 1, 2] else "Other"
        }

    def adaptive_binning(value, feature):
        model = adaptive_bin_models[feature]
        model.update_bins(value)
        return model.assign_bin(value)

    for _, event in df.iterrows():
        event_dict = event.to_dict()
        for feature in features_to_discretize:
            if feature in event_dict:
                if feature == timestamp_column:
                    for temp_feature, temp_value in extract_temporal_features(event_dict[feature]).items():
                        event_dict[f"{feature}_{temp_feature}_bin"] = temp_value
                else:
                    event_dict[f"{feature}_bin"] = adaptive_binning(event_dict[feature], feature)
        yield event_dict
        time.sleep(delay)
