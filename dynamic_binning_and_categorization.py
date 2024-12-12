from collections import defaultdict, deque
from river.cluster import DBSTREAM
from tdigest import TDigest
import time

def stream_event_log(
    df, timestamp_column, control_flow_column, resource_column, case_id_column,
    data_columns, features_to_discretize, quantiles, sliding_window_size,
    bin_density_threshold, dbstream_params, delay=1
):
    sliding_window = defaultdict(lambda: deque(maxlen=sliding_window_size))
    feature_tdigest = defaultdict(TDigest)
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
        return {'hour': timestamp.hour, 'day_of_week': timestamp.weekday()}

    def hybrid_binning(value, feature):
        feature_tdigest[feature].update(value)
        bins = [feature_tdigest[feature].percentile(q * 100) for q in quantiles]
        for i, bin_threshold in enumerate(bins):
            if value < bin_threshold:
                sliding_window[feature].append(value)
                if len(sliding_window[feature]) > bin_density_threshold:
                    bins.insert(i + 1, (bins[i] + bins[i + 1]) / 2)
                return f"{feature}_Quantile_Bin_{i}"
        return f"{feature}_Quantile_Bin_{len(bins)}"

    for _, event in df.iterrows():
        event_dict = event.to_dict()
        for feature in features_to_discretize:
            if feature in event_dict:
                if feature == timestamp_column:
                    for temp_feature, temp_value in extract_temporal_features(event_dict[feature]).items():
                        event_dict[f"{feature}_{temp_feature}_bin"] = temp_value
                else:
                    event_dict[f"{feature}_bin"] = hybrid_binning(event_dict[feature], feature)
        yield event_dict
        time.sleep(delay)
