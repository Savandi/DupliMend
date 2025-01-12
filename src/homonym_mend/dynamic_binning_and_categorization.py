from collections import defaultdict, deque
from tdigest import TDigest
import numpy as np
from river.drift import ADWIN
import time

class AdaptiveBinning:
    """
    Adaptive binning approach combining quantile-based binning with density-based adjustments.
    """
    def __init__(self, initial_bins=10, bin_density_threshold=50, drift_threshold=0.05):
        self.initial_bins = initial_bins
        self.bin_density_threshold = bin_density_threshold
        self.drift_detector = ADWIN(delta=drift_threshold)
        self.data_distribution = TDigest()  # Tracks data distribution
        self.bins = np.linspace(0, 1, initial_bins + 1)  # Initial uniform bins
        self.bin_counts = defaultdict(int)  # Count of items in each bin

    def update_bins(self, new_value):
        """
        Update bins with a new value and adjust for drift or density changes.
        """
        self.data_distribution.update(new_value)
        if self.drift_detector.update(new_value):
            self.recalculate_bins()

        # Assign the value to a bin and update bin density
        assigned_bin = self.assign_bin(new_value, update_count=True)
        if self.bin_counts[assigned_bin] > self.bin_density_threshold:
            self.split_bin(assigned_bin)

    def recalculate_bins(self):
        """
        Recalculate bins based on data distribution quantiles.
        """
        self.bins = [self.data_distribution.percentile(q * 100) for q in np.linspace(0, 1, self.initial_bins + 1)]
        self.bin_counts = defaultdict(int)  # Reset bin counts

    def assign_bin(self, value, update_count=False):
        """
        Assign a value to the appropriate bin.
        """
        for i, bin_threshold in enumerate(self.bins[:-1]):
            if value < self.bins[i + 1]:
                if update_count:
                    self.bin_counts[i] += 1
                return i
        # Assign to the last bin
        if update_count:
            self.bin_counts[len(self.bins) - 2] += 1
        return len(self.bins) - 2

    def split_bin(self, bin_index):
        """
        Split a bin into two finer bins if it exceeds the density threshold.
        """
        if bin_index >= len(self.bins) - 1:
            return  # Can't split the last bin

        lower_bound = self.bins[bin_index]
        upper_bound = self.bins[bin_index + 1]
        mid_point = (lower_bound + upper_bound) / 2

        # Stop splitting if the bin width is too small
        min_bin_width = 0.01  # Define a minimum width threshold
        if (upper_bound - lower_bound) < min_bin_width:
            print(f"Stopping split: Bin width too small [{lower_bound}, {upper_bound}]")
            return

        print(f"Splitting bin: [{lower_bound}, {upper_bound}] into [{lower_bound}, {mid_point}, {upper_bound}]")

        # Insert the new midpoint into the bins array
        self.bins = np.insert(self.bins, bin_index + 1, mid_point)

        # Recalculate bin counts (optional but keeps accurate tracking)
        self.bin_counts = defaultdict(int)

    def merge_sparse_bins(self):
        """
        Merge sparse bins to reduce fragmentation.
        """
        merged_bins = [self.bins[0]]  # Always keep the first bin
        i = 0
        while i < len(self.bins) - 1:
            lower_bound = self.bins[i]
            upper_bound = self.bins[i + 1]
            count = self.bin_counts[i]

            # Merge condition: current and next bin counts are below threshold
            if count < self.bin_density_threshold and i + 1 < len(self.bins) - 1:
                upper_bound = self.bins[i + 2]
                i += 1  # Skip the next bin as it's merged

            merged_bins.append(upper_bound)
            i += 1

        self.bins = merged_bins
        self.bin_counts = defaultdict(int)  # Reset bin counts


def extract_temporal_features(timestamp):
    """
    Extract temporal features from a timestamp, such as hour of the day, day of the week, and season.
    """
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'season': (
            "Winter" if timestamp.month in [12, 1, 2]
            else "Spring" if timestamp.month in [3, 4, 5]
            else "Summer" if timestamp.month in [6, 7, 8]
            else "Fall"
        )
    }


def stream_event_log(
    event_dict, timestamp_column, control_flow_column, resource_column, case_id_column,
    event_id_column, data_columns, features_to_discretize, sliding_window_size,
    bin_density_threshold, quantiles=None
):
    sliding_window = defaultdict(lambda: deque(maxlen=sliding_window_size))
    adaptive_bin_models = defaultdict(
        lambda: AdaptiveBinning(
            bin_density_threshold=bin_density_threshold,
            initial_bins=len(quantiles) if quantiles else 10
        )
    )
    # Apply binning for selected features
    for feature in features_to_discretize:
        if feature in event_dict:
            if feature == timestamp_column:
                # Extract and bin temporal features
                for temp_feature, temp_value in extract_temporal_features(event_dict[feature]).items():
                    event_dict[f"{feature}_{temp_feature}_bin"] = temp_value
            else:
                adaptive_model = adaptive_bin_models[feature]
                adaptive_model.update_bins(event_dict[feature])
                event_dict[f"{feature}_bin"] = adaptive_model.assign_bin(event_dict[feature])

    # Debug print to check event_id
    print(f"Yielding Event ID: {event_dict.get(event_id_column)}")

    return event_dict

