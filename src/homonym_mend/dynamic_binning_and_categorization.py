from collections import defaultdict, deque
from tdigest import TDigest
import numpy as np
from river.drift import ADWIN
import time


class DecayingTDigest:
    """
    T-Digest implementation with temporal decay for maintaining streaming quantiles.
    """

    def __init__(self, decay_factor=0.95):
        self.tdigest = TDigest()
        self.decay_factor = decay_factor
        self.last_decay = 0
        self.decay_interval = 1000  # Number of updates before applying decay
        self.updates = 0

    def update(self, value, weight=1.0):
        """Update the T-Digest with a new value and optional weight."""
        self.tdigest.update(value, weight)
        self.updates += 1

        if self.updates - self.last_decay >= self.decay_interval:
            self._apply_decay()

    def _apply_decay(self):
        """Apply decay to all centroids in the T-Digest."""
        for c in self.tdigest.C:
            c.w *= self.decay_factor
        self.last_decay = self.updates

    def percentile(self, p):
        """Get the specified percentile from the digest."""
        return self.tdigest.percentile(p)


class EnhancedAdaptiveBinning:
    """
    Enhanced adaptive binning with hybrid approach combining quantile-based binning
    with density-based adjustments and temporal decay.
    """

    def __init__(self,
                 initial_bins=10,
                 bin_density_threshold=50,
                 drift_threshold=0.05,
                 decay_factor=0.95,
                 min_bin_width=0.01,
                 quantile_points=None):
        self.initial_bins = initial_bins
        self.bin_density_threshold = bin_density_threshold
        self.drift_detector = ADWIN(delta=drift_threshold)
        self.decaying_digest = DecayingTDigest(decay_factor=decay_factor)
        self.bins = np.linspace(0, 1, initial_bins + 1)
        self.bin_counts = defaultdict(int)
        self.min_bin_width = min_bin_width
        self.quantile_points = quantile_points or [0.25, 0.5, 0.75]
        self.window_size = 1000
        self.recent_values = []

    def update_bins(self, new_value):
        """
        Update bins with a new value and adjust for drift or density changes.
        Returns the bin assignment for the new value.
        """
        # Update decay-aware data structures
        self.decaying_digest.update(new_value)
        self.recent_values.append(new_value)
        if len(self.recent_values) > self.window_size:
            self.recent_values.pop(0)

        # Check for drift
        if self.drift_detector.update(new_value):
            self._recalculate_bins()

        # Assign to bin and update density
        assigned_bin = self._assign_bin(new_value)
        self.bin_counts[assigned_bin] += 1

        # Check density-based conditions
        if self.bin_counts[assigned_bin] > self.bin_density_threshold:
            self._split_bin(assigned_bin)
        elif self._is_sparse_region(assigned_bin):
            self._merge_sparse_bins()

        return assigned_bin

    def _recalculate_bins(self):
        """
        Recalculate bins using both quantile and density information.
        """
        if not self.recent_values:
            return

        # Get quantile-based boundaries
        quantile_boundaries = [
            self.decaying_digest.percentile(q * 100)
            for q in self.quantile_points
        ]

        # Add density-based boundaries
        density_boundaries = self._compute_density_boundaries()

        # Combine and sort all boundaries
        all_boundaries = sorted(set(quantile_boundaries + density_boundaries))

        # Update bins while respecting minimum width
        self.bins = self._ensure_minimum_width(all_boundaries)
        self.bin_counts = defaultdict(int)

    def _compute_density_boundaries(self):
        """
        Compute additional bin boundaries based on data density.
        """
        if not self.recent_values:
            return []

        # Use kernel density estimation for finding density peaks
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.recent_values)
        x = np.linspace(min(self.recent_values), max(self.recent_values), 100)
        density = kde(x)

        # Find peaks in density
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density)
        return list(x[peaks])

    def _ensure_minimum_width(self, boundaries):
        """
        Ensure all bin widths meet the minimum width requirement.
        """
        result = [boundaries[0]]
        for b in boundaries[1:]:
            if b - result[-1] >= self.min_bin_width:
                result.append(b)
        return result

    def _assign_bin(self, value):
        """
        Assign a value to the appropriate bin with interpolation.
        """
        for i, bin_threshold in enumerate(self.bins[:-1]):
            if value < self.bins[i + 1]:
                # Use linear interpolation within the bin
                bin_width = self.bins[i + 1] - bin_threshold
                position = (value - bin_threshold) / bin_width
                return i + position
        return len(self.bins) - 2

    def _split_bin(self, bin_index):
        """
        Split a dense bin into smaller bins.
        """
        if bin_index >= len(self.bins) - 1:
            return

        lower = self.bins[bin_index]
        upper = self.bins[bin_index + 1]
        width = upper - lower

        if width < self.min_bin_width:
            return

        # Create new boundaries using local density
        values_in_bin = [v for v in self.recent_values
                         if lower <= v < upper]

        if not values_in_bin:
            return

        # Use quantiles within the bin for splitting
        q_values = np.quantile(values_in_bin, [0.5])
        new_boundaries = sorted([lower] + list(q_values) + [upper])

        # Update bins array
        self.bins = np.concatenate([
            self.bins[:bin_index],
            new_boundaries,
            self.bins[bin_index + 2:]
        ])

        # Reset counts for affected bins
        self.bin_counts = defaultdict(int)

    def _is_sparse_region(self, bin_index):
        """
        Check if a bin and its neighbors form a sparse region.
        """
        count = self.bin_counts[bin_index]
        neighbor_counts = [
            self.bin_counts[bin_index - 1] if bin_index > 0 else float('inf'),
            self.bin_counts[bin_index + 1] if bin_index < len(self.bins) - 2 else float('inf')
        ]
        return count < self.bin_density_threshold / 3 and any(
            nc < self.bin_density_threshold / 3 for nc in neighbor_counts
        )

    def _merge_sparse_bins(self):
        """
        Merge adjacent sparse bins to reduce fragmentation.
        """
        i = 0
        while i < len(self.bins) - 2:
            if (self._is_sparse_region(i) and
                    self.bins[i + 1] - self.bins[i] >= self.min_bin_width):
                # Merge current bin with next bin
                self.bins = np.concatenate([
                    self.bins[:i + 1],
                    self.bins[i + 2:]
                ])
                # Update counts
                self.bin_counts[i] += self.bin_counts[i + 1]
                del self.bin_counts[i + 1]
            else:
                i += 1


def extract_temporal_features(timestamp):
    """
    Extract temporal features with more granular time periods and seasonal information.
    """
    hour = timestamp.hour
    hour_bin = (
        "Early_Morning" if 4 <= hour < 8 else
        "Morning" if 8 <= hour < 12 else
        "Afternoon" if 12 <= hour < 16 else
        "Late_Afternoon" if 16 <= hour < 20 else
        "Night"
    )

    month = timestamp.month
    season = (
        "Winter" if month in [12, 1, 2] else
        "Spring" if month in [3, 4, 5] else
        "Summer" if month in [6, 7, 8] else
        "Fall"
    )

    return {
        'hour_bin': hour_bin,
        'day_period': hour_bin,
        'day_of_week': timestamp.weekday(),
        'is_weekend': timestamp.weekday() >= 5,
        'week_of_month': (timestamp.day - 1) // 7 + 1,
        'season': season,
        'month': month
    }


def stream_event_log(
        event_dict,
        timestamp_column,
        control_flow_column,
        resource_column,
        case_id_column,
        event_id_column,
        data_columns,
        features_to_discretize,
        sliding_window_size,
        bin_density_threshold,
        quantiles=None,
        decay_factor=0.95
):
    """
    Enhanced streaming event log processor with improved binning and feature creation.
    """
    # Initialize adaptive binning models with enhanced configuration
    adaptive_bin_models = defaultdict(
        lambda: EnhancedAdaptiveBinning(
            initial_bins=len(quantiles) if quantiles else 10,
            bin_density_threshold=bin_density_threshold,
            decay_factor=decay_factor,
            quantile_points=quantiles or [0.25, 0.5, 0.75]
        )
    )

    # Process each feature that needs discretization
    for feature in features_to_discretize:
        if feature not in event_dict:
            continue

        if feature == timestamp_column:
            # Extract and bin temporal features
            temporal_features = extract_temporal_features(event_dict[feature])
            event_dict.update({
                f"{feature}_{temp_feature}_bin": temp_value
                for temp_feature, temp_value in temporal_features.items()
            })
        else:
            # Apply enhanced adaptive binning
            try:
                value = float(event_dict[feature])  # Ensure numeric
                adaptive_model = adaptive_bin_models[feature]
                bin_assignment = adaptive_model.update_bins(value)
                event_dict[f"{feature}_bin"] = bin_assignment
            except (ValueError, TypeError) as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue

    # Debug print
    print(f"Processed Event ID: {event_dict.get(event_id_column)}")
    return event_dict