from collections import defaultdict
import pandas as pd
from tdigest import TDigest
import numpy as np
from river.drift import ADWIN

time_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

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
        try:
            print(f"[DEBUG] Entering DecayingTDigest.update() with value: {value} (Type: {type(value)}), weight: {weight}")

            # Ensure value is numeric
            if not isinstance(value, (int, float)):
                print(f"[ERROR] Non-numeric value detected in DecayingTDigest.update(): {value} (Type: {type(value)})")
                raise ValueError(f"Invalid value type: {value}")

            if not isinstance(weight, (int, float)):
                print(f"[ERROR] Non-numeric weight detected: {weight} (Type: {type(weight)})")
                raise ValueError(f"Invalid weight type: {weight}")

            self.tdigest.update(value, weight)
            self.updates += 1
            print(f"[DEBUG] Successfully updated TDigest with value: {value}")

            if self.updates - self.last_decay >= self.decay_interval:
                self._apply_decay()

        except Exception as e:
            print(f"[ERROR] Exception in DecayingTDigest.update(): {e}")

    def _apply_decay(self):
        """Apply decay to all centroids in the T-Digest."""
        try:
            centroids = self.tdigest.centroids_to_list()  # ✅ Get centroids

            print(f"[DEBUG] Applying decay: Current centroids before decay: {centroids}")

            updated_centroids = []

            for entry in centroids:
                # ✅ Handle both tuple and dictionary formats
                if isinstance(entry, dict):
                    if 'm' in entry and 'c' in entry:
                        mean, weight = float(entry['m']), float(entry['c'])
                    else:
                        print(f"[ERROR] Unexpected centroid format: {entry}")
                        continue
                elif isinstance(entry, tuple):
                    mean, weight = float(entry[0]), float(entry[1])
                else:
                    print(f"[ERROR] Invalid centroid type: {type(entry)}, value: {entry}")
                    continue

                updated_centroids.append((mean, weight * self.decay_factor))

            # ✅ Rebuild the T-Digest with corrected centroids
            self.tdigest = TDigest()
            for mean, weight in updated_centroids:
                self.tdigest.update(mean, weight)

            print(f"[DEBUG] Applied decay: New centroid weights after {self.decay_factor} decay factor.")
            print(f"[DEBUG] Current centroids after decay: {self.tdigest.centroids_to_list()}")

        except Exception as e:
            print(f"[ERROR] Exception in _apply_decay(): {e}")

    def percentile(self, p):
        """Get the specified percentile from the digest."""
        return self.tdigest.percentile(p)


class EnhancedAdaptiveBinning:
    """
    Enhanced adaptive binning with hybrid approach combining quantile-based binning
    with density-based adjustments and temporal decay.
    """

    def __init__(self,
                 initial_bins,
                 bin_density_threshold,
                 drift_threshold,
                 decay_factor,
                 min_bin_width,
                 quantile_points):
        self.initial_bins = initial_bins
        self.bin_density_threshold = bin_density_threshold
        self.drift_detector = ADWIN(delta=drift_threshold)
        self.decaying_digest = DecayingTDigest(decay_factor=decay_factor)
        self.bins = np.linspace(0, 1, initial_bins + 1)
        self.bin_counts = defaultdict(int)
        self.min_bin_width = min_bin_width
        self.quantile_points = quantile_points
        self.window_size = 1000
        self.recent_values = []
        self.recent_splits = {}

    def update_bins(self, new_value):
        """
        Update bins with a new value and adjust for drift or density changes.
        Returns the bin assignment for the new value.
        """
        try:
            print(f"[DEBUG] Entering update_bins() with new_value: {new_value} (Type: {type(new_value)})")

            # Ensure value remains numeric before processing
            if not isinstance(new_value, (int, float, str)):
                print(f"[ERROR] Unexpected type detected in update_bins: {new_value} (Type: {type(new_value)})")
                return 0

            # Check if new_value turns into a string unexpectedly
            if isinstance(new_value, str):
                print(f"[DEBUG] String detected inside update_bins(): {new_value}")
                new_value = new_value.strip().lower()

                # Detect when 'm' appears
                if new_value == 'm':
                    print(f"[ERROR] 'm' string detected inside update_bins()! Investigating source...")

                if new_value in ['nan', 'null', '', 'm']:
                    print(f"[ERROR] Invalid numeric conversion for {new_value}, assigning to bin 0")
                    return 0

                try:
                    new_value = float(new_value)
                    print(f"[DEBUG] Converted feature value: {new_value}")
                except ValueError:
                    print(f"[ERROR] Cannot convert {new_value} to float. Returning bin 0.")
                    return 0

            print(f"[DEBUG] Final new_value before binning: {new_value} (Type: {type(new_value)})")

            # ✅ Check if new_value changes after calling decaying_digest.update()
            print(f"[DEBUG] Before decaying_digest.update(new_value): {new_value}")
            self.decaying_digest.update(new_value)
            print(f"[DEBUG] After decaying_digest.update(new_value): {new_value}")

            # ✅ Check if new_value changes after appending to recent_values
            print(f"[DEBUG] Before appending to recent_values: {new_value}")
            self.recent_values.append(new_value)
            print(f"[DEBUG] After appending to recent_values: {self.recent_values[-1]}")

            if len(self.recent_values) > self.window_size:
                self.recent_values.pop(0)

            # ✅ Check if new_value changes after calling drift_detector.update()
            print(f"[DEBUG] Before drift_detector.update(new_value): {new_value}")
            drift_detected = self.drift_detector.update(new_value)
            print(f"[DEBUG] After drift_detector.update(new_value): {new_value}, Drift detected: {drift_detected}")

            if drift_detected:
                print(f"[DRIFT DETECTED] Recalculating bins due to change in distribution!")
                self._recalculate_bins()

            # ✅ Check if new_value changes after calling _assign_bin()
            print(f"[DEBUG] Before _assign_bin(new_value): {new_value}")
            assigned_bin = self._assign_bin(new_value)
            print(f"[DEBUG] After _assign_bin(new_value): {assigned_bin}")

            assigned_bin = int(round(assigned_bin))  # Ensure integer bin assignment
            self.bin_counts[assigned_bin] += 1

            print(f"[DEBUG] Value: {new_value} -> Assigned bin: {assigned_bin}")
            return assigned_bin

        except Exception as e:
            print(f"[ERROR] Failed to update bins: {str(e)}")
            return 0  # Return default bin on error

    def _recalculate_bins(self):
        """
        Recalculate bins using both quantile and density information.
        """
        if not self.recent_values:
            return

        # Debugging: Print the distribution before recalculating
        print(f"[DEBUG] Recalculating bins! Current values: {self.recent_values}")

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

        # Debugging: Show updated bin structure
        print(f"[DEBUG] New bin boundaries: {self.bins}")

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
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= value < self.bins[i + 1]:
                bin_width = self.bins[i + 1] - self.bins[i]
                position = (value - self.bins[i]) / bin_width if bin_width > 0 else 0

                print(f"[DEBUG] Value {value} assigned to bin {i} (position {position:.2f})")
                return i + position  # Keep floating point bin positions

        # If value is exactly equal to the last bin edge, place it in the last bin
        return len(self.bins) - 2

    def _split_bin(self, bin_index):
        """
        Split a dense bin into two smaller bins while ensuring stability in binning.
        """

        if bin_index >= len(self.bins) - 1:
            return

        current_time = self.decaying_digest.updates  # Use the update counter as a time reference

        # Cooldown mechanism: Prevent multiple splits in quick succession
        if bin_index in self.recent_splits and (current_time - self.recent_splits[bin_index] < 500):
            print(f"[DEBUG] Skipping split for bin {bin_index} (recently split)")
            return

        lower = self.bins[bin_index]
        upper = self.bins[bin_index + 1]
        width = upper - lower

        # Prevent splitting very small bins
        if width < self.min_bin_width:
            print(f"[DEBUG] Skipping split for bin {bin_index}: width {width} is too small")
            return

        # Extract values that fall within this bin
        values_in_bin = [v for v in self.recent_values if lower <= v < upper]

        if not values_in_bin:
            return  # No data points available for splitting

        # Compute the median as a new boundary for splitting
        median_boundary = np.median(values_in_bin)

        # Ensure new boundary does not create a redundant split
        if median_boundary in self.bins:
            median_boundary += self.min_bin_width  # Ensure a valid new split

        # Prevent splitting very small bins
        if median_boundary - lower < self.min_bin_width or upper - median_boundary < self.min_bin_width:
            print(f"[DEBUG] Skipping split for bin {bin_index}: median {median_boundary} is too close to boundaries")
            return

        # Insert new bin boundary and sort
        new_boundaries = np.sort(np.append(self.bins, median_boundary ))
        self.bins = new_boundaries

        # Update bin densities: Redistribute count from old bin to the new bins
        old_count = self.bin_counts[bin_index]
        new_bin_index = np.where(self.bins == median_boundary )[0][0]

        self.bin_counts[bin_index] = old_count // 2  # Assign half to the first new bin
        self.bin_counts[new_bin_index] = old_count - self.bin_counts[
            bin_index]  # Assign remaining to the second new bin

        # Mark bin as recently split
        self.recent_splits[bin_index] = current_time

        print(f"[DEBUG] Split bin {bin_index}: New boundaries {self.bins}, Updated bin counts {dict(self.bin_counts)}")

    def _is_sparse_region(self, bin_index):
        """
        Check if a bin and its neighbors form a sparse region.
        """
        count = self.bin_counts[bin_index]
        neighbor_counts = [
            self.bin_counts.get(bin_index - 1, float('inf')),
            self.bin_counts.get(bin_index + 1, float('inf'))
        ]

        # Merge only if the bin AND BOTH neighbors are sparse
        return count < self.bin_density_threshold / 3 and all(
            nc < self.bin_density_threshold / 3 for nc in neighbor_counts
        )

    def _merge_sparse_bins(self):
        """
        Merge adjacent sparse bins to reduce fragmentation while updating bin densities.
        """
        i = 0
        while i < len(self.bins) - 2:
            if (self._is_sparse_region(i) and self.bins[i + 1] - self.bins[i] >= self.min_bin_width):
                # Merge current bin with next bin
                print(f"[DEBUG] Merging bins {i} and {i + 1} due to sparse region")

                # Sum bin counts
                merged_count = self.bin_counts[i] + self.bin_counts[i + 1]

                # Remove the boundary
                self.bins = np.concatenate([self.bins[:i + 1], self.bins[i + 2:]])

                # Update bin counts
                self.bin_counts[i] = merged_count
                del self.bin_counts[i + 1]  # Remove the second bin

            else:
                i += 1  # Move to the next bin


def update_time_distribution(activity, time_features):
    """
    Update the time distribution incrementally, tracking all relevant time-based features.
    This function assumes time features have already been extracted.
    """
    hour_bin = time_features["hour_bin"]
    day_of_week = time_features["day_of_week"]
    is_weekend = time_features["is_weekend"]
    week_of_month = time_features["week_of_month"]
    season = time_features["season"]
    month = time_features["month"]

    # Incrementally update counts for all time-based features
    time_distribution[activity][hour_bin][day_of_week][month] += 1
    time_distribution[activity]["is_weekend"][is_weekend] += 1
    time_distribution[activity]["week_of_month"][week_of_month] += 1
    time_distribution[activity]["season"][season] += 1



def stream_event_log(event_dict, timestamp_column, control_flow_column, resource_column, case_id_column, event_id_column, data_columns, features_to_discretize, binning_models):
    """
    Streaming event log processor with binning for numerical features only.
    """
    try:
        print(f"[DEBUG] Processing event {event_dict.get(event_id_column, 'Unknown')}")
        processed_event = event_dict.copy()  # Create a copy to avoid modifying original

        # Process numerical features for binning
        for feature in features_to_discretize:
            print(f"[DEBUG] Processing feature {feature} with value: {processed_event.get(feature)}")
            if feature not in processed_event:
                print(f"[DEBUG] Feature {feature} not found in event dictionary")
                continue

            if feature in [case_id_column, control_flow_column]:
                print(f"[DEBUG] Skipping non-numeric column {feature}")
                continue

            try:
                # Get value and handle potential None or empty values
                value = processed_event.get(feature)
                if value is None or value == '':
                    print(f"[DEBUG] Empty or None value for feature {feature}")
                    processed_event[f"{feature}_bin"] = 0  # Default bin
                    continue

                # Ensure numeric conversion
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"[ERROR] Cannot convert value to float: {value}")
                        processed_event[f"{feature}_bin"] = 0  # Assign default bin
                        continue

                for key, val in event_dict.items():
                    print(f"[DEBUG] Event Feature: {key} = {val} (Type: {type(val)})")

                if isinstance(value, str) and 'm' in value:
                    print(f"[ERROR] 'm' detected in {feature} BEFORE calling update_bins(): {value}")

                if feature == timestamp_column and isinstance(value, pd.Timestamp):
                    bin_assignment = int(binning_models[feature].update_bins(value))
                else:
                    bin_assignment = int(binning_models[feature].update_bins(value))

                processed_event[f"{feature}_bin"] = bin_assignment
                print(f"[DEBUG] Feature {feature}: Value {value} → Bin {bin_assignment}")

            except (ValueError, TypeError) as e:
                print(f"[ERROR] Issue processing feature {feature}: {str(e)}")
                processed_event[f"{feature}_bin"] = 0  # Default bin for errors
                continue

        return processed_event

    except Exception as e:
        print(f"[ERROR] Failed to process event: {str(e)}")
        return event_dict  # Return original if processing fails
