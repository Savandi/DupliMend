from collections import defaultdict
import numpy as np
from config.config import dbstream_params, decay_after_events, lossy_counting_budget, \
    removal_threshold_events, frequency_decay_threshold
from src.utils.similarity_utils import compute_contextual_weighted_similarity

class DBStream:
    """
    DBStream: Custom version for clustering categorical/discretized feature vectors.
    Tracks the most frequent feature vector as the cluster centroid.
    """

    def __init__(self):
        """
        Initialize DBStream with parameters from config.py.
        """

        self.clustering_threshold = dbstream_params.get("clustering_threshold", 0.35)
        self.fading_factor = dbstream_params.get("fading_factor", 0.05)
        self.cleanup_interval = dbstream_params.get("cleanup_interval", 2)
        self.split_threshold = dbstream_params.get("split_threshold", 0.4)
        self.merge_threshold = dbstream_params.get("merge_threshold", 0.8)
        self.eps = dbstream_params.get("eps", 0.02)
        self.beta = dbstream_params.get("beta", 0.15)
        self.lambda_ = dbstream_params.get("lambda", 0.0001)
        self.lossy_counting_budget = lossy_counting_budget
        self.removal_threshold_events = removal_threshold_events
        self.activity_event_counters = defaultdict(int)  # Tracks event count per activity label

        # Fix missing attributes
        self.decay_after_events = decay_after_events  # Event-driven decay
        self.forgetting_threshold = frequency_decay_threshold  # Adaptive removal threshold

        self.micro_clusters = {}  # Changed from list to dictionary
        self.event_count = 0
        self.similarity_history = []
        self.max_history_size = 15

    def forget_old_clusters(self, activity_label):
        """
        Remove old micro-clusters using frequency decay, but only based on
        the event count for this specific DBStream instance's activity label.
        """
        current_activity_event_count = self.activity_event_counters[activity_label]  # Only this instance's event count
        self._apply_decay(current_activity_event_count)

        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            last_seen_activity_event = cluster.get("last_seen_activity_event", current_activity_event_count)

            events_since_last_seen = current_activity_event_count - last_seen_activity_event  # Uses only relevant events

            # Apply exponential frequency decay
            cluster["weight"] *= np.exp(-events_since_last_seen / self.decay_after_events)

            # Forget cluster if weight (frequency) is too low and hasn't been updated in `removal_threshold_events`
            if cluster["weight"] < self.forgetting_threshold and events_since_last_seen > self.removal_threshold_events:
                del self.micro_clusters[cluster_id]

            # Stop removing clusters once within the lossy counting budget
            if len(self.micro_clusters) <= self.lossy_counting_budget:
                break  #  Prevent unnecessary deletions

    def partial_fit(self, vector, activity_label):
        """
        Process a new vector, detect clusters, and update micro-clusters dynamically.
        Uses activity-specific event counts to determine cluster removal.
        """
        # Track event count only for this specific activity label
        self.activity_event_counters[activity_label] += 1

        # Forget old clusters based on activity-specific event inactivity
        self.forget_old_clusters(activity_label)

        self.event_count += 1  # Global counter (not used for forgetting anymore)

        if not self.micro_clusters:
            self.micro_clusters[0] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_activity_event": self.activity_event_counters[activity_label]  # Track activity-based timestamp
            }
            return 0

        similarities = []
        for cluster_id, cluster in self.micro_clusters.items():
            sim = compute_contextual_weighted_similarity(
                cluster["centroid"],
                vector,
                [1] * len(vector),
                [1] * len(vector),
                alpha=self.beta
            )
            similarities.append((sim, cluster_id))

        similarities.sort(reverse=True, key=lambda x: x[0])

        best_sim, best_cluster_id = similarities[0]

        if best_sim > self.merge_threshold:
            self._update_cluster(best_cluster_id, vector, activity_label)  # Ensure correct updates
            return best_cluster_id
        elif best_sim < self.split_threshold:
            new_cluster_id = len(self.micro_clusters)
            self.micro_clusters[new_cluster_id] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_activity_event": self.activity_event_counters[activity_label]  # Store last seen event per activity
            }
            return new_cluster_id
        else:
            return best_cluster_id

    def _update_cluster(self, cluster_id, new_vector, activity_label):
        """
        Update an existing cluster with a new feature vector.
        """
        cluster = self.micro_clusters[cluster_id]
        cluster["weight"] += 1
        cluster["centroid"] = (cluster["centroid"] * (cluster["weight"] - 1) + new_vector) / cluster["weight"]
        cluster["last_seen_activity_event"] = self.activity_event_counters[activity_label]  #  Uses instance-specific counter

    def _apply_decay(self, current_activity_event_count):
        """
        Apply smooth temporal decay to vector frequencies and dynamically adjust thresholds.
        Ensures that decayed clusters fade gradually instead of sudden removals.
        """
        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            time_since_update = current_activity_event_count - cluster["last_seen_activity_event"]

            # Apply adaptive decay instead of sudden drop
            decay_factor = np.exp(-time_since_update / self.decay_after_events)

            if "vector_frequencies" not in cluster:
                cluster["vector_frequencies"] = {}  # Ensure this key exists

            # Decay all stored vector frequencies in the cluster
            for vector in list(cluster["vector_frequencies"].keys()):
                cluster["vector_frequencies"][vector] *= decay_factor  # Gradual decay

                # Remove vector if frequency drops below threshold
                if cluster["vector_frequencies"][vector] < self.forgetting_threshold:
                    del cluster["vector_frequencies"][vector]

            # If all feature vectors in the cluster decay, remove the cluster
            if not cluster["vector_frequencies"]:
                del self.micro_clusters[cluster_id]  # Only remove empty clusters
            else:
                cluster["centroid"] = self._most_frequent_vector(cluster["vector_frequencies"])

    def _most_frequent_vector(self, vector_frequencies):
        if not vector_frequencies:
            return np.zeros_like(self.micro_clusters[0]["centroid"])  # Default zero vector
        return max(vector_frequencies, key=vector_frequencies.get)

    def get_micro_clusters(self):
        """
        Retrieve a list of all current micro-clusters.
        """
        return list(self.micro_clusters.values())  # Ensure method returns micro-clusters correctly
