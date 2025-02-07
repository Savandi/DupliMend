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

        # Fix missing attributes
        self.decay_after_events = decay_after_events  # Event-driven decay
        self.forgetting_threshold = frequency_decay_threshold  # Adaptive removal threshold

        self.micro_clusters = {}  # Changed from list to dictionary
        self.event_count = 0
        self.similarity_history = []
        self.max_history_size = 15

    def forget_old_clusters(self, global_event_counter):
        """
        Remove old micro-clusters from this DBStream instance using frequency decay.
        Ensures that _apply_decay() is run before final removal to prevent unnecessary drops.
        """
        # First, apply vector-level decay before removing clusters
        self._apply_decay(global_event_counter)

        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            events_since_last_update = global_event_counter - cluster.get("last_seen_event", global_event_counter)

            # Apply exponential frequency decay
            cluster["weight"] *= np.exp(-events_since_last_update / self.decay_after_events)

            # Forget cluster if weight (frequency) is too low and hasn't been updated recently
            if cluster[
                "weight"] < self.forgetting_threshold and events_since_last_update > self.removal_threshold_events:
                del self.micro_clusters[cluster_id]  # Only remove cluster after vector decay has been applied

            if len(self.micro_clusters) <= self.lossy_counting_budget:
                break  # Stop once within budget

    def partial_fit(self, vector, global_event_counter):
        """
        Process new vector, detect clusters, and update micro-clusters dynamically.
        """
        self.forget_old_clusters(global_event_counter)
        self.event_count += 1

        if not self.micro_clusters:
            self.micro_clusters[0] = {"centroid": np.array(vector), "weight": 1, "last_seen_event": global_event_counter}
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
            self._update_cluster(best_cluster_id, vector)
            return best_cluster_id
        elif best_sim < self.split_threshold:
            new_cluster_id = len(self.micro_clusters)
            self.micro_clusters[new_cluster_id] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_event": global_event_counter  # Ensure last_seen_event is stored
            }
            return new_cluster_id
        else:
            return best_cluster_id

    def _update_cluster(self, cluster_id, new_vector,global_event_counter):
        """
        Update an existing micro-cluster with a new data point.
        """
        cluster = self.micro_clusters[cluster_id]
        cluster["weight"] += 1
        cluster["centroid"] = (cluster["centroid"] * (cluster["weight"] - 1) + new_vector) / cluster["weight"]
        cluster["last_seen_event"] = global_event_counter

    def _apply_decay(self, global_event_counter):
        """
        Apply smooth temporal decay to vector frequencies and dynamically adjust thresholds.
        Ensures that decayed clusters fade gradually instead of sudden removals.
        """
        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            time_since_update = global_event_counter - cluster["last_seen_event"]

            # Apply adaptive decay instead of sudden drop
            decay_factor = np.exp(-time_since_update / self.decay_after_events)

            # Decay all stored vector frequencies in the cluster
            for vector in list(cluster["vector_frequencies"].keys()):
                cluster["vector_frequencies"][vector] *= decay_factor  # Gradual decay

                # Remove vector if frequency drops below threshold
                if cluster["vector_frequencies"][vector] < self.forgetting_threshold:
                    del cluster["vector_frequencies"][vector]

            # If all feature vectors in the cluster decay, remove the cluster
            if not cluster["vector_frequencies"]:
                del self.micro_clusters[cluster_id]  # Corrected way to remove cluster
            else:
                cluster["centroid"] = self._most_frequent_vector(
                    cluster["vector_frequencies"])  # Ensure centroid updates

    def _most_frequent_vector(self, vector_frequencies):
        """
        Determines the most frequent feature vector in a cluster.
        """
        return max(vector_frequencies, key=vector_frequencies.get)

    def get_micro_clusters(self):
        """
        Retrieve a list of all current micro-clusters.
        """
        return list(self.micro_clusters.values())  # Ensure method returns micro-clusters correctly