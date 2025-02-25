from collections import defaultdict
import numpy as np
import logging
from config.config import dbstream_params, decay_after_events, lossy_counting_budget, \
    removal_threshold_events, frequency_decay_threshold
from src.utils.logging_utils import log_traceability
from src.utils.similarity_utils import compute_contextual_weighted_similarity

# Configure logging
logging.basicConfig(filename="dbstream_debug.log", level=logging.DEBUG, format="%(asctime)s - %(message)s")

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
        self.merge_threshold = dbstream_params.get("merge_threshold", 0.7)  # Increase from 0.6
        self.split_threshold = dbstream_params.get("split_threshold", 0.1)
        self.eps = dbstream_params.get("eps", 0.02)
        self.beta = dbstream_params.get("beta", 0.15)
        self.lambda_ = dbstream_params.get("lambda", 0.001)
        self.lossy_counting_budget = lossy_counting_budget
        self.removal_threshold_events = removal_threshold_events
        self.activity_event_counters = defaultdict(int)  # Tracks event count per activity label

        self.decay_after_events = decay_after_events  # Event-driven decay
        self.forgetting_threshold = frequency_decay_threshold  # Adaptive removal threshold

        self.micro_clusters = {}  # Changed from list to dictionary
        self.event_count = 0
        self.similarity_history = []
        self.max_history_size = 15

        logging.info("DBStream initialized with merge_threshold: %.2f, split_threshold: %.2f",
                     self.merge_threshold, self.split_threshold)

    def forget_old_clusters(self, activity_label):
        """
        Remove old micro-clusters using frequency decay, but only based on
        the event count for this specific DBStream instance's activity label.
        """
        current_activity_event_count = self.activity_event_counters[activity_label]
        self._apply_decay(current_activity_event_count)

        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            last_seen_activity_event = cluster.get("last_seen_activity_event", current_activity_event_count)
            events_since_last_seen = current_activity_event_count - last_seen_activity_event

            # Apply decay
            cluster["weight"] *= np.exp(-events_since_last_seen / self.decay_after_events)

            # Log before deletion
            if cluster["weight"] < self.forgetting_threshold and events_since_last_seen > self.removal_threshold_events:
                log_traceability(
                    "CLUSTER_REMOVAL", activity_label,
                    f"Cluster {cluster_id} removed: Weight={cluster['weight']}, Inactive for {events_since_last_seen} events"
                )
                del self.micro_clusters[cluster_id]

            # Stop removing if within budget
            if len(self.micro_clusters) <= self.lossy_counting_budget:
                break

    def partial_fit(self, vector, activity_label):
        """
        Process a new vector, detect clusters, and update micro-clusters dynamically.
        Uses activity-specific event counts to determine cluster removal.
        """
        self.activity_event_counters[activity_label] += 1
        self.forget_old_clusters(activity_label)

        self.event_count += 1

        if not self.micro_clusters:
            self.micro_clusters[0] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_activity_event": self.activity_event_counters[activity_label]
            }
            logging.info("Created first cluster with vector: %s", vector)
            return 0

        similarities = []
        for cluster_id, cluster in self.micro_clusters.items():
            sim = compute_contextual_weighted_similarity(
                cluster["centroid"], vector, [1] * len(vector), [1] * len(vector), alpha=self.beta
            )
            logging.debug(f"Similarity between cluster {cluster_id} and new vector: {sim}")
            similarities.append((sim, cluster_id))
            logging.debug("Similarity with cluster %d: %.4f", cluster_id, sim)

        similarities.sort(reverse=True, key=lambda x: x[0])
        best_sim, best_cluster_id = similarities[0]

        if best_sim > self.merge_threshold:
            logging.info("Merging vector into cluster %d with similarity %.4f", best_cluster_id, best_sim)
            self._update_cluster(best_cluster_id, vector, activity_label)
            return best_cluster_id
        elif best_sim < self.split_threshold:
            new_cluster_id = len(self.micro_clusters)
            self.micro_clusters[new_cluster_id] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_activity_event": self.activity_event_counters[activity_label]
            }
            logging.info("Creating new cluster %d for vector: %s", new_cluster_id, vector)
            return new_cluster_id
        else:
            logging.info("Assigning vector to existing cluster %d (similarity: %.4f)", best_cluster_id, best_sim)
            return best_cluster_id

    def _update_cluster(self, cluster_id, new_vector, activity_label):
        """
        Update an existing cluster with a new feature vector.
        """
        cluster = self.micro_clusters[cluster_id]
        cluster["weight"] += 1
        cluster["centroid"] = (cluster["centroid"] * (cluster["weight"] - 1) + new_vector) / cluster["weight"]
        logging.debug(f"Updated cluster {cluster_id} centroid to: {cluster['centroid']}")
        cluster["last_seen_activity_event"] = self.activity_event_counters[activity_label]
        logging.info("Updated cluster %d (New weight: %.2f)", cluster_id, cluster["weight"])

    def _apply_decay(self, current_activity_event_count):
        """
        Apply smooth temporal decay to vector frequencies and dynamically adjust thresholds.
        Ensures that decayed clusters fade gradually instead of sudden removals.
        """
        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            time_since_update = current_activity_event_count - cluster["last_seen_activity_event"]
            decay_factor = np.exp(-time_since_update / self.decay_after_events)

            if "vector_frequencies" not in cluster:
                cluster["vector_frequencies"] = {}

            for vector in list(cluster["vector_frequencies"].keys()):
                cluster["vector_frequencies"][vector] *= decay_factor
                if cluster["vector_frequencies"][vector] < self.forgetting_threshold:
                    del cluster["vector_frequencies"][vector]

            if not cluster["vector_frequencies"]:
                logging.info("Removing cluster %d due to full decay", cluster_id)
                del self.micro_clusters[cluster_id]
            else:
                cluster["centroid"] = self._most_frequent_vector(cluster["vector_frequencies"])

    def _most_frequent_vector(self, vector_frequencies):
        if not vector_frequencies:
            return np.zeros_like(self.micro_clusters[0]["centroid"])
        return max(vector_frequencies, key=vector_frequencies.get)

    def get_micro_clusters(self):
        """
        Retrieve a list of all current micro-clusters.
        """
        clusters = [cluster for cluster in self.micro_clusters.values() if cluster["weight"] > 0.01]
        logging.info("Returning %d active micro-clusters", len(clusters))
        return clusters
