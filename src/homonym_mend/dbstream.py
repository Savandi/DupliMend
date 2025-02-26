from collections import defaultdict
import numpy as np
from config.config import dbstream_params, decay_after_events, lossy_counting_budget, \
    removal_threshold_events, frequency_decay_threshold
from src.utils.global_state import dbstream_clusters  # This is fine now!
from src.utils.logging_utils import log_traceability
from src.utils.similarity_utils import compute_contextual_weighted_similarity

class DBStream:
    """
    DBStream: Custom version for clustering categorical/discretized feature vectors.
    Tracks the most frequent feature vector as the cluster centroid.
    """

    def __init__(self):
        # Same DBStream initialization, no changes here
        self.clustering_threshold = dbstream_params.get("clustering_threshold")
        self.fading_factor = dbstream_params.get("fading_factor")
        self.cleanup_interval = dbstream_params.get("cleanup_interval")
        self.merge_threshold = dbstream_params.get("merge_threshold")
        self.split_threshold = dbstream_params.get("split_threshold")
        self.eps = dbstream_params.get("eps")
        self.beta = dbstream_params.get("beta")
        self.lambda_ = dbstream_params.get("lambda")
        self.lossy_counting_budget = lossy_counting_budget
        self.removal_threshold_events = removal_threshold_events
        self.activity_event_counters = defaultdict(int)

        self.decay_after_events = decay_after_events
        self.forgetting_threshold = frequency_decay_threshold

        self.micro_clusters = {}
        self.event_count = 0
        self.similarity_history = []
        self.max_history_size = 15

        log_traceability("dbstream_init", "DBStream", {
            "merge_threshold": self.merge_threshold,
            "split_threshold": self.split_threshold
        })

    def forget_old_clusters(self, activity_label):
        current_activity_event_count = self.activity_event_counters[activity_label]
        self._apply_decay(current_activity_event_count)

        for cluster_id in list(self.micro_clusters.keys()):
            cluster = self.micro_clusters[cluster_id]
            last_seen_activity_event = cluster.get("last_seen_activity_event", current_activity_event_count)
            events_since_last_seen = current_activity_event_count - last_seen_activity_event

            cluster["weight"] *= np.exp(-events_since_last_seen / self.decay_after_events)

            if cluster["weight"] < self.forgetting_threshold and events_since_last_seen > self.removal_threshold_events:
                log_traceability("cluster_removal", activity_label, {
                    "cluster_id": cluster_id,
                    "reason": "Low weight and inactivity",
                    "weight": cluster["weight"],
                    "events_since_last_seen": events_since_last_seen
                })
                del self.micro_clusters[cluster_id]

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
            print(f"[DEBUG] Created first cluster for {activity_label} with vector: {vector}")
            return 0

        print(f"[DEBUG] Computing similarity for {activity_label}: {vector}")  # ✅ Add this here

        similarities = []
        for cluster_id, cluster in self.micro_clusters.items():
            sim = compute_contextual_weighted_similarity(
                cluster["centroid"], vector, [1] * len(vector), [1] * len(vector), alpha=self.beta
            )
            print(f"[DEBUG] Similarity with cluster {cluster_id}: {sim}")  # ✅ Add this here
            similarities.append((sim, cluster_id))

        similarities.sort(reverse=True, key=lambda x: x[0])
        best_sim, best_cluster_id = similarities[0]

        if best_sim > self.merge_threshold:
            print(f"[DEBUG] Merging into cluster {best_cluster_id} with similarity {best_sim}")
            self._update_cluster(best_cluster_id, vector, activity_label)
            return best_cluster_id
        elif best_sim < self.split_threshold:
            new_cluster_id = len(self.micro_clusters)
            self.micro_clusters[new_cluster_id] = {
                "centroid": np.array(vector),
                "weight": 1,
                "last_seen_activity_event": self.activity_event_counters[activity_label]
            }
            print(f"[DEBUG] Creating new cluster {new_cluster_id} for {activity_label}")
            return new_cluster_id
        else:
            print(f"[DEBUG] Assigning {activity_label} to existing cluster {best_cluster_id} (Similarity: {best_sim})")
            return best_cluster_id

    def _update_cluster(self, cluster_id, new_vector, activity_label):
        """
        Update an existing cluster with a new feature vector.
        """
        cluster = self.micro_clusters[cluster_id]
        cluster["weight"] += 1
        cluster["centroid"] = (cluster["centroid"] * (cluster["weight"] - 1) + new_vector) / cluster["weight"]
        cluster["last_seen_activity_event"] = self.activity_event_counters[activity_label]

        log_traceability("cluster_update", activity_label, {
            "cluster_id": cluster_id,
            "new_weight": cluster["weight"],
            "updated_centroid": cluster["centroid"].tolist()
        })

    def _apply_decay(self, current_activity_event_count):
        """
        Apply smooth temporal decay to vector frequencies and dynamically adjust thresholds.
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
                log_traceability("cluster_decay_removal", cluster_id, {
                    "reason": "All vectors decayed",
                    "removed_cluster_id": cluster_id
                })
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
        log_traceability("micro_cluster_summary", "DBStream", {
            "total_active_clusters": len(clusters)
        })
        return clusters
