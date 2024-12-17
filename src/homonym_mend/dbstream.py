import numpy as np
from src.utils.logging_utils import log_traceability

class DBStream:
    """
    DBStream: Density-Based Stream Clustering Algorithm with Grace Period for Decay.
    """
    def __init__(self, clustering_threshold=1.0, fading_factor=0.01, cleanup_interval=2, intersection_factor=0.3, grace_period_events=10):
        self.clustering_threshold = clustering_threshold
        self.fading_factor = fading_factor
        self.cleanup_interval = cleanup_interval
        self.intersection_factor = intersection_factor
        self.grace_period_events = grace_period_events
        self.micro_clusters = []  # List of micro-clusters
        self.current_event_count = 0  # Track the number of processed events

    def partial_fit(self, vector):
        """
        Incrementally update micro-clusters with a new feature vector.
        """
        log_traceability("dbstream_update", "DBStream", {"vector": vector})

        matched_cluster = None
        self.current_event_count += 1  # Increment event counter for each new vector

        for cluster in self.micro_clusters:
            # Skip low-weight clusters
            if cluster["weight"] < 0.01:
                continue

            # Check if the cluster's grace period has elapsed
            if self.current_event_count - cluster["last_updated_event"] >= self.grace_period_events:
                similarity = self._similarity(cluster["centroid"], vector)
                if similarity >= self.clustering_threshold:
                    # Update existing cluster
                    cluster["centroid"] = self._update_centroid(cluster["centroid"], vector, cluster["weight"])
                    cluster["weight"] += 1
                    cluster["last_updated_event"] = self.current_event_count
                    matched_cluster = cluster
                    break

        if not matched_cluster:
            # Create a new micro-cluster
            new_cluster = {"centroid": vector, "weight": 1, "last_updated_event": self.current_event_count}
            self.micro_clusters.append(new_cluster)

        # Apply decay to clusters if grace period has elapsed
        self._decay_clusters()

        return matched_cluster or new_cluster

    def _decay_clusters(self):
        """
        Apply decay to all micro-clusters based on the fading factor.
        """
        for cluster in self.micro_clusters:
            # Only decay clusters after grace period
            if self.current_event_count - cluster["last_updated_event"] >= self.grace_period_events:
                cluster["weight"] *= self.fading_factor

        # Remove clusters with weights below threshold
        self.micro_clusters = [c for c in self.micro_clusters if c["weight"] > 0.01]

    def get_micro_clusters(self):
        """
        Return all micro-clusters.
        """
        return self.micro_clusters

    def set_micro_clusters(self, micro_clusters):
        """
        Set the list of micro-clusters.
        """
        self.micro_clusters = micro_clusters

    def _similarity(self, v1, v2):
        """
        Compute similarity between two vectors (cosine similarity).
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _update_centroid(self, centroid, vector, weight):
        """
        Update the centroid of a micro-cluster with weighted averaging.
        """
        return [(c * weight + v) / (weight + 1) for c, v in zip(centroid, vector)]


def cleanup_clusters(dbstream_instance, weight_threshold=0.5):
    """
    Remove clusters with weights below a defined threshold.
    """
    micro_clusters = dbstream_instance.get_micro_clusters()
    retained_clusters = [
        cluster for cluster in micro_clusters if cluster["weight"] >= weight_threshold
    ]
    dbstream_instance.set_micro_clusters(retained_clusters)
    log_traceability("cleanup", "Cluster Maintenance", {
        "removed_clusters": len(micro_clusters) - len(retained_clusters),
        "retained_clusters": len(retained_clusters)
    })
