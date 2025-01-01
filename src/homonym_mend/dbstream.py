import numpy as np
from collections import defaultdict, Counter
from src.utils.logging_utils import log_traceability


class DBStream:
    """
    DBStream: Custom version for clustering categorical/discretized feature vectors.
    Tracks the most frequent feature vector as the cluster centroid.
    """
    def __init__(self, params):
        """
        Initialize DBStream with configuration parameters.

        Parameters:
            params (dict): Dictionary containing DBStream configuration parameters.
        """
        self.clustering_threshold = params.get("clustering_threshold", 1.0)
        self.fading_factor = params.get("fading_factor", 0.01)
        self.grace_period_events = params.get("grace_period_events", 10)
        self.cleanup_interval = params.get("cleanup_interval", 2)
        self.micro_clusters = []  # List of micro-clusters
        self.event_count = 0  # Track the number of processed events

    def partial_fit(self, vector):
        """
        Incrementally fit a new feature vector into clusters, tracking the most frequent vector as centroid.
        """
        log_traceability("dbstream_update", "DBStream", {"vector": vector})

        # Update existing clusters or create a new cluster
        matched_cluster = None
        self.event_count += 1

        for cluster in self.micro_clusters:
            # Compare the new vector with cluster's centroid
            similarity = self._vector_similarity(vector, cluster["centroid"])
            if abs(similarity - 1.0) < 1e-6:  # Identical vectors
                cluster["vector_frequencies"][tuple(vector)] += 1  # Increase frequency
                cluster["last_updated"] = self.event_count
                cluster["centroid"] = self._most_frequent_vector(cluster["vector_frequencies"])
                matched_cluster = cluster
                break

        if not matched_cluster:
            # Create a new cluster
            new_cluster = {
                "centroid": vector,
                "vector_frequencies": Counter({tuple(vector): 1}),
                "last_updated": self.event_count
            }
            self.micro_clusters.append(new_cluster)
            log_traceability("new_cluster", "DBStream", {"centroid": vector})

        # Apply temporal decay
        self._apply_decay()

    def get_micro_clusters(self):
        """
        Return a summary of micro-clusters and their centroids.
        """
        return [{"centroid": list(cluster["centroid"]), "frequency": cluster["vector_frequencies"], "last_updated": cluster["last_updated"]}
                for cluster in self.micro_clusters]

    def _apply_decay(self):
        """
        Apply temporal decay to vector frequencies and clean up outdated clusters.
        """
        for cluster in self.micro_clusters:
            # Apply decay only to older vectors beyond the grace period
            if self.event_count - cluster["last_updated"] > self.grace_period_events:
                for vector in list(cluster["vector_frequencies"]):
                    cluster["vector_frequencies"][vector] *= self.fading_factor
                    if cluster["vector_frequencies"][vector] < 1e-2:  # Threshold for removal
                        del cluster["vector_frequencies"][vector]

                if not cluster["vector_frequencies"]:
                    self.micro_clusters.remove(cluster)  # Remove empty clusters
                else:
                    cluster["centroid"] = self._most_frequent_vector(cluster["vector_frequencies"])

    def _vector_similarity(self, v1, v2):
        """
        Compute similarity between two vectors (exact match for categorical/discretized vectors).
        """
        return 1.0 if np.array_equal(v1, v2) else 0.0

    def _most_frequent_vector(self, vector_frequencies):
        """
        Return the most frequent vector in the cluster.
        """
        return max(vector_frequencies, key=vector_frequencies.get)
