import numpy as np


class DBStream:
    """
    DBStream: Density-Based Stream Clustering Algorithm.
    """
    def __init__(self, clustering_threshold=1.0, fading_factor=0.01, cleanup_interval=2, intersection_factor=0.3):
        self.clustering_threshold = clustering_threshold
        self.fading_factor = fading_factor
        self.cleanup_interval = cleanup_interval
        self.intersection_factor = intersection_factor
        self.micro_clusters = []  # List of micro-clusters

    def partial_fit(self, vector):
        """
        Incrementally update micro-clusters with a new feature vector.
        """
        matched_cluster = None
        for cluster in self.micro_clusters:
            similarity = self._similarity(cluster["centroid"], vector)
            if similarity >= self.clustering_threshold:
                # Update existing cluster
                cluster["centroid"] = self._update_centroid(cluster["centroid"], vector)
                cluster["weight"] += 1
                matched_cluster = cluster
                break

        if not matched_cluster:
            # Create a new micro-cluster
            new_cluster = {"centroid": vector, "weight": 1}
            self.micro_clusters.append(new_cluster)

        self._decay_clusters()
        return matched_cluster or new_cluster

    def get_micro_clusters(self):
        """
        Return all micro-clusters.
        """
        return self.micro_clusters

    def _similarity(self, v1, v2):
        """
        Compute similarity between two vectors (cosine similarity).
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _update_centroid(self, centroid, vector):
        """
        Update the centroid of a micro-cluster.
        """
        return [(c + v) / 2 for c, v in zip(centroid, vector)]

    def _decay_clusters(self):
        """
        Apply decay to all micro-clusters.
        """
        for cluster in self.micro_clusters:
            cluster["weight"] *= self.fading_factor
        self.micro_clusters = [c for c in self.micro_clusters if c["weight"] > 0.1]
