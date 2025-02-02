import numpy as np
from collections import defaultdict
from config.config import dbstream_params  # Import parameters dynamically
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_history
from src.utils.logging_utils import log_traceability
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
        # Pull parameters from config.py
        self.clustering_threshold = dbstream_params.get("clustering_threshold", 0.35)
        self.fading_factor = dbstream_params.get("fading_factor", 0.05)
        self.cleanup_interval = dbstream_params.get("cleanup_interval", 2)
        self.split_threshold = dbstream_params.get("split_threshold", 0.4)
        self.merge_threshold = dbstream_params.get("merge_threshold", 0.8)
        self.eps = dbstream_params.get("eps", 0.02)  # Distance threshold for merging clusters
        self.beta = dbstream_params.get("beta", 0.15)  # Sensitivity to new clusters
        self.lambda_ = dbstream_params.get("lambda", 0.0001)  # Cluster decay rate

        self.micro_clusters = []  # Store dynamically updated micro-clusters
        self.event_count = 0
        self.similarity_history = []  # Track all similarities
        self.max_history_size = 15  # Keep track of last N events

    def partial_fit(self, vector):
        """
        Process new vector, detect clusters, and update micro-clusters dynamically.
        """
        self.event_count += 1

        # If no clusters exist, initialize one
        if not self.micro_clusters:
            self.micro_clusters.append({"centroid": np.array(vector), "weight": 1})
            return 0  # Assign first cluster ID

        # Compute similarity with existing clusters
        similarities = []
        for i, cluster in enumerate(self.micro_clusters):
            sim = compute_contextual_weighted_similarity(
                cluster["centroid"],
                vector,
                [1] * len(vector),
                [1] * len(vector),
                alpha=self.beta  # Uses `beta` from config.py
            )
            similarities.append((sim, i))

        similarities.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity

        # Identify the best match
        best_sim, best_cluster_id = similarities[0]

        if best_sim > self.merge_threshold:
            # Merge with the best matching existing cluster
            self._update_cluster(best_cluster_id, vector)
            return best_cluster_id
        elif best_sim < self.split_threshold:
            # Check past feature vectors before creating a new cluster
            variability_factor = adaptive_threshold_variability(activity_feature_history)
            if variability_factor > self.split_threshold:
                new_cluster_id = len(self.micro_clusters)
                self.micro_clusters.append({"centroid": np.array(vector), "weight": 1})
                return new_cluster_id
            else:
                return best_cluster_id
        else:
            return best_cluster_id

    def _update_cluster(self, cluster_id, new_vector):
        """
        Update an existing micro-cluster with a new data point.
        """
        cluster = self.micro_clusters[cluster_id]
        cluster["weight"] += 1
        cluster["centroid"] = (cluster["centroid"] * (cluster["weight"] - 1) + new_vector) / cluster["weight"]

    def get_micro_clusters(self):
        """
        Return a summary of micro-clusters and their centroids.
        """
        return [{"centroid": cluster["centroid"], "weight": cluster["weight"]}
                for cluster in self.micro_clusters]

    def apply_decay(self):
        """
        Apply temporal decay to cluster weights, removing outdated ones.
        """
        for cluster in self.micro_clusters:
            cluster["weight"] *= (1 - self.lambda_)  # Decay based on `lambda`
            if cluster["weight"] < 0.1:  # Remove clusters below threshold
                self.micro_clusters.remove(cluster)
