import numpy as np
from collections import defaultdict, Counter
from src.utils.logging_utils import log_traceability
from config.config import grace_period_events
from src.utils.similarity_utils import compute_contextual_weighted_similarity


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
        self.cleanup_interval = params.get("cleanup_interval", 2)
        self.micro_clusters = []
        self.event_count = 0
        self.similarity_history = []  # Track all similarities
        self.max_history_size = 15    # Keep track of last N events
        self.split_threshold = 0.4    # Threshold for splitting
        self.merge_threshold = 0.8    # Threshold for merging
        self.all_similarities = []  # Track all similarity pairs
        self.clustering_threshold = params.get("clustering_threshold", 1.0)
        self.vectors = []  # Store all vectors
        self.similarity_patterns = []  # Store similarity patterns
    def partial_fit(self, vector):
        """Process new vector and detect patterns."""
        self.vectors.append(vector)
        n_vectors = len(self.vectors)

        # Need at least 3 vectors to detect patterns
        if n_vectors < 3:
            return "no_change", n_vectors - 1

        # Compute similarity matrix for all vectors
        similarity_matrix = np.zeros((n_vectors, n_vectors))
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                sim = compute_contextual_weighted_similarity(
                    self.vectors[i],
                    self.vectors[j],
                    [1] * len(vector),
                    [1] * len(vector),
                    alpha=0.5
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

                # Store similarity pattern
                self.similarity_patterns.append((sim, i, j))

        # Sort by similarity
        self.similarity_patterns.sort(key=lambda x: x[0], reverse=True)

        # Look for groups
        high_sim_pairs = []  # sim > 0.8
        low_sim_pairs = []  # sim < 0.2

        for sim, i, j in self.similarity_patterns:
            if sim > 0.8:
                high_sim_pairs.append((i, j))
            elif sim < 0.2:
                low_sim_pairs.append((i, j))

        # If we have both very similar and very different vectors
        if high_sim_pairs and low_sim_pairs:
            split_indices = set()
            # Include both high and low similarity pairs
            for i, j in high_sim_pairs + low_sim_pairs:
                split_indices.add(i)
                split_indices.add(j)

            if len(split_indices) >= 2:
                return "split", list(split_indices)

        return "no_change", n_vectors - 1

    def update_vector_groups(self, vector, similarity_matrix):
        """Track potential homonymous groups based on similarities"""
        n = len(similarity_matrix)
        high_similarity_indices = []

        # Find highly similar vectors
        for i in range(n):
            if similarity_matrix[i][-1] > self.similarity_threshold:  # Compare with new vector
                high_similarity_indices.append(i)

        if high_similarity_indices:
            # Add to existing group or create new
            group_found = False
            for group in self.vector_groups.values():
                if any(idx in group for idx in high_similarity_indices):
                    group.extend(high_similarity_indices)
                    group.append(n - 1)  # Add current vector
                    group_found = True
                    break

            if not group_found:
                new_group_id = len(self.vector_groups)
                self.vector_groups[new_group_id] = high_similarity_indices + [n - 1]

        # Check for potential splits
        for group_id, group in self.vector_groups.items():
            if len(group) >= self.min_group_size:
                # Compute average intra-group similarity
                intra_similarities = []
                for i in group:
                    for j in group:
                        if i < j:
                            intra_similarities.append(similarity_matrix[i][j])

                if intra_similarities and np.mean(intra_similarities) > 0.8:
                    # Found a distinct group - potential homonym
                    return True, group

        return False, []

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
