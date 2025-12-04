import numpy as np
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import KMeans
from src.duplimend_framework.utils.global_state import event_cluster_mapping, cluster_event_mapping
from src.duplimend_framework.utils.logging_utils import log_traceability
from src.duplimend_framework.utils.global_state import event_embedding_mapping
from sklearn.preprocessing import normalize


class ClusteringAdapter(ABC):
    """
    Base adapter class for online clustering algorithms.
    Provides a consistent interface regardless of underlying implementation.

    Distance Metrics:
        - cosine: Range [0, 2], typical threshold ~0.01-0.95
        - euclidean: Range [0, 2] (normalized vectors), typical threshold ~0.01-0.95
        - manhattan: Unbounded [0, inf), typical threshold depends on feature scale

    Note: Euclidean distance uses L2-normalized vectors to ensure bounded range,
    allowing consistent thresholds with cosine distance. Manhattan distance remains
    unbounded and may require different threshold tuning.
    """
    def __init__(self, algorithm_name, config=None):
        self.algorithm_name = algorithm_name
        self.config = config or {}
        self.cluster_counter = 0
        self.micro_clusters = {}  # Track centroid and metadata for each cluster
        self.cluster_history = defaultdict(list)  # Track evolution of clusters
        self.last_modified = {}  # Track when each cluster was last modified
        self.vector_hash_cache = {}  # Cache vector hashes for faster lookups
        self.variance_cache = {}
        self.variance_dirty = set()

        
    @abstractmethod
    def partial_fit(self, vector, feature_names, timestamp=None):
        """
        Process a new feature vector and assign to best cluster or create a new one.
        Returns: (cluster_id, is_new_cluster)
        """
        pass
    
    @abstractmethod
    def predict(self, vector, feature_names):
        """
        Find the best cluster for a vector without updating the model.
        Returns: cluster_id or None if no suitable cluster
        """
        pass
    
    def create_new_cluster(self, vector, feature_names, reason="new_pattern", timestamp=None):
        """
        Create a new cluster with the given vector as centroid.
        Returns: new cluster ID
        """
        new_cluster_id = self.cluster_counter
        self.cluster_counter += 1
        
        # Ensure consistent timestamp handling
        if timestamp is None:
            timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            # Convert other timestamp formats if needed
            try:
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)
                else:
                    timestamp = datetime.now()
            except (ValueError, OSError):
                timestamp = datetime.now()
        
        # Initialize with zeros for sum_squares (no variance when only one point)
        self.micro_clusters[new_cluster_id] = {
            "centroid": np.array(vector, dtype=np.float64),
            "feature_names": feature_names,
            "weight": 1.0,
            "samples": 1,
            "sum_squares": np.zeros_like(vector),  # Initialize with zeros
            "creation_timestamp": timestamp,
            "last_updated": timestamp,
            "creation_reason": reason
        }
            
        # Update last modified timestamp
        self.last_modified[new_cluster_id] = timestamp
        
        # Save initial state in history
        self.cluster_history[new_cluster_id].append({
            "timestamp": timestamp,
            "centroid": np.array(vector).copy(),
            "weight": 1.0,
            "samples": 1,
            "action": "create"
        })
        
        log_traceability("cluster_creation", self.algorithm_name, {
            "cluster_id": new_cluster_id,
            "reason": reason,
            "feature_count": len(feature_names)
        })
        
        return new_cluster_id
    
    def update_cluster(self, cluster_id, new_vector, feature_names, timestamp=None):
        """
        Update an existing cluster with a new vector.
        Returns: updated cluster_id
        """
        if cluster_id not in self.micro_clusters:
            # If cluster doesn't exist, create a new one
            return self.create_new_cluster(new_vector, feature_names, "update_missing_cluster", timestamp)
        
        cluster_info = self.micro_clusters[cluster_id]
        old_centroid = cluster_info["centroid"]
        old_weight = cluster_info["weight"]
        old_samples = cluster_info["samples"]
        
        # Calculate learning rate based on cluster weight
        alpha = self.config.get("learning_rate", 0.1)
        decay = 1.0 / (1.0 + old_weight * alpha)
        
        # Update centroid with weighted average
        if len(old_centroid) == len(new_vector):
            # Store old centroid for variance calculation
            current_centroid = np.array(old_centroid)
            
            # Update the centroid
            new_centroid = (1.0 - decay) * current_centroid + decay * new_vector
            cluster_info["centroid"] = new_centroid
            
            # Update sum of squares for online variance calculation
            # Using a modified version of Welford's algorithm
            delta = new_vector - current_centroid
            if "sum_squares" not in cluster_info:
                cluster_info["sum_squares"] = np.square(delta)
            else:
                # Scale factor to correct for the moving centroid
                scale = (old_samples) / (old_samples + 1.0)
                cluster_info["sum_squares"] = scale * cluster_info["sum_squares"] + np.square(delta) * decay
        else:
            cluster_info["centroid"] = old_centroid
        
        # Update other metadata
        cluster_info["weight"] += 1.0
        cluster_info["samples"] += 1
        cluster_info["last_updated"] = timestamp or datetime.now()
                
        # Update last modified timestamp
        self.last_modified[cluster_id] = timestamp or datetime.now()
        
        # Save state in history
        self.cluster_history[cluster_id].append({
            "timestamp": timestamp or datetime.now(),
            "centroid": np.array(cluster_info["centroid"]).copy(),
            "weight": cluster_info["weight"],
            "samples": cluster_info["samples"],
            "action": "update"
        })
        self.variance_dirty.add(cluster_id)
        return cluster_id
    
    def merge_clusters(self, cid1, cid2, timestamp=None):
        """Merge two clusters"""
        c1 = self.micro_clusters.get(cid1)
        c2 = self.micro_clusters.get(cid2)
        
        if not c1 or not c2:
            return None
            
        # Calculate weighted average of centroids
        w1 = c1.get("weight", 1.0)
        w2 = c2.get("weight", 1.0)
        total_weight = w1 + w2
        
        merged_centroid = (c1["centroid"] * w1 + c2["centroid"] * w2) / total_weight
        
        # Use the ID of the heavier cluster as the merged ID
        merged_id = cid1 if w1 >= w2 else cid2
        other_id = cid2 if merged_id == cid1 else cid1
        
        # Update the merged cluster
        self.micro_clusters[merged_id] = {
            "centroid": merged_centroid,
            "feature_names": c1["feature_names"],  # Assuming same features
            "weight": total_weight,
            "samples": c1.get("samples", 1) + c2.get("samples", 1),
            "sum_squares": (c1.get("sum_squares", np.zeros_like(c1["centroid"])) * w1 + 
                            c2.get("sum_squares", np.zeros_like(c2["centroid"])) * w2) / total_weight,
            "merged_from": [cid1, cid2]
        }
        
        self.variance_dirty.update([cid1, cid2, merged_id])
        self.variance_cache.pop(cid1, None)
        self.variance_cache.pop(cid2, None)
        # Log the merge in history
        self.cluster_history[merged_id].append({
            "timestamp": timestamp,
            "action": "merge",
            "merged_with": other_id
        })
        
        # Remove the other cluster
        if other_id in self.micro_clusters:
            self.cluster_history[other_id].append({
                "timestamp": timestamp,
                "action": "merged_into",
                "target": merged_id
            })
            del self.micro_clusters[other_id]
        
        return merged_id
        
    def update_event_mappings(self, event_id, activity_label, cluster_id):
        """
        Update the mappings between events and clusters
        """
        # Map event to cluster
        if event_id is not None:
            event_cluster_mapping[event_id] = (activity_label, cluster_id)
            
            # Also track which events are in each cluster
            cluster_key = (activity_label, cluster_id)
            if cluster_key not in cluster_event_mapping:
                cluster_event_mapping[cluster_key] = []
            
            if event_id not in cluster_event_mapping[cluster_key]:
                cluster_event_mapping[cluster_key].append(event_id)
    
    def get_cached_variance(self, cluster_id):
        if cluster_id in self.variance_dirty or cluster_id not in self.variance_cache:
            cluster = self.micro_clusters.get(cluster_id)
            if cluster:
                self.variance_cache[cluster_id] = self.calculate_cluster_variance(cluster)
                self.variance_dirty.discard(cluster_id)
            else:
                return 0.0
        return self.variance_cache.get(cluster_id, 0.0)

    def get_cluster_centroid(self, cluster_id):
        """
        Get the centroid vector for a cluster
        """
        if cluster_id in self.micro_clusters:
            return self.micro_clusters[cluster_id]["centroid"]
        return None
    
    def get_cluster_feature_names(self, cluster_id):
        """
        Get the feature names for a cluster
        """
        if cluster_id in self.micro_clusters:
            return self.micro_clusters[cluster_id]["feature_names"]
        return None
    
    def calculate_distance(self, vector1, vector2, feature_names=None, hash_dimensions=None):
        """
        Calculate distance between two vectors based on configured metric.
        Supports: 'cosine', 'euclidean', 'manhattan'

        For euclidean distance, vectors are normalized to unit length,
        making the distance range [0, 2] (same as cosine), allowing
        consistent thresholds across metrics.
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        # Get distance metric from config (default to cosine for backward compatibility)
        distance_metric = self.config.get("distance_metric", "cosine").lower()

        if distance_metric == "euclidean":
            # Normalize vectors to unit length for bounded range [0, 2]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 2.0  # Maximum distance if either vector is zero

            v1_normalized = v1 / norm1
            v2_normalized = v2 / norm2

            # Euclidean distance on normalized vectors: range [0, 2]
            return np.linalg.norm(v1_normalized - v2_normalized)

        elif distance_metric == "manhattan":
            # Manhattan distance (L1 norm)
            return np.sum(np.abs(v1 - v2))

        else:  # default to cosine
            # Cosine distance (1 - cosine similarity)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance if either vector is zero
            cosine_similarity = np.dot(v1, v2) / (norm1 * norm2)
            # Clamp value to [-1, 1] to avoid floating point issues
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            return 1.0 - cosine_similarity
    
    def find_closest_cluster(self, vector, feature_names, min_weight=None):
        """
        Find the closest cluster to the given vector.
        Returns: (cluster_id, distance) or (None, float('inf')) if no clusters exist
        """
        if min_weight is None:
            min_weight = self.config.get("min_cluster_weight", 0.5)
        if not self.micro_clusters:
            return None, float('inf')
        
        best_cluster = None
        min_distance = float('inf')
        
        for cluster_id, info in self.micro_clusters.items():
            # Skip clusters with weight below threshold
            if info["weight"] < min_weight:
                continue
                
            # Check feature name compatibility
            if len(info["centroid"]) != len(vector):
                # Skip clusters with mismatched dimensions
                continue
            
            # Calculate distance
            distance = self.calculate_distance(info["centroid"], vector, feature_names)
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_id
        
        return best_cluster, min_distance
    
    def _hash_vector(self, vector):
        """Generate a stable hash for a vector"""
        # Convert to string with fixed precision to ensure stable hashing
        vec_str = ";".join([f"{v:.6f}" for v in vector])
        
        # Check cache first
        if vec_str in self.vector_hash_cache:
            return self.vector_hash_cache[vec_str]
        
        # Hash the vector string
        m = hashlib.md5(vec_str.encode())
        h = m.hexdigest()
        
        # Cache the result
        self.vector_hash_cache[vec_str] = h
        
        return h
    
    def get_significant_clusters(self, min_weight=None):
        if min_weight is None:
            min_weight = self.config.get("min_cluster_weight", 0.5)
        return [
            cid for cid, info in self.micro_clusters.items()
            if info["weight"] >= min_weight
        ]
        
    def check_for_merges(self, merge_threshold, min_weight, timestamp=None):
        """Check for and merge similar clusters"""
        merge_operations = []
        merge_candidates = []
        
        # Get clusters with sufficient weight
        significant_clusters = self.get_significant_clusters(min_weight)
        if len(significant_clusters) < 2:
            return {"merge_occurred": False, "operations": []}
            
        # Compute distances between all pairs
        for i, cid1 in enumerate(significant_clusters):
            for j, cid2 in enumerate(significant_clusters):
                if i >= j:  # Only process upper triangle
                    continue
                    
                # Get cluster centroids
                c1 = self.micro_clusters.get(cid1)
                c2 = self.micro_clusters.get(cid2)
                
                if not c1 or not c2:
                    continue
                    
                # Calculate distance
                distance = self.calculate_distance(c1["centroid"], c2["centroid"], c1["feature_names"])
                
                if distance < merge_threshold:
                    merge_candidates.append((cid1, cid2, distance))
        
        # Process merges (sorted by distance)
        merge_candidates.sort(key=lambda x: x[2])
        processed_clusters = set()
        
        for cid1, cid2, distance in merge_candidates:
            if cid1 in processed_clusters or cid2 in processed_clusters:
                continue  # Skip if either cluster already merged
                
            # Merge the clusters
            merged_id = self.merge_clusters(cid1, cid2, timestamp)
            if merged_id:
                merge_operations.append((cid1, cid2, merged_id, distance))
                processed_clusters.add(cid1)
                processed_clusters.add(cid2)
        
        return {"merge_occurred": len(merge_operations) > 0, "operations": merge_operations}

    def get_cluster_info(self, cluster_id):
        """
        Get detailed information about a cluster.
        """
        if cluster_id not in self.micro_clusters:
            return None
        
        info = self.micro_clusters[cluster_id]
        
        # Get history summary
        history = self.cluster_history.get(cluster_id, [])
        history_summary = {
            "creation_time": history[0]["timestamp"] if history else None,
            "update_count": len([h for h in history if h["action"] == "update"]),
            "merge_count": len([h for h in history if h["action"] == "merge_target"])
        }
        
        # Build result dictionary
        return {
            "cluster_id": cluster_id,
            "centroid": info["centroid"].tolist() if isinstance(info["centroid"], np.ndarray) else info["centroid"],
            "weight": info["weight"],
            "samples": info["samples"],
            "creation_reason": info.get("creation_reason", "unknown"),
            "creation_time": info.get("creation_timestamp"),
            "last_updated": info.get("last_updated"),
            "history": history_summary
        }
    
    def decay_clusters(self, decay_factor=0.9, min_weight=0.1, timestamp=None):
        """
        Apply exponential decay to all clusters.
        Returns: list of removed cluster IDs
        """
        removed_clusters = []
        
        for cluster_id in list(self.micro_clusters.keys()):
            # Apply decay
            self.micro_clusters[cluster_id]["weight"] *= decay_factor
            
            # Check if cluster should be removed
            if self.micro_clusters[cluster_id]["weight"] < min_weight:
                # Record removal in history
                self.cluster_history[cluster_id].append({
                    "timestamp": timestamp or datetime.now(),
                    "action": "decay_removed",
                    "final_weight": self.micro_clusters[cluster_id]["weight"]
                })
                
                # Remove the cluster
                del self.micro_clusters[cluster_id]
                removed_clusters.append(cluster_id)
                
                log_traceability("cluster_removal", self.algorithm_name, {
                    "cluster_id": cluster_id,
                    "reason": "decay",
                    "timestamp": timestamp or datetime.now()
                })
        self.variance_cache.pop(cluster_id, None)
        self.variance_dirty.discard(cluster_id)
        return removed_clusters
        
    def split_cluster(self, cluster_id, timestamp=None):
        """
        Split a cluster into two using KMeans, respecting the configured distance metric.
        """
        original = self.micro_clusters.get(cluster_id)
        if not original:
            return None

        # Get all event IDs for this cluster
        activity_label = None
        for (act, cid) in cluster_event_mapping:
            if cid == cluster_id:
                activity_label = act
                break
        if activity_label is None:
            return None

        event_ids = cluster_event_mapping.get((activity_label, cluster_id), [])
        if len(event_ids) < 2:
            return None  # Not enough members to split

        # Gather embeddings for all members
        member_vectors = [event_embedding_mapping[eid] for eid in event_ids if eid in event_embedding_mapping]
        if len(member_vectors) < 2:
            return None

        X = np.stack(member_vectors)

        # Get distance metric from config
        distance_metric = self.config.get("distance_metric", "cosine").lower()

        # Normalize vectors for cosine and euclidean (to ensure bounded range)
        if distance_metric in ["cosine", "euclidean"]:
            X_normalized = normalize(X, norm='l2')
        else:
            X_normalized = X  # Use raw vectors for manhattan
        
        # Check for duplicate points before clustering
        unique_vectors, inverse_indices = np.unique(X_normalized, axis=0, return_inverse=True)
        
        if len(unique_vectors) < 2:
            return None
        
        if len(unique_vectors) == 2:
            distance = np.linalg.norm(unique_vectors[0] - unique_vectors[1])
            if distance < 1e-6:
                return None
        
        try:
            # Use standard KMeans on normalized vectors (equivalent to cosine distance)
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42, max_iter=100)
            labels = kmeans.fit_predict(X_normalized)
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None

        except Exception:
            return None

        # Continue with the rest of the split logic...
        # Determine which label should keep the original cluster ID (e.g., the larger group)
        label_counts = np.bincount(labels)
        original_label = np.argmax(label_counts)
        new_label = 1 - original_label

        # Update the original cluster with its new centroid and members
        original_members = [eid for eid, label in zip(event_ids, labels) if label == original_label]
        new_members = [eid for eid, label in zip(event_ids, labels) if label == new_label]

        self.micro_clusters[cluster_id]["centroid"] = kmeans.cluster_centers_[original_label]
        self.micro_clusters[cluster_id]["samples"] = len(original_members)
        self.micro_clusters[cluster_id]["weight"] = float(len(original_members))
        cluster_event_mapping[(activity_label, cluster_id)] = original_members
        for eid in original_members:
            event_cluster_mapping[eid] = (activity_label, cluster_id)

        # Create the new cluster for the split-off members
        new_id = self.create_new_cluster(
            kmeans.cluster_centers_[new_label],
            original["feature_names"],
            "variance_split",
            timestamp
        )
        self.variance_dirty.update([cluster_id, new_id])
        cluster_event_mapping[(activity_label, new_id)] = new_members
        for eid in new_members:
            event_cluster_mapping[eid] = (activity_label, new_id)

        # Log the split operation
        self.cluster_history[cluster_id].append({
            "timestamp": timestamp,
            "action": "split",
            "split_into": [cluster_id, new_id]
        })

        return [cluster_id, new_id]
    
    def calculate_cluster_variance(self, cluster):
        """Calculate internal variance of a cluster using actual member vectors"""
        if cluster.get("samples", 0) <= 1:
            return 0.0
        
        cluster_id = None
        for cid, info in self.micro_clusters.items():
            if info is cluster:
                cluster_id = cid
                break
        
        if cluster_id is None:
            return 0.0
        
        # Get actual member vectors for more accurate variance calculation
        activity_label = None
        for (act, cid) in cluster_event_mapping:
            if cid == cluster_id:
                activity_label = act
                break
        
        if activity_label:
            event_ids = cluster_event_mapping.get((activity_label, cluster_id), [])
            if len(event_ids) > 1:
                member_vectors = [event_embedding_mapping[eid] for eid in event_ids 
                                if eid in event_embedding_mapping]
                if len(member_vectors) > 1:
                    X = np.array(member_vectors)
                    X_normalized = normalize(X, norm='l2')
                    
                    # Check for duplicate points
                    unique_vectors = np.unique(X_normalized, axis=0)
                    if len(unique_vectors) < 2:
                        return 0.0
                    
                    # FAST variance calculation using numpy
                    variance = np.var(X_normalized, axis=0).mean()
                    return variance
        
        return 0.0

    def get_event_ids_for_clusters(self, activity_label, cluster_ids):
        event_ids = []
        for cid in cluster_ids:
            key = (activity_label, cid)
            if key in cluster_event_mapping:
                event_ids.extend(cluster_event_mapping[key])
        return event_ids
    
    # def check_for_splits(self, variance_threshold, min_weight, timestamp=None):
    #     """Check if any clusters should be split based on internal variance"""
    #     split_operations = []
        
    #     # Get clusters with sufficient weight
    #     significant_clusters = self.get_significant_clusters(min_weight)
                
    #     for cluster_id in significant_clusters:
    #         cluster = self.micro_clusters.get(cluster_id)
    #         if not cluster:
    #             continue
            
    #         # Pre-check: Do we have enough diverse points to split?
    #         activity_label = None
    #         for (act, cid) in cluster_event_mapping:
    #             if cid == cluster_id:
    #                 activity_label = act
    #                 break
            
    #         if activity_label:
    #             event_ids = cluster_event_mapping.get((activity_label, cluster_id), [])
    #             if len(event_ids) < 4:  # Need at least 4 points for meaningful split
    #                 continue
                    
    #             # Check for unique vectors
    #             member_vectors = [event_embedding_mapping[eid] for eid in event_ids if eid in event_embedding_mapping]
    #             if len(member_vectors) < 4:
    #                 continue
                    
    #             X_normalized = normalize(np.array(member_vectors), norm='l2')
    #             unique_vectors = np.unique(X_normalized, axis=0)
    #             if len(unique_vectors) < 2:
    #                 print(f"[DEBUG] Skipping split for cluster {cluster_id}: only {len(unique_vectors)} unique points")
    #                 continue
                
    #         # Calculate cluster variance only after pre-checks pass
    #         variance = self.calculate_cluster_variance(cluster)
    #         print(f"[DEBUG] Cluster {cluster_id} variance: {variance}, threshold: {variance_threshold}")
            
    #         if variance > variance_threshold:
    #             print(f"[SPLIT] Cluster {cluster_id} split, variance={variance}")
    #             # Create new cluster from this one
    #             new_id = self.split_cluster(cluster_id, timestamp)
    #             if new_id:
    #                 # Make sure format matches what process_event expects
    #                 split_operations.append((cluster_id, [new_id]))
                    
    #     return {"split_occurred": len(split_operations) > 0, "operations": split_operations}
    def check_for_splits(self, variance_threshold, min_weight, timestamp=None):
        split_operations = []
        significant_clusters = self.get_significant_clusters(min_weight)

        for cluster_id in significant_clusters:
            variance = self.get_cached_variance(cluster_id)
            if variance > variance_threshold:
                result = self.split_cluster(cluster_id, timestamp)
                if result:
                    split_operations.append((cluster_id, [result]))

        return {"split_occurred": len(split_operations) > 0, "operations": split_operations}
