import numpy as np
from river import cluster
from src.duplimend_framework.cluster_adapter import ClusteringAdapter


class DenStreamAdapter(ClusteringAdapter):
    """
    Adapter for River's DenStream clustering algorithm
    """

    def __init__(self, config=None):
        super().__init__("RiverDenStream", config)

        # Default parameters
        self.decaying_factor = self.config.get("decaying_factor")
        self.beta = self.config.get("beta")
        self.mu = self.config.get("mu")
        self.epsilon = self.config.get("epsilon")
        self.n_samples_init = self.config.get("n_samples_init")

        # Initialize River's DenStream
        self.denstream = cluster.DenStream(
            decaying_factor=self.decaying_factor,
            beta=self.beta,
            mu=self.mu,
            epsilon=self.epsilon,
            n_samples_init=self.n_samples_init
        )

        # Map River's internal cluster IDs to our IDs
        self.river_to_adapter_ids = {}

    def _vector_to_dict(self, vector, feature_names):
        """Convert numpy vector to dict for River's consumption"""
        return {i: float(v) for i, v in enumerate(vector)}

    def partial_fit(self, vector, feature_names, timestamp=None):
        """Process a new vector and assign to best cluster or create new one"""
        # Convert to dict format for River
        x = self._vector_to_dict(vector, feature_names)

        # Update the model (DenStream doesn't return cluster IDs)
        self.denstream.learn_one(x)

        # Use predict_one to get the cluster
        river_id = self.denstream.predict_one(x)

        # If no cluster found, create a new one
        if river_id is None or river_id < 0:
            adapter_id = self.create_new_cluster(vector, feature_names, "denstream_new_cluster", timestamp)
            is_new_cluster = True
        elif river_id not in self.river_to_adapter_ids:
            # Map River's ID to our internal ID system
            adapter_id = self.create_new_cluster(vector, feature_names, "denstream_new_cluster", timestamp)
            self.river_to_adapter_ids[river_id] = adapter_id
            is_new_cluster = True
        else:
            # Update existing cluster
            adapter_id = self.river_to_adapter_ids[river_id]
            self.update_cluster(adapter_id, vector, feature_names, timestamp)
            is_new_cluster = False

        return adapter_id, is_new_cluster

    def predict(self, vector, feature_names):
        """Find the best cluster for a vector without updating the model"""
        # Convert to dict format for River
        x = self._vector_to_dict(vector, feature_names)

        # Use River's predict_one to get cluster ID
        river_id = self.denstream.predict_one(x)

        # Map to our internal ID system
        if river_id is not None and river_id >= 0 and river_id in self.river_to_adapter_ids:
            return self.river_to_adapter_ids[river_id]

        # Fallback: find nearest cluster
        return self.find_closest_cluster(vector, feature_names)[0]

    def split_cluster(self, cluster_id, timestamp=None):
        """DenStream-specific implementation of cluster splitting"""
        original = self.micro_clusters.get(cluster_id)
        if not original:
            return None
        
        # Create new cluster with perturbation
        perturbation = np.random.normal(0, 0.05, size=original["centroid"].shape)
        new_centroid = original["centroid"] + perturbation
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        new_centroid = np.maximum(new_centroid, 0.0)
        # Create new adapter cluster
        new_id = self.create_new_cluster(
            new_centroid,
            original["feature_names"],
            "variance_split",
            timestamp
        )
        self.variance_dirty.update([cluster_id, new_id])

        # Force River's DenStream to recognize the new cluster
        x = self._vector_to_dict(new_centroid, original["feature_names"])
        self.denstream.learn_one(x)
        
        # Map the new River ID (DenStream doesn't directly expose IDs)
        # We'll assign it when the next prediction happens
        
        # Log the split operation
        self.cluster_history[cluster_id].append({
            "timestamp": timestamp,
            "action": "split",
            "split_into": new_id
        })
        
        return new_id