import numpy as np
from river import cluster
from sklearn.cluster import KMeans
from src.duplimend_framework.cluster_adapter import ClusteringAdapter


class CluStreamAdapter(ClusteringAdapter):
    """
    Adapter for River's CluStream clustering algorithm
    """
    def __init__(self, config=None):
        super().__init__("RiverCluStream", config)
        
        # Default parameters
        self.n_macro_clusters = self.config.get("n_macro_clusters")
        self.max_micro_clusters = self.config.get("max_micro_clusters")
        self.time_gap = self.config.get("time_gap")
        self.halflife = self.config.get("halflife")
        self.split_perturbation = self.config.get("split_perturbation", 0.05)
        
        # Initialize River's CluStream
        self.clustream = cluster.CluStream(
            n_macro_clusters=self.n_macro_clusters,
            max_micro_clusters=self.max_micro_clusters,
            time_gap=self.time_gap,
            halflife=self.halflife
        )
        print(f"[DEBUG] CluStream parameters: n_macro_clusters={self.n_macro_clusters}, max_micro_clusters={self.max_micro_clusters}, time_gap={self.time_gap}, halflife={self.halflife}")
        
        # Map River's internal cluster IDs to our IDs
        self.river_to_adapter_ids = {}
        self.micro_clusters = {}
        self.feature_names = None

    def _vector_to_dict(self, vector, feature_names):
        """Convert numpy vector to dict for River's consumption"""
        return {i: float(v) for i, v in enumerate(vector)}

    def _update_micro_clusters(self):
        """
        Synchronize River's micro-clusters with the adapter's clusters.
        Ensures every River micro-cluster has a corresponding adapter cluster.
        """
        # Get River's micro-clusters (assume self.clustream.microclusters is a dict)
        river_micro_clusters = getattr(self.clustream, "microclusters", {})
        for river_id, mc in river_micro_clusters.items():
            # Defensive: skip if missing attributes
            if not hasattr(mc, "center") or not hasattr(mc, "n"):
                continue
            # Update or add micro-cluster info
            self.micro_clusters[river_id] = {
                "centroid": np.array(mc.center),
                "weight": mc.n,
                "samples": mc.n,
                "feature_names": self.feature_names
            }
            # Ensure mapping exists
            if river_id not in self.river_to_adapter_ids:
                adapter_id = self.create_new_cluster(
                    np.array(mc.center),
                    self.feature_names,
                    reason="clustream_internal",
                    timestamp=None
                )
                self.river_to_adapter_ids[river_id] = adapter_id
                print(f"[DEBUG] Created adapter cluster {adapter_id} for River ID {river_id}")
    
    def partial_fit(self, vector, feature_names, timestamp=None):
        """
        Process a new feature vector and assign to best cluster or create a new one.
        Returns: (adapter_cluster_id, is_new_cluster)
        """
        self.feature_names = feature_names
        x = self._vector_to_dict(vector, feature_names)
        self.clustream.learn_one(x)
        self._update_micro_clusters()

        river_id = self.clustream.predict_one(x)
        if river_id is None:
            # Fallback: assign to latest micro-cluster if available
            river_id = max(self.micro_clusters.keys()) if self.micro_clusters else 0

        adapter_id = self.river_to_adapter_ids.get(river_id)
        if adapter_id is None:
            # Should not happen if _update_micro_clusters is correct, but fallback
            adapter_id = self.create_new_cluster(vector, feature_names, reason="clustream_fallback", timestamp=timestamp)
            self.river_to_adapter_ids[river_id] = adapter_id
            is_new_cluster = True
        else:
            self.update_cluster(adapter_id, vector, feature_names, timestamp)
            is_new_cluster = False

        print(f"[DEBUG] Assigned event to cluster: {adapter_id}, is_new: {is_new_cluster}")
        print(f"[DEBUG] Current clusters: {list(self.micro_clusters.keys())}")
        return adapter_id, is_new_cluster
    
    def get_significant_clusters(self, min_weight):
        # Return adapter IDs, not River IDs
        print(f"[DEBUG] get_significant_clusters: river_to_adapter_ids={self.river_to_adapter_ids}, micro_clusters={self.micro_clusters}")
        return [
            self.river_to_adapter_ids[river_id]
            for river_id, info in self.micro_clusters.items()
            if river_id in self.river_to_adapter_ids and info["weight"] >= min_weight
        ]

    def get_cluster_centroid(self, cluster_id):
        # Find the River ID for this adapter ID
        for river_id, adapter_id in self.river_to_adapter_ids.items():
            if adapter_id == cluster_id:
                return self.micro_clusters[river_id]["centroid"]
        return None

    def get_macro_clusters(self, n_macro_clusters=None, min_cluster_weight=None):
        n_macro_clusters = n_macro_clusters or self.n_macro_clusters
        min_cluster_weight = min_cluster_weight or 0  # Default to 0 if not set
        centroids = []
        micro_ids = []
        for cid, info in self.micro_clusters.items():
            if info["weight"] >= min_cluster_weight:
                centroids.append(info["centroid"])
                micro_ids.append(cid)
        if len(centroids) < n_macro_clusters or len(centroids) == 0:
            return {i: [cid] for i, cid in enumerate(micro_ids)}
        kmeans = KMeans(n_clusters=n_macro_clusters, random_state=0)
        labels = kmeans.fit_predict(np.array(centroids))
        kmeans.cluster_centers_ = np.maximum(kmeans.cluster_centers_, 0.0)  
        macro_clusters = {}
        for macro_id in range(n_macro_clusters):
            macro_clusters[macro_id] = [micro_ids[i] for i, label in enumerate(labels) if label == macro_id]
        return macro_clusters

    def predict(self, vector, feature_names):
        """Find the best cluster for a vector without updating the model"""
        x = self._vector_to_dict(vector, feature_names)
        river_id = self.clustream.predict_one(x)
        if river_id is not None and river_id in self.river_to_adapter_ids:
            return self.river_to_adapter_ids[river_id]
        return self.find_closest_cluster(vector, feature_names)[0]

    def split_cluster(self, cluster_id, timestamp=None):
        """CluStream-specific implementation of cluster splitting"""
        original = self.micro_clusters.get(cluster_id)
        if not original:
            return None

        perturbation_std = self.split_perturbation
        perturbation = np.random.normal(0, perturbation_std, size=original["centroid"].shape)
        new_centroid = original["centroid"] + perturbation
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        new_centroid = np.maximum(new_centroid, 0.0)

        new_id = self.create_new_cluster(
            new_centroid,
            original["feature_names"],
            "variance_split",
            timestamp
        )
        self.variance_dirty.update([cluster_id, new_id])

        x = self._vector_to_dict(new_centroid, original["feature_names"])
        self.clustream.learn_one(x)

        river_id = self.clustream.predict_one(x)
        if river_id is not None and river_id not in self.river_to_adapter_ids:
            self.river_to_adapter_ids[river_id] = new_id

        self.cluster_history[cluster_id].append({
            "timestamp": timestamp,
            "action": "split",
            "split_into": new_id
        })

        return new_id