import numpy as np
from river import cluster
from src.duplimend_framework.cluster_adapter import ClusteringAdapter


class DBStreamAdapter(ClusteringAdapter):
    """
    Adapter for River's DBStream clustering algorithm
    """
    def __init__(self, config=None):
        super().__init__("RiverDBStream", config)
        
        # Default parameters
        self.clustering_threshold = self.config.get("clustering_threshold")
        self.fading_factor = self.config.get("fading_factor")
        
        # Initialize River's DBStream
        self.dbstream = cluster.DBSTREAM(
            clustering_threshold=self.clustering_threshold,
            fading_factor=self.fading_factor
        )
        
        # Map River's internal cluster IDs to our IDs
        self.river_to_adapter_ids = {}
        self.adapter_to_river_ids = {}
        
    def _vector_to_dict(self, vector, feature_names):
        """Convert numpy vector to dict for River's consumption"""
        return {f: float(v) for f, v in zip(feature_names, vector)}
    
    def partial_fit(self, vector, feature_names, timestamp=None):
        """Process a new vector and assign to best cluster or create new one"""
        # Convert to dict format for River
        x = self._vector_to_dict(vector, feature_names)
        
        # Get current clusters before update
        pre_update_clusters = set(self.dbstream.centers.keys())
        
        # Update the model
        self.dbstream.learn_one(x)
        
        # Get clusters after update
        post_update_clusters = set(self.dbstream.centers.keys())
        
        # Find new clusters created during this update
        new_clusters = post_update_clusters - pre_update_clusters
        is_new_cluster = len(new_clusters) > 0
        
        # Find the closest cluster for this vector
        best_distance = float('inf')
        best_river_id = None
        
        for river_id, center in self.dbstream.centers.items():
            river_vector = np.array([center.get(f, 0.0) for f in feature_names])
            distance = self.calculate_distance(vector, river_vector)
            
            if distance < best_distance:
                best_distance = distance
                best_river_id = river_id
        
        if best_river_id is None:
            adapter_id = self.create_new_cluster(vector, feature_names, "river_no_cluster_found", timestamp)
            return adapter_id, True
        
        # Map River's ID to our internal ID system
        if best_river_id not in self.river_to_adapter_ids:
            # Create a new adapter cluster
            adapter_id = self.create_new_cluster(vector, feature_names, 
                                               "river_new_cluster", timestamp)
            self.river_to_adapter_ids[best_river_id] = adapter_id
            self.adapter_to_river_ids[adapter_id] = best_river_id
            
            # Also store the River center in our format
            river_center = self.dbstream.centers[best_river_id]
            river_vector = np.array([river_center.get(f, 0.0) for f in feature_names])
            self.micro_clusters[adapter_id]["centroid"] = river_vector
        else:
            # Use existing adapter ID
            adapter_id = self.river_to_adapter_ids[best_river_id]
            
            # Update our cluster with River's updated center
            river_center = self.dbstream.centers[best_river_id]
            river_vector = np.array([river_center.get(f, 0.0) for f in feature_names])
            self.update_cluster(adapter_id, river_vector, feature_names, timestamp)
        
        # Check for deleted clusters in River
        for river_id in list(self.river_to_adapter_ids.keys()):
            if river_id not in self.dbstream.centers:
                adapter_id = self.river_to_adapter_ids[river_id]
                
                # Record removal in history
                if adapter_id in self.micro_clusters:
                    self.cluster_history[adapter_id].append({
                        "timestamp": timestamp,
                        "action": "river_removed",
                        "final_weight": self.micro_clusters[adapter_id]["weight"]
                    })
                    
                    # Remove the cluster
                    del self.micro_clusters[adapter_id]
                
                # Remove mappings
                del self.river_to_adapter_ids[river_id]
                if adapter_id in self.adapter_to_river_ids:
                    del self.adapter_to_river_ids[adapter_id]
        
        return adapter_id, is_new_cluster
    
    def predict(self, vector, feature_names):
        """Find the best cluster for a vector without updating the model"""
        # Convert to dict format for River
        x = self._vector_to_dict(vector, feature_names)
        
        # Use River's predict_one to get cluster ID
        river_id = self.dbstream.predict_one(x)
        
        # Map to our internal ID system
        if river_id is not None and river_id in self.river_to_adapter_ids:
            return self.river_to_adapter_ids[river_id]
        
        return None
    
    def split_cluster(self, cluster_id, timestamp=None):
        """DBStream-specific implementation of cluster splitting"""
        original = self.micro_clusters.get(cluster_id)
        if not original:
            return None
        
        # Get river ID for this cluster
        river_id = self.adapter_to_river_ids.get(cluster_id)
        if not river_id:
            return super().split_cluster(cluster_id, timestamp)
            
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

        # Force River to create a new cluster by adding this as a new point
        x = self._vector_to_dict(new_centroid, original["feature_names"])
        pre_update = set(self.dbstream.centers.keys())
        self.dbstream.learn_one(x)
        post_update = set(self.dbstream.centers.keys())
        
        # Find the new River cluster
        new_river_ids = post_update - pre_update
        if new_river_ids:
            new_river_id = list(new_river_ids)[0]
            self.river_to_adapter_ids[new_river_id] = new_id
            self.adapter_to_river_ids[new_id] = new_river_id
        
        # Log the split operation
        self.cluster_history[cluster_id].append({
            "timestamp": timestamp,
            "action": "split",
            "split_into": new_id
        })
        
        return new_id   