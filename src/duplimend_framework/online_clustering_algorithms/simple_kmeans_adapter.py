from duplimend_framework.cluster_adapter import ClusteringAdapter


class SimpleKMeansAdapter(ClusteringAdapter):
    """
    Simple K-Means adapter for testing. Creates at most k clusters.
    """
    def __init__(self, config=None):
        super().__init__("SimpleKMeans", config)
        
        self.k = self.config.get("k", 5)
        self.distance_threshold = self.config.get("distance_threshold", 0.5)
        
    def partial_fit(self, vector, feature_names, timestamp=None):
        """Process a new vector and assign to best cluster or create a new one"""
        closest_id, min_distance = self.find_closest_cluster(vector, feature_names)
        
        is_new_cluster = False
        if (closest_id is None or min_distance > self.distance_threshold) and len(self.micro_clusters) < self.k:
            closest_id = self.create_new_cluster(vector, feature_names, "kmeans_new_cluster", timestamp)
            is_new_cluster = True
        elif closest_id is not None:
            self.update_cluster(closest_id, vector, feature_names, timestamp)
            
        return closest_id, is_new_cluster
    
    def predict(self, vector, feature_names):
        """Find the best cluster for a vector without updating the model"""
        closest_id, _ = self.find_closest_cluster(vector, feature_names)
        return closest_id