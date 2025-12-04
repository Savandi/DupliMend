import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class ClusterAwareRegularizer:
    """
    Implements cluster-aware regularization to maintain semantic structure
    during autoencoder training by penalizing cluster collapse and merging.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Regularization weights
        self.separation_weight = self.config.get("separation_weight", 1.0)
        self.compactness_weight = self.config.get("compactness_weight", 0.5)
        self.consistency_weight = self.config.get("consistency_weight", 0.3)
        
        # Thresholds
        self.min_separation_distance = self.config.get("min_separation_distance", 0.1)
        self.max_intra_cluster_variance = self.config.get("max_intra_cluster_variance", 0.05)
        
        # State tracking
        self.previous_centroids = {}
        self.cluster_assignments = defaultdict(list)
        self.regularization_history = defaultdict(list)
        
    def calculate_regularization_loss(self, embeddings: torch.Tensor, 
                                    cluster_assignments: List[int],
                                    activity_label: str,
                                    previous_centroids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate cluster-aware regularization loss.
        
        Args:
            embeddings: Current batch embeddings [batch_size, embedding_dim]
            cluster_assignments: Cluster IDs for each embedding
            activity_label: Activity being processed
            previous_centroids: Previous centroids for consistency loss
            
        Returns:
            regularization_loss: Total regularization loss
            loss_components: Dictionary with individual loss components
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device), {}
        
        # Group embeddings by cluster
        clusters = self._group_embeddings_by_cluster(embeddings, cluster_assignments)
        
        if len(clusters) < 2:
            # Single cluster - only compactness matters
            compactness_loss = self._calculate_compactness_loss(clusters)
            return compactness_loss * self.compactness_weight, {
                'compactness_loss': compactness_loss.item(),
                'separation_loss': 0.0,
                'consistency_loss': 0.0
            }
        
        # Calculate individual loss components
        separation_loss = self._calculate_separation_loss(clusters)
        compactness_loss = self._calculate_compactness_loss(clusters)
        consistency_loss = self._calculate_consistency_loss(
            clusters, activity_label, previous_centroids
        )
        
        # Combine losses
        total_loss = (
            self.separation_weight * separation_loss +
            self.compactness_weight * compactness_loss +
            self.consistency_weight * consistency_loss
        )
        
        # Update state
        self._update_regularization_state(clusters, activity_label)
        
        loss_components = {
            'separation_loss': separation_loss.item(),
            'compactness_loss': compactness_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_regularization_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _group_embeddings_by_cluster(self, embeddings: torch.Tensor, 
                                   cluster_assignments: List[int]) -> Dict[int, torch.Tensor]:
        """Group embeddings by their cluster assignments"""
        clusters = defaultdict(list)
        
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id >= 0:  # Valid cluster assignment
                clusters[cluster_id].append(embeddings[i])
        
        # Convert lists to tensors
        cluster_tensors = {}
        for cluster_id, embedding_list in clusters.items():
            if embedding_list:
                cluster_tensors[cluster_id] = torch.stack(embedding_list)
        
        return cluster_tensors
    
    def _calculate_separation_loss(self, clusters: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss to maintain separation between clusters.
        Penalizes clusters that are too close together.
        """
        if len(clusters) < 2:
            return torch.tensor(0.0)
        
        # Calculate centroids
        centroids = {}
        for cluster_id, cluster_embeddings in clusters.items():
            centroids[cluster_id] = torch.mean(cluster_embeddings, dim=0)
        
        # Calculate pairwise distances between centroids
        cluster_ids = list(centroids.keys())
        separation_loss = torch.tensor(0.0, device=list(centroids.values())[0].device)
        pair_count = 0
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                centroid_i = centroids[cluster_ids[i]]
                centroid_j = centroids[cluster_ids[j]]
                
                # Euclidean distance between centroids
                distance = torch.norm(centroid_i - centroid_j, p=2)
                
                # Penalize if distance is below threshold
                if distance < self.min_separation_distance:
                    # Penalty increases as distance decreases
                    penalty = torch.exp(-distance / self.min_separation_distance)
                    separation_loss += penalty
                
                pair_count += 1
        
        return separation_loss / max(pair_count, 1)
    
    def _calculate_compactness_loss(self, clusters: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Calculate loss to maintain cluster compactness.
        Penalizes clusters with high intra-cluster variance.
        """
        if not clusters:
            return torch.tensor(0.0)
        
        compactness_loss = torch.tensor(0.0)
        device = list(clusters.values())[0].device
        compactness_loss = compactness_loss.to(device)
        
        for cluster_id, cluster_embeddings in clusters.items():
            if cluster_embeddings.size(0) > 1:
                # Calculate centroid
                centroid = torch.mean(cluster_embeddings, dim=0)
                
                # Calculate variance (mean squared distance from centroid)
                distances = torch.norm(cluster_embeddings - centroid.unsqueeze(0), p=2, dim=1)
                variance = torch.var(distances)
                
                # Penalize high variance
                if variance > self.max_intra_cluster_variance:
                    penalty = variance - self.max_intra_cluster_variance
                    compactness_loss += penalty
        
        return compactness_loss / len(clusters)
    
    def _calculate_consistency_loss(self, clusters: Dict[int, torch.Tensor],
                                  activity_label: str,
                                  previous_centroids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate consistency loss to prevent drastic centroid movements.
        Maintains continuity in the embedding space.
        """
        if activity_label not in self.previous_centroids or not clusters:
            return torch.tensor(0.0)
        
        prev_centroids = self.previous_centroids[activity_label]
        if prev_centroids is None or len(prev_centroids) == 0:
            return torch.tensor(0.0)
        
        # Calculate current centroids
        current_centroids = {}
        for cluster_id, cluster_embeddings in clusters.items():
            current_centroids[cluster_id] = torch.mean(cluster_embeddings, dim=0)
        
        consistency_loss = torch.tensor(0.0)
        if current_centroids:
            device = list(current_centroids.values())[0].device
            consistency_loss = consistency_loss.to(device)
        
        # Compare with previous centroids
        matched_clusters = 0
        for cluster_id in current_centroids:
            if cluster_id in prev_centroids:
                current_centroid = current_centroids[cluster_id]
                previous_centroid = prev_centroids[cluster_id]
                
                # Ensure tensors are on same device
                if previous_centroid.device != current_centroid.device:
                    previous_centroid = previous_centroid.to(current_centroid.device)
                
                # Calculate movement distance
                movement = torch.norm(current_centroid - previous_centroid, p=2)
                consistency_loss += movement
                matched_clusters += 1
        
        return consistency_loss / max(matched_clusters, 1)
    
    def _update_regularization_state(self, clusters: Dict[int, torch.Tensor], activity_label: str):
        """Update internal state for tracking"""
        # Update previous centroids
        current_centroids = {}
        for cluster_id, cluster_embeddings in clusters.items():
            current_centroids[cluster_id] = torch.mean(cluster_embeddings, dim=0).detach()
        
        self.previous_centroids[activity_label] = current_centroids
        
        # Track regularization history
        self.regularization_history[activity_label].append({
            'num_clusters': len(clusters),
            'cluster_sizes': {cid: embs.size(0) for cid, embs in clusters.items()}
        })
    
    def detect_cluster_quality_degradation(self, clusters: Dict[int, torch.Tensor],
                                         activity_label: str) -> Dict:
        """
        Detect if clustering quality has degraded significantly.
        Returns metrics and recommendations.
        """
        if len(clusters) < 2:
            return {
                'degradation_detected': False,
                'reason': 'insufficient_clusters',
                'metrics': {}
            }
        
        # Calculate current metrics
        metrics = self._calculate_cluster_quality_metrics(clusters)
        
        # Compare with historical performance
        history = self.regularization_history.get(activity_label, [])
        if len(history) < 10:  # Need sufficient history
            return {
                'degradation_detected': False,
                'reason': 'insufficient_history',
                'metrics': metrics
            }
        
        # Analyze trends
        degradation_signals = []
        
        # Check for cluster collapse (too many small clusters or too few clusters)
        recent_cluster_counts = [h['num_clusters'] for h in history[-5:]]
        avg_recent_clusters = np.mean(recent_cluster_counts)
        
        if metrics['num_clusters'] < avg_recent_clusters * 0.7:
            degradation_signals.append('cluster_collapse')
        
        # Check separation quality
        if metrics['min_inter_cluster_distance'] < self.min_separation_distance:
            degradation_signals.append('poor_separation')
        
        # Check compactness
        if metrics['max_intra_cluster_variance'] > self.max_intra_cluster_variance:
            degradation_signals.append('poor_compactness')
        
        degradation_detected = len(degradation_signals) >= 2
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_signals': degradation_signals,
            'metrics': metrics,
            'recommendation': 'retrain_autoencoder' if degradation_detected else 'continue_monitoring'
        }
    
    def _calculate_cluster_quality_metrics(self, clusters: Dict[int, torch.Tensor]) -> Dict:
        """Calculate comprehensive cluster quality metrics"""
        if not clusters:
            return {}
        
        # Calculate centroids
        centroids = {}
        for cluster_id, cluster_embeddings in clusters.items():
            centroids[cluster_id] = torch.mean(cluster_embeddings, dim=0)
        
        # Inter-cluster distances
        cluster_ids = list(centroids.keys())
        inter_distances = []
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                dist = torch.norm(centroids[cluster_ids[i]] - centroids[cluster_ids[j]], p=2)
                inter_distances.append(dist.item())
        
        # Intra-cluster variances
        intra_variances = []
        cluster_sizes = []
        
        for cluster_id, cluster_embeddings in clusters.items():
            cluster_sizes.append(cluster_embeddings.size(0))
            
            if cluster_embeddings.size(0) > 1:
                centroid = centroids[cluster_id]
                distances = torch.norm(cluster_embeddings - centroid.unsqueeze(0), p=2, dim=1)
                variance = torch.var(distances).item()
                intra_variances.append(variance)
        
        return {
            'num_clusters': len(clusters),
            'cluster_sizes': cluster_sizes,
            'min_inter_cluster_distance': min(inter_distances) if inter_distances else 0.0,
            'avg_inter_cluster_distance': np.mean(inter_distances) if inter_distances else 0.0,
            'max_intra_cluster_variance': max(intra_variances) if intra_variances else 0.0,
            'avg_intra_cluster_variance': np.mean(intra_variances) if intra_variances else 0.0,
            'cluster_balance': np.std(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes else 0.0
        }
    
    def get_regularization_summary(self, activity_label: str) -> Dict:
        """Get summary of regularization state for an activity"""
        return {
            'activity': activity_label,
            'has_previous_centroids': activity_label in self.previous_centroids,
            'num_previous_centroids': len(self.previous_centroids.get(activity_label, {})),
            'history_length': len(self.regularization_history.get(activity_label, [])),
            'recent_metrics': self.regularization_history.get(activity_label, [])[-1:] if activity_label in self.regularization_history else None
        }