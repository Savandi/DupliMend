import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.duplimend_framework.streaming_sparse_denoising_autoencoder import SparseDenoisingAutoencoder
from src.duplimend_framework.drift_retraining.cluster_aware_regularizer import ClusterAwareRegularizer
from src.duplimend_framework.drift_retraining.centroid_memory_manager import CentroidMemoryManager

class ClusterAwareAutoencoder(SparseDenoisingAutoencoder):
    """
    Enhanced autoencoder with cluster-aware regularization and centroid memory
    for catastrophic forgetting protection.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, sparsity_lambda=0.001, 
                 noise_std=0.1, cluster_regularization_config=None, memory_config=None):
        super().__init__(input_dim, latent_dim, hidden_dims, sparsity_lambda, noise_std)
        
        if cluster_regularization_config is not None:
            self.cluster_regularizer = ClusterAwareRegularizer(cluster_regularization_config)
            self.enable_cluster_regularization = True
            self.cluster_reg_weight = cluster_regularization_config.get("cluster_reg_weight", 0.1)
        else:
            self.cluster_regularizer = None
            self.enable_cluster_regularization = False
            self.cluster_reg_weight = 0.0
        
        if memory_config is not None:
            self.memory_manager = CentroidMemoryManager(memory_config)
            self.enable_memory_regularization = memory_config.get("enable_memory_regularization", True)
        else:
            self.memory_manager = None
            self.enable_memory_regularization = False
        
        self.current_activity = None
        self.training_step = 0

        
    def forward_with_clusters(self, x: torch.Tensor, cluster_assignments: Optional[List[int]] = None,
                                activity_label: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with cluster-aware regularization and memory protection.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            cluster_assignments: Cluster IDs for each sample
            activity_label: Current activity being processed
            
        Returns:
            x_reconstructed: Reconstructed input
            z: Latent embeddings
            losses: Dictionary containing all loss components
        """
        x_reconstructed, z = self.forward(x)
        
        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        
        losses = {
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'total_loss': reconstruction_loss + sparsity_loss
        }
        
        if (self.enable_cluster_regularization and
            self.cluster_regularizer is not None and
            cluster_assignments is not None and
            activity_label is not None and
            len(set(cluster_assignments)) > 1):
            
            cluster_reg_loss, cluster_components = self.cluster_regularizer.calculate_regularization_loss(
                z, cluster_assignments, activity_label
            )
            
            losses.update(cluster_components)
            losses['cluster_regularization_loss'] = cluster_reg_loss
            losses['total_loss'] = losses['total_loss'] + self.cluster_reg_weight * cluster_reg_loss
        
        if (self.enable_memory_regularization and
            self.memory_manager is not None and
            cluster_assignments is not None and
            activity_label is not None):
            
            memory_loss, memory_components = self.memory_manager.get_memory_regularization_loss(
                activity_label, z, cluster_assignments
            )
            
            losses.update(memory_components)
            losses['memory_regularization_loss'] = memory_loss
            losses['total_loss'] = losses['total_loss'] + memory_loss
        
        return x_reconstructed, z, losses
    
    def enable_enhanced_regularization(self, cluster_config=None, memory_config=None):
        """Enable regularization after warmup phase"""
        if cluster_config and self.cluster_regularizer is None:
            self.cluster_regularizer = ClusterAwareRegularizer(cluster_config)
            self.cluster_reg_weight = cluster_config.get("cluster_reg_weight", 0.1)
        
        if memory_config and self.memory_manager is None:
            self.memory_manager = CentroidMemoryManager(memory_config)
        
        self.enable_cluster_regularization = True
        self.enable_memory_regularization = True

    def update_memory_from_clusters(self, activity_label: str, z: torch.Tensor, 
                                   cluster_assignments: List[int], timestamp: int):
        """
        Update centroid memory based on current clusters.
        Call this during or after training to build memory.
        
        Args:
            activity_label: Current activity
            z: Latent embeddings from autoencoder
            cluster_assignments: Cluster assignments for embeddings
            timestamp: Current timestamp
        """
        if not cluster_assignments or z.size(0) == 0:
            return
        
        clusters = {}
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id >= 0:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(z[i].detach().cpu().numpy())
        
        for cluster_id, embeddings in clusters.items():
            if embeddings:
                self.memory_manager.update_cluster_memory(
                    activity_label, cluster_id, embeddings, timestamp
                )
    
    def check_cluster_quality(self, z: torch.Tensor, cluster_assignments: List[int],
                            activity_label: str) -> Dict:
        """Check if cluster quality has degraded and retraining is needed"""
        if not cluster_assignments or len(set(cluster_assignments)) < 2:
            return {'quality_ok': True, 'reason': 'insufficient_clusters'}
        
        clusters = {}
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id >= 0:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(z[i])
        
        cluster_tensors = {}
        for cluster_id, embedding_list in clusters.items():
            if embedding_list:
                cluster_tensors[cluster_id] = torch.stack(embedding_list)
        
        degradation_result = self.cluster_regularizer.detect_cluster_quality_degradation(
            cluster_tensors, activity_label
        )
        
        return {
            'quality_ok': not degradation_result['degradation_detected'],
            'degradation_signals': degradation_result.get('degradation_signals', []),
            'metrics': degradation_result.get('metrics', {}),
            'recommendation': degradation_result.get('recommendation', 'continue')
        }
    
    def update_training_state(self, activity_label: str):
        """Update internal training state"""
        self.current_activity = activity_label
        self.training_step += 1
    
    def save_memory_state(self, filepath: str):
        """Save memory state for persistence"""
        self.memory_manager.save_memory_state(filepath)
    
    def load_memory_state(self, filepath: str):
        """Load memory state from file"""
        self.memory_manager.load_memory_state(filepath)
    
    def get_regularization_summary(self) -> Dict:
        """Get summary of regularization state"""
        summary = {}
        
        if self.current_activity:
            summary['cluster_regularization'] = self.cluster_regularizer.get_regularization_summary(self.current_activity)
        
        summary['memory_regularization'] = self.memory_manager.get_memory_summary()
        
        return summary
    
    def enable_forgetting_protection(self, enable: bool = True):
        """Enable or disable catastrophic forgetting protection"""
        self.enable_memory_regularization = enable
    
    def get_memory_statistics(self) -> Dict:
        """Get detailed memory usage statistics"""
        return self.memory_manager.get_memory_summary()