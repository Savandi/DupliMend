import numpy as np
import torch
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class CentroidMemoryManager:
    """
    Manages centroid/exemplar memory for catastrophic forgetting protection.
    Stores representative embeddings from stable clusters to regularize during retraining.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Memory management parameters
        self.max_centroids_per_activity = self.config.get("max_centroids_per_activity", 50)
        self.centroid_stability_threshold = self.config.get("centroid_stability_threshold", 10)  # Min events to be stable
        self.memory_decay_factor = self.config.get("memory_decay_factor", 0.95)  # Decay unused centroids
        self.exemplar_count = self.config.get("exemplar_count", 3)  # Additional exemplars per cluster
        
        # Regularization parameters
        self.memory_regularization_weight = self.config.get("memory_regularization_weight", 0.1)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)  # For matching clusters
        
        # Storage
        self.cluster_centroids = defaultdict(dict)  # {activity: {cluster_id: centroid_info}}
        self.cluster_exemplars = defaultdict(dict)  # {activity: {cluster_id: [exemplar_embeddings]}}
        self.cluster_stats = defaultdict(dict)      # {activity: {cluster_id: {count, last_seen, confidence}}}
        self.memory_usage_stats = defaultdict(int)
        
        # Tracking
        self.last_cleanup = 0
        self.memory_updates = 0
        
    def update_cluster_memory(self, activity_label: str, cluster_id: int, 
                            embeddings: List[np.ndarray], timestamp: int):
        """
        Update centroid memory for a specific cluster.
        
        Args:
            activity_label: Activity this cluster belongs to
            cluster_id: Cluster identifier
            embeddings: List of embeddings assigned to this cluster
            timestamp: Current timestamp
        """
        if not embeddings:
            return
        
        # Ensure we have the activity in our memory
        if activity_label not in self.cluster_centroids:
            self.cluster_centroids[activity_label] = {}
            self.cluster_exemplars[activity_label] = {}
            self.cluster_stats[activity_label] = {}
        
        # Calculate new centroid
        embeddings_array = np.array(embeddings)
        new_centroid = np.mean(embeddings_array, axis=0)
        
        # Update or create cluster memory
        if cluster_id in self.cluster_centroids[activity_label]:
            # Update existing centroid with moving average
            existing_info = self.cluster_centroids[activity_label][cluster_id]
            existing_centroid = existing_info['centroid']
            existing_count = self.cluster_stats[activity_label][cluster_id]['count']
            
            # Weighted update: give more weight to established centroids
            alpha = min(len(embeddings) / (existing_count + len(embeddings)), 0.3)
            updated_centroid = (1 - alpha) * existing_centroid + alpha * new_centroid
            
            self.cluster_centroids[activity_label][cluster_id] = {
                'centroid': updated_centroid,
                'created_at': existing_info['created_at'],
                'updated_at': timestamp,
                'version': existing_info['version'] + 1
            }
            
            # Update stats
            self.cluster_stats[activity_label][cluster_id]['count'] += len(embeddings)
            self.cluster_stats[activity_label][cluster_id]['last_seen'] = timestamp
            self.cluster_stats[activity_label][cluster_id]['confidence'] = min(
                self.cluster_stats[activity_label][cluster_id]['confidence'] + 0.1, 1.0
            )
            
        else:
            # Create new cluster memory
            self.cluster_centroids[activity_label][cluster_id] = {
                'centroid': new_centroid,
                'created_at': timestamp,
                'updated_at': timestamp,
                'version': 1
            }
            
            self.cluster_stats[activity_label][cluster_id] = {
                'count': len(embeddings),
                'last_seen': timestamp,
                'confidence': 0.5  # Start with medium confidence
            }
        
        # Update exemplars (store diverse representative points)
        self._update_exemplars(activity_label, cluster_id, embeddings_array, new_centroid)
        
        # Update memory usage tracking
        self.memory_usage_stats[activity_label] = len(self.cluster_centroids[activity_label])
        self.memory_updates += 1
        
        # Periodic cleanup
        if self.memory_updates % 100 == 0:
            self._cleanup_memory()
    
    def _update_exemplars(self, activity_label: str, cluster_id: int, 
                         embeddings: np.ndarray, centroid: np.ndarray):
        """Update exemplar embeddings for a cluster"""
        if cluster_id not in self.cluster_exemplars[activity_label]:
            self.cluster_exemplars[activity_label][cluster_id] = []
        
        current_exemplars = self.cluster_exemplars[activity_label][cluster_id]
        
        # Find diverse points: closest to centroid, furthest from centroid, and median distance
        distances = np.linalg.norm(embeddings - centroid.reshape(1, -1), axis=1)
        
        if len(embeddings) >= self.exemplar_count:
            # Closest point (most representative)
            closest_idx = np.argmin(distances)
            # Furthest point (boundary case)
            furthest_idx = np.argmax(distances)
            # Median distance point (typical case)
            median_idx = np.argsort(distances)[len(distances) // 2]
            
            new_exemplars = [
                embeddings[closest_idx],
                embeddings[furthest_idx],
                embeddings[median_idx]
            ]
        else:
            # Use all available points
            new_exemplars = [emb for emb in embeddings]
        
        # Merge with existing exemplars and keep only the best ones
        all_exemplars = current_exemplars + new_exemplars
        if len(all_exemplars) > self.exemplar_count:
            # Keep the most diverse set
            exemplar_array = np.array(all_exemplars)
            distances_to_centroid = np.linalg.norm(exemplar_array - centroid.reshape(1, -1), axis=1)
            
            # Sort by distance and pick evenly spaced ones
            sorted_indices = np.argsort(distances_to_centroid)
            step = len(sorted_indices) // self.exemplar_count
            selected_indices = sorted_indices[::max(step, 1)][:self.exemplar_count]
            
            self.cluster_exemplars[activity_label][cluster_id] = [
                all_exemplars[i] for i in selected_indices
            ]
        else:
            self.cluster_exemplars[activity_label][cluster_id] = all_exemplars
    
    def get_memory_regularization_loss(self, activity_label: str, 
                                     embeddings: torch.Tensor,
                                     cluster_assignments: List[int]) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate memory regularization loss to prevent catastrophic forgetting.
        
        Args:
            activity_label: Current activity being processed
            embeddings: Current batch embeddings [batch_size, embedding_dim]
            cluster_assignments: Cluster IDs for each embedding
            
        Returns:
            memory_loss: Memory regularization loss
            loss_info: Dictionary with loss components and statistics
        """
        if activity_label not in self.cluster_centroids or embeddings.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device), {}
        
        stored_centroids = self.cluster_centroids[activity_label]
        stored_stats = self.cluster_stats[activity_label]
        
        memory_loss = torch.tensor(0.0, device=embeddings.device)
        matched_points = 0
        total_points = 0
        cluster_matches = defaultdict(int)
        
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id < 0:  # Invalid cluster assignment
                continue
                
            total_points += 1
            current_embedding = embeddings[i]
            
            # Find matching stored cluster
            matching_centroid = self._find_matching_centroid(
                current_embedding, cluster_id, stored_centroids, stored_stats
            )
            
            if matching_centroid is not None:
                # Apply regularization: pull current embedding toward stored centroid
                centroid_tensor = torch.tensor(matching_centroid, device=embeddings.device, dtype=embeddings.dtype)
                point_loss = torch.norm(current_embedding - centroid_tensor, p=2) ** 2
                
                # Weight by centroid confidence
                confidence = stored_stats[cluster_id]['confidence'] if cluster_id in stored_stats else 0.5
                weighted_loss = confidence * point_loss
                
                memory_loss += weighted_loss
                matched_points += 1
                cluster_matches[cluster_id] += 1
        
        # Normalize by matched points
        if matched_points > 0:
            memory_loss = memory_loss / matched_points
        
        # Apply global weight
        memory_loss = self.memory_regularization_weight * memory_loss
        
        loss_info = {
            'memory_regularization_loss': memory_loss.item(),
            'matched_points': matched_points,
            'total_points': total_points,
            'match_ratio': matched_points / max(total_points, 1),
            'cluster_matches': dict(cluster_matches),
            'stored_clusters': len(stored_centroids)
        }
        
        return memory_loss, loss_info
    
    def _find_matching_centroid(self, embedding: torch.Tensor, cluster_id: int,
                               stored_centroids: Dict, stored_stats: Dict) -> Optional[np.ndarray]:
        """
        Find the best matching stored centroid for an embedding.
        
        Args:
            embedding: Current embedding
            cluster_id: Assigned cluster ID
            stored_centroids: Dictionary of stored centroids
            stored_stats: Dictionary of cluster statistics
            
        Returns:
            Matching centroid or None if no good match found
        """
        # First, try exact cluster ID match
        if cluster_id in stored_centroids:
            cluster_info = stored_centroids[cluster_id]
            stored_centroid = cluster_info['centroid']
            
            # Check if this centroid is stable enough
            if cluster_id in stored_stats:
                stats = stored_stats[cluster_id]
                if stats['count'] >= self.centroid_stability_threshold:
                    return stored_centroid
        
        # If no exact match or unstable, try distance-based matching
        embedding_np = embedding.detach().cpu().numpy()
        best_centroid = None
        best_distance = float('inf')

        # Get distance metric from config (default to cosine)
        distance_metric = self.config.get("distance_metric", "cosine").lower()

        # Convert similarity_threshold to distance_threshold
        # For cosine: similarity 0.7 ≈ distance 0.3
        # For euclidean (normalized): similar conversion applies
        distance_threshold = 1.0 - self.similarity_threshold

        for cid, cluster_info in stored_centroids.items():
            if cid in stored_stats:
                stats = stored_stats[cid]

                # Only consider stable clusters
                if stats['count'] >= self.centroid_stability_threshold:
                    stored_centroid = cluster_info['centroid']

                    # Calculate distance based on metric
                    distance = self._calculate_distance(embedding_np, stored_centroid, distance_metric)

                    if distance < best_distance and distance < distance_threshold:
                        best_distance = distance
                        best_centroid = stored_centroid

        return best_centroid

    def _calculate_distance(self, vector1, vector2, distance_metric):
        """Calculate distance between two vectors based on configured metric."""
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        if distance_metric == "euclidean":
            # Normalize vectors to unit length for bounded range [0, 2]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 2.0

            v1_normalized = v1 / norm1
            v2_normalized = v2 / norm2
            return np.linalg.norm(v1_normalized - v2_normalized)

        elif distance_metric == "manhattan":
            return np.sum(np.abs(v1 - v2))

        else:  # cosine (default)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            cosine_similarity = np.dot(v1, v2) / (norm1 * norm2)
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            return 1.0 - cosine_similarity
    
    def _cleanup_memory(self):
        """Clean up old or unused centroids to manage memory"""
        print(f"[MEMORY-CLEANUP] Starting memory cleanup...")
        
        total_removed = 0
        
        for activity_label in list(self.cluster_centroids.keys()):
            activity_centroids = self.cluster_centroids[activity_label]
            activity_stats = self.cluster_stats[activity_label]
            
            clusters_to_remove = []
            
            # Find clusters to remove based on criteria
            for cluster_id in list(activity_centroids.keys()):
                if cluster_id in activity_stats:
                    stats = activity_stats[cluster_id]
                    
                    # Remove if:
                    # 1. Very low confidence and few samples
                    # 2. Not seen recently (implement time-based decay if needed)
                    # 3. Activity has too many clusters
                    
                    should_remove = False
                    
                    # Low confidence + few samples
                    if stats['confidence'] < 0.3 and stats['count'] < self.centroid_stability_threshold:
                        should_remove = True
                    
                    # Too many clusters - remove least confident ones
                    elif len(activity_centroids) > self.max_centroids_per_activity:
                        # This will be handled separately by sorting
                        pass
                    
                    if should_remove:
                        clusters_to_remove.append(cluster_id)
            
            # Handle too many clusters - keep most confident ones
            if len(activity_centroids) > self.max_centroids_per_activity:
                # Sort by confidence and count
                cluster_scores = []
                for cluster_id in activity_centroids.keys():
                    if cluster_id in activity_stats:
                        stats = activity_stats[cluster_id]
                        score = stats['confidence'] * np.log(1 + stats['count'])
                        cluster_scores.append((score, cluster_id))
                
                cluster_scores.sort(reverse=True)  # Highest score first
                clusters_to_keep = set(cid for _, cid in cluster_scores[:self.max_centroids_per_activity])
                
                for cluster_id in list(activity_centroids.keys()):
                    if cluster_id not in clusters_to_keep:
                        clusters_to_remove.append(cluster_id)
            
            # Remove selected clusters
            for cluster_id in clusters_to_remove:
                if cluster_id in activity_centroids:
                    del activity_centroids[cluster_id]
                if cluster_id in activity_stats:
                    del activity_stats[cluster_id]
                if cluster_id in self.cluster_exemplars.get(activity_label, {}):
                    del self.cluster_exemplars[activity_label][cluster_id]
                
                total_removed += 1
            
            # Update memory usage stats
            self.memory_usage_stats[activity_label] = len(activity_centroids)
        
        self.last_cleanup = self.memory_updates
        print(f"[MEMORY-CLEANUP] Removed {total_removed} outdated clusters")
        print(f"[MEMORY-CLEANUP] Current memory usage: {dict(self.memory_usage_stats)}")
    
    def save_memory_state(self, filepath: str):
        """Save centroid memory state for persistence"""
        memory_state = {
            'cluster_centroids': {},
            'cluster_exemplars': {},
            'cluster_stats': dict(self.cluster_stats),
            'memory_usage_stats': dict(self.memory_usage_stats),
            'config': self.config,
            'last_cleanup': self.last_cleanup,
            'memory_updates': self.memory_updates,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for activity_label, clusters in self.cluster_centroids.items():
            memory_state['cluster_centroids'][activity_label] = {}
            for cluster_id, cluster_info in clusters.items():
                memory_state['cluster_centroids'][activity_label][str(cluster_id)] = {
                    'centroid': cluster_info['centroid'].tolist(),
                    'created_at': cluster_info['created_at'],
                    'updated_at': cluster_info['updated_at'],
                    'version': cluster_info['version']
                }
        
        for activity_label, clusters in self.cluster_exemplars.items():
            memory_state['cluster_exemplars'][activity_label] = {}
            for cluster_id, exemplars in clusters.items():
                memory_state['cluster_exemplars'][activity_label][str(cluster_id)] = [
                    exemplar.tolist() for exemplar in exemplars
                ]
        
        with open(filepath, 'w') as f:
            json.dump(memory_state, f, indent=2)
        
        print(f"[MEMORY] Saved centroid memory state to: {filepath}")
    
    def load_memory_state(self, filepath: str):
        """Load centroid memory state from file"""
        if not os.path.exists(filepath):
            print(f"[MEMORY] No memory state file found at: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            memory_state = json.load(f)
        
        # Restore configuration
        self.config.update(memory_state.get('config', {}))
        self.last_cleanup = memory_state.get('last_cleanup', 0)
        self.memory_updates = memory_state.get('memory_updates', 0)
        self.memory_usage_stats = defaultdict(int, memory_state.get('memory_usage_stats', {}))
        self.cluster_stats = defaultdict(dict, memory_state.get('cluster_stats', {}))
        
        # Restore centroids
        for activity_label, clusters in memory_state.get('cluster_centroids', {}).items():
            self.cluster_centroids[activity_label] = {}
            for cluster_id_str, cluster_info in clusters.items():
                cluster_id = int(cluster_id_str)
                self.cluster_centroids[activity_label][cluster_id] = {
                    'centroid': np.array(cluster_info['centroid']),
                    'created_at': cluster_info['created_at'],
                    'updated_at': cluster_info['updated_at'],
                    'version': cluster_info['version']
                }
        
        # Restore exemplars
        for activity_label, clusters in memory_state.get('cluster_exemplars', {}).items():
            self.cluster_exemplars[activity_label] = {}
            for cluster_id_str, exemplars in clusters.items():
                cluster_id = int(cluster_id_str)
                self.cluster_exemplars[activity_label][cluster_id] = [
                    np.array(exemplar) for exemplar in exemplars
                ]
        
        total_centroids = sum(len(clusters) for clusters in self.cluster_centroids.values())
        print(f"[MEMORY] Loaded centroid memory state: {total_centroids} centroids across {len(self.cluster_centroids)} activities")
    
    def get_memory_summary(self) -> Dict:
        """Get comprehensive summary of memory state"""
        total_centroids = sum(len(clusters) for clusters in self.cluster_centroids.values())
        total_exemplars = sum(
            sum(len(exemplars) for exemplars in clusters.values()) 
            for clusters in self.cluster_exemplars.values()
        )
        
        activity_summaries = {}
        for activity_label in self.cluster_centroids.keys():
            centroids_count = len(self.cluster_centroids[activity_label])
            exemplars_count = sum(len(exemplars) for exemplars in self.cluster_exemplars.get(activity_label, {}).values())
            
            activity_summaries[activity_label] = {
                'centroids': centroids_count,
                'exemplars': exemplars_count,
                'memory_usage': self.memory_usage_stats.get(activity_label, 0)
            }
        
        return {
            'total_centroids': total_centroids,
            'total_exemplars': total_exemplars,
            'activities': len(self.cluster_centroids),
            'memory_updates': self.memory_updates,
            'last_cleanup': self.last_cleanup,
            'activity_summaries': activity_summaries,
            'config': self.config
        }