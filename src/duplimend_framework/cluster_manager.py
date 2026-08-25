from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np

from config.config import use_control_flow_features
from src.duplimend_framework.drift_retraining.hybrid_drift_detector import HybridDriftDetector
from src.duplimend_framework.online_clustering_algorithms.clustream_adapter import CluStreamAdapter
from src.duplimend_framework.online_clustering_algorithms.dbstream_adapter import DBStreamAdapter
from src.duplimend_framework.online_clustering_algorithms.denstream_adapter import DenStreamAdapter
from src.duplimend_framework.online_clustering_algorithms.streamKMeans_adapter import STREAMKMeansAdapter
from src.duplimend_framework.utils.logging_utils import log_traceability
from src.duplimend_framework.utils.global_state import event_cluster_mapping, cluster_event_mapping
from src.duplimend_framework.drift_retraining.centroid_memory_manager import CentroidMemoryManager


class ClusterManager:
    """
    Manages clustering per activity, detects homonyms, and handles cluster lifecycle
    """

    def __init__(self, config=None, tracker=None, offline_mode=False):
        self.config = config or {}
        self.tracker = tracker
        self.offline_mode = offline_mode
        self.clustering_adapters = {}
        self.cluster_algorithm = self.config.get("algorithm", "river_dbstream")
        self.min_cluster_weight = self.config.get("min_cluster_weight")
        self.merge_threshold = self.config.get("merge_threshold")
        self.variance_threshold = self.config.get("variance_threshold")
        self.decay_interval = self.config.get("decay_interval", 100)
        self.event_counter = 0

        if not self.offline_mode:
            drift_config = self.config.get("drift_detection", {})
            self.drift_detector = HybridDriftDetector(drift_config)
            self.drift_check_interval = drift_config.get("check_interval", 1000)
            self.last_drift_check = {}

            memory_config = self.config.get("memory_management", {})

            self.recent_embeddings = defaultdict(
                lambda: deque(maxlen=memory_config.get("recent_embeddings_maxlen", 1000)))
            self.memory_manager = CentroidMemoryManager(memory_config)
            self.memory_update_interval = memory_config.get("update_interval", 100)
        else:
            self.drift_detector = None
            self.drift_check_interval = None
            self.last_drift_check = {}
            self.recent_embeddings = {}
            self.memory_manager = None
            self.memory_update_interval = None

    def get_or_create_adapter(self, activity_label):
        """Get or create a clustering adapter for this activity"""
        if activity_label not in self.clustering_adapters:
            if self.cluster_algorithm == "river_dbstream":
                self.clustering_adapters[activity_label] = DBStreamAdapter(self.config)
            elif self.cluster_algorithm == "river_streamkmeans":
                self.clustering_adapters[activity_label] = STREAMKMeansAdapter(self.config)
            elif self.cluster_algorithm == "river_denstream":
                self.clustering_adapters[activity_label] = DenStreamAdapter(self.config)
            elif self.cluster_algorithm == "river_clustream":
                self.clustering_adapters[activity_label] = CluStreamAdapter(self.config)
            else:
                self.clustering_adapters[activity_label] = DBStreamAdapter(self.config)

            log_traceability("cluster_manager", activity_label, {
                "action": "created_adapter",
                "algorithm": self.cluster_algorithm
            })

        return self.clustering_adapters[activity_label]

    def get_all_activities(self):
        """Return a list of all activity labels currently managed."""
        return list(self.clustering_adapters.keys())

    def _check_drift_for_activity(self, activity_label: str, timestamp: int):
        """Check for drift in a specific activity's embedding space"""
        adapter = self.clustering_adapters.get(activity_label)
        if not adapter:
            return

        significant_clusters = adapter.get_significant_clusters(self.min_cluster_weight)
        if not significant_clusters:
            return

        centroids = []
        for cluster_id in significant_clusters:
            centroid = adapter.get_cluster_centroid(cluster_id)
            if centroid is not None:
                centroids.append(centroid)

        if centroids:
            self.drift_detector.update_centroids(activity_label, centroids, timestamp)

            recent_embs = list(self.recent_embeddings[activity_label])
            if recent_embs:
                memory_config = self.config.get("memory_management", {})
                n_norm = memory_config.get("recent_embeddings_for_norm", 100)
                embeddings = [emb['embedding'] for emb in recent_embs[-n_norm:]]
                self.drift_detector.update_embedding_norms(activity_label, embeddings, timestamp)

            drift_result = self.drift_detector.detect_drift(activity_label, timestamp)

            if drift_result.get('drift_detected', False):
                log_traceability("drift_detection", activity_label, {
                    "drift_detected": True,
                    "drift_type": drift_result.get('primary_drift_type'),
                    "severity": drift_result.get('severity'),
                    "recommendation": drift_result.get('recommendation')
                })


    def _update_memory_for_activity(self, activity_label: str, timestamp: int):
        """Update centroid memory for an activity"""
        adapter = self.clustering_adapters.get(activity_label)
        if not adapter:
            return

        significant_clusters = adapter.get_significant_clusters(self.min_cluster_weight)
        if not significant_clusters:
            return

        recent_embs = list(self.recent_embeddings[activity_label])
        memory_config = self.config.get("memory_management", {})
        if len(recent_embs) < memory_config.get("min_recent_embeddings_for_memory", 10):
            return
        cluster_embeddings = defaultdict(list)

        n_mem = memory_config.get("recent_embeddings_for_memory", 50)
        for emb_data in recent_embs[-n_mem:]:
            embedding = emb_data['embedding']

            best_cluster = None
            best_distance = float('inf')

            for cluster_id in significant_clusters:
                centroid = adapter.get_cluster_centroid(cluster_id)
                if centroid is not None:
                    distance = np.linalg.norm(embedding - centroid)
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = cluster_id

            if best_cluster is not None:
                cluster_embeddings[best_cluster].append(embedding)

        min_per_cluster = memory_config.get("min_embeddings_per_cluster_for_memory", 3)
        for cluster_id, embeddings in cluster_embeddings.items():
            if len(embeddings) >= min_per_cluster:
                self.memory_manager.update_cluster_memory(
                    activity_label, cluster_id, embeddings, timestamp
                )

        pass

    def get_memory_for_autoencoder(self, activity_label: str):
        """Get memory manager instance for autoencoder use"""
        if self.offline_mode or self.memory_manager is None:
            return None
        return self.memory_manager

    def save_memory_state(self, filepath: str):
        """Save memory state for persistence"""
        if self.offline_mode or self.memory_manager is None:
            return
        self.memory_manager.save_memory_state(filepath)

    def load_memory_state(self, filepath: str):
        """Load memory state from file"""
        if self.offline_mode or self.memory_manager is None:
            return
        self.memory_manager.load_memory_state(filepath)

    def get_memory_summary(self) -> Dict:
        """Get memory management summary"""
        if self.offline_mode or self.memory_manager is None:
            return {"offline_mode": True, "memory_management": "disabled"}
        return self.memory_manager.get_memory_summary()

    def process_event(self, event_id, activity_label, feature_vector, feature_names, timestamp,
                      reconstruction_error=None, autoencoder_embedding=None):
        """
        Process an event by finding its cluster and identifying potential homonyms
        Returns: (cluster_id, combined_details)
        """
        if len(feature_vector) != len(feature_names):
            raise ValueError(
                f"[ERROR] Feature vector length ({len(feature_vector)}) does not match feature names ({len(feature_names)})")

        self.event_counter += 1

        adapter = self.get_or_create_adapter(activity_label)

        if not self.offline_mode and self.recent_embeddings is not None:
            self.recent_embeddings[activity_label].append({
                'embedding': feature_vector.copy(),
                'timestamp': timestamp,
                'event_id': event_id
            })

        cluster_id, is_new_cluster = adapter.partial_fit(feature_vector, feature_names, timestamp)
        adapter.update_event_mappings(event_id, activity_label, cluster_id)

        drift_result = None
        drift_triggered = False
        drift_info = None

        if not self.offline_mode and self.drift_detector is not None:
            if reconstruction_error is None:
                reconstruction_error = 0.0

            embeddings_array = autoencoder_embedding if autoencoder_embedding is not None else None
            warmup_events = self.config.get("warmup_global_events", 4000)

            if self.event_counter >= warmup_events:
                significant_clusters = adapter.get_significant_clusters(self.min_cluster_weight)
                centroids = [adapter.get_cluster_centroid(cid) for cid in significant_clusters
                             if adapter.get_cluster_centroid(cid) is not None]

                if embeddings_array is None:
                    recent_embs = [e['embedding'] for e in list(self.recent_embeddings[activity_label])[-100:]]
                    if recent_embs:
                        embeddings_array = np.array(recent_embs)

                if embeddings_array is not None:
                    drift_result = self.drift_detector.update_and_detect_drift(
                        activity_label=activity_label,
                        centroids=centroids,
                        embeddings=embeddings_array,
                        reconstruction_error=reconstruction_error,
                        timestamp=timestamp
                    )
                    drift_triggered = drift_result.get('drift_detected', False)

                    if drift_triggered:
                        drift_info = drift_result
                        severity = drift_result.get('severity', 'unknown')
                        recommendation = drift_result.get('recommendation', 'monitor')

                        log_traceability("drift_detection", activity_label, {
                            "drift_detected": True,
                            "severity": severity,
                            "recommendation": recommendation,
                            "timestamp": timestamp,
                            "event_count": self.event_counter
                        })

            pass

        if not self.offline_mode and self.memory_update_interval is not None and self.event_counter % self.memory_update_interval == 0:
            self._update_memory_for_activity(activity_label, timestamp)

        if self.tracker is not None:
            self.tracker.track_assignment(event_id, activity_label, cluster_id, timestamp)

        if not is_new_cluster and not self.offline_mode:
            merge_details = adapter.check_for_merges(self.merge_threshold, self.min_cluster_weight, timestamp)
            if merge_details.get("merge_occurred", False) and self.tracker:
                for op in merge_details.get("operations", []):
                    cid1, cid2, merged_id, distance = op
                    before_centroids = {
                        cid1: adapter.get_cluster_centroid(cid1),
                        cid2: adapter.get_cluster_centroid(cid2)
                    }
                    after_centroid = adapter.get_cluster_centroid(merged_id)
                    feature_names = adapter.get_cluster_feature_names(merged_id)
                    self.tracker.track_merge(
                        activity_label,
                        [cid1, cid2],
                        merged_id,
                        [],
                        before_centroids,
                        after_centroid,
                        feature_names
                    )
            split_details = adapter.check_for_splits(
                self.variance_threshold,
                self.min_cluster_weight,
                timestamp
            )
            if split_details.get("split_occurred", False) and self.tracker:
                for op in split_details.get("operations", []):
                    original_cluster_id, new_cluster_ids = op
                    if any(isinstance(cid, list) for cid in new_cluster_ids):
                        flat_new_cluster_ids = [item for sublist in new_cluster_ids for item in
                                                (sublist if isinstance(sublist, list) else [sublist])]
                    else:
                        flat_new_cluster_ids = new_cluster_ids
                    before_centroids = {
                        original_cluster_id: adapter.get_cluster_centroid(original_cluster_id)
                    }
                    after_centroids = {
                        cid: adapter.get_cluster_centroid(cid) for cid in flat_new_cluster_ids
                    }
                    try:
                        feature_names = adapter.get_cluster_feature_names(
                            flat_new_cluster_ids[0]) if flat_new_cluster_ids else []
                        affected_event_ids = adapter.get_event_ids_for_clusters(activity_label, flat_new_cluster_ids)
                        self.tracker.track_split(
                            activity_label,
                            original_cluster_id,
                            flat_new_cluster_ids,
                            affected_event_ids,
                            before_centroids,
                            after_centroids,
                            feature_names
                        )
                    except Exception:
                        continue
        else:
            merge_details = {"merge_occurred": False, "operations": []}
            split_details = {"split_occurred": False, "operations": []}

        if not self.offline_mode and self.event_counter % self.decay_interval == 0:
            adapter.decay_clusters(0.9, self.min_cluster_weight / 2, timestamp)

        homonym_details = {}
        if len(adapter.get_significant_clusters(self.min_cluster_weight)) > 1:
            homonym_details = self.analyze_cluster_relationships(
                activity_label, adapter, cluster_id, feature_vector, feature_names
            )

        if not self.offline_mode and drift_result is None and self.drift_detector is not None:
            drift_triggered, drift_info = self.drift_detector.should_trigger_retraining(activity_label)

        combined_details = {
            "cluster_id": cluster_id,
            "is_new_cluster": is_new_cluster,
            "homonym_detected": homonym_details.get("detected", False),
            "merge_occurred": merge_details.get("merge_occurred", False),
            "split_occurred": split_details.get("split_occurred", False),
            "drift_detected": drift_triggered,
            "drift_info": drift_info if drift_triggered else None
        }

        return cluster_id, combined_details

    def analyze_cluster_relationships(self, activity_label, adapter, current_cluster_id,
                                      feature_vector, feature_names=None):
        """
        Analyze relationships between clusters to detect potential homonyms
        Returns: dict with analysis details
        """
        significant_clusters = adapter.get_significant_clusters(self.min_cluster_weight)

        if len(significant_clusters) <= 1:
            return {"detected": False, "reason": "insufficient_clusters"}

        distances = {}
        for i, cid1 in enumerate(significant_clusters):
            for j, cid2 in enumerate(significant_clusters):
                if i < j:
                    centroid1 = adapter.get_cluster_centroid(cid1)
                    centroid2 = adapter.get_cluster_centroid(cid2)

                    if centroid1 is None or centroid2 is None or len(centroid1) != len(centroid2):
                        continue

                    dist = adapter.calculate_distance(
                        centroid1, centroid2, feature_names
                    )
                    distances[(cid1, cid2)] = dist

        sorted_distances = sorted(distances.items(), key=lambda x: x[1])

        if len(sorted_distances) > 0:
            most_distant = sorted_distances[-1]
            if most_distant[1] > self.variance_threshold:
                return {
                    "detected": True,
                    "reason": "cluster_separation",
                    "clusters": most_distant[0],
                    "distance": most_distant[1]
                }

            current_distances = []
            for (cid1, cid2), dist in distances.items():
                if cid1 == current_cluster_id or cid2 == current_cluster_id:
                    current_distances.append(dist)

            if current_distances and max(current_distances) > self.variance_threshold:
                return {
                    "detected": True,
                    "reason": "current_cluster_separation",
                    "cluster_id": current_cluster_id,
                    "max_distance": max(current_distances)
                }

        return {"detected": False, "reason": "clusters_too_similar"}

    def get_macro_cluster_summary(self, activity_label):
        adapter = self.get_or_create_adapter(activity_label)
        if hasattr(adapter, "get_macro_clusters"):
            macro_clusters = adapter.get_macro_clusters(n_macro_clusters=self.config.get("n_macro_clusters"),
                                                        min_cluster_weight=self.config.get("min_cluster_weight", 0))
            summary = []
            for macro_id, micro_ids in macro_clusters.items():
                valid_micro_ids = [cid for cid in micro_ids if cid in adapter.micro_clusters]
                centroids = []
                for cid in valid_micro_ids:
                    try:
                        centroids.append(adapter.get_cluster_centroid(cid))
                    except KeyError:
                        continue
                if not centroids:
                    continue

                samples = sum(
                    adapter.micro_clusters[cid]["samples"] for cid in valid_micro_ids if cid in adapter.micro_clusters)
                macro_centroid = np.mean(centroids, axis=0)
                macro_centroid = np.maximum(macro_centroid, 0.0)
                summary.append({
                    "macro_id": macro_id,
                    "micro_ids": valid_micro_ids,
                    "samples": samples,
                    "macro_centroid": macro_centroid
                })
            return summary
        else:
            clusters = adapter.get_significant_clusters(self.min_cluster_weight)
            summary = []
            for i, cid in enumerate(clusters):
                centroid = adapter.get_cluster_centroid(cid)
                samples = adapter.micro_clusters[cid]["samples"]
                summary.append({
                    "macro_id": i,
                    "micro_ids": [cid],
                    "samples": samples,
                    "macro_centroid": centroid
                })
            return summary

    def reset_all_clustering_states(self):
        """Reset all clustering states for clean inference phase"""
        from src.duplimend_framework.utils.global_state import event_cluster_mapping, cluster_event_mapping
        event_cluster_mapping.clear()
        cluster_event_mapping.clear()
        self.clustering_adapters.clear()
        self.event_counter = 0

        for activity_label, adapter in self.clustering_adapters.items():

            adapter.cluster_counter = 0
            adapter.micro_clusters.clear()
            adapter.cluster_history.clear()
            adapter.last_modified.clear()
            adapter.vector_hash_cache.clear()

            if hasattr(adapter, 'river_to_adapter_ids'):
                adapter.river_to_adapter_ids.clear()
            if hasattr(adapter, 'adapter_to_river_ids'):
                adapter.adapter_to_river_ids.clear()

            if hasattr(adapter, '_recreate_algorithm'):
                adapter._recreate_algorithm()

            log_traceability("cluster_reset", "ClusterManager", {
                "activity": activity_label,
                "algorithm": adapter.algorithm_name,
                "reason": "autoencoder_retraining"
            })

    def should_trigger_retraining(self, activity_label: str) -> Tuple[bool, Dict]:
        """Check if retraining should be triggered for an activity"""
        if self.offline_mode or self.drift_detector is None:
            return False, {"reason": "offline_mode_no_retraining"}
        return self.drift_detector.should_trigger_retraining(activity_label)

    def update_drift_detection(self, activity_label: str, centroids: List[np.ndarray],
                               embeddings: List[np.ndarray], timestamp: int):
        """Update drift detection with new data"""
        if self.offline_mode or self.drift_detector is None:
            return
        self.drift_detector.update_centroids(activity_label, centroids, timestamp)
        self.drift_detector.update_embedding_norms(activity_label, embeddings, timestamp)

    def get_drift_summary(self) -> Dict:
        """Get comprehensive drift detection summary"""
        if self.offline_mode or self.drift_detector is None:
            return {"offline_mode": True, "drift_detection": "disabled"}
        return self.drift_detector.get_drift_summary()

    def reset_drift_detection(self, activity_label: Optional[str] = None):
        """Reset drift detection state"""
        if self.offline_mode or self.drift_detector is None:
            return
        if activity_label:
            if activity_label in self.drift_detector.activity_drift_scores:
                del self.drift_detector.activity_drift_scores[activity_label]
            if activity_label in self.recent_embeddings:
                self.recent_embeddings[activity_label].clear()
        else:
            self.drift_detector = HybridDriftDetector(self.config.get("drift_detection", {}))
            self.recent_embeddings.clear()
            self.last_drift_check.clear()
