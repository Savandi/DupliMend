import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from river.drift import ADWIN
import json
from datetime import datetime

class ADWINDriftDetector:
    """
    Advanced drift detector using ADWIN algorithm for both embedding space 
    and reconstruction error monitoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self.adwin_delta = self.config.get("adwin_delta", 0.002)
        self.adwin_clock = self.config.get("adwin_clock", 32)
        self.adwin_max_buckets = self.config.get("adwin_max_buckets", 5)
        self.adwin_min_window_length = self.config.get("adwin_min_window_length", 5)
        self.adwin_grace_period = self.config.get("adwin_grace_period", 10)
        
        self.adwin_detectors = {
            'reconstruction_error': defaultdict(lambda: ADWIN(
                delta=self.adwin_delta,
                clock=self.adwin_clock,
                max_buckets=self.adwin_max_buckets,
                min_window_length=self.adwin_min_window_length,
                grace_period=self.adwin_grace_period
            )),
            'embedding_norm': defaultdict(lambda: ADWIN(
                delta=self.adwin_delta,
                clock=self.adwin_clock,
                max_buckets=self.adwin_max_buckets,
                min_window_length=self.adwin_min_window_length,
                grace_period=self.adwin_grace_period
            )),
            'embedding_variance': defaultdict(lambda: ADWIN(
                delta=self.adwin_delta,
                clock=self.adwin_clock,
                max_buckets=self.adwin_max_buckets,
                min_window_length=self.adwin_min_window_length,
                grace_period=self.adwin_grace_period
            )),
            'inter_cluster_distance': defaultdict(lambda: ADWIN(
                delta=self.adwin_delta,
                clock=self.adwin_clock,
                max_buckets=self.adwin_max_buckets,
                min_window_length=self.adwin_min_window_length,
                grace_period=self.adwin_grace_period
            ))
        }
        
        self.drift_history = defaultdict(list)
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.last_drift_timestamp = defaultdict(int)
        self.drift_count = defaultdict(int)
        
        self.window_stats = defaultdict(dict)
        
        self.warmup_events = self.config.get("warmup_events", 100)
        self.event_counters = defaultdict(int)
        
    def update_reconstruction_error(self, activity_label: str, error: float, timestamp: int) -> Dict:
        """
        Update ADWIN with reconstruction error and detect drift.
        
        Args:
            activity_label: Activity identifier
            error: Reconstruction error value
            timestamp: Current timestamp
            
        Returns:
            Dictionary with drift detection results
        """
        self.event_counters[activity_label] += 1
        
        self.metric_history[f"{activity_label}_reconstruction_error"].append({
            'timestamp': timestamp,
            'value': error
        })
        
        detector = self.adwin_detectors['reconstruction_error'][activity_label]
        detector.update(error)
        
        drift_detected = detector.drift_detected
        
        if drift_detected:
            self._record_drift_event(activity_label, 'reconstruction_error', timestamp, error)
            
        stats = self._get_window_statistics(detector, 'reconstruction_error', activity_label)
        
        return {
            'drift_detected': drift_detected,
            'metric_type': 'reconstruction_error',
            'current_value': error,
            'window_mean': stats['mean'],
            'window_variance': stats['variance'],
            'window_width': stats['width'],
            'total_samples': self.event_counters[activity_label]
        }
    
    def update_embedding_metrics(self, activity_label: str, embeddings: np.ndarray, 
                                timestamp: int) -> Dict:
        """
        Update ADWIN with embedding-based metrics and detect drift.
        
        Args:
            activity_label: Activity identifier
            embeddings: Array of embeddings [n_samples, embedding_dim]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with drift detection results for all embedding metrics
        """
        embeddings = np.asarray(embeddings)

        if embeddings.size == 0:
            return {'drift_detected': False, 'reason': 'no_embeddings'}

        embeddings = np.atleast_2d(embeddings)

        embedding_norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(embedding_norms)
        embedding_variance = np.var(embeddings.flatten())
        
        norm_detector = self.adwin_detectors['embedding_norm'][activity_label]
        norm_detector.update(mean_norm)
        norm_drift = norm_detector.drift_detected
        
        var_detector = self.adwin_detectors['embedding_variance'][activity_label]
        var_detector.update(embedding_variance)
        var_drift = var_detector.drift_detected
        
        self.metric_history[f"{activity_label}_embedding_norm"].append({
            'timestamp': timestamp,
            'value': mean_norm
        })
        self.metric_history[f"{activity_label}_embedding_variance"].append({
            'timestamp': timestamp,
            'value': embedding_variance
        })
        
        if norm_drift:
            self._record_drift_event(activity_label, 'embedding_norm', timestamp, mean_norm)
        if var_drift:
            self._record_drift_event(activity_label, 'embedding_variance', timestamp, embedding_variance)
        
        norm_stats = self._get_window_statistics(norm_detector, 'embedding_norm', activity_label)
        var_stats = self._get_window_statistics(var_detector, 'embedding_variance', activity_label)
        
        return {
            'drift_detected': norm_drift or var_drift,
            'norm_drift': {
                'detected': norm_drift,
                'current_value': mean_norm,
                'window_stats': norm_stats
            },
            'variance_drift': {
                'detected': var_drift,
                'current_value': embedding_variance,
                'window_stats': var_stats
            }
        }
    
    def update_cluster_metrics(self, activity_label: str, centroids: List[np.ndarray], 
                              timestamp: int) -> Dict:
        """
        Update ADWIN with cluster-based metrics and detect drift.
        
        Args:
            activity_label: Activity identifier
            centroids: List of cluster centroids
            timestamp: Current timestamp
            
        Returns:
            Dictionary with drift detection results
        """
        if len(centroids) < 2:
            return {'drift_detected': False, 'reason': 'insufficient_clusters'}
        
        min_distance = float('inf')
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                min_distance = min(min_distance, dist)
        
        detector = self.adwin_detectors['inter_cluster_distance'][activity_label]
        detector.update(min_distance)
        drift_detected = detector.drift_detected
        
        self.metric_history[f"{activity_label}_inter_cluster_distance"].append({
            'timestamp': timestamp,
            'value': min_distance
        })
        
        if drift_detected:
            self._record_drift_event(activity_label, 'inter_cluster_distance', timestamp, min_distance)
        
        stats = self._get_window_statistics(detector, 'inter_cluster_distance', activity_label)
        
        return {
            'drift_detected': drift_detected,
            'metric_type': 'inter_cluster_distance',
            'current_value': min_distance,
            'window_stats': stats,
            'num_clusters': len(centroids)
        }
    
    def detect_combined_drift(self, activity_label: str, reconstruction_error: float,
                            embeddings: Optional[np.ndarray] = None,
                            centroids: Optional[List[np.ndarray]] = None,
                            timestamp: int = 0) -> Dict:
        """
        Perform combined drift detection across all metrics.
        
        Args:
            activity_label: Activity identifier
            reconstruction_error: Current reconstruction error
            embeddings: Optional embeddings for embedding-based metrics
            centroids: Optional centroids for cluster-based metrics
            timestamp: Current timestamp
            
        Returns:
            Comprehensive drift detection results
        """
        results = {
            'activity': activity_label,
            'timestamp': timestamp,
            'drift_detected': False,
            'drift_types': [],
            'metrics': {}
        }
        
        if self.event_counters[activity_label] < self.warmup_events:
            results['warmup'] = True
            results['warmup_progress'] = self.event_counters[activity_label] / self.warmup_events
            
        recon_result = self.update_reconstruction_error(activity_label, reconstruction_error, timestamp)
        results['metrics']['reconstruction_error'] = recon_result
        if recon_result['drift_detected']:
            results['drift_types'].append('reconstruction_error')
        
        if embeddings is not None:
            embed_result = self.update_embedding_metrics(activity_label, embeddings, timestamp)
            results['metrics']['embeddings'] = embed_result
            if embed_result['drift_detected']:
                if embed_result['norm_drift']['detected']:
                    results['drift_types'].append('embedding_norm')
                if embed_result['variance_drift']['detected']:
                    results['drift_types'].append('embedding_variance')
        
        if centroids is not None and len(centroids) > 1:
            cluster_result = self.update_cluster_metrics(activity_label, centroids, timestamp)
            results['metrics']['clusters'] = cluster_result
            if cluster_result['drift_detected']:
                results['drift_types'].append('inter_cluster_distance')
        
        results['drift_detected'] = len(results['drift_types']) > 0
        results['drift_severity'] = self._calculate_drift_severity(results['drift_types'])
        
        results['recommendation'] = self._get_drift_recommendation(results)
        
        return results
    
    def _record_drift_event(self, activity_label: str, metric_type: str, 
                           timestamp: int, value: float):
        """Record a drift event for tracking"""
        event = {
            'activity': activity_label,
            'metric_type': metric_type,
            'timestamp': timestamp,
            'value': value,
            'drift_count': self.drift_count[f"{activity_label}_{metric_type}"]
        }
        
        self.drift_history[activity_label].append(event)
        self.drift_count[f"{activity_label}_{metric_type}"] += 1
        self.last_drift_timestamp[f"{activity_label}_{metric_type}"] = timestamp
    
    def _get_window_statistics(self, detector: ADWIN, metric_type: str, 
                               activity_label: str) -> Dict:
        """Get statistics from ADWIN window"""
        try:
            return {
                'mean': detector.mean,
                'variance': detector.variance,
                'width': detector.width,
                'n_samples': detector.n_samples,
                'total_drift_detected': self.drift_count[f"{activity_label}_{metric_type}"]
            }
        except:
            return {
                'mean': 0.0,
                'variance': 0.0,
                'width': 0,
                'n_samples': 0,
                'total_drift_detected': 0
            }
    
    def _calculate_drift_severity(self, drift_types: List[str]) -> str:
        """Calculate overall drift severity based on detected drift types"""
        if len(drift_types) == 0:
            return 'none'
        elif len(drift_types) == 1:
            return 'low'
        elif len(drift_types) == 2:
            return 'medium'
        else:
            return 'high'
    
    def _get_drift_recommendation(self, results: Dict) -> str:
        """Generate recommendation based on drift detection results"""
        if not results['drift_detected']:
            return 'continue_monitoring'
        
        severity = results['drift_severity']
        drift_types = results['drift_types']
        
        if severity == 'high':
            return 'immediate_retraining_recommended'
        elif severity == 'medium':
            if 'reconstruction_error' in drift_types:
                return 'schedule_retraining'
            else:
                return 'increase_monitoring'
        else:
            return 'monitor_closely'
    
    def get_drift_summary(self, activity_label: Optional[str] = None) -> Dict:
        """Get comprehensive drift detection summary"""
        if activity_label:
            activities = [activity_label]
        else:
            activities = set(k.split('_')[0] for k in self.drift_count.keys())
        
        summary = {
            'activities': {},
            'total_drift_events': sum(self.drift_count.values()),
            'metric_types': ['reconstruction_error', 'embedding_norm', 
                           'embedding_variance', 'inter_cluster_distance']
        }
        
        for activity in activities:
            activity_summary = {
                'total_events': self.event_counters[activity],
                'drift_counts': {},
                'last_drift': {},
                'recent_history': []
            }
            
            for metric_type in summary['metric_types']:
                key = f"{activity}_{metric_type}"
                activity_summary['drift_counts'][metric_type] = self.drift_count.get(key, 0)
                activity_summary['last_drift'][metric_type] = self.last_drift_timestamp.get(key, 0)
            
            if activity in self.drift_history:
                activity_summary['recent_history'] = self.drift_history[activity][-5:]
            
            summary['activities'][activity] = activity_summary
        
        return summary
    
    def reset_detector(self, activity_label: str, metric_type: Optional[str] = None):
        """Reset ADWIN detector for specific activity and metric"""
        if metric_type:
            if metric_type in self.adwin_detectors:
                self.adwin_detectors[metric_type][activity_label] = ADWIN(
                    delta=self.adwin_delta,
                    clock=self.adwin_clock,
                    max_buckets=self.adwin_max_buckets,
                    min_window_length=self.adwin_min_window_length,
                    grace_period=self.adwin_grace_period
                )
        else:
            for metric_dict in self.adwin_detectors.values():
                if activity_label in metric_dict:
                    del metric_dict[activity_label]
        
        self.event_counters[activity_label] = 0
    
    def save_drift_state(self, filepath: str):
        """Save drift detection state to file"""
        state = {
            'config': self.config,
            'drift_counts': dict(self.drift_count),
            'last_drift_timestamps': dict(self.last_drift_timestamp),
            'event_counters': dict(self.event_counters),
            'drift_history': {k: v[-100:] for k, v in self.drift_history.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_drift_state(self, filepath: str):
        """Load drift detection state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state.get('config', self.config)
        self.drift_count = defaultdict(int, state.get('drift_counts', {}))
        self.last_drift_timestamp = defaultdict(int, state.get('last_drift_timestamps', {}))
        self.event_counters = defaultdict(int, state.get('event_counters', {}))
        
        for activity, events in state.get('drift_history', {}).items():
            self.drift_history[activity] = events