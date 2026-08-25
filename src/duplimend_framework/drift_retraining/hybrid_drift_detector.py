import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from .adwin_drift_detector import ADWINDriftDetector

class HybridDriftDetector:
    """
    Combines embedding-space drift detection with comprehensive ADWIN-based drift detection.
    Monitors both reconstruction errors and embedding space changes.
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        self.adwin_detector = ADWINDriftDetector(config)
        
        self.use_embedding_detector = self.config.get("use_embedding_detector", False)
        self.use_adwin_detector = self.config.get("use_adwin_detector", True)
        
        self.combination_strategy = self.config.get("drift_combination_strategy", "any")
        
        self.min_observations = self.config.get("min_observations_for_drift", 100)
        self.warmup_events = self.config.get("warmup_global_events", 4000)
        self.event_counter = 0
        self.drift_history = []

    def update_centroids(self, activity_label: str, centroids: List[np.ndarray], timestamp: int):
        """Update centroid history for an activity - forwards to embedding detector"""
        if self.use_embedding_detector:
            self.embedding_detector.update_centroids(activity_label, centroids, timestamp)
    
    def update_embedding_norms(self, activity_label: str, embeddings: List[np.ndarray], timestamp: int):
        """Update embedding norms for an activity - forwards to embedding detector"""
        if self.use_embedding_detector:
            self.embedding_detector.update_embedding_norms(activity_label, embeddings, timestamp)
    
    def detect_drift(self, activity_label: str, timestamp: int) -> Dict:
        """Detect drift for an activity using configured detectors"""
        return self.update_and_detect_drift(
            activity_label=activity_label,
            centroids=[],
            embeddings=np.array([]),
            reconstruction_error=0.0,
            timestamp=timestamp
        )
    
    def update_and_detect_drift(self, activity_label: str, centroids: List[np.ndarray], 
                                embeddings: np.ndarray, reconstruction_error: float, 
                                timestamp: int) -> Dict:
        """
        Perform comprehensive drift detection using both embedding and ADWIN methods.
        
        Args:
            activity_label: Activity identifier
            centroids: List of cluster centroids
            embeddings: Current embeddings array
            reconstruction_error: Current reconstruction error
            timestamp: Current timestamp
            
        Returns:
            Dictionary with comprehensive drift detection results
        """
        self.event_counter += 1
        
        if self.event_counter < self.warmup_events:
            return {
                'drift_detected': False,
                'reason': 'warmup_phase',
                'warmup_progress': self.event_counter / self.warmup_events
            }
        
        results = {
            'activity': activity_label,
            'timestamp': timestamp,
            'drift_detected': False,
            'detection_methods': {}
        }
        
        if self.use_embedding_detector:
            self.embedding_detector.update_centroids(activity_label, centroids, timestamp)
            self.embedding_detector.update_embedding_norms(activity_label, embeddings, timestamp)
            embedding_drift = self.embedding_detector.detect_drift(activity_label, timestamp)
            results['detection_methods']['embedding'] = embedding_drift
        
        if self.use_adwin_detector:
            adwin_drift = self.adwin_detector.detect_combined_drift(
                activity_label=activity_label,
                reconstruction_error=reconstruction_error,
                embeddings=embeddings,
                centroids=centroids,
                timestamp=timestamp
            )
            results['detection_methods']['adwin'] = adwin_drift
        
        results['drift_detected'] = self._combine_drift_signals(results['detection_methods'])
        
        results['severity'] = self._calculate_overall_severity(results['detection_methods'])
        results['recommendation'] = self._get_recommendation(results)
        
        if results['drift_detected']:
            self.drift_history.append({
                'activity': activity_label,
                'timestamp': timestamp,
                'severity': results['severity'],
                'methods_triggered': self._get_triggered_methods(results['detection_methods'])
            })
        
        return results
    
    def _combine_drift_signals(self, detection_methods: Dict) -> bool:
        """Combine drift signals from different methods based on strategy"""
        drift_signals = []
        
        if 'embedding' in detection_methods:
            drift_signals.append(detection_methods['embedding'].get('drift_detected', False))
        
        if 'adwin' in detection_methods:
            drift_signals.append(detection_methods['adwin'].get('drift_detected', False))
        
        if not drift_signals:
            return False
        
        if self.combination_strategy == "any":
            return any(drift_signals)
        elif self.combination_strategy == "all":
            return all(drift_signals)
        elif self.combination_strategy == "majority":
            return sum(drift_signals) > len(drift_signals) / 2
        else:
            return any(drift_signals)
    
    def _calculate_overall_severity(self, detection_methods: Dict) -> str:
        """Calculate overall drift severity from all detection methods"""
        severities = []
        
        if 'embedding' in detection_methods:
            embed_data = detection_methods['embedding']
            if embed_data.get('drift_detected'):
                severities.append(embed_data.get('severity', 0.5))
        
        if 'adwin' in detection_methods:
            adwin_data = detection_methods['adwin']
            severity_map = {'none': 0, 'low': 0.3, 'medium': 0.6, 'high': 0.9}
            severities.append(severity_map.get(adwin_data.get('drift_severity', 'none'), 0))
        
        if not severities:
            return 'none'
        
        avg_severity = sum(severities) / len(severities)
        
        if avg_severity < 0.3:
            return 'low'
        elif avg_severity < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _get_triggered_methods(self, detection_methods: Dict) -> List[str]:
        """Get list of methods that detected drift"""
        triggered = []
        
        if detection_methods.get('embedding', {}).get('drift_detected'):
            triggered.append('embedding')
        
        if detection_methods.get('adwin', {}).get('drift_detected'):
            adwin_types = detection_methods['adwin'].get('drift_types', [])
            triggered.extend([f'adwin_{t}' for t in adwin_types])
        
        return triggered
    
    def _get_recommendation(self, results: Dict) -> str:
        """Generate recommendation based on drift detection results"""
        if not results['drift_detected']:
            return 'continue_monitoring'
        
        severity = results['severity']
        triggered_methods = self._get_triggered_methods(results['detection_methods'])
        
        if severity == 'high' or len(triggered_methods) >= 3:
            return 'immediate_retraining'
        
        if severity == 'medium' and 'adwin_reconstruction_error' in triggered_methods:
            return 'schedule_retraining'
        
        if severity == 'low':
            return 'increase_monitoring'
        
        return 'monitor_closely'
    
    def get_drift_summary(self) -> Dict:
        """Get comprehensive drift detection summary"""
        summary = {
            'total_events': self.event_counter,
            'warmup_complete': self.event_counter >= self.warmup_events,
            'drift_history': self.drift_history[-10:],
            'total_drifts': len(self.drift_history)
        }
        
        if self.use_embedding_detector:
            summary['embedding_summary'] = self.embedding_detector.get_drift_summary()
        
        if self.use_adwin_detector:
            summary['adwin_summary'] = self.adwin_detector.get_drift_summary()
        
        return summary
    
    def should_trigger_retraining(self, activity_label: str) -> Tuple[bool, Dict]:
        """
        Check if retraining should be triggered for an activity based on drift detection.
        ADWIN automatically adapts window size - no fixed windows needed.
        
        Args:
            activity_label: Activity to check
            
        Returns:
            Tuple of (should_retrain: bool, info: dict with details)
        """
        recent_drifts = [d for d in self.drift_history 
                        if d['activity'] == activity_label]
        
        if not recent_drifts:
            return False, {'reason': 'no_drift_history'}
        
        latest_drift = recent_drifts[-1]
        
        should_retrain = False
        info = {
            'latest_drift': latest_drift,
            'total_drifts': len(recent_drifts)
        }
        
        if latest_drift['severity'] == 'high':
            should_retrain = True
            info['reason'] = 'high_severity_drift'
        elif latest_drift['severity'] == 'medium' and len(recent_drifts) >= 2:
            should_retrain = True
            info['reason'] = 'repeated_medium_severity_drift'
        elif len(latest_drift.get('methods_triggered', [])) >= 3:
            should_retrain = True
            info['reason'] = 'multiple_drift_types_detected'
        
        if self.use_adwin_detector:
            adwin_summary = self.adwin_detector.get_drift_summary(activity_label)
            if adwin_summary and 'activities' in adwin_summary:
                activity_data = adwin_summary['activities'].get(activity_label, {})
                drift_counts = activity_data.get('drift_counts', {})
                
                reconstruction_drifts = drift_counts.get('reconstruction_error', 0)
                embedding_drifts = drift_counts.get('embedding_norm', 0) + drift_counts.get('embedding_variance', 0)
                cluster_drifts = drift_counts.get('inter_cluster_distance', 0)
                
                total_adwin_drifts = sum(drift_counts.values())
                info['adwin_drift_counts'] = drift_counts
                info['total_adwin_drifts'] = total_adwin_drifts
                
                if reconstruction_drifts >= 2:
                    should_retrain = True
                    info['reason'] = f'multiple_reconstruction_error_drifts ({reconstruction_drifts})'
                elif total_adwin_drifts >= 4:
                    should_retrain = True
                    info['reason'] = f'excessive_total_drifts ({total_adwin_drifts})'
                elif embedding_drifts >= 3:
                    should_retrain = True
                    info['reason'] = f'embedding_space_instability ({embedding_drifts})'
        
        return should_retrain, info
    
    def reset_detectors(self, activity_label: Optional[str] = None):
        """Reset drift detectors for specific activity or all activities"""
        if self.use_adwin_detector:
            if activity_label:
                self.adwin_detector.reset_detector(activity_label)
            else:
                for activity in self.adwin_detector.event_counters.keys():
                    self.adwin_detector.reset_detector(activity)
        
        
    def save_state(self, filepath: str):
        """Save drift detection state"""
        import json
        from datetime import datetime
        
        state = {
            'config': self.config,
            'event_counter': self.event_counter,
            'drift_history': self.drift_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        if self.use_embedding_detector:
            self.embedding_detector.save_drift_state(filepath.replace('.json', '_embedding.json'))
        
        if self.use_adwin_detector:
            self.adwin_detector.save_drift_state(filepath.replace('.json', '_adwin.json'))