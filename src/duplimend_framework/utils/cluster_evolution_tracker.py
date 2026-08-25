import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

from src.duplimend_framework.utils.global_state import event_cluster_mapping, cluster_event_mapping


class ClusterEvolutionTracker:
    def __init__(self, output_dir="./src/evaluation_results"):
        if output_dir is None:
            from config.config import evaluation_config
            base_dir = evaluation_config.get("results_base_dir", "./src/evaluation_results")
        else:
            base_dir = output_dir
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, f"tracking_{timestamp}")
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.operations_file = os.path.join(self.output_dir, "split_merge_operations.jsonl")
        self.config_file = os.path.join(self.output_dir, "config_snapshot.json")
        self.event_history_file = os.path.join(self.output_dir, "event_history.csv")
        self.centroid_history_file = os.path.join(self.output_dir, "centroid_history.jsonl")


        self.split_merge_operations = []
        self.cluster_snapshots = []
        self.event_cluster_history = defaultdict(list)
        
        pd.DataFrame(columns=[
            'timestamp', 'event_id', 'activity', 'cluster_id', 
            'operation', 'details'
        ]).to_csv(self.event_history_file, index=False)
        
        pd.DataFrame(columns=["event_id", "timestamp", "activity", "cluster_id", "operation", "operation_id"]
                    ).to_csv(self.event_history_file, index=False)

        with open(self.operations_file, 'w') as f:
            pass

        self.operation_counter = 0
        self.feature_selection_counter = 0

    def save_final_centroids(self, cluster_manager, centroids_path, test_file_name):
        """Save final centroids for each activity in the current test file"""
        
        final_centroids = {}
        
        for activity_label in cluster_manager.get_all_activities():
            activity_centroids = []
            
            macro_summary = cluster_manager.get_macro_cluster_summary(activity_label)
            
            for macro_cluster in macro_summary:
                centroid = macro_cluster.get('macro_centroid', [])
                if hasattr(centroid, 'tolist'):
                    centroid = centroid.tolist()
                centroid_data = {
                    "cluster_id": macro_cluster['macro_id'],
                    "centroid": centroid,
                    "samples": macro_cluster['samples'],
                    "weight": macro_cluster.get('weight', 0),
                    "activity": activity_label
                }
                activity_centroids.append(centroid_data)
            
            if activity_centroids:
                final_centroids[activity_label] = activity_centroids
        
        centroids_data = {
            "test_file": test_file_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_activities": len(final_centroids),
            "centroids_by_activity": final_centroids
        }
        
        with open(centroids_path, 'w') as f:
            json.dump(centroids_data, f, indent=2, default=str)

    def track_assignment(self, event_id, activity, cluster_id, timestamp=None):
        """Record when an event is assigned to a cluster"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        self.event_cluster_history[event_id].append((timestamp, cluster_id, "assign"))

        with open(self.event_history_file, 'a') as f:
            f.write(f"{event_id},{timestamp},{activity},{cluster_id},assign,0\n")

    def track_centroid(self, activity, cluster_id, centroid, feature_names, timestamp=None):
        """Record a cluster's centroid at a point in time"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        centroid_record = {
            "timestamp": timestamp,
            "activity": activity,
            "cluster_id": cluster_id,
            "centroid": centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
            "feature_names": feature_names
        }


        with open(self.centroid_history_file, 'a') as f:
            f.write(json.dumps(centroid_record) + '\n')

    def save_config_snapshot(self, config_values):
        """Save configuration snapshot to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_values, f, indent=2, default=str)
        except Exception:
            pass
            
    def track_split(self, activity, original_cluster_id, new_cluster_ids, affected_event_ids,
                    before_centroids, after_centroids, feature_names, events_by_cluster=None):
        """Record a cluster split operation with detailed information"""
        timestamp = datetime.now().isoformat()
        self.operation_counter += 1
        operation_id = self.operation_counter

        if events_by_cluster is None:
            events_by_cluster = {cid: [] for cid in new_cluster_ids}

            for event_id in affected_event_ids:
                current_assignment = event_cluster_mapping.get(event_id)
                if current_assignment:
                    current_activity, current_cluster_id = current_assignment
                    if current_activity == activity and current_cluster_id in new_cluster_ids:
                        events_by_cluster[current_cluster_id].append(event_id)

        split_record = {
            "timestamp": timestamp,
            "operation_id": operation_id,
            "type": "split",
            "activity": activity,
            "original_cluster_id": original_cluster_id,
            "new_cluster_ids": new_cluster_ids,
            "affected_event_ids": affected_event_ids,
            "events_by_cluster": events_by_cluster,
            "before_centroids": {
                str(original_cluster_id): before_centroids[original_cluster_id].tolist()
                if isinstance(before_centroids[original_cluster_id], np.ndarray)
                else before_centroids[original_cluster_id]
            },
            "after_centroids": {
                str(cid): after_centroids[cid].tolist() if isinstance(after_centroids[cid], np.ndarray) else
                after_centroids[cid]
                for cid in new_cluster_ids
            },
            "feature_names": feature_names
        }

        self.split_merge_operations.append(split_record)

        with open(self.operations_file, 'a') as f:
            f.write(json.dumps(split_record) + '\n')

        for event_id in affected_event_ids:
            assigned_cluster = None
            for cid, events in events_by_cluster.items():
                if event_id in events:
                    assigned_cluster = cid
                    break

            assigned_cluster = assigned_cluster if assigned_cluster is not None else "split_pending"

            self.event_cluster_history[event_id].append((timestamp, assigned_cluster, "split"))
            with open(self.event_history_file, 'a') as f:
                f.write(f"{event_id},{timestamp},{activity},{assigned_cluster},split,{operation_id}\n")

    def track_merge(self, activity, original_cluster_ids, new_cluster_id, affected_event_ids,
                    before_centroids, after_centroid, feature_names, events_by_source_cluster=None):
        """Record a cluster merge operation with detailed information"""
        timestamp = datetime.now().isoformat()
        self.operation_counter += 1
        operation_id = self.operation_counter

        if events_by_source_cluster is None:
            events_by_source_cluster = {cid: [] for cid in original_cluster_ids}

            for cid in original_cluster_ids:
                cluster_key = (activity, cid)
                if cluster_key in cluster_event_mapping:
                    for event_id in cluster_event_mapping[cluster_key]:
                        if event_id in affected_event_ids:
                            events_by_source_cluster[cid].append(event_id)

        merge_record = {
            "timestamp": timestamp,
            "operation_id": operation_id,
            "type": "merge",
            "activity": activity,
            "original_cluster_ids": original_cluster_ids,
            "new_cluster_id": new_cluster_id,
            "affected_event_ids": affected_event_ids,
            "events_by_source_cluster": events_by_source_cluster,
            "before_centroids": {
                str(cid): before_centroids[cid].tolist() if isinstance(before_centroids[cid], np.ndarray) else
                before_centroids[cid]
                for cid in original_cluster_ids
            },
            "after_centroid": after_centroid.tolist() if isinstance(after_centroid, np.ndarray) else after_centroid,
            "feature_names": feature_names
        }

        self.split_merge_operations.append(merge_record)

        with open(self.operations_file, 'a') as f:
            f.write(json.dumps(merge_record) + '\n')

        for event_id in affected_event_ids:
            self.event_cluster_history[event_id].append((timestamp, new_cluster_id, "merge"))
            with open(self.event_history_file, 'a') as f:
                f.write(f"{event_id},{timestamp},{activity},{new_cluster_id},merge,{operation_id}\n")
    

    def generate_summary_report(self):
        """Generate a consolidated CSV with all event tracking information"""
        consolidated_records = []

        for event_id, history in self.event_cluster_history.items():
            if not history:
                continue
            
            def safe_timestamp_key(x):
                timestamp = x[0]
                if timestamp is None:
                    return 0.0
                
                if isinstance(timestamp, str):
                    try:
                        return pd.to_datetime(timestamp).timestamp()
                    except Exception:
                        return 0.0
                elif hasattr(timestamp, "timestamp"):
                    try:
                        return timestamp.timestamp()
                    except Exception:
                        return 0.0
                elif isinstance(timestamp, (int, float)):
                    return float(timestamp)
                else:
                    return 0.0
            
            try:
                sorted_history = sorted(history, key=safe_timestamp_key)
            except Exception:
                sorted_history = history

            final_record = sorted_history[-1]

            first_assignment = next((h for h in sorted_history if h[2] == "assign"), None)
            first_timestamp = first_assignment[0] if first_assignment else sorted_history[0][0]

            activity = event_cluster_mapping.get(event_id, (None, None))[0]

            num_splits = sum(1 for h in sorted_history if h[2] == "split")
            num_merges = sum(1 for h in sorted_history if h[2] == "merge")

            record = {
                "event_id": event_id,
                "activity": activity,
                "first_assignment_time": first_timestamp,
                "initial_cluster": sorted_history[0][1],
                "final_cluster": final_record[1],
                "num_splits": num_splits,
                "num_merges": num_merges,
                "total_cluster_changes": len(sorted_history) - 1,
                "cluster_journey": "→".join(str(h[1]) for h in sorted_history)
            }

            for op in self.split_merge_operations:
                if event_id in op.get("affected_event_ids", []):
                    op_type = op["type"]
                    if op_type == "split":
                        for cluster_id, events in op.get("events_by_cluster", {}).items():
                            if event_id in events:
                                record[
                                    f"split_op_{op['operation_id']}"] = f"from_{op['original_cluster_id']}_to_{cluster_id}"
                                break
                    elif op_type == "merge":
                        for cluster_id, events in op.get("events_by_source_cluster", {}).items():
                            if event_id in events:
                                record[
                                    f"merge_op_{op['operation_id']}"] = f"from_{cluster_id}_to_{op['new_cluster_id']}"
                                break

            consolidated_records.append(record)
