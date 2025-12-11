import os
import sys
import json
import numpy as np
import pandas as pd
import datetime
import glob
import re
import csv
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_core import (
    load_event_vectors,
    load_centroids,
    load_ground_truth,
    assign_events_to_clusters,
    merge_with_ground_truth,
    calculate_expected_entropy,
    calculate_clustering_metrics,
    create_pm4py_log_from_reassigned_events,
    compute_process_mining_metrics,
    compute_process_mining_metrics_with_label_mapping,
    has_ground_truth_labels,
    create_ground_truth_log,
    compute_ground_truth_model_metrics
)

from config.config import evaluation_config

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter

single_eval_config = evaluation_config.get("single_evaluation_config", {})

default_tracking_dir = single_eval_config.get("default_tracking_dir", "./src/evaluation_results/tracking_20250705_153857")
default_output_dir = single_eval_config.get("default_output_dir", "./evaluation_results")

tracking_dir = os.environ.get('TRACKING_DIR', default_tracking_dir)
event_vector_path = f"{tracking_dir}/event_feature_vectors.jsonl"
centroid_path = f"{tracking_dir}/final_centroids.json"

ground_truth_path = os.environ.get('GROUND_TRUTH_PATH', single_eval_config.get("default_ground_truth_path", "./src/synthetic_logs/ipalia_groundtruth.csv"))
activity_of_interest = os.environ.get('ACTIVITY', single_eval_config.get("default_activity", "A"))
event_id_column = os.environ.get('EVENT_ID_COLUMN', single_eval_config.get("event_id_column", "EventID"))
control_flow_column = os.environ.get('CONTROL_FLOW_COLUMN', single_eval_config.get("control_flow_column", "concept:name"))
control_flow_column_ground_truth = os.environ.get('GROUND_TRUTH_COLUMN', single_eval_config.get("control_flow_column_ground_truth", "ground_truth_activity"))
case_id_column = os.environ.get('CASE_ID_COLUMN', single_eval_config.get("case_id_column", "case:concept:name"))

print(f"[CONFIG] Tracking directory: {tracking_dir}")
print(f"[CONFIG] Output directory: {default_output_dir}")
print(f"[CONFIG] Ground truth: {ground_truth_path}")
print(f"[CONFIG] Activity of interest: {activity_of_interest}")
print(f"[CONFIG] Column mappings:")
print(f"  - Event ID: {event_id_column}")
print(f"  - Control flow: {control_flow_column}")
print(f"  - Ground truth: {control_flow_column_ground_truth}")
print(f"  - Case ID: {case_id_column}")

os.makedirs(default_output_dir, exist_ok=True)

import evaluation_core
evaluation_core.event_id_column = event_id_column
evaluation_core.case_id_column = case_id_column
evaluation_core.control_flow_column = control_flow_column
evaluation_core.control_flow_column_ground_truth = control_flow_column_ground_truth


def find_checkpoints(tracking_dir):
    import glob
    import re

    checkpoint_dirs = []
    pattern = os.path.join(tracking_dir, "checkpoint_*")

    for checkpoint_path in glob.glob(pattern):
        if os.path.isdir(checkpoint_path):
            match = re.search(r'checkpoint_(\d+)', os.path.basename(checkpoint_path))
            if match:
                checkpoint_num = int(match.group(1))
                checkpoint_dirs.append((checkpoint_num, checkpoint_path))

    checkpoint_dirs.sort(key=lambda x: x[0])

    print(f"Found {len(checkpoint_dirs)} checkpoints in {tracking_dir}")
    return checkpoint_dirs


def evaluate_checkpoint(checkpoint_path, checkpoint_num, activity_of_interest, gt_df, output_dir=None):
    print(f"\n[CHECKPOINT {checkpoint_num}] Evaluating...")

    checkpoint_event_vectors_path = os.path.join(checkpoint_path, f"event_feature_vectors_checkpoint_{checkpoint_num}.jsonl")
    checkpoint_centroids_path = os.path.join(checkpoint_path, f"centroids_checkpoint_{checkpoint_num}.json")

    if not os.path.exists(checkpoint_event_vectors_path):
        print(f"  Event vectors not found: {checkpoint_event_vectors_path}")
        return None

    if not os.path.exists(checkpoint_centroids_path):
        print(f"  Centroids not found: {checkpoint_centroids_path}")
        return None

    events = load_event_vectors(checkpoint_event_vectors_path)
    centroids = load_centroids(checkpoint_centroids_path)

    if not events or not centroids:
        print(f"  Failed to load checkpoint data")
        return None

    assigned_events = assign_events_to_clusters(events, centroids)

    if not assigned_events:
        print(f"  No events assigned to clusters")
        return None

    reassigned = merge_with_ground_truth(assigned_events, gt_df)

    if not reassigned:
        print(f"  No events matched with ground truth")
        return None

    feature_vectors = {str(e.get('event_id', '')): e.get('feature_vector', [])
                       for e in events if e.get('feature_vector')}

    expected_entropy_clusters, expected_entropy_labels, label_distribution = calculate_expected_entropy(reassigned)
    clustering_metrics = calculate_clustering_metrics(reassigned, feature_vectors=feature_vectors)

    result = create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=True)
    if result is not None and len(result) == 3:
        refined_log, imprecise_log, label_mapping = result
        process_mining_metrics = compute_process_mining_metrics_with_label_mapping(
            refined_log, imprecise_log, label_mapping
        )
    else:
        process_mining_metrics = None

    silhouette_val = clustering_metrics.get('silhouette')
    checkpoint_results = {
        "checkpoint": checkpoint_num,
        "expected_entropy_clusters_perspective": float(expected_entropy_clusters),
        "expected_entropy_labels_perspective": float(expected_entropy_labels),
        "normalized_mutual_info_score": float(clustering_metrics.get('nmi', 0.0)),
        "adjusted_rand_score": float(clustering_metrics.get('ari', 0.0)),
        "silhouette_score": float(silhouette_val) if silhouette_val is not None else None,
        "total_events_analyzed": len(reassigned),
        "total_clusters": len(set(r['nearest_cid'] for r in reassigned)),
    }

    if process_mining_metrics:
        checkpoint_results.update({
            "log_fitness": float(process_mining_metrics.get('fitness', 0.0)),
            "log_precision": float(process_mining_metrics.get('precision', 0.0)),
            "fscore": float(process_mining_metrics.get('fscore', 0.0)),
            "generalization": float(process_mining_metrics.get('generalization', 0.0)),
            "simplicity": float(process_mining_metrics.get('simplicity', 0.0))
        })
    else:
        checkpoint_results.update({
            "log_fitness": 0.0,
            "log_precision": 0.0,
            "fscore": 0.0,
            "generalization": 0.0,
            "simplicity": 0.0
        })

    print(f"  Entropy (Clusters): {checkpoint_results['expected_entropy_clusters_perspective']:.4f}")
    print(f"  Entropy (Labels): {checkpoint_results['expected_entropy_labels_perspective']:.4f}")
    print(f"  NMI: {checkpoint_results['normalized_mutual_info_score']:.4f}")
    print(f"  ARI: {checkpoint_results['adjusted_rand_score']:.4f}")
    if silhouette_val is not None:
        print(f"  Silhouette: {silhouette_val:.4f}")
    if process_mining_metrics:
        print(f"  Log Fitness: {checkpoint_results['log_fitness']:.4f}")
        print(f"  Log Precision: {checkpoint_results['log_precision']:.4f}")
        print(f"  F-Score: {checkpoint_results['fscore']:.4f}")

    if output_dir:
        checkpoint_eval_dir = os.path.join(output_dir, f"checkpoint_{checkpoint_num}")
        os.makedirs(checkpoint_eval_dir, exist_ok=True)
        metrics_file = os.path.join(checkpoint_eval_dir, f"evaluation_metrics_checkpoint_{checkpoint_num}.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(convert_numpy_types(checkpoint_results), f, indent=2)
            print(f"  Metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"  Failed to save metrics: {e}")

    return checkpoint_results


def find_refined_log_path(tracking_dir):
    pattern1 = os.path.join(tracking_dir, "refined_*.csv")
    matches = glob.glob(pattern1)

    if matches:
        return matches[0]

    pattern2 = os.path.join(tracking_dir, "test_results_*", "refined_*.csv")
    matches = glob.glob(pattern2)

    if matches:
        return matches[0]

    return None


def create_pm4py_log_from_refined_log_checkpoint(log_path):
    try:
        import pandas as pd
        from pm4py.objects.log.obj import EventLog, Trace, Event
        import datetime

        df = pd.read_csv(log_path)

        if 'CaseID' in df.columns:
            case_col = 'CaseID'
        elif 'case:concept:name' in df.columns:
            case_col = 'case:concept:name'
        elif 'SYSCALL_pid' in df.columns:
            case_col = 'SYSCALL_pid'
        else:
            print(f"Warning: No case ID column found, using first column")
            case_col = df.columns[0]

        if 'refined_activity' in df.columns:
            activity_col = 'refined_activity'
        elif 'Activity' in df.columns:
            activity_col = 'Activity'
        elif 'concept:name' in df.columns:
            activity_col = 'concept:name'
        else:
            print(f"Warning: No activity column found, using second column")
            activity_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        event_log = EventLog()

        for case_id, case_group in df.groupby(case_col):
            trace = Trace()
            trace.attributes['concept:name'] = str(case_id)

            if 'EventID' in case_group.columns:
                case_group = case_group.sort_values('EventID')

            for _, row in case_group.iterrows():
                event = Event()
                event['concept:name'] = row[activity_col]

                timestamp_cols = ['time:timestamp', 'timestamp', 'Timestamp', 'SYSCALL_timestamp']
                for ts_col in timestamp_cols:
                    if ts_col in row and pd.notnull(row[ts_col]):
                        try:
                            if ts_col == 'SYSCALL_timestamp':
                                event['time:timestamp'] = pd.to_datetime(row[ts_col], unit='s')
                            else:
                                event['time:timestamp'] = pd.to_datetime(row[ts_col])
                        except:
                            pass
                        break

                for col in df.columns:
                    if col not in [case_col, activity_col] and col not in timestamp_cols:
                        if pd.notnull(row[col]):
                            event[col] = row[col]

                trace.append(event)

            if len(trace) > 0:
                event_log.append(trace)

        print(f"Created PM4Py log with {len(event_log)} traces from {log_path}")
        return event_log

    except Exception as e:
        print(f"Error creating PM4Py log from {log_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results_to_csv(tracking_dir, checkpoint_results, final_results, csv_output_path):
    rows = []

    for result in checkpoint_results:
        row_data = {'tracking_dir': tracking_dir}

        prefix = f"checkpoint_{result['checkpoint']}_"

        for metric in ['expected_entropy_clusters_perspective', 'expected_entropy_labels_perspective',
                      'normalized_mutual_info_score', 'adjusted_rand_score', 'log_fitness',
                      'log_precision', 'fscore', 'generalization', 'simplicity',
                      'total_events_analyzed', 'total_clusters']:
            row_data[f'{prefix}{metric}'] = result.get(metric, 0.0)

        rows.append(row_data)

    if final_results:
        final_row = {
            'tracking_dir': tracking_dir,
            'final_expected_entropy_clusters': final_results.get('expected_entropy_clusters_perspective', 0.0),
            'final_expected_entropy_labels': final_results.get('expected_entropy_labels_perspective', 0.0),
            'final_nmi': final_results.get('normalized_mutual_info_score', 0.0),
            'final_ari': final_results.get('adjusted_rand_score', 0.0),
            'final_log_fitness': final_results.get('process_mining', {}).get('log_fitness', 0.0),
            'final_log_precision': final_results.get('process_mining', {}).get('log_precision', 0.0),
            'final_fscore': final_results.get('process_mining', {}).get('fscore', 0.0),
            'final_generalization': final_results.get('process_mining', {}).get('generalization', 0.0),
            'final_simplicity': final_results.get('process_mining', {}).get('simplicity', 0.0)
        }
        rows.append(final_row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_output_path, index=False)
        print(f"Results saved to: {csv_output_path}")
        return csv_output_path

    return None


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def run_evaluation(tracking_dir_path, ground_truth_file_path, activity, output_dir,
                   event_id_col=None, control_flow_col=None,
                   control_flow_gt_col=None, case_id_col=None):
    import evaluation_core

    evaluation_core.event_id_column = event_id_col or event_id_column
    evaluation_core.case_id_column = case_id_col or case_id_column
    evaluation_core.control_flow_column = control_flow_col or control_flow_column
    evaluation_core.control_flow_column_ground_truth = control_flow_gt_col or control_flow_column_ground_truth

    event_vector_file = os.path.join(tracking_dir_path, "event_feature_vectors.jsonl")
    centroid_file = os.path.join(tracking_dir_path, "final_centroids.json")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CHECKPOINT-AWARE EVALUATION")
    print(f"{'='*60}")
    print(f"Tracking Directory: {tracking_dir_path}")
    print(f"Activity: {activity}")
    print(f"Output Directory: {output_dir}")

    gt_df = load_ground_truth(ground_truth_file_path)
    if gt_df.empty:
        print("Warning: No ground truth loaded. Continuing without ground truth metrics.")

    checkpoints = find_checkpoints(tracking_dir_path)
    checkpoint_results = []

    if checkpoints:
        print(f"\n=== EVALUATING {len(checkpoints)} CHECKPOINTS ===")
        for checkpoint_num, checkpoint_path in checkpoints:
            result = evaluate_checkpoint(checkpoint_path, checkpoint_num, activity, gt_df, output_dir=output_dir)
            if result:
                checkpoint_results.append(result)
    else:
        print("No checkpoints found.")

    print(f"\n=== EVALUATING FINAL STATE ===")

    events = load_event_vectors(event_vector_file)
    centroids = load_centroids(centroid_file)

    if events and centroids:
        assigned_events = assign_events_to_clusters(events, centroids)

        if assigned_events:
            reassigned = merge_with_ground_truth(assigned_events, gt_df)

            if reassigned:
                feature_vectors = {str(e.get('event_id', '')): e.get('feature_vector', [])
                                   for e in events if e.get('feature_vector')}

                expected_entropy_clusters, expected_entropy_labels, label_distribution = calculate_expected_entropy(reassigned)
                clustering_metrics = calculate_clustering_metrics(reassigned, feature_vectors=feature_vectors)

                print("\n=== Computing Process Mining Metrics ===")
                result = create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=True)
                if result is not None and len(result) == 3:
                    refined_log, imprecise_log, label_mapping = result

                    print("\n[1/2] Computing Mlab = D(Llab)...")
                    imprecise_metrics = compute_process_mining_metrics(imprecise_log)

                    print("\n[2/2] Computing Mre = β(D(Lre))...")
                    refined_metrics = compute_process_mining_metrics_with_label_mapping(
                        refined_log, imprecise_log, label_mapping
                    )
                else:
                    imprecise_metrics = None
                    refined_metrics = None

                silhouette_val = clustering_metrics.get('silhouette')
                final_results = {
                    "expected_entropy_clusters_perspective": float(expected_entropy_clusters),
                    "expected_entropy_labels_perspective": float(expected_entropy_labels),
                    "normalized_mutual_info_score": float(clustering_metrics.get('nmi', 0.0)),
                    "adjusted_rand_score": float(clustering_metrics.get('ari', 0.0)),
                    "silhouette_score": float(silhouette_val) if silhouette_val is not None else None,
                    "total_events_analyzed": len(reassigned),
                    "total_clusters": len(set(r['nearest_cid'] for r in reassigned)),
                }

                if imprecise_metrics:
                    final_results["imprecise_model"] = convert_numpy_types(imprecise_metrics)

                if refined_metrics:
                    final_results["refined_model"] = convert_numpy_types(refined_metrics)

                if imprecise_metrics and refined_metrics:
                    final_results["improvements"] = {
                        "precision": float(refined_metrics['precision'] - imprecise_metrics['precision']),
                        "fitness": float(refined_metrics['fitness'] - imprecise_metrics['fitness']),
                        "fscore": float(refined_metrics['fscore'] - imprecise_metrics['fscore'])
                    }

                print(f"\n{'='*70}")
                print(f"FINAL EVALUATION RESULTS")
                print(f"{'='*70}")

                print(f"\n--- Clustering Quality Metrics ---")
                print(f"Expected Entropy (Clusters): {expected_entropy_clusters:.6f}")
                print(f"Expected Entropy (Labels): {expected_entropy_labels:.6f}")
                print(f"NMI: {clustering_metrics.get('nmi', 0.0):.6f}")
                print(f"ARI: {clustering_metrics.get('ari', 0.0):.6f}")
                if silhouette_val is not None:
                    print(f"Silhouette: {silhouette_val:.6f}")

                print(f"\n--- Process Mining Quality Metrics ---")
                if imprecise_metrics:
                    print("\nMlab = D(Llab):")
                    print(f"  Log Precision: {imprecise_metrics['precision']:.6f}")
                    print(f"  Log Fitness:   {imprecise_metrics['fitness']:.6f}")
                    print(f"  F-Score:       {imprecise_metrics['fscore']:.6f}")

                if refined_metrics:
                    print("\nMre = β(D(Lre)):")
                    print(f"  Log Precision: {refined_metrics['precision']:.6f}")
                    print(f"  Log Fitness:   {refined_metrics['fitness']:.6f}")
                    print(f"  F-Score:       {refined_metrics['fscore']:.6f}")

                    if imprecise_metrics:
                        precision_improvement = refined_metrics['precision'] - imprecise_metrics['precision']
                        fitness_improvement = refined_metrics['fitness'] - imprecise_metrics['fitness']
                        fscore_improvement = refined_metrics['fscore'] - imprecise_metrics['fscore']

                        print(f"\n{'─'*70}")
                        print("IMPROVEMENT OVER BASELINE:")
                        print(f"{'─'*70}")
                        print(f"  Δ Log Precision: {precision_improvement:+.6f}")
                        print(f"  Δ Log Fitness:   {fitness_improvement:+.6f}")
                        print(f"  Δ F-Score:       {fscore_improvement:+.6f}")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                full_results_data = convert_numpy_types({
                    "configuration": {
                        "tracking_dir": tracking_dir_path,
                        "ground_truth_path": ground_truth_file_path,
                        "activity_of_interest": activity,
                        "evaluation_timestamp": timestamp
                    },
                    "checkpoint_results": checkpoint_results,
                    "final_results": final_results
                })

                json_output = os.path.join(output_dir, f"evaluation_results_{activity}_{timestamp}.json")
                csv_output = os.path.join(output_dir, f"checkpoint_results_{activity}_{timestamp}.csv")

                with open(json_output, 'w') as f:
                    json.dump(full_results_data, f, indent=2)
                print(f"\nEvaluation results saved to: {json_output}")

                if checkpoint_results or final_results:
                    csv_path = save_results_to_csv(tracking_dir_path, checkpoint_results, final_results, csv_output)
                    if csv_path:
                        print(f"CSV results saved to: {csv_path}")

                if checkpoint_results:
                    print("\nCheckpoint Progression:")
                    print("Checkpoint | Entropy(C) | Entropy(L) | NMI    | ARI    | Fitness | Precision | F-Score")
                    print("-" * 90)
                    for result in checkpoint_results[-10:]:
                        print(f"{result['checkpoint']:10d} | "
                              f"{result['expected_entropy_clusters_perspective']:10.4f} | "
                              f"{result['expected_entropy_labels_perspective']:10.4f} | "
                              f"{result['normalized_mutual_info_score']:6.4f} | "
                              f"{result['adjusted_rand_score']:6.4f} | "
                              f"{result.get('log_fitness', 0):7.4f} | "
                              f"{result.get('log_precision', 0):9.4f} | "
                              f"{result.get('fscore', 0):7.4f}")

                print(f"\n{'='*60}")
                print("EVALUATION COMPLETE")
                print(f"{'='*60}")

                return {
                    "checkpoint_results": checkpoint_results,
                    "final_results": final_results,
                    "output_files": {
                        "json": json_output,
                        "csv": csv_output
                    }
                }
            else:
                print("No events matched with ground truth.")
                return None
        else:
            print("No events were assigned to clusters.")
            return None
    else:
        print("Failed to load final event vectors or centroids.")
        return None


def main():
    print(f"\n{'='*60}")
    print(f"CHECKPOINT-AWARE EVALUATION")
    print(f"{'='*60}")
    print(f"Tracking Directory: {tracking_dir}")
    print(f"Activity: {activity_of_interest}")

    results = run_evaluation(
        tracking_dir_path=tracking_dir,
        ground_truth_file_path=ground_truth_path,
        activity=activity_of_interest,
        output_dir=default_output_dir,
        event_id_col=event_id_column,
        control_flow_col=control_flow_column,
        control_flow_gt_col=control_flow_column_ground_truth,
        case_id_col=case_id_column
    )

    return results


if __name__ == "__main__":
    main()
