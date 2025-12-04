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

# Add parent directory to path to import evaluate_expected_entropy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core evaluation functions from evaluate_expected_entropy
from evaluate_expected_entropy import (
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

# Import config
from config.config import evaluation_config

# PM4Py imports for checkpoint-specific functionality
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter

# === CONFIGURATION ===
single_eval_config = evaluation_config.get("single_evaluation_config", {})

# Use config values with fallbacks
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

# Create output directory if it doesn't exist
os.makedirs(default_output_dir, exist_ok=True)

# Export these as module-level variables for the imported functions
import evaluate_expected_entropy
evaluate_expected_entropy.event_id_column = event_id_column
evaluate_expected_entropy.case_id_column = case_id_column
evaluate_expected_entropy.control_flow_column = control_flow_column
evaluate_expected_entropy.control_flow_column_ground_truth = control_flow_column_ground_truth

# === CHECKPOINT-SPECIFIC FUNCTIONS ===

def find_checkpoints(tracking_dir):
    """
    Find all checkpoint directories in the tracking directory.

    Args:
        tracking_dir: Directory to search for checkpoints

    Returns:
        list: Sorted list of (checkpoint_number, checkpoint_path) tuples
    """
    import glob
    import re

    checkpoint_dirs = []
    pattern = os.path.join(tracking_dir, "checkpoint_*")

    for checkpoint_path in glob.glob(pattern):
        if os.path.isdir(checkpoint_path):
            # Extract checkpoint number from directory name
            match = re.search(r'checkpoint_(\d+)', os.path.basename(checkpoint_path))
            if match:
                checkpoint_num = int(match.group(1))
                checkpoint_dirs.append((checkpoint_num, checkpoint_path))

    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: x[0])

    print(f"Found {len(checkpoint_dirs)} checkpoints in {tracking_dir}")
    return checkpoint_dirs

def evaluate_checkpoint(checkpoint_path, checkpoint_num, activity_of_interest, gt_df, output_dir=None):
    """
    Evaluate a single checkpoint and return metrics.

    Args:
        checkpoint_path: Path to checkpoint directory (in tracking dir)
        checkpoint_num: Checkpoint number
        activity_of_interest: Activity to evaluate
        gt_df: Ground truth dataframe
        output_dir: Directory to save checkpoint metrics (evaluation_results/duplimend/tracking_XXX/)

    Returns:
        dict: Metrics for this checkpoint
    """
    print(f"\n[CHECKPOINT {checkpoint_num}] Evaluating...")

    # Load checkpoint data
    checkpoint_event_vectors_path = os.path.join(checkpoint_path, f"event_feature_vectors_checkpoint_{checkpoint_num}.jsonl")
    checkpoint_centroids_path = os.path.join(checkpoint_path, f"centroids_checkpoint_{checkpoint_num}.json")

    if not os.path.exists(checkpoint_event_vectors_path):
        print(f"  Event vectors not found: {checkpoint_event_vectors_path}")
        return None

    if not os.path.exists(checkpoint_centroids_path):
        print(f"  Centroids not found: {checkpoint_centroids_path}")
        return None

    # Load checkpoint data using imported functions
    events = load_event_vectors(checkpoint_event_vectors_path)
    centroids = load_centroids(checkpoint_centroids_path)

    if not events or not centroids:
        print(f"  Failed to load checkpoint data")
        return None

    # Assign events to clusters
    assigned_events = assign_events_to_clusters(events, centroids)

    if not assigned_events:
        print(f"  No events assigned to clusters")
        return None

    # Merge with ground truth
    reassigned = merge_with_ground_truth(assigned_events, gt_df)

    if not reassigned:
        print(f"  No events matched with ground truth")
        return None

    # Calculate metrics using imported functions
    expected_entropy_clusters, expected_entropy_labels, label_distribution = calculate_expected_entropy(reassigned)
    clustering_metrics = calculate_clustering_metrics(reassigned)

    # Create PM4Py log for process mining metrics
    result = create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=True)
    if result is not None and len(result) == 3:
        refined_log, imprecise_log, label_mapping = result
        process_mining_metrics = compute_process_mining_metrics_with_label_mapping(
            refined_log, imprecise_log, label_mapping
        )
    else:
        process_mining_metrics = None

    # Compile checkpoint results
    checkpoint_results = {
        "checkpoint": checkpoint_num,
        "expected_entropy_clusters_perspective": float(expected_entropy_clusters),
        "expected_entropy_labels_perspective": float(expected_entropy_labels),
        "normalized_mutual_info_score": float(clustering_metrics.get('nmi', 0.0)),
        "adjusted_rand_score": float(clustering_metrics.get('ari', 0.0)),
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
    if process_mining_metrics:
        print(f"  Log Fitness: {checkpoint_results['log_fitness']:.4f}")
        print(f"  Log Precision: {checkpoint_results['log_precision']:.4f}")
        print(f"  F-Score: {checkpoint_results['fscore']:.4f}")

    # Save metrics to evaluation_results/duplimend/tracking_XXX/checkpoint_XXX/
    if output_dir:
        checkpoint_eval_dir = os.path.join(output_dir, f"checkpoint_{checkpoint_num}")
        os.makedirs(checkpoint_eval_dir, exist_ok=True)
        metrics_file = os.path.join(checkpoint_eval_dir, f"evaluation_metrics_checkpoint_{checkpoint_num}.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(convert_numpy_types(checkpoint_results), f, indent=2)
            print(f"  ✓ Metrics saved to: {metrics_file}")
        except Exception as e:
            print(f"  ✗ Failed to save metrics: {e}")

    return checkpoint_results

def find_refined_log_path(tracking_dir):
    """Find the refined log CSV file in the tracking directory or test results subdirectory."""
    # Look for refined log in main tracking directory
    pattern1 = os.path.join(tracking_dir, "refined_*.csv")
    matches = glob.glob(pattern1)

    if matches:
        return matches[0]  # Return first match

    # Look in test_results subdirectories
    pattern2 = os.path.join(tracking_dir, "test_results_*", "refined_*.csv")
    matches = glob.glob(pattern2)

    if matches:
        return matches[0]  # Return first match

    return None

def create_pm4py_log_from_refined_log_checkpoint(log_path):
    """
    Create a PM4Py EventLog object from a refined CSV log file for checkpoint evaluation.

    Args:
        log_path: Path to the refined CSV log file

    Returns:
        PM4Py EventLog object
    """
    try:
        import pandas as pd
        from pm4py.objects.log.obj import EventLog, Trace, Event
        import datetime

        # Read CSV
        df = pd.read_csv(log_path)

        # Determine case column
        if 'CaseID' in df.columns:
            case_col = 'CaseID'
        elif 'case:concept:name' in df.columns:
            case_col = 'case:concept:name'
        elif 'SYSCALL_pid' in df.columns:
            case_col = 'SYSCALL_pid'
        else:
            print(f"Warning: No case ID column found, using first column")
            case_col = df.columns[0]

        # Determine activity column
        if 'refined_activity' in df.columns:
            activity_col = 'refined_activity'
        elif 'Activity' in df.columns:
            activity_col = 'Activity'
        elif 'concept:name' in df.columns:
            activity_col = 'concept:name'
        else:
            print(f"Warning: No activity column found, using second column")
            activity_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Create PM4Py log
        event_log = EventLog()

        for case_id, case_group in df.groupby(case_col):
            trace = Trace()
            trace.attributes['concept:name'] = str(case_id)

            # Sort by EventID or index
            if 'EventID' in case_group.columns:
                case_group = case_group.sort_values('EventID')

            for _, row in case_group.iterrows():
                event = Event()
                event['concept:name'] = row[activity_col]

                # Add timestamp if available
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

                # Add other attributes
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
    """
    Save checkpoint and final results to a CSV file.

    Args:
        tracking_dir: Base tracking directory
        checkpoint_results: List of checkpoint metric dictionaries
        final_results: Dictionary of final metrics
        csv_output_path: Path to save CSV file

    Returns:
        str: Path to the saved CSV file
    """
    rows = []

    # Add checkpoint results
    for result in checkpoint_results:
        row_data = {'tracking_dir': tracking_dir}

        # Add checkpoint-specific metrics
        prefix = f"checkpoint_{result['checkpoint']}_"

        for metric in ['expected_entropy_clusters_perspective', 'expected_entropy_labels_perspective',
                      'normalized_mutual_info_score', 'adjusted_rand_score', 'log_fitness',
                      'log_precision', 'fscore', 'generalization', 'simplicity',
                      'total_events_analyzed', 'total_clusters']:
            row_data[f'{prefix}{metric}'] = result.get(metric, 0.0)

        rows.append(row_data)

    # Add final results as a separate row
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

    # Write to CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_output_path, index=False)
        print(f"Results saved to: {csv_output_path}")
        return csv_output_path

    return None

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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

# === MAIN EVALUATION WORKFLOW ===

def run_evaluation(tracking_dir_path, ground_truth_file_path, activity, output_dir,
                   event_id_col=None, control_flow_col=None,
                   control_flow_gt_col=None, case_id_col=None):
    """
    Run evaluation on a tracking directory with ground truth.

    Args:
        tracking_dir_path: Path to tracking directory containing event vectors and centroids
        ground_truth_file_path: Path to ground truth CSV file
        activity: Activity of interest to evaluate
        output_dir: Directory to save evaluation results
        event_id_col: Event ID column name (defaults to config)
        control_flow_col: Control flow column name (defaults to config)
        control_flow_gt_col: Ground truth column name (defaults to config)
        case_id_col: Case ID column name (defaults to config)

    Returns:
        dict: Evaluation results including final_results and checkpoint_results
    """
    import evaluate_expected_entropy

    # Use provided column names or fall back to defaults
    evaluate_expected_entropy.event_id_column = event_id_col or event_id_column
    evaluate_expected_entropy.case_id_column = case_id_col or case_id_column
    evaluate_expected_entropy.control_flow_column = control_flow_col or control_flow_column
    evaluate_expected_entropy.control_flow_column_ground_truth = control_flow_gt_col or control_flow_column_ground_truth

    # Construct paths
    event_vector_file = os.path.join(tracking_dir_path, "event_feature_vectors.jsonl")
    centroid_file = os.path.join(tracking_dir_path, "final_centroids.json")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CHECKPOINT-AWARE EVALUATION")
    print(f"{'='*60}")
    print(f"Tracking Directory: {tracking_dir_path}")
    print(f"Activity: {activity}")
    print(f"Output Directory: {output_dir}")

    # Load ground truth
    gt_df = load_ground_truth(ground_truth_file_path)
    if gt_df.empty:
        print("Warning: No ground truth loaded. Continuing without ground truth metrics.")

    # Find and evaluate checkpoints
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

    # Evaluate final state
    print(f"\n=== EVALUATING FINAL STATE ===")

    # Load final data
    events = load_event_vectors(event_vector_file)
    centroids = load_centroids(centroid_file)

    if events and centroids:
        # Assign events to clusters
        assigned_events = assign_events_to_clusters(events, centroids)

        if assigned_events:
            # Merge with ground truth
            reassigned = merge_with_ground_truth(assigned_events, gt_df)

            if reassigned:
                # Calculate all metrics
                expected_entropy_clusters, expected_entropy_labels, label_distribution = calculate_expected_entropy(reassigned)
                clustering_metrics = calculate_clustering_metrics(reassigned)

                # Process mining metrics with label mapping
                print("\n=== Computing Process Mining Metrics ===")
                result = create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=True)
                if result is not None and len(result) == 3:
                    refined_log, imprecise_log, label_mapping = result

                    # 1. Compute Mlab: Model from imprecise log
                    print("\n[1/2] Computing Mlab = D(Llab)...")
                    imprecise_metrics = compute_process_mining_metrics(imprecise_log)

                    # 2. Compute Mre = β(D(Lre)): Model from refined log with relabeling
                    print("\n[2/2] Computing Mre = β(D(Lre))...")
                    refined_metrics = compute_process_mining_metrics_with_label_mapping(
                        refined_log, imprecise_log, label_mapping
                    )
                else:
                    imprecise_metrics = None
                    refined_metrics = None

                # Compile final results
                final_results = {
                    "expected_entropy_clusters_perspective": float(expected_entropy_clusters),
                    "expected_entropy_labels_perspective": float(expected_entropy_labels),
                    "normalized_mutual_info_score": float(clustering_metrics.get('nmi', 0.0)),
                    "adjusted_rand_score": float(clustering_metrics.get('ari', 0.0)),
                    "total_events_analyzed": len(reassigned),
                    "total_clusters": len(set(r['nearest_cid'] for r in reassigned)),
                }

                # Store process mining metrics
                if imprecise_metrics:
                    final_results["imprecise_model"] = convert_numpy_types(imprecise_metrics)

                if refined_metrics:
                    final_results["refined_model"] = convert_numpy_types(refined_metrics)

                # Calculate improvements if both metrics available
                if imprecise_metrics and refined_metrics:
                    final_results["improvements"] = {
                        "precision": float(refined_metrics['precision'] - imprecise_metrics['precision']),
                        "fitness": float(refined_metrics['fitness'] - imprecise_metrics['fitness']),
                        "fscore": float(refined_metrics['fscore'] - imprecise_metrics['fscore'])
                    }

                # Print results summary
                print(f"\n{'='*70}")
                print(f"FINAL EVALUATION RESULTS")
                print(f"{'='*70}")

                print(f"\n--- Clustering Quality Metrics ---")
                print(f"Expected Entropy (Clusters): {expected_entropy_clusters:.6f}")
                print(f"Expected Entropy (Labels): {expected_entropy_labels:.6f}")
                print(f"NMI: {clustering_metrics.get('nmi', 0.0):.6f}")
                print(f"ARI: {clustering_metrics.get('ari', 0.0):.6f}")

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
                        print(f"  Δ Log Precision: {precision_improvement:+.6f}  {'✓' if precision_improvement >= 0 else '✗'}")
                        print(f"  Δ Log Fitness:   {fitness_improvement:+.6f}  {'✓' if fitness_improvement >= 0 else '✗'}")
                        print(f"  Δ F-Score:       {fscore_improvement:+.6f}  {'✓' if fscore_improvement >= 0 else '✗'}")

                # Save results
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # Prepare full results data
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

                # Save to evaluation_results directory (for comparison with baselines)
                json_output = os.path.join(output_dir, f"evaluation_results_{activity}_{timestamp}.json")
                csv_output = os.path.join(output_dir, f"checkpoint_results_{activity}_{timestamp}.csv")

                with open(json_output, 'w') as f:
                    json.dump(full_results_data, f, indent=2)
                print(f"\n✓ Evaluation results saved to: {json_output}")

                # Save CSV results
                if checkpoint_results or final_results:
                    csv_path = save_results_to_csv(tracking_dir_path, checkpoint_results, final_results, csv_output)
                    if csv_path:
                        print(f"✓ CSV results saved to: {csv_path}")

                # Show checkpoint progression if available
                if checkpoint_results:
                    print("\nCheckpoint Progression:")
                    print("Checkpoint | Entropy(C) | Entropy(L) | NMI    | ARI    | Fitness | Precision | F-Score")
                    print("-" * 90)
                    for result in checkpoint_results[-10:]:  # Show last 10 checkpoints
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

                # Return results for programmatic access
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
    """Main function to run checkpoint-aware evaluation using config defaults."""

    print(f"\n{'='*60}")
    print(f"CHECKPOINT-AWARE EVALUATION")
    print(f"{'='*60}")
    print(f"Tracking Directory: {tracking_dir}")
    print(f"Activity: {activity_of_interest}")

    # Call the parameterized evaluation function
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