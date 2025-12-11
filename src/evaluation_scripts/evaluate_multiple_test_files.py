import os
import json
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import glob
from collections import defaultdict
import tempfile
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import evaluation_config

class MultiTestFileEvaluator:
    def __init__(self, base_tracking_dir, ground_truth_dir, output_dir="evaluation_results"):
        self.base_tracking_dir = base_tracking_dir
        self.ground_truth_dir = ground_truth_dir

        tracking_dir_name = os.path.basename(os.path.normpath(base_tracking_dir))
        self.output_dir = os.path.join(output_dir, f"multi_test_evaluation_{tracking_dir_name}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.test_result_dirs = self._find_test_result_directories()
        print(f"[EVALUATOR] Found {len(self.test_result_dirs)} test result directories")

    def _find_test_result_directories(self):
        pattern = os.path.join(self.base_tracking_dir, "test_results_*")
        test_dirs = glob.glob(pattern)
        return sorted(test_dirs)

    def _extract_test_file_name(self, test_result_dir):
        dir_name = os.path.basename(test_result_dir)
        if dir_name.startswith("test_results_"):
            return dir_name[len("test_results_"):]
        return dir_name

    def _find_ground_truth_file(self, test_file_name):
        exact_match_patterns = [
            f"{test_file_name}.csv",
            f"{test_file_name}",
        ]

        for pattern in exact_match_patterns:
            gt_path = os.path.join(self.ground_truth_dir, pattern)
            if os.path.exists(gt_path):
                print(f"[FOUND] Using exact match ground truth: {gt_path}")
                return gt_path

        fallback_patterns = [
            f"{test_file_name}_groundtruth.csv",
            f"{test_file_name}_ground_truth.csv",
            f"groundtruth_{test_file_name}.csv",
            f"ground_truth_{test_file_name}.csv",
        ]

        for pattern in fallback_patterns:
            gt_path = os.path.join(self.ground_truth_dir, pattern)
            if os.path.exists(gt_path):
                print(f"[FOUND] Using pattern match ground truth: {gt_path}")
                return gt_path

        general_patterns = ["groundtruth.csv", "ground_truth.csv", "ipalia_groundtruth.csv"]
        for pattern in general_patterns:
            gt_path = os.path.join(self.ground_truth_dir, pattern)
            if os.path.exists(gt_path):
                print(f"[WARNING] Using general ground truth file {gt_path} for test {test_file_name}")
                return gt_path

        return None

    def _extract_activities_from_centroids(self, centroid_path):
        try:
            import json
            with open(centroid_path, 'r') as f:
                data = json.load(f)

            activities = list(data.get('centroids_by_activity', {}).keys())
            print(f"  Found {len(activities)} activities in centroids: {activities}")
            return activities
        except Exception as e:
            print(f"  Warning: Could not extract activities from centroids: {e}")
            return []

    def _create_temp_evaluate_script(self, test_result_dir, ground_truth_path, activity_of_interest="A", test_output_dir=None):
        event_vector_path = os.path.join(test_result_dir, "event_feature_vectors.jsonl")
        centroid_path = os.path.join(test_result_dir, "final_centroids.json")

        original_script_path = "src/evaluation_scripts/evaluation_core.py"
        with open(original_script_path, 'r') as f:
            script_content = f.read()

        try:
            import sys as sys_import
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys_import.path.insert(0, project_root)
            from config.config import evaluation_config

            eval_config = evaluation_config.get('multi_test_evaluation_config', {})
            event_id_col = eval_config.get('event_id_column', 'EventID')
            control_flow_col = eval_config.get('control_flow_column', 'base_activity_label')
            control_flow_gt_col = eval_config.get('control_flow_column_ground_truth', 'ground_truth_activity_label')
            case_id_col = eval_config.get('case_id_column', 'SYSCALL_pid')

            sys_import.path.remove(project_root)
        except Exception as e:
            print(f"Warning: Could not load config, using fallback values: {e}")
            event_id_col = "EventID"
            control_flow_col = "base_activity_label"
            control_flow_gt_col = "ground_truth_activity_label"
            case_id_col = "SYSCALL_pid"

        new_config = f'''# === CONFIGURATION ===
tracking_dir = r"{self.base_tracking_dir.replace(os.sep, '/')}"
event_vector_path = r"{event_vector_path.replace(os.sep, '/')}"
centroid_path = r"{centroid_path.replace(os.sep, '/')}"
ground_truth_path = r"{ground_truth_path.replace(os.sep, '/')}"
activity_of_interest = "{activity_of_interest}"
event_id_column = "{event_id_col}"
control_flow_column = "{control_flow_col}"
control_flow_column_ground_truth = "{control_flow_gt_col}"
case_id_column = "{case_id_col}"
test_output_dir = r"{test_output_dir.replace(os.sep, '/') if test_output_dir else ''}"'''

        enhanced_functions = '''

# === ENHANCED ANALYSIS FUNCTIONS ===
import os
import json
from collections import defaultdict

def save_cluster_perspective_analysis(label_distribution, test_output_dir):
    if not test_output_dir:
        return

    cluster_analysis = []
    total_events = sum(len(labels) for labels in label_distribution.values())

    for cid, labels in label_distribution.items():
        cluster_size = len(labels)
        cluster_entropy = compute_cluster_entropy(labels, base=2)
        weight = cluster_size / total_events

        label_counts = pd.Series(labels).value_counts()
        label_distribution_dict = {}
        for label, count in label_counts.items():
            prob = count / cluster_size
            label_distribution_dict[label] = {
                'count': count,
                'probability': prob
            }

        cluster_analysis.append({
            'cluster_id': cid,
            'cluster_size': cluster_size,
            'entropy': cluster_entropy,
            'weight': weight,
            'label_distribution': label_distribution_dict
        })

    output_file = os.path.join(test_output_dir, "cluster_perspective_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=2)

    print(f"[SAVED] Cluster perspective analysis to {output_file}")

def save_label_perspective_analysis(reassigned, test_output_dir):
    if not test_output_dir:
        return

    label_to_clusters = defaultdict(list)
    for r in reassigned:
        if r["ground_truth"] != "unknown":
            label_to_clusters[r["ground_truth"]].append(r["nearest_cid"])

    label_analysis = []
    total_events = sum(len(clusters) for clusters in label_to_clusters.values())

    for label, cluster_list in label_to_clusters.items():
        cluster_counts = pd.Series(cluster_list).value_counts()
        probabilities = cluster_counts / len(cluster_list)
        entropy = shannon_entropy(probabilities, base=2)
        if len(probabilities) > 1:
            max_entropy = np.log2(len(probabilities))
            entropy /= max_entropy
        weight = len(cluster_list) / total_events

        cluster_distribution_dict = {}
        for cluster_id, count in cluster_counts.items():
            prob = count / len(cluster_list)
            cluster_distribution_dict[cluster_id] = {
                'count': count,
                'probability': prob
            }

        label_analysis.append({
            'ground_truth_label': label,
            'label_size': len(cluster_list),
            'entropy': entropy,
            'weight': weight,
            'cluster_distribution': cluster_distribution_dict
        })

    output_file = os.path.join(test_output_dir, "label_perspective_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(label_analysis, f, indent=2)

    print(f"[SAVED] Label perspective analysis to {output_file}")
'''

        lines = script_content.splitlines()
        new_lines = []
        in_config = False
        config_end_found = False

        for line in lines:
            if line.strip() == "# === CONFIGURATION ===":
                in_config = True
                new_lines.extend(new_config.splitlines())
                continue
            elif in_config and line.strip() == "# === LOAD DATA ===":
                in_config = False
                config_end_found = True
                new_lines.extend(enhanced_functions.splitlines())
                new_lines.append(line)
                continue
            elif not in_config or config_end_found:
                new_lines.append(line)

        for i, line in enumerate(new_lines):
            if "metrics = calculate_clustering_metrics(reassigned)" in line:
                analysis_calls = [
                    "",
                    "    # === SAVE ENHANCED ANALYSIS ===",
                    "    if test_output_dir:",
                    "        save_cluster_perspective_analysis(label_distribution, test_output_dir)",
                    "        save_label_perspective_analysis(reassigned, test_output_dir)"
                ]
                new_lines[i+1:i+1] = analysis_calls
                break

        return "\n".join(new_lines)

    def _run_evaluation_for_test_file(self, test_result_dir, ground_truth_path, activity_of_interest="ALL"):

        test_file_name = self._extract_test_file_name(test_result_dir)
        print(f"\n[EVALUATING] Test file: {test_file_name}")

        test_output_dir = os.path.join(self.output_dir, test_file_name)
        os.makedirs(test_output_dir, exist_ok=True)

        event_vector_path = os.path.join(test_result_dir, "event_feature_vectors.jsonl")
        centroid_path = os.path.join(test_result_dir, "final_centroids.json")

        if not os.path.exists(event_vector_path):
            print(f"[ERROR] Event vectors not found: {event_vector_path}")
            return None

        if not os.path.exists(centroid_path):
            print(f"[ERROR] Centroids not found: {centroid_path}")
            return None

        available_activities = self._extract_activities_from_centroids(centroid_path)

        if not available_activities:
            print(f"[WARNING] No activities found in centroids, using default activity: {activity_of_interest}")
            activities_to_evaluate = [activity_of_interest] if activity_of_interest != "ALL" else []
        else:
            activities_to_evaluate = available_activities

        all_metrics = {}

        if activity_of_interest == "ALL" or len(activities_to_evaluate) > 1:
            combined_metrics = self._run_single_activity_evaluation(
                test_result_dir, ground_truth_path, "ALL", test_output_dir
            )
            if combined_metrics:
                all_metrics['ALL_ACTIVITIES'] = combined_metrics

        for activity in activities_to_evaluate:
            activity_metrics = self._run_single_activity_evaluation(
                test_result_dir, ground_truth_path, activity, test_output_dir
            )
            if activity_metrics:
                all_metrics[activity] = activity_metrics

        if not all_metrics:
            print(f"[ERROR] No successful evaluations for {test_file_name}")
            return None

        primary_result = all_metrics.get('ALL_ACTIVITIES', list(all_metrics.values())[0])

        per_activity_clean = {}
        for activity, metrics in all_metrics.items():
            per_activity_clean[activity] = {
                key: value for key, value in metrics.items()
                if key not in ['per_activity_results', 'test_output_dir']
            }

        primary_result['per_activity_results'] = per_activity_clean
        primary_result['evaluated_activities'] = list(activities_to_evaluate)

        return primary_result

    def _run_single_activity_evaluation(self, test_result_dir, ground_truth_path, activity_of_interest, test_output_dir):

        event_vector_path = os.path.join(test_result_dir, "event_feature_vectors.jsonl")
        centroid_path = os.path.join(test_result_dir, "final_centroids.json")

        temp_script_content = self._create_temp_evaluate_script(
            test_result_dir, ground_truth_path, activity_of_interest, test_output_dir
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(temp_script_content)
            temp_script_path = temp_file.name

        try:
            result = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )

            if result.returncode != 0:
                print(f"[ERROR] Evaluation failed for activity {activity_of_interest}")
                print(f"Error output: {result.stderr}")
                return None

            output_file = os.path.join(test_output_dir, f"evaluation_output_{activity_of_interest}.txt")
            with open(output_file, 'w') as f:
                f.write(result.stdout)

            metrics = self._parse_evaluation_output(result.stdout)
            metrics['activity'] = activity_of_interest
            metrics['test_output_dir'] = test_output_dir

            print(f"[SUCCESS] Evaluated activity '{activity_of_interest}'")
            entropy_clusters = metrics.get('expected_entropy_clusters', 'N/A')
            entropy_labels = metrics.get('expected_entropy_labels', 'N/A')
            nmi = metrics.get('nmi', 'N/A')
            ari = metrics.get('ari', 'N/A')

            print(f"  Expected Entropy (Clusters): {entropy_clusters if entropy_clusters == 'N/A' else f'{entropy_clusters:.4f}'}")
            print(f"  Expected Entropy (Labels): {entropy_labels if entropy_labels == 'N/A' else f'{entropy_labels:.4f}'}")
            print(f"  NMI: {nmi if nmi == 'N/A' else f'{nmi:.4f}'}")
            print(f"  ARI: {ari if ari == 'N/A' else f'{ari:.4f}'}")

            if 'log_precision_imprecise' in metrics or 'log_precision' in metrics:
                print(f"  Models:")
                if 'log_precision_imprecise' in metrics:
                    print(f"    Mlab: Precision={metrics.get('log_precision_imprecise', 'N/A'):.4f}, "
                          f"Fitness={metrics.get('log_fitness_imprecise', 'N/A'):.4f}")
                if 'log_precision' in metrics:
                    print(f"    Mre:  Precision={metrics.get('log_precision', 'N/A'):.4f}, "
                          f"Fitness={metrics.get('log_fitness', 'N/A'):.4f}")
                if 'improvement_precision' in metrics:
                    imp_prec = metrics.get('improvement_precision', 0)
                    imp_fit = metrics.get('improvement_fitness', 0)
                    print(f"    Improvement: Δ Precision={imp_prec:+.4f}, Δ Fitness={imp_fit:+.4f}")

            print(f"  Output saved to: {output_file}")

            return metrics

        finally:
            try:
                os.unlink(temp_script_path)
            except:
                pass

    def _parse_evaluation_output(self, output):
        metrics = {}

        lines = output.split('\n')
        current_model = None

        for i, line in enumerate(lines):
            line = line.strip()

            if "Expected Entropy (Clusters" in line and ":" in line:
                try:
                    value = float(line.split(':')[1].strip())
                    metrics['expected_entropy_clusters'] = value
                except:
                    pass

            elif "Expected Entropy (Labels" in line and ":" in line:
                try:
                    value = float(line.split(':')[1].strip())
                    metrics['expected_entropy_labels'] = value
                except:
                    pass

            elif "Normalized Mutual Information (NMI):" in line:
                try:
                    value = float(line.split(':')[1].strip())
                    metrics['nmi'] = value
                except:
                    pass

            elif "Adjusted Rand Index (ARI):" in line:
                try:
                    value = float(line.split(':')[1].strip())
                    metrics['ari'] = value
                except:
                    pass

            elif "Mlab = D(Llab)" in line:
                current_model = "imprecise"
            elif "Mre = β(D(Lre))" in line:
                current_model = "refined"

            elif "Log Precision:" in line and current_model:
                try:
                    value = float(line.split(':')[1].strip())
                    if current_model == "imprecise":
                        metrics['log_precision_imprecise'] = value
                    elif current_model == "refined":
                        metrics['log_precision'] = value
                except:
                    pass

            elif "Log Fitness:" in line and current_model:
                try:
                    value = float(line.split(':')[1].strip())
                    if current_model == "imprecise":
                        metrics['log_fitness_imprecise'] = value
                    elif current_model == "refined":
                        metrics['log_fitness'] = value
                except:
                    pass

            elif "F-Score:" in line and current_model:
                try:
                    value = float(line.split(':')[1].strip())
                    if current_model == "imprecise":
                        metrics['fscore_imprecise'] = value
                    elif current_model == "refined":
                        metrics['fscore'] = value
                except:
                    pass

            elif "Generalization:" in line and current_model:
                try:
                    value = float(line.split(':')[1].strip())
                    if current_model == "imprecise":
                        metrics['generalization_imprecise'] = value
                    elif current_model == "refined":
                        metrics['generalization'] = value
                except:
                    pass

            elif "Simplicity:" in line and current_model:
                try:
                    value = float(line.split(':')[1].strip())
                    if current_model == "imprecise":
                        metrics['simplicity_imprecise'] = value
                    elif current_model == "refined":
                        metrics['simplicity'] = value
                except:
                    pass

            elif "Δ Log Precision:" in line:
                try:
                    value_str = line.split(':')[1].strip().split()[0]
                    value = float(value_str)
                    metrics['improvement_precision'] = value
                except:
                    pass
            elif "Δ Log Fitness:" in line:
                try:
                    value_str = line.split(':')[1].strip().split()[0]
                    value = float(value_str)
                    metrics['improvement_fitness'] = value
                except:
                    pass
            elif "Δ F-Score:" in line:
                try:
                    value_str = line.split(':')[1].strip().split()[0]
                    value = float(value_str)
                    metrics['improvement_fscore'] = value
                except:
                    pass

        return metrics

    def evaluate_all_test_files(self, activity_of_interest="ALL"):

        print(f"\n{'='*80}")
        print(f"STARTING MULTI-TEST FILE EVALUATION")
        print(f"{'='*80}")
        print(f"Activity of interest: {activity_of_interest}")
        print(f"Base tracking directory: {self.base_tracking_dir}")
        print(f"Ground truth directory: {self.ground_truth_dir}")

        all_metrics = []
        failed_evaluations = []

        for test_result_dir in self.test_result_dirs:
            test_file_name = self._extract_test_file_name(test_result_dir)

            ground_truth_path = self._find_ground_truth_file(test_file_name)
            if not ground_truth_path:
                print(f"[ERROR] No ground truth file found for test {test_file_name}")
                failed_evaluations.append(test_file_name)
                continue

            metrics = self._run_evaluation_for_test_file(
                test_result_dir, ground_truth_path, activity_of_interest
            )

            if metrics:
                all_metrics.append(metrics)
            else:
                failed_evaluations.append(test_file_name)

        if not all_metrics:
            print("[ERROR] No successful evaluations completed!")
            return None

        aggregate_stats = self._calculate_aggregate_statistics(all_metrics)

        self._save_results(all_metrics, aggregate_stats, failed_evaluations)

        return aggregate_stats

    def _calculate_aggregate_statistics(self, all_metrics):

        print(f"\n{'='*60}")
        print(f"CALCULATING AGGREGATE STATISTICS")
        print(f"{'='*60}")

        metrics_arrays = {
            'expected_entropy_clusters': [],
            'expected_entropy_labels': [],
            'nmi': [],
            'ari': [],
            'log_fitness_imprecise': [],
            'log_precision_imprecise': [],
            'fscore_imprecise': [],
            'generalization_imprecise': [],
            'simplicity_imprecise': [],
            'log_fitness': [],
            'log_precision': [],
            'fscore': [],
            'generalization': [],
            'simplicity': [],
            'improvement_precision': [],
            'improvement_fitness': [],
            'improvement_fscore': []
        }

        for metrics in all_metrics:
            for key in metrics_arrays.keys():
                if key in metrics and metrics[key] is not None:
                    metrics_arrays[key].append(metrics[key])

        aggregate_stats = {
            'num_test_files': len(all_metrics),
            'successful_evaluations': len(all_metrics),
        }

        for metric_name, values in metrics_arrays.items():
            if values:
                aggregate_stats[f'{metric_name}_mean'] = np.mean(values)
                aggregate_stats[f'{metric_name}_std'] = np.std(values)
                aggregate_stats[f'{metric_name}_min'] = np.min(values)
                aggregate_stats[f'{metric_name}_max'] = np.max(values)
                aggregate_stats[f'{metric_name}_values'] = values
            else:
                aggregate_stats[f'{metric_name}_mean'] = None
                aggregate_stats[f'{metric_name}_std'] = None

        print(f"Number of test files: {aggregate_stats['num_test_files']}")
        print()

        print("Clustering Metrics:")
        for metric_name in ['expected_entropy_clusters', 'expected_entropy_labels', 'nmi', 'ari']:
            mean_val = aggregate_stats.get(f'{metric_name}_mean')
            std_val = aggregate_stats.get(f'{metric_name}_std')

            if mean_val is not None and std_val is not None:
                print(f"  {metric_name.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        print()

        print("Mlab (Imprecise Model):")
        for metric_name in ['log_precision_imprecise', 'log_fitness_imprecise', 'fscore_imprecise']:
            mean_val = aggregate_stats.get(f'{metric_name}_mean')
            std_val = aggregate_stats.get(f'{metric_name}_std')

            if mean_val is not None and std_val is not None:
                display_name = metric_name.replace('_imprecise', '').replace('_', ' ').title()
                print(f"  {display_name}: {mean_val:.4f} ± {std_val:.4f}")
        print()

        print("Mre (Refined Model):")
        for metric_name in ['log_precision', 'log_fitness', 'fscore']:
            mean_val = aggregate_stats.get(f'{metric_name}_mean')
            std_val = aggregate_stats.get(f'{metric_name}_std')

            if mean_val is not None and std_val is not None:
                display_name = metric_name.replace('_', ' ').title()
                print(f"  {display_name}: {mean_val:.4f} ± {std_val:.4f}")
        print()

        print("Improvements (Mre vs Mlab):")
        for metric_name in ['improvement_precision', 'improvement_fitness', 'improvement_fscore']:
            mean_val = aggregate_stats.get(f'{metric_name}_mean')
            std_val = aggregate_stats.get(f'{metric_name}_std')

            if mean_val is not None and std_val is not None:
                display_name = metric_name.replace('improvement_', 'Δ ').replace('_', ' ').title()
                print(f"  {display_name}: {mean_val:+.4f} ± {std_val:.4f}")
        print()

        return aggregate_stats

    def _save_results(self, all_metrics, aggregate_stats, failed_evaluations):

        detailed_results_path = os.path.join(self.output_dir, "detailed_test_results.json")
        with open(detailed_results_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        aggregate_results_path = os.path.join(self.output_dir, "aggregate_statistics.json")
        with open(aggregate_results_path, 'w') as f:
            json.dump(aggregate_stats, f, indent=2)

        summary_report_path = os.path.join(self.output_dir, "evaluation_summary.md")
        with open(summary_report_path, 'w') as f:
            f.write("# Multi-Test File Evaluation Summary\n\n")
            f.write(f"**Total Test Files:** {len(self.test_result_dirs)}\n")
            f.write(f"**Successful Evaluations:** {aggregate_stats['successful_evaluations']}\n")
            f.write(f"**Failed Evaluations:** {len(failed_evaluations)}\n\n")

            if failed_evaluations:
                f.write("## Failed Evaluations\n\n")
                for failed in failed_evaluations:
                    f.write(f"- {failed}\n")
                f.write("\n")

            f.write("## Aggregate Statistics\n\n")

            f.write("### Clustering Metrics\n\n")
            f.write("| Metric | Mean | Std Dev | Min | Max |\n")
            f.write("|--------|------|---------|-----|-----|\n")

            for metric_name in ['expected_entropy_clusters', 'expected_entropy_labels', 'nmi', 'ari']:
                mean_val = aggregate_stats.get(f'{metric_name}_mean')
                std_val = aggregate_stats.get(f'{metric_name}_std')
                min_val = aggregate_stats.get(f'{metric_name}_min')
                max_val = aggregate_stats.get(f'{metric_name}_max')

                if mean_val is not None:
                    f.write(f"| {metric_name.replace('_', ' ').title()} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} |\n")
                else:
                    f.write(f"| {metric_name.replace('_', ' ').title()} | N/A | N/A | N/A | N/A |\n")

            f.write("\n### Process Mining Metrics\n\n")
            f.write("| Model | Metric | Mean | Std Dev |\n")
            f.write("|-------|--------|------|----------|\n")

            for model_prefix, model_name in [('', 'Mre'), ('_imprecise', 'Mlab')]:
                for base_metric in ['log_precision', 'log_fitness', 'fscore']:
                    metric_name = base_metric + model_prefix
                    mean_val = aggregate_stats.get(f'{metric_name}_mean')
                    std_val = aggregate_stats.get(f'{metric_name}_std')

                    if mean_val is not None:
                        display_name = base_metric.replace('_', ' ').title()
                        f.write(f"| {model_name} | {display_name} | {mean_val:.4f} | {std_val:.4f} |\n")

            f.write("\n### Improvements (Mre vs Mlab)\n\n")
            f.write("| Metric | Mean | Std Dev |\n")
            f.write("|--------|------|----------|\n")

            for metric_name in ['improvement_precision', 'improvement_fitness', 'improvement_fscore']:
                mean_val = aggregate_stats.get(f'{metric_name}_mean')
                std_val = aggregate_stats.get(f'{metric_name}_std')

                if mean_val is not None:
                    display_name = 'Δ ' + metric_name.replace('improvement_', '').replace('_', ' ').title()
                    f.write(f"| {display_name} | {mean_val:+.4f} | {std_val:.4f} |\n")

            f.write("\n## Individual Test File Results\n\n")
            f.write("| Test File | Entropy(C) | Entropy(L) | NMI | ARI | Mlab Prec | Mlab Fit | Mre Prec | Mre Fit | Δ Prec | Δ Fit |\n")
            f.write("|-----------|------------|------------|-----|-----|-----------|----------|----------|---------|--------|-------|\n")

            for metrics in all_metrics:
                test_file = metrics.get('test_file', 'Unknown')

                def fmt(val):
                    return f"{val:.4f}" if isinstance(val, float) else str(val)

                f.write(f"| {test_file} | "
                       f"{fmt(metrics.get('expected_entropy_clusters', 'N/A'))} | "
                       f"{fmt(metrics.get('expected_entropy_labels', 'N/A'))} | "
                       f"{fmt(metrics.get('nmi', 'N/A'))} | "
                       f"{fmt(metrics.get('ari', 'N/A'))} | "
                       f"{fmt(metrics.get('log_precision_imprecise', 'N/A'))} | "
                       f"{fmt(metrics.get('log_fitness_imprecise', 'N/A'))} | "
                       f"{fmt(metrics.get('log_precision', 'N/A'))} | "
                       f"{fmt(metrics.get('log_fitness', 'N/A'))} | "
                       f"{fmt(metrics.get('improvement_precision', 'N/A'))} | "
                       f"{fmt(metrics.get('improvement_fitness', 'N/A'))} |\n")

        print(f"\n[RESULTS SAVED]")
        print(f"Detailed results: {detailed_results_path}")
        print(f"Aggregate statistics: {aggregate_results_path}")
        print(f"Summary report: {summary_report_path}")


class MultiTrackingEvaluator:

    def __init__(self, tracking_base_dir, ground_truth_dir, output_dir="multi_tracking_evaluation_results"):
        self.tracking_base_dir = tracking_base_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.tracking_dirs = self._find_tracking_directories()
        print(f"[MULTI-TRACKING] Found {len(self.tracking_dirs)} tracking directories")

    def _find_tracking_directories(self):
        pattern = os.path.join(self.tracking_base_dir, "tracking_*")
        tracking_dirs = glob.glob(pattern)
        return sorted([d for d in tracking_dirs if os.path.isdir(d)])

    def evaluate_all_tracking_directories(self, activity_of_interest="A"):

        print(f"\n{'='*100}")
        print(f"STARTING MULTI-TRACKING DIRECTORY EVALUATION")
        print(f"{'='*100}")
        print(f"Base directory: {self.tracking_base_dir}")
        print(f"Found {len(self.tracking_dirs)} tracking directories")

        all_tracking_results = []

        for tracking_dir in self.tracking_dirs:
            tracking_name = os.path.basename(tracking_dir)
            print(f"\n{'='*80}")
            print(f"PROCESSING TRACKING DIRECTORY: {tracking_name}")
            print(f"{'='*80}")

            tracking_output_dir = os.path.join(self.output_dir, tracking_name)

            evaluator = MultiTestFileEvaluator(
                tracking_dir,
                self.ground_truth_dir,
                tracking_output_dir
            )

            aggregate_stats = evaluator.evaluate_all_test_files(activity_of_interest)

            if aggregate_stats:
                aggregate_stats['tracking_directory'] = tracking_name
                aggregate_stats['tracking_path'] = tracking_dir
                all_tracking_results.append(aggregate_stats)
                print(f"[SUCCESS] Completed evaluation for {tracking_name}")
            else:
                print(f"[ERROR] Failed evaluation for {tracking_name}")

        if all_tracking_results:
            self._save_overall_results(all_tracking_results)

        return all_tracking_results

    def _save_overall_results(self, all_tracking_results):

        print(f"\n{'='*80}")
        print(f"CALCULATING OVERALL AGGREGATE STATISTICS")
        print(f"{'='*80}")

        overall_results_path = os.path.join(self.output_dir, "all_tracking_detailed_results.json")
        with open(overall_results_path, 'w') as f:
            json.dump(all_tracking_results, f, indent=2)

        summary_path = os.path.join(self.output_dir, "multi_tracking_summary.md")
        with open(summary_path, 'w') as f:
            f.write("# Multi-Tracking Directory Evaluation Summary\n\n")
            f.write(f"**Total Tracking Directories Processed:** {len(all_tracking_results)}\n\n")

            f.write("## Results by Tracking Directory\n\n")
            f.write("| Tracking Directory | Test Files | Entropy (Clusters) | Entropy (Labels) | NMI | ARI | Log Fitness | Log Precision | F-Score |\n")
            f.write("|-------------------|------------|-------------------|------------------|-----|-----|-------------|---------------|---------|\n")

            for result in all_tracking_results:
                tracking_name = result.get('tracking_directory', 'Unknown')
                num_files = result.get('successful_evaluations', 0)

                entropy_c = result.get('expected_entropy_clusters_mean', 'N/A')
                entropy_l = result.get('expected_entropy_labels_mean', 'N/A')
                nmi = result.get('nmi_mean', 'N/A')
                ari = result.get('ari_mean', 'N/A')
                fitness = result.get('log_fitness_mean', 'N/A')
                precision = result.get('log_precision_mean', 'N/A')
                fscore = result.get('fscore_mean', 'N/A')

                entropy_c_str = f"{entropy_c:.4f}" if isinstance(entropy_c, (int, float)) else str(entropy_c)
                entropy_l_str = f"{entropy_l:.4f}" if isinstance(entropy_l, (int, float)) else str(entropy_l)
                nmi_str = f"{nmi:.4f}" if isinstance(nmi, (int, float)) else str(nmi)
                ari_str = f"{ari:.4f}" if isinstance(ari, (int, float)) else str(ari)
                fitness_str = f"{fitness:.4f}" if isinstance(fitness, (int, float)) else str(fitness)
                precision_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else str(precision)
                fscore_str = f"{fscore:.4f}" if isinstance(fscore, (int, float)) else str(fscore)

                f.write(f"| {tracking_name} | {num_files} | {entropy_c_str} | {entropy_l_str} | {nmi_str} | {ari_str} | {fitness_str} | {precision_str} | {fscore_str} |\n")

        print(f"[OVERALL RESULTS SAVED]")
        print(f"Detailed results: {overall_results_path}")
        print(f"Summary report: {summary_path}")


def main():

    try:
        multi_test_config = evaluation_config.get("multi_test_evaluation_config", {})
    except ImportError:
        print("[WARNING] Could not import config, using fallback defaults")
        multi_test_config = {}

    parser = argparse.ArgumentParser(description='Evaluate clustering quality across multiple test files')

    default_tracking_base_dir = multi_test_config.get(
        "default_tracking_base_dir"
    )
    default_ground_truth_dir = multi_test_config.get(
        "default_ground_truth_dir"    )
    default_output_dir = multi_test_config.get(
        "default_output_dir",
    )
    default_activity = multi_test_config.get(
        "default_activity",
    )

    parser.add_argument('--tracking-base-dir', type=str,
                       default=default_tracking_base_dir,
                       help='Base directory containing tracking_* folders with test results')
    parser.add_argument('--ground-truth-dir', type=str,
                       default=default_ground_truth_dir,
                       help='Directory containing ground truth files')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                       help='Output directory for results')
    parser.add_argument('--activity', type=str, default=default_activity,
                       help='Activity of interest for evaluation')

    args = parser.parse_args()

    print(f"[CONFIG] Using tracking base directory: {args.tracking_base_dir}")
    print(f"[CONFIG] Using ground truth directory: {args.ground_truth_dir}")
    print(f"[CONFIG] Using output directory: {args.output_dir}")

    if not os.path.exists(args.tracking_base_dir):
        print(f"[ERROR] Tracking base directory not found: {args.tracking_base_dir}")
        return

    if not os.path.exists(args.ground_truth_dir):
        print(f"[ERROR] Ground truth directory not found: {args.ground_truth_dir}")
        return

    tracking_pattern = os.path.join(args.tracking_base_dir, "tracking_*")
    tracking_dirs = glob.glob(tracking_pattern)
    test_results_pattern = os.path.join(args.tracking_base_dir, "test_results_*")
    test_results_dirs = glob.glob(test_results_pattern)

    if test_results_dirs and not tracking_dirs:
        print(f"[MODE] Single tracking directory mode - found {len(test_results_dirs)} test result directories")
        evaluator = MultiTestFileEvaluator(
            args.tracking_base_dir,
            args.ground_truth_dir,
            args.output_dir
        )
        aggregate_stats = evaluator.evaluate_all_test_files(args.activity)

        if aggregate_stats:
            print(f"\n{'='*80}")
            print(f"EVALUATION COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Results saved to: {args.output_dir}")
        else:
            print(f"\n[ERROR] Evaluation failed!")

    elif tracking_dirs:
        print(f"[MODE] Multiple tracking directory mode - found {len(tracking_dirs)} tracking directories")
        evaluator = MultiTrackingEvaluator(
            args.tracking_base_dir,
            args.ground_truth_dir,
            args.output_dir
        )
        aggregate_stats = evaluator.evaluate_all_tracking_directories(args.activity)

        if aggregate_stats:
            print(f"\n{'='*80}")
            print(f"EVALUATION COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Results saved to: {args.output_dir}")
        else:
            print(f"\n[ERROR] Evaluation failed!")
    else:
        print(f"[ERROR] No tracking directories or test result directories found in {args.tracking_base_dir}")
        print(f"Expected either:")
        print(f"  - Multiple tracking_* directories (for multi-tracking mode)")
        print(f"  - test_results_* directories (for single tracking mode)")
        return


if __name__ == "__main__":
    main()
