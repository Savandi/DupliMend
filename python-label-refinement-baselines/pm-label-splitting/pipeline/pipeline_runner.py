import copy
import csv
import time
import os
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import threading

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import pipeline.clustering_method
from evaluation.apply_im import apply_im_without_noise_and_evaluate
from pipeline import clustering_method

# Import path configuration from centralized module (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR, RESULTS_BASE_DIR, BEST_RESULTS_DIR
from pipeline.input_preprocessor import InputPreprocessor
from label_splitter.distance_metrics import Distance
from label_splitter.event_graphs_variant_based import get_event_graphs_from_event_log, EventGraphsVariantBased
from label_splitter.label_splitter_event_based import LabelSplitter as LabelSplitterEventBased
from label_splitter.label_splitter_variant_based import LabelSplitter as LabelSplitterVariantBased
from label_splitter.label_splitter_variant_multiplex import LabelSplitter as LabelSplitterVariantMultiplex
from pipeline.pipeline_helpers import get_tuples_for_folder, get_community_similarity, filter_duplicate_xor
from pipeline.pipeline_variant import PipelineVariant
from utils.file_writer_helper import get_config_string, write_summary_file, \
    write_summary_file_with_parameters, run_start_string, setup_result_folder, export_models_and_pngs
from utils.input_data import InputData

# Try to import clustering evaluation functions for enhanced metrics
try:
    # Test basic functionality first before complex imports
    print("Testing basic Python functionality...")
    
    # Clustering evaluation functions that filter to ONLY imprecise events
    def simple_get_clustering_from_refined_log(refined_log, imprecise_labels):
        """Extract clustering from refined log - ONLY for imprecise events"""
        clustering = []
        seen_labels = []
        try:
            for trace in refined_log:
                for event in trace:
                    activity = event.get("concept:name", "Unknown")

                    # Check if this activity is imprecise or a refined version of one
                    is_imprecise_related = False

                    # Direct match with imprecise labels
                    if activity in imprecise_labels:
                        is_imprecise_related = True
                    else:
                        # Check if it's a refined version (e.g., "D_0", "D_1" for imprecise label "D")
                        for imprecise_label in imprecise_labels:
                            if activity.startswith(f"{imprecise_label}_"):
                                is_imprecise_related = True
                                break

                    # Only include imprecise-related events
                    if is_imprecise_related:
                        if activity not in seen_labels:
                            seen_labels.append(activity)
                        clustering.append(seen_labels.index(activity))

            print(f"[CLUSTERING] Extracted {len(clustering)} imprecise events with {len(seen_labels)} unique labels")
            return clustering
        except Exception as e:
            print(f"[CLUSTERING] Error: {e}")
            return []

    def simple_get_ground_truth_clustering_from_log(original_log, imprecise_labels):
        """Extract ground truth clustering from original log - ONLY for imprecise events"""
        clustering = []
        original_labels = []
        try:
            for trace in original_log:
                for event in trace:
                    current_activity = event.get("concept:name", "Unknown")

                    # Only include events that have been made imprecise
                    if current_activity in imprecise_labels:
                        # Use OrgLabel for ground truth
                        if 'OrgLabel' in event:
                            ground_truth_label = event['OrgLabel']
                            if ground_truth_label not in original_labels:
                                original_labels.append(ground_truth_label)
                            clustering.append(original_labels.index(ground_truth_label))
                        else:
                            # Fallback if no OrgLabel
                            if current_activity not in original_labels:
                                original_labels.append(current_activity)
                            clustering.append(original_labels.index(current_activity))

            print(f"[GROUND_TRUTH] Extracted {len(clustering)} imprecise events with {len(original_labels)} unique ground truth labels")
            return clustering
        except Exception as e:
            print(f"[GROUND_TRUTH] Error: {e}")
            return []

    def compute_cluster_entropy(labels, base=2, normalized=True):
        """Compute entropy for a list of cluster labels"""
        import pandas as pd
        from scipy.stats import entropy as shannon_entropy

        if not labels:
            return 0.0

        label_counts = pd.Series(labels).value_counts()
        total = len(labels)
        probabilities = label_counts / total
        entropy_val = shannon_entropy(probabilities, base=base)

        if normalized and len(label_counts) > 1:
            max_entropy = shannon_entropy([1/len(label_counts)] * len(label_counts), base=base)
            entropy_val = entropy_val / max_entropy if max_entropy > 0 else 0.0

        return entropy_val

    def simple_compute_expected_entropy_clusters_perspective(ground_truth_clustering, refined_clustering, base=2):
        """
        Compute expected entropy from clusters perspective:
        For each cluster, calculate entropy of ground truth labels within it.

        Parameters:
        - ground_truth_clustering: List of ground truth cluster IDs
        - refined_clustering: List of refined cluster IDs from algorithm
        """
        import numpy as np
        from collections import defaultdict, Counter

        if not refined_clustering or not ground_truth_clustering:
            return 0.0

        if len(refined_clustering) != len(ground_truth_clustering):
            print(f"[WARNING] Clustering length mismatch: {len(refined_clustering)} vs {len(ground_truth_clustering)}")
            return 0.0

        # Group ground truth labels by refined clusters
        cluster_to_ground_truth = defaultdict(list)
        for refined_cluster, ground_truth_label in zip(refined_clustering, ground_truth_clustering):
            cluster_to_ground_truth[refined_cluster].append(ground_truth_label)

        total_events = len(refined_clustering)
        cluster_entropies = []
        cluster_weights = []

        print(f"\n=== Expected Entropy Analysis: Clusters Perspective ===")

        for cluster_id, ground_truth_labels in cluster_to_ground_truth.items():
            cluster_size = len(ground_truth_labels)
            cluster_entropy = compute_cluster_entropy(ground_truth_labels, base=base, normalized=False)
            cluster_weight = cluster_size / total_events

            cluster_entropies.append(cluster_entropy)
            cluster_weights.append(cluster_weight)

            # Print cluster details
            label_counts = Counter(ground_truth_labels)
            print(f"Cluster {cluster_id}: {cluster_size} events, entropy = {cluster_entropy:.4f}")
            for label, count in label_counts.items():
                prob = count / cluster_size
                print(f"  {label}: {count} events ({prob:.1%})")

        expected_entropy = np.sum(np.array(cluster_entropies) * np.array(cluster_weights))
        print(f"\nExpected Entropy (Clusters Perspective): {expected_entropy:.4f}")

        return expected_entropy

    def simple_compute_expected_entropy_labels_perspective(ground_truth_clustering, refined_clustering, base=2):
        """
        Compute expected entropy from labels perspective:
        For each ground truth label, calculate entropy of clusters assigned to it.

        Parameters:
        - ground_truth_clustering: List of ground truth cluster IDs
        - refined_clustering: List of refined cluster IDs from algorithm
        """
        import numpy as np
        from collections import defaultdict, Counter

        if not refined_clustering or not ground_truth_clustering:
            return 0.0

        if len(refined_clustering) != len(ground_truth_clustering):
            print(f"[WARNING] Clustering length mismatch: {len(refined_clustering)} vs {len(ground_truth_clustering)}")
            return 0.0

        # Group refined clusters by ground truth labels
        label_to_clusters = defaultdict(list)
        for refined_cluster, ground_truth_label in zip(refined_clustering, ground_truth_clustering):
            label_to_clusters[ground_truth_label].append(refined_cluster)

        total_events = len(ground_truth_clustering)
        expected_entropy = 0.0

        print(f"\n=== Expected Entropy Analysis: Labels Perspective ===")

        for ground_truth_label, cluster_assignments in label_to_clusters.items():
            label_size = len(cluster_assignments)
            label_entropy = compute_cluster_entropy(cluster_assignments, base=base, normalized=False)
            label_weight = label_size / total_events

            expected_entropy += label_weight * label_entropy

            # Print label details
            cluster_counts = Counter(cluster_assignments)
            print(f"Label {ground_truth_label}: {label_size} events, entropy = {label_entropy:.4f}")
            for cluster, count in cluster_counts.items():
                prob = count / label_size
                print(f"  {cluster}: {count} events ({prob:.1%})")

        print(f"\nExpected Entropy (Labels Perspective): {expected_entropy:.4f}")

        return expected_entropy

    # Use simple functions by default
    get_clustering_from_refined_log = simple_get_clustering_from_refined_log
    get_ground_truth_clustering_from_log = simple_get_ground_truth_clustering_from_log
    compute_expected_entropy_clusters_perspective = simple_compute_expected_entropy_clusters_perspective
    compute_expected_entropy_labels_perspective = simple_compute_expected_entropy_labels_perspective
    
    CLUSTERING_EVALUATION_AVAILABLE = True
    print("Enhanced clustering evaluation available (simplified version)")
    
except ImportError as e:
    print(f"Warning: Could not import clustering evaluation functions: {e}")
    print("Full metrics will not be available for synthetic data.")
    CLUSTERING_EVALUATION_AVAILABLE = False


def run_pipeline_for_artificial_event_logs(input_paths: List[Tuple[str, str]]) -> None:
    """
    Runs pipeline for set of artificial data
    :param input_paths: list of tuples of path and input prefix
    """
    for path, prefix in input_paths:
        input_list = get_tuples_for_folder(path, prefix)[::-1]
        apply_pipeline_to_folder(input_list, prefix, PipelineVariant.VARIANTS, labels_to_split=[], use_noise=False,
                                 use_frequency=True)


def run_pipeline_for_real_log(input_name: str, log_path: str, folder_name: str, labels_to_split: list = None) -> None:
    """
    Runs the pipeline for a real log / individual log

    :param input_name: name to identify the log with
    :param log_path: path to the input event log
    :param folder_name: folder to associate with the log for the outputs
    :param labels_to_split: list of labels to split, or None for auto-detection
    """
    # Auto-detect imprecise labels if not specified
    if labels_to_split is None:
        print("[LABEL_DETECTION] Auto-detecting imprecise labels from log...")
        log = xes_importer.apply(log_path)
        imprecise_labels = set()

        for trace in log:
            for event in trace:
                activity = event.get('concept:name', '')
                org_label = event.get('OrgLabel', '')

                # If OrgLabel exists and differs from concept:name, this is imprecise
                if org_label and org_label != activity:
                    imprecise_labels.add(activity)

        labels_to_split = list(imprecise_labels)
        print(f"[LABEL_DETECTION] Detected imprecise labels: {labels_to_split}")

        if not labels_to_split:
            print("[WARNING] No imprecise labels detected! Using empty list (will process all labels)")
            labels_to_split = []

    apply_pipeline_to_folder([(input_name, log_path)], folder_name,
                             PipelineVariant.VARIANTS,
                             labels_to_split=labels_to_split,
                             use_frequency=True,
                             use_noise=False)


def apply_pipeline_to_folder(input_list: List[Tuple[str, str]], folder_name: str, pipeline_variant: PipelineVariant,
                             labels_to_split: List[str] = None, use_frequency: bool = True,
                             use_noise: bool = True) -> None:
    """
    Apply the whole pipeline to a folder of artificial event log.
    Sets up the output folder, input data for each log and applies the algorithm on the defined parameter space.
    """
    if labels_to_split is None:
        labels_to_split = []

    setup_result_folder(folder_name=folder_name, pipeline_variant=pipeline_variant)

    print("Starting pipeline")
    for (name, path) in input_list:
        input_data, input_preprocessor = set_up_input_data(folder_name, labels_to_split, name, path, pipeline_variant,
                                                           use_frequency, use_noise)

        if input_preprocessor.has_duplicate_xor():
            print('############## Skipped ######################')
            print('Duplicate XOR found, skipping model')
            with open(f'{BEST_RESULTS_DIR}/{input_data.summary_file_name}', 'a') as outfile:
                outfile.write(
                    f'´\n----------------Skipped Model {input_data.input_name} because of duplicate label ------------------------\n')
            continue

        best_score, best_precision, best_configs = run_pipeline_on_parameter_space(input_data)
        try:
            write_summary_file_with_parameters(best_configs, best_score, best_precision, name,
                                               input_data.summary_file_name)
            write_summary_file(best_score, best_precision, input_data.ground_truth_precision, name,
                               input_data.summary_file_name,
                               input_data.xixi_precision, input_data.xixi_ari)
        except Exception as e:
            with open(f'{BEST_RESULTS_DIR}/{input_data.summary_file_name}', 'a') as outfile:
                outfile.write(
                    f'´\n----------------Exception occurred while writing summary file------------------------\n')
                outfile.write(f'{repr(e)}\n')
            continue


def apply_pipeline_to_folder_enhanced(input_list: List[Tuple[str, str]], folder_name: str, pipeline_variant: PipelineVariant,
                                       labels_to_split: List[str] = None, use_frequency: bool = True,
                                       use_noise: bool = True, is_synthetic_data: bool = False) -> None:
    """
    Enhanced version of apply_pipeline_to_folder that supports different metrics for synthetic vs real data.
    
    For synthetic data: Full metrics including Expected Entropy and clustering metrics
    For real data: Simplified metrics focusing on precision, fitness, fscore
    """
    if labels_to_split is None:
        labels_to_split = []

    # Create output directories if they don't exist
    from pathlib import Path
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    Path('./results').mkdir(parents=True, exist_ok=True)

    print("Starting enhanced pipeline")
    print(f"Mode: {'Synthetic' if is_synthetic_data else 'Real-life'} data")
    
    for (name, path) in input_list:
        input_data, input_preprocessor = set_up_input_data(folder_name, labels_to_split, name, path, pipeline_variant,
                                                           use_frequency, use_noise)

        if input_preprocessor.has_duplicate_xor():
            print('############## Skipped ######################')
            print('Duplicate XOR found, skipping model')
            with open(f'{BEST_RESULTS_DIR}/{input_data.summary_file_name}', 'a') as outfile:
                outfile.write(
                    f'´\n----------------Skipped Model {input_data.input_name} because of duplicate label ------------------------\n')
            continue

        best_score, best_precision, best_configs = run_pipeline_on_parameter_space_enhanced(input_data, is_synthetic_data)
        try:
            write_summary_file_with_parameters(best_configs, best_score, best_precision, name,
                                               input_data.summary_file_name)
            write_summary_file(best_score, best_precision, input_data.ground_truth_precision, name,
                               input_data.summary_file_name,
                               input_data.xixi_precision, input_data.xixi_ari)
        except Exception as e:
            with open(f'{BEST_RESULTS_DIR}/{input_data.summary_file_name}', 'a') as outfile:
                outfile.write(
                    f'´\n----------------Exception occurred while writing summary file------------------------\n')
                outfile.write(f'{repr(e)}\n')
            continue


def run_pipeline_on_parameter_space_enhanced(input_data: InputData, is_synthetic_data: bool = False):
    """
    Enhanced version that supports different metrics for synthetic vs real data.
    
    For synthetic data: Full metrics including Expected Entropy and clustering metrics
    For real data: Simplified metrics focusing on precision, fitness, fscore
    """
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        print(f'Starting pipeline for {input_data.input_name}')
        outfile.write(run_start_string())

    best_precision = 0
    best_score = 0
    best_configs = []
    input_data.use_frequency = True

    event_graphs_variant_based = get_event_graphs_from_event_log(input_data.original_log, input_data.labels_to_split)

    for label in input_data.labels_to_split:
        # SYNTHETIC LOGS: Full parameter space k={1,2,3,4,5}, threshold={0,0.1,...,1.0}
        for window_size in [1, 2, 3, 4, 5]:
            for distance in [Distance.EDIT_DISTANCE,
                             Distance.SET_DISTANCE,
                             Distance.MULTISET_DISTANCE]:
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    try:
                        # Pass original log reference - copy will be made inside apply_pipeline_to_log_enhanced
                        found_score, precision, f1_scores_refined = apply_pipeline_to_log_enhanced(
                            input_data, input_data.original_log, distance, window_size, threshold, best_score,
                            event_graphs_variant_based, is_synthetic_data
                        )

                        config_string = get_config_string(clustering_method.ClusteringMethod.COMMUNITY_DETECTION,
                                                          distance,
                                                          input_data.labels_to_split, input_data.max_number_of_traces,
                                                          input_data.log_path, threshold, window_size,
                                                          use_frequency=input_data.use_frequency)
                        if found_score > best_score:
                            best_configs = [config_string]
                            best_score = found_score
                            best_precision = precision
                        elif found_score == best_score:
                            best_configs.append(config_string)

                    except Exception as ex:
                        import traceback
                        error_msg = f"\n{'='*80}\nERROR in parameter combination:\n"
                        error_msg += f"  Label: {label}\n"
                        error_msg += f"  Window: {window_size}, Distance: {distance}, Threshold: {threshold}\n"
                        error_msg += f"  Exception: {ex}\n"
                        error_msg += f"  Traceback:\n{traceback.format_exc()}\n{'='*80}\n"
                        print(error_msg)

                        # Also log to file
                        try:
                            with open(f'{OUTPUT_BASE_DIR}/ERRORS.log', 'a') as error_log:
                                error_log.write(error_msg)
                        except:
                            pass
                    finally:
                        # Force garbage collection after each parameter combination to free memory
                        import gc
                        gc.collect()

    return best_score, best_precision, best_configs


def apply_pipeline_to_log_enhanced(input_data: InputData, log: EventLog, distance_variant: Distance, window_size: int,
                                   threshold: float, current_best_score: float,
                                   event_graphs_variant_based: EventGraphsVariantBased,
                                   is_synthetic_data: bool = False) -> Tuple[float, float, List[float]]:
    """
    Enhanced version of apply_pipeline_to_log with support for different metrics.
    """
    start_time = time.time()

    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        outfile.write(f'\n\n###### Starting pipeline for {threshold}, {distance_variant}, {window_size}######\n')

        # Create a copy of the log here, just before processing
        # This ensures only one copy exists at a time and can be garbage collected immediately after
        log_copy = copy.deepcopy(log)

        # Get label splitter and split the log
        label_splitter = get_label_splitter(distance_variant, input_data, outfile, threshold, window_size,
                                            event_graphs_variant_based)
        split_log = label_splitter.split_labels(log_copy)
        labels_to_original = label_splitter.get_split_labels_to_original_labels()

        # Calculate unrefined log metrics
        calculate_unrefined_log_precision(input_data, label_splitter, labels_to_original, outfile)

        # Get evaluation metrics
        final_marking, initial_marking, final_net, precision, simplicity, generalization, fitness = apply_im_without_noise_and_evaluate(
            labels_to_original, input_data.original_log, split_log, outfile, label_splitter.short_label_to_original_label)

        # Calculate ARI score
        ari_score = 0
        if input_data.ground_truth_clustering and len(label_splitter.found_clustering) > 0:
            # Extract membership arrays from igraph Clustering objects
            # igraph Clustering objects need .membership to get flat cluster assignments
            xixi_membership = input_data.xixi_clustering.membership if hasattr(input_data.xixi_clustering, 'membership') else input_data.xixi_clustering
            found_membership = label_splitter.found_clustering.membership if hasattr(label_splitter.found_clustering, 'membership') else label_splitter.found_clustering

            # Check lengths match
            if len(xixi_membership) != len(found_membership):
                outfile.write(f'\n⚠️  Clustering length mismatch: xixi={len(xixi_membership)}, found={len(found_membership)}\n')
                outfile.write(f'   Using 0 for ARI score\n')
                ari_score = 0
            else:
                ari_score = adjusted_rand_score(xixi_membership, found_membership)

        runtime = time.time() - start_time

        # Calculate additional metrics for synthetic data
        additional_metrics = {}
        if is_synthetic_data and CLUSTERING_EVALUATION_AVAILABLE:
            try:
                # Get clustering from refined log (only imprecise events)
                refined_clustering = get_clustering_from_refined_log(split_log, input_data.labels_to_split)
                ground_truth_clustering = get_ground_truth_clustering_from_log(input_data.original_log, input_data.labels_to_split)
                
                # Calculate expected entropy metrics
                expected_entropy_clusters = compute_expected_entropy_clusters_perspective(
                    ground_truth_clustering, refined_clustering)
                expected_entropy_labels = compute_expected_entropy_labels_perspective(
                    ground_truth_clustering, refined_clustering)
                
                # Calculate NMI
                if len(ground_truth_clustering) > 0 and len(refined_clustering) > 0:
                    nmi_score = normalized_mutual_info_score(ground_truth_clustering, refined_clustering)
                else:
                    nmi_score = 0
                    
                additional_metrics = {
                    'expected_entropy_clusters': expected_entropy_clusters,
                    'expected_entropy_labels': expected_entropy_labels,
                    'nmi': nmi_score
                }
                
                print(f"Expected Entropy Clusters: {expected_entropy_clusters:.4f}")
                print(f"Expected Entropy Labels: {expected_entropy_labels:.4f}")
                print(f"NMI Score: {nmi_score:.4f}")
                
            except Exception as e:
                print(f"Warning: Could not calculate enhanced metrics: {e}")
                additional_metrics = {
                    'expected_entropy_clusters': 0,
                    'expected_entropy_labels': 0,
                    'nmi': 0
                }

        # Write results using enhanced CSV function
        write_results_to_csv_enhanced(ari_score, distance_variant, fitness, generalization, input_data,
                                      label_splitter, precision, runtime, simplicity, threshold, window_size,
                                      is_synthetic_data, additional_metrics)

        # Export models if new best score found
        if ari_score > current_best_score:
            print(f'\nHigher Adjusted Rand Index found: {ari_score}')
            print(f'\nPrecision of found clustering: {precision}')
            tree = inductive_miner.apply(split_log)
            export_models_and_pngs(final_marking, initial_marking, final_net, tree, input_data.input_name,
                                   f'{input_data.input_name}_{threshold}_{distance_variant}_{window_size}_split_log')

        f1_scores_refined = []  # Placeholder for compatibility
        return ari_score, precision, f1_scores_refined


def set_up_input_data(folder_name: str, labels_to_split: List[str], name: str, path: str,
                      pipeline_variant: PipelineVariant, use_frequency: bool, use_noise: bool) -> Tuple[
    InputData, InputPreprocessor]:
    """
    Generates the input data used throughout the pipeline for one event log.
    Calculates the original performance, unrefined performance, xixi performance and sets up naming

    :return: Preprocessed input data and input preprocessor
    """
    input_data = InputData(original_input_name=name,
                           log_path=path,
                           pipeline_variant=pipeline_variant,
                           labels_to_split=labels_to_split,
                           use_frequency=use_frequency,
                           use_noise=use_noise,
                           max_number_of_traces=100,
                           folder_name=folder_name)
    input_data.original_log = xes_importer.apply(input_data.log_path, parameters={
        xes_importer.Variants.ITERPARSE.value.Parameters.MAX_TRACES: input_data.max_number_of_traces})
    input_data.input_name = f'{input_data.original_input_name}_{input_data.pipeline_variant}' if use_frequency else f'{input_data.original_input_name}_{input_data.pipeline_variant}'
    input_data.use_combined_context = False
    input_data.concurrent_labels = []
    summary_file_name = f'{folder_name}_{pipeline_variant}.txt' if use_frequency else f'{folder_name}_{pipeline_variant}.txt'
    input_data.summary_file_name = summary_file_name

    input_preprocessor = InputPreprocessor(input_data)
    input_preprocessor.preprocess_input()

    return input_data, input_preprocessor


def run_pipeline_on_parameter_space(input_data: InputData):
    """
    Runs the pipeline for one input data object on the whole parameter space.

    :return: Best results found and the configs used for the result
    """
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        print(f'Starting pipeline for {input_data.input_name}')
        outfile.write(run_start_string())

    best_precision = 0
    best_score = 0
    best_configs = []
    input_data.use_frequency = True

    event_graphs_variant_based = get_event_graphs_from_event_log(input_data.original_log, input_data.labels_to_split)

    for label in input_data.labels_to_split:
        # SYNTHETIC LOGS: Full parameter space k={1,2,3,4,5}, threshold={0,0.1,...,1.0}
        for window_size in [1, 2, 3, 4, 5]:
            for distance in [Distance.EDIT_DISTANCE,
                             Distance.SET_DISTANCE,
                             Distance.MULTISET_DISTANCE]:
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    try:
                        # Make a copy only when calling the function to avoid keeping multiple copies in memory
                        # The copy will be created inside apply_pipeline_to_log where it's needed
                        found_score, precision, f1_scores_refined = apply_pipeline_to_log(input_data, input_data.original_log, distance,
                                                                                          window_size, threshold,
                                                                                          best_score,
                                                                                          event_graphs_variant_based)

                        config_string = get_config_string(clustering_method.ClusteringMethod.COMMUNITY_DETECTION,
                                                          distance,
                                                          input_data.labels_to_split, input_data.max_number_of_traces,
                                                          input_data.log_path, threshold, window_size,
                                                          use_frequency=input_data.use_frequency)
                        if found_score > best_score:
                            best_configs = [config_string]
                            best_score = found_score
                            best_precision = precision
                        elif round(found_score, 2) == round(best_score, 2):
                            best_configs.append(config_string)
                    except Exception as e:
                        print('----------------Exception occurred while running pipeline ------------------------')
                        print(repr(e))

                        with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
                            outfile.write(f'´\n----------------Exception occurred------------------------\n')
                            outfile.write(f'{window_size}\n')
                            outfile.write(f'{threshold}\n')
                            outfile.write(f'{distance}\n')
                            outfile.write(f'{input_data.use_combined_context}\n')
                        continue
                    finally:
                        # Force garbage collection after each parameter combination to free memory
                        import gc
                        gc.collect()

    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        outfile.write('Best precision found:\n')
        outfile.write(str(best_precision))
    return best_score, best_precision, best_configs


def apply_pipeline_to_log(input_data: InputData,
                          log: EventLog,
                          distance_variant: Distance,
                          window_size: int,
                          threshold: float,
                          best_score: float,
                          event_graphs_variant_based: EventGraphsVariantBased):
    """
    Applies the algorithm with the specified to the input event log

    :return: ARI, precision and F1-scores of the model generated from the refined event log
    """
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        outfile.write(get_config_string(clustering_method.ClusteringMethod.COMMUNITY_DETECTION, distance_variant,
                                        input_data.labels_to_split, input_data.max_number_of_traces,
                                        input_data.log_path, threshold, window_size, input_data.use_frequency))
        start = time.time()

        # Create a copy of the log here, just before processing
        # This ensures only one copy exists at a time and can be garbage collected immediately after
        log_copy = copy.deepcopy(log)

        # Apply the label splitting algorithm
        label_splitter = get_label_splitter(distance_variant, input_data, outfile,
                                            threshold, window_size, event_graphs_variant_based)

        split_log = label_splitter.split_labels(log_copy)
        split_log_clustering = filter_duplicate_xor(split_log, input_data.labels_to_split,
                                                    label_splitter.found_clustering)
        end = time.time()
        runtime = end - start
        outfile.write(f'\nRuntime: {runtime}\n')
        outfile.write('\nPerformance split_log:\n')
        outfile.write('\nIM without threshold:\n')
        labels_to_original = label_splitter.get_split_labels_to_original_labels()

        # Export the split log
        xes_exporter.apply(split_log,
                           f'{OUTPUT_BASE_DIR}/{input_data.input_name}_{threshold}_{window_size}_{distance_variant}.xes.gz')

        calculate_unrefined_log_precision(input_data, label_splitter, labels_to_original, outfile)

        # Generate a model from the split data and evaluate it.
        final_marking, initial_marking, final_net, precision, simplicity, generalization, fitness = apply_im_without_noise_and_evaluate(
            labels_to_original,
            split_log,
            input_data.original_log,
            outfile,
            label_splitter.short_label_to_original_label)

        f1_scores_refined = []
        if input_data.ground_truth_clustering:
            ari_score = get_community_similarity(input_data.ground_truth_clustering, split_log_clustering)
        else:
            ari_score = precision
        outfile.write(f'\nAdjusted Rand Index:\n')
        outfile.write(f'{ari_score}\n\n')

        write_results_to_csv(ari_score, distance_variant, fitness, generalization, input_data, label_splitter,
                             precision, runtime, simplicity, threshold, window_size)

        if ari_score > best_score:
            # Export everything is new best model was found
            print(f'\nHigher Adjusted Rand Index found: {ari_score}')
            print(f'\nPrecision of found clustering: {precision}')
            # Fix: Use apply instead of apply_tree
            tree = inductive_miner.apply(split_log)
            export_models_and_pngs(final_marking, initial_marking, final_net, tree, input_data.input_name,
                                   f'{input_data.input_name}_{threshold}_{distance_variant}_{window_size}_split_log')

        return ari_score, precision, f1_scores_refined

def calculate_unrefined_log_precision(input_data, label_splitter, labels_to_original, outfile):
    if input_data.original_log_precision == 0:
        print('Starting to get original performance')
        final_marking, initial_marking, final_net, original_precision, original_simplicity, original_generalization, original_fitness = apply_im_without_noise_and_evaluate(
            labels_to_original,
            input_data.original_log,
            input_data.original_log,
            outfile,
            label_splitter.short_label_to_original_label)
        input_data.original_log_precision = original_precision
        input_data.original_log_simplicity = original_simplicity
        input_data.original_log_generalization = original_generalization
        input_data.original_log_fitness = original_fitness
        print('finished original log calculation')


def write_results_to_csv(ari_score: float, distance_variant: Distance, fitness: float, generalization: float,
                         input_data: InputData, label_splitter: LabelSplitterVariantBased, precision: float,
                         runtime: float, simplicity: float, threshold: float, window_size: int) -> None:
    with open(f'{RESULTS_BASE_DIR}/{input_data.folder_name}_{input_data.pipeline_variant}_NEW.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if input_data.ground_truth_clustering:
            row = [input_data.original_input_name, input_data.max_number_of_traces,
                   ' '.join(input_data.labels_to_split),
                   ', '.join(input_data.original_labels), input_data.original_log_precision,
                   input_data.original_log_simplicity, input_data.original_log_generalization,
                   input_data.original_log_fitness,
                   len(input_data.xixi_clustering),
                   input_data.xixi_precision, input_data.xixi_ari,
                   input_data.use_combined_context, input_data.use_frequency, window_size, distance_variant, threshold,
                   len(label_splitter.found_clustering), precision, ari_score, simplicity, generalization, fitness,
                   runtime]
        else:
            row = [input_data.original_input_name, input_data.max_number_of_traces,
                   ' '.join(input_data.labels_to_split),
                   '[]', input_data.original_log_precision,
                   input_data.original_log_simplicity, input_data.original_log_generalization,
                   input_data.original_log_fitness,
                   0, 0, 0,
                   input_data.use_combined_context, input_data.use_frequency, window_size, distance_variant, threshold,
                   len(label_splitter.found_clustering), precision, ari_score, simplicity, generalization,
                   fitness, runtime]
        writer.writerow(row)


def write_results_to_csv_enhanced(ari_score: float, distance_variant: Distance, fitness: float, generalization: float,
                                  input_data: InputData, label_splitter, precision: float,
                                  runtime: float, simplicity: float, threshold: float, window_size: int,
                                  is_synthetic_data: bool = False, additional_metrics: dict = None) -> None:
    """
    Enhanced CSV writer that supports different output formats for synthetic vs real data.

    For synthetic data: Full metrics including Expected Entropy and clustering metrics
    For real data: Simplified metrics focusing on precision, fitness, fscore

    Saves one CSV per log file (matching Label-Refinement format)
    """
    # Extract log name from input_data.original_input_name (e.g., "feb16-1625/A_1" -> "A_1")
    log_name = input_data.original_input_name.split('/')[-1] if '/' in input_data.original_input_name else input_data.original_input_name

    # Create one CSV per log file: {folder_name}_{log_name}_{variant}_ENHANCED.csv
    csv_filename = f'{RESULTS_BASE_DIR}/{input_data.folder_name}_{log_name}_{input_data.pipeline_variant}_ENHANCED.csv'

    # Check if we need to write headers (file doesn't exist or is empty)
    write_headers = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0

    # Safety: Handle None clustering
    num_clusters_found = len(label_splitter.found_clustering) if label_splitter.found_clustering is not None else 0

    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_headers:
            if is_synthetic_data:
                # Full headers for synthetic data (26 columns)
                headers = [
                    'Name', 'max_number_of_traces', 'labels_to_split', 'original labels',
                    'original_precision', 'original_simplicity', 'original_generalization', 'original_fitness',
                    'Xixi number of Clusters found', 'Xixi Precision', 'Xixi ARI',
                    'use_combined_context', 'use_frequency', 'window_size', 'distance_metric', 'threshold',
                    'Number of Clusters found', 'Precision Align', 'ARI', 'Simplicity', 'Generalization',
                    'Fitness', 'Runtime', 'Expected Entropy Clusters', 'Expected Entropy Labels', 'NMI (Normalized Mutual Information)'
                ]
            else:
                # Simplified headers for real data (13 columns)
                headers = [
                    'Log_Name', 'Max_Traces', 'Labels_To_Split',
                    'Original_Precision', 'Original_Fitness', 'Original_FScore',
                    'Use_Combined_Context', 'Use_Frequency', 'Window_Size', 'Distance_Variant', 'Threshold',
                    'Refined_Precision', 'Refined_Fitness', 'Refined_FScore', 'Runtime'
                ]
            writer.writerow(headers)
        
        if is_synthetic_data:
            # Calculate additional metrics for synthetic data
            expected_entropy_clusters = 0
            expected_entropy_labels = 0
            nmi_score = 0
            
            if CLUSTERING_EVALUATION_AVAILABLE and additional_metrics:
                expected_entropy_clusters = additional_metrics.get('expected_entropy_clusters', 0)
                expected_entropy_labels = additional_metrics.get('expected_entropy_labels', 0)
                nmi_score = additional_metrics.get('nmi', 0)
            
            # Full row for synthetic data
            if input_data.ground_truth_clustering:
                row = [
                    input_data.original_input_name, input_data.max_number_of_traces,
                    ' '.join(input_data.labels_to_split), ', '.join(input_data.original_labels),
                    input_data.original_log_precision, input_data.original_log_simplicity,
                    input_data.original_log_generalization, input_data.original_log_fitness,
                    len(input_data.xixi_clustering), input_data.xixi_precision, input_data.xixi_ari,
                    input_data.use_combined_context, input_data.use_frequency, window_size, str(distance_variant).replace('Distance.', ''), threshold,
                    num_clusters_found, precision, ari_score, simplicity, generalization, fitness,
                    runtime, expected_entropy_clusters, expected_entropy_labels, nmi_score
                ]
            else:
                row = [
                    input_data.original_input_name, input_data.max_number_of_traces,
                    ' '.join(input_data.labels_to_split), '[]',
                    input_data.original_log_precision, input_data.original_log_simplicity,
                    input_data.original_log_generalization, input_data.original_log_fitness,
                    0, 0, 0,
                    input_data.use_combined_context, input_data.use_frequency, window_size, str(distance_variant).replace('Distance.', ''), threshold,
                    num_clusters_found, precision, ari_score, simplicity, generalization, fitness,
                    runtime, expected_entropy_clusters, expected_entropy_labels, nmi_score
                ]
        else:
            # Simplified row for real data - calculate F-scores
            original_fscore = 2 * (input_data.original_log_precision * input_data.original_log_fitness) / (
                input_data.original_log_precision + input_data.original_log_fitness
            ) if (input_data.original_log_precision + input_data.original_log_fitness) > 0 else 0
            
            refined_fscore = 2 * (precision * fitness) / (precision + fitness) if (precision + fitness) > 0 else 0
            
            row = [
                input_data.original_input_name, input_data.max_number_of_traces,
                ' '.join(input_data.labels_to_split),
                input_data.original_log_precision, input_data.original_log_fitness, original_fscore,
                input_data.use_combined_context, input_data.use_frequency, window_size, distance_variant, threshold,
                precision, fitness, refined_fscore, runtime
            ]
        
        writer.writerow(row)


def get_label_splitter(distance_variant: Distance, input_data: InputData, outfile, threshold, window_size,
                       event_graphs_variant_based: EventGraphsVariantBased):
    if input_data.pipeline_variant == PipelineVariant.VARIANTS:
        label_splitter = LabelSplitterVariantBased(outfile,
                                                   input_data.labels_to_split,
                                                   threshold=threshold,
                                                   window_size=window_size,
                                                   distance_variant=distance_variant,
                                                   clustering_variant=clustering_method.ClusteringMethod.COMMUNITY_DETECTION,
                                                   use_frequency=input_data.use_frequency,
                                                   concurrent_labels=input_data.concurrent_labels,
                                                   use_combined_context=input_data.use_combined_context,
                                                   event_graphs_variant_based=event_graphs_variant_based)
    elif input_data.pipeline_variant == PipelineVariant.VARIANTS_MULTIPLEX:
        label_splitter = LabelSplitterVariantMultiplex(outfile,
                                                       input_data.labels_to_split,
                                                       threshold=threshold,
                                                       window_size=window_size,
                                                       distance_variant=distance_variant,
                                                       clustering_variant=clustering_method.ClusteringMethod.COMMUNITY_DETECTION,
                                                       use_frequency=input_data.use_frequency,
                                                       use_combined_context=input_data.use_combined_context)
    else:
        label_splitter = LabelSplitterEventBased(outfile,
                                                 input_data.labels_to_split,
                                                 threshold=threshold,
                                                 window_size=window_size,
                                                 distance_variant=distance_variant,
                                                 clustering_variant=clustering_method.ClusteringMethod.COMMUNITY_DETECTION,
                                                 use_combined_context=input_data.use_combined_context)
    return label_splitter
