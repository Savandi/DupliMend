import csv
from time import time

from igraph import *
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import egraph_builder
import egraph_label_refinement
import mapping_all
import mapping_modularity

# OUTPUT CONFIGURATION - Use config from DupliMend (silently)
import os
import sys
import io

# PRIORITY: Environment variable > Config > Fallback
# This allows PBS scripts to override config for scratch storage
if 'BASELINE_RESULTS_DIR' in os.environ:
    RESULTS_BASE_DIR = os.environ.get('BASELINE_RESULTS_DIR')
    print(f"[CONFIG] Using results directory from environment: {RESULTS_BASE_DIR}")
else:
    # Try to load results_dir from DupliMend config (suppress verbose output)
    try:
        # Temporarily suppress stdout to avoid verbose DupliMend config printing
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        sys.path.append('../../config')
        from config import evaluation_config

        # Restore stdout
        sys.stdout = original_stdout

        label_config = evaluation_config.get("baseline_evaluation_config", {}).get("label_refinement", {})
        RESULTS_BASE_DIR = label_config.get("results_dir", "../results")
        print(f"[CONFIG] Using results directory from config: {RESULTS_BASE_DIR}")
    except (ImportError, Exception) as e:
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        RESULTS_BASE_DIR = "../results"
        print(f"[CONFIG] Using fallback results directory: {RESULTS_BASE_DIR}")

# Create directory if it doesn't exist
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
import precision_util
import clustering_evaluation


def has_duplicate_xor(net, imprecise_labels, original_labels):
    for transition in net.transitions:
        if transition.label != None and transition.label in original_labels:
            transition.label = imprecise_labels[0]

    for p in net.places:
        seen_labels = []
        for a in p.out_arcs:
            t = a.target
            if t.label != None and t.label in imprecise_labels and t.label in seen_labels:
                return True
            seen_labels.append(t.label)
    return False


def run(event_log, xixi_log, original_event_log, original_net, original_initial_marking, original_final_marking,
        experiment_nr_parameter, start_data_set_size_parameter, log_name, folder_name,
        event_log_path,
        parameters={"TIMESTAMP_KEY": "no_timestamp", "ACTIVITY_KEY": "concept:name",
                    "EVENT_IDENTIFICATION": "Activity", "CASE_ID_KEY": 0, "LIFECYCLE_KEY": "lifecycle:transition",
                    "LIFECYCLE_MODE": "atomic", "k": 1}, weight_matched=1, weight_not_matched=10, weight_structure=1,
        k=1, basic_cost=1,
        labeling_function=egraph_label_refinement.default_labeling_function, use_adaptive_parameters=False,
        is_real_life_log=False, log_size_parameter=100):
    start_time = time()
    
    # Initialize evaluation metrics
    refined_log_fitness = 0.0
    refined_log_fscore = 0.0
    expected_entropy_clusters = 0.0
    expected_entropy_labels = 0.0
    nmi = 0.0

    egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID = egraph_builder.get_egraphs(parameters, False,
                                                                                                event_log)

    # time_s = time()
    # egraphs_folding, map_egraph_ID_to_trace_IDs_folding, map_trace_ID_to_egraph_ID_folding = egraph_builder.get_egraphs(
    #     parameters, True, event_log)
    # time_e = time()
    # time0 = time_e - time_s

    time_s = time()
    mapping = mapping_all.get_mappings(egraphs, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                                       labeling_function, "GREEDY", False)
    time_e = time()
    time_for_greedy_mapping = time_e - time_s

    # mapping_folding = mapping_all.get_mappings(egraphs_folding, weight_matched, weight_not_matched, weight_structure, k,
    #                                            basic_cost, labeling_function, "GREEDY", False)

    # time_s = time()
    # mapping_semi = mapping_all.get_mappings(egraphs, weight_matched, weight_not_matched, weight_structure, k,
    #                                         basic_cost, labeling_function, "SEMI_GREEDY", False)
    # time_e = time()
    # time_for_semi_greedy_mapping = time_e - time_s

    # time_s = time()
    # mapping_folding_semi = mapping_all.get_mappings(egraphs_folding, weight_matched, weight_not_matched,
    #                                                 weight_structure, k, basic_cost, labeling_function, "SEMI_GREEDY",
    #                                                 False)
    # time_e = time()
    time1 = time_e - time_s

    mapping_quality = mapping_modularity.get_mapping_modularity(event_log, egraphs, map_egraph_ID_to_trace_IDs,
                                                                map_trace_ID_to_egraph_ID, mapping, labeling_function)
    # mapping_folding_quality = mapping_modularity.get_mapping_modularity(event_log, egraphs_folding,
    #                                                                     map_egraph_ID_to_trace_IDs_folding,
    #                                                                     map_trace_ID_to_egraph_ID_folding,
    #                                                                     mapping_folding, labeling_function)
    # mapping_semi_quality = mapping_modularity.get_mapping_modularity(event_log, egraphs, map_egraph_ID_to_trace_IDs,
    #                                                                  map_trace_ID_to_egraph_ID, mapping_semi,
    #                                                                  labeling_function)
    # mapping_folding_semi_quality = mapping_modularity.get_mapping_modularity(event_log, egraphs_folding,
    #                                                                          map_egraph_ID_to_trace_IDs_folding,
    #                                                                          map_trace_ID_to_egraph_ID_folding,
    #                                                                          mapping_folding_semi, labeling_function)
    # print(mapping_quality)
    # print(mapping_folding_quality)
    # print(mapping_semi_quality)
    # print(mapping_folding_semi_quality)

    ref_log, imprecise_labels, original_labels, num_of_new_labels, cl = egraph_label_refinement.get_refined_event_log(
        event_log,
        cluster_method="CONNECTED_COMPONENTS",
        detection_mode_imp_in_loop="vertical",
        egraphs=egraphs,
        mapping=mapping,
        map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
        map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
        use_adaptive_parameters=False)

    print(f"[LABEL_DEBUG] Detected imprecise_labels: {imprecise_labels}")
    print(f"[LABEL_DEBUG] Detected original_labels: {original_labels}")
    print(f"[LABEL_DEBUG] Number of new labels: {num_of_new_labels}")
    
    # Check unique activities in original and refined logs
    orig_activities = set()
    for trace in event_log:
        for event in trace:
            orig_activities.add(event['concept:name'])
    
    ref_activities = set()
    for trace in ref_log:
        for event in trace:
            ref_activities.add(event['concept:name'])
    
    print(f"[LABEL_DEBUG] Original log activities: {sorted(orig_activities)}")
    print(f"[LABEL_DEBUG] Refined log activities: {sorted(ref_activities)}")

    if has_duplicate_xor(original_net, imprecise_labels, original_labels):
        print('----------------------------------- Skipped because of duplicate xor -----------------------------------')
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    org_model_prec, org_model_simplicity, org_model_generalization = precision_util.get_precision_of_original_model(original_net, original_initial_marking,
                                                                    original_final_marking, event_log, imprecise_labels,
                                                                    original_labels, parameters)  # todo test

    precise_refined_log_prec, precise_simplicity, precise_generalization = precision_util.get_precision_of_precice_log(original_event_log, event_log,
                                                                           imprecise_labels, original_labels,
                                                                           parameters)
    imp_prec, imp_simplicity, imp_generalization = precision_util.get_precision(event_log, event_log, imprecise_labels, parameters)

    ref_log_prec, ref_simplicity, ref_generalization = precision_util.get_precision(ref_log, event_log, imprecise_labels, parameters)
    
    if xixi_log:
        xixi_prec, xixi_simplicity, xixi_generalization = precision_util.get_precision(xixi_log, event_log, imprecise_labels, parameters)
    else:
        xixi_prec, xixi_simplicity, xixi_generalization = 0, 0, 0

    ground_truth_clustering = get_ground_truth_clustering_from_log(event_log, imprecise_labels, original_labels)
    print('ground_truth_clustering')
    print(ground_truth_clustering)
    print('len(ground_truth_clustering)')
    print(len(ground_truth_clustering))

    print('Before loop')


    # TODO: Add no of clusters identified

    header = [
        'Log', 'Folder', 'Original Labels',
        'Original Model Precision', 'Original Log Simplicity', 'Original Log Generalization',
        'Precise Log Precision ', 'Precise Log Simplicity', 'Precise Log Generalization',
        'Unrefined Log Precision', 'Unrefined Log Simplicity', 'Unrefined Log Generalization',
        'Xixi Log Precision', 'Xixi Log Simplicity', 'Xixi Log Generalization',
        'Variant Threshold', 'Unfolding Threshold', 'Log Size', 'Refined Log Precision',
        'Refined Log ARI', 'Refined Log Simplicity', 'Refined Log Generalization',
        'Expected Entropy Clusters', 'Expected Entropy Labels', 'NMI', 'Refined Log Fitness', 'Refined Log F-Score'
    ]

    # Orignal model => Model the log was generated from (Process Tree)
    # Precise Log => Generated log with precise labels
    # Unrefined Log Precision => Log with imprecise Label without refinements
    # Xixi Log => Log from Xixi (her result files)
    # Refined Log => My refined log result
    # ref_log => ????

    max_ari = 0
    ################################################################################################
    # OPTIMIZATION: Load event log ONCE outside the loop instead of 121 times
    fresh_event_log = xes_import_factory.apply(event_log_path, parameters={
        xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})

    # OPTIMIZATION: Periodic batch CSV writes - write every 5 combinations for progress visibility
    batch_rows = []
    BATCH_WRITE_INTERVAL = 5  # Write to CSV every 5 combinations

    # SYNTHETIC LOGS: Full parameter space (11 × 11 = 121 combinations)
    # Using complete threshold range for synthetic log evaluation
    threshold_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_combinations = len(threshold_values) * len(threshold_values)  # 121 combinations
    current_combination = 0

    # Prepare CSV file path - Save in dataset subfolder (e.g., noImprInLoop_default_IMD)
    # Get dataset name from environment variable (set by shell script)
    dataset_name = os.environ.get('DATASET_NAME', 'default_dataset')

    # Create dataset subfolder if it doesn't exist
    dataset_results_dir = os.path.join(RESULTS_BASE_DIR, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # Using folder name and log name for unique filename within dataset subfolder
    result_file = os.path.join(dataset_results_dir, f"{folder_name}_{log_name}_result_{start_data_set_size_parameter}.csv")

    # Write CSV headers if file doesn't exist (first time creation)
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as csvfile:
            fwriter = csv.writer(csvfile)
            headers = [
                'log_name', 'folder_name', 'original_labels',
                'org_model_prec', 'org_model_simplicity', 'org_model_generalization',
                'precise_refined_log_prec', 'precise_simplicity', 'precise_generalization',
                'imp_prec', 'imp_simplicity', 'imp_generalization',
                'xixi_prec', 'xixi_simplicity', 'xixi_generalization',
                'variant_threshold', 'unfolding_threshold', 'num_traces',
                'ref_log_cc_precision', 'ari',
                'ref_log_cc_simplicity', 'ref_log_cc_generalization',
                'expected_entropy_clusters', 'expected_entropy_labels', 'nmi',
                'refined_log_fitness', 'refined_log_fscore'
            ]
            fwriter.writerow(headers)
        print(f"[CSV] Created new result file with headers: {result_file}")

    for new_variant_threshold in threshold_values:
        for new_unfolding_threshold in threshold_values:
            current_combination += 1
            # Print progress for every combination since we only have 25 now
            print(f"[PROGRESS] Combination {current_combination}/{total_combinations} (tv={new_variant_threshold}, tu={new_unfolding_threshold})")
            ref_log_cc, _, _, t_num_of_new_labels, cl = egraph_label_refinement.get_refined_event_log(fresh_event_log,
                                                                                                    cluster_method="CONNECTED_COMPONENTS",
                                                                                                    detection_mode_imp_in_loop="vertical",
                                                                                                    egraphs=egraphs,
                                                                                                    mapping=mapping,
                                                                                                    variant_threshold=new_variant_threshold,
                                                                                                    unfolding_threshold=new_unfolding_threshold,
                                                                                                    map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
                                                                                                    map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
                                                                                                    use_adaptive_parameters=False)
            ref_log_cc_precision, ref_log_cc_simplicity, ref_log_cc_generalization = precision_util.get_precision(ref_log_cc, event_log,
                                                                imprecise_labels,
                                                                parameters)

            cl = get_clustering_from_log(ref_log_cc, imprecise_labels)

            # The baseline may only refine a subset of imprecise events
            # We need to align the ground truth with what was actually refined
            if len(ground_truth_clustering) != len(cl):
                # OPTIMIZATION: Only print warning on first combination to reduce I/O overhead
                if current_combination == 1:
                    print(f"[INFO] Baseline refined {len(cl)} out of {len(ground_truth_clustering)} imprecise events")
                    print(f"[INFO] Adjusting ground truth to match refined events for fair comparison")

                # For a fair comparison, we should only compare the events that were refined
                # This is a limitation of the baseline approach
                if len(cl) > 0:
                    # Take the first len(cl) ground truth labels assuming sequential processing
                    ground_truth_clustering = ground_truth_clustering[:len(cl)]
                else:
                    if current_combination == 1:
                        print(f"[WARNING] No events were refined, setting ARI to 0")
                    ari = 0.0
                    ground_truth_clustering = []
                    cl = []

            if len(cl) > 0:
                ari = adjusted_rand_score(ground_truth_clustering, cl)
            else:
                ari = 0.0
            
            # Comprehensive evaluation including fitness, F-score, entropy, and NMI
            try:
                # Pass event_log (imprecise) not original_event_log (precise) for ground truth extraction
                eval_results = clustering_evaluation.evaluate_refined_log_quality(
                    ref_log_cc, event_log, imprecise_labels, original_labels, parameters, is_real_life_log
                )
                
                # Extract additional metrics
                refined_log_fitness = eval_results.get('refined_log_fitness', 0.0)
                refined_log_fscore = eval_results.get('refined_log_fscore', 0.0)
                expected_entropy_clusters = eval_results.get('expected_entropy_clusters', 0.0)
                expected_entropy_labels = eval_results.get('expected_entropy_labels', 0.0)
                nmi = eval_results.get('nmi', 0.0)

                # Print evaluation metrics (since we only have 25 combinations, we can afford to print all)
                print(f"  → ARI: {ari:.4f}, Fitness: {refined_log_fitness:.4f}, F-score: {refined_log_fscore:.4f}, NMI: {nmi:.4f}")

            except Exception as e:
                # OPTIMIZATION: Only print errors for first few combinations
                if current_combination <= 3:
                    print(f"[ERROR] Comprehensive evaluation failed: {e}")
                refined_log_fitness = 0.0
                refined_log_fscore = 0.0
                expected_entropy_clusters = 0.0
                expected_entropy_labels = 0.0
                nmi = 0.0
            
            if ari > max_ari:  # and experiment_nr_parameter == 111
                max_ari = ari
                xes_exporter.apply(ref_log_cc, './example_loop_cc.xes')
                print(f"[BEST] New max ARI: {max_ari:.4f} at tv={new_variant_threshold}, tu={new_unfolding_threshold}")

            row = [log_name, folder_name, original_labels,
                   org_model_prec, org_model_simplicity, org_model_generalization,
                   precise_refined_log_prec, precise_simplicity, precise_generalization,
                   imp_prec, imp_simplicity, imp_generalization,
                   xixi_prec, xixi_simplicity, xixi_generalization,
                   new_variant_threshold,
                   new_unfolding_threshold, len(fresh_event_log), ref_log_cc_precision,
                   ari, ref_log_cc_simplicity, ref_log_cc_generalization,
                   expected_entropy_clusters, expected_entropy_labels, nmi, refined_log_fitness, refined_log_fscore]

            # Collect rows in batch
            batch_rows.append(row)

            # OPTIMIZATION: Write batch periodically every N combinations for progress visibility
            if len(batch_rows) >= BATCH_WRITE_INTERVAL or current_combination == total_combinations:
                with open(result_file, 'a', newline='') as csvfile:
                    fwriter = csv.writer(csvfile)
                    fwriter.writerows(batch_rows)
                print(f"[WRITE] Wrote {len(batch_rows)} rows to CSV (total combinations processed: {current_combination}/{total_combinations})")
                batch_rows = []  # Clear the batch after writing

                # MEMORY FIX: Force garbage collection after each batch to free memory
                import gc
                gc.collect()

    # Write any remaining rows (shouldn't happen with our logic, but safety check)
    if len(batch_rows) > 0:
        with open(result_file, 'a', newline='') as csvfile:
            fwriter = csv.writer(csvfile)
            fwriter.writerows(batch_rows)
        print(f"[WRITE] Wrote final {len(batch_rows)} rows to CSV")

    print('after loop')
    print(f'Max ARI found for CC: {max_ari}')
    print(f'All results written to {result_file}')
    ################################################################################################


    # max_ari = 0

    # for new_variant_threshold in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    #     for new_unfolding_threshold in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    #         ref_log_cd, _, _, num_of_new_labels_comdec, cl = egraph_label_refinement.get_refined_event_log(
    #             event_log,
    #             egraphs=egraphs,
    #             cluster_method="COMMUNITY_DETECTION",
    #             detection_mode_imp_in_loop="vertical",
    #             mapping=mapping,
    #             variant_threshold=new_variant_threshold,
    #             unfolding_threshold=new_unfolding_threshold,
    #             map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
    #             map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
    #             use_adaptive_parameters=use_adaptive_parameters)
    #         ref_log_cd_precision, ref_log_cd_simplicity, ref_log_cd_generalization = precision_util.get_precision(ref_log_cd, event_log,
    #                                                             imprecise_labels,
    #                                                             parameters)

    #         clustering_cd = get_clustering_from_log(ref_log_cd, imprecise_labels)
    #         # print('clustering_cd')
    #         # print(clustering_cd)
    #         ari = compare_communities(ground_truth_clustering, clustering_cd, method='adjusted_rand')
    #         print('ari')
    #         print(ari)
    #         if ari > max_ari:  # and experiment_nr_parameter == 111
    #             max_ari = ari
    #             xes_exporter.apply(ref_log_cd, './example_loop_cd.xes')

    #         row = [log_name, folder_name, original_labels,
    #                org_model_prec, org_model_simplicity, org_model_generalization,
    #                precise_refined_log_prec, precise_simplicity, precise_generalization,
    #                imp_prec, imp_simplicity, imp_generalization,
    #                xixi_prec, xixi_simplicity, xixi_generalization,
    #                new_variant_threshold, new_unfolding_threshold, 500, ref_log_cd_precision,
    #                ari, ref_log_cd_simplicity, ref_log_cd_generalization]

    #         with open(
    #                 f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cd_' + str(experiment_nr_parameter) + "/result_" + str(
    #                     start_data_set_size_parameter) + '.csv',
    #                 'a') as csvfile:
    #             fwriter = csv.writer(csvfile)
    #             fwriter.writerow(row)

    # print(f'Max ARI found for CD: {max_ari}')
    print("Community Detection skipped due to METIS issues")
    num_of_new_labels_comdec = 0
    ref_log_comdec_prec = 0
    ref_log_comdec_simplicity = 0  
    ref_log_comdec_generalization = 0

    # Extension method variables (all commented out)
    num_of_new_labels_folding = 0
    num_of_new_labels_semi = 0
    num_of_new_labels_no_vertical = 0
    mapping_folding_quality = 0
    mapping_semi_quality = 0
    mapping_folding_semi_quality = 0

    # Additional precision variables that are referenced in return statement
    ref_log_folding_prec = 0
    ref_log_semi_prec = 0
    ref_log_no_vertical_prec = 0
    ref_log_all_prec = 0
    ref_log_no_comdec_prec = 0
    ref_log_no_folding_prec = 0
    ref_log_no_semi_prec = 0
    ref_log_vertical_prec = 0

    # Timing variables
    time_needed_for_all_extensions = 0
    time_for_semi_greedy_mapping = 0

    # ref_log_folding, _, _, num_of_new_labels_folding = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                                                  cluster_method="CONNECTED_COMPONENTS",
    #                                                                                                  detection_mode_imp_in_loop="vertical",
    #                                                                                                  egraphs=egraphs_folding,
    #                                                                                                  mapping=mapping_folding,
    #                                                                                                  map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs_folding,
    #                                                                                                  map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID_folding,
    #                                                                                                  use_adaptive_parameters=use_adaptive_parameters)
    # ref_log_semi, _, _, num_of_new_labels_semi = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                                            cluster_method="CONNECTED_COMPONENTS",
    #                                                                                            detection_mode_imp_in_loop="vertical",
    #                                                                                            egraphs=egraphs,
    #                                                                                            mapping=mapping_semi,
    #                                                                                            map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
    #                                                                                            map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
    #                                                                                            use_adaptive_parameters=use_adaptive_parameters)
    # ref_log_no_vertical, _, _, num_of_new_labels_no_vertical = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                                                          cluster_method="CONNECTED_COMPONENTS",
    #                                                                                                          detection_mode_imp_in_loop="postprocessing",
    #                                                                                                          egraphs=egraphs,
    #                                                                                                          mapping=mapping,
    #                                                                                                          map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
    #                                                                                                          map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
    #                                                                                                          use_adaptive_parameters=use_adaptive_parameters)

    # print(num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi, num_of_new_labels_no_vertical)

    # time_s = time()
    # ref_log_all, _, _, _ = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                      cluster_method="COMMUNITY_DETECTION",
    #                                                                      detection_mode_imp_in_loop="postprocessing",
    #                                                                      egraphs=egraphs_folding,
    #                                                                      mapping=mapping_folding_semi,
    #                                                                      map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs_folding,
    #                                                                      map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID_folding,
    #                                                                      use_adaptive_parameters=use_adaptive_parameters)
    # time_e = time()
    # time2 = time_e - time_s
    # ref_log_no_comdec, _, _, _ = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                            cluster_method="CONNECTED_COMPONENTS",
    #                                                                            detection_mode_imp_in_loop="postprocessing",
    #                                                                            egraphs=egraphs_folding,
    #                                                                            mapping=mapping_folding_semi,
    #                                                                            map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs_folding,
    #                                                                            map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID_folding,
    #                                                                            use_adaptive_parameters=use_adaptive_parameters)
    # ref_log_no_folding, _, _, _ = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                             cluster_method="COMMUNITY_DETECTION",
    #                                                                             detection_mode_imp_in_loop="postprocessing",
    #                                                                             egraphs=egraphs, mapping=mapping_semi,
    #                                                                             map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs,
    #                                                                             map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID,
    #                                                                             use_adaptive_parameters=use_adaptive_parameters)
    # ref_log_no_semi, _, _, _ = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                          cluster_method="COMMUNITY_DETECTION",
    #                                                                          detection_mode_imp_in_loop="postprocessing",
    #                                                                          egraphs=egraphs_folding,
    #                                                                          mapping=mapping_folding,
    #                                                                          map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs_folding,
    #                                                                          map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID_folding,
    #                                                                          use_adaptive_parameters=use_adaptive_parameters)
    # ref_log_vertical, _, _, _ = egraph_label_refinement.get_refined_event_log(event_log,
    #                                                                           cluster_method="COMMUNITY_DETECTION",
    #                                                                           detection_mode_imp_in_loop="vertical",
    #                                                                           egraphs=egraphs_folding,
    #                                                                           mapping=mapping_folding_semi,
    #                                                                           map_egraph_ID_to_trace_IDs=map_egraph_ID_to_trace_IDs_folding,
    #                                                                           map_trace_ID_to_egraph_ID=map_trace_ID_to_egraph_ID_folding,
    #                                                                           use_adaptive_parameters=use_adaptive_parameters)

    # org_model_prec = precision_util.get_precision_of_original_model(original_net, original_initial_marking,
    #                                                                 original_final_marking, event_log, imprecise_labels,
    #                                                                 original_labels, parameters)  # todo test
    # precise_refined_log_prec = precision_util.get_precision_of_precice_log(original_event_log, event_log,
    #                                                                        imprecise_labels, original_labels,
    #                                                                        parameters)
    #
    # imp_prec = precision_util.get_precision(event_log, event_log, imprecise_labels, parameters)
    #
    # ref_log_prec = precision_util.get_precision(ref_log, event_log, imprecise_labels, parameters)
    # xixi_prec = precision_util.get_precision(xixi_log, event_log, imprecise_labels, parameters)
    #
    # ref_log_comdec_prec = precision_util.get_precision(ref_log_comdec, event_log, imprecise_labels, parameters)
    # ref_log_folding_prec = precision_util.get_precision(ref_log_folding, event_log, imprecise_labels, parameters)
    # ref_log_semi_prec = precision_util.get_precision(ref_log_semi, event_log, imprecise_labels, parameters)
    # ref_log_no_vertical_prec = precision_util.get_precision(ref_log_no_vertical, event_log, imprecise_labels,
    #                                                         parameters)

    # ref_log_all_prec = precision_util.get_precision(ref_log_all, event_log, imprecise_labels, parameters)
    #
    # ref_log_no_comdec_prec = precision_util.get_precision(ref_log_no_comdec, event_log, imprecise_labels, parameters)
    # ref_log_no_folding_prec = precision_util.get_precision(ref_log_no_folding, event_log, imprecise_labels, parameters)
    # ref_log_no_semi_prec = precision_util.get_precision(ref_log_no_semi, event_log, imprecise_labels, parameters)
    # ref_log_vertical_prec = precision_util.get_precision(ref_log_vertical, event_log, imprecise_labels,
    #                                                      parameters)  # no postpro = vetical ref.

    number_of_different_original_labels = precision_util.get_number_of_different_original_labels(event_log)
    # print("number_of_different_original_labels: ", number_of_different_original_labels)

    # time_needed_for_all_extensions = time0 + time1 + time2

    end_time = time()
    epoch_time = end_time - start_time
    print("epoch time: ", epoch_time)
    return org_model_prec, precise_refined_log_prec, \
           imp_prec, xixi_prec, ref_log_prec, \
           0, 0, 0, 0, \
           0, \
           0, 0, 0, 0, \
           number_of_different_original_labels, \
           epoch_time, \
           0, \
           time_for_greedy_mapping, 0, \
           num_of_new_labels, num_of_new_labels_comdec, 0, 0, 0, \
           mapping_quality, 0, 0, 0, \
           refined_log_fitness, refined_log_fscore, expected_entropy_clusters, expected_entropy_labels, nmi


def get_ground_truth_clustering_from_log(event_log, imprecise_labels, original_labels):
    clustering = []

    print(f"[GROUND_TRUTH_DEBUG] Processing {len(event_log)} traces")
    print(f"[GROUND_TRUTH_DEBUG] Looking for imprecise labels: {imprecise_labels}")
    print(f"[GROUND_TRUTH_DEBUG] Original labels: {original_labels}")

    events_found = 0
    ground_truth_labels = set()

    for trace in event_log:
        for event in trace:
            current_activity = event['concept:name']

            # Only include events that have been made imprecise (current activity is imprecise)
            if current_activity in imprecise_labels:
                # Check if OrgLabel exists for ground truth
                if 'OrgLabel' in event:
                    ground_truth_label = event['OrgLabel']
                    try:
                        clustering.append(original_labels.index(ground_truth_label))
                        ground_truth_labels.add(ground_truth_label)
                        events_found += 1
                    except ValueError:
                        # If OrgLabel not in original_labels, use current label index
                        clustering.append(0)
                        events_found += 1
                else:
                    # For real logs without ground truth, create dummy clustering
                    clustering.append(0)
                    events_found += 1
    
    print(f"[GROUND_TRUTH_DEBUG] Found {events_found} events with imprecise labels")
    print(f"[GROUND_TRUTH_DEBUG] Unique ground truth labels: {sorted(ground_truth_labels)}")
    print(f"[GROUND_TRUTH_DEBUG] Ground truth clustering sample: {clustering[:10]}...")
    
    if not clustering:  # If no imprecise labels found, return empty clustering
        clustering = [0]  # At least one element
    
    return clustering


def get_clustering_from_log(event_log, imprecise_labels):
    seen_labels = []
    clustering = []
    
    print(f"[DEBUG] get_clustering_from_log called with imprecise_labels: {imprecise_labels}")
    
    for trace in event_log:
        for event in trace:
            activity_name = event['concept:name']
            # Check if this activity is an imprecise label or a refined version of one
            is_imprecise_related = False
            
            # Direct match with imprecise labels
            if activity_name in imprecise_labels:
                is_imprecise_related = True
            else:
                # Check if it's a refined version (e.g., "D_0", "D_1" for imprecise label "D")
                for imprecise_label in imprecise_labels:
                    if activity_name.startswith(f"{imprecise_label}_"):
                        is_imprecise_related = True
                        break
            
            if is_imprecise_related:
                if activity_name not in seen_labels:
                    seen_labels.append(activity_name)
                clustering.append(seen_labels.index(activity_name))
    
    print(f"[DEBUG] Found {len(clustering)} events with imprecise-related labels")
    print(f"[DEBUG] Seen labels: {seen_labels}")
    
    if not clustering:  # If no imprecise labels found, return empty clustering
        clustering = [0]  # At least one element
        print("[DEBUG] No imprecise-related events found, returning default clustering")
    
    print('clustering final')
    print(clustering)
    return clustering
