import copy
from time import time

import egraph_builder
import egraph_horizontal_refinement_cc2
import egraph_horizontal_refinement_community_detection
import egraph_postprocessing
import egraph_vertical_refinement
import imprecise_label_detector
import mapping_all
import mappings_evaluator
import precision_util
from igraph import *


def default_labeling_function(event):
    # if "Activity" in event:
    #    return event["Activity"]
    return event["concept:name"]


def get_refined_event_log(event_log,
                          parameters={"TIMESTAMP_KEY": "no_timestamp", "ACTIVITY_KEY": "concept:name",
                                                 "EVENT_IDENTIFICATION": "Activity", "CASE_ID_KEY": 0,
                                                 "LIFECYCLE_KEY": "lifecycle:transition",
                                                 "LIFECYCLE_MODE": "atomic", "k": "do_not_use_this_k"},
                          weight_matched=1,
                          weight_not_matched=10,
                          weight_structure=1,
                          k=1,
                          variant_threshold=0.8,
                          basic_cost=1,
                          labeling_function=default_labeling_function,
                          unfolding_threshold=0.6,
                          cluster_method="COMMUNITY_DETECTION",
                          use_trace_folding=False,
                          use_local_cost=True,
                          mapping_search_mode="SEMI-GREEDY",
                          detection_mode="EXPERIMENT",
                          mapping_evaluation_mode=False,
                          use_mapping_and_label_neighbors=False,
                          original_labels=[],
                          imprecise_labels=[],
                          detection_mode_imp_in_loop="postprocessing",
                          speedup_mode=True,
                          egraphs="not defined",
                          map_egraph_ID_to_trace_IDs="not defined",
                          map_trace_ID_to_egraph_ID="not defined",
                          mapping="not defined",
                          use_adaptive_parameters=True):
    parameters["k"] = k

    print('#################### Entered ########################')
    print('use_adaptive_parameters')
    print(use_adaptive_parameters)
    print('cluster_method')
    print(cluster_method)
    if detection_mode == "EXPERIMENT":
        original_labels, imprecise_labels = imprecise_label_detector.oracle_detection(event_log)
    elif detection_mode == "FLOWER":
        original_labels, imprecise_labels = imprecise_label_detector.label_in_flower_detector(event_log)

    if egraphs == "not defined":
        egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID = egraph_builder.get_egraphs(parameters,
                                                                                                    use_trace_folding,
                                                                                                    event_log)

    if not use_adaptive_parameters:
        label_refinements, egraphs, _, _, clustering = get_label_refinements(egraphs, map_egraph_ID_to_trace_IDs,
                                                                 map_trace_ID_to_egraph_ID, original_labels,
                                                                 imprecise_labels, weight_matched, weight_not_matched,
                                                                 weight_structure, k, basic_cost,
                                                                 labeling_function, variant_threshold,
                                                                 unfolding_threshold, cluster_method, use_trace_folding,
                                                                 use_local_cost, mapping_search_mode, detection_mode,
                                                                 mapping_evaluation_mode,
                                                                 use_mapping_and_label_neighbors,
                                                                 detection_mode_imp_in_loop, speedup_mode, mapping,
                                                                 do_vertical_refinement=True)

        # MEMORY FIX: Create ONE copy at the start instead of multiple copies
        refined_log = copy.deepcopy(event_log)
        refined_log = transform_event_log(refined_log, parameters, egraphs, label_refinements)

        # if detection_mode_imp_in_loop == "postprocessing":
        post_processed_label_refinements = egraph_postprocessing.postprocess(refined_log, imprecise_labels,
                                                                             label_refinements, parameters)
        # print(post_processed_label_refinements[0][1])
        number_of_new_labels = len(post_processed_label_refinements[0][1])
        # MEMORY FIX: Reuse the same refined_log instead of creating another deepcopy
        refined_log = transform_event_log(refined_log, parameters, egraphs,
                                          post_processed_label_refinements)

    else:
        horizontal_refinement = "not defined"
        precision = 0
        if cluster_method == "CONNECTED_COMPONENTS":
            print('Entering 1st loop')
            for new_variant_threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
                new_label_refinements, new_egraphs, mapping, new_horizontal_refinement, clustering = get_label_refinements(egraphs,
                                                                                                               map_egraph_ID_to_trace_IDs,
                                                                                                               map_trace_ID_to_egraph_ID,
                                                                                                               original_labels,
                                                                                                               imprecise_labels,
                                                                                                               weight_matched,
                                                                                                               weight_not_matched,
                                                                                                               weight_structure,
                                                                                                               k,
                                                                                                               basic_cost,
                                                                                                               labeling_function,
                                                                                                               new_variant_threshold,
                                                                                                               unfolding_threshold,
                                                                                                               cluster_method,
                                                                                                               use_trace_folding,
                                                                                                               use_local_cost,
                                                                                                               mapping_search_mode,
                                                                                                               detection_mode,
                                                                                                               mapping_evaluation_mode,
                                                                                                               use_mapping_and_label_neighbors,
                                                                                                               detection_mode_imp_in_loop,
                                                                                                               speedup_mode,
                                                                                                               mapping,
                                                                                                               do_vertical_refinement=False)

                # MEMORY FIX: Create ONE copy per iteration instead of two
                new_refined_log = copy.deepcopy(event_log)
                new_refined_log = transform_event_log(new_refined_log, parameters, new_egraphs,
                                                      new_label_refinements)
                new_new_label_refinements = egraph_postprocessing.postprocess(new_refined_log, imprecise_labels,
                                                                              new_label_refinements, parameters)
                # MEMORY FIX: Reuse the same log instead of creating another deepcopy
                new_refined_log = transform_event_log(new_refined_log, parameters, new_egraphs,
                                                      new_new_label_refinements)

                '''
                if detection_mode_imp_in_loop == "postprocessing":
                    post_processed_label_refinements = egraph_postprocessing.postprocess(new_refined_log, imprecise_labels, new_label_refinements, parameters)
                    new_refined_log = transform_event_log(copy.deepcopy(event_log), parameters, new_egraphs, post_processed_label_refinements)
                '''
                new_precision, _, _ = precision_util.get_precision(new_refined_log, event_log, imprecise_labels, parameters)
                print('new_precision')
                print(new_precision)
                if new_precision >= precision:
                    variant_threshold = new_variant_threshold
                    refined_log = new_refined_log
                    precision = new_precision
                    horizontal_refinement = new_horizontal_refinement

                else:
                    print('Broke out of loop')
                    ...  # break
                print("real num of dif labels: ", len(new_label_refinements[0][1]), "num of dif labels: ",
                      len(new_new_label_refinements[0][1]), " --- variant threshold: ", new_variant_threshold,
                      " --- prec: ", new_precision)
            print("opt variant_threshold: ", variant_threshold)

        if detection_mode_imp_in_loop == "vertical":
            print('Starting 2nd loop')
            for new_unfolding_threshold in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
                new_label_refinements, new_egraphs, mapping, horizontal_refinement, clustering = get_label_refinements(egraphs,
                                                                                                           map_egraph_ID_to_trace_IDs,
                                                                                                           map_trace_ID_to_egraph_ID,
                                                                                                           original_labels,
                                                                                                           imprecise_labels,
                                                                                                           weight_matched,
                                                                                                           weight_not_matched,
                                                                                                           weight_structure,
                                                                                                           k,
                                                                                                           basic_cost,
                                                                                                           labeling_function,
                                                                                                           variant_threshold,
                                                                                                           new_unfolding_threshold,
                                                                                                           cluster_method,
                                                                                                           use_trace_folding,
                                                                                                           use_local_cost,
                                                                                                           mapping_search_mode,
                                                                                                           detection_mode,
                                                                                                           mapping_evaluation_mode,
                                                                                                           use_mapping_and_label_neighbors,
                                                                                                           detection_mode_imp_in_loop,
                                                                                                           speedup_mode,
                                                                                                           mapping,
                                                                                                           horizontal_refinement)

                # MEMORY FIX: Create copy here for vertical loop
                new_refined_log = copy.deepcopy(event_log)
                new_refined_log = transform_event_log(new_refined_log, parameters, new_egraphs,
                                                      new_label_refinements)
                new_precision, _, _ = precision_util.get_precision(new_refined_log, event_log, imprecise_labels, parameters)
                print('new_precision')
                print(new_precision)
                if new_precision >= precision:
                    unfolding_threshold = new_unfolding_threshold
                    refined_log = new_refined_log
                    precision = new_precision
                else:
                    print('Broke out of 2nd loop')
                    break
            # print("opt_unfolding_threshold: ", unfolding_threshold)

        elif detection_mode_imp_in_loop != "vertical" and cluster_method != "COMMUNITY_DETECTION":
            new_label_refinements, new_egraphs, _, horizontal_refinement, clustering = get_label_refinements(egraphs,
                                                                                                 map_egraph_ID_to_trace_IDs,
                                                                                                 map_trace_ID_to_egraph_ID,
                                                                                                 original_labels,
                                                                                                 imprecise_labels,
                                                                                                 weight_matched,
                                                                                                 weight_not_matched,
                                                                                                 weight_structure,
                                                                                                 k,
                                                                                                 basic_cost,
                                                                                                 labeling_function,
                                                                                                 variant_threshold,
                                                                                                 unfolding_threshold,
                                                                                                 cluster_method,
                                                                                                 use_trace_folding,
                                                                                                 use_local_cost,
                                                                                                 mapping_search_mode,
                                                                                                 detection_mode,
                                                                                                 mapping_evaluation_mode,
                                                                                                 use_mapping_and_label_neighbors,
                                                                                                 detection_mode_imp_in_loop,
                                                                                                 speedup_mode,
                                                                                                 mapping,
                                                                                                 horizontal_refinement)

            # MEMORY FIX: Create copy once for this path
            refined_log = copy.deepcopy(event_log)
            refined_log = transform_event_log(refined_log, parameters, new_egraphs, new_label_refinements)

        if detection_mode_imp_in_loop == "postprocessing" and cluster_method == "COMMUNITY_DETECTION":
            label_refinements, egraphs, _, _, clustering = get_label_refinements(egraphs, map_egraph_ID_to_trace_IDs,
                                                                     map_trace_ID_to_egraph_ID, original_labels,
                                                                     imprecise_labels, weight_matched,
                                                                     weight_not_matched,
                                                                     weight_structure, k,
                                                                     basic_cost, labeling_function, variant_threshold,
                                                                     unfolding_threshold, cluster_method,
                                                                     use_trace_folding, use_local_cost,
                                                                     mapping_search_mode,
                                                                     detection_mode, mapping_evaluation_mode,
                                                                     use_mapping_and_label_neighbors,
                                                                     detection_mode_imp_in_loop,
                                                                     speedup_mode, mapping)

            # MEMORY FIX: Create copy once for CD path
            refined_log = copy.deepcopy(event_log)
            refined_log = transform_event_log(refined_log, parameters, egraphs, label_refinements)
    post_processed_label_refinements = egraph_postprocessing.postprocess(refined_log, imprecise_labels,
                                                                         label_refinements, parameters)
    number_of_new_labels = len(post_processed_label_refinements[0][1])
    refined_log = transform_event_log(refined_log, parameters, egraphs, post_processed_label_refinements)

    return refined_log, imprecise_labels, original_labels, number_of_new_labels, clustering


def get_label_refinements(egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID, original_labels, labels,
                          weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                          labeling_function, variant_threshold, unfolding_threshold, cluster_method, use_trace_folding,
                          use_local_cost, mapping_search_mode, detection_mode, mapping_evaluation_mode,
                          use_mapping_and_label_neighbors, detection_mode_imp_in_loop, speedup_mode, mappings,
                          horizontal_refinement="not defined", do_vertical_refinement=True):
    # print(use_local_cost)
    # todo using parameter horiz ref causes trouble
    label_refinements = []
    if mappings == "not defined":
        mappings = mapping_all.get_mappings(egraphs, weight_matched, weight_not_matched, weight_structure, k,
                                            basic_cost,
                                            labeling_function, mapping_search_mode, use_mapping_and_label_neighbors)

    if mapping_evaluation_mode and detection_mode == "EXPERIMENT":
        mappings_evaluator.evaluate_mappings(egraphs, mappings, original_labels, labeling_function)

    '''
    if not speedup_mode:
        log_range = range(0, len(map_trace_ID_to_egraph_ID))
        new_egraphs = []
        new_mappings = [[0 for _ in log_range] for _ in log_range]
        for trace_ID in log_range:
            new_egraphs.append(egraphs[map_trace_ID_to_egraph_ID[trace_ID]])
            for trace_ID2 in log_range:
                new_mappings[trace_ID][trace_ID2] = mappings[map_trace_ID_to_egraph_ID[trace_ID]][map_trace_ID_to_egraph_ID[trace_ID2]]

        mappings = new_mappings
        egraphs = new_egraphs
    '''
    for label in labels:
        if horizontal_refinement == "not defined":
            time0 = time()
            label_refinement, clustering = refine_label(egraphs, mappings, label, labeling_function, variant_threshold,
                                            cluster_method, use_trace_folding,
                                            use_local_cost, mapping_search_mode, detection_mode,
                                            mapping_evaluation_mode,
                                            use_mapping_and_label_neighbors)  # list of lists containing (log_position, trace_position) tuples
            time1 = time()
            # print(time1-time0)
            if speedup_mode:
                log_range = range(0, len(map_trace_ID_to_egraph_ID))
                new_egraphs = []
                for trace_ID in log_range:
                    new_egraphs.append(egraphs[map_trace_ID_to_egraph_ID[trace_ID]])
                egraphs = new_egraphs

                new_label_refinement = []
                for com in label_refinement:
                    new_com = []
                    for egraph_nr, node_nr in com:
                        trace_ids = map_egraph_ID_to_trace_IDs[egraph_nr]
                        for trace_id in trace_ids:
                            new_com.append((trace_id, node_nr))
                    new_label_refinement.append(new_com)
                label_refinement = new_label_refinement
                # print('Took horizontal_refinement')
                # print(label_refinement)
            horizontal_refinement = label_refinement
            # print("length of horiz. ref.:", len(horizontal_refinement))
        else:  # todo
            label_refinement = horizontal_refinement
            log_range = range(0, len(map_trace_ID_to_egraph_ID))
            new_egraphs = []
            for trace_ID in log_range:
                new_egraphs.append(egraphs[map_trace_ID_to_egraph_ID[trace_ID]])
            egraphs = new_egraphs

        if detection_mode_imp_in_loop == "vertical" and do_vertical_refinement:
            vertical_label_refinement = egraph_vertical_refinement.get_vertical_refinement(egraphs,
                                                                                           map_egraph_ID_to_trace_IDs,
                                                                                           map_trace_ID_to_egraph_ID,
                                                                                           label_refinement,
                                                                                           label, labeling_function,
                                                                                           unfolding_threshold,
                                                                                           speedup_mode)
            label_refinement = vertical_label_refinement
            # print('Took vertical_label_refinement')
            # print(label_refinement)
            # print("length of vertic. ref.:", len(label_refinement))
        label_refinements.append((label, label_refinement))
    return label_refinements, egraphs, mappings, horizontal_refinement, clustering


def refine_label(egraphs, mappings, label, labeling_function, variant_threshold, cluster_method, use_trace_folding,
                 use_local_cost, mapping_search_mode, detection_mode, mapping_evaluation_mode,
                 use_mapping_and_label_neighbors):
    clustering = Clustering([])
    if cluster_method == "CONNECTED_COMPONENTS":
        horizontal_refinement, clustering = egraph_horizontal_refinement_cc2.get_connected_components(egraphs, mappings, label,
                                                                                          labeling_function,
                                                                                          variant_threshold,
                                                                                          use_local_cost,
                                                                                          mapping_evaluation_mode,
                                                                                          use_mapping_and_label_neighbors)
    if cluster_method == "COMMUNITY_DETECTION":
        horizontal_refinement, clustering = egraph_horizontal_refinement_community_detection.get_communities(egraphs, mappings,
                                                                                                 label,
                                                                                                 labeling_function,
                                                                                                 variant_threshold,
                                                                                                 use_local_cost,
                                                                                                 mapping_evaluation_mode,
                                                                                                 use_mapping_and_label_neighbors)
    return horizontal_refinement, clustering


def transform_event_log(event_log, parameters, egraphs, label_refinements):
    for label, label_refinement in label_refinements:
        for label_nr in range(0, len(label_refinement)):
            if len(label_refinement[label_nr]) >= 0:  # todo fix
                for egraph_index, nodeID in label_refinement[label_nr]:
                    # for trace_index in map_egraph_ID_to_trace_IDs[egraph_index]:
                    print(f"Debug: egraph_index={egraph_index}, nodeID={nodeID}")
                    print(f"Debug: Available nodeIDs in egraph {egraph_index}: {list(egraphs[egraph_index].nodeID_to_event_dict.keys())}")
                    if nodeID not in egraphs[egraph_index].nodeID_to_event_dict:
                        print(f"Warning: nodeID {nodeID} not found in egraph {egraph_index}. Skipping.")
                        continue
                    start_position = egraphs[egraph_index].nodeID_to_event_dict[nodeID]["start_position"]

                    # print('egraph_index')
                    # print(egraph_index)
                    # print('nodeID')
                    # print(nodeID)
                    # print('start_position')
                    # print(start_position)

                    # Check if egraph_index is within bounds of event_log
                    if egraph_index >= len(event_log):
                        print(f"Warning: egraph_index {egraph_index} exceeds event_log length {len(event_log)}. Skipping.")
                        continue

                    # Check if start_position is within bounds of the trace
                    if start_position >= len(event_log[egraph_index]):
                        print(f"Warning: start_position {start_position} exceeds trace length {len(event_log[egraph_index])} at egraph_index {egraph_index}. Skipping.")
                        continue

                    event_log[egraph_index][start_position][parameters['ACTIVITY_KEY']] = \
                    event_log[egraph_index][start_position][parameters['ACTIVITY_KEY']][0] + "_X_" + str(label_nr + 1)

                    if parameters["LIFECYCLE_MODE"] == "full":
                        print('Lifecycle full')
                        end_position = egraphs[egraph_index].nodeID_to_event_dict[nodeID]["end_position"]
                        # Check if end_position is within bounds
                        if end_position < len(event_log[egraph_index]):
                            event_log[egraph_index][end_position][parameters['ACTIVITY_KEY']] = \
                            event_log[egraph_index][end_position][parameters['ACTIVITY_KEY']][0] + "_X_" + str(label_nr + 1)
                        else:
                            print(f"Warning: end_position {end_position} exceeds trace length {len(event_log[egraph_index])} at egraph_index {egraph_index}. Skipping end position update.")
    return event_log
