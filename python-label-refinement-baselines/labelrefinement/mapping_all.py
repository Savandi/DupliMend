import egraph_mapping_advanced
import egraph_mapping_dynamic
import egraph_vertical_refinement
import egraph_class
import copy
from time import time
import egraph_horizontal_refinement_cc as egraph_horizontal_refinement


def get_mappings(egraphs, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                                        labeling_function, mapping_search_mode, use_mapping_and_label_neighbors):
    mappings = get_mappings_helper(egraphs, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                                        labeling_function, mapping_search_mode, use_mapping_and_label_neighbors)
    mappings = normalize_mappings(mappings)

    new_mappings = mappings
    for i in range(0, len(mappings)):  # transforms triangle into symmetric quadrat
        for j in range(i + 1, len(mappings)):
            new_mappings[i].append(([(m[1], m[0]) for m in mappings[j][i][0]], mappings[j][i][1]))

    return new_mappings


def normalize_mappings(mappings): #todo: remove normalization function
    max_cost = 1
    #for mappings_for_single_trace in mappings:
    #    for mapping, mapping_cost in mappings_for_single_trace:
    #        if mapping_cost != 'nm' and max_cost < mapping_cost:
    #            max_cost = mapping_cost
    #average_cost = 0

    costs = []
    for mappings_for_single_trace_position in range(0, len(mappings)):
        for mapping_position in range(0, mappings_for_single_trace_position+1):
            mapping, mapping_cost = mappings[mappings_for_single_trace_position][mapping_position]
            #average_cost += mapping_cost
            if mapping_cost != "inf":
                costs.append(mapping_cost)
            if mapping_cost != "inf" and mapping_cost >= max_cost:
                max_cost = mapping_cost
    #average_cost = average_cost/((len(mappings) ** 2)/2)

    #print("max:", max_cost)
    #print("average cost:", sum(costs) / len(costs))
    for mappings_for_single_trace_position in range(0, len(mappings)):
        for mapping_position in range(0, mappings_for_single_trace_position+1):

            mapping, mapping_cost = mappings[mappings_for_single_trace_position][mapping_position]
            #if mapping_cost != 'inf' and mapping_cost != "nm":
            mappings[mappings_for_single_trace_position][mapping_position] = mapping,mapping_cost#/max_cost #/ (3 * average_cost)#max_cost
            if mapping_cost == "inf":
                mappings[mappings_for_single_trace_position][mapping_position] = mapping, max_cost
            #else:
            #    print("allo")
            #    mappings[mappings_for_single_trace_position][mapping_position] = mapping, 1

    cost_sum = 0
    for mappins in mappings:
        for m, cost in mappins:
            cost_sum += cost
    #print("summed mapping cost: ", cost_sum)
    return mappings

def get_mappings_helper(egraphs, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                               labeling_function, mapping_search_mode, use_mapping_and_label_neighbors):

    mappings = []
    for egraph_index1 in range(0, len(egraphs)):
        mapping_for_egraph = []
        for egraph_index2 in range(0, len(egraphs)):
            if egraph_index2 > egraph_index1:
                break
                #mapping_for_egraph.append("none")
            if egraph_index1 == egraph_index2:
                mapping_for_egraph.append(([(index, index) for index in range(0, egraphs[egraph_index1].size)], 0))
            else:
                time_before = time()
                if mapping_search_mode == "GREEDY":
                    mapping_for_egraph.append(egraph_mapping_dynamic.get_optimal_mapping(egraphs[egraph_index1],
                                              egraphs[egraph_index2], weight_matched, weight_not_matched, weight_structure,
                                              k, basic_cost, labeling_function, use_mapping_and_label_neighbors))
                else: #if ... "SEMI-GREEDY"
                    mapping_for_egraph.append(egraph_mapping_advanced.get_optimal_mapping(egraphs[egraph_index1],
                                              egraphs[egraph_index2], weight_matched, weight_not_matched, weight_structure,
                                              k, basic_cost, labeling_function, use_mapping_and_label_neighbors))
                time_after = time()
                #print("mapping time: ", time_after-time_before)
        mappings.append(mapping_for_egraph)
    return mappings
    #return [[egraph_mapping.get_optimal_mapping_greedy(egraph_1, egraph_2, weight_matched, weight_not_matched, weight_structure,
    #                                            k, basic_cost, labeling_function) for egraph_2 in egraphs] for egraph_1 in egraphs]
