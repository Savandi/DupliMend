import egraph_mapping_cost_recursive
import trace_util as graph_util
from time import time

def get_optimal_mapping(egraph_1, egraph_2, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                               labeling_function=egraph_mapping_cost_recursive.default_labeling_function, use_mapping_and_label_neighbors=False):
    old_cost_matched = 4 * k    #todo: context size of dummy start and dummy end. assumption: context of dummy start and end do not overlap  # problematic with real context comparison
    old_cost_not_matched = (egraph_1.size + egraph_2.size) * (basic_cost + 2*k)#"start_value"
    old_cost_structure = abs(egraph_1.size - egraph_2.size)  #todo
    old_mapping_cost = old_cost_matched * weight_matched + old_cost_structure * weight_structure + old_cost_not_matched * weight_not_matched
    #old_mapping_cost = 999999
    #old_cost_matched = 4 * k    #todo: context size of dummy start and dummy end. assumption: context of dummy start and end do not overlap  # problematic with real context comparison
    #old_cost_not_matched = "start_value"
    #old_cost_structure = abs(egraph_1.size - egraph_2.size)

    time0 = time()
    #todo: initialize maximal mapping cost / cost for empty mapping
    priority_list = graph_util.get_nodes_sorted_by_importance(egraph_1, egraph_2, k, labeling_function)
    #print("priority_list: ", priority_list)
    time1 = time()
    #print("timeprio: ", time1 - time0)
    mapping = [] #initial mapping consists of matched dummy start and end
    final_cost = old_mapping_cost
    already_mapped_1 = []
    already_mapped_2 = []
    #print("priority list: ", priority_list)
    #if (egraph_1.graph_id == 29 and egraph_2.graph_id == 7) or (
    #        egraph_2.graph_id == 29 and egraph_1.graph_id == 7):
    #    print("priority_list: ", priority_list)
    for (nodeID, graph_nr, importance) in priority_list:

        #print("old mc: ",old_mapping_cost)
        if already_mapped_1 == range(0, egraph_1.size) or already_mapped_2 == range(0, egraph_2.size):
            break
        if graph_nr == 1 and nodeID not in already_mapped_1:
            mapped_position, new_mapping_cost, new_cost_matched, new_cost_not_matched, new_cost_structure = \
                get_optimal_mapping_for_single_event(egraph_1, egraph_2, nodeID, mapping, old_cost_matched,
                                                     old_cost_not_matched, old_cost_structure, already_mapped_1, already_mapped_2,
                                                                   weight_matched, weight_not_matched, weight_structure,
                                                                   k, graph_nr, basic_cost,
                                                                   labeling_function, use_mapping_and_label_neighbors)


            if mapped_position != 'nm' and new_mapping_cost <= old_mapping_cost: # and mapping_cost != 'inf': #nm = no mapping found (optimum = inf cost)
                old_mapping_cost = new_mapping_cost
                old_cost_matched = new_cost_matched
                old_cost_not_matched = new_cost_not_matched
                old_cost_structure = new_cost_structure
                mapping.append((nodeID, mapped_position))
                already_mapped_2.append(mapped_position)
                final_cost = old_mapping_cost
                already_mapped_1.append(nodeID)

        if graph_nr == 2 and nodeID not in already_mapped_2:
            mapped_position, new_mapping_cost, new_cost_matched, new_cost_not_matched, new_cost_structure =\
                get_optimal_mapping_for_single_event(egraph_1, egraph_2, nodeID, mapping, old_cost_matched,
                                                     old_cost_not_matched, old_cost_structure, already_mapped_1, already_mapped_2,
                                                                           weight_matched, weight_not_matched, weight_structure,
                                                                   k, graph_nr, basic_cost,
                                                                   labeling_function, use_mapping_and_label_neighbors)
            if mapped_position != 'nm' and new_mapping_cost <= old_mapping_cost: # and mapping_cost != 'inf': #nm = no mapping found (optimum = inf cost)
                old_mapping_cost = new_mapping_cost
                old_cost_matched = new_cost_matched
                old_cost_not_matched = new_cost_not_matched
                old_cost_structure = new_cost_structure
                mapping.append((mapped_position, nodeID))
                already_mapped_1.append(mapped_position)
                final_cost = old_mapping_cost
                already_mapped_2.append(nodeID)


        '''
            #print("new mc: ", new_mapping_cost)
        if (egraph_1.graph_id == 29 and egraph_2.graph_id == 7) or (
                    egraph_2.graph_id == 29 and egraph_1.graph_id == 7):
                print("position: ", nodeID)
                print("mapped_position: ", mapped_position)
                print("new_mapping_cost: ", new_mapping_cost)
                print("new_cost_matched: ", new_cost_matched)
                print("new_cost_not_matched: ", new_cost_not_matched)
                print("new_cost_structure: ", new_cost_structure)

        if (egraph_1.graph_id == 29 and egraph_2.graph_id == 7) or (egraph_2.graph_id == 29 and egraph_1.graph_id == 7):

            print("mapping", mapping)  #todo marking
            print("final_cost: ", final_cost)
            print("old_cost_matched: ", old_cost_matched)
            print("old_cost_not_matched: ", old_cost_not_matched)
            print("old_cost_structure: ", old_cost_structure)
            print("------------------------------------------------------------------------")
        '''

        #if (graph_nr == 1 and egraph_1.nodeID_to_event_dict[nodeID]["OrgLabel"] == "D") or\
        #        (graph_nr == 2 and egraph_2.nodeID_to_event_dict[nodeID]["OrgLabel"] == "D"):
        #print("mappin", mapping)
        #print("mapping_cost", final_cost)
        #print("old_cost_matched", old_cost_matched)
        #print("old_cost_not_matched", old_cost_not_matched)
        #print("old_cost_structure", old_cost_structure)
    #final_cost = egraph_mapping_cost_recursive.get_mapping_cost(egraph_1, egraph_2, mapping, weight_matched, weight_not_matched,
    #                                      weight_structure, k, basic_cost,
    #                                      labeling_function)
    time2 = time()
    #print("timerest: ", time2-time1)
    #print("mapping time: ", gomg3 - gomg1)
    #print("final_cost", final_cost)
    #final_cost = old_cost_matched + old_cost_not_matched + old_cost_structure

    #if egraph_2.size > 20 or egraph_1.size > 20:
    '''
    for a, b in mapping:

        if ((egraph_1.nodeID_to_event_dict[a]["OrgLabel"] == "D") and (
                                egraph_2.nodeID_to_event_dict[b]["OrgLabel"] != "D")) \
                                or ((egraph_1.nodeID_to_event_dict[a]["OrgLabel"] != "D") and (
                                egraph_2.nodeID_to_event_dict[b]["OrgLabel"] == "D")):
                print(egraph_1.nodeID_to_event_dict[a]["OrgLabel"])
                print(egraph_2.nodeID_to_event_dict[b]["OrgLabel"])
                print("priority_list", priority_list)
                print("mapping_cost", final_cost)
                print("old_cost_matched", old_cost_matched)
                print("old_cost_not_matched", old_cost_not_matched)
                print("old_cost_structure", old_cost_structure)
                s1 = ""
                for e in egraph_1.nodeID_to_event_dict.values():
                    s1 += labeling_function(e)
                print(s1)
                s2 = ""
                for e in egraph_2.nodeID_to_event_dict.values():
                    s2 += labeling_function(e)
                print(s2)

                lm = []
                for m1, m2 in mapping:
                    lm.append((labeling_function(egraph_1.nodeID_to_event_dict[m1]),labeling_function(egraph_2.nodeID_to_event_dict[m2])))
                print(lm)
                print(mapping)
                print("______")
    '''
    return mapping, final_cost #/ (4*((egraph_1.size + egraph_2.size) * basic_cost + (egraph_1.size + egraph_2.size) * 2 * k-4))

def get_optimal_mapping_for_single_event(egraph_1, egraph_2, nodeID_source, mapping, old_cost_matched,
                                         old_cost_not_matched, old_cost_structure, already_mapped_1, already_mapped_2,
                                            weight_matched, weight_not_matched, weight_structure,
                                            k, graph_nr, basic_cost,
                                            labeling_function=egraph_mapping_cost_recursive.default_labeling_function,
                                            use_mapping_and_label_neighbors=False):

    final_mapping_node = 'nm'
    final_mapping_cost = "inf"
    final_cost_matched = old_cost_matched
    final_cost_structure = old_cost_structure
    final_cost_not_matched = old_cost_not_matched

    if graph_nr == 1:
        already_mapped = already_mapped_2
        egraph_source = egraph_1
        egraph_target = egraph_2
    else:
        already_mapped = already_mapped_1
        egraph_source = egraph_2
        egraph_target = egraph_1

    source_label = labeling_function(egraph_source.nodeID_to_event_dict[nodeID_source])
    #print("sourcelabel: ", source_label)
    possible_match_nodes = [nodeID for nodeID in range(0, egraph_target.size) if nodeID not in already_mapped] #todo
    #print("possible_match_nodes", possible_match_nodes)
    for nodeID_target in possible_match_nodes:
        if labeling_function(egraph_target.nodeID_to_event_dict[nodeID_target]) == source_label:
            #print("targetlabel: ", labeling_function(egraph_target.nodeID_to_event_dict[nodeID_target]))
            if graph_nr == 1:
                new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure = \
                    egraph_mapping_cost_recursive.get_mapping_cost(egraph_1, egraph_2, mapping, old_cost_matched,
                                                                   old_cost_not_matched, old_cost_structure, nodeID_source, nodeID_target,
                                                                weight_matched, weight_not_matched,
                                                                weight_structure, k, basic_cost, labeling_function,
                                                                   already_mapped_1, already_mapped_2,
                                                                   use_mapping_and_label_neighbors)
            else:
                new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure = \
                    egraph_mapping_cost_recursive.get_mapping_cost(egraph_1, egraph_2, mapping, old_cost_matched,
                                                                   old_cost_not_matched, old_cost_structure, nodeID_target, nodeID_source,
                                                                weight_matched, weight_not_matched,
                                                                weight_structure, k, basic_cost, labeling_function,
                                                                   already_mapped_1, already_mapped_2,
                                                                   use_mapping_and_label_neighbors)


            #print("nodeID_target", nodeID_target)
            #print(new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure)
            if final_mapping_cost == 'inf' or new_cost <= final_mapping_cost:
                final_cost_matched = new_cost_matched
                final_cost_structure = new_cost_structure
                final_cost_not_matched = new_cost_not_matched
                final_mapping_cost = new_cost
                final_mapping_node = nodeID_target
            '''
            if egraph_1.graph_id == 85 and egraph_2.graph_id == 56:
                    print("mapping_cost:", new_cost)
                    print("mapping_node:", nodeID_target)
                    print("matched:", new_cost_matched)
                    print("struc:", new_cost_structure)
                    print("notmatched:", new_cost_not_matched)
                    print("...................")
            '''
    #final_mapping_cost = final_cost_matched + 0.3 * final_cost_not_matched + 0.2 * final_cost_structure
    return final_mapping_node, final_mapping_cost, final_cost_matched, final_cost_not_matched, final_cost_structure
