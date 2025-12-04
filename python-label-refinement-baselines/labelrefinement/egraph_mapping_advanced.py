import egraph_mapping_cost_recursive
import trace_util as graph_util
from time import time

def get_optimal_mapping(egraph_1, egraph_2, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                               labeling_function=egraph_mapping_cost_recursive.default_labeling_function,
                                use_mapping_and_label_neighbors=False):
    cost_matched = 4 * k    #todo: context size of dummy start and dummy end. assumption: context of dummy start and end do not overlap  # problematic with real context comparison
    cost_not_matched = (egraph_1.size + egraph_2.size) * (basic_cost + 2*k)#"start_value"
    cost_structure = abs(egraph_1.size - egraph_2.size)  #todo
    old_mapping_cost = cost_matched * weight_matched + cost_structure * weight_structure + cost_not_matched * weight_not_matched
    cost = old_mapping_cost
    time0 = time()
    time1 = time()
    #print("timeprio: ", time1 - time0)
    mapping = [] #initial mapping consists of matched dummy start and end
    final_cost = 'inf'

    label_list, source_position_dict, target_position_dict = get_label_list(egraph_1, egraph_2, labeling_function)
    #print("label list: ", label_list)
    #print("source_position_dict: ", source_position_dict)
    #print("target_position_dict: ", target_position_dict)
    for label in label_list:
        if label in source_position_dict.keys() and label in target_position_dict.keys():
            cost, cost_matched, cost_not_matched, cost_structure, mapping = rec_call(source_position_dict[label][0], egraph_1,
                                                                egraph_2, cost, cost_matched, cost_not_matched, cost_structure,
                                                                mapping, source_position_dict[label], target_position_dict[label],
                                                                weight_matched, weight_not_matched,
                                                                weight_structure, k, basic_cost, labeling_function,use_mapping_and_label_neighbors)

            #print(mapping, cost, cost_matched, cost_not_matched, cost_structure)

    time2 = time()
    #print("timerest: ", time2 - time1)

    return mapping, cost

def rec_call(source_nodeID, egraph_source, egraph_target, old_cost, old_cost_matched, old_cost_not_matched, old_cost_structure,
             mapping, free_source_domain, free_target_domain, weight_matched, weight_not_matched,
                                                           weight_structure, k, basic_cost, labeling_function, use_mapping_and_label_neighbors):
    new_free_source_domain = free_source_domain.copy()
    new_free_source_domain.remove(source_nodeID)
    #if egraph_source.graph_id == 126 and egraph_target.graph_id == 108:
        #print("-------bein----------------")
        #print("free_target_domain: ", free_target_domain)
    cost_dict = {}
    counter = 0
    for target_nodeID in free_target_domain:
        #if egraph_source.graph_id == 126 and egraph_target.graph_id == 108:
            #print("free_target_domain in loop: ", free_target_domain)
            #print("counter: ", counter)
            #counter += 1

        new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure = \
            egraph_mapping_cost_recursive.get_mapping_cost(egraph_source, egraph_target, mapping, old_cost_matched,
                                                           old_cost_not_matched, old_cost_structure, source_nodeID,
                                                           target_nodeID,
                                                           weight_matched, weight_not_matched,
                                                           weight_structure, k, basic_cost, labeling_function, [], [], use_mapping_and_label_neighbors)

        new_free_target_domain = free_target_domain.copy()
        new_free_target_domain.remove(target_nodeID)

        if new_free_source_domain != [] and new_free_target_domain != []:
            new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure, new_mapping = \
                rec_call(new_free_source_domain[0], egraph_source, egraph_target, new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure,
                     mapping + [(source_nodeID, target_nodeID)], new_free_source_domain, new_free_target_domain, weight_matched, weight_not_matched,
                     weight_structure, k, basic_cost, labeling_function, use_mapping_and_label_neighbors)
            cost_dict[target_nodeID] = (new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure, new_mapping)
        else:
            cost_dict[target_nodeID] = (new_cost, new_cost_matched, new_cost_not_matched, new_cost_structure, mapping + [(source_nodeID, target_nodeID)])

    if new_free_source_domain != [] and free_target_domain != []:
        best_decision = rec_call(new_free_source_domain[0], egraph_source, egraph_target, old_cost, old_cost_matched, old_cost_not_matched, old_cost_structure,
                         mapping, new_free_source_domain, free_target_domain, weight_matched, weight_not_matched,
                         weight_structure, k, basic_cost, labeling_function, use_mapping_and_label_neighbors)

    else:
        best_decision = (old_cost, old_cost_matched, old_cost_not_matched, old_cost_structure, mapping) #initialy old mapping is the best

    for targetID_key in cost_dict.keys():
        cost, cost_matched, cost_not_matched, cost_structure, mapping = cost_dict[targetID_key]
        if cost < best_decision[0]:
            best_decision = (cost, cost_matched, cost_not_matched, cost_structure, mapping)
    #if egraph_source.graph_id == 126 and egraph_target.graph_id == 108:
        #print("cost_dict     : ", cost_dict)
        #print("best decision : ", best_decision)
        #print("-------end----------------")
    return best_decision


def get_label_list(egraph_source, egraph_target, labeling_function):
    source_position_dict = {}
    target_position_dict = {}
    label_set = set()
    for nodeID in range(0, egraph_source.size):
        label = labeling_function(egraph_source.nodeID_to_event_dict[nodeID])
        label_set.add(label)
        if label not in source_position_dict.keys():
            source_position_dict[label] = [nodeID]
            target_position_dict[label] = []
        else:
            source_position_dict[label].append(nodeID)


    for nodeID in range(0, egraph_target.size):
        label = labeling_function(egraph_target.nodeID_to_event_dict[nodeID])
        if label in target_position_dict.keys():
            target_position_dict[label].append(nodeID)



    label_list = list(label_set)
    label_list = sorted(label_list, key=lambda x: (len(target_position_dict[x]) + len(source_position_dict[x])), reverse=False)
    return label_list, source_position_dict, target_position_dict