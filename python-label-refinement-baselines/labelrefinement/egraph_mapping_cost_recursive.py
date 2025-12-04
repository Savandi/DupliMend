import trace_util as graph_util
from time import time

def default_labeling_function(event):
    if "Activity" in event:
        return event["Activity"]
    return event["concept:name"]


def get_mapping_cost(egraph_1, egraph_2, mapping, old_cost_matched, old_cost_not_matched, old_cost_structure,
                     nodeID_1, nodeID_2, weight_matched, weight_not_matched, weight_structure, k, basic_cost,
                     labeling_function, already_mapped_1, already_mapped_2, use_mapping_and_label_neighbors):
    gmc1 = time()
    new_cost_matched = get_cost_matched(egraph_1, egraph_2, mapping, old_cost_matched, nodeID_1, nodeID_2, k, labeling_function, use_mapping_and_label_neighbors)
    cost = weight_matched * new_cost_matched
    gmc2 = time()
    #print("t1: ", gmc2-gmc1)

    #print("c1: ", cost)
    new_cost_not_matched = get_cost_not_matched(egraph_1, egraph_2, mapping, old_cost_not_matched, nodeID_1, nodeID_2, k, basic_cost, already_mapped_1, already_mapped_2)
    cost += weight_not_matched * new_cost_not_matched
    #print("c2: ", cost)
    gmc3 = time()
    #print("t2: ", gmc3-gmc2)
    new_cost_structure = get_cost_structure(egraph_1, egraph_2, mapping, old_cost_structure, nodeID_1, nodeID_2)
    cost += weight_structure * new_cost_structure
    #print("c3: ", cost)
    gmc4 = time()
    #print("t3: ", gmc4-gmc3)
    return cost, new_cost_matched, new_cost_not_matched, new_cost_structure

def get_cost_matched(egraph_1, egraph_2, mapping, old_cost_matched, nodeID_1, nodeID_2, k,
                     labeling_function = default_labeling_function, use_mapping_and_label_neighbors=False):
    cost = old_cost_matched

    #cost += graph_util.get_dissimilar_neighbors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function)


    # print(graph_util.get_dissimilar_neighbors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function) -
    #       graph_util.get_dissimilar_successors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function)
    #       - graph_util.get_dissimilar_predecessors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function))
    #print("gcm: ", cost)
    #cost += get_label_cost(egraph_1.nodeID_to_event_dict[nodeID_1], egraph_2.nodeID_to_event_dict[nodeID_2], labeling_function)
    cost += graph_util.get_dissimilar_predecessors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function, use_mapping_and_label_neighbors)
    cost += graph_util.get_dissimilar_successors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function, use_mapping_and_label_neighbors)
    #cost += graph_util.get_dissimilar_concurrencies(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function) #enable when using egraphs instead traces


    return cost

def get_cost_not_matched(egraph_1, egraph_2, mapping, old_mapping_cost, nodeID_1, nodeID_2, k, basic_cost, already_mapped_1, already_mapped_2):
    return old_mapping_cost - 4 * k - 2 * basic_cost
    '''
    if mapping != []:
        cost = old_mapping_cost
        #if nodeID_1 not in already_mapped_1:  #todo enable if non injective mappings are allowed
        cost -= basic_cost
        cost -= 2 * k #graph_util.get_neighbor_size(egraph_1, nodeID_1, k)


        #if nodeID_2 not in already_mapped_2:   #todo enable if non injective mappings are allowed
        cost -= - basic_cost
        cost -= 2 * k #graph_util.get_neighbor_size(egraph_2, nodeID_2, k)

        return cost
    '''
    #else:
    #    return (egraph_1.size + egraph_2.size) * (basic_cost + 2 * k) #neighborsize always 2*k if totaly ordered egraph is considered


    '''
    not_matched_nodes1 = range(0, egraph_1.size)
    not_matched_nodes2 = range(0, egraph_2.size)

    for node in not_matched_nodes1:
        cost += basic_cost
        cost += graph_util.get_neighbor_size(egraph_1, node, k)
    for node in not_matched_nodes2:
        cost += basic_cost
        cost += graph_util.get_neighbor_size(egraph_2, node, k)
    cost = cost - 2 * basic_cost                #remove cost for first mapping element
    cost = cost - graph_util.get_neighbor_size(egraph_1, nodeID_1, k)
    cost = cost - graph_util.get_neighbor_size(egraph_2, nodeID_2, k)

    return cost
    '''
def get_cost_structure(egraph_1, egraph_2, mapping, old_mapping_cost, nodeID_1A, nodeID_2A):
    cost = old_mapping_cost
    cost = cost + abs(egraph_1.dist_to_start[nodeID_1A] - egraph_2.dist_to_start[nodeID_2A])
    cost = cost + abs(egraph_1.dist_to_end[nodeID_1A]) - (egraph_2.dist_to_end[nodeID_2A])          #dummy start and end
    if mapping == []:
        return cost
    for (nodeID_1B, nodeID_2B) in mapping:
        cost = cost + abs(graph_util.dist(egraph_1, nodeID_1A, nodeID_1B) - graph_util.dist(egraph_2, nodeID_2A, nodeID_2B)) #/ 2

    return cost