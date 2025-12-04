import graph_util
from time import time

def default_labeling_function(event):
    if "Activity" in event:
        return event["Activity"]
    return event["concept:name"]

def get_label_cost(event1, event2, labeling_function = default_labeling_function):
    if labeling_function(event1) == labeling_function(event2):
        return 0
    else:
        return 999999999


def get_mapping_cost(egraph_1, egraph_2, mapping, weight_matched, weight_not_matched, weight_structure, k, basic_cost, labeling_function):
    gmc1 = time()
    cost = weight_matched * get_cost_matched(egraph_1, egraph_2, mapping, k, labeling_function)
    cost += weight_not_matched * get_cost_not_matched(egraph_1, egraph_2, mapping, k, basic_cost)
    cost += weight_structure * get_cost_structure(egraph_1, egraph_2, mapping)
    gmc2 = time()
    print("get mapping cost time:", gmc2 - gmc1)
    return cost

def get_cost_matched(egraph_1, egraph_2, mapping, k, labeling_function = default_labeling_function):
    cost = 0
    if mapping == None:
        return 0
    gcm1 = time()
    #print("egraph1: ", egraph_1.transitive_reduction.nodes())
    #print("egraph2: ", egraph_2.transitive_reduction.nodes())
    #print("mapping:", mapping)
    for (nodeID_1, nodeID_2) in mapping:
        cost += get_label_cost(egraph_1.nodeID_to_event_dict[nodeID_1], egraph_2.nodeID_to_event_dict[nodeID_2], labeling_function)
        cost += graph_util.get_dissimilar_predecessors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function)
        cost += graph_util.get_dissimilar_successors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function)
        cost += graph_util.get_dissimilar_concurrencies(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function)
    gcm2 = time()
    print("get cost matched time: ", gcm2 - gcm1)
    return cost

def get_cost_not_matched(egraph_1, egraph_2, mapping, k, basic_cost = 2):
    cost = 0
    gcnm1 = time()
    if mapping == None:
        not_matched_nodes1 = range(0, len(egraph_1))
        not_matched_nodes2 = range(0, len(egraph_2))
    else:
        matched_trace1 = [match[0] for match in mapping]
        not_matched_nodes1 = [node for node in range(0, egraph_1.transitive_reduction.number_of_nodes()) if node not in matched_trace1]
        matched_trace2 = [match[1] for match in mapping]
        not_matched_nodes2 = [node for node in range(0, egraph_2.transitive_reduction.number_of_nodes()) if node not in matched_trace2]


    for node in not_matched_nodes1:
        cost += basic_cost
        cost += graph_util.get_neighbor_size(egraph_1, node, k)
    for node in not_matched_nodes2:
        cost += basic_cost
        cost += graph_util.get_neighbor_size(egraph_2, node, k)

    gcnm2 = time()
    print("get cost not matched time: ", gcnm2 - gcnm1)
    return cost

def get_cost_structure(egraph_1, egraph_2, mapping):
    gcs1 = time()
    cost = 0
    if mapping == None:
        return 0
    for (node1_a, node1_b) in mapping:
        for (node2_a, node2_b) in mapping:
            cost += abs(graph_util.dist(egraph_1, node1_a, node2_a) - graph_util.dist(egraph_2, node1_b, node2_b))#/2

    gcs2 = time()
    print("get cost structure time: ", gcs2 - gcs1)
    return cost