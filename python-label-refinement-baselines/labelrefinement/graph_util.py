import networkx as nx

def get_dissimilar_predecessors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function):
    predecessors1 = get_predecessors(egraph_1, nodeID_1, k)
    predecessors2 = get_predecessors(egraph_2, nodeID_2, k)

    similar_mapping_counter = 0
    for mapped_node1, mapped_node2 in mapping:
        if mapped_node1 in predecessors1 and mapped_node2 in predecessors2:
            similar_mapping_counter = similar_mapping_counter + 1

    return len(predecessors1) + len(predecessors2) - 4 * similar_mapping_counter
    #predecessors1 = set([labeling_function(egraph_1.nodeID_to_event_dict[event]) for event in predecessors1])
    #predecessors2 = set([labeling_function(egraph_2.nodeID_to_event_dict[event]) for event in predecessors2])
    #return len(predecessors1.symmetric_difference(predecessors2))

def get_dissimilar_successors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function):
    successors1 = get_successors(egraph_1, nodeID_1, k)
    successors2 = get_successors(egraph_2, nodeID_2, k)

    similar_mapping_counter = 0
    for mapped_node1, mapped_node2 in mapping:
        if mapped_node1 in successors1 and mapped_node2 in successors2:
            similar_mapping_counter = similar_mapping_counter + 1

    return len(successors1) + len(successors2) - 4 * similar_mapping_counter

    #successors1 = set([labeling_function(egraph_1.nodeID_to_event_dict[event]) for event in successors1])
    #successors2 = set([labeling_function(egraph_2.nodeID_to_event_dict[event]) for event in successors2])
    #return len(successors1.symmetric_difference(successors2))

def get_dissimilar_concurrencies(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function):
    return 0        #todo return 0 wen no timestamp and atomic
    concurrencies1 = get_concurrencies(egraph_1, nodeID_1)
    concurrencies2 = get_concurrencies(egraph_2, nodeID_2)

    similar_mapping_counter = 0
    for mapped_node1, mapped_node2 in mapping:
        if mapped_node1 in concurrencies1 and mapped_node2 in concurrencies2:
            similar_mapping_counter = similar_mapping_counter + 1

    return len(concurrencies1) + len(concurrencies2) - 4 * similar_mapping_counter
    #concurrencies1 = set([labeling_function(egraph_2.nodeID_to_event_dict[event]) for event in concurrencies1])
    #concurrencies2 = set([labeling_function(egraph_2.nodeID_to_event_dict[event]) for event in concurrencies2])
    #return len(concurrencies1.symmetric_difference(concurrencies2))

def get_neighbor_size(egraph, nodeID, k):     #paper is unclear, we here take number of preds and succs to measure the importance of event
    return len(get_predecessors(egraph, nodeID, k)) + len(get_successors(egraph, nodeID, k))

'''
def get_cost_not_matched(trace1, trace2, mapping, k, basic_cost = 2):
    cost = 0

    matched_trace1 = [match[0] for match in mapping]
    not_matched_trace1 = [position for position in range(0, len(trace1)) if position not in matched_trace1]

    matched_trace2 = [match[1] for match in mapping]
    not_matched_trace2 = [position for position in range(0, len(trace2)) if position not in matched_trace2]
    #print(not_matched_trace1)
    #print(not_matched_trace2)
    for position in not_matched_trace1:
        #print(position)
        cost += basic_cost
        cost += get_neighbor_size(trace1, position, k)
    for position in not_matched_trace2:
        #print(position)
        cost += basic_cost
        cost += get_neighbor_size(trace2, position, k)

    return cost

'''
def dist(egraph, nodeID_1, nodeID_2): #symmetric
    try:
        #print("hallo1")
        return nx.shortest_path_length(egraph.transitive_reduction, nodeID_1, nodeID_2)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    try:
        #print("hallo2")
        return nx.shortest_path_length(egraph.transitive_reduction, nodeID_2, nodeID_1)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        #print("hallo3")
        return 0

def get_predecessors(egraph, nodeID, k):
    preds = {nodeID}
    for counter in range(0, k):
        new_preds = preds
        for pred in preds:
            new_preds = new_preds.union(set(egraph.transitive_reduction.predecessors(pred)))
        preds = new_preds
    preds.remove(nodeID)
    return preds

def get_successors(egraph, nodeID, k):
    succs = {nodeID}
    for counter in range(0, k):
        new_succs = succs
        for succ in succs:
            new_succs = new_succs.union(set(egraph.transitive_reduction.successors(succ)))
        succs = new_succs
    succs.remove(nodeID)
    return succs

def get_concurrencies(egraph, nodeID):
    return set()
    neighbors = set(egraph.partial_order.successors(nodeID))
    neighbors = neighbors.union(set(egraph.partial_order.predecessors(nodeID)))
    concurrencies = set(egraph.partial_order.nodes()).difference(neighbors)
    concurrencies.remove(nodeID)
    return concurrencies

def get_nodes_sorted_by_importance(egraph_1, egraph_2, k, labeling_function): #todo increase speed by 1/2

    #max_context_frequency = {label: get_maximum_context_frequency(trace1, trace2, label, k, labeling_function}
    importance_list = [] #(position, trace_nr, importance)
    for nodeID_1 in range(0, egraph_1.size):
        importance = 0
        label = labeling_function(egraph_1.nodeID_to_event_dict[nodeID_1])
        predecessors = set(labeling_function(egraph_1.nodeID_to_event_dict[pred]) for pred in get_predecessors(egraph_1, nodeID_1, k))
        successors = set(labeling_function(egraph_1.nodeID_to_event_dict[succ]) for succ in get_successors(egraph_1, nodeID_1, k))
        concurrencies = set(labeling_function(egraph_1.nodeID_to_event_dict[conc]) for conc in get_concurrencies(egraph_1, nodeID_1))
        #print("label: " + str(label))
        #print("preds: " + str(predecessors))
        #print("succs: " + str(successors))
        #print("conc: " + str(concurrencies))

        for current_nodeID in range(0, egraph_1.size):
            current_label = labeling_function(egraph_1.nodeID_to_event_dict[current_nodeID])
            current_predecessors = set(labeling_function(egraph_1.nodeID_to_event_dict[pred]) for pred in get_predecessors(egraph_1, current_nodeID, k))
            current_successors = set(labeling_function(egraph_1.nodeID_to_event_dict[succ]) for succ in get_successors(egraph_1, current_nodeID, k))
            current_concurrencies = set(labeling_function(egraph_1.nodeID_to_event_dict[conc]) for conc in get_concurrencies(egraph_1, current_nodeID))
            if label == current_label and predecessors == current_predecessors \
                    and successors == current_successors and concurrencies == current_concurrencies:  # todo : change to continuous measurement
                importance = importance + 1

        for current_nodeID in range(0, egraph_2.size):
            current_label = labeling_function(egraph_2.nodeID_to_event_dict[current_nodeID])
            current_predecessors = set(labeling_function(egraph_2.nodeID_to_event_dict[pred]) for pred in get_predecessors(egraph_2, current_nodeID, k))
            current_successors = set(labeling_function(egraph_2.nodeID_to_event_dict[succ]) for succ in get_successors(egraph_2, current_nodeID, k))
            current_concurrencies = set(labeling_function(egraph_2.nodeID_to_event_dict[conc]) for conc in get_concurrencies(egraph_2, current_nodeID))
            if label == current_label and predecessors == current_predecessors \
                    and successors == current_successors and concurrencies == current_concurrencies:
                importance = importance + 1

        importance_list.append((nodeID_1, 1, importance))

    for nodeID_2 in range(0, egraph_2.size):
        importance = 0
        label = labeling_function(egraph_2.nodeID_to_event_dict[nodeID_2])
        predecessors = set(labeling_function(egraph_2.nodeID_to_event_dict[pred]) for pred in
                           get_predecessors(egraph_2, nodeID_2, k))
        successors = set(labeling_function(egraph_2.nodeID_to_event_dict[succ]) for succ in
                         get_successors(egraph_2, nodeID_2, k))
        concurrencies = set(labeling_function(egraph_2.nodeID_to_event_dict[conc]) for conc in
                            get_concurrencies(egraph_2, nodeID_2))
        # print("label: " + str(label))
        # print("preds: " + str(predecessors))
        # print("succs: " + str(successors))
        # print("conc: " + str(concurrencies))

        for current_nodeID in range(0, egraph_1.size):
            current_label = labeling_function(egraph_1.nodeID_to_event_dict[current_nodeID])
            current_predecessors = set(labeling_function(egraph_1.nodeID_to_event_dict[pred]) for pred in
                                       get_predecessors(egraph_1, current_nodeID, k))
            current_successors = set(labeling_function(egraph_1.nodeID_to_event_dict[succ]) for succ in
                                     get_successors(egraph_1, current_nodeID, k))
            current_concurrencies = set(labeling_function(egraph_1.nodeID_to_event_dict[conc]) for conc in
                                        get_concurrencies(egraph_1, current_nodeID))
            if label == current_label and predecessors == current_predecessors \
                    and successors == current_successors and concurrencies == current_concurrencies:
                importance = importance + 1

        for current_nodeID in range(0, egraph_2.size):
            current_label = labeling_function(egraph_2.nodeID_to_event_dict[current_nodeID])
            current_predecessors = set(labeling_function(egraph_2.nodeID_to_event_dict[pred]) for pred in
                                    get_predecessors(egraph_2, current_nodeID, k))
            current_successors = set(labeling_function(egraph_2.nodeID_to_event_dict[succ]) for succ in
                                    get_successors(egraph_2, current_nodeID, k))
            current_concurrencies = set(labeling_function(egraph_2.nodeID_to_event_dict[conc]) for conc in
                                        get_concurrencies(egraph_2, current_nodeID))
            if label == current_label and predecessors == current_predecessors \
                        and successors == current_successors and concurrencies == current_concurrencies:
                importance = importance + 1
        importance_list.append((nodeID_2, 2, importance))
    #importance_list = sorted(importance_list, key=lambda x: x[2], reverse=True)
    if egraph_1.size > egraph_2.size:
        importance_list = sorted(importance_list, key=lambda x: (x[2], x[1]), reverse=True) #todo: fix error ttat nodeids are sorted by rap size first
    else:
        importance_list = sorted(importance_list, key=lambda x: (x[2], -x[1]), reverse=True)
    #print(importance_list)

    return importance_list