#todo: completely overwork the egraph layer by calculating all distances, contexts and importances for an egraph in the beginning and then always using networkx graphs

import networkx as nx
from time import time
'''
def get_dissimilar_neighbors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function, local_mode=False):

    predecessors1 = egraph_1.contexts[nodeID_1][1]#get_predecessors(egraph_1, nodeID_1, k)
    predecessors2 = egraph_2.contexts[nodeID_2][1]#get_predecessors(egraph_2, nodeID_2, k)
    successors1 = egraph_1.contexts[nodeID_1][3]#get_successors(egraph_1, nodeID_1, k)
    successors2 = egraph_2.contexts[nodeID_2][3]#get_successors(egraph_2, nodeID_2, k)

    neighbors1 = set(predecessors1).union(successors1)
    neighbors2 = set(predecessors2).union(successors2)


    neighbor_labels1 = set()
    neighbor_labels2 = set()

    #print("n1: ", neighbors1)
    for n in neighbors1:
        #print(labeling_function(egraph_1.nodeID_to_event_dict[n]))
        neighbor_labels1.add(labeling_function(egraph_1.nodeID_to_event_dict[n]))
    #print("n2: ", neighbors2)
    for n in neighbors2:
        #print(labeling_function(egraph_2.nodeID_to_event_dict[n]))
        neighbor_labels2.add(labeling_function(egraph_2.nodeID_to_event_dict[n]))

    #print("len: ", len(neighbor_labels1 ^ neighbor_labels2))
    return len(neighbor_labels1 ^ neighbor_labels2)

    predecessors1 = egraph_1.contexts[nodeID_1][1]#get_predecessors(egraph_1, nodeID_1, k)
    predecessors2 = egraph_2.contexts[nodeID_2][1]#get_predecessors(egraph_2, nodeID_2, k)
    successors1 = egraph_1.contexts[nodeID_1][3]#get_successors(egraph_1, nodeID_1, k)
    successors2 = egraph_2.contexts[nodeID_2][3]#get_successors(egraph_2, nodeID_2, k)

    neighbors1 = set(predecessors1).union(successors1)
    neighbors2 = set(predecessors2).union(successors2)


    similar_mapping_counter = 0
    for mapped_node1, mapped_node2 in mapping:
        if mapped_node1 in neighbors1 and mapped_node2 in neighbors2:
            similar_mapping_counter = similar_mapping_counter + 1

    if nodeID_1 < k and nodeID_2 < k:
        similar_mapping_counter = similar_mapping_counter + 1
    if egraph_1.size - k < nodeID_1 and egraph_2.size - k < nodeID_2:
        similar_mapping_counter = similar_mapping_counter + 1

    return len(neighbors1) + len(neighbors2) - 4 * similar_mapping_counter # 4* because if we find a similar mapping, then an other mapped pair also has one more similar mapping
'''
def get_dissimilar_predecessors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function,
                                count_mapped_neighbors=False ,local_mode=False, use_mapping_and_label_neighbors=False):
    predecessors1 = egraph_1.contexts[nodeID_1][1]  # get_predecessors(egraph_1, nodeID_1, k)
    predecessors2 = egraph_2.contexts[nodeID_2][1]  # get_predecessors(egraph_2, nodeID_2, k)

    similar_mapping_counter = 0
    for mapped_node1, mapped_node2 in mapping:
        if mapped_node1 in predecessors1 and mapped_node2 in predecessors2:
            similar_mapping_counter = similar_mapping_counter + 1

    if nodeID_1 < k and nodeID_2 < k:
        similar_mapping_counter = similar_mapping_counter + 1
    added_size = 0
    if nodeID_1 < k:
        added_size += 1
    if nodeID_2 < k:
        added_size += 1

    if local_mode:  # no dynamic calc needed for local mapping cost calculation
        mapping_cost = added_size + len(predecessors1) + len(predecessors2) - 2 * similar_mapping_counter
    else:
        mapping_cost = added_size + len(predecessors1) + len(predecessors2) - 4 * similar_mapping_counter

    if not use_mapping_and_label_neighbors:
        return mapping_cost

    else:
        #if not count_mapped_neighbors:
        predecessors1 = egraph_1.contexts[nodeID_1][1]#get_predecessors(egraph_1, nodeID_1, k)
        predecessors2 = egraph_2.contexts[nodeID_2][1]#get_predecessors(egraph_2, nodeID_2, k)

        neighbors1 = set(predecessors1)
        neighbors2 = set(predecessors2)

        neighbor_labels1 = set()
        neighbor_labels2 = set()

        #print("n1: ", neighbors1)
        for n in neighbors1:
            #print(egraph_1.nodeID_to_event_dict[n]["concept:name"])
            neighbor_labels1.add(labeling_function(egraph_1.nodeID_to_event_dict[n]))
        #print("n2: ", neighbors2)
        for n in neighbors2:
            #print(egraph_2.nodeID_to_event_dict[n]["concept:name"])
            neighbor_labels2.add(labeling_function(egraph_2.nodeID_to_event_dict[n]))

        if nodeID_1 < k:
            neighbor_labels1.add("dummy_start")
        if nodeID_2 < k:
            neighbor_labels2.add("dummy_start")
        label_cost = len(neighbor_labels1 ^ neighbor_labels2)

        return (1 / 2) * mapping_cost + (1 / 2) * label_cost

def get_dissimilar_successors(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function,
                              count_mapped_neighbors=False, local_mode=False, use_mapping_and_label_neighbors=False):

    #else:
        successors1 = egraph_1.contexts[nodeID_1][3]#get_successors(egraph_1, nodeID_1, k)
        successors2 = egraph_2.contexts[nodeID_2][3]#get_successors(egraph_2, nodeID_2, k)
        similar_mapping_counter = 0

        added_size = 0
        if nodeID_1 > egraph_1.size - k:
            added_size += 1
        if nodeID_2 > egraph_2.size - k:
            added_size += 1
        if egraph_1.size - k < nodeID_1 and egraph_2.size - k < nodeID_2:
            similar_mapping_counter = similar_mapping_counter + 1


        for mapped_node1, mapped_node2 in mapping:
            if mapped_node1 in successors1 and mapped_node2 in successors2:
                similar_mapping_counter = similar_mapping_counter + 1

        if local_mode:
            mapping_cost = added_size + len(successors1) + len(successors2) - 2 * similar_mapping_counter  # -4*... because all simmilar neighbors also have one more similar neighbor
        else:
            mapping_cost = added_size + len(successors1) + len(successors2) - 4 * similar_mapping_counter

        if not use_mapping_and_label_neighbors:
            return mapping_cost

        else:
            #if not count_mapped_neighbors:
            successors1 = egraph_1.contexts[nodeID_1][3]#get_successors(egraph_1, nodeID_1, k)
            successors2 = egraph_2.contexts[nodeID_2][3]#get_successors(egraph_2, nodeID_2, k)
            neighbors1 = set(successors1)
            neighbors2 = set(successors2)
            neighbor_labels1 = set()
            neighbor_labels2 = set()

            for n in neighbors1:
                neighbor_labels1.add(labeling_function(egraph_1.nodeID_to_event_dict[n]))
            for n in neighbors2:
                neighbor_labels2.add(labeling_function(egraph_2.nodeID_to_event_dict[n]))
            if nodeID_1 > egraph_1.size - k:
                neighbor_labels1.add("dummy_end")
            if nodeID_2 > egraph_2.size - k:
                neighbor_labels2.add("dummy_end")
            label_cost =  len (neighbor_labels1 ^ neighbor_labels2)

            return (1 / 2) * mapping_cost + (1 / 2) * label_cost

def get_dissimilar_concurrencies(egraph_1, egraph_2, mapping, nodeID_1, nodeID_2, k, labeling_function):
    return 0        #todo return 0 when no timestamp and atomic
    '''
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
    '''

def get_neighbor_size(egraph, nodeID, k):     #paper is unclear, we here take number of preds and succs to measure the importance of event
    #size = 0
    #if nodeID < k:
    #    size += 1
    #if egraph.size - k < nodeID:
    #    size += 1
    return len(egraph.contexts[nodeID][1]) + len(egraph.contexts[nodeID][3]) # + len(egraph.contexts[nodeID][2]) ???


def dist(trace, nodeID_1, nodeID_2): #symmetric
    return trace.distances[nodeID_1][nodeID_2]#max(nodeID_1 - nodeID_2, nodeID_2 - nodeID_1)

def get_predecessors(trace, position, k):
    #print(list(range(max(0, position - k), position)))
    return trace.contexts[position][1]#range(max(0, position - k), position)


def get_successors(trace, position, k):
    return trace.contexts[position][3]#range(position + 1, min(position + k + 1, trace.size))

def get_concurrencies(trace, position):
    return []
    #neighbors = set(egraph.partial_order.successors(nodeID))
    #neighbors = neighbors.union(set(egraph.partial_order.predecessors(nodeID)))
    #concurrencies = set(egraph.partial_order.nodes()).difference(neighbors)
    #concurrencies.remove(nodeID)
    return concurrencies

def get_nodes_sorted_by_importance(egraph_1, egraph_2, k, labeling_function):
    contexts1 = egraph_1.contexts
    contexts2 = egraph_2.contexts
    visited1 = []
    visited2 = []
    priority_list = []

    for nodeID_1 in range(0, egraph_1.size):
        if nodeID_1 not in visited1:
            importance = 0
            visited1.append(nodeID_1)
            context = contexts1[nodeID_1]
            nodes_with_same_context = [(nodeID_1, 1)]

            for current_nodeID in range(0, egraph_1.size):
                if current_nodeID not in visited1:
                    current_context = contexts1[current_nodeID]
                    if equal_context(context, current_context):
                        importance = importance + 1
                        nodes_with_same_context.append((current_nodeID, 1))
                        visited1.append(current_nodeID)

            for current_nodeID in range(0, egraph_2.size):
                    if current_nodeID not in visited2:
                        current_context = contexts2[current_nodeID]
                        if equal_context(context, current_context):
                            importance = importance + 1
                            nodes_with_same_context.append((current_nodeID, 2))
                            visited2.append(current_nodeID)
            for node, graph_nr in nodes_with_same_context:
                priority_list.append((node, graph_nr, importance))

    for nodeID_2 in range(0, egraph_2.size):
        if nodeID_2 not in visited2:
            importance = 0
            visited2.append(nodeID_2)
            context = contexts2[nodeID_2]
            nodes_with_same_context = [(nodeID_2, 2)]
            for current_nodeID in range(0, egraph_1.size):
                if current_nodeID not in visited1:
                    current_context = contexts1[current_nodeID]
                    if equal_context(context, current_context):
                        importance = importance + 1
                        nodes_with_same_context.append((current_nodeID, 1))
                        visited1.append(current_nodeID)

            for current_nodeID in range(0, egraph_2.size):
                if current_nodeID not in visited2:
                    current_context = contexts2[current_nodeID]
                    if equal_context(context, current_context):
                        importance = importance + 1
                        nodes_with_same_context.append((current_nodeID, 2))
                        visited2.append(current_nodeID)
            for node, graph_nr in nodes_with_same_context:
                priority_list.append((node, graph_nr, importance))




    if egraph_1.size > egraph_2.size:
        priority_list = sorted(priority_list, key=lambda x: (-x[1], x[2]), reverse=True)
    else:
        priority_list = sorted(priority_list, key=lambda x: (x[1], x[2]), reverse=True)
    #print(importance_list)



    #print("importance list: ", priority_list)
    return priority_list




def get_nodes_sorted_by_importance_reworked(egraph_1, egraph_2, k, labeling_function):
    contexts1 = egraph_1.contexts
    contexts2 = egraph_2.contexts
    priority_list = []

    for nodeID_1 in range(0, egraph_1.size):
        min_dissimilarity = 99999

        context = contexts1[nodeID_1]

        for current_nodeID in range(0, egraph_2.size):
            current_context = contexts2[current_nodeID]
            dissimilarity = get_dissimilarity(egraph_1, egraph_2, nodeID_1, current_nodeID, context, current_context, k, labeling_function)

            if min_dissimilarity > dissimilarity:
                min_dissimilarity = dissimilarity
        if min_dissimilarity != 99999:
            priority_list.append((nodeID_1, 1, min_dissimilarity))

    priority_list = sorted(priority_list, key=lambda x: (x[2]), reverse=False)
    return priority_list




def get_dissimilarity(egraph_1, egraph_2, nodeID_1, nodeID_2, context1, context2, k, labeling_function):

    if context1[0] != context2[0]:
        return 99999

    struct_dissimilarity = abs(nodeID_1 - nodeID_2) + abs((egraph_1.size - nodeID_1) - (egraph_2.size - nodeID_2))

    successors1 = context1[3]
    successors2 = context2[3]
    neighbors1 = set(successors1)
    neighbors2 = set(successors2)
    neighbor_labels1 = set()
    neighbor_labels2 = set()

    for n in neighbors1:
        neighbor_labels1.add(labeling_function(egraph_1.nodeID_to_event_dict[n]))
    for n in neighbors2:
        neighbor_labels2.add(labeling_function(egraph_2.nodeID_to_event_dict[n]))
    if nodeID_1 > egraph_1.size - k:
        neighbor_labels1.add("dummy_end")
    if nodeID_2 > egraph_2.size - k:
        neighbor_labels2.add("dummy_end")
    succ_dissimilarity = len(neighbor_labels1 ^ neighbor_labels2)

    predecessors1 = context1[1]
    predecessors2 = context2[1]
    neighbors1 = set(predecessors1)
    neighbors2 = set(predecessors2)
    neighbor_labels1 = set()
    neighbor_labels2 = set()

    for n in neighbors1:
        neighbor_labels1.add(labeling_function(egraph_1.nodeID_to_event_dict[n]))
    for n in neighbors2:
        neighbor_labels2.add(labeling_function(egraph_2.nodeID_to_event_dict[n]))

    if nodeID_1 < k:
        neighbor_labels1.add("dummy_start")
    if nodeID_2 < k:
        neighbor_labels2.add("dummy_start")
    pred_dissimilarity = len(neighbor_labels1 ^ neighbor_labels2)


    return pred_dissimilarity + succ_dissimilarity + struct_dissimilarity




def equal_context(context1, context2):
    if set(context1[0]) != set(context2[0]):
        return False
    if set(context1[1])!= set(context2[1]):
        return False
    if set(context1[2]) != set(context2[2]):
        return False
    if set(context1[3]) != set(context2[3]):
        return False
    return True