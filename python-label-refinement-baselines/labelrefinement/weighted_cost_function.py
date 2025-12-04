import trace_util as graph_util
from egraph_mapping_cost_recursive import default_labeling_function as labeling_function

def get_weight(egraphs, egraph_index1, node_index1, egraph_index2, node_index2, mappings, k=1, weight_single_matching=5,
               weight_single_structure=1, use_mapping_and_label_neighbors=False, use_local_cost=False):#weight_single_matching=0.7, weight_single_structure=0.3):
    k = 3
    use_mapping_and_label_neighbors = True
    mapping, mapping_cost = mappings[egraph_index1][egraph_index2]
    if not use_local_cost:
        return 1, mapping_cost
    else:
        egraph1 = egraphs[egraph_index1]
        egraph2 = egraphs[egraph_index2]

        cost_matching = graph_util.get_dissimilar_predecessors(egraph1, egraph2, mapping, node_index1, node_index2, k,
                                                               labeling_function, local_mode=True, use_mapping_and_label_neighbors=use_mapping_and_label_neighbors) #"local" = non dynamic
        cost_matching += graph_util.get_dissimilar_successors(egraph1, egraph2, mapping, node_index1, node_index2, k,
                                                               labeling_function, local_mode=True, use_mapping_and_label_neighbors=use_mapping_and_label_neighbors)

        cost_structure = abs(egraph1.dist_to_start[node_index1] - egraph2.dist_to_start[node_index2])  #distance to dummy start
        cost_structure += abs(egraph1.dist_to_end[node_index1] - egraph2.dist_to_end[node_index2])          #distance to dummy end
        for (nodeID_1B, nodeID_2B) in mapping:
            cost_structure += abs(graph_util.dist(egraph1, node_index1, nodeID_1B) - graph_util.dist(egraph2, node_index2, nodeID_2B))#/2

        local_cost = weight_single_matching * cost_matching + weight_single_structure * cost_structure



        return local_cost, mapping_cost
