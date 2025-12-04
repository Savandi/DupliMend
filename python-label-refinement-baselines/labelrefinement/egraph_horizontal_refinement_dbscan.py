import egraph_mapping_dynamic as egraph_mapping
import egraph_vertical_refinement
import egraph_class
import copy
from time import time
import egraph_horizontal_refinement_cc as egraph_horizontal_refinement
import trace_util as graph_util
from egraph_mapping_cost_recursive import default_labeling_function as labeling_function

def get_connected_components(egraphs, mappings, label, labeling_function, variant_threshold):
    eps = variant_threshold
    #MinPts = len(egraphs) * 0.05
    MinPts = 1#len(egraphs) * 0.05
    #MinPts = 0
    #eps=1
    visited_nodes = []
    noise = []
    clusters = []
    mappings = transform_mappings_triangle_to_qmatrix(mappings)
    for egraph_index in (range(0, len(egraphs))):
        for node_index in (range(0, egraphs[egraph_index].size)): #we iterate over all "points"
            if labeling_function(egraphs[egraph_index].nodeID_to_event_dict[node_index]) == label and (egraph_index, node_index) not in visited_nodes:
                visited_nodes.append((egraph_index, node_index))
                neighbors = get_neighbors(egraphs, egraph_index, node_index, mappings, eps)

                if len(neighbors) < MinPts: # if no core point and not visited: add to NOISE
                    noise.append((egraph_index, node_index))
                else:
                    noise, mappings, egraph_index, node_index, visited_nodes, neighbors, cluster, clusters, eps, MinPts =\
                        expand_cluster(egraphs, noise, mappings, egraph_index, node_index, visited_nodes, neighbors, [], clusters, eps, MinPts)
                    clusters.append(cluster)

    #print("clusters: ", clusters)
    print("noise: ", noise)
    return clusters, noise
    '''
    remove_from_noise = []
    for noise_graph, noise_node in noise:
        distance_to_cluster = 99999
        closest_cluster = "nc"

        for c_index in range(0, len(clusters)):
            cluster = clusters[c_index]
            for cluster_graph, cluster_node in cluster:
                mapping, mapping_cost = mappings[noise_graph][cluster_graph]
                if (noise_node, cluster_node) in mapping:
                    if mapping_cost < distance_to_cluster:
                        distance_to_cluster = mapping_cost
                        closest_cluster = c_index

        if closest_cluster != "nc":
            clusters[closest_cluster].append((noise_graph, noise_node))
            remove_from_noise.append((noise_graph, noise_node))

    
    print("final noise: ", set(noise) - set(remove_from_noise))
    return clusters, set(noise) - set(remove_from_noise)
    '''
def get_neighbors(egraphs, egraph_index, node_index, mappings, eps):

    neighbors = {(egraph_index, node_index)}#set()
    for mapped_egraph_index in (range(0, len(mappings))):
        mapping, mapping_cost = mappings[egraph_index][mapped_egraph_index]
        for source_node, target_node in mapping:
            #if source_node == node_index and mapping_cost < eps:
            #    neighbors.add((mapped_egraph_index, target_node))
            if source_node == node_index:
                cost = get_weight(egraphs, egraph_index, source_node, mapped_egraph_index, target_node, mappings)
                if cost <= eps:
                    neighbors.add((mapped_egraph_index, target_node))

    #print(len(neighbors))
    return neighbors



def expand_cluster(egraphs, noise, mappings, egraph_index, node_index, visited_nodes, neighbors, cluster, clusters, eps, MinPts):
    cluster.append((egraph_index, node_index))
    while len(neighbors) != 0:
        neighbor_graph_index, neighbor_node_index = neighbors.pop()
        if (neighbor_graph_index, neighbor_node_index) not in visited_nodes:
            visited_nodes.append((neighbor_graph_index, neighbor_node_index))
            new_neighbors = get_neighbors(egraphs, neighbor_graph_index, neighbor_node_index, mappings, eps)
            #print("new_neighbors",new_neighbors)
            if len(new_neighbors) >= MinPts:
                #print("n  ", neighbors)
                #print("nn ", new_neighbors)
                neighbors = neighbors.union(new_neighbors)
            in_c = False
            for c in clusters: #stop if "point" is allready in a cluster
                if (neighbor_graph_index, neighbor_node_index) in c:
                    in_c = True

            #        noise.append((neighbor_graph_index, neighbor_node_index))
            #        #return noise, mappings, egraph_index, node_index, visited_nodes, neighbors, cluster, clusters, eps, MinPts
            if not in_c:
                cluster.append((neighbor_graph_index, neighbor_node_index))

            #if (neighbor_graph_index, neighbor_node_index) in noise:
            #   noise.remove((neighbor_graph_index, neighbor_node_index))
        #else:
        #
        #    for c in clusters: #stop if "point" is allready in a cluster
        #        if (neighbor_graph_index, neighbor_node_index) in c:
        #            noise.append((neighbor_graph_index, neighbor_node_index))

    return noise, mappings, egraph_index, node_index, visited_nodes, neighbors, cluster, clusters, eps, MinPts


def get_weight(egraphs, egraph_index1, node_index1, egraph_index2, node_index2, mappings, k=1, weight_single_matching=0.9, weight_single_structure=0.1):#weight_single_matching=0.7, weight_single_structure=0.3):
    k = 2
    mapping, mapping_cost = mappings[egraph_index1][egraph_index2]
    #print("weight: ", mapping_cost)
    egraph1 = egraphs[egraph_index1]
    egraph2 = egraphs[egraph_index2]

    cost_matching = graph_util.get_dissimilar_neighbors(egraph1, egraph2, mapping, node_index1, node_index2, k, labeling_function)
    #cost_matching += graph_util.get_dissimilar_successors(egraph1, egraph2, mapping, node_index1, node_index2, k, labeling_function)

    cost_structure = abs(node_index1 - node_index2)  #distance to dummy start
    cost_structure += abs(abs(egraph1.size - node_index1) - abs(egraph2.size - node_index2))          #distance to dummy end
    for (nodeID_1B, nodeID_2B) in mapping:
        cost_structure += abs(graph_util.dist(egraph1, node_index1, nodeID_1B) - graph_util.dist(egraph2, node_index2, nodeID_2B))#/2


    cost = weight_single_matching * cost_matching + weight_single_structure * cost_structure
    normalization_coefficient = 2 * k # maximum for dissimilarity in post and pre contexts
    normalization_coefficient += 1/3 * (max(egraph1.size, egraph2.size) - 2) * max(egraph1.size, egraph2.size)#worst case for structure cost

    normalization_coefficient = normalization_coefficient / 5
    weight_global_mapping = 1
    weight_local_mapping = 0

    #print("mapping cost", mapping_cost)
    #print("local   cost", cost/normalization_coefficient)

    return weight_global_mapping * mapping_cost + weight_local_mapping * cost/normalization_coefficient




def transform_mappings_triangle_to_qmatrix(mappings):
    #print(mappings)
    new_mappings = mappings
    for i in range(0, len(mappings)):
        for j in range(i+1, len(mappings)):
            new_mappings[i].append(([(m[1], m[0]) for m in mappings[j][i][0]], mappings[j][i][1]))

    return new_mappings



