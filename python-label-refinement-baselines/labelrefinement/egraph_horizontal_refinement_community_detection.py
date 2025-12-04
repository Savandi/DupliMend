import egraph_mapping_dynamic as egraph_mapping
import egraph_vertical_refinement
import egraph_class
import copy
from time import time
import egraph_horizontal_refinement_cc as egraph_horizontal_refinement
import trace_util as graph_util
from egraph_mapping_cost_recursive import default_labeling_function as labeling_function
import networkx as nx
import matplotlib.pyplot as plt
# import metis
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    print("[WARNING] python-louvain not installed - community detection will not work")
import matplotlib.cm as cm
import weighted_cost_function
import igraph as ig
import leidenalg as la


def get_communities(egraphs, mappings, label, labeling_function, variant_threshold,
                          use_local_cost, mapping_evaluation_mode, use_mapping_and_label_neighbors):
    #print("incom",use_local_cost)

    G = nx.Graph()
    index_to_node_dict = {}
    node_to_index_dict = {}
    index = 0
    for egraph_index in (range(0, len(egraphs))):
        for nodeID in range(0, egraphs[egraph_index].size):
            if labeling_function(egraphs[egraph_index].nodeID_to_event_dict[nodeID]) == label:
                G.add_node(index)
                index_to_node_dict[index] = (egraph_index, nodeID)
                node_to_index_dict[(egraph_index, nodeID)] = index
                index += 1
    # print('index_to_node_dict')
    # print(index_to_node_dict)
    #
    # print('len(egraphs) CD')
    # print(len(egraphs))

    max_local_cost = 0
    max_gloabal_cost = 0
    for egraph_index1 in (range(0, len(egraphs))):
        for egraph_index2 in (range(0, len(egraphs))):
            mapping, cost = mappings[egraph_index1][egraph_index2]
            for m1, m2 in mapping:
                if labeling_function(egraphs[egraph_index1].nodeID_to_event_dict[m1]) == label:
                    local_cost, global_cost = weighted_cost_function.get_weight(egraphs, egraph_index1, m1, egraph_index2, m2, mappings,
                                                                                use_mapping_and_label_neighbors, use_local_cost=use_local_cost)
                    if max_local_cost < local_cost:
                        max_local_cost = local_cost
                    if max_gloabal_cost < global_cost:
                        max_gloabal_cost = global_cost
                    G.add_edge(node_to_index_dict[(egraph_index1, m1)], node_to_index_dict[(egraph_index2, m2)],
                               local_cost=local_cost, global_cost=global_cost)

    max_cost = 0
    for u, v, d in G.edges(data=True):
        #print("moin ", G[u][v]["local_cost"], G[u][v]["global_cost"])
        if max_local_cost > 0 and G[u][v]["global_cost"] > 0:
            G[u][v]["local_cost"] = 20 * G[u][v]["local_cost"]/G[u][v]["global_cost"]
            #G[u][v]["local_cost"] = G[u][v]["local_cost"]/max_local_cost
        else:
            G[u][v]["local_cost"] = 0
        if max_gloabal_cost > 0:
            G[u][v]["global_cost"] = G[u][v]["global_cost"]/max_gloabal_cost
        else:
            G[u][v]["global_cost"] = 0

        if use_local_cost:
            G[u][v]["cost"] =  G[u][v]["local_cost"] + G[u][v]["global_cost"]
            if G[u][v]["cost"] > max_cost:
                max_cost = G[u][v]["cost"]
        else:
            G[u][v]["weight"] = 1 - G[u][v]["global_cost"]


    if use_local_cost:
        for u, v, d in G.edges(data=True):
            G[u][v]["weight"] = 1 - G[u][v]["cost"]/max_cost
            #print(G[u][v]["local_cost"], G[u][v]["global_cost"], G[u][v]["weight"])

    # partitions = community_louvain.best_partition(G, weight="weight", randomize=False)
    # print("partitions: ", partitions)
    # todo: take CC if modularity bellow treshold
    # modularity = community_louvain.modularity(partitions, G)
    # print("modularity: ", modularity)
    # if mapping_evaluation_mode:
    #     draw_partitioned_graph(G, partitions)
    # clusters = transform_cluster_dict_to_list(partitions)
    # print('clusters networkx')
    # print(clusters)

    # TODO: Add edge pruning with variant threshold!!
    edge_weights = nx.get_edge_attributes(G, 'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w > variant_threshold))


    g = ig.Graph.from_networkx(G)

    partitions = la.find_partition(g, la.ModularityVertexPartition, weights=g.es['weight'], seed=396482)

    partitions_new = []
    # print('partitions')
    # print(partitions)
    for com in partitions:
        partition = []
        # print('community')
        # print(com)
        for index in com:
            partition.append(index_to_node_dict[index])
        # print('partition')
        # print(partition)
        partitions_new.append(partition)
        # print('partitions_new after appending')
        # print(partitions_new)
        # print('len(partition)')
        # print(len(partition))

    # print('partitions_new')
    # print(partitions_new)
    # print('len(partitions_new)')
    # print(len(partitions_new))

    return partitions_new, partitions


    '''
    leiden_coms = algorithms.louvain(G)
    print("leiden_coms: ", leiden_coms)

    partitions = []
    for com in leiden_coms.communities:
        partition = []
        for index in com:
            partition.append(index_to_node_dict[index])
        partitions.append(partition)
    print("partitions: ", partitions)

    return partitions, []
    '''










def draw_partitioned_graph(G, partitions):

    elarge = [(u, v) for (u, v, d) in G.edges(data=True)]
    cmap = cm.get_cmap('viridis', max(partitions.values()) + 1)

    pos = nx.spring_layout(G, weight="weight")  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=10)

    # edges

    nx.draw_networkx_edges(
        G, pos, edgelist=elarge, width=1, alpha=0.5, edge_color="b", style="dashed"
    )
    # labels
    #nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    plt.axis("off")
    plt.show()


    nx.draw_networkx_nodes(G, pos, node_size=10, cmap=cmap, node_color=list(partitions.values()))
    nx.draw_networkx_edges(
        G, pos, edgelist=elarge, width=1, alpha=0.5, edge_color="b", style="dashed"
    )



    plt.axis("off")
    plt.show()





def transform_cluster_dict_to_list(partitions):
    cluster_dict = {}
    for node in partitions.keys():
        cluster_id = partitions[node]
        if cluster_id not in cluster_dict.keys():
            cluster_dict[cluster_id] = [node]
        else:
            cluster_dict[cluster_id].append(node)
    clusters = []
    for v in cluster_dict.values():
        clusters.append(v)
    return clusters

