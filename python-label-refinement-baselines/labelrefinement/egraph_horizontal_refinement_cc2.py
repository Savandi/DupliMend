import networkx as nx
import matplotlib.pyplot as plt
import weighted_cost_function
import igraph as ig



def get_connected_components(egraphs, mappings, label, labeling_function, variant_threshold, use_local_cost, mapping_evaluation_mode, use_mapping_and_label_neighbors):
        G = nx.Graph()

        max_local_cost = 0
        max_gloabal_cost = 0
        print('len(egraphs)')
        print(len(egraphs))
        for egraph_index1 in (range(0, len(egraphs))):
            for egraph_index2 in (range(0, len(egraphs))):
                mapping, cost = mappings[egraph_index1][egraph_index2]
                for m1, m2 in mapping:
                    if labeling_function(egraphs[egraph_index1].nodeID_to_event_dict[m1]) == label:
                        local_cost, global_cost = weighted_cost_function.get_weight(egraphs, egraph_index1, m1,
                                                                                    egraph_index2, m2, mappings, use_mapping_and_label_neighbors, use_local_cost)
                        if max_local_cost < local_cost:
                            max_local_cost = local_cost
                        if max_gloabal_cost < global_cost:
                            max_gloabal_cost = global_cost
                        G.add_edge((egraph_index1, m1), (egraph_index2, m2), local_cost=local_cost, global_cost=global_cost)

        for u, v, d in G.edges(data=True):
            if max_local_cost != 0:
                G[u][v]["local_cost"] = G[u][v]["local_cost"] / max_local_cost
            else:
                G[u][v]["local_cost"] = 0
            if max_gloabal_cost != 0:
                G[u][v]["global_cost"] = G[u][v]["global_cost"] / max_gloabal_cost
            else:
                G[u][v]["global_cost"] = 0
            if use_local_cost:
                G[u][v]["weight"] = (0.5 * G[u][v]["local_cost"] + 0.5 * G[u][v]["global_cost"])**2 #todo: change hardcoding
            else:
                G[u][v]["weight"] = G[u][v]["global_cost"]
        edge_weights = nx.get_edge_attributes(G, 'weight')
        G.remove_edges_from((e for e, w in edge_weights.items() if w > variant_threshold))
        clusters = nx.connected_components(G)

        c = []
        for cl in clusters:
            c.append(list(cl))

        g = ig.Graph.from_networkx(G)
        clustering = g.clusters()
        print('CC clustering')
        print(clustering)


        #print("number of new labels: ", len(c))

        # elarge = [(u, v) for (u, v) in G.edges(data=True)]
        if mapping_evaluation_mode:
            elarge = G.edges


            pos = nx.spring_layout(G, weight="weight")  # positions for all nodes

            # nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)

            # edges
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
            nx.draw_networkx_edges(
                G, pos, edgelist=elarge, width=6, alpha=0.5, edge_color="b", style="dashed"
            )

            # labels
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

            plt.axis("off")
            plt.show()

        return c, clustering
        '''
        mappings = transform_mappings_triangle_to_qmatrix(mappings)
        G = nx.Graph()
    
        for egraph_index1 in (range(0, len(egraphs))):
            for egraph_index2 in (range(0, len(egraphs))):
                mapping, cost = mappings[egraph_index1][egraph_index2]
                for m1, m2 in mapping:
                    if labeling_function(egraphs[egraph_index1].nodeID_to_event_dict[m1]) == label:
                        #w = get_weight(egraphs, egraph_index1, m1, egraph_index2, m2, mappings)
                        G.add_node((egraph_index2, m2))
                        G.add_node((egraph_index1, m1))
                        #if w <= variant_threshold:
                        G.add_edge((egraph_index1, m1), (egraph_index2, m2))
    
        clusters = nx.connected_components(G)
        c = []
        for cl in clusters:
            c.append(list(cl))
    
    
    
        #elarge = [(u, v) for (u, v) in G.edges(data=True)]
        elarge = G.edges
    
        pos = nx.spring_layout(G)  # positions for all nodes
    
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)
    
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, pos, edgelist=elarge, width=6, alpha=0.5, edge_color="b", style="dashed"
        )
    
        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    
        plt.axis("off")
        plt.show()
    
        return c, []
        '''

