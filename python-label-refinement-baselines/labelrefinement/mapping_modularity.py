import networkx as nx
import networkx.algorithms.community as nx_comm

def get_mapping_modularity(log, egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID, mappings, labeling_function):
    G = nx.Graph()
    index_to_node_dict = {}
    node_to_index_dict = {}

    max_cost = 0
    if True:
        for egraph1_id, egraph1 in enumerate(egraphs):
            for egraph2_id, egraph2 in enumerate(egraphs):
                mapping, mapping_cost = mappings[egraph1_id][egraph2_id]
                for (node1_id, node2_id) in mapping:
                    if max_cost < mapping_cost:
                        max_cost = mapping_cost

                    for real_egraphs1_id in map_egraph_ID_to_trace_IDs[egraph1_id]:
                        for real_egraphs2_id in map_egraph_ID_to_trace_IDs[egraph2_id]:
                            G.add_edge((real_egraphs1_id, node1_id), (real_egraphs2_id, node2_id), mapping_cost=mapping_cost)

    for u, v, d in G.edges(data=True):
        if max_cost > 0:
            G[u][v]["weight"] = 1 - G[u][v]["mapping_cost"] / max_cost
        else:
            G[u][v]["weight"] = 0

        #print(G[u][v]["weight"])



    partition = {}

    if True:
        for case_id, case in enumerate(log):
            for event_id, event in enumerate(case):
                if event["OrgLabel"] not in partition.keys():
                    partition[event["OrgLabel"]] = [(case_id, event_id)]
                else:
                    partition[event["OrgLabel"]].append((case_id, event_id))



    #print("before: ", partition)
    partition = partition.values()
    #print("before: ", partition)
    modularity1 = nx_comm.modularity(G, partition, weight="weight")




    '''
    G = nx.Graph()
    index_to_node_dict = {}
    node_to_index_dict = {}

    max_cost = 0
    if True:
        for egraph1_id, egraph1 in enumerate(egraphs):
            for egraph2_id, egraph2 in enumerate(egraphs):
                mapping, mapping_cost = mappings[egraph1_id][egraph2_id]
                for (node1_id, node2_id) in mapping:
                    if max_cost < mapping_cost:
                        max_cost = mapping_cost

                    G.add_edge((egraph1_id, node1_id), (egraph2_id, node2_id),
                                       mapping_cost=mapping_cost)


    for u, v, d in G.edges(data=True):
        if G[u][v]["mapping_cost"] < 0:
            print("negative!!!!: ", G[u][v]["mapping_cost"])
        if max_cost > 0:
            G[u][v]["weight"] = 1 - G[u][v]["mapping_cost"] / max_cost
        else:
            G[u][v]["weight"] = 0

        #print(G[u][v]["weight"])



    partition = {}

    if True:
        for egraph_id, egraph in enumerate(egraphs):
            for event_id, event in enumerate(egraph.events):
                if event["OrgLabel"] not in partition.keys():
                    partition[event["OrgLabel"]] = [(egraph_id, event_id)]
                else:
                    partition[event["OrgLabel"]].append((egraph_id, event_id))


    #print("before: ", partition)
    partition = partition.values()
    #print("before: ", partition)
    modularity2 = nx_comm.modularity(G, partition, weight="weight")
    '''

    return modularity1#, modularity2