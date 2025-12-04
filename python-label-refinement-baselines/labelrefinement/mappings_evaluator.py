import weighted_cost_function
import trace_util as graph_util

def evaluate_mappings(egraphs, mappings, original_labels, labeling_function):
    k = 1
    global_equal_counter = 0
    global_dissimilar_counter = 0
    for original_label in original_labels:
        print("original_label: ", original_label)
        equal_counter = 0
        equal_cost = 0
        dissimilar_counter = 0
        dissimilar_cost = 0

        for e1, egraph_1 in enumerate(egraphs):
            for e2, egraph_2 in enumerate(egraphs):



                mapping, cost = mappings[e1][e2]
                for a, b in mapping:
                    _, w = weighted_cost_function.get_weight(egraphs, e1, a, e2, b, mappings)
                    #if w < variant_threshold:
                    if ((egraph_1.nodeID_to_event_dict[a]["OrgLabel"] == original_label) and (
                                egraph_2.nodeID_to_event_dict[b]["OrgLabel"] != original_label)) \
                                or ((egraph_1.nodeID_to_event_dict[a]["OrgLabel"] != original_label) and (
                                egraph_2.nodeID_to_event_dict[b]["OrgLabel"] == original_label)):

                            dissimilar_counter += 1
                            global_dissimilar_counter += 1
                            dissimilar_cost += w
                            if (egraph_1.nodeID_to_event_dict[a]["OrgLabel"] == "A1") and (egraph_2.nodeID_to_event_dict[b]["OrgLabel"] != "A1"): # 8 < abs(egraph_1.size - egraph_2.size):
                                print(e1)
                                print(e2)
                                #priolist = graph_util.get_nodes_sorted_by_importance_reworked(egraph_1, egraph_2, k,
                                                                                          #labeling_function)
                                #print(priolist)
                                print("0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 ")
                                s1 = ""
                                for e in egraph_1.nodeID_to_event_dict.values():
                                    s1 += labeling_function(e)
                                    s1 += " "
                                print(s1)
                                s2 = ""
                                for e in egraph_2.nodeID_to_event_dict.values():
                                    s2 += labeling_function(e)
                                    s2 += " "
                                print(s2)

                                lm = []
                                for m1, m2 in mapping:
                                    lm.append((labeling_function(egraph_1.nodeID_to_event_dict[m1]),
                                               labeling_function(egraph_2.nodeID_to_event_dict[m2])))
                                print(lm)
                                print(mapping)
                                print("______")
                    elif ((egraph_1.nodeID_to_event_dict[a]["OrgLabel"] == original_label) and (
                                egraph_2.nodeID_to_event_dict[b]["OrgLabel"] == original_label)):
                            equal_counter += 1
                            equal_cost += w
                            global_equal_counter += 1

        print("equal counter: ", equal_counter)
        print("dissimilar_counter: ", dissimilar_counter)
        if dissimilar_counter == 0:
            print("correct ratio: ", 1)
        else:
            print("correct ratio: ", equal_counter/(equal_counter + dissimilar_counter))
        print("---")
        print("equal_cost: ", equal_cost)
        print("dissimilar_cost: ", dissimilar_cost)
        print("----------------------------------------------")
    print("global_equal_counter: ", global_equal_counter)
    print("global_dissimilar_counter: ", global_dissimilar_counter)

def get_original_labels(egraphs, label):
    return
