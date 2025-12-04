import egraph_class
import egraph_label_refinement
from time import time


# a hashing is used between the traces and the egraphs we are using for mapping calculation and refinements.
def get_egraphs(parameters, use_trace_folding, event_log):

    time0 = time()
    egraphs = []
    seen_traces = []
    map_egraph_ID_to_trace_IDs = {}
    map_trace_ID_to_egraph_ID = {}
    new_id = 0
    for id, trace in enumerate(event_log):
        labels_of_trace_list = [egraph_label_refinement.default_labeling_function(event) for event in trace]
        labels_of_trace_string = ""
        for label in labels_of_trace_list:
            labels_of_trace_string += label

        if labels_of_trace_string not in seen_traces:
            seen_traces.append(labels_of_trace_string)
            egraph = egraph_class.egraph(trace, new_id, parameters, use_trace_folding)
            egraphs.append(egraph)
            map_egraph_ID_to_trace_IDs[len(egraphs) - 1] = [id]
            map_trace_ID_to_egraph_ID[id] = len(egraphs) - 1
            new_id += 1

        else:
            egraph_index = seen_traces.index(labels_of_trace_string)  #todo very dangerous :O
            map_egraph_ID_to_trace_IDs[egraph_index].append(id)
            map_trace_ID_to_egraph_ID[id] = egraph_index


    time1 = time()

    #print("overhead: ", time1 - time0)
    print("actual size :", len(egraphs))
    return egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID