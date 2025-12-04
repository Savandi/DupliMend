from statistics import mean

def get_vertical_refinement(egraphs, map_egraph_ID_to_trace_IDs, map_trace_ID_to_egraph_ID, label_refinement, label,
                            labeling_function, unfolding_threshold, speedup_mode):
    for egraph_ID in range(0, len(egraphs)):
        #one egraph is a variant
        node_IDs_to_consider = []
        component_IDs_in_this_egraph = []
        for node_ID in range(0, egraphs[egraph_ID].size):
            if labeling_function(egraphs[egraph_ID].nodeID_to_event_dict[node_ID]) == label:
                for component_ID in range(0, len(label_refinement)):
                    if (egraph_ID, node_ID) in label_refinement[component_ID] and component_ID not in component_IDs_in_this_egraph:
                        component_IDs_in_this_egraph.append(component_ID)
                        #if speedup_mode:
                        #    component_size = sum([len(map_egraph_ID_to_trace_IDs[egraph_nr]) for (egraph_nr, node_nr) in  label_refinement[component_ID]])
                        #else:
                        component_size = len(label_refinement[component_ID])
                        component_position = mean([elem[1] for elem in label_refinement[component_ID]])
                        node_IDs_to_consider.append({"node_ID": node_ID, "component_ID": component_ID,
                                                     "component_size": component_size, "component_position": component_position})

        if len(node_IDs_to_consider) > 1:# != []:
            max_size = max([element["component_size"] for element in node_IDs_to_consider])

            node_IDs_to_consider.sort(key=lambda x: x["component_position"], reverse=False)

            vertical_refinement = [[0]]
            for index in range(1, len(node_IDs_to_consider)):
                curr_component_size = node_IDs_to_consider[index]["component_size"]
                if curr_component_size >= unfolding_threshold * max_size and node_IDs_to_consider[index]["component_ID"] != node_IDs_to_consider[index-1]["component_ID"]:
                    vertical_refinement.append([index])
                elif node_IDs_to_consider[index]["component_ID"] != node_IDs_to_consider[index-1]["component_ID"]:
                    vertical_refinement[-1].append(index)

            new_label_refinement = []
            components_in_variant = []
            for indexes in vertical_refinement:
                new_component = set()
                for index in indexes:
                    components_in_variant.append(node_IDs_to_consider[index]["component_ID"])
                    new_component = new_component.union(set(label_refinement[node_IDs_to_consider[index]["component_ID"]]))
                new_label_refinement.append(list(new_component))
            components_not_in_variant = set(range(0, len(label_refinement))).difference(set(components_in_variant))
            for comp in components_not_in_variant: #add components wic are not in te variant
                new_label_refinement.append(label_refinement[comp])
            label_refinement = new_label_refinement
    return label_refinement