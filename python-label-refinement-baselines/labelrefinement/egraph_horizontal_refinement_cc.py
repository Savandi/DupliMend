import egraph_mapping_dynamic as egraph_mapping
import egraph_vertical_refinement
import egraph_class
import copy
from time import time
import egraph_horizontal_refinement_cc as egraph_horizontal_refinement

def get_connected_components(egraphs, mappings, label, labeling_function, variant_threshold):

    gcc1 = time()
    connected_components = []
    for egraphs_index in range(0, len(egraphs)):   #todo: Going only once over traces occuring multiple times
        for nodeID in range(0, egraphs[egraphs_index].size):
            if labeling_function(egraphs[egraphs_index].nodeID_to_event_dict[nodeID]) == label:
                #print("connected_components: ", connected_components)
                #print("ei, ni: ", egraphs_index, ", ", nodeID)
                if connected_components == []:
                    connected_components = [[(egraphs_index, nodeID)]]
                else:
                    distances_to_components = []
                    for component in connected_components:
                        distances_to_components.append(get_minimal_distance_to_component(egraphs_index, nodeID, mappings, component))
                    added = False
                    added_component_nr = "not added"
                    components_to_be_merged = []
                    for component_nr in range(0, len(distances_to_components)):
                        if not added and distances_to_components[component_nr] != "nm" and distances_to_components[component_nr] <= variant_threshold:
                            connected_components[component_nr].append((egraphs_index, nodeID))
                            added = True
                            added_component_nr = component_nr
                        elif added and distances_to_components[component_nr] != "nm" and distances_to_components[component_nr] <= variant_threshold: #If event is allready added to other component, we have to merge the components
                            components_to_be_merged.append(component_nr)
                    if not added:
                        connected_components.append([(egraphs_index, nodeID)])
                    for com in components_to_be_merged:
                        connected_components[added_component_nr].extend(connected_components[com])
                    #for com in components_to_be_merged:
                    #    connected_components.pop(com)
                    connected_components = [connected_components[index] for index in range(0, len(connected_components)) if index not in components_to_be_merged]
    gcc2 = time()
    return connected_components, []


def get_minimal_distance_to_component(egraphs_index, nodeID, mappings, component):
    minimal_distance = "nm"
    for component_log_index, component_nodeID in component:
        allo = mappings[egraphs_index][component_log_index]
        #print(component_log_index)
        #print(egraphs_index)
        #print("asdasda ", allo)
        mapping, mapping_cost = allo
        for nodeID_source, nodeID_target in mapping:
            if nodeID == nodeID_source and component_nodeID == nodeID_target:
                if minimal_distance == "nm" or mapping_cost < minimal_distance:
                    minimal_distance = mapping_cost
                break #stop walking through mapping if mapped position is found
    return minimal_distance