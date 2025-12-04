from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter

def postprocess(event_log, imprecise_labels, label_refinements, parameters):
    # Apply inductive miner to get process tree, then convert to Petri net
    tree = inductive_miner.apply(event_log, parameters={
        "activity_key": parameters["ACTIVITY_KEY"]})
    net, initial_marking, final_marking = pt_converter.apply(tree)

    seen_transitions = []
    new_label_refinements = []
    for label, label_refinement in label_refinements:
        new_label_refinement = []
        for transition_1 in net.transitions:

            if transition_1.label != None and transition_1.label.split("_X_", 1)[0] == label and\
                    len(transition_1.label.split("_X_", 1)) > 1 and transition_1.name not in seen_transitions:
                pre_places_1 = set()
                post_places_1 = set()

                for arc in transition_1.in_arcs:
                    pre_places_1.add(arc.source.name)
                for arc in transition_1.out_arcs:
                    post_places_1.add(arc.target.name)

                ref_nr_1 = int(transition_1.label.split("_X_", 1)[1])
                seen_transitions.append(transition_1.name)
                new_label_refinement.append(label_refinement[ref_nr_1-1])
                for transition_2 in net.transitions:

                    if transition_2.name not in seen_transitions and transition_2.label != None and \
                            len(transition_2.label.split("_X_", 1)) > 1 and label == transition_2.label.split("_X_", 1)[0]:
                        ref_nr_2 = int(transition_2.label.split("_X_", 1)[1])       # ERROR HERE TODO

                        pre_places_2 = set()
                        post_places_2 = set()
                        for arc in transition_2.in_arcs:
                            pre_places_2.add(arc.source.name)
                        for arc in transition_2.out_arcs:
                            post_places_2.add(arc.target.name)


                        if pre_places_1 == pre_places_2 and post_places_1 == post_places_2:
                            seen_transitions.append(transition_2.name)
                            if len(new_label_refinement) > 0:
                                new_label_refinement[-1].extend(label_refinement[ref_nr_2-1])
                            else:
                                new_label_refinement = [label_refinement[ref_nr_2-1]]
        new_label_refinements.append((label, new_label_refinement))

    return new_label_refinements