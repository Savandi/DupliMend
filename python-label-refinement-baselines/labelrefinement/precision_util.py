import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.petri_net.util import performance_map 
from time import time
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.conversion.process_tree import converter as pt_converter  

def get_precision(ref_log, event_log, imprecise_labels, graph_parameters):
    # for case_id, case in enumerate(ref_log):
    #    for event_id, event in enumerate(case):
    #        print(event)
    #        print(event_log[case_id][event_id])
    #        print(".")
    #    print("---------")
    # print("---------------------------------------------------------------------------------")

    # for case in event_log:
    #    for event in case:
    #        print(event)
    #    print("---------")

    # print("get_prec")
    # print("a")
    tree = inductive_miner.apply(ref_log, parameters={
        "activity_key": graph_parameters["ACTIVITY_KEY"]})
    net, initial_marking, final_marking = pt_converter.apply(tree)
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)

    # print("b")
    # RELABELING: Map refined labels back to their original imprecise labels
    for transition in net.transitions:
        if transition.label != None and "_X_" in transition.label:
            base_label = transition.label.split("_X_", 1)[0]
            if base_label in imprecise_labels:
                transition.label = base_label  # Map D_X_1 → D, E_X_1 → E (correct for multiple imprecise labels)
    # print("c")
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)

    # prec = precision_evaluator.apply(event_log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    time1 = time()
    # print("moin1: ")
    # Use ETCONFORMANCE_TOKEN for consistency with PM-label-splitting and DupliMend
    prec = precision_evaluator.apply(event_log, net, initial_marking, final_marking,
                                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)

    time2 = time()
    print("precision time: ", time2 - time1)
    # prec = pm4py.evaluation.precision.versions.align_etconformance.apply(event_log, net, initial_marking, final_marking)
    # print("d")
    return prec, simplicity, generalization


def get_number_of_different_original_labels(event_log):
    original_labels = []
    for trace in event_log:
        for event in trace:
            orig_label = event["OrgLabel"]
            if orig_label not in original_labels:
                original_labels.append(orig_label)

    return len(original_labels)


def get_precision_of_original_model(net, initial_marking, final_marking, event_log, imprecise_labels, original_labels,
                                    graph_parameters):
    # print("get_prec_of_original_model")
    print(original_labels)
    print(imprecise_labels)
    # RELABELING: Map original precise labels to imprecise label for evaluation
    # Note: original_labels should be a list with one element per imprecise label
    for transition in net.transitions:
        print(transition.label)
        if transition.label != None and transition.label in original_labels:
            # If original_labels is a list, find the corresponding imprecise label
            # For single imprecise label case: all original labels map to imprecise_labels[0]
            transition.label = imprecise_labels[0]
            print('after swap')
            print(transition.label)

    generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)
    prec = precision_evaluator.apply(event_log, net, initial_marking, final_marking,
                                     variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    return prec, simplicity, generalization


def get_precision_of_precice_log(original_event_log, event_log, imprecise_labels, original_labels, graph_parameters):
    # print("get_prec_of_precise_log")
    tree = inductive_miner.apply(original_event_log, parameters={
        "activity_key": graph_parameters["ACTIVITY_KEY"]})
    net, initial_marking, final_marking = pt_converter.apply(tree)

    # RELABELING: Map original precise labels to imprecise label for evaluation
    for transition in net.transitions:
        if transition.label != None and transition.label in original_labels:
            transition.label = imprecise_labels[0]  # For single imprecise label case

    prec = precision_evaluator.apply(event_log, net, initial_marking, final_marking,
                                     variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)

    generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
    simplicity = simplicity_evaluator.apply(net)
    return prec,  simplicity, generalization
