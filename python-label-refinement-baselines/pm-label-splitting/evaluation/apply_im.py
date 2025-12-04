import os
from typing import TextIO, Dict, List

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.process_tree import converter as pt_converter
from evaluation.performance_evaluator import PerformanceEvaluator
from pipeline.pipeline_helpers import get_clustering_from_xixi_log, filter_duplicate_xor
from pipeline.post_processor import PostProcessor
from utils.file_writer_helper import write_exception, export_models_and_pngs
from pm4py.objects.log.importer.xes import importer as xes_importer

from utils.input_data import InputData

# Import OUTPUT_BASE_DIR from path_config (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR


def apply_im_without_noise_and_export(input_name: str, suffix: str, split_log: EventLog, original_log: EventLog,
                                      outfile: TextIO, labels_to_original: Dict[str, str],
                                      short_labels_to_original_labels: Dict[str, str] = None):
    """
    Applies the Inductive Miner without noise threshold and exports the result.
    """
    outfile.write(f'\n IM without noise threshold:\n')

    final_marking, initial_marking, final_net, precision, simplicity, generalization, _ = apply_im_without_noise_and_evaluate(
        labels_to_original,
        split_log,
        original_log,
        outfile,
        short_labels_to_original_labels)
    
    # Fix: Use apply instead of apply_tree
    tree = inductive_miner.apply(split_log)

    export_models_and_pngs(final_marking, initial_marking, final_net, tree, input_name, suffix)
    return precision, final_net, initial_marking, final_marking

def apply_im_without_noise_and_evaluate(labels_to_original: Dict[str, str], split_log: EventLog, original_log: EventLog,
                                        outfile: TextIO, short_labels_to_original_labels: Dict[str, str] = None):
    """
    Applies the Inductive Miner without noise threshold and evaluates the result.
    """
    process_tree = inductive_miner.apply(split_log)
    net, initial_marking, final_marking = pt_converter.apply(process_tree)
    post_processor = PostProcessor(labels_to_original, short_labels_to_original_labels)
    final_net = post_processor.post_process_petri_net(net)

    performance_evaluator = PerformanceEvaluator(final_net,
                                                 initial_marking,
                                                 final_marking,
                                                 original_log,
                                                 outfile,
                                                 skip_fitness=True)
    performance_evaluator.evaluate_performance()

    final_net = post_processor.rename_short_labels_to_original_labels(final_net)
    return final_marking, initial_marking, final_net, performance_evaluator.precision, \
           performance_evaluator.simplicity, performance_evaluator.generalization, performance_evaluator.fitness


def apply_im_with_noise_and_export(input_name: str, suffix: str, split_log: EventLog, original_log: EventLog,
                                   outfile: TextIO, labels_to_original: Dict[str, str],
                                   short_labels_to_original_labels: Dict[str, str] = None):
    """
    Applies the Inductive miner with multiple noise thresholds to the input.
    The mined models are evaluated and results and models exported.
    """
    f1_scores = []
    for noise_threshold in [0, 0.1, 0.2, 0.3, 0.4]:
        outfile.write(f'\nnoise_threshold: {noise_threshold}\n')

        try:
            process_tree = inductive_miner.apply(split_log,
                                               variant=inductive_miner.Variants.IMf,
                                               parameters={
                                                   inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: noise_threshold})
            net, initial_marking, final_marking = pt_converter.apply(process_tree)
        except Exception as e:
            print('Exception occurred while applying IM with noise')
            write_exception(e, outfile)
            continue

        post_processor = PostProcessor(labels_to_original, short_labels_to_original_labels)
        final_net = post_processor.post_process_petri_net(net)

        performance_evaluator = PerformanceEvaluator(final_net, initial_marking, final_marking, original_log,
                                                     outfile)

        performance_evaluator.evaluate_performance()

        f1_score = get_f1_score(performance_evaluator.precision, performance_evaluator.fitness)
        f1_scores.append(f1_score)
        tree = inductive_miner.apply(original_log,
                                        variant=inductive_miner.Variants.IMf,
                                        parameters={
                                            inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: noise_threshold})
        export_models_and_pngs(final_marking, initial_marking, final_net, tree, input_name, suffix)
    return f1_scores


def get_f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall)


def write_data_from_original_log_with_imprecise_labels(input_name: str, original_log: EventLog,
                                                       use_noise: bool = True) -> List[float]:
    print(f'{OUTPUT_BASE_DIR}/{input_name}.txt')
    with open(f'{OUTPUT_BASE_DIR}/{input_name}.txt', 'a') as outfile:
        outfile.write('\nOriginal Data Performance:\n')
        f1_scores = apply_im_with_noise_and_export(input_name, 'original_log_imprecise_labels', original_log,
                                                   original_log,
                                                   outfile) if use_noise else []
        apply_im_without_noise_and_export(input_name, 'original_log_imprecise_labels', original_log, original_log,
                                          outfile)
    return f1_scores


def get_xixi_metrics(labels_to_split, input_data: InputData):
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        original_log = xes_importer.apply(input_data.log_path)
        xixi_refined_log_path = input_data.log_path.replace('LogD', 'LogR', 1)
        if not os.path.isfile(xixi_refined_log_path):
            xixi_refined_log_path = xixi_refined_log_path.replace('LogR', 'LogR_IM', 1)

        log = xes_importer.apply(xixi_refined_log_path)

        clustering = get_clustering_from_xixi_log(log, labels_to_split, outfile, input_data)
        clustering = filter_duplicate_xor(log, labels_to_split, clustering)

        labels_to_original = {}

        for label in labels_to_split:
            labels_to_original[label] = label

        outfile.write('\n Xixi refined log results:\n')
        precision, final_net, initial_marking, final_marking = apply_im_without_noise_and_export(input_data.input_name, 'xixi',
                                                                                                 log, original_log,
                                                                                                 outfile,
                                                                                                 labels_to_original=labels_to_original)

        outfile.write('\n Xixi clustering:\n')
        outfile.write(f'{str(clustering)}\n')
    return precision, clustering

