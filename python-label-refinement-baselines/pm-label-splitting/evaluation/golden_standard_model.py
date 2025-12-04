import re
import os

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.process_tree import converter as pt_converter

from evaluation.performance_evaluator import PerformanceEvaluator
from utils.file_writer_helper import export_models_and_pngs
from pipeline.pipeline_variant import remove_pipeline_variant_from_string

# Import OUTPUT_BASE_DIR from path_config (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR


class GoldenStandardModel:
    """
    Representation of the golden standard, i.e., the event log with precise labels, before label modification
    """

    def __init__(self, input_name: str, original_input_name: str, path: str, labels_to_split):
        self._input_name = input_name
        self._path = path
        self._labels_to_split = labels_to_split
        self._input_identifier = original_input_name if original_input_name != '' \
            else get_input_identifier_from_variant_input_name(self._input_name)
        self._imprecise_log = xes_importer.apply(self._path)
        self.net = None
        self.im = None
        self.fm = None

    def evaluate_golden_standard_model(self):
        """
        Gets the path to the precise event log, generates the golden standard model and evaluates it

        :return: Precision of the golden standard model
        """
        # Ensure output directory exists before writing
        output_file_path = f'{OUTPUT_BASE_DIR}/{self._input_name}.txt'
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'a') as outfile:
            outfile.write('\n Performance of golden standard model:\n')
            log = get_log_from_input_identifier(self._input_identifier, path=self._path)

            imprecise_labels = attributes_filter.get_attribute_values(self._imprecise_log, 'concept:name')

            process_tree = inductive_miner.apply(log)
            net, initial_marking, final_marking = pt_converter.apply(process_tree)

            rename_transitions_to_original_label(imprecise_labels, net, self._labels_to_split)

            self.net = net
            self.im = initial_marking
            self.fm = final_marking
            performance_evaluator = PerformanceEvaluator(net, initial_marking, final_marking, self._imprecise_log,
                                                         outfile, skip_fitness=True)
            performance_evaluator.evaluate_performance()

            original_tree = inductive_miner.apply(log)
            export_models_and_pngs(final_marking, initial_marking, net, original_tree, self._input_name,
                                   'no_noise_golden')

        return performance_evaluator.precision


def rename_transitions_to_original_label(imprecise_labels, net, labels_to_split):
    for transition in net.transitions:
        if transition.label is None:
            continue

        if transition.label not in imprecise_labels:
            transition.label = labels_to_split[0]


def get_input_identifier_from_variant_input_name(input_name: str):
    original_input_name = remove_pipeline_variant_from_string(input_name)
    original_input_name = re.sub(r'.*/', '', original_input_name)
    return original_input_name


def get_log_from_input_identifier(input_identifier, path):
    pattern = input_identifier + r'.*'
    log_path = re.sub(pattern, f'{input_identifier}_Log.xes.gz', path)
    log = xes_importer.apply(log_path)
    return log
