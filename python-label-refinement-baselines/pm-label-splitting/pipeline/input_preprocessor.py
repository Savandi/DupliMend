import re
import os

from igraph import *
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer as xes_importer

from evaluation.apply_im import apply_im_with_noise_and_export, \
    apply_im_without_noise_and_export, get_xixi_metrics
from evaluation.golden_standard_model import GoldenStandardModel
from utils.input_data import InputData
from pipeline.pipeline_helpers import get_imprecise_labels, get_community_similarity
from pipeline.pipeline_variant import remove_pipeline_variant_from_string, PipelineVariant

# Import OUTPUT_BASE_DIR from path_config (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR


class InputPreprocessor:
    def __init__(self, input_data: InputData):
        self.input_data: InputData = input_data

    def preprocess_input(self) -> None:
        # Ensure output directory exists before writing
        output_file_path = f'{OUTPUT_BASE_DIR}/{self.input_data.input_name}.txt'
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, 'a') as outfile:
            original_log = self.input_data.original_log

            xixi_precision = 0
            xixi_ari = 0
            ground_truth_precision = 0
            labels_to_split = self.input_data.labels_to_split
            ground_truth_model = None
            xixi_clustering = None
            original_labels = None
            ground_truth_clustering = None

            if not self.input_data.labels_to_split:
                # Try to auto-detect imprecise labels
                labels_to_split = get_imprecise_labels(original_log)
                
                # If no labels found, try comparison method
                if not labels_to_split:
                    from pipeline.pipeline_helpers import get_imprecise_labels_from_comparison, get_corresponding_original_log_path
                    
                    # Get the correct original log path
                    original_log_path = get_corresponding_original_log_path(self.input_data.log_path)
                    
                    if original_log_path and os.path.exists(original_log_path):
                        labels_to_split = get_imprecise_labels_from_comparison(self.input_data.log_path, original_log_path)
                        outfile.write(f'\nFound imprecise labels by comparison: {labels_to_split}\n')
                        outfile.write(f'Compared {self.input_data.log_path} with {original_log_path}\n')
                    else:
                        outfile.write(f'\nOriginal log not found. Expected at: {original_log_path}\n')
                        outfile.write('Please specify labels_to_split manually\n')
                        return
                
                # Rest of the preprocessing logic remains the same
                ground_truth_model = GoldenStandardModel(self.input_data.input_name, '', self.input_data.log_path,
                                                        labels_to_split)
                ground_truth_precision = ground_truth_model.evaluate_golden_standard_model()

                xixi_precision, xixi_clustering = get_xixi_metrics(labels_to_split, self.input_data)

                export_model_from_original_log_with_precise_labels(self.input_data.input_name, self.input_data.log_path,
                                                                self.input_data.use_noise)

                original_labels = self.get_original_labels(labels_to_split)
                outfile.write('\n Original Labels:\n')
                outfile.write(f'{str(original_labels)}\n')

                ground_truth_clustering = self.get_ground_truth_clustering(original_labels, labels_to_split)
                outfile.write('\n Ground truth clustering clustering:\n')
                outfile.write(f'{str(ground_truth_clustering)}\n')

                # Try to compare clusterings - handle case where vectors have different lengths
                try:
                    xixi_ari = get_community_similarity(ground_truth_clustering, xixi_clustering)
                    outfile.write('\n Xixi Adjusted Rand Index:\n')
                    outfile.write(f'{xixi_ari}\n')
                except ValueError as e:
                    # Clustering vectors have different lengths - comparison not possible
                    outfile.write(f'\n⚠️  Xixi ARI comparison skipped: {e}\n')
                    outfile.write(f'   Ground truth clustering length: {len(ground_truth_clustering.membership) if hasattr(ground_truth_clustering, "membership") else "unknown"}\n')
                    outfile.write(f'   Xixi clustering length: {len(xixi_clustering.membership) if hasattr(xixi_clustering, "membership") else "unknown"}\n')
                    xixi_ari = 0.0  # Use 0 as sentinel value when comparison not possible

            # Set the processed data
            self.input_data.original_labels = original_labels
            self.input_data.xixi_precision = xixi_precision
            self.input_data.ground_truth_precision = ground_truth_precision
            self.input_data.ground_truth_model = ground_truth_model
            self.input_data.labels_to_split = labels_to_split
            self.input_data.original_labels = original_labels
            self.input_data.ground_truth_clustering = ground_truth_clustering
            self.input_data.xixi_clustering = xixi_clustering
            self.input_data.xixi_ari = xixi_ari

    def get_original_labels(self, labels_to_split):
        original_labels = set()
        log = self.input_data.original_log

        variants = variants_filter.get_variants(log)

        for variant in variants:
            filtered_log = variants_filter.apply(log, [variant])
            for event in filtered_log[0]:
                if event['concept:name'] in labels_to_split:
                    original_labels.add(event['OrgLabel'])
        return list(original_labels)

    def get_ground_truth_clustering(self, original_labels, labels_to_split):
        log = self.input_data.original_log
        ground_truth_clustering = []
        variants = variants_filter.get_variants(log)

        if self.input_data.pipeline_variant != PipelineVariant.EVENTS:
            for variant in variants:
                filtered_log = variants_filter.apply(log, [variant])
                for event in filtered_log[0]:
                    if event['concept:name'] in labels_to_split:
                        ground_truth_clustering.append(original_labels.index(event['OrgLabel']))
        else:
            for trace in log:
                for event in trace:
                    if event['concept:name'] in labels_to_split:
                        ground_truth_clustering.append(original_labels.index(event['OrgLabel']))
        return Clustering(ground_truth_clustering)

    def has_duplicate_xor(self):
        if not self.input_data.ground_truth_model:
            return False
        for p in self.input_data.ground_truth_model.net.places:
            seen_labels = []
            for a in p.out_arcs:
                t = a.target
                if t.label != None and t.label in self.input_data.labels_to_split and t.label in seen_labels:
                    return True
                seen_labels.append(t.label)
        return False


def export_model_from_original_log_with_precise_labels(input_name, path, use_noise=True):
    with open(f'{OUTPUT_BASE_DIR}/{input_name}.txt', 'a') as outfile:
        outfile.write('\n Data from log without imprecise labels\n')
        original_input_name = remove_pipeline_variant_from_string(input_name)
        original_input_name = re.sub(r'.*/', '', original_input_name)

        pattern = original_input_name + r'.*'
        log_path = re.sub(pattern, f'{original_input_name}_Log.xes.gz', path)
        original_log = xes_importer.apply(log_path)
        if use_noise:
            apply_im_with_noise_and_export(input_name, 'original_log_precise_labels', original_log, original_log,
                                           outfile, labels_to_original={})
        apply_im_without_noise_and_export(input_name, 'original_log_precise_labels', original_log, original_log,
                                          outfile, labels_to_original={})
