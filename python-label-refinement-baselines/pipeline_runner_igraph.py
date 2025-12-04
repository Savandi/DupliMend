import copy
import csv
import os
from pathlib import Path

from pm4py.objects.log.importer.xes import importer as xes_importer

import clustering_variant
from apply_im import apply_im_with_noise_and_export, apply_im_without_noise
from distance_metrics import DistanceVariant
from file_writer_helper import get_config_string, write_summary_file, \
    write_summary_file_with_parameters, run_start_string
from goldenstandardmodel import export_models_and_pngs
from input_data import InputData
from input_preprocessor import InputPreprocessor

# CONFIGURATION - Load from DupliMend config
try:
    import sys
    sys.path.append('../config')
    from config import evaluation_config
    
    # Use DupliMend's PM label splitting config (since this runs PM splitting)
    pm_config = evaluation_config.get("baseline_evaluation_config", {}).get("pm_label_splitting", {})
    
    # Get data paths
    DATA_PATH_SYNTHETIC = pm_config.get("data_path_synthetic", "./data/noImprInLoop_default_OD")
    DATA_PATH_REAL = pm_config.get("data_path_real", "C:\\Users\\drana\\Documents\\GitHub\\DupliMend\\data\\BPI_Challenge_2013_closed_problems.xes")
    
    # Get output paths  
    OUTPUT_BASE_DIR = pm_config.get("output_dir", r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\outputs")
    RESULTS_BASE_DIR = pm_config.get("results_dir", r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\results")
    BEST_RESULTS_DIR = pm_config.get("best_results_dir", r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\best_results")
    
    print(f"✅ Using DupliMend PM Label Splitting config")
    print(f"   Synthetic data: {DATA_PATH_SYNTHETIC}")
    print(f"   Real data: {DATA_PATH_REAL}")
    print(f"   Results will be saved to: {RESULTS_BASE_DIR}")
    
except ImportError:
    # Fallback if config import fails
    print("⚠️  Could not import DupliMend config, using fallback paths")
    DATA_PATH_SYNTHETIC = "./data/noImprInLoop_default_OD"
    DATA_PATH_REAL = "C:\\Users\\drana\\Documents\\GitHub\\DupliMend\\data\\BPI_Challenge_2013_closed_problems.xes"
    OUTPUT_BASE_DIR = r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\outputs"
    RESULTS_BASE_DIR = r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\results"
    BEST_RESULTS_DIR = r"C:\Users\drana\Documents\GitHub\DupliMend\src\evaluation_results\baselines\pm_label_splitting\best_results"
from label_splitter_event_based_igraph import LabelSplitter as LabelSplitterEventBased
from label_splitter_variant_based_igraph import LabelSplitter as LabelSplitterVariantBased
from label_splitter_variant_multiplex import LabelSplitter as LabelSplitterVariantMultiplex
from pipeline_helpers_shared import get_tuples_for_folder, get_community_similarity, filter_duplicate_xor
from pipeline_variant import PipelineVariant
from plot_helpers import plot_noise_to_f1_score
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner



def run_pipeline_multi_layer_igraph(input_paths) -> None:

    if not os.path.exists(BEST_RESULTS_DIR):
        os.makedirs(BEST_RESULTS_DIR)

    apply_pipeline_to_folder([('real_logs/road_traffic_fines',
                               './real_logs/Road_Traffic_Fine_Management_Process_shortened_labels.xes.gz')],
                             'real_logs.txt',
                             PipelineVariant.VARIANTS,
                             labels_to_split=['F'],
                             use_frequency=True,
                             use_noise=False)

    for path, prefix in input_paths:
        input_list = get_tuples_for_folder(path, prefix)
        apply_pipeline_to_folder(input_list, prefix, PipelineVariant.VARIANTS, labels_to_split=[], use_noise=False,
                                 use_frequency=True)

    for path, prefix in input_paths:
        input_list = get_tuples_for_folder(path, prefix)
        apply_pipeline_to_folder(input_list, prefix, PipelineVariant.VARIANTS, labels_to_split=[], use_noise=False,
                                 use_frequency=False)

    for path, prefix in input_paths:
        input_list = get_tuples_for_folder(path, prefix)
        apply_pipeline_to_folder(input_list, prefix, PipelineVariant.EVENTS, labels_to_split=[], use_noise=False,
                                 use_frequency=True)


def apply_pipeline_to_folder(input_list, folder_name, pipeline_variant, labels_to_split=[], use_frequency=True,
                             use_noise=True):
    header = [
        'Name', 'max_number_of_traces', 'labels_to_split', 'original labels', 'original_precision', 'original_simplicity',
        'original_generalization', 'original_fitness' ,'Xixi number of Clusters found', 'Xixi Precision', 'Xixi ARI',
        'use_combined_context', 'use_frequency', 'window_size', 'distance_metric', 'threshold', 'Number of Clusters found',
        'Precision Align', 'ARI', 'Simplicity', 'Generalization', 'Fitness']

    Path(f'{OUTPUT_BASE_DIR}/{folder_name}').mkdir(parents=True, exist_ok=True)


    csv_file_path = Path(f'{RESULTS_BASE_DIR}/{folder_name}_{pipeline_variant}_NEW.csv')
    if csv_file_path.is_file():
        print(csv_file_path)
        print('Warning: File already existis exiting')
        return

    with open(f'{RESULTS_BASE_DIR}/{folder_name}_{pipeline_variant}_NEW.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print("Starting pipeline")
    for (name, path) in input_list:
        input_data = InputData(original_input_name=name,
                               log_path=path,
                               pipeline_variant=pipeline_variant,
                               labels_to_split=labels_to_split,
                               use_frequency=use_frequency,
                               use_noise=use_noise,
                               max_number_of_traces=500,
                               folder_name=folder_name)

        input_data.original_log = xes_importer.apply(input_data.log_path, parameters={
            xes_importer.Variants.ITERPARSE.value.Parameters.MAX_TRACES: input_data.max_number_of_traces})
        input_data.input_name = f'{input_data.original_input_name}_{input_data.pipeline_variant}' if use_frequency else f'{input_data.original_input_name}_{input_data.pipeline_variant}'

        input_data.use_combined_context = False

        input_preprocessor = InputPreprocessor(input_data)
        input_preprocessor.preprocess_input()
        summary_file_name = f'{folder_name}_{pipeline_variant}.txt' if use_frequency else f'{folder_name}_{pipeline_variant}.txt'
        input_data.summary_file_name = summary_file_name

        if input_preprocessor.has_duplicate_xor():
            print('############## Skipped ######################')
            print('Duplicate XOR found, skipping this model')
            with open(f'{OUTPUT_BASE_DIR}/best_results/{summary_file_name}', 'a') as outfile:
                outfile.write(
                    f'´\n----------------Skipped Model {input_data.input_name} because of duplicate label ------------------------\n')
            continue

        concurrent_labels = []

        input_data.concurrent_labels = concurrent_labels
        print('concurrent_labels')
        print(concurrent_labels)

        best_score, best_precision, best_configs = apply_pipeline_multi_layer_igraph_to_log_with_multiple_parameters(
            input_data)
        try:
            write_summary_file_with_parameters(best_configs, best_score, best_precision, name, summary_file_name)
            write_summary_file(best_score, best_precision, input_data.ground_truth_precision, name, summary_file_name,
                               input_data.xixi_precision, input_data.xixi_ari)
        except Exception as e:
            print('----------------Exception occurred while writing summary file ------------------------')
            print(repr(e))
            with open(f'{OUTPUT_BASE_DIR}/best_results/{summary_file_name}', 'a') as outfile:
                outfile.write(f'´\n----------------Exception occurred while writing summary file------------------------\n')
                outfile.write(f'{repr(e)}\n')
            continue


def apply_pipeline_multi_layer_igraph_to_log_with_multiple_parameters(input_data: InputData):
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        print(f'Starting pipeline for {input_data.input_name}')
        outfile.write(run_start_string())

    best_precision = 0
    best_score = 0
    best_configs = []
    y_f1_scores_refined = []
    x_noises = [0, 0.1, 0.2, 0.3, 0.4]

    for label in input_data.labels_to_split:
        for window_size in [1, 2, 3, 4, 5]:
            for distance in [DistanceVariant.EDIT_DISTANCE,
                             DistanceVariant.SET_DISTANCE,
                             DistanceVariant.MULTISET_DISTANCE]:
                for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    try:
                        log = copy.deepcopy(input_data.original_log)

                        found_score, precision, f1_scores_refined = apply_pipeline_multi_layer_igraph_to_log(
                            input_data,
                            log,
                            distance,
                            window_size,
                            threshold,
                            best_score)
                        if len(f1_scores_refined) > 0:
                            y_f1_scores_refined = f1_scores_refined

                        if found_score > best_score:
                            best_configs = [
                                get_config_string(clustering_variant.ClusteringVariant.COMMUNITY_DETECTION,
                                                  distance,
                                                  input_data.labels_to_split,
                                                  input_data.max_number_of_traces,
                                                  input_data.log_path,
                                                  threshold,
                                                  window_size,
                                                  use_frequency=input_data.use_frequency)]
                            best_score = found_score
                            best_precision = precision
                        elif round(found_score, 2) == round(best_score, 2):
                            best_configs.append(
                                get_config_string(clustering_variant.ClusteringVariant.COMMUNITY_DETECTION,
                                                  distance,
                                                  input_data.labels_to_split,
                                                  input_data.max_number_of_traces,
                                                  input_data.log_path,
                                                  threshold,
                                                  window_size,
                                                  use_frequency=input_data.use_frequency))
                    except Exception as e:
                        print('----------------Exception occurred while running pipeline ------------------------')
                        print(repr(e))
                        with open(f'{OUTPUT_BASE_DIR}/best_results/{input_data.summary_file_name}', 'a') as outfile:
                            outfile.write(
                                f'´\n----------------Exception occurred while running pipeline------------------------\n')
                            outfile.write(f'{repr(e)}\n')
                            outfile.write('Error parameters:')
                            outfile.write(f'{input_data.input_name}')
                            outfile.write(f'{window_size}\n')
                            outfile.write(f'{threshold}\n')
                            outfile.write(f'{distance}\n')
                            outfile.write(f'{input_data.use_combined_context}\n')

                        with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
                            outfile.write(f'´\n----------------Exception occurred------------------------\n')
                            outfile.write(f'{repr(e)}\n')
                            outfile.write(f'{window_size}\n')
                            outfile.write(f'{threshold}\n')
                            outfile.write(f'{distance}\n')
                            outfile.write(f'{input_data.use_combined_context}\n')
                        continue

    print('best_score of all iterations:')
    print(best_score)

    print('best_precision of all iterations:')
    print(best_precision)

    if len(y_f1_scores_refined) > 0:
        plot_noise_to_f1_score(x_noises, input_data.y_f1_scores_unrefined, y_f1_scores_refined, input_data.input_name)
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        outfile.write('Best precision found:\n')
        outfile.write(str(best_precision))
    return best_score, best_precision, best_configs


def apply_pipeline_multi_layer_igraph_to_log(input_data: InputData,
                                             log,
                                             distance_variant: DistanceVariant,
                                             window_size: int,
                                             threshold: float,
                                             best_score: float):
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        outfile.write(get_config_string(clustering_variant.ClusteringVariant.COMMUNITY_DETECTION, distance_variant,
                                        input_data.labels_to_split, input_data.max_number_of_traces,
                                        input_data.log_path, threshold, window_size, input_data.use_frequency))

        if input_data.pipeline_variant == PipelineVariant.VARIANTS:
            label_splitter = LabelSplitterVariantBased(outfile,
                                                       input_data.labels_to_split,
                                                       threshold=threshold,
                                                       window_size=window_size,
                                                       distance_variant=distance_variant,
                                                       clustering_variant=clustering_variant.ClusteringVariant.COMMUNITY_DETECTION,
                                                       use_frequency=input_data.use_frequency,
                                                       concurrent_labels=input_data.concurrent_labels,
                                                       use_combined_context=input_data.use_combined_context)
        elif input_data.pipeline_variant == PipelineVariant.VARIANTS_MULTIPLEX:
            label_splitter = LabelSplitterVariantMultiplex(outfile,
                                                           input_data.labels_to_split,
                                                           threshold=threshold,
                                                           window_size=window_size,
                                                           distance_variant=distance_variant,
                                                           clustering_variant=clustering_variant.ClusteringVariant.COMMUNITY_DETECTION,
                                                           use_frequency=input_data.use_frequency,
                                                           use_combined_context=input_data.use_combined_context)
        else:
            label_splitter = LabelSplitterEventBased(outfile,
                                                     input_data.labels_to_split,
                                                     threshold=threshold,
                                                     window_size=window_size,
                                                     distance_variant=distance_variant,
                                                     clustering_variant=clustering_variant.ClusteringVariant.COMMUNITY_DETECTION,
                                                     use_combined_context=input_data.use_combined_context)

        split_log = label_splitter.split_labels(log)
        split_log_clustering = filter_duplicate_xor(split_log, input_data.labels_to_split,
                                                    label_splitter.found_clustering)

        outfile.write('\nPerformance split_log:\n')
        outfile.write('\nIM without threshold:\n')

        labels_to_original = label_splitter.get_split_labels_to_original_labels()

        if input_data.original_log_precision == 0:
            print('Starting to get original performance')
            final_marking, initial_marking, final_net, original_precision, original_simplicity, original_generalization, original_fitness = apply_im_without_noise(labels_to_original,
                                                                                      input_data.original_log,
                                                                                      input_data.original_log,
                                                                                      outfile,
                                                                                      label_splitter.short_label_to_original_label)
            input_data.original_log_precision = original_precision
            input_data.original_log_simplicity = original_simplicity
            input_data.original_log_generalization = original_generalization
            input_data.original_log_fitness = original_fitness
            print('finished original log calculation')

        final_marking, initial_marking, final_net, precision, simplicity, generalization, fitness = apply_im_without_noise(labels_to_original,
                                                                                      split_log,
                                                                                      input_data.original_log,
                                                                                      outfile,
                                                                                      label_splitter.short_label_to_original_label)

        f1_scores_refined = []
        ari_score = 0
        if input_data.ground_truth_clustering:
            ari_score = get_community_similarity(input_data.ground_truth_clustering, split_log_clustering)
        else:
            ari_score = precision
        outfile.write(f'\nAdjusted Rand Index:\n')
        outfile.write(f'{ari_score}\n\n')

        print(f'\nAdjusted Rand Index:\n')
        print(f'{ari_score}')

        with open(f'{RESULTS_BASE_DIR}/{input_data.folder_name}_{input_data.pipeline_variant}_NEW.csv', 'a') as f:
            writer = csv.writer(f)
            if input_data.ground_truth_clustering:
                row = [input_data.original_input_name, input_data.max_number_of_traces,
                       ' '.join(input_data.labels_to_split),
                       ', '.join(input_data.original_labels), input_data.original_log_precision,
                       input_data.original_log_simplicity, input_data.original_log_generalization,
                       input_data.original_log_fitness,
                       len(input_data.xixi_clustering),
                       input_data.xixi_precision, input_data.xixi_ari,
                       input_data.use_combined_context, input_data.use_frequency, window_size, distance_variant, threshold,
                       len(label_splitter.found_clustering), precision, ari_score, simplicity, generalization, fitness]
            else:
                row = [input_data.original_input_name, input_data.max_number_of_traces,
                       ' '.join(input_data.labels_to_split),
                       '[]', input_data.original_log_precision,
                       input_data.original_log_simplicity, input_data.original_log_generalization,
                       input_data.original_log_fitness,
                       0, 0, 0,
                       input_data.use_combined_context, input_data.use_frequency, window_size, distance_variant, threshold,
                       len(label_splitter.found_clustering), precision, ari_score, simplicity, generalization,
                       fitness]
            writer.writerow(row)

        if ari_score > best_score:
            print(f'\nHigher Adjusted Rand Index found: {ari_score}')
            print(f'\nPrecision of found clustering: {precision}')

            if input_data.use_noise:
                outfile.write('\nIM with noise threshold:\n')
                f1_scores_refined = apply_im_with_noise_and_export(input_data.input_name, 'split_log', split_log,
                                                                   input_data.original_log,
                                                                   outfile,
                                                                   label_splitter.get_split_labels_to_original_labels(),
                                                                   label_splitter.short_label_to_original_label)
            outfile_name = f'{input_data.input_name}_{threshold}_{distance_variant}_{window_size}'

            xes_exporter.apply(split_log,
                            f'{OUTPUT_BASE_DIR}/{outfile_name}_split_log.xes')

            tree = inductive_miner.apply_tree(split_log)
            export_models_and_pngs(final_marking, initial_marking, final_net, tree, input_data.input_name, f'{input_data.input_name}_{threshold}_{distance_variant}_{window_size}_split_log')
        return ari_score, precision, f1_scores_refined
