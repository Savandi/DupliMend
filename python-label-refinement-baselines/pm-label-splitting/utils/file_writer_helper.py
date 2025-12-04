import csv
import datetime
import os
from datetime import datetime
from pathlib import Path

from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.process_tree.exporter import exporter as ptml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer

from pipeline.pipeline_variant import PipelineVariant

# Import paths from path_config (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR, RESULTS_BASE_DIR, BEST_RESULTS_DIR


def write_summary_file_with_parameters(best_configs, best_score, best_precision, name, summary_file_name):
    with open(f'{BEST_RESULTS_DIR}/With_Parameters_{summary_file_name}', 'a') as outfile:
        outfile.write(get_result_header(name))
        outfile.write(f'\nBest found configs for {name}:')
        for config in best_configs:
            outfile.write(config)
        outfile.write('Score:\n')
        outfile.write(str(best_score))
        outfile.write('\nPrecision of best_score model :\n')
        outfile.write(str(best_precision))


def setup_result_folder(folder_name: str, pipeline_variant: PipelineVariant):
    if not os.path.exists('../../outputs/best_results'):
        os.makedirs('../../outputs/best_results')

    header = [
        'Name', 'max_number_of_traces', 'labels_to_split', 'original labels', 'original_precision',
        'original_simplicity',
        'original_generalization', 'original_fitness', 'Xixi number of Clusters found', 'Xixi Precision', 'Xixi ARI',
        'use_combined_context', 'use_frequency', 'window_size', 'distance_metric', 'threshold',
        'Number of Clusters found',
        'Precision Align', 'ARI', 'Simplicity', 'Generalization', 'Fitness', 'Runtime']

    Path(f'{OUTPUT_BASE_DIR}/{folder_name}').mkdir(parents=True, exist_ok=True)

    csv_file_path = Path(f'{RESULTS_BASE_DIR}/{folder_name}_{pipeline_variant}_NEW.csv')
    if csv_file_path.is_file():
        print(csv_file_path)
        print('Warning: File already existis exiting')
        return

    with open(f'{RESULTS_BASE_DIR}/{folder_name}_{pipeline_variant}_NEW.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def run_start_string():
    return '''



----------------------------------------------------------------------------------------------
Output from {date}
----------------------------------------------------------------------------------------------


                '''.format(date=datetime.now())


def get_config_string(clustering_variant, distance_variant, labels_to_split, number_of_traces, original_log_path,
                      threshold, window_size, use_frequency=True):
    return '''

Parameters of this run:

Window size: {window_size}
Threshold for edges: {threshold}
Split candidates: {labels_to_split}
Max number of traces: {number_of_traces}
Method for distance calculation: {distance_variant}
Method for finding clusters: {clustering_variant}
Original log location: {original_log_path}
Use frequency: {use_frequency}

'''.format(threshold=threshold,
           window_size=window_size,
           labels_to_split=''.join(labels_to_split),
           number_of_traces=number_of_traces,
           distance_variant=distance_variant,
           clustering_variant=clustering_variant,
           original_log_path=original_log_path,
           use_frequency=use_frequency)


def get_result_header(name):
    return '''

-------------------------------------------------------
Results for {name} from {date}
-------------------------------------------------------

'''.format(date=datetime.now(), name=name)


def write_summary_file(best_score, best_precision, golden_standard_precision, name, summary_file_name, xixi_precision, xixi_ari):
    with open(f'{BEST_RESULTS_DIR}/{summary_file_name}', 'a') as outfile:
        outfile.write(f'\n\nBest score found for {name}:\n')
        outfile.write(f'{str(best_score)}\n')
        outfile.write(f'Precision of best_score model found for {name}:\n')
        outfile.write(f'{str(best_precision)}\n')
        if xixi_precision != 0:
            outfile.write(f'Precision found by Xixi for {name}:\n')
            outfile.write(f'{str(xixi_precision)}\n')
            outfile.write(f'Adjusted Rand Index from Xixi for {name}:\n')
            outfile.write(f'{str(xixi_ari)}\n')
        if golden_standard_precision != 0:
            outfile.write(f'Golden_standard_precision for {name}:\n')
            outfile.write(f'{str(golden_standard_precision)}\n')


def write_exception(e, outfile):
    print('----------------Exception occurred------------------------')
    print(e)
    outfile.write(f'´\n----------------Exception occurred------------------------\n')
    outfile.write(f'{repr(e)}\n')


def export_models_and_pngs(final_marking, initial_marking, net, original_tree, input_name, suffix):
    """Export process models (PNML, PTML, PNG) - continues even if some exports fail"""
    try:
        pnml_exporter.apply(net, initial_marking,
                            f'{OUTPUT_BASE_DIR}/{suffix}.pnml', final_marking=final_marking)
        print(f"[EXPORT] Saved PNML: {suffix}.pnml")
    except Exception as e:
        print(f"[EXPORT] Failed to save PNML: {e}")

    try:
        ptml_exporter.apply(original_tree, f'{OUTPUT_BASE_DIR}/{suffix}.ptml')
        print(f"[EXPORT] Saved PTML: {suffix}.ptml")
    except Exception as e:
        print(f"[EXPORT] Failed to save PTML: {e}")

    # PNG export already has try-except inside
    save_models_as_png(f'{OUTPUT_BASE_DIR}/{suffix}',
                       final_marking,
                       initial_marking,
                       net,
                       original_tree)


def save_models_as_png(name, final_marking, initial_marking, net, tree):
    """Save process models as PNG (optional - skips if Graphviz not available)"""
    try:
        gviz = pt_visualizer.apply(tree)
        pt_visualizer.save(gviz,
                          f'{name}_tree.png')
        parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
        gviz_petri_net = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
        pn_visualizer.save(gviz_petri_net,
                          f'{name}_net.png')
        print(f"[PNG] Saved visualization: {name}_tree.png, {name}_net.png")
    except Exception as e:
        # Graphviz not available - skip PNG generation (not essential for CSV results)
        print(f"[PNG] Skipping visualization (Graphviz not available): {e}")
    return
