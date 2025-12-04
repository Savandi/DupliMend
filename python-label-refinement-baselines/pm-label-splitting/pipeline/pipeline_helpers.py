import os
import re

from igraph import Clustering, compare_communities
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from utils.input_data import InputData
from pipeline.pipeline_variant import PipelineVariant

# Import OUTPUT_BASE_DIR from path_config (avoids circular imports)
from utils.path_config import OUTPUT_BASE_DIR


def get_clustering_from_xixi_log(log, labels_to_split, outfile, input_data: InputData):
    """Extract clustering from Xixi refined log - handles labels with/without numeric suffixes"""
    variants = variants_filter.get_variants(log)
    clustering = []
    split_labels = []
    label_to_cluster = {}  # Map labels to cluster IDs
    next_cluster_id = 0

    def get_cluster_id(label):
        """Extract cluster ID from label, or assign new ID if no numeric suffix"""
        nonlocal next_cluster_id

        # Try to extract numeric suffix (e.g., "O1" -> 1, "O2" -> 2)
        match = re.search(r'\d+$', label)
        if match:
            return int(match.group(0))
        else:
            # No numeric suffix - assign unique ID based on label itself
            if label not in label_to_cluster:
                label_to_cluster[label] = next_cluster_id
                next_cluster_id += 1
            return label_to_cluster[label]

    if input_data.pipeline_variant != PipelineVariant.EVENTS:
        for variant in variants:
            filtered_log = variants_filter.apply(log, [variant])
            for event in filtered_log[0]:
                label = event['concept:name']
                if label[0] in labels_to_split:
                    if label not in split_labels:
                        split_labels.append(label)
                    clustering.append(get_cluster_id(label))
    else:
        for trace in log:
            for event in trace:
                label = event['concept:name']
                if label[0] in labels_to_split:
                    if label not in split_labels:
                        split_labels.append(label)
                    clustering.append(get_cluster_id(label))

    outfile.write('\n Xixi split labels:\n')
    outfile.write(f'{str(split_labels)}\n')
    return Clustering(clustering)


def get_tuples_for_folder(folder_path, prefix):
    log_list = []
    identifier_pattern = f'^(\w+_\d+)'
    for f in os.listdir(folder_path):
        if 'LogD' in f:
            log_list.append((f'{prefix}/{re.match(identifier_pattern, f).group(1)}', f'{folder_path}{f}'))
    return log_list


def get_community_similarity(comm1: Clustering, comm2: Clustering, method='adjusted_rand'):
    return compare_communities(comm1, comm2, method)


def get_concurrent_labels(input_data: InputData, threshold: float = 0.85):
    with open(f'{OUTPUT_BASE_DIR}/{input_data.input_name}.txt', 'a') as outfile:
        variants = variants_filter.get_variants(input_data.original_log)
        predecessor_count = {}
        successor_count = {}
        concurrent_labels = []

        for variant in variants:
            filtered_log = variants_filter.apply(input_data.original_log, [variant])
            last_label = ''
            for event in filtered_log[0]:
                label = event['concept:name']
                if not label in predecessor_count:
                    predecessor_count[label] = 0
                    successor_count[label] = 0

                if last_label:
                    if label in input_data.labels_to_split:
                        predecessor_count[last_label] += len(variants[variant])
                    if last_label in input_data.labels_to_split:
                        successor_count[label] += len(variants[variant])
                last_label = label

        labels = set(successor_count.keys()) | set(predecessor_count.keys())

        for label in labels:
            total_count = predecessor_count[label] + successor_count[label]
            if total_count == 0:
                continue
            directly_follows_ratio = abs((predecessor_count[label] - successor_count[label]) / total_count)
            if directly_follows_ratio < threshold and label not in input_data.labels_to_split:
                concurrent_labels.append(label)
        outfile.write('\n Concurrent labels:\n')
        outfile.write(f'{str(concurrent_labels)}\n')
    return concurrent_labels


def filter_duplicate_xor(event_log, labels_to_split, clustering: Clustering):
    """Filter duplicate XOR transitions - handles labels with/without numeric suffixes"""

    def extract_suffix(label):
        """Safely extract numeric suffix from label, return label itself if no suffix"""
        match = re.search(r'\d+$', label)
        return match.group(0) if match else label

    process_tree = inductive_miner.apply(event_log)
    net, initial_marking, final_marking = pt_converter.apply(process_tree)

    seen_transitions = []
    updated_label_mapping = {}
    must_update_log = False

    for t_1 in net.transitions:
        if t_1.label is not None and t_1.label[0] in labels_to_split and t_1.label not in seen_transitions:
            pre_places_1 = set()
            post_places_1 = set()
            for arc in t_1.in_arcs:
                pre_places_1.add(arc.source)
            for arc in t_1.out_arcs:
                post_places_1.add(arc.target)

            seen_transitions.append(t_1.label)
            updated_label_mapping[extract_suffix(t_1.label)] = t_1.label

            for t_2 in net.transitions:
                if t_2.label is not None and t_2.label not in seen_transitions and t_2.label[0] in labels_to_split and t_2.label != t_1.label:
                    pre_places_2 = set()
                    post_places_2 = set()
                    for arc in t_2.in_arcs:
                        pre_places_2.add(arc.source)
                    for arc in t_2.out_arcs:
                        post_places_2.add(arc.target)

                    if pre_places_1 == pre_places_2 and post_places_1 == post_places_2:
                        print(f'Merging {t_2.label} and {t_1.label}')
                        must_update_log = True
                        seen_transitions.append(t_2.label)
                        updated_label_mapping[extract_suffix(t_2.label)] = t_1.label

    if must_update_log:
        for trace in event_log:
            for event in trace:
                label = event['concept:name']
                if label[0] in labels_to_split:
                    suffix = extract_suffix(label)
                    if suffix in updated_label_mapping:
                        event['concept:name'] = updated_label_mapping[suffix]
        new_clustering = []
        for i in range(len(clustering.membership)):
            m = clustering.membership[i]
            mapped_label = updated_label_mapping.get(f'{m}', f'{m}')
            suffix = extract_suffix(mapped_label)
            try:
                new_clustering.append(int(suffix))
            except ValueError:
                # If suffix is not numeric, use original value
                new_clustering.append(m)
        clustering = Clustering(new_clustering)

    return clustering


def get_imprecise_labels(log):
    print('Getting imprecise labels')
    imprecise_labels = set()
    for trace in log:
        for event in trace:
            # Check if OrgLabel exists
            if 'OrgLabel' in event:
                if event['OrgLabel'] != event['concept:name']:
                    imprecise_labels.add(event['concept:name'])
            else:
                # If no OrgLabel, we need to determine imprecise labels differently
                print("Warning: No 'OrgLabel' attribute found. Cannot auto-detect imprecise labels.")
                return []
    return list(imprecise_labels)


def get_imprecise_labels_from_comparison(imprecise_log_path, original_log_path):
    """
    Compare imprecise log with original log to find imprecise labels
    """
    print(f'Getting imprecise labels by comparing {imprecise_log_path} with {original_log_path}')
    from pm4py.objects.log.importer.xes import importer as xes_importer
    
    if not os.path.exists(original_log_path):
        print(f"Warning: Original log not found at {original_log_path}")
        return []
    
    imprecise_log = xes_importer.apply(imprecise_log_path)
    original_log = xes_importer.apply(original_log_path)
    
    imprecise_labels = set()
    
    # Compare corresponding traces and events
    for i, (imprecise_trace, original_trace) in enumerate(zip(imprecise_log, original_log)):
        for j, (imprecise_event, original_event) in enumerate(zip(imprecise_trace, original_trace)):
            if imprecise_event['concept:name'] != original_event['concept:name']:
                imprecise_labels.add(imprecise_event['concept:name'])
                print(f"Found imprecise label: {imprecise_event['concept:name']} (original: {original_event['concept:name']})")
    
    return list(imprecise_labels)

def get_corresponding_original_log_path(logd_path):
    """
    Get the path to the corresponding original log file
    
    Examples:
    A_1_LogD_Sequence_feb16-1625.xes.gz → A_1_Log.xes.gz
    B_1_LogD_Sequence_feb16-1625.xes.gz → B_1_Log.xes.gz
    """
    import re
    
    # Extract the identifier (e.g., A_1, B_1) from the LogD filename
    filename = os.path.basename(logd_path)
    
    # Match pattern like A_1_LogD_Sequence_feb16-1625.xes.gz
    match = re.match(r'^([A-Z]+_\d+)_LogD_.*\.(xes(?:\.gz)?)$', filename)
    
    if match:
        identifier = match.group(1)  # e.g., A_1
        extension = match.group(2)   # e.g., xes.gz
        
        # Construct the original log filename
        original_filename = f"{identifier}_Log.{extension}"
        
        # Get the directory and construct full path
        directory = os.path.dirname(logd_path)
        original_log_path = os.path.join(directory, original_filename)
        
        return original_log_path
    else:
        print(f"Warning: Could not parse LogD filename pattern: {filename}")
        return None
