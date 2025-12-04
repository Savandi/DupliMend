import os
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import pm4py
def oracle_detection(log):
    print("[ORACLE_DEBUG] Starting oracle_detection")
    imprecise_labels = set()
    original_labels = set()
    
    # Check if this is a real-life log (no OrgLabel attribute or OrgLabel == concept:name for all events)
    is_real_life_log = True
    total_events = 0
    different_labels = 0
    
    for trace in log:
        for event in trace:
            total_events += 1
            # Check if OrgLabel exists first
            if 'OrgLabel' not in event:
                # No OrgLabel attribute means this is a real-life log
                is_real_life_log = True
                break
            elif event["OrgLabel"] != event["concept:name"]:
                original_labels.add(event["OrgLabel"])
                imprecise_labels.add(event["concept:name"])
                different_labels += 1
                is_real_life_log = False
        if 'OrgLabel' not in event:  # Break out of outer loop too
            break

    print(f"[ORACLE_DEBUG] Total events: {total_events}, Different labels: {different_labels}")
    print(f"[ORACLE_DEBUG] Is real life log: {is_real_life_log}")
    
    # If this is a real-life log (no OrgLabel or no difference between OrgLabel and concept:name)
    if is_real_life_log or different_labels == 0:
        print("Detected real-life log - using frequency-based imprecise label detection")
        return frequency_based_detection(log)

    original_labels = list(original_labels)
    imprecise_labels = list(imprecise_labels)
    
    print(f"[ORACLE_DEBUG] Original labels: {original_labels}")
    print(f"[ORACLE_DEBUG] Imprecise labels: {imprecise_labels}")
    
    # Safety check for synthetic logs
    if len(imprecise_labels) == 0:
        print("Warning: No imprecise labels detected, using frequency-based detection as fallback")
        return frequency_based_detection(log)
        
    original_labels.append(imprecise_labels[0])
    print(f"[ORACLE_DEBUG] Final original labels: {original_labels}")
    print(f"[ORACLE_DEBUG] Final imprecise labels: {imprecise_labels}")
    return original_labels, imprecise_labels

def frequency_based_detection(log):
    """
    For real-life logs, detect potentially imprecise labels based on frequency.
    Labels that appear very frequently might be imprecise (e.g., generic activities).
    """
    from collections import Counter
    
    # Count frequency of each activity label
    label_counts = Counter()
    total_events = 0
    
    for trace in log:
        for event in trace:
            label_counts[event["concept:name"]] += 1
            total_events += 1
    
    # Calculate frequency percentages
    label_frequencies = {label: count/total_events for label, count in label_counts.items()}
    
    # Identify potentially imprecise labels (appear in more than 15% of events)
    frequency_threshold = 0.15
    
    imprecise_labels = []
    for label, frequency in label_frequencies.items():
        if frequency > frequency_threshold:
            imprecise_labels.append(label)
    
    # If no labels meet the frequency criterion, select the most frequent ones
    if len(imprecise_labels) == 0:
        # Take the top 2 most frequent labels as potentially imprecise
        most_frequent = label_counts.most_common(2)
        imprecise_labels = [label for label, count in most_frequent]
    
    # For real-life logs, original_labels = all unique labels
    original_labels = list(label_counts.keys())
    
    print(f"Frequency-based detection results:")
    print(f"  Total unique labels: {len(original_labels)}")
    print(f"  Potentially imprecise labels: {imprecise_labels}")
    print(f"  Label frequencies: {dict(list(label_frequencies.items())[:5])}...")  # Show first 5
    
    return original_labels, imprecise_labels

def label_in_flower_detector(log):
    #todo
    '''
    tree = inductive_miner.apply_tree(log)
    for node in tree.children:
        print(node)
    print("label: ", tree.label)
    bottomup =pm4py.objects.process_tree.bottomup.get_bottomup_nodes(tree)
    print("bottomup: ", bottomup)
    for b in bottomup:
        b.parent
    return 0'''
    return [], []
