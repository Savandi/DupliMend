import numpy as np
import math
import hashlib
from collections import defaultdict
from src.duplimend_framework.utils.global_state import directly_follows_graph, case_last_seen_global
from config.config import control_flow_context_window, parallelization_threshold, unfolding_threshold, forgetting_params, control_flow_pattern_window_size, max_control_flow_patterns, enable_self_loop_handling
from src.duplimend_framework.utils.global_state import (
    per_case_directly_follows_graph,
    case_activity_sequences,
    activity_positions,
    pattern_frequencies,
    activity_instance_counts, activity_hash_mapping, case_context_freeze, self_loop_cases
)
from src.duplimend_framework.utils import global_state


def forget_old_cases(global_event_counter):
    """
    Remove globally inactive cases and outdated feature tracking using event-driven forgetting.
    Ensures that expired cases no longer influence historical tracking.
    """
    for case_id, transitions in list(per_case_directly_follows_graph.items()):
        for (src, dst) in list(transitions.keys()):
            transitions[(src, dst)] *= forgetting_params["forgetting_factor"]
            if transitions[(src, dst)] < forgetting_params["forgetting_threshold"]:
                del transitions[(src, dst)]
            
        if not transitions:
            del per_case_directly_follows_graph[case_id]

    for case_id in list(case_last_seen_global.keys()):
        last_event_counter = case_last_seen_global.get(case_id, 0)
        events_since_last_seen = global_event_counter - last_event_counter
        frequency = np.exp(-events_since_last_seen / max(forgetting_params["decay_after_events"], 1))

        if frequency < forgetting_params["frequency_decay_threshold"] and events_since_last_seen > forgetting_params["removal_threshold_events"]:
            directly_follows_graph.remove_case_transitions(case_id)

            per_case_directly_follows_graph.pop(case_id, None)

            case_last_seen_global.pop(case_id, None)
            
            case_activity_sequences.pop(case_id, None)
            activity_positions.pop(case_id, None)
            pattern_frequencies.pop(case_id, None)
            activity_instance_counts.pop(case_id, None)
            
            for activity in list(self_loop_cases):
                self_loop_cases[activity].pop(case_id, None)
                if not self_loop_cases[activity]:
                    del self_loop_cases[activity]
                    
            case_context_freeze.pop(case_id, None)

def hash_activity(activity, max_value=1000000):
    """Hash an activity name to a consistent numerical ID"""
    hash_obj = hashlib.md5(str(activity).encode('utf-8'))
    hash_value = int(hash_obj.hexdigest(), 16)
    hashed = hash_value % max_value
    activity_hash_mapping[activity] = hashed
    return hashed


def update_case_sequence(case_id, activity, timestamp, global_event_counter):
    """
    Update the activity sequence for a case and track positional information.
    Call this whenever a new event is processed.
    """
    case_activity_sequences[case_id].append({
        'activity': activity,
        'timestamp': timestamp,
        'position': len(case_activity_sequences[case_id])
    })
    curr_position = len(case_activity_sequences[case_id]) - 1
    activity_positions[case_id][activity].append(curr_position)
    instance_num = len(activity_positions[case_id][activity])
    activity_instance_counts[case_id][activity] = instance_num

    if global_event_counter is not None:
        case_last_seen_global[case_id] = global_event_counter




def compute_entropy(counts):
    """Compute entropy of a distribution"""
    values = list(counts.values()) if isinstance(counts, dict) else counts
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in values if c > 0)

def detect_loop_context(sequence, current_pos, current_activity):
    """
    Detect if the current activity is part of a loop and determine loop characteristics.
    Returns dict with detailed loop information including type and length.
    """
    if current_pos < 1:
        return {"in_loop": False, "loop_length": 0, "loop_type": "none", "loop_repetitions": 0}
    
    activities = [item['activity'] for item in sequence]
    
    if current_pos > 0 and activities[current_pos - 1] == current_activity:
        loop_length = 1
        i = current_pos - 1
        while i >= 0 and activities[i] == current_activity:
            loop_length += 1
            i -= 1
        return {
            "in_loop": True, 
            "loop_length": 1, 
            "loop_type": "self_loop", 
            "loop_repetitions": loop_length
        }
    
    if current_pos >= 3:
        if (activities[current_pos - 2] == current_activity and 
            activities[current_pos - 1] != current_activity):
            
            other_activity = activities[current_pos - 1]
            repetitions = 1
            
            i = current_pos - 3
            while i >= 1:
                if (activities[i] == other_activity and 
                    activities[i - 1] == current_activity):
                    repetitions += 1
                    i -= 2
                else:
                    break
            
            if repetitions >= 2:
                return {
                    "in_loop": True,
                    "loop_length": 2,
                    "loop_type": "alternating_2",
                    "loop_repetitions": repetitions
                }
    
    for loop_len in range(3, min(8, current_pos + 1)):
        if current_pos >= (loop_len * 2) - 1:
            current_pattern = activities[current_pos - loop_len + 1:current_pos + 1]
            prev_pattern = activities[current_pos - (2 * loop_len) + 1:current_pos - loop_len + 1]
            
            if current_pattern == prev_pattern:
                repetitions = 2
                check_pos = current_pos - (2 * loop_len)
                
                while check_pos >= loop_len - 1:
                    check_pattern = activities[check_pos - loop_len + 1:check_pos + 1]
                    if check_pattern == current_pattern:
                        repetitions += 1
                        check_pos -= loop_len
                    else:
                        break
                
                return {
                    "in_loop": True,
                    "loop_length": loop_len,
                    "loop_type": f"repeating_{loop_len}",
                    "loop_repetitions": repetitions
                }
    
    return {"in_loop": False, "loop_length": 0, "loop_type": "none", "loop_repetitions": 0}

def extract_control_flow_features(case_id, curr_activity):
    """
    Extract enhanced control flow features that capture context for homonym detection.
    Returns a dict with all features, including prev/next activities as labels.
    Now robust to parallelism: skips parallel activities in the context window.
    Implements conditional context freezing for self-loops.
    """

    features = {}

    case_graph = per_case_directly_follows_graph.get(case_id, {})
    incoming_counts = defaultdict(int)
    outgoing_counts = defaultdict(int)
    for (src, dst), count in case_graph.items():
        if dst == curr_activity:
            incoming_counts[src] += count
        if src == curr_activity:
            outgoing_counts[dst] += count

    features["num_incoming_total"] = float(sum(incoming_counts.values()))
    features["entropy_in"] = float(compute_entropy(incoming_counts.values()))

    if case_id in case_activity_sequences and case_id in activity_positions:
        sequence = case_activity_sequences[case_id]
        positions = activity_positions[case_id].get(curr_activity, [])
        if sequence and positions:
            seq_len = len(sequence)
            if seq_len > 0:
                latest_pos = positions[-1]
                features["rel_position"] = float(latest_pos) / max(1, seq_len - 1) if seq_len > 1 else 0.0
                features["is_first"] = 1.0 if latest_pos == 0 else 0.0
                features["is_last"] = 1.0 if latest_pos == seq_len - 1 else 0.0
                features["occurrence_num"] = float(len(positions))
                if len(positions) > 1:
                    features["dist_from_prev"] = float(latest_pos - positions[-2])
                else:
                    features["dist_from_prev"] = -1.0
                    
                loop_info = detect_loop_context(sequence, latest_pos, curr_activity)
                features["is_in_loop"] = 1.0 if loop_info["in_loop"] else 0.0
                features["loop_length"] = float(loop_info["loop_length"])
                features["loop_repetitions"] = float(loop_info["loop_repetitions"])
                
                features["is_self_loop"] = 1.0 if loop_info["loop_type"] == "self_loop" else 0.0
                features["is_alternating_2_loop"] = 1.0 if loop_info["loop_type"] == "alternating_2" else 0.0
                features["is_repeating_pattern"] = 1.0 if loop_info["loop_type"].startswith("repeating_") else 0.0

                parallel_acts = directly_follows_graph.get_parallel_activities(curr_activity, parallelization_threshold) \
                    if hasattr(directly_follows_graph, "get_parallel_activities") else set()
                
                features["num_parallel_activities"] = float(len(parallel_acts))
                before_ctx = []
                i = latest_pos - 1
                seen_parallel = set(
                    item['activity'] for item in sequence[:latest_pos] if item['activity'] in parallel_acts
                )
                while i >= 0 and len(before_ctx) < control_flow_context_window:
                    act = sequence[i]['activity']
                    if act in parallel_acts and act in seen_parallel:
                        i -= 1
                        continue
                    before_ctx.append(act)
                    i -= 1

                if enable_self_loop_handling:
                    prev_activity = sequence[latest_pos - 1]['activity'] if latest_pos > 0 else None

                    if prev_activity == curr_activity:
                        self_loop_cases[curr_activity][case_id].add(latest_pos)
                    num_cases_with_self_loop = len(self_loop_cases[curr_activity])
                    total_active_cases = len(case_activity_sequences)
                    self_loop_ratio = num_cases_with_self_loop / max(1, total_active_cases)

                    if prev_activity == curr_activity and self_loop_ratio < unfolding_threshold:
                        frozen = case_context_freeze[case_id].get(curr_activity)
                        if frozen:
                            for idx in range(control_flow_context_window):
                                features[f"prev_{idx+1}"] = frozen.get(f"prev_{idx+1}")
                        else:
                            for idx in range(control_flow_context_window):
                                features[f"prev_{idx+1}"] = before_ctx[idx] if idx < len(before_ctx) else None
                                if curr_activity not in case_context_freeze[case_id]:
                                    case_context_freeze[case_id][curr_activity] = {}
                                case_context_freeze[case_id][curr_activity][f"prev_{idx+1}"] = features[f"prev_{idx+1}"]
                    else:
                        for idx in range(control_flow_context_window):
                            features[f"prev_{idx+1}"] = before_ctx[idx] if idx < len(before_ctx) else None
                else:
                    for idx in range(control_flow_context_window):
                        features[f"prev_{idx+1}"] = before_ctx[idx] if idx < len(before_ctx) else None

        else:
            features.update({
                "rel_position": 0.0,
                "is_first": 1.0,
                "is_last": 1.0,
                "occurrence_num": 1.0,
                "dist_from_prev": -1.0,
                "is_in_loop": 0.0,
                "loop_length": 0.0,
                "loop_repetitions": 0.0,
                "is_self_loop": 0.0,
                "is_alternating_2_loop": 0.0,
                "is_repeating_pattern": 0.0,
                "num_parallel_activities": 0.0
            })
            
            for idx in range(control_flow_context_window):
                features[f"prev_{idx+1}"] = None
            
    else:
        features.update({
            "rel_position": 0.0,
            "is_first": 1.0,
            "is_last": 1.0,
            "occurrence_num": 1.0,
            "dist_from_prev": -1.0,
            "is_in_loop": 0.0,
            "loop_length": 0.0,
            "loop_repetitions": 0.0,
            "is_self_loop": 0.0,
            "is_alternating_2_loop": 0.0,
            "is_repeating_pattern": 0.0,
            "num_parallel_activities": 0.0
        })
        
        for idx in range(control_flow_context_window):
            features[f"prev_{idx+1}"] = None
        
    
    return features


def get_control_flow_feature_names():
    """Return the names of the enhanced control flow features"""
    base_features = [
        "num_incoming_total",
        "entropy_in",         
        "rel_position",         
        "is_first",
        "is_last", 
        "occurrence_num",
        "dist_from_prev",
        "is_in_loop", 
        "loop_length",
        "loop_repetitions",
        "is_self_loop",
        "is_alternating_2_loop", 
        "is_repeating_pattern",
        "num_parallel_activities"
    ]
    
    context_features = [f"prev_{i+1}" for i in range(control_flow_context_window)]
    
    
    return base_features + context_features

