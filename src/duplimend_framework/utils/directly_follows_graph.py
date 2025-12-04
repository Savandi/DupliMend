from collections import defaultdict
from config.config import forgetting_params
from typing import DefaultDict, Tuple, Dict
from src.duplimend_framework.utils.logging_utils import log_traceability

def _default_transition_dict():
    """Factory function for default transition dictionary - picklable"""
    return {"count": 0, "last_seen_event": 0}

class DirectlyFollowsGraph:
    def __init__(self):
        """
        Initialize the directly follows graph with incremental transition tracking.
        """
        # Core transition matrix: (prev_activity, curr_activity) -> {"count": int, "last_seen_event": int}
        self.graph: DefaultDict[Tuple[str, str], Dict[str, int]] = defaultdict(_default_transition_dict)
        
        # Track transitions per case for forgetting
        self.case_transitions = defaultdict(list)  # case_id -> [(prev, curr), ...]
        
        # Statistics for monitoring
        self.total_transitions = 0
        self.unique_transitions = 0

    def reset_timestamps_for_testing(self, new_base_event=0):
        """
        Reset all timestamps to start fresh for testing while preserving transition counts.
        This allows proper forgetting behavior during inference phase.
        """
        print(f"[DFG] Resetting timestamps for testing phase (base_event={new_base_event})")
        
        transitions_reset = 0
        for transition_key, meta in self.graph.items():
            # Reset last_seen_event to the new base
            meta["last_seen_event"] = new_base_event
            transitions_reset += 1
        
        # Clear case transitions since test cases are different from training cases
        cases_cleared = len(self.case_transitions)
        self.case_transitions.clear()
        
        print(f"[DFG] Reset {transitions_reset} transition timestamps")
        print(f"[DFG] Cleared {cases_cleared} case transition mappings")
        print(f"[DFG] Preserved {self.unique_transitions} unique transitions with counts")

    def add_transition(self, case_id, prev_activity, curr_activity, global_event_counter):
        """
        Incrementally add a transition to the global directly follows matrix.
        
        Args:
            case_id: The case this transition belongs to
            prev_activity: Previous activity in the sequence
            curr_activity: Current activity in the sequence
            global_event_counter: Current global event number for recency tracking
        """
        key = (prev_activity, curr_activity)
        
        # Update global transition count
        if self.graph[key]["count"] == 0:
            self.unique_transitions += 1
            
        self.graph[key]["count"] += 1
        self.graph[key]["last_seen_event"] = global_event_counter
        self.total_transitions += 1
        
        # Track this transition for the case (for forgetting)
        self.case_transitions[case_id].append(key)
        
        print(f"[DFG] Added transition: {prev_activity} -> {curr_activity} (count: {self.graph[key]['count']})")

    def apply_forgetting(self, global_event_counter):
        """
        Apply temporal decay to transition counts based on recency and frequency thresholds.
        """
        print(f"[DFG] Applying forgetting at event {global_event_counter}")
        
        transitions_to_remove = []
        
        for transition_key, meta in list(self.graph.items()):
            last_seen = meta["last_seen_event"]
            current_count = meta["count"]
            age = global_event_counter - last_seen
            
            # Apply decay based on age and frequency
            if (current_count < forgetting_params["frequency_decay_threshold"] and 
                age > forgetting_params["removal_threshold_events"]):
                
                # Mark for removal
                transitions_to_remove.append(transition_key)
                print(f"[DFG] Removing old transition: {transition_key[0]} -> {transition_key[1]} (age: {age}, count: {current_count})")
            
            elif age > forgetting_params["removal_threshold_events"] // 2:
                # Apply gradual decay to older transitions
                decay_factor = forgetting_params["temporal_decay_rate"]
                new_count = max(1, int(current_count * (1 - decay_factor)))
                
                if new_count != current_count:
                    self.graph[transition_key]["count"] = new_count
                    self.total_transitions -= (current_count - new_count)
                    print(f"[DFG] Decayed transition: {transition_key[0]} -> {transition_key[1]} ({current_count} -> {new_count})")
        
        # Remove transitions marked for deletion
        for transition_key in transitions_to_remove:
            removed_count = self.graph[transition_key]["count"]
            self.total_transitions -= removed_count
            self.unique_transitions -= 1
            del self.graph[transition_key]

    def remove_case_transitions(self, case_id):
        """
        Remove all transitions associated with a forgotten case.
        This is called when a case is deemed too old and should be forgotten.
        """
        if case_id not in self.case_transitions:
            return
            
        print(f"[DFG] Removing transitions for forgotten case: {case_id}")
        
        for transition_key in self.case_transitions[case_id]:
            if transition_key in self.graph:
                # Decrement the count
                self.graph[transition_key]["count"] -= 1
                self.total_transitions -= 1
                
                # Remove if count reaches zero
                if self.graph[transition_key]["count"] <= 0:
                    del self.graph[transition_key]
                    self.unique_transitions -= 1
                    print(f"[DFG] Removed transition: {transition_key[0]} -> {transition_key[1]}")
        
        # Clean up case tracking
        del self.case_transitions[case_id]

    def get_global_frequency(self, prev_activity, curr_activity):
        """
        Get the frequency count of a specific transition.
        
        Returns:
            int: Number of times prev_activity -> curr_activity has occurred
        """
        return self.graph.get((prev_activity, curr_activity), {}).get("count", 0)

    def get_incoming_distribution(self, curr_activity):
        """
        Get the distribution of activities that directly precede curr_activity.
        
        Returns:
            dict: {prev_activity: count, ...}
        """
        distribution = defaultdict(int)
        for (prev, curr), meta in self.graph.items():
            if curr == curr_activity:
                distribution[prev] += meta.get("count", 0)
        return dict(distribution)

    def get_outgoing_distribution(self, prev_activity):
        """
        Get the distribution of activities that directly follow prev_activity.
        
        Returns:
            dict: {next_activity: count, ...}
        """
        distribution = defaultdict(int)
        for (prev, curr), meta in self.graph.items():
            if prev == prev_activity:
                distribution[curr] += meta.get("count", 0)
        return dict(distribution)
    
    def get_parallel_activities(self, activity, threshold):
        """
        Find activities that appear to be parallel with the given activity.
        Parallel activities have bidirectional transitions above the threshold.
        
        Args:
            activity: The activity to find parallels for
            threshold: Minimum count for both directions to be considered parallel
            
        Returns:
            set: Activities that are parallel with the input activity
        """
        parallel_acts = set()
        for (a, b), meta in self.graph.items():
            if a == activity:
                count_ab = meta.get("count", 0)
                count_ba = self.graph.get((b, a), {}).get("count", 0)
                
                # Both directions must meet threshold
                if count_ab >= threshold and count_ba >= threshold:
                    parallel_acts.add(b)
                    
        return parallel_acts

    def get_most_frequent_transitions(self, top_n=10):
        """
        Get the most frequent transitions in the graph.
        
        Args:
            top_n: Number of top transitions to return
            
        Returns:
            list: [(prev_activity, curr_activity, count), ...] sorted by count desc
        """
        transitions = []
        for (prev, curr), meta in self.graph.items():
            transitions.append((prev, curr, meta["count"]))
        
        return sorted(transitions, key=lambda x: x[2], reverse=True)[:top_n]

    def get_activity_statistics(self, activity):
        """
        Get statistics for a specific activity.
        
        Returns:
            dict: Statistics including incoming/outgoing counts, unique predecessors/successors
        """
        incoming = self.get_incoming_distribution(activity)
        outgoing = self.get_outgoing_distribution(activity)
        
        return {
            "total_incoming": sum(incoming.values()),
            "total_outgoing": sum(outgoing.values()),
            "unique_predecessors": len(incoming),
            "unique_successors": len(outgoing),
            "incoming_activities": incoming,
            "outgoing_activities": outgoing
        }

    def get_graph_statistics(self):
        """
        Get overall graph statistics.
        
        Returns:
            dict: Overall statistics about the graph
        """
        return {
            "total_transitions": self.total_transitions,
            "unique_transitions": self.unique_transitions,
            "unique_activities": len(set([a for (a, b) in self.graph.keys()] + [b for (a, b) in self.graph.keys()])),
            "active_cases": len(self.case_transitions)
        }