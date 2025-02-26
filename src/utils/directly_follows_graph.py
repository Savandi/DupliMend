from collections import defaultdict
import numpy as np
from config.config import lossy_counting_budget, frequency_decay_threshold, decay_after_events, removal_threshold_events


class DirectlyFollowsGraph:
    def __init__(self):
        """
        Initialize the directly follows graph with adaptive forgetting.
        """
        self.graph = defaultdict(lambda: {"count": 0, "last_seen_event": 0})  # Initialize to 0
        self.case_transitions = defaultdict(list)  #  Track transitions per case


    def add_transition(self, case_id, prev_activity, curr_activity, global_event_counter):
        """
        Track transition frequency globally while associating transitions with specific cases.
        """
        print(f"[DEBUG] Add transitions with specific case")
        key = (prev_activity, curr_activity)
        self.graph[key]["count"] += 1
        self.graph[key]["last_seen_event"] = global_event_counter  # Use explicit parameter

        # Store the transition in case-specific tracking
        self.case_transitions[case_id].append(key)

        #  Apply forgetting every `decay_after_events`
        if global_event_counter % decay_after_events == 0:
            self.apply_forgetting(global_event_counter)  #  Pass explicitly
        print(f"[DEBUG] End Add transitions with specific case")

    def apply_forgetting(self, global_event_counter):
        """
        Decay transition counts and remove rarely used transitions based on event counts.
        """
        print(f"[DEBUG] apply decay and forgetting for transitions")
        for key in list(self.graph.keys()):
            events_since_last_seen = global_event_counter - self.graph[key]["last_seen_event"]
            self.graph[key]["count"] *= np.exp(-events_since_last_seen / decay_after_events)

            if self.graph[key]["count"] < frequency_decay_threshold and events_since_last_seen > removal_threshold_events:
                del self.graph[key]

            if len(self.graph) <= lossy_counting_budget:
                break

    def get_global_frequency(self, prev_activity, curr_activity):
        """
        Retrieve the total frequency of a transition (prev_activity → curr_activity) across all cases.
        """
        return self.graph.get((prev_activity, curr_activity), {}).get("count", 0)

    def remove_case_transitions(self, case_id):
        """
        Remove all transitions contributed by a forgotten case.
        """
        print(f"[DEBUG] remove_case_transitions")
        if case_id in self.case_transitions:
            for key in self.case_transitions[case_id]:
                if key in self.graph:
                    self.graph[key]["count"] -= 1  #
                    if self.graph[key]["count"] <= 0:
                        del self.graph[key]  #
            #  Remove case-specific transition tracking
            del self.case_transitions[case_id]


