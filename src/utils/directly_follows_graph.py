from collections import defaultdict

from config.config import lossy_counting_error_rate


class DirectlyFollowsGraph:
    def __init__(self, use_lossy_counting=False, error_rate=lossy_counting_error_rate):
        """
        Initialize the directly follows graph.

        :param use_lossy_counting: Boolean to enable or disable lossy counting.
        :param error_rate: Error rate for lossy counting (used only if lossy counting is enabled).
        """
        self.graph = defaultdict(int)
        self.use_lossy_counting = use_lossy_counting
        self.error_rate = error_rate
        self.total_events = 0
        self.threshold = 0 if not use_lossy_counting else 1 / error_rate

    def add_transition(self, prev_activity, curr_activity):
        """
        Track transition frequency globally for feature scoring.
        """
        self.graph[(prev_activity, curr_activity)] = self.graph.get((prev_activity, curr_activity), 0) + 1

    def get_global_frequency(self, prev_activity, curr_activity):
        """
        Retrieve the total frequency of a transition (prev_activity → curr_activity) across all cases.
        """
        return self.graph.get((prev_activity, curr_activity), 0)

    def _apply_lossy_counting(self):
        """
        Apply lossy counting to prune less frequent transitions.
        """
        for from_activity in list(self.graph.keys()):
            for to_activity in list(self.graph[from_activity].keys()):
                if self.graph[from_activity][to_activity] < self.threshold:
                    del self.graph[from_activity][to_activity]

            if not self.graph[from_activity]:  # Remove empty nodes
                del self.graph[from_activity]
