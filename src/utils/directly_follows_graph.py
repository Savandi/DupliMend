from collections import defaultdict

from config.config import lossy_counting_error_rate


class DirectlyFollowsGraph:
    def __init__(self, use_lossy_counting=False, error_rate=lossy_counting_error_rate):
        """
        Initialize the directly follows graph.

        :param use_lossy_counting: Boolean to enable or disable lossy counting.
        :param error_rate: Error rate for lossy counting (used only if lossy counting is enabled).
        """
        self.graph = defaultdict(lambda: defaultdict(int))
        self.use_lossy_counting = use_lossy_counting
        self.error_rate = error_rate
        self.total_events = 0
        self.threshold = 0 if not use_lossy_counting else 1 / error_rate

    def add_transition(self, from_activity, to_activity):
        """
        Add a transition to the directly follows graph.

        :param from_activity: The activity from which the transition starts.
        :param to_activity: The activity to which the transition goes.
        """
        self.graph[from_activity][to_activity] += 1
        self.total_events += 1

        if self.use_lossy_counting:
            self._apply_lossy_counting()

    def get_frequency(self, from_activity, to_activity):
        """
        Get the frequency of a specific transition.

        :param from_activity: The activity from which the transition starts.
        :param to_activity: The activity to which the transition goes.
        :return: Frequency of the transition.
        """
        return self.graph[from_activity][to_activity]

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
