from enum import Enum
from functools import cache

import editdistance


class Distance(Enum):
    EDIT_DISTANCE = 1
    SET_DISTANCE = 2
    MULTISET_DISTANCE = 3


class DistanceCalculator:
    """
    Calculates (contextual) distance between two event
    """

    def __init__(self, window_size: int = 3, use_combined_context=False):
        self.window_size = window_size
        self.use_combined_context = use_combined_context

    def get_edit_distance(self, event_a, event_b) -> float:
        """
        Get the edit distance between the prefix and suffix with size window_size from two events

        :return: Edit distance with current parameters between A and B
        """
        if not self.use_combined_context:
            prefix_a = event_a['prefix'][(-1) * min(self.window_size, len(event_a['prefix'])):]
            prefix_b = event_b['prefix'][(-1) * min(self.window_size, len(event_b['prefix'])):]
            prefix_distance = self.get_cached_edit_distance(prefix_a, prefix_b)
            suffix_a = event_a['suffix'][:min(self.window_size, len(event_a['suffix'])) - 1]
            suffix_b = event_b['suffix'][:min(self.window_size, len(event_b['suffix'])) - 1]
            suffix_distance = self.get_cached_edit_distance(suffix_a, suffix_b)
            return prefix_distance * 0.5 + suffix_distance * 0.5
        string_a = event_a['prefix'][(-1) * min(self.window_size, len(event_a['prefix'])):] + event_a['suffix'][
                                                                                             :min(self.window_size,
                                                                                                  len(event_a[
                                                                                                          'suffix'])) - 1]
        string_b = event_b['prefix'][(-1) * min(self.window_size, len(event_b['prefix'])):] + event_b['suffix'][
                                                                                             :min(self.window_size,
                                                                                                  len(event_b[
                                                                                                          'suffix'])) - 1]
        return self.get_cached_edit_distance(string_a, string_b)

    @cache
    def get_cached_edit_distance(self, suffix_a, suffix_b):
        """
        Wrapper to cache the results of the edit distance calculation
        """
        return editdistance.eval(suffix_a, suffix_b)

    def get_set_distance(self, event_a, event_b) -> float:
        """
        Calculate set based contextual distance between two events

        :return: distance between event A and event B
        """
        prefix_a = event_a['prefix'][(-1) * min(self.window_size, len(event_a['prefix'])):]
        prefix_b = event_b['prefix'][(-1) * min(self.window_size, len(event_b['prefix'])):]
        suffix_a = event_a['suffix'][:min(self.window_size, len(event_a['suffix'])) - 1]
        suffix_b = event_b['suffix'][:min(self.window_size, len(event_b['suffix'])) - 1]
        if not self.use_combined_context:
            prefix_distance = _get_set_distance_for_strings(prefix_a, prefix_b)
            suffix_distance = _get_set_distance_for_strings(suffix_a, suffix_b)

            return prefix_distance * 0.5 + suffix_distance * 0.5
        context_a = prefix_a + suffix_a
        context_b = prefix_b + suffix_b
        return _get_set_distance_for_strings(context_a, context_b)

    def get_multiset_distance(self, event_a, event_b) -> float:
        """
        Calculate multi-set based contextual distance between two events

        :return: distance between event A and event B
        """
        if not self.use_combined_context:
            prefix_distance = _get_multiset_distance_for_strings(
                event_a['prefix'][(-1) * min(self.window_size, len(event_a['prefix'])):],
                event_b['prefix'][(-1) * min(self.window_size, len(event_b['prefix'])):])
            suffix_distance = _get_multiset_distance_for_strings(
                event_a['suffix'][:min(self.window_size, len(event_a['suffix'])) - 1],
                event_b['suffix'][:min(self.window_size, len(event_b['suffix'])) - 1])

            return prefix_distance * 0.5 + suffix_distance * 0.5
        return _get_multiset_distance_for_strings(
            event_a['prefix'][(-1) * min(self.window_size, len(event_a['prefix'])):] + event_a['suffix'][
                                                                                       :min(self.window_size, len(
                                                                                           event_a['suffix'])) - 1],
            event_b['prefix'][(-1) * min(self.window_size, len(event_b['prefix'])):] + event_b['suffix'][
                                                                                       :min(self.window_size, len(
                                                                                           event_b['suffix'])) - 1])


@cache
def _get_set_distance_for_strings(string_a: str, string_b: str) -> int:
    set_a = set(string_a)
    set_b = set(string_b)

    distance = 0
    if len(set_a) > len(set_b):
        long_sting = set_a
        short_string = set_b
    else:
        long_sting = set_b
        short_string = set_a
    for label in long_sting:
        if label not in short_string:
            distance += 1
    return distance


@cache
def _get_multiset_distance_for_strings(string_a, string_b) -> int:
    distance = 0
    if len(string_a) > len(string_b):
        long_sting = string_a
        short_string = string_b
    else:
        long_sting = string_b
        short_string = string_a

    for label in long_sting:
        if label not in short_string:
            distance += 1
        else:
            short_string = short_string.replace(label, '', 1)

    return distance
