import copy
import json
import operator as op
import string
from functools import reduce
from itertools import combinations
from typing import TextIO, List

import leidenalg as la

from label_splitter.distance_metrics import Distance, DistanceCalculator
from label_splitter.event_graphs_variant_based import EventGraphsVariantBased
from pipeline.clustering_method import ClusteringMethod


class LabelSplitter:
    """
    Applies the label splitting algorithm, with the variant compression.
    Generates the event graph, applies the clustering method and applies the label splitting to the event log
    """

    def __init__(self,
                 outfile: TextIO,
                 labels_to_split,
                 window_size: int = 3,
                 threshold: float = 0.75,
                 prefix_weight: float = 0.5,
                 distance_variant=Distance.EDIT_DISTANCE,
                 clustering_variant=ClusteringMethod.COMMUNITY_DETECTION,
                 use_frequency: bool = False,
                 concurrent_labels: List[str] = None,
                 use_combined_context: bool = False,
                 event_graphs_variant_based: EventGraphsVariantBased = None
                 ):
        if concurrent_labels is None:
            concurrent_labels = []
        self.concurrent_labels = concurrent_labels
        self.labels_to_split = labels_to_split
        self.window_size = window_size
        self.threshold = threshold
        self.prefix_weight = prefix_weight
        self._split_labels_to_original_labels = {}
        self.outfile = outfile
        self.label_and_id_to_event = event_graphs_variant_based.label_and_id_to_event
        self.variants_to_count = event_graphs_variant_based.variants_to_count
        self.distance_variant = distance_variant
        self.distance_calculator = DistanceCalculator(window_size, use_combined_context)
        self.clustering_variant = clustering_variant
        self._variant_to_label = {}
        self.use_frequency = use_frequency
        self.short_label_to_original_label = event_graphs_variant_based.short_label_to_original_label
        self.found_clustering = None
        self.event_graphs = event_graphs_variant_based.event_graphs

        if distance_variant is Distance.EDIT_DISTANCE:
            self.get_distance = self.distance_calculator.get_edit_distance
        elif distance_variant is Distance.SET_DISTANCE:
            self.get_distance = self.distance_calculator.get_set_distance
        elif distance_variant is Distance.MULTISET_DISTANCE:
            self.get_distance = self.distance_calculator.get_multiset_distance
        else:
            print('Warning: Distance metric not found, fallback to default distance')
            self.get_distance = self.distance_calculator.get_edit_distance

    def _write(self, log_entry: string) -> None:
        self.outfile.write(f'{log_entry}\n')

    def get_split_labels_to_original_labels(self):
        self._write('Map:')
        self._write(json.dumps(self._split_labels_to_original_labels))
        return self._split_labels_to_original_labels

    def split_labels(self, log):
        print('Starting label splitting')
        event_graphs = copy.deepcopy(self.event_graphs)

        self.calculate_edges(event_graphs)
        self.get_communities_leiden(event_graphs=event_graphs)
        self.set_split_labels(log)

        return log

    def calculate_edges(self, event_graphs) -> None:
        for (label, graph) in event_graphs.items():
            print(f'Calculating edges for {label}')
            # Use dynamic lists instead of pre-allocated arrays to save memory
            # Only store edges that meet the threshold, not all possible combinations
            edges = []
            weights = []

            for vertex_a, vertex_b in combinations(range(len(graph.vs)), 2):
                distance = self.get_distance(self.label_and_id_to_event[label][vertex_a],
                                             self.label_and_id_to_event[label][vertex_b])

                normalized_distance = (1 - distance / self.window_size)
                if self.use_frequency:
                    weight = normalized_distance * (
                            self.variants_to_count[self.label_and_id_to_event[label][vertex_a]['variant']] *
                            self.variants_to_count[self.label_and_id_to_event[label][vertex_b]['variant']])
                else:
                    weight = normalized_distance
                if normalized_distance >= self.threshold:
                    edges.append((vertex_a, vertex_b))
                    weights.append(weight)

            if self.use_frequency:
                # Use dynamic lists for self-edges too
                for vertex_a in range(len(graph.vs)):
                    count = self.variants_to_count[self.label_and_id_to_event[label][vertex_a]['variant']]
                    if count > 1:
                        weight = ncr(count, 2) * 2
                        edges.append((vertex_a, vertex_a))
                        weights.append(weight)

            graph.add_edges(edges)
            graph.es['weight'] = weights
        print('Finished calculating edges')

    def get_communities_leiden(self, event_graphs) -> None:
        print('Starting community detection')
        for (label, graph) in event_graphs.items():
            print(f'Getting communities for {label}')
            partition = la.find_partition(graph, la.ModularityVertexPartition, weights=graph.es['weight'], seed=396482)
            self.found_clustering = partition
            self._write('Found communities: \n')
            self._write(str(partition))

            for count, cluster in enumerate(partition):
                self._split_labels_to_original_labels[f'{label}_{count}'] = label
                for vertex in cluster:
                    self.label_and_id_to_event[label][vertex]['label'] = f'{label}_{count}'
                    self._variant_to_label[self.label_and_id_to_event[label][vertex]['variant']] = f'{label}_{count}'

        self._write('\nReassigned labels')
        print('Finished community detection')

    def set_split_labels(self, log):
        for trace in log:
            occurrence_counters = {}
            for event in trace:
                label = event['concept:name']
                if label not in occurrence_counters:
                    occurrence_counters[label] = 0
                else:
                    occurrence_counters[label] += 1
                event['variant'] = label + '_' + event['variant_raw'] + f'_{occurrence_counters[label]}'
                if event['variant'] in self._variant_to_label:
                    event['concept:name'] = self._variant_to_label[event['variant']]
        print('Finished setting labels')


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom
