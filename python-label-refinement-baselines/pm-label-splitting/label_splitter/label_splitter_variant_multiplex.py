import json
import string
from itertools import combinations
from typing import TextIO

import igraph
import leidenalg as la
from pm4py.algo.filtering.log.variants import variants_filter

from pipeline.clustering_method import ClusteringMethod
from label_splitter.distance_metrics import Distance, DistanceCalculator
from label_splitter.label_splitter_variant_based import ncr


class LabelSplitter:
    """
    EXPERIMENTAL:
    Applies the label splitting algorithm, with the variant compression and a multiplex graph, using multiple attributes.
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
                 use_frequency=False,
                 use_combined_context=False):
        self.labels_to_split = labels_to_split
        self.window_size = window_size
        self.threshold = threshold
        self.prefix_weight = prefix_weight
        self._split_labels_to_original_labels = {}
        self.outfile = outfile
        self.label_and_id_to_event = {}
        self.variants_to_count = {}
        self.distance_variant = distance_variant
        self.distance_calculator = DistanceCalculator(window_size, use_combined_context)
        self.clustering_variant = clustering_variant
        self._variant_to_label = {}
        self.use_frequency = use_frequency
        self.short_label_to_original_label = {}
        self.found_clustering = None

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
        layers = []
        event_graphs = self.get_event_graphs_from_event_log(log)
        layers.append(event_graphs)
        for i in range(2):
            layers.append(event_graphs.copy())

        for index, layer in enumerate(layers):
            self.calculate_edges(layer, index)
        self.get_communities_leiden_multiplex(layers=layers)
        self.set_split_labels(log)

        return log

    def get_event_graphs_from_event_log(self, log):
        print('Variants based approach')
        variants = variants_filter.get_variants(log)
        event_graphs = {}

        for variant in variants:
            filtered_log = variants_filter.apply(log, [variant])

            prefix = ''
            processed_events = []
            occurrence_counters = {}
            for event in filtered_log[0]:
                label = event['concept:name']
                if 'original_label' in event.keys():
                    self.short_label_to_original_label[label] = event['original_label']

                if label not in list(event_graphs.keys()) and label in self.labels_to_split:
                    event_graphs[label] = igraph.Graph()
                    self.label_and_id_to_event[label] = []

                for preceding_event in processed_events:
                    preceding_event['suffix'] = preceding_event['suffix'] + label

                if label not in occurrence_counters:
                    occurrence_counters[label] = 0
                else:
                    occurrence_counters[label] += 1

                event['prefix'] = prefix
                event['suffix'] = ''
                event['label'] = label
                event['variant'] = label + '_' + str(variant).replace(',', '') + f'_{occurrence_counters[label]}'
                self.variants_to_count[event['variant']] = len(variants[variant])
                processed_events.append(event)
                prefix = prefix + label
            for event in processed_events:
                label = event['concept:name']
                if label in self.labels_to_split:
                    self.label_and_id_to_event[label].append(event)
                    event_graphs[label].add_vertices(1)

        return event_graphs

    def get_distance_by_index(self, index: int, event_a, event_b):
        if index == 0:
            return self.distance_calculator.get_edit_distance(event_a, event_b)
        elif index == 1:
            return self.distance_calculator.get_set_distance(event_a, event_b)
        elif index == 2:
            return self.distance_calculator.get_multiset_distance(event_a, event_b)
        else:
            print('########### Warning: Distance metric not found, fallback to default distance ###########')
            return self.distance_calculator.get_edit_distance(event_a, event_b)

    def calculate_edges(self, event_graphs, index) -> None:
        for (label, graph) in event_graphs.items():
            print(f'Calculating edges for {label}')
            edges = []
            weights = []

            for (vertex_a, vertex_b) in combinations(range(len(graph.vs)), 2):
                edit_distance = self.get_distance_by_index(index, self.label_and_id_to_event[label][vertex_a],
                                                           self.label_and_id_to_event[label][vertex_b])

                normalized_distance = (1 - edit_distance / self.window_size)
                if self.use_frequency:
                    weight = normalized_distance * (
                            self.variants_to_count[self.label_and_id_to_event[label][vertex_a]['variant']] *
                            self.variants_to_count[self.label_and_id_to_event[label][vertex_b]['variant']])
                else:
                    weight = normalized_distance
                if normalized_distance > self.threshold:
                    edges.append((vertex_a, vertex_b))
                    weights.append(weight)
            if self.use_frequency:
                for vertex_a in range(len(graph.vs)):
                    count = self.variants_to_count[self.label_and_id_to_event[label][vertex_a]['variant']]
                    if count > 1:
                        ########################################################################
                        # TODO: Check if times 2 or not!!!!!!!
                        ########################################################################
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
            print(partition)
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

    def get_communities_leiden_multiplex(self, layers) -> None:
        print('Starting community detection')
        for (label, graph) in layers[0].items():
            print(f'Getting communities for {label}')
            graphs = [graph, layers[1][label], layers[2][label]]
            membership, improvement = la.find_partition_multiplex(graphs, la.ModularityVertexPartition,
                                                                  weights=graph.es['weight'], seed=396482)
            print('membership')
            print(membership)
            print('improvement')
            print(improvement)
            partition = []

            for index, cluster in enumerate(membership):
                if not len(partition) > cluster:
                    while not len(partition) > cluster:
                        partition.append([])
                partition[cluster].append(index)

            print('partition')
            print(partition)

            self._write('Found multiplex partition: \n')
            self._write(str(partition))

            self._write('Found multiplex memberships: \n')
            self._write(str(membership))

            self._write('Improvement made through multiplex optimizer:')
            self._write(str(improvement))

            for count, cluster in enumerate(partition):
                self._split_labels_to_original_labels[f'{label}_{count}'] = label
                for vertex in cluster:
                    self.label_and_id_to_event[label][vertex]['label'] = f'{label}_{count}'
                    self._variant_to_label[self.label_and_id_to_event[label][vertex]['variant']] = f'{label}_{count}'

        self._write('\nReassigned labels')
        print('Finished community detection')

    def set_split_labels(self, log):
        variants = variants_filter.get_variants(log)
        for variant in variants:
            filtered_log = variants_filter.apply(log, [variant])
            for trace in filtered_log:
                occurrence_counters = {}
                for event in trace:
                    label = event['concept:name']
                    if label not in occurrence_counters:
                        occurrence_counters[label] = 0
                    else:
                        occurrence_counters[label] += 1
                    event['variant'] = label + '_' + str(variant).replace(',', '') + f'_{occurrence_counters[label]}'
                    if event['variant'] in self._variant_to_label:
                        event['concept:name'] = self._variant_to_label[event['variant']]
        print('Finished setting labels')
