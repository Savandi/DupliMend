import json
import string
from itertools import combinations
from typing import TextIO

import igraph

from pipeline.clustering_method import ClusteringMethod
from label_splitter.distance_metrics import Distance, DistanceCalculator
import leidenalg as la


class LabelSplitter:
    """
    Applies the label splitting algorithm, without the variant compression.
    Generates the event graph, applies the clustering method and applies the label splitting to the event log
    """
    def __init__(self,
                 outfile: TextIO,
                 labels_to_split,
                 window_size: int = 3,
                 threshold: float = 0.75,
                 prefix_weight: float = 0.5,
                 distance_variant = Distance.EDIT_DISTANCE,
                 clustering_variant = ClusteringMethod.COMMUNITY_DETECTION,
                 concurrent_labels=None,
                 use_combined_context=False):
        if concurrent_labels is None:
            concurrent_labels = []
        self.concurrent_labels = concurrent_labels
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
        event_graphs = self.get_event_graphs_from_event_log(log)

        self.calculate_edges(event_graphs)
        self.get_communities_louvain(event_graphs=event_graphs, log=log)

        return log

    def get_event_graphs_from_event_log(self, log):
        print('Event based approach')
        event_graphs = {}
        for case_id, trace in enumerate(log):
            prefix = ''
            processed_events = []
            for position, event in enumerate(trace):
                label = event['concept:name']
                if 'original_label' in event.keys():
                    self.short_label_to_original_label[label] = event['original_label']

                event['case_id'] = case_id
                event['position'] = position
                if label not in list(event_graphs.keys()) and label in self.labels_to_split:
                    event_graphs[label] = igraph.Graph()
                    self.label_and_id_to_event[label] = []

                for preceding_event in processed_events:
                    if label in self.concurrent_labels:
                        break
                    preceding_event['suffix'] = preceding_event['suffix'] + label

                event['prefix'] = prefix
                event['suffix'] = ''
                event['label'] = label
                processed_events.append(event)
                if label not in self.concurrent_labels:
                    prefix = prefix + label

            for event in processed_events:
                label = event['concept:name']
                if label in self.labels_to_split:
                    self.label_and_id_to_event[label].append(event)
                    event_graphs[label].add_vertices(1)
        print('Finished calculating event_graphs')
        return event_graphs

    def calculate_edges(self, event_graphs) -> None:
        for (label, graph) in event_graphs.items():
            print(f'Calculating edges for {label}')
            edges = []
            weights = []

            for (vertex_a, vertex_b) in combinations(range(len(graph.vs)), 2):
                edit_distance = self.get_distance(self.label_and_id_to_event[label][vertex_a],
                                                  self.label_and_id_to_event[label][vertex_b])
                weight = (1 - edit_distance / self.window_size)
                if weight > self.threshold:
                    edges.append((vertex_a, vertex_b))
                    weights.append(weight)
            graph.add_edges(edges)
            graph.es['weight'] = weights
        print('Finished calculating edges')

    def get_communities_louvain(self, event_graphs, log) -> None:
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
                    event = self.label_and_id_to_event[label][vertex]
                    log[event['case_id']][event['position']]['concept:name'] = f'{label}_{count}'
        self._write('\nReassigned labels')
        print('Finished setting labels')
        print('Finished community detection')
