import networkx as nx
from egraph_mapping_cost_recursive import default_labeling_function as labeling_function
# citations
# wikipedia: The transitive reduction of a finite directed acyclic graph (a directed graph without directed cycles) is unique
class egraph:

    def __init__(self, trace, id, Parameters, use_trace_folding): #requieres consistent trace with events ordered by timestamp

        self.Parameters = Parameters
        #.Parameters.CASE_ID_KEY: "case:concept:name",
        #timestamp_filter.Parameters.TIMESTAMP_KEY: "time:timestamp"})

        self.time_stamp_key = self.Parameters["TIMESTAMP_KEY"]
        self.activity_key = self.Parameters["ACTIVITY_KEY"]
        self.lifecycle_key = self.Parameters["LIFECYCLE_KEY"]
        self.event_identification_key = self.Parameters["EVENT_IDENTIFICATION"]
        self.lifecycle_mode = self.Parameters["LIFECYCLE_MODE"] #eather atomic or full (start and complete)
        self.case_id_key = self.Parameters["CASE_ID_KEY"]
        self.graph_id = id
        self.trace = trace

        if self.lifecycle_mode == "full":
            self.events = self.get_events_full()


            self.nodeID_to_event_dict = {}

            self.partial_order = self.construct_partial_order_full()
            self.transitive_reduction = nx.transitive_reduction(self.partial_order)
        elif self.lifecycle_mode == "atomic":
            self.events = self.get_events_atomic()


            self.nodeID_to_event_dict = {}

            self.partial_order = self.construct_partial_order_atomic()
            #self.transitive_reduction = nx.transitive_reduction(self.partial_order)
        self.size = len(self.events)


        self.distances, self.dist_to_start, self.dist_to_end = self.calculate_distances_advanced(use_trace_folding)
        self.contexts = self.calculate_contexts(self.Parameters)
    def calculate_distances(self, use_trace_folding):
        if not use_trace_folding:
            distances = []
            for nodeID_1 in range(0, len(self.events)):
                d = []
                for nodeID_2 in range(0, len(self.events)):
                    d.append((nodeID_1 - nodeID_2))
                distances.append(d)
            return distances, [id for id, _ in enumerate(self.events)], [self.size - id for id, _ in enumerate(self.events)]  # todo implement folding
        else:
            pair_position_dict = {}
            for node_ID, node in enumerate(self.events):
                if node_ID + 1 < len(self.events):
                    pair = [labeling_function(self.nodeID_to_event_dict[node_ID]), labeling_function(self.nodeID_to_event_dict[node_ID + 1])]
                    pair = ''.join(sorted(pair))

                    if pair in pair_position_dict.keys():
                        end = pair_position_dict[pair][-1]
                        if end < node_ID:
                            pair_position_dict[pair].append(node_ID)
                    else:
                        pair_position_dict[pair] = [node_ID]
            '''
            s1 = ""
            for e in self.nodeID_to_event_dict.values():
                s1 += labeling_function(e)
                s1 += " "
            print(s1)
            print("pair_position_dict: ", pair_position_dict)
            '''
            dist_to_start = {}
            dist_to_end = {}
            distances = []
            for node_ID1, node1 in enumerate(self.events):
                d = []
                for node_ID2, node2 in enumerate(self.events):
                        higher_node_ID = max(node_ID1, node_ID2)
                        lower_node_ID = min(node_ID1, node_ID2)
                        max_in_between_loop_length = 0
                        for pair in pair_position_dict.keys():
                            positions = pair_position_dict[pair]
                            if len(positions) > 2:
                                start_loop_position = positions[-1]
                                end_loop_position = positions[0]
                                for position in positions[1:]:
                                    if position <= start_loop_position and lower_node_ID < position:
                                        start_loop_position = position
                                    if end_loop_position <= position and position < higher_node_ID:
                                        end_loop_position = position

                                if end_loop_position - start_loop_position > max_in_between_loop_length:
                                    max_in_between_loop_length = end_loop_position - start_loop_position
                        if node_ID1 > node_ID2:
                            d.append(node_ID1 - node_ID2 - max_in_between_loop_length)
                        else:
                            d.append(node_ID1 - node_ID2 + max_in_between_loop_length)


                        if node_ID2 == self.size - 1:
                            dist_to_end[node_ID1] = self.size - node_ID1 - max_in_between_loop_length
                        if node_ID2 == 0:
                            dist_to_start[node_ID1] = node_ID2 - node_ID1 - max_in_between_loop_length
                distances.append(d)

            #print("distances" ,distances)



            return distances, dist_to_start, dist_to_end



    def calculate_distances_advanced(self, use_trace_folding):
        if not use_trace_folding:
            distances = []
            for nodeID_1 in range(0, len(self.events)):
                d = []
                for nodeID_2 in range(0, len(self.events)):
                    d.append((nodeID_1 - nodeID_2))
                distances.append(d)
            return distances, [id for id, _ in enumerate(self.events)], [self.size - id for id, _ in enumerate(self.events)]
        else:
            pair_position_dict = {}
            for node_ID, node in enumerate(self.events):
                if node_ID + 1 < len(self.events):
                    pair = [labeling_function(self.nodeID_to_event_dict[node_ID]), labeling_function(self.nodeID_to_event_dict[node_ID + 1])]
                    pair = ''.join(sorted(pair))

                    if pair in pair_position_dict.keys():
                        end = pair_position_dict[pair][-1]
                        if end < node_ID:
                            pair_position_dict[pair].append(node_ID)
                    else:
                        pair_position_dict[pair] = [node_ID]
            '''
            s1 = ""
            for e in self.nodeID_to_event_dict.values():
                s1 += labeling_function(e)
                s1 += " "
            print(s1)
            print("pair_position_dict: ", pair_position_dict)
            '''
            dist_to_start = {}
            dist_to_end = {}
            distances = []
            for node_ID1, node1 in enumerate(self.events):
                d = []
                for node_ID2, node2 in enumerate(self.events):
                        higher_node_ID = max(node_ID1, node_ID2)
                        lower_node_ID = min(node_ID1, node_ID2)
                        max_in_between_loop_length = 0

                        for pair in pair_position_dict.keys():
                            positions = pair_position_dict[pair]
                            if len(positions) > 1:
                                start_loop_position = "none"#positions[-1]
                                end_loop_position = "none"#positions[0]
                                loop_iterations = 0
                                for position in positions:
                                    if (start_loop_position == "none" or position <= start_loop_position) and lower_node_ID  <= position + 1 and position <= higher_node_ID:
                                        start_loop_position = position
                                    if (end_loop_position == "none" or end_loop_position <= position) and lower_node_ID <= position and position <= higher_node_ID + 1:
                                        end_loop_position = position
                                    #if lower_node_ID < position and position < higher_node_ID: # penalty for each iteration
                                if start_loop_position != "none" and end_loop_position != "none":
                                    loop_iterations = positions.index(end_loop_position) - positions.index(start_loop_position)
                                    loop_length = end_loop_position - start_loop_position
                                    #todo incldue +length of 1 iteration
                                if (start_loop_position != "none" and end_loop_position != "none") and end_loop_position - start_loop_position - loop_iterations > max_in_between_loop_length:
                                    # max_in_between_loop_length = end_loop_position - start_loop_position - loop_iterations
                                    # max_in_between_loop_length = loop_iterations * (1 - loop_iterations/(end_loop_position - start_loop_position))
                                    max_in_between_loop_length = end_loop_position - start_loop_position - loop_iterations * (1 - loop_iterations/(end_loop_position - start_loop_position))
                        if node_ID1 > node_ID2:
                            d.append(node_ID1 - node_ID2 - max_in_between_loop_length)
                        else:
                            d.append(node_ID1 - node_ID2 + max_in_between_loop_length)


                        if node_ID2 == self.size - 1:
                            dist_to_end[node_ID1] = self.size - node_ID1 - max_in_between_loop_length
                        if node_ID2 == 0:
                            dist_to_start[node_ID1] = node_ID2 - node_ID1 - max_in_between_loop_length
                distances.append(d)

            #for d in distances:
            #    print(d)



            return distances, dist_to_start, dist_to_end




    def calculate_distances_new(self, use_trace_folding):
        if not use_trace_folding:
            distances = []
            for nodeID_1 in range(0, len(self.events)):
                d = []
                for nodeID_2 in range(0, len(self.events)):
                    d.append((nodeID_1 - nodeID_2))
                distances.append(d)
            return distances, [id for id, _ in enumerate(self.events)], [self.size - id for id, _ in enumerate(
                self.events)]  # todo implement folding
        else:

            '''
            s1 = ""
            for e in self.nodeID_to_event_dict.values():
                s1 += labeling_function(e)
                s1 += " "
            print(s1)
            '''

            dist_to_start = {}
            dist_to_end = {}
            distances = []
            for node_ID1, node1 in enumerate(self.events):
                d = []
                for node_ID2, node2 in enumerate(self.events):
                        #print(node_ID1)
                        #print(node_ID2)
                        pair_position_dict = {}
                        for node_ID in range(min(node_ID1, node_ID2), max(node_ID1, node_ID2)+1):
                            if node_ID + 1 < len(self.events):
                                pair = [labeling_function(self.nodeID_to_event_dict[node_ID]),
                                        labeling_function(self.nodeID_to_event_dict[node_ID + 1])]
                                pair = ''.join(sorted(pair))

                                if pair in pair_position_dict.keys():
                                    start, second_start, end = pair_position_dict[pair]
                                    #if node_ID < start:
                                    #    pair_position_dict[pair] = node_ID, end
                                    if end < node_ID:
                                        pair_position_dict[pair] = start, second_start, node_ID
                                        if start == second_start:
                                            pair_position_dict[pair] = start, node_ID, node_ID
                                else:
                                    pair_position_dict[pair] = (node_ID, node_ID, node_ID)

                        #print("pair_position_dict: ", pair_position_dict)
                        max_in_between_loop_length = 0
                        for pair in pair_position_dict.keys():
                            start, second_start, end = pair_position_dict[pair]
                            if (node_ID1 < node_ID2) and (node_ID1 <= start and end <= node_ID2):
                                if max_in_between_loop_length <= end - second_start:
                                    max_in_between_loop_length = end - second_start

                            if (node_ID1 >= node_ID2) and (node_ID2 <= start and end <= node_ID1):
                                if max_in_between_loop_length <= end - second_start:
                                    max_in_between_loop_length = end - second_start

                        if node_ID1 > node_ID2:
                            d.append(node_ID1 - node_ID2 - max_in_between_loop_length)
                            #print("distance: ", node_ID1 - node_ID2 - max_in_between_loop_length)
                        else:
                            d.append(node_ID1 - node_ID2 + max_in_between_loop_length)
                            #print("distance: ", node_ID1 + node_ID2 - max_in_between_loop_length)

                        if node_ID2 == self.size-1:
                            dist_to_end[node_ID1] = self.size - node_ID1 - max_in_between_loop_length
                        if node_ID2 == 0:
                            dist_to_start[node_ID1] = node_ID2 - node_ID1 - max_in_between_loop_length
                distances.append(d)


            '''
            for node_ID1, node1 in enumerate(self.events):
                max_in_between_loop_length = 0
                for pair in pair_position_dict.keys():
                    start, end = pair_position_dict[pair]
                    if node_ID1 <= start:
                        if max_in_between_loop_length < end - start:
                            max_in_between_loop_length = end - start
                dist_to_end[node_ID1] = self.size - node_ID1 - max_in_between_loop_length

            for node_ID1, node1 in enumerate(self.events):
                max_in_between_loop_length = 0
                for pair in pair_position_dict.keys():
                    start, end = pair_position_dict[pair]
                    if node_ID1 >= end:
                        if max_in_between_loop_length < end - start:
                            max_in_between_loop_length = end - start
                dist_to_start[node_ID1] = node_ID1 - max_in_between_loop_length
            '''
            #print("dist_to_start", dist_to_start)
            #print("dist_to_end", dist_to_end)
            return distances, dist_to_start, dist_to_end
            print(distances)





    def get_predecessors(self, position, k):
        # print(list(range(max(0, position - k), position)))
        return range(max(0, position - k), position)

    def get_successors(self, position, k):
        return range(position + 1, min(position + k + 1, self.size))

    def get_concurrencies(self, position):
        return []
        #neighbors = set(egraph.partial_order.successors(nodeID))
        #neighbors = neighbors.union(set(egraph.partial_order.predecessors(nodeID)))
        #concurrencies = set(egraph.partial_order.nodes()).difference(neighbors)
        #concurrencies.remove(nodeID)
        #return concurrencies


    def calculate_contexts(self, Parameters):
        contexts =[]
        for nodeID in range(0, len(self.events)):
            contexts.append((labeling_function(self.nodeID_to_event_dict[nodeID]), self.get_predecessors(nodeID, Parameters["k"]),self.get_concurrencies(nodeID), self.get_successors(nodeID, Parameters["k"])))
        return contexts

    def construct_partial_order_full(self):
        partial_order = nx.DiGraph()
        #todo:  no timestamp mode
        for nodeID_1 in range(0, len(self.events)):
            self.nodeID_to_event_dict[nodeID_1] = self.events[nodeID_1]
            for nodeID_2 in range(nodeID_1+1, len(self.events)):
                if self.events[nodeID_1]["end_timestamp"] <= self.events[nodeID_2]["start_timestamp"]:
                    partial_order.add_edge(nodeID_1, nodeID_2)
        return partial_order

    def construct_partial_order_atomic(self):
        partial_order = nx.DiGraph()

        if self.time_stamp_key != "no_timestamp":
            for nodeID_1 in range(0, len(self.events)):
                self.nodeID_to_event_dict[nodeID_1] = self.events[nodeID_1]
                for nodeID_2 in range(nodeID_1+1, len(self.events)):
                    if self.events[nodeID_1]["end_timestamp"] < self.events[nodeID_2]["end_timestamp"]:
                        partial_order.add_edge(nodeID_1, nodeID_2)
        else:
            for nodeID_1 in range(0, len(self.events)):
                self.nodeID_to_event_dict[nodeID_1] = self.events[nodeID_1]
                #for nodeID_2 in range(nodeID_1+1, len(self.events)):                               \todo markin
                #        partial_order.add_edge(nodeID_1, nodeID_2)
        return partial_order


    def has_started(self, event):
        return event[self.lifecycle_key] == "start"

    def is_finished(self, event):
        return event[self.lifecycle_key] == "complete"

    def has_lifecycle(self, event):
        return self.lifecycle_key in event

    def get_events_full(self):
        #returns: lists of events sorted by start-timestamp
        visited_positions = []
        events = []

        for start_position in range(0, len(self.trace)):
            if start_position not in visited_positions and self.has_started(self.trace[start_position]):
                visited_positions.append(start_position)
                for end_position in range(start_position+1, len(self.trace)): #trace already sorted by timeframe  (problems can oocur if
                    if self.is_finished(self.trace[end_position]) and \
                            self.trace[start_position][self.event_identification_key] == self.trace[end_position][self.event_identification_key]:
                        new_event = self.trace[start_position]
                        new_event["start_position"] = start_position
                        new_event["end_position"] = end_position
                        new_event["start_timestamp"] = self.trace[start_position][self.time_stamp_key]
                        new_event["end_timestamp"] = self.trace[end_position][self.time_stamp_key]
                        events.append(new_event)
                        visited_positions.append(end_position)
                        print(new_event)
                        break
        return events

    def get_events_atomic(self):
        #returns: lists of events sorted by start-timestamp
        visited_positions = []
        events = []


        if self.time_stamp_key != "no_timestamp":
            for position in range(0, len(self.trace)):
                new_event = self.trace[position]
                new_event["start_position"] = position
                new_event["end_position"] = position
                new_event["start_timestamp"] = self.trace[position][self.time_stamp_key]
                new_event["end_timestamp"] = self.trace[position][self.time_stamp_key]
                events.append(new_event)

        else:
            for position in range(0, len(self.trace)):
                new_event = self.trace[position]
                new_event["start_position"] = position #needed for log transformation
                new_event["end_position"] = position
                events.append(new_event)

        return events


