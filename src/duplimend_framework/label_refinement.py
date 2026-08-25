import pandas as pd
from collections import defaultdict
from config.config import control_flow_column


class LabelRefiner:
    def __init__(self, output_file_path, input_columns):
        self.output_file_path = output_file_path
        self.cluster_mapping = defaultdict(lambda: defaultdict(int))
        self.input_columns = input_columns
        self.semantic_mapping = defaultdict(dict) 
        self.cluster_representatives = defaultdict(lambda: defaultdict(list))
        self.initialize_csv()

    def initialize_csv(self):
        try:
            refined_df = pd.DataFrame(columns=self.input_columns)
            refined_df.to_csv(self.output_file_path, index=False)
        except Exception:
            pass

    def refine_label(self, event_label, cluster_id, dbstream_instance, force_split=False, force_relabel=False):
        """
        Simplified method that always uses actual cluster ID in the refined label.
        """
        if force_relabel and cluster_id in self.semantic_mapping.get(event_label, {}):
            del self.semantic_mapping[event_label][cluster_id]

        if cluster_id is None:
            return event_label
        
        return f"{event_label}_{cluster_id}"
    

    def add_representative(self, event_label, cluster_id, event_dict, max_representatives=10):
        """Add an event as a representative for its cluster (for future label generation)"""
        representatives = self.cluster_representatives[event_label][cluster_id]
        representatives.append(event_dict.copy())

        if len(representatives) > max_representatives:
            representatives.pop(0)

    def process_event(self, event, cluster_id, dbstream_instance, split_merge_result):
        """
        Process an event and assign a refined activity label.
        """
        event_label = event.get(control_flow_column)
        if not event_label:
            raise ValueError("Activity label is missing in the event.")
        
        force_relabel = split_merge_result.get("merge_occurred", False)
        force_split = split_merge_result.get("split_occurred", False)
        
        self.add_representative(event_label, cluster_id, event)
        
        refined_label = self.refine_label(
            event_label, 
            cluster_id, 
            dbstream_instance, 
            force_split=force_split,
            force_relabel=force_relabel
        )
        
        clean_event = {}
        for column in self.input_columns:
            if column == "refined_activity":
                clean_event[column] = refined_label
            elif column in event:
                clean_event[column] = event[column]
        
        return clean_event

    def append_event_to_csv(self, event):
        df = pd.DataFrame([event])
        df.to_csv(self.output_file_path, mode="a", header=False, index=False)

    def process_and_save_event(self, event, cluster_id, dbstream_instance, split_merge_result):
        refined_event = self.process_event(event, cluster_id, dbstream_instance, split_merge_result)
        self.append_event_to_csv(refined_event)