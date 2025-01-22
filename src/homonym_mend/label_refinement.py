import pandas as pd
from collections import defaultdict

class LabelRefiner:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.cluster_mapping = defaultdict(int)  # Maps activity labels to cluster IDs

    def refine_label(self, event_label, cluster_id):
        """Generate a refined label with a cluster-specific suffix."""
        return f"{event_label}_{cluster_id}"

    def process_event(self, event, cluster_id):
        """
        Refine the label of a single event.
        :param event: A dictionary representing a single event.
        :param cluster_id: Cluster ID assigned to this event.
        :return: Event with a refined label.
        """
        event_label = event["Activity"]
        refined_label = self.refine_label(event_label, cluster_id)
        event["refined_activity"] = refined_label
        return event

    def append_event_to_log(self, event):
        """
        Append the refined event incrementally to the output log.
        :param event: Refined event dictionary.
        """
        df = pd.DataFrame([event])
        with open(self.output_file_path, "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    def process_and_save_event(self, event, cluster_id):
        """
        Refine the event label and append it to the log file.
        :param event: Event dictionary.
        :param cluster_id: Cluster ID.
        """
        refined_event = self.process_event(event, cluster_id)
        self.append_event_to_log(refined_event)
