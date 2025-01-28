import pandas as pd
from collections import defaultdict

class LabelRefiner:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.cluster_mapping = defaultdict(lambda: defaultdict(int))  # Maps activity labels to their cluster-specific suffix mapping
        self.initialize_csv()

    def initialize_csv(self):
        """
        Create the CSV file with necessary headers if it does not already exist.
        """
        df = pd.DataFrame(columns=["EventID", "Activity", "Timestamp", "Resource", "CaseID", "refined_activity"])
        df.to_csv(self.output_file_path, index=False)

    def refine_label(self, event_label, cluster_id):
        """
        Generate a refined label with a cluster-specific suffix.
        :param event_label: The original activity label.
        :param cluster_id: The cluster ID associated with the event.
        :return: Refined activity label.
        """
        cluster_suffix = self.cluster_mapping[event_label]

        if cluster_id not in cluster_suffix:
            cluster_suffix[cluster_id] = len(cluster_suffix) + 1

        return f"{event_label}_{cluster_suffix[cluster_id]}"

    def process_event(self, event, cluster_id):
        """
        Refine the label of a single event.
        :param event: A dictionary representing a single event.
        :param cluster_id: Cluster ID assigned to this event.
        :return: Event with a refined label.
        """
        event_label = event.get("Activity")
        if not event_label:
            raise ValueError("Activity label is missing in the event.")

        refined_label = self.refine_label(event_label, cluster_id)
        event["refined_activity"] = refined_label
        return event

    def append_event_to_csv(self, event):
        """
        Append the refined event incrementally to the output CSV file.
        :param event: Refined event dictionary.
        """
        df = pd.DataFrame([event])
        df.to_csv(self.output_file_path, mode="a", header=False, index=False)

    def process_and_save_event(self, event, cluster_id):
        """
        Refine the event label and append it to the CSV file.
        :param event: Event dictionary.
        :param cluster_id: Cluster ID.
        """
        refined_event = self.process_event(event, cluster_id)
        self.append_event_to_csv(refined_event)
