import pandas as pd
from collections import defaultdict
from datetime import datetime
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_metadata

class LabelRefiner:
    def __init__(self, output_file_path):
        self.output_file_path = output_file_path
        self.cluster_mapping = defaultdict(lambda: defaultdict(int))  # Maps activity labels to refined cluster labels
        self.initialize_csv()

    def initialize_csv(self):
        """
        Create the CSV file with necessary headers if it does not already exist.
        """
        df = pd.DataFrame(columns=["EventID", "Activity", "Timestamp", "Resource", "CaseID", "refined_activity"])
        df.to_csv(self.output_file_path, index=False)

    def refine_label(self, event_label, cluster_id):
        """
        Generate a refined label but only apply suffixes if a split has been detected.
        """
        cluster_suffix = self.cluster_mapping[event_label]

        # Retrieve existing cluster assignments for this activity label
        existing_clusters = activity_feature_metadata.get(event_label, {})

        # If there is only one cluster (no split), return the activity label as-is
        if len(existing_clusters) <= 1:
            return event_label  # No suffix yet since a split hasn't occurred

        # If multiple clusters exist, refine the label
        if cluster_id in cluster_suffix:
            return f"{event_label}_{cluster_suffix[cluster_id]}"

        # Check if the new cluster is distinct enough from previous ones
        most_frequent_cluster = max(existing_clusters, key=lambda cid: existing_clusters[cid]["frequency"], default=cluster_id)
        if existing_clusters[most_frequent_cluster]["frequency"] > 2:  # If the split has happened, assign a suffix
            cluster_suffix[cluster_id] = cluster_suffix[most_frequent_cluster]
            return f"{event_label}_{cluster_suffix[most_frequent_cluster]}"

        # Otherwise, assign a new suffix since a confirmed split has happened
        cluster_suffix[cluster_id] = len(cluster_suffix)
        return f"{event_label}_{cluster_suffix[cluster_id]}"

    def process_event(self, event, cluster_id):
        """
        Refine the label of a single event, applying suffixes only if a split has occurred.
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
        """
        print(f"[DEBUG] Writing to CSV: {event}")  # Debugging
        df = pd.DataFrame([event])
        df.to_csv(self.output_file_path, mode="a", header=False, index=False)

    def process_and_save_event(self, event, cluster_id):
        """
        Refine the event label and append it to the CSV file.
        """
        refined_event = self.process_event(event, cluster_id)
        self.append_event_to_csv(refined_event)
