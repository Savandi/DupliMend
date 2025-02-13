import pandas as pd
from collections import defaultdict
from src.homonym_mend.dynamic_feature_vector_construction import activity_feature_metadata

class LabelRefiner:
    def __init__(self, output_file_path, input_columns):
        """
        Initialize LabelRefiner with a dynamically provided list of input columns.
        """
        print(f"Label Refiner initialized")
        self.output_file_path = output_file_path
        self.cluster_mapping = defaultdict(lambda: defaultdict(int))
        self.input_columns = input_columns  # ✅ Store input columns
        self.initialize_csv()

    def initialize_csv(self):
        """
        Create the CSV file with headers retrieved from `main.py`.
        """
        try:
            # Use column names provided from main.py
            refined_df = pd.DataFrame(columns=self.input_columns)
            refined_df.to_csv(self.output_file_path, index=False)

            print(f"[INFO] Output CSV initialized with columns: {self.input_columns}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize output CSV: {str(e)}")

    def refine_label(self, event_label, cluster_id, dbstream_instance):
        """
        Generate a refined label but return existing mappings before assigning new ones.
        Ensures '_0', '_1' is only appended if multiple clusters exist.
        """
        cluster_suffix = self.cluster_mapping[event_label]

        # Retrieve existing cluster assignments for this activity label
        existing_clusters = activity_feature_metadata.get(event_label, {})

        active_clusters = dbstream_instance.get_micro_clusters()  # ✅ Fetch actual active clusters
        if len(active_clusters) <= 1:
            return event_label  # ✅ Do not append "_0" if only one cluster exist

        # Check if this cluster ID already has an assigned refined label
        if cluster_id in cluster_suffix:
            return f"{event_label}_{cluster_suffix[cluster_id]}"

        # Handle merges: If clusters reduce back to one, revert to the base label
        active_clusters = [cid for cid in existing_clusters if existing_clusters[cid]["frequency"] > 0]

        if len(active_clusters) == 1:
            return event_label  # ✅ Revert to base label when only one cluster remains

        # Use the most frequently seen cluster's suffix
        most_frequent_cluster = max(existing_clusters, key=lambda cid: existing_clusters[cid]["frequency"],
                                    default=cluster_id)

        if existing_clusters[most_frequent_cluster]["frequency"] > 2:
            cluster_suffix[cluster_id] = cluster_suffix[most_frequent_cluster]
            return f"{event_label}_{cluster_suffix[most_frequent_cluster]}"

        # ✅ Fix: Assign a new suffix **only if multiple clusters exist**
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
