import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters for synthetic log generation
num_cases = 50
num_events = 300
start_time = datetime.now() - timedelta(days=30)

# Possible activity labels (with "Submit" as the homonymous label to be detected)
activity_labels = ["Submit", "Review", "Approve", "Archive", "Finalize"]

# Resources and categorical feature options
resources = ["User_1", "User_2", "User_3", "User_4"]
categorical_features_1 = ["Team_A", "Team_B", "Team_C"]
categorical_features_2 = ["High", "Medium", "Low"]

# Numerical feature ranges
numeric_feature_1_range = (1, 100)
numeric_feature_2_range = (0, 50)

# Generate synthetic log
event_log = []

def generate_case_id(case_number):
    return f"Case_{case_number}"

def generate_event_id(event_number):
    return f"E_{event_number}"

def generate_timestamp(base_time, event_idx):
    return base_time + timedelta(minutes=random.randint(1, 120) * event_idx)

for case_number in range(1, num_cases + 1):
    case_id = generate_case_id(case_number)
    num_events_in_case = random.randint(5, 10)
    base_time = start_time + timedelta(days=random.randint(0, 30))

    for event_idx in range(1, num_events_in_case + 1):
        event_id = generate_event_id(len(event_log) + 1)
        activity = random.choice(activity_labels)
        timestamp = generate_timestamp(base_time, event_idx)

        event_data = {
            "EventID": event_id,
            "CaseID": case_id,
            "Activity": activity,
            "Timestamp": timestamp,
            "Resource": random.choice(resources),
            "NumericFeature_1": random.randint(*numeric_feature_1_range),
            "NumericFeature_2": random.uniform(*numeric_feature_2_range),
            "CategoricalFeature_1": random.choice(categorical_features_1),
            "CategoricalFeature_2": random.choice(categorical_features_2),
        }

        event_log.append(event_data)

# Shuffle the log to simulate unordered streaming data
random.shuffle(event_log)

# Convert to DataFrame and save to CSV
event_log_df = pd.DataFrame(event_log)
event_log_df.sort_values(by="Timestamp", inplace=True)
event_log_df.to_csv("synthetic_log_with_homonyms.csv", index=False)

print("Synthetic log with homonymous labels generated: synthetic_log_with_homonyms.csv")
