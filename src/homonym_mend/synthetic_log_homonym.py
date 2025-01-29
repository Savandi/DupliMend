import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters for synthetic log generation
num_cases = 200  # Ensuring a larger number of cases
num_events = 1000  # At least 1000 events
start_time = datetime.now() - timedelta(days=30)

# Possible activity labels
activity_labels = ["Submit", "Review", "Approve", "Archive", "Finalize"]

# Resources, locations, shifts, and departments
resources = ["User_1", "User_2", "User_3", "User_4"]
locations = ["Branch_A", "Branch_B", "Branch_C", "Headquarters"]
shifts = ["Morning", "Afternoon", "Night"]
departments = ["HR", "Finance", "IT", "Operations"]

# Numerical feature ranges
processing_time_range = (1, 120)  # Processing time in minutes
priority_level_range = (1, 5)  # Priority levels (1: Low, 5: High)
file_size_range = (0.1, 10.0)  # File size in MB

# Define trace variants
trace_variants = [
    ["Submit", "Archive", "Review", "Submit", "Approve", "Finalize"],
    ["Submit", "Archive", "Review", "Submit", "Approve"],
    ["Submit", "Review", "Archive", "Submit", "Approve"],
    ["Submit", "Review", "Archive", "Submit", "Approve", "Finalize"]
]

# Generate synthetic log
event_log = []


def generate_case_id(case_number):
    return f"Case_{case_number}"


def generate_event_id(event_number):
    return f"E_{event_number}"


def generate_timestamp(base_time, previous_timestamp=None):
    increment_minutes = random.randint(1, 120)
    return previous_timestamp + timedelta(minutes=increment_minutes) if previous_timestamp else base_time


def extract_temporal_features(timestamp):
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'season': (
            "Winter" if timestamp.month in [12, 1, 2]
            else "Spring" if timestamp.month in [3, 4, 5]
            else "Summer" if timestamp.month in [6, 7, 8]
            else "Fall"
        )
    }


for case_number in range(1, num_cases + 1):
    case_id = generate_case_id(case_number)
    base_time = start_time + timedelta(days=random.randint(0, 30))
    previous_timestamp = None
    variant = random.choice(trace_variants)

    for activity in variant:
        timestamp = generate_timestamp(base_time, previous_timestamp)
        previous_timestamp = timestamp
        temporal_features = extract_temporal_features(timestamp)

        event_data = {
            "EventID": generate_event_id(len(event_log) + 1),
            "CaseID": case_id,
            "Activity": activity,
            "Timestamp": timestamp,
            "Resource": random.choice(resources),
            "Location": random.choice(locations),
            "Shift": random.choice(shifts),
            "Department": random.choice(departments),
            "ProcessingTime": random.randint(*processing_time_range),
            "PriorityLevel": random.randint(*priority_level_range),
            "FileSize": round(random.uniform(*file_size_range), 2),
        }

        event_data.update(temporal_features)
        event_log.append(event_data)

# Shuffle to simulate streaming data
random.shuffle(event_log)

# Convert to DataFrame and save
log_df = pd.DataFrame(event_log)
log_df.sort_values(by="Timestamp", inplace=True)
log_df.to_csv("synthetic_log_with_homonyms.csv", index=False)

print("Synthetic log with homonymous labels and control-flow variants generated: synthetic_log_with_homonyms.csv")
