import pandas as pd
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
    ["Submit", "Archive", "Review", "Submit", "Approve", "Submit", "Finalize"],
    ["Submit", "Archive", "Submit", "Review", "Submit", "Approve", "Finalize"],
    ["Submit", "Review", "Archive", "Submit", "Review", "Submit", "Approve"],
    ["Submit", "Review", "Archive", "Submit", "Approve", "Submit", "Finalize"],
    ["Submit", "Submit", "Review", "Submit", "Approve", "Submit", "Finalize"],
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

for case_number in range(1, num_cases + 1):
    case_id = generate_case_id(case_number)
    base_time = start_time + timedelta(days=random.randint(0, 30))
    variant = random.choice(trace_variants)

    timestamp = base_time  # ✅ Use base_time directly for the first event
    for activity in variant:
        event_data = {
            "EventID": generate_event_id(len(event_log) + 1),
            "CaseID": case_id,
            "Activity": activity,
            "Timestamp": timestamp,  # ✅ Will be converted to ISO format before saving
            "Resource": random.choice(resources),
            "Location": random.choice(locations),
            "Shift": random.choice(shifts),
            "Department": random.choice(departments),
            "ProcessingTime": random.randint(*processing_time_range),
            "PriorityLevel": random.randint(*priority_level_range),
            "FileSize": round(random.uniform(*file_size_range), 2),
        }

        event_log.append(event_data)

        # ✅ Update timestamp for the next event
        timestamp += timedelta(minutes=random.randint(1, 120))

# Sort first by CaseID, then by Timestamp
log_df = pd.DataFrame(event_log)
log_df.sort_values(by=["CaseID", "Timestamp"], inplace=True)

# ✅ Convert Timestamp to ISO format for CSV consistency
log_df["Timestamp"] = log_df["Timestamp"].apply(lambda x: x.isoformat())

# Save to CSV
log_df.to_csv("synthetic_log_with_homonyms.csv", index=False)

print("✅ Synthetic log with homonymous labels and control-flow variants generated: synthetic_log_with_homonyms.csv")
