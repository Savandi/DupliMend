import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_log(version: int, num_cases=10, events_per_case=5):
    activities_v1 = ["Submit", "Review", "Approve", "Archive"]
    activities_v2 = ["Submit", "Review", "Approve", "Archive", "Finalize"]
    activities_v3 = ["Submit", "Submit", "Review", "Approve", "Archive", "Submit"]

    activity_versions = {
        1: activities_v1,
        2: activities_v2,
        3: activities_v3  # Version 3 introduces multiple contexts for "Submit"
    }

    activities = activity_versions.get(version, activities_v1)

    log = []
    for case_id in range(1, num_cases + 1):
        start_time = datetime(2023, 1, 1, 8, 0, 0)
        for event_id in range(events_per_case):
            activity = random.choice(activities)
            # Introduce homonyms (different contexts for "Submit")
            if activity == "Submit" and version == 3:
                # Randomly vary contextual features to mimic homonymy
                resource = random.choice(["UserA", "UserB", "System"])
                numeric_feature_1 = random.uniform(1, 10)
                numeric_feature_2 = random.uniform(1, 5)
            else:
                resource = random.choice(["UserA", "UserB", "UserC"])
                numeric_feature_1 = random.uniform(5, 15)
                numeric_feature_2 = random.uniform(10, 20)

            log.append({
                "EventID": len(log) + 1,
                "CaseID": case_id,
                "Timestamp": start_time + timedelta(minutes=event_id * 5),
                "Activity": activity,
                "Resource": resource,
                "NumericFeature_1": round(numeric_feature_1, 2),
                "NumericFeature_2": round(numeric_feature_2, 2)
            })

    return pd.DataFrame(log)


# Generate logs for three versions
log_v1 = generate_synthetic_log(version=1, num_cases=10, events_per_case=10)
log_v2 = generate_synthetic_log(version=2, num_cases=10, events_per_case=10)
log_v3 = generate_synthetic_log(version=3, num_cases=10, events_per_case=10)

# Save logs for reference
log_v1.to_csv("synthetic_log_v1.csv", index=False)
log_v2.to_csv("synthetic_log_v2.csv", index=False)
log_v3.to_csv("synthetic_log_v3.csv", index=False)

print("Logs generated and saved: synthetic_log_v1.csv, synthetic_log_v2.csv, synthetic_log_v3.csv")
