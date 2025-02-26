# global_state.py
from collections import defaultdict, deque
import pandas as pd
from src.utils.directly_follows_graph import DirectlyFollowsGraph

# Move Directly Follows Graph initialization here to prevent circular import
directly_follows_graph = DirectlyFollowsGraph()
# Tracks historical feature vectors per activity
activity_feature_history = defaultdict(list)
# Tracks feature metadata: frequency & recency
activity_feature_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": None}))
# Track per-activity event counts
activity_event_counters = defaultdict(int)
# Tracks feature importance dynamically
feature_relevance_tracker = defaultdict(float)

# Track previous activities per case (for control-flow)
previous_events = defaultdict(lambda: deque(maxlen=3))

# Feature weights for adaptive scoring
feature_weights = defaultdict(lambda: 1.0)

# Historical tracking for resources
resource_usage_history = defaultdict(lambda: defaultdict(int))

dbstream_clusters = {}

def extract_temporal_features(timestamp):
    if not isinstance(timestamp, pd.Timestamp):
        try:
            timestamp = pd.to_datetime(timestamp, errors='coerce')
        except Exception as e:
            print(f"[ERROR] Failed to convert timestamp: {timestamp}, Error: {e}")
            return {}

    if pd.isna(timestamp):
        print(f"[ERROR] NaT detected in timestamp: {timestamp}")
        return {}

    if not isinstance(timestamp, pd.Timestamp):
        print(f"[ERROR] Unexpected type for timestamp: {type(timestamp)}")
        return {}

    hour = timestamp.hour
    hour_bin = (
        "Early_Morning" if 4 <= hour < 8 else
        "Morning" if 8 <= hour < 12 else
        "Afternoon" if 12 <= hour < 16 else
        "Late_Afternoon" if 16 <= hour < 20 else
        "Night"
    )

    temporal_features = {
        'hour_bin': hour_bin,
        'day_of_week': timestamp.weekday(),
        'is_weekend': timestamp.weekday() >= 5,
        'week_of_month': (timestamp.day - 1) // 7 + 1,
        'season': (
            "Winter" if timestamp.month in [12, 1, 2]
            else "Spring" if timestamp.month in [3, 4, 5]
            else "Summer" if timestamp.month in [6, 7, 8]
            else "Fall"
        ),
        'month': timestamp.month
    }

    # Add debug printing
    print("[DEBUG] Temporal features extracted:")
    for key, value in temporal_features.items():
        print(f"  {key}: {value}")

    return temporal_features