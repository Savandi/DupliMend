import pandas as pd
from collections import defaultdict

from src.duplimend_framework.utils.directly_follows_graph import DirectlyFollowsGraph

directly_follows_graph = DirectlyFollowsGraph()
event_cluster_mapping = {}
cluster_event_mapping = {}
event_embedding_mapping = {}
activity_hash_mapping = {}

self_loop_cases = defaultdict(lambda: defaultdict(set))
case_context_freeze = defaultdict(dict)
case_activity_sequences = defaultdict(list)
activity_positions = defaultdict(lambda: defaultdict(list))  
pattern_frequencies = defaultdict(lambda: defaultdict(int))
activity_instance_counts = defaultdict(lambda: defaultdict(int))


per_case_directly_follows_graph = defaultdict(lambda: defaultdict(int))
case_last_seen_global = defaultdict(int)
last_activity_per_case = defaultdict(lambda: None)


def extract_temporal_features(timestamp):
    if not isinstance(timestamp, pd.Timestamp):
        try:
            timestamp = pd.to_datetime(timestamp, errors='coerce')
        except Exception:
            return {}

    if pd.isna(timestamp):
        return {}

    hour_bin = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    week_of_month = (timestamp.day - 1) // 7 + 1

    season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
    month = timestamp.month

    if month in [12, 1, 2]:
        season = 1
    elif month in [3, 4, 5]:
        season = 2
    elif month in [6, 7, 8]:
        season = 3
    else:
        season = 4

    temporal_features = {
        'hour_bin': hour_bin,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'week_of_month': week_of_month,
        'season': season,
        'month': month
    }

    return temporal_features
