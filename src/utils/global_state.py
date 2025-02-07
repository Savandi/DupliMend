# global_state.py
from collections import defaultdict, deque

from src.utils.directly_follows_graph import DirectlyFollowsGraph

# Move Directly Follows Graph initialization here to prevent circular import
directly_follows_graph = DirectlyFollowsGraph()

activity_feature_history = defaultdict(list)
activity_feature_metadata = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "recency": None}))

# Track previous activities per case (for control-flow)
previous_events = defaultdict(lambda: deque(maxlen=3))

# Feature weights for adaptive scoring
feature_weights = defaultdict(lambda: 1.0)

# Historical tracking for resources
resource_usage_history = defaultdict(lambda: defaultdict(int))