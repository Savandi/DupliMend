import pm4py
import pandas as pd
import numpy as np
import random
import datetime
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to generate a single trace following the BPMN structure
def generate_trace(case_id, start_time, activity_durations):
    events = []
    
    # Start with activity A
    current_time = start_time
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_start' 
    })
    
    current_time += datetime.timedelta(minutes=random.randint(*activity_durations['A']))
    
    # Parallel split to activities C, D, and B1
    # For parallel flows, we'll process them one by one, but their execution can overlap in time
    
    # Flow 1: C
    c_start_time = current_time
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'C',
        'time:timestamp': c_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'C'  # Ground truth label
    })
    c_end_time = c_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['C']))
    
    # Flow 2: D
    d_start_time = current_time + datetime.timedelta(minutes=random.randint(1, 5))  # Small delay for realistic interleaving
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'D',
        'time:timestamp': d_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'D'  # Ground truth label
    })
    d_end_time = d_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['D']))
    
    # Flow 3: B1 -> B2
    b1_start_time = current_time + datetime.timedelta(minutes=random.randint(2, 7))  # More delay for realistic interleaving
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'B1',
        'time:timestamp': b1_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'B1'  # Ground truth label
    })
    
    b1_end_time = b1_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['B1']))
    b2_start_time = b1_end_time
    
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'B2',
        'time:timestamp': b2_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'B2'  # Ground truth label
    })
    
    b2_end_time = b2_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['B2']))
    
    # Find the maximum end time from parallel paths
    current_time = max(c_end_time, d_end_time, b2_end_time)
    
    # After parallel merge, continue with A again
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_middle'  # Ground truth label
    })
    
    current_time += datetime.timedelta(minutes=random.randint(*activity_durations['A']))
    
    # Exclusive gateway - either path F or path E1->E2
    if random.random() < 0.5:  # 50% chance to take either path
        # Path F
        events.append({
            'case:concept:name': case_id,
            'concept:name': 'F',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'F'  # Ground truth label
        })
        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['F']))
    else:
        # Path E1 -> E2
        events.append({
            'case:concept:name': case_id,
            'concept:name': 'E1',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'E1'  # Ground truth label
        })
        
        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['E1']))
        
        events.append({
            'case:concept:name': case_id,
            'concept:name': 'E2',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'E2'  # Ground truth label
        })
        
        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['E2']))
    
    # Final activity A
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_end'  # Ground truth label
    })
    
    return events

# Define activity durations (in minutes) as (min, max) ranges
activity_durations = {
    'A': (10, 30),
    'B1': (15, 45),
    'B2': (20, 60),
    'C': (30, 90),
    'D': (25, 75),
    'E1': (20, 50),
    'E2': (15, 40),
    'F': (35, 80)
}

# Generate 990 traces
all_events = []
start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)

# We'll generate case start times over a period to ensure interleaving
case_starts = []
for i in range(990):
    # Distribute case starts over a 30-day period
    start_time = start_date + datetime.timedelta(
        minutes=random.randint(0, 30 * 24 * 60)  # Random minutes within 30 days
    )
    case_starts.append((f"Case_{i+1}", start_time))

# Sort by start time to help with interleaving
case_starts.sort(key=lambda x: x[1])

# Generate traces
for case_id, start_time in case_starts:
    trace_events = generate_trace(case_id, start_time, activity_durations)
    all_events.extend(trace_events)

# Create DataFrame from all events
df = pd.DataFrame(all_events)

# Sort all events by timestamp to ensure proper interleaving
df = df.sort_values('time:timestamp')

# Check the event count (should be close to 8415)
print(f"Total number of events: {len(df)}")
print(f"Total number of traces: {df['case:concept:name'].nunique()}")

# Save the event log in XES format
event_log = pm4py.format_dataframe(
    df,
    case_id='case:concept:name',
    activity_key='concept:name',
    timestamp_key='time:timestamp'
)
pm4py.write_xes(event_log, 'bpmn_event_log.xes')

# Also save as CSV for easier inspection
df.to_csv('bpmn_event_log.csv', index=False)

# Generate ground truth model (simplified version of the original BPMN)
# This is optional but can be useful for conformance checking
ground_truth_df = df.copy()
ground_truth_df['concept:name'] = ground_truth_df['ground_truth']  # Use ground truth labels as activities

ground_truth_log = pm4py.format_dataframe(
    ground_truth_df,
    case_id='case:concept:name',
    activity_key='concept:name',
    timestamp_key='time:timestamp'
)

# Save ground truth log
pm4py.write_xes(ground_truth_log, 'ground_truth_event_log.xes')

print("Event logs generated successfully!")

# Option to visualize the process model from the generated log
try:
    # Discover process model from event log
    discovered_net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
    
    # Save the discovered model as an image
    pm4py.save_vis_petri_net(discovered_net, initial_marking, final_marking, 'discovered_model.png')
    print("Process model visualization saved as 'discovered_model.png'")
except Exception as e:
    print(f"Couldn't generate visualization: {e}")