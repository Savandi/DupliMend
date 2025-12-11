import pm4py
import pandas as pd
import numpy as np
import random
import datetime
from collections import defaultdict

random.seed(42)
np.random.seed(42)


def generate_trace(case_id, start_time, activity_durations):
    events = []

    current_time = start_time
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_start'
    })

    current_time += datetime.timedelta(minutes=random.randint(*activity_durations['A']))

    c_start_time = current_time
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'C',
        'time:timestamp': c_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'C'
    })
    c_end_time = c_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['C']))

    d_start_time = current_time + datetime.timedelta(minutes=random.randint(1, 5))
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'D',
        'time:timestamp': d_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'D'
    })
    d_end_time = d_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['D']))

    b1_start_time = current_time + datetime.timedelta(minutes=random.randint(2, 7))
    events.append({
        'case:concept:name': case_id,
        'concept:name': 'B1',
        'time:timestamp': b1_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'B1'
    })

    b1_end_time = b1_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['B1']))
    b2_start_time = b1_end_time

    events.append({
        'case:concept:name': case_id,
        'concept:name': 'B2',
        'time:timestamp': b2_start_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'B2'
    })

    b2_end_time = b2_start_time + datetime.timedelta(minutes=random.randint(*activity_durations['B2']))

    current_time = max(c_end_time, d_end_time, b2_end_time)

    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_middle'
    })

    current_time += datetime.timedelta(minutes=random.randint(*activity_durations['A']))

    if random.random() < 0.5:
        events.append({
            'case:concept:name': case_id,
            'concept:name': 'F',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'F'
        })
        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['F']))
    else:
        events.append({
            'case:concept:name': case_id,
            'concept:name': 'E1',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'E1'
        })

        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['E1']))

        events.append({
            'case:concept:name': case_id,
            'concept:name': 'E2',
            'time:timestamp': current_time,
            'lifecycle:transition': 'complete',
            'ground_truth': 'E2'
        })

        current_time += datetime.timedelta(minutes=random.randint(*activity_durations['E2']))

    events.append({
        'case:concept:name': case_id,
        'concept:name': 'A',
        'time:timestamp': current_time,
        'lifecycle:transition': 'complete',
        'ground_truth': 'A_end'
    })

    return events


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

all_events = []
start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)

case_starts = []
for i in range(990):
    start_time = start_date + datetime.timedelta(
        minutes=random.randint(0, 30 * 24 * 60)
    )
    case_starts.append((f"Case_{i+1}", start_time))

case_starts.sort(key=lambda x: x[1])

for case_id, start_time in case_starts:
    trace_events = generate_trace(case_id, start_time, activity_durations)
    all_events.extend(trace_events)

df = pd.DataFrame(all_events)
df = df.sort_values('time:timestamp')

print(f"Total number of events: {len(df)}")
print(f"Total number of traces: {df['case:concept:name'].nunique()}")

event_log = pm4py.format_dataframe(
    df,
    case_id='case:concept:name',
    activity_key='concept:name',
    timestamp_key='time:timestamp'
)
pm4py.write_xes(event_log, 'bpmn_event_log.xes')

df.to_csv('bpmn_event_log.csv', index=False)

ground_truth_df = df.copy()
ground_truth_df['concept:name'] = ground_truth_df['ground_truth']

ground_truth_log = pm4py.format_dataframe(
    ground_truth_df,
    case_id='case:concept:name',
    activity_key='concept:name',
    timestamp_key='time:timestamp'
)

pm4py.write_xes(ground_truth_log, 'ground_truth_event_log.xes')

print("Event logs generated successfully!")

try:
    discovered_net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
    pm4py.save_vis_petri_net(discovered_net, initial_marking, final_marking, 'discovered_model.png')
    print("Process model visualization saved as 'discovered_model.png'")
except Exception as e:
    print(f"Couldn't generate visualization: {e}")
