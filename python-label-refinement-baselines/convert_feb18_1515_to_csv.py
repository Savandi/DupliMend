#!/usr/bin/env python3
"""
Convert feb18-1515 XES.gz files to CSV format for DupliMend evaluation
Extracts ground truth from OrgLabel attribute
"""

import os
import gzip
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer

# Paths
INPUT_DIR = "/mnt/c/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD/feb18-1515/logs"
OUTPUT_DIR = "/mnt/c/Users/drana/Downloads/baseline_csv_logs/feb18-1515"

def extract_xes_gz(xes_gz_path):
    """Extract .xes.gz to temporary .xes file"""
    xes_path = xes_gz_path.replace('.gz', '_temp')
    with gzip.open(xes_gz_path, 'rb') as f_in:
        with open(xes_path, 'wb') as f_out:
            f_out.write(f_in.read())
    return xes_path

def convert_xes_to_csv(xes_path, output_csv_path, output_gt_path):
    """
    Convert XES file to CSV format matching baseline structure

    CSV columns: EventID, CaseID, Activity, OrgLabel, Resource, Lifecycle, Timestamp
    Ground Truth columns: EventID, ground_truth_activity
    """
    print(f"  Loading XES file...")
    log = xes_importer.apply(xes_path)

    events = []
    ground_truth = []
    event_id = 1

    for case in log:
        case_id = case.attributes.get('concept:name', 'unknown')

        for event in case:
            # Extract attributes
            activity = event.get('concept:name', '')
            org_label = event.get('OrgLabel', activity)  # Ground truth
            resource = event.get('org:resource', 'artificial')
            lifecycle = event.get('lifecycle:transition', 'complete')
            timestamp = event.get('time:timestamp', event_id)

            # Add to events list
            events.append({
                'EventID': event_id,
                'CaseID': case_id,
                'Activity': activity,  # Observed (potentially duplicated)
                'OrgLabel': org_label,  # Keep for baselines
                'Resource': resource,
                'Lifecycle': lifecycle,
                'Timestamp': timestamp
            })

            # Add to ground truth list
            ground_truth.append({
                'EventID': event_id,
                'ground_truth_activity': org_label  # True label
            })

            event_id += 1

    # Convert to DataFrames
    df_events = pd.DataFrame(events)
    df_gt = pd.DataFrame(ground_truth)

    # Save to CSV
    df_events.to_csv(output_csv_path, index=False)
    df_gt.to_csv(output_gt_path, index=False)

    print(f"  ✅ Saved: {len(df_events)} events")
    print(f"     CSV: {output_csv_path}")
    print(f"     GT:  {output_gt_path}")

    return len(df_events)

def main():
    """Convert all _LogD_Sequence XES.gz files"""
    print("="*80)
    print("XES.gz to CSV Conversion - feb18-1515")
    print("="*80)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all _LogD_Sequence files
    log_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('_LogD_Sequence_feb18-1515.xes.gz')])

    print(f"Found {len(log_files)} log files to convert")
    print()

    total_events = 0
    event_counts = {}

    for i, log_file in enumerate(log_files, 1):
        log_name = log_file.replace('_LogD_Sequence_feb18-1515.xes.gz', '')
        print(f"[{i}/{len(log_files)}] Processing: {log_name}")

        # Paths
        xes_gz_path = os.path.join(INPUT_DIR, log_file)
        output_csv = os.path.join(OUTPUT_DIR, f"{log_name}_LogD_train.csv")
        output_gt = os.path.join(OUTPUT_DIR, f"{log_name}_ground_truth.csv")

        try:
            # Extract .gz
            print(f"  Extracting .xes.gz...")
            xes_path = extract_xes_gz(xes_gz_path)

            # Convert to CSV
            num_events = convert_xes_to_csv(xes_path, output_csv, output_gt)
            total_events += num_events
            event_counts[log_name] = num_events

            # Clean up temp file
            os.remove(xes_path)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

        print()

    # Summary
    print("="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"Total files converted: {len(event_counts)}")
    print(f"Total events: {total_events}")
    print()
    print("Event counts per log:")
    for log_name, count in sorted(event_counts.items()):
        print(f"  {log_name}: {count} events")
    print()
    print(f"Average events per log: {total_events / len(event_counts):.0f}")
    print(f"Min events: {min(event_counts.values())}")
    print(f"Max events: {max(event_counts.values())}")
    print()
    print(f"✅ All files saved to: {OUTPUT_DIR}")
    print()

    # Warmup recommendations
    avg_events = total_events / len(event_counts)
    print("WARMUP RECOMMENDATIONS:")
    if avg_events < 500:
        print(f"  Recommended warmup: 200-300 events (small logs)")
    elif avg_events < 2000:
        print(f"  Recommended warmup: 500-1000 events (medium logs)")
    else:
        print(f"  Recommended warmup: 1000-2000 events (large logs)")

if __name__ == "__main__":
    main()
