"""
XES to CSV Converter for Label Refinement Baseline Dataset

Converts XES.gz files from the BPM2016 dataset to CSV format for DupliMend.
Handles both ground truth logs and imprecise (LogD) logs.
"""

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import os
from pathlib import Path
import argparse


def convert_xes_to_csv(xes_path, output_csv_path, is_ground_truth=False):
    """
    Convert XES file to CSV format for DupliMend

    Args:
        xes_path: Path to XES or XES.GZ file
        output_csv_path: Path to save CSV file
        is_ground_truth: If True, only extract OrgLabel as ground_truth_activity
    """
    print(f"Reading XES file: {xes_path}")

    # Import XES log
    log = xes_importer.apply(str(xes_path))

    print(f"Loaded {len(log)} traces")

    # Convert to dataframe
    rows = []
    event_id = 0

    for trace_idx, trace in enumerate(log):
        case_id = trace.attributes.get('concept:name', f'case_{trace_idx}')

        for event in trace:
            event_id += 1

            # Get activity label
            activity = event.get('concept:name', 'UNKNOWN')
            org_label = event.get('OrgLabel', activity)  # Ground truth label
            resource = event.get('org:resource', 'artificial')
            lifecycle = event.get('lifecycle:transition', 'complete')

            row = {
                'EventID': event_id,
                'CaseID': case_id,
                'Activity': activity,  # Imprecise label in LogD files
                'OrgLabel': org_label,  # Ground truth
                'Resource': resource,
                'Lifecycle': lifecycle,
                'Timestamp': event_id  # Use event order as timestamp since no real time
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # If ground truth file, create simplified version with just EventID and ground truth
    if is_ground_truth:
        df_gt = df[['EventID', 'OrgLabel']].copy()
        df_gt.rename(columns={'OrgLabel': 'ground_truth_activity'}, inplace=True)
        df_gt.to_csv(output_csv_path, index=False)
        print(f"Saved ground truth CSV: {output_csv_path}")
        print(f"  Columns: {list(df_gt.columns)}")
        print(f"  Events: {len(df_gt)}")
    else:
        # For training/test logs, save full CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Saved CSV: {output_csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Events: {len(df)}")
        print(f"  Cases: {df['CaseID'].nunique()}")
        print(f"  Unique activities: {df['Activity'].nunique()}")
        print(f"  Activity distribution:")
        print(df['Activity'].value_counts())

    return df


def convert_folder(folder_path, output_folder, log_prefix="A_1", convert_all=False):
    """
    Convert logs from a folder (e.g., feb16-1625)

    Args:
        folder_path: Path to folder containing logs/ subdirectory
        output_folder: Where to save CSV files
        log_prefix: Which log to convert (e.g., "A_1", "B_1")
        convert_all: If True, convert all logs A-Q
    """
    logs_dir = Path(folder_path) / "logs"
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if convert_all:
        # Convert all logs A through Q
        log_prefixes = [f"{letter}_1" for letter in "ABCDEFGHIJKLMNOPQ"]
    else:
        log_prefixes = [log_prefix]

    for prefix in log_prefixes:
        print(f"\n{'='*60}")
        print(f"Converting log: {prefix}")
        print(f"{'='*60}")

        # Convert LogD (imprecise training data)
        # NOTE: LogD already contains ground truth in OrgLabel column, so separate GT file is optional
        logd_path = logs_dir / f"{prefix}_LogD_Sequence_feb16-1625.xes.gz"
        if logd_path.exists():
            logd_csv = output_path / f"{prefix}_LogD_train.csv"
            convert_xes_to_csv(logd_path, logd_csv, is_ground_truth=False)
            print(f"  → LogD CSV contains both Activity (imprecise) and OrgLabel (ground truth)")
        else:
            print(f"Warning: LogD file not found: {logd_path}")

        # Convert ground truth log (OPTIONAL - LogD already has OrgLabel with ground truth)
        # This creates a standalone ground truth file from the perfectly labeled Log.xes
        # Optional: creates standalone ground truth file from perfectly labeled Log.xes
        log_path = logs_dir / f"{prefix}_Log.xes.gz"
        if log_path.exists():
            log_csv = output_path / f"{prefix}_ground_truth.csv"
            convert_xes_to_csv(log_path, log_csv, is_ground_truth=True)
            print(f"  → Created separate ground truth file (optional - LogD already has OrgLabel)")
        else:
            print(f"Note: Separate ground truth log not found (optional): {log_path}")

    print(f"\n{'='*60}")
    print(f"Conversion complete! Files saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert XES logs to CSV for DupliMend')
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to folder containing logs/ directory (e.g., feb16-1625)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder for CSV files')
    parser.add_argument('--log', type=str, default='A_1',
                        help='Log prefix to convert (default: A_1)')
    parser.add_argument('--all', action='store_true',
                        help='Convert all logs A through Q')

    args = parser.parse_args()

    convert_folder(args.folder, args.output, args.log, args.all)
