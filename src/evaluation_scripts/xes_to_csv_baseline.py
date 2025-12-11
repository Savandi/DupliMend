import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import os
from pathlib import Path
import argparse


def convert_xes_to_csv(xes_path, output_csv_path, is_ground_truth=False):
    print(f"Reading XES file: {xes_path}")

    log = xes_importer.apply(str(xes_path))

    print(f"Loaded {len(log)} traces")

    rows = []
    event_id = 0

    for trace_idx, trace in enumerate(log):
        case_id = trace.attributes.get('concept:name', f'case_{trace_idx}')

        for event in trace:
            event_id += 1

            activity = event.get('concept:name', 'UNKNOWN')
            org_label = event.get('OrgLabel', activity)
            resource = event.get('org:resource', 'artificial')
            lifecycle = event.get('lifecycle:transition', 'complete')

            row = {
                'EventID': event_id,
                'CaseID': case_id,
                'Activity': activity,
                'OrgLabel': org_label,
                'Resource': resource,
                'Lifecycle': lifecycle,
                'Timestamp': event_id
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if is_ground_truth:
        df_gt = df[['EventID', 'OrgLabel']].copy()
        df_gt.rename(columns={'OrgLabel': 'ground_truth_activity'}, inplace=True)
        df_gt.to_csv(output_csv_path, index=False)
        print(f"Saved ground truth CSV: {output_csv_path}")
        print(f"  Columns: {list(df_gt.columns)}")
        print(f"  Events: {len(df_gt)}")
    else:
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
    logs_dir = Path(folder_path) / "logs"
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if convert_all:
        log_prefixes = [f"{letter}_1" for letter in "ABCDEFGHIJKLMNOPQ"]
    else:
        log_prefixes = [log_prefix]

    for prefix in log_prefixes:
        print(f"\n{'='*60}")
        print(f"Converting log: {prefix}")
        print(f"{'='*60}")

        logd_path = logs_dir / f"{prefix}_LogD_Sequence_feb16-1625.xes.gz"
        if logd_path.exists():
            logd_csv = output_path / f"{prefix}_LogD_train.csv"
            convert_xes_to_csv(logd_path, logd_csv, is_ground_truth=False)
            print(f"  -> LogD CSV contains both Activity (imprecise) and OrgLabel (ground truth)")
        else:
            print(f"Warning: LogD file not found: {logd_path}")

        log_path = logs_dir / f"{prefix}_Log.xes.gz"
        if log_path.exists():
            log_csv = output_path / f"{prefix}_ground_truth.csv"
            convert_xes_to_csv(log_path, log_csv, is_ground_truth=True)
            print(f"  -> Created separate ground truth file (optional - LogD already has OrgLabel)")
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
