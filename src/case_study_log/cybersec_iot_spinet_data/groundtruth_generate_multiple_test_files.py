import pandas as pd
import ast
import os
from pathlib import Path

def extract_filenames_preserve_order(path_str):
    try:
        paths = ast.literal_eval(path_str)
        return [os.path.basename(p).strip() for p in paths if isinstance(p, str)]
    except:
        return []

def transform_ground_truth_log(input_file, output_file):
    df = pd.read_csv(input_file, dtype=str)
    df["SYSCALL_exit"] = pd.to_numeric(df["SYSCALL_exit"], errors="coerce")

    df.dropna(subset=["SYSCALL_syscall", "PROCESS_uid", "SYSCALL_exit"], inplace=True)

    df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].fillna("[]")
    df["CUSTOM_libs"] = df["CUSTOM_libs"].fillna("[]")

    df.dropna(how='all', inplace=True)

    df.drop_duplicates(inplace=True)

    filename_prefix = Path(input_file).stem
    df["SYSCALL_pid"] = filename_prefix + "_" + df["SYSCALL_pid"]

    df["ground_truth_activity_label"] = (
        df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"] + "_" + df["SYSCALL_success"]
    )

    df["base_activity_label"] = (
        df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"]
    )

    df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].apply(extract_filenames_preserve_order)
    df["CUSTOM_libs"] = df["CUSTOM_libs"].apply(extract_filenames_preserve_order)

    selected_columns = [
        "SYSCALL_pid",
        "SYSCALL_timestamp",
        "SYSCALL_syscall",
        "PROCESS_uid",
        "SYSCALL_success",
        "SYSCALL_exit",
        "SYSCALL_exit_hint",
        "PROCESS_comm",
        "CUSTOM_openFiles",
        "CUSTOM_libs",
        "base_activity_label",
        "ground_truth_activity_label"
    ]

    final_df = df[selected_columns].copy()
    final_df["EventID"] = range(1, len(final_df) + 1)

    final_df.to_csv(output_file, index=False)
    print(f"Transformed file saved to: {output_file}")

def batch_process_test_logs():
    input_path = Path(r"C:\Users\drana\Downloads\cybersec_iot_spinet_data\test_data")
    output_path = Path(r"C:\Users\drana\Downloads\cybersec_iot_spinet_data\processed_groundtruth_test_data")
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        output_file = output_path / f"transformed_{csv_file.name}"
        try:
            transform_ground_truth_log(csv_file, output_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    batch_process_test_logs()