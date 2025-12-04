import pandas as pd
import ast
import os
from pathlib import Path

def extract_filenames_preserve_order(path_str):
    try:
        paths = ast.literal_eval(path_str)
        filenames = [os.path.basename(p).strip() for p in paths if isinstance(p, str)]
        return filenames
    except:
        return []

def transform_event_log(input_file, output_file):
    df = pd.read_csv(input_file, dtype=str)
    # Convert SYSCALL_exit to numeric (supports +ve and -ve), invalids become NaN
    df["SYSCALL_exit"] = pd.to_numeric(df["SYSCALL_exit"], errors="coerce")

    # Drop rows with missing essential values
    df.dropna(subset=["SYSCALL_syscall", "PROCESS_uid", "SYSCALL_exit"], inplace=True)

    # Fill optional list fields if null
    df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].fillna("[]")
    df["CUSTOM_libs"] = df["CUSTOM_libs"].fillna("[]")

    # Drop rows that are entirely empty
    df.dropna(how='all', inplace=True)

    # Remove duplicated rows (exact matches)
    df.drop_duplicates(inplace=True)

    filename_prefix = Path(input_file).stem
    df["SYSCALL_pid"] = filename_prefix + "_" + df["SYSCALL_pid"]

    df["activity_label"] = df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"]

    df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].apply(extract_filenames_preserve_order)
    df["CUSTOM_libs"] = df["CUSTOM_libs"].apply(extract_filenames_preserve_order)

    selected_columns = [
        "SYSCALL_pid",
        "SYSCALL_timestamp",
        "SYSCALL_syscall",
        "PROCESS_uid",
        "SYSCALL_exit",
        "CUSTOM_openFiles",
        "SYSCALL_exit_hint",
        "PROCESS_comm",
        "CUSTOM_libs",
        "activity_label"
    ]

    transformed_df = df[selected_columns].copy()
    transformed_df.drop(columns=["SYSCALL_syscall", "PROCESS_uid"], inplace=True)
    transformed_df["EventID"] = range(1, len(transformed_df) + 1)

    transformed_df.to_csv(output_file, index=False)
    print(f"Transformed file saved to: {output_file}")

def batch_process_event_logs():
    input_path = Path(r"C:\Users\drana\Downloads\cybersec_iot_spinet_data\train_data")
    output_path = Path(r"C:\Users\drana\Downloads\cybersec_iot_spinet_data\processed_train_data")
    output_path.mkdir(parents=True, exist_ok=True)

    for csv_file in input_path.glob("*.csv"):
        output_file = output_path / f"transformed_{csv_file.name}"
        transform_event_log(csv_file, output_file)

if __name__ == "__main__":
    batch_process_event_logs()
