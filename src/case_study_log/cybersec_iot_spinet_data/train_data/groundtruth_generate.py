import pandas as pd
import ast
import os

input_path = "49df5a5f-edd9-4467-b1d7-9bd4aed21e6b.csv"
df = pd.read_csv(input_path, dtype=str)

df["ground_truth_activity_label"] = (
    df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"] + "_" + df["SYSCALL_success"]
)

df["base_activity_label"] = (
    df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"]
)

def extract_filenames_preserve_order(path_str):
    try:
        paths = ast.literal_eval(path_str)
        return [os.path.basename(p).strip() for p in paths if isinstance(p, str)]
    except:
        return []

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

output_path = "ground_truth_event_log.csv"
final_df.to_csv(output_path, index=False)

print(f"Ground truth event log saved to: {output_path}")
