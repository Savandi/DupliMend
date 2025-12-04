import pandas as pd
import ast
import os

# Load original data with all columns as strings
input_path = "49df5a5f-edd9-4467-b1d7-9bd4aed21e6b.csv"
df = pd.read_csv(input_path, dtype=str)

# Create ground truth activity label: SYSCALL_syscall_PROCESS_uid_SYSCALL_success
df["ground_truth_activity_label"] = (
    df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"] + "_" + df["SYSCALL_success"]
)

# Create base activity label: SYSCALL_syscall_PROCESS_uid
df["base_activity_label"] = (
    df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"]
)

# Extract just filenames from list-based paths, preserving original order
def extract_filenames_preserve_order(path_str):
    try:
        paths = ast.literal_eval(path_str)
        return [os.path.basename(p).strip() for p in paths if isinstance(p, str)]
    except:
        return []

df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].apply(extract_filenames_preserve_order)
df["CUSTOM_libs"] = df["CUSTOM_libs"].apply(extract_filenames_preserve_order)

# Select relevant columns for saving
selected_columns = [
    "SYSCALL_pid",             # Case ID
    "SYSCALL_timestamp",       # Timestamp
    "SYSCALL_syscall",
    "PROCESS_uid",
    "SYSCALL_success",         # Used in ground truth
    "SYSCALL_exit",            # Feature
    "SYSCALL_exit_hint",       # Feature
    "PROCESS_comm",            # Feature
    "CUSTOM_openFiles",        # Preprocessed
    "CUSTOM_libs",             # Preprocessed
    "base_activity_label",     # What the model will see
    "ground_truth_activity_label"  # For evaluation
]

# Subset the DataFrame
final_df = df[selected_columns].copy()

# Add EventID
final_df["EventID"] = range(1, len(final_df) + 1)

# Save
output_path = "ground_truth_event_log.csv"
final_df.to_csv(output_path, index=False)

print(f"Ground truth event log saved to: {output_path}")
