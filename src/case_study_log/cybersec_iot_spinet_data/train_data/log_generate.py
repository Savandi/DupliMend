import pandas as pd
import ast
import os

# Load original data, forcing all columns to be string type
input_path = "49df5a5f-edd9-4467-b1d7-9bd4aed21e6b.csv"
df = pd.read_csv(input_path, dtype=str)

# Create activity label using only SYSCALL_syscall and PROCESS_uid
df["activity_label"] = (
    df["SYSCALL_syscall"] + "_" + df["PROCESS_uid"]
)

# Helper function to extract filenames (preserve original list order, no sorting)
def extract_filenames_preserve_order(path_str):
    try:
        paths = ast.literal_eval(path_str)
        filenames = [os.path.basename(p).strip() for p in paths if isinstance(p, str)]
        return filenames
    except:
        return []

# Apply filename extraction to list-based columns
df["CUSTOM_openFiles"] = df["CUSTOM_openFiles"].apply(extract_filenames_preserve_order)
df["CUSTOM_libs"] = df["CUSTOM_libs"].apply(extract_filenames_preserve_order)

# Select columns to include before final drop
selected_columns = [
    "SYSCALL_pid",            # Case ID
    "SYSCALL_timestamp",      # Timestamp
    "SYSCALL_syscall",        # (temporary, dropped before save)
    "PROCESS_uid",            # (temporary, dropped before save)
    "SYSCALL_exit",           # Discriminative feature
    "CUSTOM_openFiles",       # Filenames, order preserved
    "SYSCALL_exit_hint",      # Discriminative feature
    "PROCESS_comm",           # Discriminative feature
    "CUSTOM_libs",            # Filenames, order preserved
    "activity_label"          # Final label (syscall + uid)
]

# Subset and make a copy before dropping columns
transformed_df = df[selected_columns].copy()

# Drop SYSCALL_syscall and PROCESS_uid before saving
transformed_df.drop(columns=["SYSCALL_syscall", "PROCESS_uid"], inplace=True)

# Add EventID column
transformed_df["EventID"] = range(1, len(transformed_df) + 1)

# Save to CSV
output_path = "transformed_event_log.csv"
transformed_df.to_csv(output_path, index=False)

print(f"Final transformed event log saved to: {output_path}")
