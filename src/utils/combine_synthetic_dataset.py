from datetime import timedelta

import pandas as pd
import os

# Check current working directory
print("Current Working Directory:", os.getcwd())

# List files in the directory to verify existence
file_list = os.listdir("C:/Users/drana/Documents/GitHub/HomonymMend/src/homonym_mend/")
print("Files in directory:", file_list)

# Change to the correct directory
os.chdir("C:/Users/drana/Documents/GitHub/HomonymMend/src/homonym_mend/")

# Load the three versions of logs
try:
    log_v1 = pd.read_csv("../homonym_mend/synthetic_log_v1.csv")
    log_v2 = pd.read_csv("../homonym_mend/synthetic_log_v2.csv")
    log_v3 = pd.read_csv("../homonym_mend/synthetic_log_v3.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Add a 'Version' column to distinguish logs
log_v1["Version"] = 1
log_v2["Version"] = 2
log_v3["Version"] = 3

# Combine logs into a single dataframe
combined_log = pd.concat([log_v1, log_v2, log_v3], ignore_index=True)

# Ensure homonymous labels appear from the start
combined_log.loc[:4, "Activity"] = "Submit"
combined_log.loc[:4, "Resource"] = ["UserA", "UserB", "System", "UserC", "UserA"]
combined_log.loc[:4, "NumericFeature_1"] = [5.5, 8.2, 12.3, 4.1, 6.7]
combined_log.loc[:4, "NumericFeature_2"] = [3.1, 2.9, 1.2, 4.4, 3.3]
# Ensure unique EventIDs
combined_log['EventID'] = range(1, len(combined_log) + 1)

# Ensure unique timestamps
combined_log['Timestamp'] = pd.to_datetime(combined_log['Timestamp'])  # Convert to datetime if not already
df = combined_log.sort_values('Timestamp')  # Sort by Timestamp
df['Timestamp'] = [df['Timestamp'].iloc[0] + timedelta(seconds=i) for i in range(len(df))]

# Save the updated dataset
output_path = '../homonym_mend/updated_combined_synthetic_log.csv'
df.to_csv(output_path, index=False, encoding='ISO-8859-1')


print("Combined synthetic log generated and saved as 'updated_combined_synthetic_log.csv'.")
