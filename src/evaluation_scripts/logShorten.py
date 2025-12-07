import pandas as pd

# Load the CSV file
df = pd.read_csv("src/synthetic_logs/ipalia.csv", encoding='ISO-8859-1', parse_dates=['time:timestamp'])
# Sort by the Timestamp columnh
df_sorted = df.sort_values(by='time:timestamp')

# df_subset = df_sorted.drop(columns=["ground_truth_activity"])

# # Select the first 1000 rows
# df_subset = df_sorted.head(2000).copy()

# Add a new EventID column (starting from 1)
df_sorted['EventID'] = range(1, len(df_sorted) + 1)

# Output to a new CSV
df_sorted.to_csv("src/synthetic_logs/ipalia.csv", index=False)
                                                                                                    