import time
import pandas as pd

# Load the CSV file
# Assuming the CSV has columns like 'case_id', 'activity', 'timestamp', etc.
df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')


# Convert the 'timestamp' column to datetime format (if not already in datetime)
df_event_log['Timestamp'] = pd.to_datetime(df_event_log['Timestamp'])

# Sort the DataFrame by the 'timestamp' column
df_event_log = df_event_log.sort_values(by='Timestamp')

# Define a function to stream events one by one in timestamp order
def stream_event_log(df, delay=1):
    for _, event in df.iterrows():
        yield event  # Yield each event row as a dictionary-like Series
        time.sleep(delay)  # Simulate real-time streaming with a delay

# Process each event in the sorted, timestamp-ordered stream
for event in stream_event_log(df_event_log):
    print("Processing event: \n", event)
    # Insert your online processing logic here