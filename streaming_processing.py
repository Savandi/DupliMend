import time
import pandas as pd


df_event_log = pd.read_csv('C:/Users/drana/Downloads/Mine Log Abstract 2.csv', encoding='ISO-8859-1')


df_event_log['Timestamp'] = pd.to_datetime(df_event_log['Timestamp'])

df_event_log = df_event_log.sort_values(by='Timestamp')

def stream_event_log(df, delay=1):
    for _, event in df.iterrows():
        yield event
        time.sleep(delay)


for event in stream_event_log(df_event_log):
    print("Processing event: \n", event)
    # Online processing logic here