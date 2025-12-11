import pandas as pd

df = pd.read_csv("src/synthetic_logs/ipalia.csv", encoding='ISO-8859-1', parse_dates=['time:timestamp'])
df_sorted = df.sort_values(by='time:timestamp')

df_sorted['EventID'] = range(1, len(df_sorted) + 1)

df_sorted.to_csv("src/synthetic_logs/ipalia.csv", index=False)
