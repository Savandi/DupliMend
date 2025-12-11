import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_vis

df = pd.read_csv("generated_event_log_homonyms_interleaved_groundtruth.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

df = df.rename(columns={
    "CaseID": "case:concept:name",
    "Activity": "concept:name",
    "Timestamp": "time:timestamp"
})
df = dataframe_utils.convert_timestamp_columns_in_df(df)
log = log_converter.apply(df)

dfg = dfg_discovery.apply(log)

gviz = dfg_vis.apply(dfg, log=log, variant=dfg_vis.Variants.FREQUENCY)
dfg_vis.view(gviz)
