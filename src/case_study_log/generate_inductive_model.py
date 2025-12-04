import pandas as pd
import pm4py

df = pm4py.format_dataframe(
    pd.read_csv('Mine_Log_Abstract 2_GroundTruth_twoComb_Incident_Inspection_Action.csv', sep=','),
    case_id='CaseID',
    activity_key='ground_truth_activity',
    timestamp_key='Timestamp'
)
bpmn_model = pm4py.discover_bpmn_inductive(df)

# Save as SVG
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
gviz = bpmn_visualizer.apply(bpmn_model)
bpmn_visualizer.save(gviz, "bpmn_model.svg")
print("✅ BPMN model saved as bpmn_model.svg")