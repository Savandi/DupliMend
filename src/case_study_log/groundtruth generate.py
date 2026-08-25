import os
import pandas as pd

def generate_ground_truth_labels(df, activity_column, activity_feature_map, new_column='ground_truth_activity'):
    """
    Append ground truth labels based on specific columns per activity.
    """
    def build_ground_truth(row):
        activity = row[activity_column]
        if activity in activity_feature_map:
            cols = activity_feature_map[activity]
            try:
                values = [str(row[col]) for col in cols]
            except KeyError:
                values = ['MISSING' for _ in cols]
            return f"{activity}_" + "_".join(values)
        else:
            return activity

    df[new_column] = df.apply(build_ground_truth, axis=1)
    return df

if __name__ == "__main__":
    input_file = 'Mine Log Abstract 2.csv'
    output_file = 'Mine_Log_Abstract 2_GroundTruth_only_Incident_HazardClassEventType(1A).csv'
    activity_column = 'Activity'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}. Please check the path and try again.")


    activity_feature_map = {
        'Incident': ['HazardClass', 'EventType (1A)'],
    }

    df = pd.read_csv(input_file, encoding='ISO-8859-1')
    inspection_rows = df[df['Activity'] == 'Inspection']
    print("Number of Inspection rows:", len(inspection_rows))
    print(inspection_rows[['InspectionType (MRE)', 'MineType']].head(10))

    df = generate_ground_truth_labels(df, activity_column, activity_feature_map)

    df.to_csv(output_file, index=False)
    print(f"✅ Saved updated log to: {output_file}")
