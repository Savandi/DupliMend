import pandas as pd
import numpy as np
from collections import defaultdict


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_ground_truth_log(df):
    incident_clusters = {}
    inspection_clusters = {}
    action_clusters = {}

    result_df = df.copy()

    result_df['ground_truth_activity'] = ''
    result_df['cluster_id'] = ''

    incident_mask = result_df['Activity'] == 'Incident'
    incident_rows = result_df[incident_mask]

    for idx, row in incident_rows.iterrows():
        incident_features = []
        incident_features.append('Incident')

        if pd.notna(row['HazardClass']) and row['HazardClass'] != '':
            incident_features.append(f"HazardClass_{row['HazardClass']}")

        if pd.notna(row['HazardType']) and row['HazardType'] != '':
            incident_features.append(f"HazardType_{row['HazardType']}")

        if pd.notna(row['Hazard Code Description (5A)']) and row['Hazard Code Description (5A)'] != '':
            incident_features.append(f"HazardCodeDesc_{row['Hazard Code Description (5A)']}")

        ground_truth = '_'.join(incident_features)
        result_df.at[idx, 'ground_truth_activity'] = ground_truth

        cluster_key = (
            str(row['HazardClass']) if pd.notna(row['HazardClass']) else 'MISSING',
            str(row['HazardType']) if pd.notna(row['HazardType']) else 'MISSING',
            str(row['Hazard Code Description (5A)']) if pd.notna(row['Hazard Code Description (5A)']) else 'MISSING'
        )

        if cluster_key not in incident_clusters:
            incident_clusters[cluster_key] = len(incident_clusters) + 1

        result_df.at[idx, 'cluster_id'] = f"Incident_Cluster_{incident_clusters[cluster_key]}"

    inspection_mask = result_df['Activity'] == 'Inspection'
    inspection_rows = result_df[inspection_mask]

    for idx, row in inspection_rows.iterrows():
        inspection_features = []
        inspection_features.append('Inspection')

        if pd.notna(row['InspectionType (MRE)']) and row['InspectionType (MRE)'] != '':
            inspection_features.append(f"InspectionType_{row['InspectionType (MRE)']}")

        ground_truth = '_'.join(inspection_features)
        result_df.at[idx, 'ground_truth_activity'] = ground_truth

        cluster_key = (
            str(row['InspectionType (MRE)']) if pd.notna(row['InspectionType (MRE)']) else 'MISSING',
            str(row['Status (MRE)']) if pd.notna(row['Status (MRE)']) else 'MISSING'
        )

        if cluster_key not in inspection_clusters:
            inspection_clusters[cluster_key] = len(inspection_clusters) + 1

        result_df.at[idx, 'cluster_id'] = f"Inspection_Cluster_{inspection_clusters[cluster_key]}"

    action_mask = result_df['Activity'] == 'Action'
    action_rows = result_df[action_mask]

    for idx, row in action_rows.iterrows():
        action_features = []
        action_features.append('Action')

        if pd.notna(row['DisplayTitle (CA)']) and row['DisplayTitle (CA)'] != '':
            action_features.append(f"DisplayTitle_{row['DisplayTitle (CA)']}")

        if pd.notna(row['Section (CA)']) and row['Section (CA)'] != '':
            action_features.append(f"Section_{row['Section (CA)']}")

        if pd.notna(row['Directive_disp (CA)']) and row['Directive_disp (CA)'] != '':
            action_features.append(f"Directive_{row['Directive_disp (CA)']}")

        if pd.notna(row['CANumber']) and row['CANumber'] != '':
            action_features.append(f"CANumber_{row['CANumber']}")

        ground_truth = '_'.join(action_features)
        result_df.at[idx, 'ground_truth_activity'] = ground_truth

        cluster_key = (
            str(row['DisplayTitle (CA)']) if pd.notna(row['DisplayTitle (CA)']) else 'MISSING',
            str(row['Section (CA)']) if pd.notna(row['Section (CA)']) else 'MISSING',
            str(row['Directive_disp (CA)']) if pd.notna(row['Directive_disp (CA)']) else 'MISSING',
            str(row['CANumber']) if pd.notna(row['CANumber']) else 'MISSING'
        )

        if cluster_key not in action_clusters:
            action_clusters[cluster_key] = len(action_clusters) + 1

        result_df.at[idx, 'cluster_id'] = f"Action_Cluster_{action_clusters[cluster_key]}"

    print(f"Incident clusters: {len(incident_clusters)}")
    print(f"Inspection clusters: {len(inspection_clusters)}")
    print(f"Action clusters: {len(action_clusters)}")

    return result_df


def analyze_clusters(df):
    print("\nCluster Analysis:")

    for activity in df['Activity'].unique():
        activity_df = df[df['Activity'] == activity]

        cluster_counts = activity_df['cluster_id'].value_counts()

        print(f"\n{activity} Activity:")
        print(f"  Total events: {len(activity_df)}")
        print(f"  Number of clusters: {len(cluster_counts)}")
        print(f"  Average cluster size: {len(activity_df) / len(cluster_counts):.2f}")
        print(f"  Minimum cluster size: {cluster_counts.min()}")
        print(f"  Maximum cluster size: {cluster_counts.max()}")

        print(f"\n  Top 5 largest clusters:")
        top_clusters = cluster_counts.sort_values(ascending=False).head(5)
        for cluster, count in top_clusters.items():
            percentage = count / len(activity_df) * 100
            print(f"    {cluster}: {count} events ({percentage:.2f}%)")

            sample_events = df[df['cluster_id'] == cluster].head(2)
            for _, event in sample_events.iterrows():
                print(f"      Example: {event['ground_truth_activity']}")


def extract_temporal_features(df):
    if df['Timestamp'].dtype != 'datetime64[ns]':
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month
    df['year'] = df['Timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_business_hours'] = df['hour_of_day'].apply(lambda x: 1 if 8 <= x <= 17 else 0)

    print("Temporal features extracted:")
    print("  - hour_of_day (0-23)")
    print("  - day_of_week (0=Monday, 6=Sunday)")
    print("  - month (1-12)")
    print("  - year")
    print("  - is_weekend (0=No, 1=Yes)")
    print("  - is_business_hours (0=No, 1=Yes)")

    return df


def simulate_control_flow_features(df):
    print("\nSimulating control flow context features...")

    case_events = df.sort_values(['CaseID', 'Timestamp']).groupby('CaseID')

    df['prev_activity'] = None
    df['next_activity'] = None
    df['activity_position'] = 0
    df['case_length'] = 0

    for case_id, events in case_events:
        case_length = len(events)

        for i, (idx, event) in enumerate(events.iterrows()):
            df.at[idx, 'activity_position'] = i + 1
            df.at[idx, 'case_length'] = case_length

            if i > 0:
                df.at[idx, 'prev_activity'] = events.iloc[i - 1]['Activity']

            if i < case_length - 1:
                df.at[idx, 'next_activity'] = events.iloc[i + 1]['Activity']

    print("Control flow features added:")
    print("  - prev_activity: Previous activity in the case")
    print("  - next_activity: Next activity in the case")
    print("  - activity_position: Position of the activity in the case")
    print("  - case_length: Total number of activities in the case")

    return df


def enhanced_clustering(df):
    print("\nPerforming enhanced clustering with temporal and control flow features...")

    result_df = create_ground_truth_log(df)
    result_df = extract_temporal_features(result_df)
    result_df = simulate_control_flow_features(result_df)
    result_df['enhanced_cluster_id'] = result_df['cluster_id']

    analyze_clusters(result_df)

    return result_df


def main():
    print("Mining Log Ground Truth Generator")
    print("=================================")

    file_path = 'Mine Log Abstract 2_2000.csv'
    df = load_data(file_path)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    result_df = enhanced_clustering(df)

    output_file = 'mine_log_ground_truth.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\nGround truth log saved to {output_file}")

    print("\nDataset Summary:")
    print(f"  Total events: {len(result_df)}")
    print(f"  Activity types: {', '.join(result_df['Activity'].unique())}")
    print(f"  Number of cases: {result_df['CaseID'].nunique()}")
    print(f"  Date range: {result_df['Timestamp'].min()} to {result_df['Timestamp'].max()}")


if __name__ == "__main__":
    main()
