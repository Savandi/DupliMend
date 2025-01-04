import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_dataset(num_clusters=3, num_samples_per_cluster=100, overlap=False):
    """
    Generate a synthetic dataset with defined clusters and optional overlap.

    Parameters:
        num_clusters (int): Number of clusters to generate.
        num_samples_per_cluster (int): Number of samples per cluster.
        overlap (bool): If True, clusters will overlap slightly.

    Returns:
        pd.DataFrame: Synthetic dataset with features and labels.
    """
    np.random.seed(42)  # For reproducibility
    data = []
    base_time = datetime(2023, 1, 1, 8, 0)  # Start of the process
    activity_labels = [f"Activity_{i}" for i in range(num_clusters)]
    resources = [f"Resource_{i}" for i in range(5)]  # Example resources

    for cluster_id in range(num_clusters):
        # Generate cluster center
        center = np.random.uniform(0, 10, size=3)
        if overlap:
            # Shift center slightly for overlap
            center += np.random.uniform(-1, 1, size=3)
        for sample in range(num_samples_per_cluster):
            # Generate data points around the center with slight noise
            point = center + np.random.normal(scale=0.5, size=3)

            # Add a timestamp with incremental time steps
            timestamp = base_time + timedelta(minutes=sample + cluster_id * 10)

            # Randomly assign a resource
            resource = np.random.choice(resources)

            # Append event with features
            data.append([timestamp, resource, *point, activity_labels[cluster_id]])

    # Create a DataFrame
    columns = ['Timestamp', 'Resource', 'Feature_1', 'Feature_2', 'Feature_3', 'Activity']
    return pd.DataFrame(data, columns=columns)

# Generate synthetic data with clear clusters
clear_clusters = generate_synthetic_dataset(num_clusters=3, num_samples_per_cluster=50)

# Generate synthetic data with overlapping clusters
overlapping_clusters = generate_synthetic_dataset(num_clusters=3, num_samples_per_cluster=50, overlap=True)

# Save for testing
clear_clusters.to_csv("clear_clusters.csv", index=False)
overlapping_clusters.to_csv("overlapping_clusters.csv", index=False)
