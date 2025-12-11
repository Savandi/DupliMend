import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_confusion_matrix(confusion_matrix, output_path="confusion_matrix.png"):
    plt.figure(figsize=(12, 10))

    cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#ffffff', '#0343df'])

    ax = sns.heatmap(confusion_matrix, annot=True, cmap=cmap, fmt='.2f', linewidths=.5)
    ax.set_title('Cluster to Ground Truth Confusion Matrix', fontsize=16)
    ax.set_xlabel('Ground Truth Labels', fontsize=14)
    ax.set_ylabel('Cluster Labels', fontsize=14)

    if len(confusion_matrix.columns) > 10:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix visualization saved to {output_path}")

def plot_cluster_metrics(metrics, cluster_stats, output_dir="cluster_visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    clusters = [stat['cluster'] for stat in cluster_stats]
    sizes = [stat['size'] for stat in cluster_stats]

    sorted_indices = np.argsort(sizes)[::-1]
    clusters = [clusters[i] for i in sorted_indices]
    sizes = [sizes[i] for i in sorted_indices]

    bars = plt.bar(clusters, sizes, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Events')
    plt.title('Cluster Sizes')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    purities = [stat['purity'] for stat in cluster_stats]
    purities = [purities[i] for i in sorted_indices]

    bars = plt.bar(clusters, purities, color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Cluster')
    plt.ylabel('Purity')
    plt.title('Cluster Purities')
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_purities.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    entropies = [stat['entropy'] for stat in cluster_stats]
    entropies = [entropies[i] for i in sorted_indices]

    bars = plt.bar(clusters, entropies, color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Cluster')
    plt.ylabel('Entropy')
    plt.title('Cluster Entropies (Lower is Better)')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_entropies.png'), dpi=300)
    plt.close()

    print(f"Cluster visualizations saved to {output_dir}")

def visualize_cluster_results(evaluation_report, output_dir="cluster_visualizations"):
    metrics = {}
    cluster_stats = []

    with open(evaluation_report, 'r') as f:
        lines = f.readlines()

    metrics_section = False
    for line in lines:
        if '## Overall Metrics' in line:
            metrics_section = True
            continue
        if metrics_section and line.startswith('##'):
            metrics_section = False
            continue
        if metrics_section and line.startswith('*'):
            parts = line.strip('* \n').split(':')
            if len(parts) >= 2:
                metric_name = parts[0].strip()
                metric_value = float(parts[1].strip())
                metrics[metric_name] = metric_value

    summary_section = False
    header_line = 0
    for i, line in enumerate(lines):
        if '## Cluster Summary' in line:
            summary_section = True
            header_line = i + 3
            continue
        if summary_section and line.startswith('##'):
            summary_section = False
            continue
        if summary_section and i > header_line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 6:
                cluster = parts[1].strip()
                size = int(parts[2].strip())
                dominant_gt = parts[3].strip()
                purity = float(parts[4].strip())
                entropy = float(parts[5].strip())

                cluster_stats.append({
                    'cluster': cluster,
                    'size': size,
                    'dominant_ground_truth': dominant_gt,
                    'purity': purity,
                    'entropy': entropy
                })

    plot_cluster_metrics(metrics, cluster_stats, output_dir)

    print(f"Visualizations based on {evaluation_report} saved to {output_dir}")

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py evaluation_report.md [output_directory]")
        sys.exit(1)

    evaluation_report = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cluster_visualizations"

    if not os.path.exists(evaluation_report):
        print(f"Error: Evaluation report not found: {evaluation_report}")
        sys.exit(1)

    visualize_cluster_results(evaluation_report, output_dir)
