#!/usr/bin/env python3
"""
Extract Best Parameters from Bayesian Optimization Results
===========================================================

This script extracts optimal hyperparameters from group-level Bayesian
optimization results and formats them for use with main.py.

Usage:
    # Extract from specific group CSV
    python extract_best_params.py --group 1

    # Extract from all groups
    python extract_best_params.py --all

    # Extract from JSON results file
    python extract_best_params.py --json <path_to_results.json>

Output:
    - Prints formatted command for main.py
    - Prints LaTeX table row for paper
    - Shows parameter summary
"""

import json
import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# ============================================================================
# DATASET GROUP CONFIGURATIONS (matches optimize_duplimend_bayesian.py)
# ============================================================================

DATASET_GROUPS = {
    1: {
        'name': 'Synthetic PESs',
        'description': 'DuplicatedTasks, I-PALIA, DocReview',
        'datasets': ['DuplicatedTasks', 'I-PALIA', 'DocReview'],
        'objective': 'ARI',
        'mode': 'online-online',
    },
    2: {
        'name': 'Real-life Low-Moderate Complexity',
        'description': 'BPIC2012, BPIC2013I, Env. Permits',
        'datasets': ['BPIC2012', 'BPIC2013I', 'Env_Permits'],
        'objective': 'Silhouette',
        'mode': 'online-online',
    },
    3: {
        'name': 'Real-life High Complexity',
        'description': 'BPIC2017, Road Fines',
        'datasets': ['BPIC2017', 'Road_Fines'],
        'objective': 'Silhouette',
        'mode': 'online-online',
    },
    4: {
        'name': 'Large-scale PESs',
        'description': 'CybersecIoT',
        'datasets': ['CybersecIoT'],
        'objective': 'ARI',
        'mode': 'offline-online',
    },
}

# Parameter names matching Table 3 from paper
PARAMETER_NAMES = [
    'num_hidden_layers',
    'layer_size',
    'latent_dim',
    'batch_size',
    'dropout_rate',
    'learning_rate',
    'noise_std',
    'sparsity_lambda',
    'variance_threshold',
    'merge_threshold',
    'cluster_regularisation_weight',
    'memory_regularisation_weight',
    'control_flow_context_window',
]


def get_results_path():
    """Get path to Bayesian optimization results directory."""
    script_dir = Path(__file__).parent
    return script_dir / "evaluation_results" / "bayesian_optimization"


def load_group_results(group_id: int) -> dict:
    """Load optimization results for a specific group from CSV."""
    results_dir = get_results_path()
    csv_file = results_dir / f"bayesian_optimization_results_group{group_id}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Results file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    if len(df) == 0:
        raise ValueError(f"Empty results file: {csv_file}")

    row = df.iloc[0]

    # Extract parameters
    params = {}
    for col in df.columns:
        if col.startswith('optimal_'):
            param_name = col.replace('optimal_', '')
            params[param_name] = row[col]

    return {
        'group_id': int(row['group_id']),
        'group_name': row['group_name'],
        'description': row['description'],
        'datasets': row['datasets'],
        'n_trials': int(row['n_trials']),
        'objective_metric': row['objective_metric'],
        'optimization_mode': row['optimization_mode'],
        'best_objective_value': float(row['best_objective_value']),
        'best_params': params,
    }


def load_all_group_results() -> dict:
    """Load optimization results for all groups from combined CSV."""
    results_dir = get_results_path()
    csv_file = results_dir / "bayesian_optimization_results_all_groups.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Combined results file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    results = {}
    for _, row in df.iterrows():
        group_id = int(row['group_id'])

        # Extract parameters
        params = {}
        for col in df.columns:
            if col.startswith('optimal_'):
                param_name = col.replace('optimal_', '')
                params[param_name] = row[col]

        results[group_id] = {
            'group_id': group_id,
            'group_name': row['group_name'],
            'description': row['description'],
            'datasets': row['datasets'],
            'n_trials': int(row['n_trials']),
            'objective_metric': row['objective_metric'],
            'optimization_mode': row['optimization_mode'],
            'best_objective_value': float(row['best_objective_value']),
            'best_params': params,
        }

    return results


def load_json_results(json_path: str) -> dict:
    """Load optimization results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def print_group_summary(results: dict):
    """Print formatted summary for a group's optimization results."""
    print("=" * 70)
    print(f"GROUP {results['group_id']}: {results['group_name']}")
    print("=" * 70)
    print(f"Description: {results['description']}")
    print(f"Datasets: {results['datasets']}")
    print(f"Optimization Mode: {results['optimization_mode']}")
    print(f"Trials: {results['n_trials']}")
    print(f"Objective Metric: {results['objective_metric']}")
    print(f"Best {results['objective_metric']}: {results['best_objective_value']:.4f}")

    print("\nOptimal Hyperparameters:")
    print("-" * 50)
    for param in PARAMETER_NAMES:
        if param in results['best_params']:
            value = results['best_params'][param]
            if isinstance(value, float):
                if value < 0.001:
                    print(f"  {param:40s}: {value:.2e}")
                else:
                    print(f"  {param:40s}: {value:.4f}")
            else:
                print(f"  {param:40s}: {value}")


def print_main_py_command(results: dict, dataset_path: str = None, output_dir: str = None):
    """Print command to run main.py with optimized parameters."""
    params = results['best_params']

    print("\n" + "=" * 70)
    print("COMMAND TO RUN MAIN.PY WITH OPTIMIZED PARAMETERS")
    print("=" * 70)

    # Determine default paths based on group
    if dataset_path is None:
        dataset_path = "<PATH_TO_YOUR_INPUT_CSV>"
    if output_dir is None:
        output_dir = f"./output_group{results['group_id']}_optimized"

    print(f"\npython main.py \\")
    print(f'  --input_csv "{dataset_path}" \\')
    print(f'  --output_dir "{output_dir}" \\')
    print(f'  --event_id_column EventID \\')
    print(f'  --case_id_column CaseID \\')
    print(f'  --timestamp_column Timestamp \\')
    print(f'  --control_flow_column Activity \\')

    # Autoencoder parameters
    print(f'  --num_hidden_layers {int(params.get("num_hidden_layers", 2))} \\')
    print(f'  --layer_size {int(params.get("layer_size", 128))} \\')
    print(f'  --latent_dim {int(params.get("latent_dim", 64))} \\')
    print(f'  --batch_size {int(params.get("batch_size", 32))} \\')
    print(f'  --dropout_rate {params.get("dropout_rate", 0.2)} \\')
    print(f'  --learning_rate {params.get("learning_rate", 0.001)} \\')
    print(f'  --noise_std {params.get("noise_std", 0.1)} \\')
    print(f'  --sparsity_lambda {params.get("sparsity_lambda", 0.001)} \\')

    # Clustering parameters
    print(f'  --variance_threshold {params.get("variance_threshold", 1e-5)} \\')
    print(f'  --merge_threshold {params.get("merge_threshold", 0.01)} \\')

    # Online adaptation parameters
    print(f'  --cluster_regularisation_weight {params.get("cluster_regularisation_weight", 0.1)} \\')
    print(f'  --memory_regularisation_weight {params.get("memory_regularisation_weight", 0.1)} \\')

    # Context parameters
    print(f'  --control_flow_context_window {int(params.get("control_flow_context_window", 7))} \\')

    # Training mode
    mode = results.get('optimization_mode', 'online-online')
    if mode == 'offline-online':
        print(f'  --training_approach offline')
    else:
        print(f'  --training_approach online')


def print_latex_table_row(results: dict):
    """Print LaTeX table row for paper."""
    params = results['best_params']

    print("\n" + "=" * 70)
    print("LATEX TABLE ROW (for paper)")
    print("=" * 70)

    # Format: Group & Layers & Size & Latent & Batch & Dropout & LR & Noise & Sparsity & VarThr & MergeThr & ClusterReg & MemReg & Window & Score
    row_parts = [
        f"Group {results['group_id']}",
        str(int(params.get('num_hidden_layers', 2))),
        str(int(params.get('layer_size', 128))),
        str(int(params.get('latent_dim', 64))),
        str(int(params.get('batch_size', 32))),
        f"{params.get('dropout_rate', 0.2):.2f}",
        f"{params.get('learning_rate', 0.001):.4f}",
        f"{params.get('noise_std', 0.1):.2f}",
        f"{params.get('sparsity_lambda', 0.001):.4f}",
        f"{params.get('variance_threshold', 1e-5):.2e}",
        f"{params.get('merge_threshold', 0.01):.3f}",
        f"{params.get('cluster_regularisation_weight', 0.1):.2f}",
        f"{params.get('memory_regularisation_weight', 0.1):.2f}",
        str(int(params.get('control_flow_context_window', 7))),
        f"{results['best_objective_value']:.2f}",
    ]

    print(" & ".join(row_parts) + " \\\\")


def print_all_groups_summary(all_results: dict):
    """Print summary table for all groups."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS SUMMARY - ALL GROUPS")
    print("=" * 70)

    print(f"\n{'Group':<8} {'Name':<35} {'Trials':<8} {'Metric':<12} {'Score':<10}")
    print("-" * 75)

    for group_id in sorted(all_results.keys()):
        r = all_results[group_id]
        print(f"{group_id:<8} {r['group_name']:<35} {r['n_trials']:<8} {r['objective_metric']:<12} {r['best_objective_value']:<10.4f}")

    print("\n" + "-" * 75)
    print(f"Total trials: {sum(r['n_trials'] for r in all_results.values())}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract best parameters from Bayesian optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract results for Group 1 (Synthetic PESs)
    python extract_best_params.py --group 1

    # Extract results for all groups
    python extract_best_params.py --all

    # Extract from a specific JSON results file
    python extract_best_params.py --json optimization_results.json

Groups:
    1 - Synthetic PESs (DuplicatedTasks, I-PALIA, DocReview)
    2 - Real-life Low-Moderate (BPIC2012, BPIC2013I, Env. Permits)
    3 - Real-life High Complexity (BPIC2017, Road Fines)
    4 - Large-scale (CybersecIoT)
        """
    )

    parser.add_argument('--group', type=int, choices=[1, 2, 3, 4],
                        help='Dataset group to extract parameters for')
    parser.add_argument('--all', action='store_true',
                        help='Extract and display all groups')
    parser.add_argument('--json', type=str,
                        help='Path to JSON results file')
    parser.add_argument('--latex', action='store_true',
                        help='Also print LaTeX table row')
    parser.add_argument('--command', action='store_true',
                        help='Also print main.py command')

    args = parser.parse_args()

    # Default behavior: show all groups summary
    if not args.group and not args.all and not args.json:
        args.all = True

    try:
        if args.json:
            # Load from JSON file
            results = load_json_results(args.json)
            print_group_summary(results)
            if args.command:
                print_main_py_command(results)
            if args.latex:
                print_latex_table_row(results)

        elif args.all:
            # Load and display all groups
            all_results = load_all_group_results()
            print_all_groups_summary(all_results)

            for group_id in sorted(all_results.keys()):
                print("\n")
                print_group_summary(all_results[group_id])
                if args.latex:
                    print_latex_table_row(all_results[group_id])

        elif args.group:
            # Load specific group
            results = load_group_results(args.group)
            print_group_summary(results)
            if args.command:
                print_main_py_command(results)
            if args.latex:
                print_latex_table_row(results)

        print("\n" + "=" * 70)
        print("Results files location:")
        print(f"  {get_results_path()}")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure optimization results CSV files exist in:")
        print(f"  {get_results_path()}")
        print("\nYou can generate them by running:")
        print("  python optimize_duplimend_bayesian.py --generate-results-only")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
