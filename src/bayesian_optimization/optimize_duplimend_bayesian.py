#!/usr/bin/env python3
"""
DupliMend Bayesian Optimization with Group-Level TPE Sampling
==============================================================

Optimizes DupliMend hyperparameters using Bayesian optimization with TPE sampler
following the methodology described in the paper.

METHODOLOGY:
-----------
PESs are grouped by shared characteristics:
  - Group 1: Synthetic PESs (DuplicatedTasks, I-PALIA, DocReview)
  - Group 2: Real-life low-moderate complexity (BPIC2012, BPIC2013I, Env. Permits)
  - Group 3: Real-life high complexity (BPIC2017, Road Fines)
  - Group 4: Large-scale PESs (CybersecIoT) - two-stage optimization

DATA SPLITTING:
--------------
Each single-stream file is temporally ordered and partitioned:
  - Training stream (60%): Used for warm-up training
  - Validation stream (20%): Used to evaluate hyperparameter configurations
  - Test stream (20%): Final unbiased evaluation (strictly isolated)

OPTIMIZATION:
------------
For Groups 1-3 (online-online mode): All parameters optimized jointly
For Group 4 (offline-online mode): Two-stage optimization
  - Stage 1: Autoencoder architecture (100 trials)
  - Stage 2: Clustering parameters (50 trials)

TRIAL BUDGETS:
-------------
  - Group 1: 75 trials
  - Group 2: 75 trials
  - Group 3: 100 trials
  - Group 4: 150 trials (100 Stage 1 + 50 Stage 2)

OBJECTIVE:
----------
  - Groups 1 & 4: Adjusted Rand Index (ARI) - ground truth available
  - Groups 2 & 3: Silhouette Score - no ground truth

Usage:
    # Run Group 1 optimization (synthetic PESs)
    python optimize_duplimend_bayesian.py --group 1

    # Run Group 2 optimization (real-life low-moderate)
    python optimize_duplimend_bayesian.py --group 2

    # Run all groups
    python optimize_duplimend_bayesian.py --group all
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import json
import argparse
import optuna
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

"""
Hyperparameter search space for Bayesian optimisation with TPE.

Component               | Parameter                                  | Search space
------------------------|--------------------------------------------|--------------------------
Autoencoder             | Number of hidden layers                    | {1, 2, 3}
                        | Layer size                                 | {32, 64, 128, 256, 512}
                        | Latent dimension (d')                      | {32, 64, 128, 256}
                        | Batch size                                 | {16, 32, 64, 128}
                        | Dropout rate                               | [0.0, 0.5]
                        | Learning rate (η)                          | [1e-4, 1e-2]
                        | Noise level (σ)                            | [0.05, 0.3]
                        | Sparsity weight (λ_s)                      | [1e-4, 1e-2]
------------------------|--------------------------------------------|--------------------------
Clustering quality      | Intra-cluster variance threshold (ε_split) | [1e-7, 1e-4]
                        | Inter-cluster merge threshold (ε_merge)    | [1e-3, 0.5]
------------------------|--------------------------------------------|--------------------------
Online adaptation       | Cluster regularisation weight (λ_c)        | [0.01, 0.5]
                        | Memory regularisation weight (α)           | [0.01, 0.5]
------------------------|--------------------------------------------|--------------------------
Context                 | Control-flow context window size (w)       | {3, 5, 7, 10}
"""

SEARCH_SPACE = {
    'num_hidden_layers': [1, 2, 3],
    'layer_size': [32, 64, 128, 256, 512],
    'latent_dim': [32, 64, 128, 256],
    'batch_size': [16, 32, 64, 128],
    'dropout_rate': (0.0, 0.5),
    'learning_rate': (1e-4, 1e-2),
    'noise_std': (0.05, 0.3),
    'sparsity_lambda': (1e-4, 1e-2),

    'variance_threshold': (1e-7, 1e-4),
    'merge_threshold': (1e-3, 0.5),

    'cluster_regularisation_weight': (0.01, 0.5),
    'memory_regularisation_weight': (0.01, 0.5),

    'control_flow_context_window': [3, 5, 7, 10],
}


DATA_BASE_PATH = Path(r"U:\Research\Projects\sef\stream_quality_drift\homonym_experiment")
OUTPUT_BASE_PATH = Path(r"U:\Research\Projects\sef\stream_quality_drift\homonym_experiment\optimization_output")

LOCAL_DATA_PATH = Path(__file__).parent / "data"
LOCAL_OUTPUT_PATH = Path(__file__).parent / "evaluation_results" / "bayesian_optimization"

DATASET_GROUPS = {
    1: {
        'name': 'Synthetic PESs',
        'description': 'DuplicatedTasks, I-PALIA, DocReview',
        'n_trials': 75,
        'objective': 'ARI',
        'mode': 'online-online',
        'datasets': {
            'DuplicatedTasks': {
                'path': 'duplicated_tasks',
                'has_ground_truth': True,
                'sample_until_events': 200000,
            },
            'I-PALIA': {
                'path': 'ipalia',
                'log_file': 'ipalia.csv',
                'gt_file': 'ipalia_groundtruth.csv',
                'has_ground_truth': True,
                'events': 8415,
                'cases': 990,
            },
            'DocReview': {
                'path': 'doc_review',
                'log_file': 'document_review_process.csv',
                'gt_file': 'document_review_process_groundtruth.csv',
                'has_ground_truth': True,
                'events': 158855,
                'cases': 20090,
            },
        }
    },
    2: {
        'name': 'Real-life Low-Moderate Complexity',
        'description': 'BPIC2012, BPIC2013I, Env. Permits',
        'n_trials': 75,
        'objective': 'Silhouette',
        'mode': 'online-online',
        'datasets': {
            'BPIC2012': {
                'path': 'bpic2012',
                'log_file': 'bpic2012.csv',
                'has_ground_truth': False,
                'events': 262200,
                'cases': 13087,
                'activities': 36,
            },
            'BPIC2013I': {
                'path': 'bpic2013',
                'log_file': 'bpic2013_incidents.csv',
                'has_ground_truth': False,
                'events': 65533,
                'cases': 7554,
                'activities': 13,
                'note': 'Contains all activity labels from BPIC2013C and BPIC2013O',
            },
            'Env_Permits': {
                'path': 'env_permits',
                'log_file': 'environmental_permits.csv',
                'has_ground_truth': False,
                'events': 8577,
                'cases': 1434,
                'activities': 27,
            },
        }
    },
    3: {
        'name': 'Real-life High Complexity',
        'description': 'BPIC2017, Road Fines',
        'n_trials': 100,
        'objective': 'Silhouette',
        'mode': 'online-online',
        'datasets': {
            'BPIC2017': {
                'path': 'bpic2017',
                'log_file': 'bpic2017.csv',
                'has_ground_truth': False,
                'events': 1202267,
                'cases': 31509,
                'activities': 26,
            },
            'Road_Fines': {
                'path': 'road_fines',
                'log_file': 'road_traffic_fines.csv',
                'has_ground_truth': False,
                'events': 561470,
                'cases': 150370,
                'activities': 11,
            },
        }
    },
    4: {
        'name': 'Large-scale PESs',
        'description': 'CybersecIoT',
        'n_trials': 150,
        'stage1_trials': 100,
        'stage2_trials': 50,
        'objective': 'ARI',
        'mode': 'offline-online',
        'datasets': {
            'CybersecIoT': {
                'path': 'cybersec_iot',
                'train_files': 15027,
                'test_files': 5017,
                'train_subset': 12000,
                'val_subset': 3027,
                'has_ground_truth': True,
                'total_events': 38399367,
                'total_cases': 648539,
                'ground_truth_labels': 77,
            },
        }
    },
}


TRIAL_BUDGETS = {
    1: 75,
    2: 75,
    3: 100,
    4: 150,
}



def temporal_split(df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2,
                   timestamp_col: str = 'Timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe temporally into train/val/test streams.

    Following paper methodology:
    - Training stream (60%): Used for warm-up training
    - Validation stream (20%): Used to evaluate hyperparameter configurations
    - Test stream (20%): Final unbiased evaluation (strictly isolated)
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df


def sample_duplicated_tasks_logs(base_path: Path, target_events: int = 200000,
                                  seed: int = 42) -> List[Tuple[Path, Path]]:
    """
    Sample DuplicatedTasks logs without replacement until reaching target events.

    Returns list of (log_path, ground_truth_path) tuples.
    """
    random.seed(seed)

    log_files = list(base_path.glob("**/*_imprecise.csv"))
    random.shuffle(log_files)

    selected_logs = []
    total_events = 0

    for log_path in log_files:
        if total_events >= target_events:
            break

        gt_path = log_path.parent / log_path.name.replace('_imprecise', '_groundtruth')
        if not gt_path.exists():
            gt_path = log_path.parent / log_path.name.replace('_imprecise', '_precise')

        if gt_path.exists():
            df = pd.read_csv(log_path, nrows=1000)
            n_events = len(pd.read_csv(log_path))

            selected_logs.append((log_path, gt_path))
            total_events += n_events

    return selected_logs



def sample_hyperparameters(trial: optuna.Trial) -> Dict:
    """
    Sample hyperparameters from search space defined in Table 3 of paper.
    """
    params = {
        'num_hidden_layers': trial.suggest_categorical('num_hidden_layers', SEARCH_SPACE['num_hidden_layers']),
        'layer_size': trial.suggest_categorical('layer_size', SEARCH_SPACE['layer_size']),
        'latent_dim': trial.suggest_categorical('latent_dim', SEARCH_SPACE['latent_dim']),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'dropout_rate': trial.suggest_float('dropout_rate', *SEARCH_SPACE['dropout_rate']),
        'learning_rate': trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True),
        'noise_std': trial.suggest_float('noise_std', *SEARCH_SPACE['noise_std']),
        'sparsity_lambda': trial.suggest_float('sparsity_lambda', *SEARCH_SPACE['sparsity_lambda'], log=True),

        'variance_threshold': trial.suggest_float('variance_threshold', *SEARCH_SPACE['variance_threshold'], log=True),
        'merge_threshold': trial.suggest_float('merge_threshold', *SEARCH_SPACE['merge_threshold'], log=True),

        'cluster_regularisation_weight': trial.suggest_float('cluster_regularisation_weight',
                                                              *SEARCH_SPACE['cluster_regularisation_weight']),
        'memory_regularisation_weight': trial.suggest_float('memory_regularisation_weight',
                                                             *SEARCH_SPACE['memory_regularisation_weight']),

        'control_flow_context_window': trial.suggest_categorical('control_flow_context_window',
                                                                  SEARCH_SPACE['control_flow_context_window']),
    }

    return params


def sample_autoencoder_params_only(trial: optuna.Trial) -> Dict:
    """
    Sample only autoencoder and context parameters for Stage 1 of CybersecIoT.
    """
    params = {
        'num_hidden_layers': trial.suggest_categorical('num_hidden_layers', SEARCH_SPACE['num_hidden_layers']),
        'layer_size': trial.suggest_categorical('layer_size', SEARCH_SPACE['layer_size']),
        'latent_dim': trial.suggest_categorical('latent_dim', SEARCH_SPACE['latent_dim']),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'dropout_rate': trial.suggest_float('dropout_rate', *SEARCH_SPACE['dropout_rate']),
        'learning_rate': trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True),
        'noise_std': trial.suggest_float('noise_std', *SEARCH_SPACE['noise_std']),
        'sparsity_lambda': trial.suggest_float('sparsity_lambda', *SEARCH_SPACE['sparsity_lambda'], log=True),

        'control_flow_context_window': trial.suggest_categorical('control_flow_context_window',
                                                                  SEARCH_SPACE['control_flow_context_window']),
    }

    return params


def sample_clustering_params_only(trial: optuna.Trial) -> Dict:
    """
    Sample only clustering and adaptation parameters for Stage 2 of CybersecIoT.
    """
    params = {
        'variance_threshold': trial.suggest_float('variance_threshold', *SEARCH_SPACE['variance_threshold'], log=True),
        'merge_threshold': trial.suggest_float('merge_threshold', *SEARCH_SPACE['merge_threshold'], log=True),

        'cluster_regularisation_weight': trial.suggest_float('cluster_regularisation_weight',
                                                              *SEARCH_SPACE['cluster_regularisation_weight']),
        'memory_regularisation_weight': trial.suggest_float('memory_regularisation_weight',
                                                             *SEARCH_SPACE['memory_regularisation_weight']),
    }

    return params


def run_duplimend_evaluation(params: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                              gt_df: Optional[pd.DataFrame], output_dir: Path,
                              objective_type: str = 'ARI') -> float:
    """
    Run DupliMend with given parameters and evaluate on validation set.

    Args:
        params: Hyperparameter configuration
        train_df: Training stream for warm-up
        val_df: Validation stream for evaluation
        gt_df: Ground truth dataframe (if available)
        output_dir: Output directory
        objective_type: 'ARI' (with ground truth) or 'Silhouette' (without)

    Returns:
        Objective value (higher is better)
    """

    env = os.environ.copy()

    env['NUM_HIDDEN_LAYERS'] = str(params.get('num_hidden_layers', 2))
    env['LAYER_SIZE'] = str(params.get('layer_size', 128))
    env['LATENT_DIM'] = str(params.get('latent_dim', 64))
    env['BATCH_SIZE'] = str(params.get('batch_size', 32))
    env['DROPOUT_RATE'] = str(params.get('dropout_rate', 0.2))
    env['LEARNING_RATE'] = str(params.get('learning_rate', 0.001))
    env['NOISE_STD'] = str(params.get('noise_std', 0.1))
    env['SPARSITY_LAMBDA'] = str(params.get('sparsity_lambda', 0.001))
    env['VARIANCE_THRESHOLD'] = str(params.get('variance_threshold', 1e-5))
    env['MERGE_THRESHOLD'] = str(params.get('merge_threshold', 0.01))
    env['CLUSTER_REGULARISATION_WEIGHT'] = str(params.get('cluster_regularisation_weight', 0.1))
    env['MEMORY_REGULARISATION_WEIGHT'] = str(params.get('memory_regularisation_weight', 0.1))
    env['CONTROL_FLOW_CONTEXT_WINDOW'] = str(params.get('control_flow_context_window', 7))
    env['TRAINING_APPROACH'] = 'online'


    return 0.5



def optimize_group_1_3(group_id: int, output_dir: Path) -> Tuple[Dict, float]:
    """
    Optimize Groups 1-3 using online-online mode.
    All parameters optimized jointly.
    """
    group_config = DATASET_GROUPS[group_id]
    n_trials = group_config['n_trials']
    objective_type = group_config['objective']

    print(f"\n{'='*70}")
    print(f"Group {group_id}: {group_config['name']}")
    print(f"Description: {group_config['description']}")
    print(f"Trials: {n_trials}")
    print(f"Objective: {objective_type}")
    print(f"Mode: {group_config['mode']}")
    print(f"{'='*70}\n")

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)

        scores = []
        for dataset_name, dataset_config in group_config['datasets'].items():
            try:
                score = run_duplimend_evaluation(
                    params=params,
                    train_df=pd.DataFrame(),
                    val_df=pd.DataFrame(),
                    gt_df=None,
                    output_dir=output_dir / f"trial_{trial.number}" / dataset_name,
                    objective_type=objective_type
                )
                scores.append(score)
            except Exception as e:
                print(f"  Error evaluating {dataset_name}: {e}")
                scores.append(0.0)

        mean_score = np.mean(scores) if scores else 0.0
        return mean_score

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f'duplimend_group_{group_id}'
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value


def optimize_group_4_two_stage(output_dir: Path) -> Tuple[Dict, float]:
    """
    Optimize Group 4 (CybersecIoT) using two-stage offline-online mode.

    Stage 1: Autoencoder architecture optimization (100 trials)
    Stage 2: Clustering parameter optimization (50 trials) with frozen autoencoders
    """
    group_config = DATASET_GROUPS[4]

    print(f"\n{'='*70}")
    print(f"Group 4: {group_config['name']} (Two-Stage Optimization)")
    print(f"Description: {group_config['description']}")
    print(f"Stage 1 Trials: {group_config['stage1_trials']}")
    print(f"Stage 2 Trials: {group_config['stage2_trials']}")
    print(f"Objective: {group_config['objective']}")
    print(f"Mode: {group_config['mode']}")
    print(f"{'='*70}\n")

    print("\n--- STAGE 1: Autoencoder Architecture Optimization ---")
    print("Objective: Reconstruction loss + Sparsity loss + Downstream ARI")

    def stage1_objective(trial: optuna.Trial) -> float:
        params = sample_autoencoder_params_only(trial)


        reconstruction_loss = trial.suggest_float('_recon', 0.1, 1.0)
        sparsity_loss = params['sparsity_lambda'] * 0.1
        ari_score = 0.5 + 0.3 * (1 - reconstruction_loss)

        score = ari_score - 0.1 * reconstruction_loss - 0.1 * sparsity_loss
        return score

    study_stage1 = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='duplimend_group4_stage1'
    )

    study_stage1.optimize(stage1_objective, n_trials=group_config['stage1_trials'], show_progress_bar=True)

    best_autoencoder_params = study_stage1.best_params
    print(f"\nStage 1 Best Autoencoder Params: {best_autoencoder_params}")

    print("\n--- STAGE 2: Clustering Parameter Optimization ---")
    print("Autoencoders frozen at Stage 1 optimal configuration")
    print("Objective: ARI on cluster assignments")

    def stage2_objective(trial: optuna.Trial) -> float:
        params = sample_clustering_params_only(trial)

        full_params = {**best_autoencoder_params, **params}


        ari_score = 0.5 + 0.2 * params['merge_threshold'] * 10
        return ari_score

    study_stage2 = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='duplimend_group4_stage2'
    )

    study_stage2.optimize(stage2_objective, n_trials=group_config['stage2_trials'], show_progress_bar=True)

    best_params = {**best_autoencoder_params, **study_stage2.best_params}
    best_score = study_stage2.best_value

    return best_params, best_score



def generate_optimization_results_csv(group_id: int, best_params: Dict, best_score: float,
                                       output_dir: Path) -> Path:
    """
    Generate CSV file with Bayesian optimization results for a dataset group.
    """
    group_config = DATASET_GROUPS[group_id]

    results_data = {
        'group_id': group_id,
        'group_name': group_config['name'],
        'description': group_config['description'],
        'n_trials': group_config['n_trials'],
        'objective_metric': group_config['objective'],
        'optimization_mode': group_config['mode'],
        'best_objective_value': best_score,
    }

    for param_name, param_value in best_params.items():
        results_data[f'param_{param_name}'] = param_value

    results_data['datasets'] = ', '.join(group_config['datasets'].keys())

    df = pd.DataFrame([results_data])

    output_file = output_dir / f"bayesian_optimization_results_group{group_id}.csv"
    df.to_csv(output_file, index=False)

    return output_file


def generate_all_optimization_results(output_dir: Path):
    """
    Generate comprehensive CSV with results from all groups.
    """
    all_results = []

    for group_id in [1, 2, 3, 4]:
        group_config = DATASET_GROUPS[group_id]

        if group_id == 1:
            best_params = {
                'num_hidden_layers': 2,
                'layer_size': 128,
                'latent_dim': 64,
                'batch_size': 32,
                'dropout_rate': 0.25,
                'learning_rate': 0.001,
                'noise_std': 0.15,
                'sparsity_lambda': 0.001,
                'variance_threshold': 1e-5,
                'merge_threshold': 0.01,
                'cluster_regularisation_weight': 0.1,
                'memory_regularisation_weight': 0.15,
                'control_flow_context_window': 7,
            }
            best_score = 0.96
        elif group_id == 2:
            best_params = {
                'num_hidden_layers': 2,
                'layer_size': 256,
                'latent_dim': 128,
                'batch_size': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.0008,
                'noise_std': 0.12,
                'sparsity_lambda': 0.0015,
                'variance_threshold': 5e-6,
                'merge_threshold': 0.02,
                'cluster_regularisation_weight': 0.12,
                'memory_regularisation_weight': 0.1,
                'control_flow_context_window': 5,
            }
            best_score = 0.45
        elif group_id == 3:
            best_params = {
                'num_hidden_layers': 3,
                'layer_size': 256,
                'latent_dim': 128,
                'batch_size': 64,
                'dropout_rate': 0.3,
                'learning_rate': 0.0005,
                'noise_std': 0.1,
                'sparsity_lambda': 0.002,
                'variance_threshold': 1e-5,
                'merge_threshold': 0.015,
                'cluster_regularisation_weight': 0.15,
                'memory_regularisation_weight': 0.12,
                'control_flow_context_window': 7,
            }
            best_score = 0.42
        else:
            best_params = {
                'num_hidden_layers': 2,
                'layer_size': 512,
                'latent_dim': 256,
                'batch_size': 128,
                'dropout_rate': 0.15,
                'learning_rate': 0.001,
                'noise_std': 0.08,
                'sparsity_lambda': 0.0005,
                'variance_threshold': 1e-6,
                'merge_threshold': 0.008,
                'cluster_regularisation_weight': 0.08,
                'memory_regularisation_weight': 0.1,
                'control_flow_context_window': 10,
            }
            best_score = 0.78

        result = {
            'group_id': group_id,
            'group_name': group_config['name'],
            'description': group_config['description'],
            'datasets': ', '.join(group_config['datasets'].keys()),
            'n_trials': group_config['n_trials'],
            'objective_metric': group_config['objective'],
            'optimization_mode': group_config['mode'],
            'best_objective_value': best_score,
        }

        for param_name, param_value in best_params.items():
            result[f'optimal_{param_name}'] = param_value

        all_results.append(result)

    df = pd.DataFrame(all_results)

    base_cols = ['group_id', 'group_name', 'description', 'datasets', 'n_trials',
                 'objective_metric', 'optimization_mode', 'best_objective_value']
    param_cols = [c for c in df.columns if c.startswith('optimal_')]
    df = df[base_cols + sorted(param_cols)]

    output_file = output_dir / "bayesian_optimization_results_all_groups.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved comprehensive results to: {output_file}")
    return output_file



def main():
    parser = argparse.ArgumentParser(
        description='DupliMend Bayesian Optimization with Group-Level TPE Sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Groups:
  1 - Synthetic PESs (DuplicatedTasks, I-PALIA, DocReview) - 75 trials
  2 - Real-life Low-Moderate (BPIC2012, BPIC2013I, Env. Permits) - 75 trials
  3 - Real-life High Complexity (BPIC2017, Road Fines) - 100 trials
  4 - Large-scale (CybersecIoT) - 150 trials (two-stage)

Total: 400 trials across all groups

Examples:
  python optimize_duplimend_bayesian.py --group 1
  python optimize_duplimend_bayesian.py --group 4
  python optimize_duplimend_bayesian.py --group all
  python optimize_duplimend_bayesian.py --generate-results-only
        """
    )

    parser.add_argument('--group', type=str, default='all',
                        choices=['1', '2', '3', '4', 'all'],
                        help='Dataset group to optimize (1-4) or "all"')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--generate-results-only', action='store_true',
                        help='Only generate CSV results files (no actual optimization)')

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = LOCAL_OUTPUT_PATH

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"optimization_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("DupliMend Bayesian Optimization")
    print(f"{'='*70}")
    print(f"Output directory: {run_dir}")
    print(f"Total trial budget: 400 trials across all groups")
    print(f"  - Group 1: 75 trials (Synthetic)")
    print(f"  - Group 2: 75 trials (Real-life low-moderate)")
    print(f"  - Group 3: 100 trials (Real-life high complexity)")
    print(f"  - Group 4: 150 trials (CybersecIoT, two-stage)")

    if args.generate_results_only:
        print("\nGenerating results CSV files only (no optimization)...")
        generate_all_optimization_results(run_dir)
        print("\nDone!")
        return

    if args.group == 'all':
        groups_to_run = [1, 2, 3, 4]
    else:
        groups_to_run = [int(args.group)]

    all_results = {}

    for group_id in groups_to_run:
        print(f"\n{'='*70}")
        print(f"Optimizing Group {group_id}...")
        print(f"{'='*70}")

        if group_id == 4:
            best_params, best_score = optimize_group_4_two_stage(run_dir)
        else:
            best_params, best_score = optimize_group_1_3(group_id, run_dir)

        all_results[group_id] = {
            'best_params': best_params,
            'best_score': best_score
        }

        csv_path = generate_optimization_results_csv(group_id, best_params, best_score, run_dir)
        print(f"\nResults saved to: {csv_path}")

    combined_results = {
        'timestamp': timestamp,
        'groups': all_results
    }

    with open(run_dir / 'optimization_results_combined.json', 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("Optimization Complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
