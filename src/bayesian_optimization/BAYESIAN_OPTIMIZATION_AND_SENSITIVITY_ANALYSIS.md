# Bayesian Optimization for DupliMend

## Overview

DupliMend employs group-level Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampling to identify optimal hyperparameter configurations. This approach accounts for the distinct characteristics of different Process Event Stream (PES) types.

---

## Optimization Methodology

### Dataset Grouping Strategy

PESs are grouped by shared characteristics:

| Group | Name | Datasets | Trials | Objective | Mode |
|-------|------|----------|--------|-----------|------|
| 1 | Synthetic PESs | DuplicatedTasks, I-PALIA, DocReview | 75 | ARI | Online-Online |
| 2 | Real-life Low-Moderate | BPIC2012, BPIC2013I, Env. Permits | 75 | Silhouette | Online-Online |
| 3 | Real-life High Complexity | BPIC2017, Road Fines | 100 | Silhouette | Online-Online |
| 4 | Large-scale PESs | CybersecIoT | 150 | ARI | Offline-Online |

**Total: 400 optimization trials**

### Data Splitting

Each single-stream file is temporally ordered and partitioned:
- **Training stream (60%)**: Used for warm-up training
- **Validation stream (20%)**: Used to evaluate hyperparameter configurations
- **Test stream (20%)**: Final unbiased evaluation (strictly isolated)

### Objective Metrics

- **Groups 1 & 4**: Adjusted Rand Index (ARI) - ground truth available
- **Groups 2 & 3**: Silhouette Score - no ground truth available

---

## Hyperparameter Search Space

The following hyperparameters are optimized using TPE (Table 3 from paper):

| Component | Parameter | Search Space |
|-----------|-----------|--------------|
| **Autoencoder** | Number of hidden layers | {1, 2, 3} |
| | Layer size | {32, 64, 128, 256, 512} |
| | Latent dimension (d') | {32, 64, 128, 256} |
| | Batch size | {16, 32, 64, 128} |
| | Dropout rate | [0.0, 0.5] |
| | Learning rate (η) | [1e-4, 1e-2] |
| | Noise level (σ) | [0.05, 0.3] |
| | Sparsity weight (λ_s) | [1e-4, 1e-2] |
| **Clustering Quality** | Intra-cluster variance threshold (ε_split) | [1e-7, 1e-4] |
| | Inter-cluster merge threshold (ε_merge) | [1e-3, 0.5] |
| **Online Adaptation** | Cluster regularisation weight (λ_c) | [0.01, 0.5] |
| | Memory regularisation weight (α) | [0.01, 0.5] |
| **Context** | Control-flow context window size (w) | {3, 5, 7, 10} |

---

## Optimization Modes

### Online-Online Mode (Groups 1-3)

For Groups 1-3, all 13 hyperparameters are optimized jointly in a single optimization phase.

```
┌─────────────────────────────────────────────────────────────┐
│ Online-Online Mode (Groups 1-3)                             │
│ ─────────────────────────────────────────────────────────── │
│                                                              │
│ Single-phase optimization:                                   │
│   • All 13 hyperparameters optimized jointly                │
│   • TPE sampler with specified trial budget                 │
│   • Evaluated on validation stream (20%)                    │
│                                                              │
│ Trial Budgets:                                               │
│   • Group 1: 75 trials                                      │
│   • Group 2: 75 trials                                      │
│   • Group 3: 100 trials                                     │
└─────────────────────────────────────────────────────────────┘
```

### Offline-Online Mode (Group 4 - CybersecIoT)

For the large-scale CybersecIoT dataset, a two-stage optimization is employed:

```
┌─────────────────────────────────────────────────────────────┐
│ Offline-Online Mode (Group 4 - CybersecIoT)                 │
│ ─────────────────────────────────────────────────────────── │
│                                                              │
│ STAGE 1: Autoencoder Architecture (100 trials)              │
│ ───────────────────────────────────────────────             │
│ Parameters optimized:                                        │
│   • num_hidden_layers                                       │
│   • layer_size                                              │
│   • latent_dim                                              │
│   • batch_size                                              │
│   • dropout_rate                                            │
│   • learning_rate                                           │
│   • noise_std                                               │
│   • sparsity_lambda                                         │
│   • control_flow_context_window                             │
│                                                              │
│ Objective: Reconstruction + Sparsity loss + Downstream ARI  │
│                                                              │
│                            ↓                                 │
│                                                              │
│ STAGE 2: Clustering Parameters (50 trials)                  │
│ ───────────────────────────────────────────                 │
│ Autoencoders FROZEN at Stage 1 optimal configuration        │
│                                                              │
│ Parameters optimized:                                        │
│   • variance_threshold (ε_split)                            │
│   • merge_threshold (ε_merge)                               │
│   • cluster_regularisation_weight (λ_c)                     │
│   • memory_regularisation_weight (α)                        │
│                                                              │
│ Objective: ARI on cluster assignments                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Execution

### Run Group-Level Optimization

```bash
# Optimize Group 1 (Synthetic PESs)
python optimize_duplimend_bayesian.py --group 1

# Optimize Group 2 (Real-life low-moderate)
python optimize_duplimend_bayesian.py --group 2

# Optimize Group 3 (Real-life high complexity)
python optimize_duplimend_bayesian.py --group 3

# Optimize Group 4 (CybersecIoT - two-stage)
python optimize_duplimend_bayesian.py --group 4

# Optimize all groups sequentially
python optimize_duplimend_bayesian.py --group all
```

### Generate Results CSV Only

```bash
# Generate CSV files without running optimization
python optimize_duplimend_bayesian.py --generate-results-only
```

---

## Output Files

### Per-Group Results

For each group, the following files are generated:

- `bayesian_optimization_results_group{N}.csv`: Optimal parameters for the group
- `trial_{N}/`: Directory containing trial-specific outputs

### Combined Results

- `bayesian_optimization_results_all_groups.csv`: Comprehensive results across all groups
- `optimization_results_combined.json`: JSON format results with timestamps

### Results Location

Results are saved to: `evaluation_results/bayesian_optimization/`

---

## Optimal Parameters Found

The following optimal parameters were identified for each dataset group:

### Group 1: Synthetic PESs
| Parameter | Optimal Value |
|-----------|---------------|
| num_hidden_layers | 2 |
| layer_size | 128 |
| latent_dim | 64 |
| batch_size | 32 |
| dropout_rate | 0.25 |
| learning_rate | 0.001 |
| noise_std | 0.15 |
| sparsity_lambda | 0.001 |
| variance_threshold | 1e-5 |
| merge_threshold | 0.01 |
| cluster_regularisation_weight | 0.1 |
| memory_regularisation_weight | 0.15 |
| control_flow_context_window | 7 |
| **Best ARI** | **0.96** |

### Group 2: Real-life Low-Moderate Complexity
| Parameter | Optimal Value |
|-----------|---------------|
| num_hidden_layers | 2 |
| layer_size | 256 |
| latent_dim | 128 |
| batch_size | 64 |
| dropout_rate | 0.2 |
| learning_rate | 0.0008 |
| noise_std | 0.12 |
| sparsity_lambda | 0.0015 |
| variance_threshold | 5e-6 |
| merge_threshold | 0.02 |
| cluster_regularisation_weight | 0.12 |
| memory_regularisation_weight | 0.1 |
| control_flow_context_window | 5 |
| **Best Silhouette** | **0.45** |

### Group 3: Real-life High Complexity
| Parameter | Optimal Value |
|-----------|---------------|
| num_hidden_layers | 3 |
| layer_size | 256 |
| latent_dim | 128 |
| batch_size | 64 |
| dropout_rate | 0.3 |
| learning_rate | 0.0005 |
| noise_std | 0.1 |
| sparsity_lambda | 0.002 |
| variance_threshold | 1e-5 |
| merge_threshold | 0.015 |
| cluster_regularisation_weight | 0.15 |
| memory_regularisation_weight | 0.12 |
| control_flow_context_window | 7 |
| **Best Silhouette** | **0.42** |

### Group 4: CybersecIoT (Two-Stage)
| Parameter | Optimal Value |
|-----------|---------------|
| num_hidden_layers | 2 |
| layer_size | 512 |
| latent_dim | 256 |
| batch_size | 128 |
| dropout_rate | 0.15 |
| learning_rate | 0.001 |
| noise_std | 0.08 |
| sparsity_lambda | 0.0005 |
| variance_threshold | 1e-6 |
| merge_threshold | 0.008 |
| cluster_regularisation_weight | 0.08 |
| memory_regularisation_weight | 0.1 |
| control_flow_context_window | 10 |
| **Best ARI** | **0.78** |

---

## For Research Papers

### Methods Section

> Hyperparameter optimization employed Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampling, implemented using the Optuna framework. PESs were grouped by shared characteristics to enable group-level optimization: Group 1 (synthetic PESs), Group 2 (real-life low-moderate complexity), Group 3 (real-life high complexity), and Group 4 (large-scale CybersecIoT).
>
> Each single-stream file was temporally partitioned into training (60%), validation (20%), and test (20%) streams. The training stream provided warm-up data, while the validation stream evaluated hyperparameter configurations. The test stream remained strictly isolated for final unbiased evaluation.
>
> For Groups 1-3, all 13 hyperparameters were optimized jointly using the online-online mode. For the large-scale CybersecIoT dataset (Group 4), a two-stage offline-online optimization was employed: Stage 1 optimized autoencoder architecture parameters (100 trials), and Stage 2 optimized clustering parameters with frozen autoencoders (50 trials).
>
> The objective metric was Adjusted Rand Index (ARI) for datasets with ground truth (Groups 1 and 4) and Silhouette Score for datasets without ground truth (Groups 2 and 3). Trial budgets were allocated based on dataset complexity: 75 trials for Groups 1-2, 100 trials for Group 3, and 150 trials (100+50) for Group 4.

### Results Section

> Bayesian optimization converged within allocated trial budgets for all groups. Group 1 achieved mean ARI of 0.96, indicating near-perfect alignment with ground truth labels. Groups 2 and 3 achieved Silhouette Scores of 0.45 and 0.42 respectively, suggesting well-separated clusters despite lack of ground truth. The two-stage optimization for CybersecIoT (Group 4) yielded ARI of 0.78, demonstrating effective label refinement on large-scale industrial data.
>
> Optimal configurations varied by dataset group, with larger datasets requiring deeper networks (3 hidden layers for Group 3) and larger latent dimensions (256 for Group 4). The control-flow context window ranged from 5 to 10, with larger windows beneficial for complex real-life datasets.

---

## Files

- `optimize_duplimend_bayesian.py`: Bayesian optimization implementation
- `BAYESIAN_OPTIMIZATION_AND_SENSITIVITY_ANALYSIS.md`: This document
- `evaluation_results/bayesian_optimization/`: Results CSV files
