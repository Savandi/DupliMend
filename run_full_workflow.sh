#!/bin/bash
# ==============================================================================
# DupliMend Complete Workflow Script
# ==============================================================================
#
# This script runs the complete DupliMend workflow:
#   1. Bayesian optimization (group-level with TPE)
#   2. Extract optimal parameters
#   3. Run DupliMend with optimized parameters
#   4. Evaluate results
#
# Usage:
#   ./run_full_workflow.sh --group 1              # Run Group 1 (Synthetic)
#   ./run_full_workflow.sh --group 4              # Run Group 4 (CybersecIoT)
#   ./run_full_workflow.sh --group all            # Run all groups
#   ./run_full_workflow.sh --extract-only         # Only extract/display params
#
# ==============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

GROUP="1"                           # Default: Group 1 (Synthetic PESs)
EXTRACT_ONLY=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --group)
            GROUP="$2"
            shift 2
            ;;
        --extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --group N        Dataset group to optimize (1, 2, 3, 4, or all)"
            echo "  --extract-only   Only extract and display parameters (no optimization)"
            echo "  --help           Show this help message"
            echo ""
            echo "Groups:"
            echo "  1 - Synthetic PESs (DuplicatedTasks, I-PALIA, DocReview) - 75 trials"
            echo "  2 - Real-life Low-Moderate (BPIC2012, BPIC2013I, Env. Permits) - 75 trials"
            echo "  3 - Real-life High Complexity (BPIC2017, Road Fines) - 100 trials"
            echo "  4 - Large-scale (CybersecIoT) - 150 trials (two-stage)"
            echo ""
            echo "Total: 400 optimization trials across all groups"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# GROUP CONFIGURATIONS
# ============================================================================

# Trial budgets (from paper)
declare -A TRIAL_BUDGETS
TRIAL_BUDGETS[1]=75
TRIAL_BUDGETS[2]=75
TRIAL_BUDGETS[3]=100
TRIAL_BUDGETS[4]=150

# Group names
declare -A GROUP_NAMES
GROUP_NAMES[1]="Synthetic PESs"
GROUP_NAMES[2]="Real-life Low-Moderate Complexity"
GROUP_NAMES[3]="Real-life High Complexity"
GROUP_NAMES[4]="Large-scale PESs (CybersecIoT)"

# Objective metrics
declare -A OBJECTIVES
OBJECTIVES[1]="ARI"
OBJECTIVES[2]="Silhouette"
OBJECTIVES[3]="Silhouette"
OBJECTIVES[4]="ARI"

# Optimization modes
declare -A MODES
MODES[1]="online-online"
MODES[2]="online-online"
MODES[3]="online-online"
MODES[4]="offline-online"

echo "=============================================================="
echo "DupliMend Bayesian Optimization Workflow"
echo "=============================================================="
echo "Timestamp: $TIMESTAMP"
echo "Group: $GROUP"
if [ "$GROUP" != "all" ]; then
    echo "Group Name: ${GROUP_NAMES[$GROUP]}"
    echo "Trials: ${TRIAL_BUDGETS[$GROUP]}"
    echo "Objective: ${OBJECTIVES[$GROUP]}"
    echo "Mode: ${MODES[$GROUP]}"
fi
echo "=============================================================="

# ============================================================================
# EXTRACT ONLY MODE
# ============================================================================

if [ "$EXTRACT_ONLY" = true ]; then
    echo ""
    echo "Extracting optimal parameters from existing results..."
    echo ""

    if [ "$GROUP" = "all" ]; then
        python extract_best_params.py --all --latex
    else
        python extract_best_params.py --group "$GROUP" --command --latex
    fi

    exit 0
fi

# ============================================================================
# STEP 1: Bayesian Optimization
# ============================================================================

echo ""
echo "STEP 1: Running Bayesian Optimization..."
echo "=============================================================="

if [ "$GROUP" = "all" ]; then
    echo "Running optimization for ALL groups (400 total trials)"
    echo "  - Group 1: 75 trials (Synthetic PESs)"
    echo "  - Group 2: 75 trials (Real-life low-moderate)"
    echo "  - Group 3: 100 trials (Real-life high complexity)"
    echo "  - Group 4: 150 trials (CybersecIoT, two-stage)"
    echo ""
    python optimize_duplimend_bayesian.py --group all
else
    echo "Running optimization for Group $GROUP: ${GROUP_NAMES[$GROUP]}"
    echo "Trials: ${TRIAL_BUDGETS[$GROUP]}"
    echo "Objective: ${OBJECTIVES[$GROUP]}"
    echo "Mode: ${MODES[$GROUP]}"
    echo ""

    if [ "$GROUP" = "4" ]; then
        echo "Note: Group 4 uses two-stage optimization:"
        echo "  - Stage 1: Autoencoder architecture (100 trials)"
        echo "  - Stage 2: Clustering parameters (50 trials)"
        echo ""
    fi

    python optimize_duplimend_bayesian.py --group "$GROUP"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Optimization failed!"
    exit 1
fi

echo ""
echo "Optimization complete!"

# ============================================================================
# STEP 2: Extract Best Parameters
# ============================================================================

echo ""
echo "STEP 2: Extracting Best Parameters..."
echo "=============================================================="

if [ "$GROUP" = "all" ]; then
    python extract_best_params.py --all --latex
else
    python extract_best_params.py --group "$GROUP" --command --latex
fi

# ============================================================================
# STEP 3: Summary
# ============================================================================

echo ""
echo "=============================================================="
echo "WORKFLOW COMPLETE!"
echo "=============================================================="
echo ""
echo "Results locations:"
echo "  - Optimization results: evaluation_results/bayesian_optimization/"
echo "  - Group CSV files: bayesian_optimization_results_group{1,2,3,4}.csv"
echo "  - Combined results: bayesian_optimization_results_all_groups.csv"
echo ""
echo "To view optimal parameters:"
echo "  python extract_best_params.py --all"
echo "  python extract_best_params.py --group $GROUP --command"
echo ""
echo "To generate main.py command with optimal parameters:"
echo "  python extract_best_params.py --group $GROUP --command"
echo ""
echo "Hyperparameter search space (Table 3 from paper):"
echo "  - Autoencoder: layers, size, latent_dim, batch, dropout, lr, noise, sparsity"
echo "  - Clustering: variance_threshold, merge_threshold"
echo "  - Online adaptation: cluster_reg_weight, memory_reg_weight"
echo "  - Context: control_flow_context_window"
echo ""
echo "Data splitting: 60% train / 20% validation / 20% test (temporal)"
echo ""
echo "Done!"
