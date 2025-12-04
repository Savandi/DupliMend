#!/bin/bash

# ============================================================================
# Run Label Refinement on Real-World XES Log WITHOUT Ground Truth (WSL)
# Reads all configuration from config/config.py
# Only calculates process mining metrics (Precision, Fitness, Simplicity, etc.)
# ============================================================================

echo "=========================================="
echo "Label Refinement - Real Log (No GT)"
echo "=========================================="

# Read configuration from config.py
echo "Reading configuration from config/config.py..."
eval $(python3.11 -c "
import sys
import os
sys.path.insert(0, '../../config')

from config import evaluation_config

# Get label refinement config
lr_config = evaluation_config.get('baseline_evaluation_config', {}).get('label_refinement', {})

# Convert Windows paths to WSL format
def win_to_wsl(path):
    if not path:
        return path
    path = path.replace(chr(92), '/')
    if len(path) > 1 and path[1] == ':':
        drive = path[0].lower()
        path = '/mnt/' + drive + path[2:]
    return path

# Extract values
log_file = win_to_wsl(lr_config.get('data_path_real', ''))
has_ground_truth = lr_config.get('has_ground_truth', False)

# Extract dataset name from log filename (without extension)
dataset_name = os.path.splitext(os.path.basename(log_file))[0]

# Output as shell exports
print(f\"export LOG_FILE='{log_file}'\")
print(f\"export HAS_GROUND_TRUTH='{has_ground_truth}'\")
print(f\"export DATASET_NAME='{dataset_name}'\")
")

if [ -z "$LOG_FILE" ]; then
    echo "❌ Error: Could not read configuration from config.py"
    exit 1
fi

echo "✓ Log file: $LOG_FILE"
echo "✓ Has ground truth: $HAS_GROUND_TRUTH"
echo "✓ Dataset name: $DATASET_NAME"
echo ""

# Validate file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Check ground truth flag
if [ "$HAS_GROUND_TRUTH" = "True" ]; then
    echo "⚠️  WARNING: has_ground_truth is True in config.py"
    echo "   For real-world logs without ground truth, set has_ground_truth: False"
    echo "   Otherwise clustering metrics (ARI, NMI) will be attempted"
    echo ""
fi

echo ""
echo "=========================================="
echo "Running Label Refinement..."
echo "Parameters: 25 combinations"
echo "Note: Only process mining metrics (Precision, Fitness, etc.)"
echo "=========================================="
echo ""

# Export for Python to use
export DATASET_NAME

# Run label refinement
python3.11 test_main.py 100000 10 0 -1 real

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS: $DATASET_NAME processed"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/${DATASET_NAME}/${DATASET_NAME}_result_-1.csv"
    echo ""
    echo "Metrics: Precision, Fitness, Simplicity, Generalization, F-score"
    echo "(No ARI, NMI, Entropy metrics - no ground truth)"
else
    echo ""
    echo "=========================================="
    echo "❌ ERROR processing $DATASET_NAME"
    echo "=========================================="
    exit 1
fi
