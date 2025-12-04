#!/bin/bash

# ============================================================================
# Run Label Refinement on Single CSV File with Ground Truth (WSL)
# Reads all configuration from config/config.py
# ============================================================================

echo "=========================================="
echo "Label Refinement - Single CSV (WSL)"
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
csv_file = win_to_wsl(lr_config.get('data_path_real', ''))
csv_config = lr_config.get('csv_config_name', 'document_review_config')
ground_truth_file = win_to_wsl(lr_config.get('ground_truth_path', ''))

# Extract dataset name from CSV filename
dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

# Output as shell exports
print(f\"export CSV_FILE='{csv_file}'\")
print(f\"export CSV_CONFIG='{csv_config}'\")
print(f\"export GROUND_TRUTH_FILE='{ground_truth_file}'\")
print(f\"export DATASET_NAME='{dataset_name}'\")
")

if [ -z "$CSV_FILE" ]; then
    echo "❌ Error: Could not read configuration from config.py"
    exit 1
fi

echo "✓ CSV file: $CSV_FILE"
echo "✓ CSV config: $CSV_CONFIG"
echo "✓ Ground truth: $GROUND_TRUTH_FILE"
echo "✓ Dataset name: $DATASET_NAME"
echo ""

# Validate files exist
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "❌ Error: Ground truth file not found: $GROUND_TRUTH_FILE"
    exit 1
fi

# Update csv_config.py active_config
echo "Setting csv_config.py active_config to: $CSV_CONFIG"
python3.11 -c "
import sys
sys.path.insert(0, '..')

# Read current config
with open('../csv_config.py', 'r') as f:
    lines = f.readlines()

# Update active_config line
with open('../csv_config.py', 'w') as f:
    for line in lines:
        if '\"active_config\"' in line and '#' not in line.split('\"active_config\"')[0]:
            f.write(f'    \"active_config\": \"$CSV_CONFIG\"  # Auto-updated by run_single_csv.sh\n')
        else:
            f.write(line)

print('✓ Updated csv_config.py')
"

if [ $? -ne 0 ]; then
    echo "❌ Error updating csv_config.py"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running Label Refinement..."
echo "Parameters: Full parameter space (121 combinations)"
echo "Mode: Synthetic (with ground truth metrics)"
echo "=========================================="
echo ""

# Export for Python to use
export DATASET_NAME

# Run label refinement with synthetic mode (calculates ARI, NMI, Expected Entropy)
python3.11 test_main.py 100000 10 0 -1 synthetic

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS: $DATASET_NAME processed"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/${DATASET_NAME}/${DATASET_NAME}_result_-1.csv"
else
    echo ""
    echo "=========================================="
    echo "❌ ERROR processing $DATASET_NAME"
    echo "=========================================="
    exit 1
fi
