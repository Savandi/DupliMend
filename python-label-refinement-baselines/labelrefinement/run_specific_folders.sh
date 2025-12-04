#!/bin/bash

# Process specific folders only - Edit the FOLDERS array below

echo "=========================================="
echo "Label Refinement - Specific Folders Mode"
echo "=========================================="

# Read paths from config.py and convert Windows paths to WSL format
echo "Reading paths from config.py..."
eval $(python3.11 -c "
import sys
import os
import re
import io

# Suppress all stdout from config import (it prints validation messages)
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Add config directory to path
config_dir = os.path.abspath('../../config')
sys.path.insert(0, config_dir)

def windows_to_wsl_path(windows_path):
    '''Convert Windows path to WSL path format'''
    if not windows_path:
        return windows_path

    # Convert backslashes to forward slashes
    path = windows_path.replace('\\\\', '/')

    # Extract drive letter and path (match: C:/path or C:\path)
    match = re.match(r'^([A-Za-z]):/(.*)$', path)

    if match:
        drive = match.group(1).lower()
        rest_of_path = match.group(2)
        return f'/mnt/{drive}/{rest_of_path}'
    else:
        return path

try:
    from config import evaluation_config

    # Get label refinement config
    label_config = evaluation_config.get('baseline_evaluation_config', {}).get('label_refinement', {})

    # Extract and convert paths
    data_path_synthetic = windows_to_wsl_path(label_config.get('data_path_synthetic', ''))
    results_dir_config = windows_to_wsl_path(label_config.get('results_dir', '../results'))

    # Extract dataset name from data path (last folder name)
    # e.g., from 'C:/Users/.../noImprInLoop_default_IMD' -> 'noImprInLoop_default_IMD'
    dataset_name = os.path.basename(data_path_synthetic.rstrip('/'))

    # Check if results path is a network drive (U:, Z:, etc.) which may not be accessible in WSL
    # Use local path instead for WSL execution
    if results_dir_config.startswith('/mnt/u') or results_dir_config.startswith('/mnt/z'):
        # Use local results directory for WSL
        results_dir = '../results'
    else:
        results_dir = results_dir_config

    # Restore stdout and output ONLY the export statements
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Output as shell export statements
    print(f\"export DATA_PATH_SYNTHETIC='{data_path_synthetic}'\")
    print(f\"export BASELINE_RESULTS_DIR='{results_dir}'\")
    print(f\"export DATASET_NAME='{dataset_name}'\")

except Exception as e:
    # Restore stdout/stderr in case of error
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Fallback to default paths if config fails
    print(\"export DATA_PATH_SYNTHETIC='/mnt/c/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD'\")
    print(\"export BASELINE_RESULTS_DIR='../results'\")
    print(\"export DATASET_NAME='noImprInLoop_default_OD'\")
")

# Check if we're using local paths
if [[ "$BASELINE_RESULTS_DIR" == "../results" ]] || [[ "$BASELINE_RESULTS_DIR" == "./results" ]]; then
    echo "ℹ️  Using local relative paths (config has network drive U: or Z:, using local instead)"
fi

echo "✓ Data path: $DATA_PATH_SYNTHETIC"
echo "✓ Dataset name: $DATASET_NAME"
echo "✓ Results will be saved to: $BASELINE_RESULTS_DIR/$DATASET_NAME/"

# ===================================================================
# EDIT THIS ARRAY: Add or remove folder names to process
#  cd /mnt/c/Users/drana/Documents/GitHub/DupliMend/python-label-refinement-baselines/labelrefinement
# ./run_specific_folders.sh
# noImprInLoop_default_OD
# ===================================================================
FOLDERS=(
#   "feb16-1625"
#   "feb17-1101"
#   "feb17-1124"
#   "feb17-1147"
#   "feb17-1236"
#   "feb18-1515"
#   "feb18-1645"
#   "feb27-1517"
#   "feb29-1439"
#   "feb29-1548"
#   "feb29-1614"
#   "feb29-1654"
#   "mrt02-1452"
#   "mrt03-1655"
#   "mrt03-2247"
#   "mrt04-0910"
  "mrt04-1607"
  "mrt04-1632"
  "mrt04-1713"
)
# ===================================================================

# ===================================================================
# noImprInLoop_default_IMD
# ===================================================================
# FOLDERS=(
#   "feb19-1230"
#   "feb19-1333"
#   "feb19-1404"
#   "feb19-1414"
#   "feb19-1420"
#   "feb19-1503"
#   "feb19-1517"
#   "feb23-0904"
#   "feb23-0914"
#   "feb24-0914"
#   "feb29-1439"
#   "feb29-1500"
#   "feb29-1608"
#   "feb29-1627"
#   "mrt02-1452"
#   "mrt04-1802"
#   "mrt04-2002"
#   "mrt04-2029"
#   "mrt04-2053"
#   "mrt05-0004"
# )
# ===================================================================
echo ""
echo "Will process the following folders:"
for folder in "${FOLDERS[@]}"; do
    echo "  - $folder"
done

echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Process each folder
for folder in "${FOLDERS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing folder: $folder"
    echo "=========================================="

    python3.11 test_main.py 100 10 0 -1 --folder "$folder"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $folder"
    else
        echo "✗ Error processing: $folder"
    fi
done

echo ""
echo "=========================================="
echo "All specified folders processed!"
echo "Check results in: $BASELINE_RESULTS_DIR/$DATASET_NAME/"
echo "=========================================="
