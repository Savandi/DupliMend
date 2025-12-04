#!/bin/bash

# Process specific folders only for PM-Label-Splitting - Edit the FOLDERS array below

echo "=========================================="
echo "PM-Label-Splitting - Specific Folders Mode"
echo "=========================================="

# Read ALL paths from config.py (single source of truth)
# Using pipe delimiter to handle spaces in paths
echo "Reading paths from config.py..."
IFS='|' read -r DATA_PATH_SYNTHETIC PM_LABEL_OUTPUTS PM_LABEL_RESULTS PM_LABEL_BEST_RESULTS DATASET_NAME < <(python3.11 -c "
import sys
import os
import io

# Suppress config import output
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Add config path
config_path = os.path.join(os.path.dirname('$PWD'), '..', 'config')
sys.path.insert(0, config_path)

try:
    from config import evaluation_config
    pm_config = evaluation_config.get('baseline_evaluation_config', {}).get('pm_label_splitting', {})

    data_path = pm_config.get('data_path_synthetic', r'C:\Users\drana\Downloads\Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all\data\noImprInLoop_default_OD')
    outputs_config = pm_config.get('output_dir', r'U:\Research\Projects\sef\stream_quality_drift\homonym_experiment\evaluation_results\baselines\pm_label_splitting\outputs')
    results_config = pm_config.get('results_dir', r'U:\Research\Projects\sef\stream_quality_drift\homonym_experiment\evaluation_results\baselines\pm_label_splitting\results')
    best_results_config = pm_config.get('best_results_dir', r'U:\Research\Projects\sef\stream_quality_drift\homonym_experiment\evaluation_results\baselines\pm_label_splitting\best_results')

    # Convert Windows paths to WSL paths
    def win_to_wsl(path):
        path = path.replace(chr(92), '/')
        if len(path) > 1 and path[1] == ':':
            drive = path[0].lower()
            path = '/mnt/' + drive + path[2:]
        return path

    # Convert paths
    data_path_wsl = win_to_wsl(data_path)
    outputs_wsl = win_to_wsl(outputs_config)
    results_wsl = win_to_wsl(results_config)
    best_results_wsl = win_to_wsl(best_results_config)

    # Extract dataset name from data path (last folder name)
    dataset_name = os.path.basename(data_path_wsl.rstrip('/'))

    # Check if paths are network drives (U:, Z:, etc.) which may not be accessible in WSL
    # Use local paths instead for WSL execution
    if outputs_wsl.startswith('/mnt/u') or outputs_wsl.startswith('/mnt/z'):
        outputs_wsl = './outputs'
    if results_wsl.startswith('/mnt/u') or results_wsl.startswith('/mnt/z'):
        results_wsl = './results'
    if best_results_wsl.startswith('/mnt/u') or best_results_wsl.startswith('/mnt/z'):
        best_results_wsl = './best_results'

    # Restore stdout and output paths
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Use pipe delimiter to handle spaces in paths
    print('|'.join([data_path_wsl, outputs_wsl, results_wsl, best_results_wsl, dataset_name]))
except Exception as e:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Fallback using pipe delimiter
    print('|'.join(['/mnt/c/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD', './outputs', './results', './best_results', 'noImprInLoop_default_OD']))
")

if [ -z "$DATA_PATH_SYNTHETIC" ] || [ -z "$PM_LABEL_OUTPUTS" ]; then
    echo "Error: Could not read paths from config.py"
    exit 1
fi

# Export for use by Python scripts
export DATA_PATH_SYNTHETIC
export DATASET_NAME
export BASELINE_RESULTS_DIR="$PM_LABEL_RESULTS"
export PM_LABEL_OUTPUTS
export PM_LABEL_RESULTS
export PM_LABEL_BEST_RESULTS

# Create base output directories if they don't exist
echo "Creating output directories..."
mkdir -p "$PM_LABEL_OUTPUTS"
mkdir -p "$PM_LABEL_RESULTS"
mkdir -p "$PM_LABEL_BEST_RESULTS"

# Create dataset-specific subdirectories
mkdir -p "$PM_LABEL_OUTPUTS/$DATASET_NAME"
mkdir -p "$PM_LABEL_RESULTS/$DATASET_NAME"
mkdir -p "$PM_LABEL_BEST_RESULTS/$DATASET_NAME"

# Check if we're using local paths
if [[ "$PM_LABEL_RESULTS" == "./results" ]] || [[ "$PM_LABEL_RESULTS" == "../results" ]]; then
    echo "ℹ️  Using local relative paths (config has network drive U: or Z:, using local instead)"
fi

echo "✓ Data path: $DATA_PATH_SYNTHETIC"
echo "✓ Dataset name: $DATASET_NAME"
echo "✓ Results will be saved to: $PM_LABEL_RESULTS/$DATASET_NAME/"
echo "✓ Outputs will be saved to: $PM_LABEL_OUTPUTS/$DATASET_NAME/"
echo "✓ Best results will be saved to: $PM_LABEL_BEST_RESULTS/$DATASET_NAME/"

# ===================================================================
# EDIT THIS ARRAY: Add or remove folder names to process
#  cd /mnt/c/Users/drana/Documents/GitHub/DupliMend/python-label-refinement-baselines/pm-label-splitting
# ./run_specific_folders.sh
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
#   "mrt04-1713"
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

# Process each specified folder
for folder in "${FOLDERS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing folder: $folder"
    echo "=========================================="

    # Check if folder exists
    if [ ! -d "$DATA_PATH_SYNTHETIC/$folder" ]; then
        echo "✗ Error: Folder '$folder' not found in $DATA_PATH_SYNTHETIC"
        continue
    fi

    # Get the index by asking main.py what folders it sees, then finding our folder
    echo "Finding folder index in main.py's view..."
    FOLDER_INDEX=$(python3.11 -c "
import os
import sys
data_folder = '$DATA_PATH_SYNTHETIC'
folder_names = sorted(os.listdir(data_folder))
# Only count directories that have a 'logs' subfolder (valid datasets)
valid_folders = [f for f in folder_names if os.path.isdir(os.path.join(data_folder, f, 'logs'))]
try:
    idx = valid_folders.index('$folder')
    print(idx)
except ValueError:
    print(-1)
")

    if [ "$FOLDER_INDEX" = "-1" ]; then
        echo "✗ Error: Folder '$folder' not found in valid datasets or missing 'logs' subdirectory"
        continue
    fi

    echo "Folder index: $FOLDER_INDEX"

    # Create folder-specific output directory within dataset subfolder
    echo "Creating folder-specific output directory: $PM_LABEL_OUTPUTS/$DATASET_NAME/$folder"
    mkdir -p "$PM_LABEL_OUTPUTS/$DATASET_NAME/$folder"

    echo "Running PM-Label-Splitting pipeline..."

    # Run the PM-Label-Splitting main.py with the folder index
    # The main.py script is in the parent directory
    # The main.py script expects: python3.11 main.py <folder_index> [real/sim] [data_folder]
    # We pass "synthetic" as mode and the full data path
    python3.11 ../main.py "$FOLDER_INDEX" "synthetic" "$DATA_PATH_SYNTHETIC"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $folder"
    else
        echo "✗ Error processing: $folder"
    fi
done

echo ""
echo "=========================================="
echo "All specified folders processed!"
echo "Check results in: $PM_LABEL_RESULTS/$DATASET_NAME/"
echo "=========================================="
