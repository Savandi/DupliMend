# Folder-Based Execution Guide - Synthetic Event Logs

This document describes how to execute the baseline algorithms (Label Refinement and PM-Label-Splitting) on folder-structured synthetic event log datasets. This execution mode is designed for batch processing multiple XES files organized in a hierarchical directory structure, where each folder contains synthetic event logs with ground truth process models.

---

## Execution Context

Folder-based execution is the primary mode for evaluating baselines on synthetic datasets. Unlike single-file execution, this mode processes multiple event logs sequentially, each organized in separate folders within a parent dataset directory. This structure is typical of synthetic benchmark datasets generated for process mining research.

### Typical Use Cases
- Batch evaluation on synthetic benchmark datasets
- Processing multiple experimental conditions or parameter settings
- Systematic comparison across different synthetic log configurations
- Large-scale experiments requiring automated folder traversal

---

## Prerequisites

### System Requirements

**Python Version:** Python 3.11 or higher recommended (Python 3.12+ supported)

**Operating System:** WSL (Windows Subsystem for Linux) environment

### Installation Steps

**1. Install Python Package Manager**
```bash
sudo apt update
sudo apt install -y python3-pip
```

**2. Install Python Dependencies**
```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines
pip3 install -r requirements.txt
```

**Required Packages:**
- pandas (data manipulation)
- pm4py (process mining algorithms)
- igraph (graph algorithms)
- matplotlib (visualization)
- python-louvain (community detection)
- networkx (graph operations)
- editdistance (string distance calculation)
- numpy (numerical operations)
- leidenalg (clustering)

**3. Fix Shell Script Line Endings (WSL Compatibility)**

Windows text editors create files with CRLF line endings, which cause execution errors in Unix environments. Convert to LF line endings:

```bash
# Label Refinement scripts
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
find . -name "*.sh" -exec chmod +x {} \;

# PM-Label-Splitting scripts
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
find . -name "*.sh" -exec chmod +x {} \;
```

This fixes the `cannot execute: required file not found` error by removing carriage returns and setting execute permissions.

---

## Dataset Structure

Folder-based execution expects a specific hierarchical directory structure:

```
noImprInLoop_default_OD/          # Root dataset directory (configured in config.py)
├── feb16-1625/                   # Individual experimental run folder
│   ├── log.xes                   # Event log file
│   ├── models/                   # Ground truth process models directory
│   │   ├── A_ModelGenfeb16-1625.ptml
│   │   ├── B_ModelGenfeb16-1625.ptml
│   │   ├── C_ModelGenfeb16-1625.ptml
│   │   └── ...
│   └── setting.txt               # Experimental parameters/configuration
├── feb17-0936/                   # Another experimental run
│   ├── log.xes
│   ├── models/
│   └── setting.txt
├── mrt02-1452/
│   └── ...
└── mrt03-1655/
    └── ...
```

**Directory Components:**

- **Root Dataset Directory**: Top-level directory containing all experimental runs (e.g., `noImprInLoop_default_OD`)
- **Run Folders**: Individual experimental conditions (e.g., `feb16-1625`, `feb17-0936`)
- **log.xes**: XES event log file for the run
- **models/**: Directory containing ground truth process models in PTML format
- **setting.txt**: Text file documenting experimental parameters for reproducibility

This structure enables systematic batch processing across multiple experimental conditions.

---

## Configuration

All configuration is centralized in `config/config.py`. Both baseline algorithms read their settings from this file.

### Configuration Location

Edit lines 308-343 in `config/config.py` (separate sections for Label Refinement and PM-Label-Splitting).

### Required Settings

```python
"baseline_evaluation_config": {
    "label_refinement": {
        "data_path_synthetic": r"C:\path\to\noImprInLoop_default_OD",
        "output_dir": r"U:\Research\Results\labelrefinement\outputs",
        "results_dir": r"U:\Research\Results\labelrefinement\results",
        "best_results_dir": r"U:\Research\Results\labelrefinement\best_results"
    },
    "pm_label_splitting": {
        "data_path_synthetic": r"C:\path\to\noImprInLoop_default_OD",
        "output_dir": r"U:\Research\Results\pm-label-splitting\outputs",
        "results_dir": r"U:\Research\Results\pm-label-splitting\results",
        "best_results_dir": r"U:\Research\Results\pm-label-splitting\best_results"
    }
}
```

### Configuration Parameters Explained

**data_path_synthetic**
- Path to root dataset directory containing multiple run folders
- Use Windows path format (scripts convert to WSL paths automatically)
- This is the parent directory of all run folders (e.g., `feb16-1625/`, `mrt03-1655/`)

**output_dir**
- Directory for intermediate outputs (text logs, models, visualizations)
- Each run creates subdirectories here

**results_dir**
- Directory for final CSV result files
- Organized by dataset and run folder names

**best_results_dir**
- Directory for summary files of best-performing configurations
- Contains aggregated results across parameter combinations

---

## Execution Methods

Two execution methods are available for processing folder-based datasets.

### Method 1: Process Specific Folders

This method allows selective processing of particular run folders by editing the `FOLDERS` array in the execution script.

**Label Refinement (25 parameter combinations per folder):**
```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
nano run_specific_folders.sh  # Edit FOLDERS array (lines 128-149)
./run_specific_folders.sh
```

**PM-Label-Splitting (45 parameter combinations per folder):**
```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
nano run_specific_folders.sh  # Edit FOLDERS array
./run_specific_folders.sh
```

**Example FOLDERS array configuration:**
```bash
FOLDERS=(
  "feb16-1625"
  "feb17-0936"
  "mrt02-1452"
  "mrt03-1655"
)
```

This approach is useful when:
- Testing specific experimental conditions
- Processing a subset of folders
- Debugging or validating results for particular runs

### Method 2: Process All Folders Automatically

This method automatically discovers and processes all folders in the dataset directory:

```bash
# Label Refinement
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
./start.sh

# PM-Label-Splitting
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
./start.sh
```

The scripts automatically:
- Read configuration from `config/config.py`
- Convert Windows paths to WSL format
- Scan the dataset directory for all subdirectories
- Process each folder sequentially with all parameter combinations
- Save results to organized output directories

This approach is useful when:
- Processing entire benchmark datasets
- Conducting comprehensive evaluations
- Ensuring all experimental conditions are processed uniformly

---

## Execution Parameters

### Label Refinement

**Parameter Space:** 25 combinations
- 5 variant thresholds: {0.0, 0.25, 0.5, 0.75, 1.0}
- 5 unfolding thresholds: {0.0, 0.25, 0.5, 0.75, 1.0}
- Total combinations: 5 × 5 = 25

**Estimated Runtime:** 5-20 minutes per folder (depends on log size and complexity)

### PM-Label-Splitting

**Parameter Space:** 45 combinations
- 3 window sizes: {1, 3, 5}
- 3 distance metrics: {EDIT_DISTANCE, SET_DISTANCE, MULTISET_DISTANCE}
- 5 thresholds: {0.0, 0.25, 0.5, 0.75, 1.0}
- Total combinations: 3 × 3 × 5 = 45

**Estimated Runtime:** 10-30 minutes per folder (depends on log size and complexity)

---

## Output Structure

Results are organized in a hierarchical directory structure:

```
results/
└── {dataset_name}/
    └── {folder_name}/
        ├── {folder_name}_result_{model_id}.csv      # Label Refinement
        └── {folder_name}_VARIANTS_ENHANCED.csv      # PM-Label-Splitting
```

**Example for `noImprInLoop_default_OD/feb16-1625`:**
```
results/noImprInLoop_default_OD/feb16-1625/feb16-1625_result_-1.csv
results/noImprInLoop_default_OD/feb16-1625/feb16-1625_VARIANTS_ENHANCED.csv
```

### CSV File Contents

Each CSV file contains one row per parameter combination with columns for:

**Process Mining Conformance Metrics:**
- Precision
- Fitness
- Simplicity
- Generalization
- F-score

**Clustering Quality Metrics** (synthetic data with ground truth):
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- Expected Entropy (Clusters Perspective)
- Expected Entropy (Labels Perspective)

**Experimental Parameters:**
- Threshold values
- Distance metrics (PM-Label-Splitting)
- Window sizes (PM-Label-Splitting)
- Runtime

---

## Resumable Execution

For large-scale batch processing, resumable execution scripts are available that automatically skip folders with existing results:

```bash
# Label Refinement
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
./run_specific_folders_resumable.sh

# PM-Label-Splitting
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
./run_specific_folders_resumable.sh
```

**Features:**
- Checks for existence of result files before processing
- Skips folders that have already been successfully processed
- Continues from interruption point if execution was terminated
- Useful for long-running batch jobs on unstable connections or shared systems

**Use Cases:**
- Processing datasets with 50+ folders
- Recovery from system crashes or interruptions
- Incremental processing of newly added folders

---

## Batch Processing Strategies

### For Large Datasets (100+ folders)

**Strategy 1: Batch Subdivision**
Divide folders into manageable batches of 10-20 folders:

```bash
# Batch 1: First 10 folders
FOLDERS=("feb16-1625" "feb17-0936" ... "mrt02-1452")
./run_specific_folders.sh

# Batch 2: Next 10 folders
FOLDERS=("mrt03-1655" "mrt04-0910" ... "mrt04-1607")
./run_specific_folders.sh
```

**Strategy 2: Resumable Processing**
Use the resumable script for automatic continuation:

```bash
./run_specific_folders_resumable.sh
```

This approach automatically processes only folders without existing results.

**Strategy 3: Resource Monitoring**
Use the monitored execution script to track system resource usage:

```bash
./run_specific_folders_monitored.sh
```

This variant logs memory and CPU usage during execution, useful for:
- Identifying resource bottlenecks
- Optimizing execution parameters
- Planning computational resource allocation

---

## Expected Behavior

**Successful Execution:**
1. Scripts read configuration from `config.py` without errors
2. Dataset directory is scanned for run folders
3. For each folder:
   - XES log is loaded and validated
   - All parameter combinations are executed sequentially
   - Results are written incrementally to CSV files
   - Progress messages display current parameter values
4. Final summary is displayed with processing statistics

**Output Verification:**
- Each folder should have corresponding CSV files in `results/` directory
- CSV row count should match parameter combination count (25 or 45)
- Metric values should be in valid ranges (0.0 to 1.0 for most metrics)
- Different parameter combinations should produce varying results

---

## Troubleshooting

**Error: Dataset directory not found**
- Verify `data_path_synthetic` in `config.py` is correct
- Ensure the directory exists and contains run folders
- Check that path uses Windows format (scripts handle WSL conversion)

**Error: log.xes not found in folder**
- Verify each run folder contains a `log.xes` file
- Check file naming matches exactly (case-sensitive in Linux)
- Ensure XES files are valid and not corrupted

**Script execution error: "required file not found"**
- Run the line ending fix commands in Prerequisites section
- This indicates CRLF line endings in shell scripts

**Incomplete results (missing rows in CSV)**
- Check execution output for errors during specific parameter combinations
- Review memory availability (process mining can be memory-intensive)
- Consider processing fewer folders simultaneously

**Long execution times**
- Execution time scales with log size and complexity
- Consider using resumable execution for overnight runs
- Process folders in smaller batches to manage runtime

---

## Execution Modes Comparison

The project supports three distinct execution modes:

| Mode | Script | Input Format | Organization | Ground Truth | Metrics |
|------|--------|--------------|--------------|--------------|---------|
| Folder-based | `run_specific_folders.sh` | Directory of XES files | Hierarchical folders | Yes | All |
| Single CSV | `run_single_csv.sh` | Single CSV file | Single file | Yes | All |
| Real-world | `run_real_log.sh` | Single XES file | Single file | No | Process mining only |

This document covers the first mode (folder-based), designed for systematic batch evaluation on synthetic benchmark datasets with multiple experimental runs.
