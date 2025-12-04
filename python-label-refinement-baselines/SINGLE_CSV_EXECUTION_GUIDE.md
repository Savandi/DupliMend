# Single CSV File Execution Guide

This document describes how to execute the baseline algorithms (Label Refinement and PM-Label-Splitting) on single CSV event log files that include ground truth labels. This execution mode is used for synthetic datasets where true activity labels are known, enabling calculation of both conformance and clustering quality metrics.

---

## Execution Context

Single CSV execution differs from folder-based execution in that it processes one CSV file at a time rather than batch-processing multiple files. However, the underlying methodology and metrics remain the same. This mode is particularly useful for:

- Testing baseline performance on specific datasets
- Evaluating new synthetic logs individually
- Running experiments with different CSV column configurations

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

## Configuration

All configuration is centralized in `config/config.py`. The shell scripts automatically read settings from this file.

### Configuration Location

Edit lines 308-343 in `config/config.py` (separate sections for Label Refinement and PM-Label-Splitting).

### Required Settings for Single CSV

```python
"data_path_real": r"U:\Research\...\document_review_process.csv",
"csv_config_name": "document_review_config",
"has_ground_truth": True,
"ground_truth_path": r"U:\Research\...\document_review_process_groundtruth.csv",
"ground_truth_activity_column": "ground_truth_activity",
"event_id_column": "EventID",
```

### Configuration Parameters Explained

**data_path_real**
- Path to the main CSV event log file
- Use Windows path format (scripts convert to WSL paths automatically)
- This parameter is specifically for single file execution (vs. `data_path_synthetic` for folder-based)

**csv_config_name**
- References a configuration section in `csv_config.py`
- Defines column mappings for the CSV file
- Two predefined configs available: `"ipalia_config"` and `"document_review_config"`
- Set to `None` for XES files

**has_ground_truth**
- Boolean flag indicating whether ground truth labels are available
- Must be `True` for synthetic datasets to enable clustering metrics
- When `False`, only process mining metrics are calculated

**ground_truth_path**
- Path to CSV file containing ground truth labels
- Must contain EventID column for matching to main log

**ground_truth_activity_column**
- Column name in ground truth CSV containing true activity labels
- Used for calculating ARI, NMI, and entropy metrics

**event_id_column**
- Column name used to match events between main log and ground truth
- Must exist in both main CSV and ground truth CSV

---

## CSV Column Configuration

The `csv_config.py` file defines how CSV columns map to event log attributes. The shell scripts automatically update the active configuration based on `csv_config_name`.

### Available Configurations

**ipalia_config:**
```python
{
    "case_id": "CaseID",
    "activity": "Activity",
    "timestamp": "Timestamp",
    "event_id": "EventID"
}
```

**document_review_config:**
```python
{
    "case_id": "case_id",
    "activity": "activity",
    "timestamp": "timestamp",
    "event_id": "EventID"
}
```

When `run_single_csv.sh` executes, it automatically updates line 38 of `csv_config.py` to activate the specified configuration.

---

## Execution Procedure (WSL Environment)

### Label Refinement

```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
./run_single_csv.sh
```

**Parameter Space:** 25 combinations
- 5 variant thresholds (0.0, 0.25, 0.5, 0.75, 1.0)
- 5 unfolding thresholds (0.0, 0.25, 0.5, 0.75, 1.0)
- Total: 5 × 5 = 25 experiments

**Estimated Runtime:** 10-30 minutes depending on log size

### PM-Label-Splitting

```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
./run_single_csv.sh
```

**Parameter Space:** 45 combinations
- 3 window sizes (1, 3, 5)
- 3 distance metrics (EDIT_DISTANCE, SET_DISTANCE, MULTISET_DISTANCE)
- 5 thresholds (0.0, 0.25, 0.5, 0.75, 1.0)
- Total: 3 × 3 × 5 = 45 experiments

**Estimated Runtime:** 20-45 minutes depending on log size

### Shell Script Operations

Both `run_single_csv.sh` scripts perform:

1. **Configuration Extraction**: Read settings from `config/config.py` using Python
2. **CSV Config Update**: Modify `csv_config.py` to activate the specified column mapping
3. **Path Conversion**: Convert Windows paths to WSL format (e.g., `U:\...` → `/mnt/u/...`)
4. **Dataset Name Derivation**: Extract filename without extension for output organization
5. **Validation**: Verify CSV file and ground truth file exist
6. **Execution**: Run baseline with all parameter combinations
7. **Results Organization**: Save to dataset-specific subdirectory

---

## Output Structure

Results are saved in a hierarchical directory structure:

```
results/
└── {dataset_name}/
    ├── {dataset_name}_result_-1.csv           # Label Refinement
    └── {dataset_name}_VARIANTS_ENHANCED.csv   # PM-Label-Splitting
```

**Example for document_review_process:**
```
results/document_review_process/document_review_process_result_-1.csv
results/document_review_process/document_review_process_VARIANTS_ENHANCED.csv
```

This structure matches the folder-based execution output format, ensuring consistency across execution modes.

---

## Metrics Calculated

Since ground truth is available (`has_ground_truth: True`), the following metrics are calculated:

### Process Mining Conformance Metrics
- **Precision**: Measures unnecessary behavior allowed by the model
- **Fitness**: Measures how well the model can replay the log
- **Simplicity**: Measures model structural complexity
- **Generalization**: Measures model's ability to generalize
- **F-score**: Harmonic mean of precision and fitness

### Clustering Quality Metrics
- **ARI (Adjusted Rand Index)**: Similarity between predicted and ground truth clusterings
- **NMI (Normalized Mutual Information)**: Information-theoretic clustering similarity
- **Expected Entropy (Clusters Perspective)**: Purity of predicted clusters
- **Expected Entropy (Labels Perspective)**: Consistency of label assignments

These metrics enable comprehensive evaluation of label refinement quality.

---

## Ground Truth File Format

The ground truth CSV must follow this structure:

```csv
EventID,ground_truth_activity
1,Review_Senior
2,Review_Junior
3,Approve_Manager
4,Review_Senior
...
```

**Requirements:**
- Must contain the EventID column (or column specified in `event_id_column`)
- Must contain ground truth labels (column specified in `ground_truth_activity_column`)
- EventIDs must match those in the main event log CSV
- One row per event in the main log

---

## Switching Between Datasets

To switch from one dataset to another, edit `config/config.py` in **both** baseline sections:

**Current Configuration (document_review_process):**
```python
"data_path_real": r"U:\...\document_review_process.csv",
"csv_config_name": "document_review_config",
"ground_truth_path": r"U:\...\document_review_process_groundtruth.csv",
```

**To Switch to ipalia:**
```python
"data_path_real": r"U:\...\ipalia.csv",
"csv_config_name": "ipalia_config",
"ground_truth_path": r"U:\...\ipalia_groundtruth.csv",
```

Ensure both Label Refinement and PM-Label-Splitting sections are updated identically.

---

## Execution Modes Comparison

The project supports three distinct execution modes:

| Mode | Script | Input Format | Ground Truth | Metrics |
|------|--------|--------------|--------------|---------|
| Folder-based | `run_specific_folders.sh` | Directory of XES files | Yes | All |
| Single CSV | `run_single_csv.sh` | Single CSV file | Yes | All |
| Real-world | `run_real_log.sh` | Single XES file | No | Process mining only |

This document covers the second mode (Single CSV), which combines the ground truth availability of folder-based execution with the single-file simplicity of real-world execution.

---

## Expected Behavior

**Successful Execution:**
1. Scripts read configuration without errors
2. CSV and ground truth files are located and loaded
3. Both baselines execute all parameter combinations sequentially
4. Progress messages display parameter values and intermediate results
5. CSV files are created in `results/{dataset_name}/` directory
6. Each CSV contains one row per parameter combination

**Output Verification:**
- CSV files should contain 25 rows (Label Refinement) or 45 rows (PM-Label-Splitting)
- Metric values should be in valid ranges (0.0 to 1.0 for most metrics)
- Different parameter combinations should produce varying metric values
- Higher quality refinements should show improved precision and/or fitness

---

## Troubleshooting

**Error: CSV file not found**
- Verify `data_path_real` path in `config.py`
- Check that the CSV file exists at the specified location
- Ensure path uses Windows format (backslashes), not WSL format

**Error: Ground truth file not found**
- Verify `ground_truth_path` in `config.py`
- Ensure the ground truth CSV exists
- Check that EventID column exists in both files

**Error: Column not found in CSV**
- Verify `csv_config_name` matches your CSV structure
- Check that column names in `csv_config.py` match your CSV headers
- Consider creating a new config section for custom CSV formats

**Script execution error: "required file not found"**
- Run the line ending fix commands in the Prerequisites section
- This error indicates CRLF line endings in shell scripts

**Low metric scores**
- This may indicate poor label refinement quality for the parameter combination
- Review different parameter combinations to find optimal settings
- Consider the nature of your dataset (some logs are inherently difficult)
