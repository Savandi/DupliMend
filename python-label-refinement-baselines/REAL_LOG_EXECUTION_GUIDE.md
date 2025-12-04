# Real-World Event Log Execution Guide

This document describes how to execute the baseline algorithms (Label Refinement and PM-Label-Splitting) on real-world XES event logs that do not have ground truth labels. This scenario is typical when evaluating process mining approaches on industrial datasets such as the BPI Challenge logs.

---

## Execution Context

Real-world event logs differ from synthetic datasets in a key aspect: they lack ground truth information about the true activity labels. Therefore, the evaluation focuses exclusively on process mining conformance metrics rather than clustering quality metrics.

### Metrics Available

**Process Mining Metrics:**
- Precision
- Fitness
- Simplicity
- Generalization
- F-score

**Clustering Metrics (Not Available):**
- Adjusted Rand Index (ARI) - requires ground truth
- Normalized Mutual Information (NMI) - requires ground truth
- Expected Entropy Clusters - requires ground truth
- Expected Entropy Labels - requires ground truth

---

## Configuration

All configuration is centralized in `config/config.py`. Both baseline algorithms read their settings from this file.

### Configuration Location

Edit lines 319-322 and 346-349 in `config/config.py` (separate sections for Label Refinement and PM-Label-Splitting).

### Required Settings

For real-world XES logs, configure:

```python
"data_path_real": r"U:\Research\...\BPI_Challenge_2013_closed_problems.xes",
"csv_config_name": None,  # Not needed for XES files
"has_ground_truth": False,  # Critical: prevents clustering metric calculation
```

### Important Notes

1. **data_path_real**: Path to the XES event log file (use Windows path format, scripts convert to WSL)
2. **csv_config_name**: Set to `None` for XES files (only used for CSV inputs)
3. **has_ground_truth**: Must be set to `False` to prevent attempting clustering metric calculations

If `has_ground_truth` is mistakenly set to `True`, the pipeline will attempt to calculate ARI, NMI, and entropy metrics, which will fail or produce meaningless results.

---

## Execution Procedure (WSL Environment)

The execution uses shell scripts that automatically read configuration, convert paths, and execute the baselines.

### Label Refinement

```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/labelrefinement
./run_real_log.sh
```

**Parameter Space:** 25 combinations
- 5 variant thresholds × 5 unfolding thresholds

### PM-Label-Splitting

```bash
cd /mnt/c/Git/DupliMend/python-label-refinement-baselines/pm-label-splitting
./run_real_log.sh
```

**Parameter Space:** 45 combinations
- 3 window sizes × 3 distance metrics × 5 thresholds

---

## Shell Script Behavior

Both `run_real_log.sh` scripts perform the following operations:

1. **Configuration Reading**: Extract settings from `config/config.py` using Python
2. **Path Conversion**: Convert Windows paths (e.g., `U:\Research\...`) to WSL format (`/mnt/u/research/...`)
3. **Dataset Name Extraction**: Derive dataset name from filename (excluding extension)
4. **Validation**: Check that the log file exists at the specified path
5. **Ground Truth Warning**: Display warning if `has_ground_truth` is `True` (should be `False` for real logs)
6. **Execution**: Run the baseline algorithm with appropriate parameters
7. **Results Reporting**: Display location of output CSV file

---

## Output Structure

Results are saved to dataset-specific subdirectories:

```
results/{dataset_name}/{dataset_name}_result_-1.csv           # Label Refinement
results/{dataset_name}/{dataset_name}_VARIANTS_ENHANCED.csv   # PM-Label-Splitting
```

**Example for BPI Challenge 2013:**
```
results/BPI_Challenge_2013_closed_problems/BPI_Challenge_2013_closed_problems_result_-1.csv
results/BPI_Challenge_2013_closed_problems/BPI_Challenge_2013_closed_problems_VARIANTS_ENHANCED.csv
```

### CSV Contents

Each row represents one parameter combination with columns for:
- Dataset name
- Parameter values (threshold, distance metric, window size, etc.)
- Precision
- Fitness
- Simplicity
- Generalization
- F-score
- Runtime

Clustering metric columns (ARI, NMI, Expected Entropy) will be empty or zero when `has_ground_truth` is `False`.

---

## Execution Comparison

The project includes three execution modes for different data types:

| Script | Data Type | Ground Truth | Input Format | Metrics |
|--------|-----------|--------------|--------------|---------|
| `run_specific_folders.sh` | Synthetic | Yes | Folder of XES files | All metrics |
| `run_single_csv.sh` | Synthetic | Yes | Single CSV file | All metrics |
| `run_real_log.sh` | Real-world | No | Single XES file | Process mining only |

This document covers the third mode (`run_real_log.sh`), designed for evaluating on industrial/real-world datasets.

---

## Example Configuration

Complete example for BPI Challenge 2013 log:

```python
"baseline_evaluation_config": {
    "label_refinement": {
        "data_path_real": r"U:\Research\Datasets\BPI\BPI_Challenge_2013_closed_problems.xes",
        "csv_config_name": None,
        "has_ground_truth": False,
        "output_dir": r"U:\Research\Results\labelrefinement\outputs",
        "results_dir": r"U:\Research\Results\labelrefinement\results",
        "best_results_dir": r"U:\Research\Results\labelrefinement\best_results"
    },
    "pm_label_splitting": {
        "data_path_real": r"U:\Research\Datasets\BPI\BPI_Challenge_2013_closed_problems.xes",
        "csv_config_name": None,
        "has_ground_truth": False,
        "output_dir": r"U:\Research\Results\pm-label-splitting\outputs",
        "results_dir": r"U:\Research\Results\pm-label-splitting\results",
        "best_results_dir": r"U:\Research\Results\pm-label-splitting\best_results"
    }
}
```

---

## Expected Runtime

Runtime depends on log size and complexity:
- Small logs (< 1000 events): 5-15 minutes per baseline
- Medium logs (1000-10000 events): 15-60 minutes per baseline
- Large logs (> 10000 events): 1-3 hours per baseline

Both baselines execute all parameter combinations sequentially.

---

## Troubleshooting

**Error: Log file not found**
- Verify the path in `config.py` is correct
- Ensure the XES file exists at the specified location
- Check that the path uses proper Windows format (scripts handle WSL conversion)

**Warning: has_ground_truth is True**
- Change `has_ground_truth: False` in `config.py`
- This prevents attempting to calculate clustering metrics on real-world logs

**Empty results CSV**
- Check execution output for errors
- Verify log file is valid XES format
- Ensure sufficient memory is available (process mining can be memory-intensive)
