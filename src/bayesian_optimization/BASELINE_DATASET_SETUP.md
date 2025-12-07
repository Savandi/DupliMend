# Baseline Dataset Setup Guide

## Overview

This guide explains how to run DupliMend on the BPM2016 benchmark dataset used by the Label Refinement and PM-Label-Splitting baselines for fair comparison.

## Dataset Information

**Source**: "Handling Duplicated Tasks in Process Discovery by Refining Event Labels" (BPM 2016)

**Location**: `C:\Users\drana\Downloads\Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all\data\noImprInLoop_default_OD`

**Structure**:
- Each folder (e.g., `feb16-1625`) contains multiple log variants (A-Q)
- Each log has 3 files:
  - `X_1_Log.xes.gz` = Ground truth (original activities)
  - `X_1_LogD_Sequence_feb16-1625.xes.gz` = Training data (imprecise labels)
  - `X_1_LogR_Sequence_feb16-1625.xes.gz` = Refined (after baseline processing)

**Imprecise Labels** (from `setting.txt`):
```
Log A: Activities C,B,E are all labeled as "D" in LogD
Log B: Activities M,K,J are all labeled as "N" in LogD
Log C: Activities C,I,H are all labeled as "D" in LogD
```

## Step 1: Convert XES to CSV

DupliMend uses CSV format. The XES files must be converted before processing.

### Run the Converter

**Single log conversion**:
```bash
cd C:\Users\drana\Documents\GitHub\DupliMend

python src/evaluation/xes_to_csv_baseline.py \
  --folder "C:/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD/feb16-1625" \
  --output "C:/Users/drana/Downloads/baseline_csv_logs/feb16-1625" \
  --log "A_1"
```

**Batch conversion (all logs A through Q)**:
```bash
python src/evaluation/xes_to_csv_baseline.py \
  --folder "C:/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD/feb16-1625" \
  --output "C:/Users/drana/Downloads/baseline_csv_logs/feb16-1625" \
  --all
```

### Output Files

- `A_1_LogD_train.csv` - Training data with imprecise labels
- `A_1_ground_truth.csv` - Ground truth for evaluation

### CSV Format

**Training file (A_1_LogD_train.csv)**:
```csv
EventID,CaseID,Activity,OrgLabel,Resource,Lifecycle,Timestamp
1,case_0,A,A,artificial,complete,1
2,case_0,D,B,artificial,complete,2   <- Imprecise! Should be B
3,case_0,D,D,artificial,complete,3
4,case_0,D,E,artificial,complete,4   <- Imprecise! Should be E
5,case_0,D,C,artificial,complete,5   <- Imprecise! Should be C
```

**Ground truth file (A_1_ground_truth.csv)**:
```csv
EventID,ground_truth_activity
1,A
2,B
3,D
4,E
5,C
```

## Step 2: Update Configuration

The config has been updated with baseline dataset paths. Key settings:

```python
# Training data location
training_mode_config = {
    "training_folder": r"C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625",
    "training_file_pattern": "*_LogD_train.csv",
    "default_input_file": r"C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625\A_1_LogD_train.csv",
}

# Evaluation configuration
evaluation_config = {
    "single_evaluation_config": {
        "default_ground_truth_path": r"C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625\A_1_ground_truth.csv",
        "default_activity": "D",  # The homonymous label
        "control_flow_column": "Activity",
        "control_flow_column_ground_truth": "ground_truth_activity",
        "case_id_column": "CaseID",
        "event_id_column": "EventID",
    }
}

# Column mappings
control_flow_column = 'Activity'
case_id_column = 'CaseID'
event_id_column = 'EventID'
timestamp_column = 'Timestamp'
resource_column = 'Resource'
```

## Step 3: Run DupliMend

### Single Log Execution

```bash
python main.py
```

This will:
1. Load `A_1_LogD_train.csv` (with imprecise label "D")
2. Train autoencoder and cluster the events
3. Refine label "D" into sub-activities (hopefully D, B, C, E)
4. Save results to tracking directory

### Multiple Log Processing

To test different logs, update `config.py`:

```python
# For Log B_1 (where M,K,J are labeled as N):
"default_input_file": r"C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625\B_1_LogD_train.csv",
"default_ground_truth_path": r"C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625\B_1_ground_truth.csv",
"default_activity": "N",  # The homonymous label for log B
```

## Step 4: Evaluate Results

After DupliMend finishes, run evaluation:

```bash
python src/evaluation/evaluate_single_test_file.py \
  --tracking_dir "path/to/tracking_YYYYMMDD_HHMMSS" \
  --ground_truth "C:/Users/drana/Downloads/baseline_csv_logs/feb16-1625/A_1_ground_truth.csv" \
  --activity "D" \
  --output_dir "U:/Research/Projects/sef/stream_quality_drift/homonym_experiment/evaluation_results/duplimend_baseline"
```

This computes:
- Expected Entropy (Clusters perspective)
- Expected Entropy (Labels perspective)
- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)
- Precision, Fitness, F-score

## Step 5: Compare with Baselines

DupliMend results can be compared with baseline results:

**Label Refinement baseline results**:
`python-label-refinement-baseline/results/`

**PM-Label-Splitting baseline results**:
`U:/Research/Projects/sef/stream_quality_drift/homonym_experiment/evaluation_results/pm-label-splitting/`

## Imprecise Activities by Log

```
A: D (composed of C, B, E)
B: N (composed of M, K, J)
C: D (composed of C, I, H)
D: J (composed of H, G, F)
E: C (composed of O, K, H)
F: N (composed of K, I, H)
G: I (composed of H, G, F)
H: N (composed of M, J, E)
I: O (composed of N, M, L)
J: O (composed of N)
K: D (composed of O, N, J)
L: O (composed of L, K, I)
M: B (composed of N, K, I)
N: O (composed of N, H, G)
O: D (composed of K, J, I)
P: L (composed of I, G, F)
Q: B (composed of A, O, M)
```

## Dataset Characteristics

1. **No real timestamps**: Events are ordered sequentially (1, 2, 3...)
2. **Synthetic data**: Generated from process trees
3. **Ground truth available**: OrgLabel column has true activity names
4. **Single imprecise label per log**: Only one activity is homonymous
5. **No resource variation**: All events have "artificial" resource

## Expected Performance

For perfect clustering:
- **Expected Entropy = 0.0** (no confusion in clusters)
- **NMI = 1.0** (perfect agreement with ground truth)
- **ARI = 1.0** (perfect cluster matching)

## Troubleshooting

**Issue**: Converter fails with "ModuleNotFoundError: No module named 'pm4py'"

**Solution**: Install pm4py
```bash
pip install pm4py
```

**Issue**: Config shows wrong paths

**Solution**: Verify paths in `config.py` point to:
- Training folder: `C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625`
- Ground truth: `C:\Users\drana\Downloads\baseline_csv_logs\feb16-1625\A_1_ground_truth.csv`

**Issue**: No clusters found for activity "D"

**Solution**: Check that:
1. `control_flow_column = 'Activity'` (not 'concept:name')
2. `default_activity = "D"` matches the imprecise label in the log

## Quick Start

```bash
# 1. Convert XES to CSV
cd C:\Users\drana\Documents\GitHub\DupliMend
python src/evaluation/xes_to_csv_baseline.py \
  --folder "C:/Users/drana/Downloads/Handling Duplicated Tasks in Process Discovery by Refining Event Labels (BPM2016)_1_all/data/noImprInLoop_default_OD/feb16-1625" \
  --output "C:/Users/drana/Downloads/baseline_csv_logs/feb16-1625" \
  --log "A_1"

# 2. Run DupliMend
python main.py

# 3. Evaluate results
python src/evaluation/evaluate_single_test_file.py \
  --tracking_dir "path/to/output" \
  --ground_truth "C:/Users/drana/Downloads/baseline_csv_logs/feb16-1625/A_1_ground_truth.csv" \
  --activity "D"
```
