import copy as c
import csv
import os
import sys
from pathlib import Path
from time import time
import tempfile
import shutil
import atexit

from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from pm4py.objects.process_tree.importer import importer as ptml_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

import test_epoch

# Import CSV support
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from csv_adapter import CSVToXESAdapter
    from csv_config import get_active_column_mapping
    CSV_SUPPORT_AVAILABLE = True
    print("[CSV_SUPPORT] CSV adapter available")
except ImportError as e:
    print(f"[CSV_SUPPORT] CSV adapter not available: {e}")
    CSV_SUPPORT_AVAILABLE = False

# CONFIGURATION - Load from DupliMend config (silently to avoid verbose output)
try:
    import sys
    import os
    # Temporarily suppress stdout to avoid verbose DupliMend config printing
    import io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    sys.path.append('../../config')
    from config import evaluation_config

    # Restore stdout
    sys.stdout = original_stdout

    # Use DupliMend's label refinement config
    label_config = evaluation_config.get("baseline_evaluation_config", {}).get("label_refinement", {})

    DATA_PATH_SYNTHETIC = label_config.get("data_path_synthetic", "../data/noImprInLoop_default_OD")
    DATA_PATH_REAL = label_config.get("data_path_real", "../data/noImprInLoop_default_OD/feb16-1625/logs/A_1_Log.xes.gz")

except (ImportError, Exception) as e:
    # Fallback if config not available
    DATA_PATH_SYNTHETIC = "../data/noImprInLoop_default_OD"
    DATA_PATH_REAL = "../data/noImprInLoop_default_OD/feb16-1625/logs/A_1_Log.xes.gz"
    label_config = {}

# Check environment variable override for data paths (for WSL compatibility)
if 'DATA_PATH_SYNTHETIC' in os.environ:
    DATA_PATH_SYNTHETIC = os.environ['DATA_PATH_SYNTHETIC']
    print(f"[CONFIG] Using data path from environment: {DATA_PATH_SYNTHETIC}")

# Check command line arguments for data type selection
if len(sys.argv) > 5 and sys.argv[5] == "real":
    DATA_PATH = DATA_PATH_REAL
    IS_REAL_LIFE_LOG = True
    print(f"🔍 Using REAL data: {DATA_PATH}")
else:
    DATA_PATH = DATA_PATH_SYNTHETIC
    IS_REAL_LIFE_LOG = False
    print(f"🔍 Using SYNTHETIC data: {DATA_PATH}")

log_size_parameter = int(sys.argv[1])
batch_size_parameter = int(sys.argv[2])
experiment_nr_parameter = int(sys.argv[3])
start_data_set_size_parameter = int(sys.argv[4])

# Optional filtering parameters
# Usage: python test_main.py 1000 10 1 -1 --max-datasets 50
# Usage: python test_main.py 1000 10 1 -1 --per-folder 3
FILTER_FOLDER = None
FILTER_LOG_TYPES = None
MAX_DATASETS = None
PER_FOLDER_LIMIT = None

for i, arg in enumerate(sys.argv):
    if arg == "--folder" and i + 1 < len(sys.argv):
        FILTER_FOLDER = sys.argv[i + 1]
        print(f"🔍 FILTER: Only processing folder: {FILTER_FOLDER}")
    elif arg == "--logs" and i + 1 < len(sys.argv):
        FILTER_LOG_TYPES = sys.argv[i + 1].split(",")
        print(f"🔍 FILTER: Only processing log types: {FILTER_LOG_TYPES}")
    elif arg == "--max-datasets" and i + 1 < len(sys.argv):
        MAX_DATASETS = int(sys.argv[i + 1])
        print(f"🔍 LIMIT: Processing maximum {MAX_DATASETS} datasets")
    elif arg == "--per-folder" and i + 1 < len(sys.argv):
        PER_FOLDER_LIMIT = int(sys.argv[i + 1])
        print(f"🔍 LIMIT: Processing {PER_FOLDER_LIMIT} datasets per folder")

import sys
sys.setrecursionlimit(10000)

# Track temporary directories for cleanup
_temp_directories = []

def cleanup_temp_directories():
    """Clean up all temporary directories created during execution"""
    for temp_dir in _temp_directories:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[CLEANUP] Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not remove {temp_dir}: {e}")
    _temp_directories.clear()

# Register cleanup function to run on exit
atexit.register(cleanup_temp_directories) 

def is_single_log_file(path):
    """Check if path points to a single log file"""
    return os.path.isfile(path) and (path.endswith('.xes') or path.endswith('.xes.gz') or path.endswith('.csv'))

def convert_csv_to_xes_if_needed(file_path):
    """Convert CSV to XES if needed, return XES path"""
    if not file_path.lower().endswith('.csv'):
        return file_path  # Already XES
    
    if not CSV_SUPPORT_AVAILABLE:
        raise Exception("CSV file provided but CSV support not available. Please ensure csv_adapter.py and csv_config.py are available.")
    
    print(f"[CSV_DETECTION] Detected CSV file: {file_path}")
    
    # Show column mapping being used
    column_mapping = get_active_column_mapping()
    print(f"[CSV_DETECTION] Using column mapping:")
    print(f"  Case ID: {column_mapping['case_id']}")
    print(f"  Activity: {column_mapping['activity']}")
    print(f"  Timestamp: {column_mapping['timestamp']}")
    print(f"  Resource: {column_mapping['resource']}")
    
    # Create temporary XES file
    temp_dir = tempfile.mkdtemp()
    _temp_directories.append(temp_dir)  # Track for cleanup
    temp_xes_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(file_path))[0] + '_converted.xes')

    print(f"[CSV_CONVERSION] Converting to temporary XES: {temp_xes_path}")

    # Convert CSV to XES
    adapter = CSVToXESAdapter()
    adapter.csv_to_xes(file_path, temp_xes_path)

    print(f"[CSV_CONVERSION] Conversion completed successfully")
    return temp_xes_path

def load_and_merge_ground_truth(event_log, ground_truth_path, ground_truth_activity_column, event_id_column):
    """Load ground truth from separate CSV file and merge it into event log as OrgLabel"""
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        print(f"[GROUND_TRUTH] Ground truth file not found: {ground_truth_path}")
        return event_log

    print(f"[GROUND_TRUTH] Loading ground truth from: {ground_truth_path}")

    # Load ground truth CSV
    import pandas as pd
    ground_truth_df = pd.read_csv(ground_truth_path)

    print(f"[GROUND_TRUTH] Loaded {len(ground_truth_df)} ground truth events")
    print(f"[GROUND_TRUTH] Ground truth columns: {list(ground_truth_df.columns)}")

    # Create mapping from EventID to ground truth activity
    event_id_to_gt = {}
    for _, row in ground_truth_df.iterrows():
        event_id = str(row[event_id_column])
        gt_activity = str(row[ground_truth_activity_column])
        event_id_to_gt[event_id] = gt_activity

    print(f"[GROUND_TRUTH] Created mapping for {len(event_id_to_gt)} events")

    # Merge ground truth into event log as OrgLabel
    merged_count = 0
    missing_count = 0

    for trace in event_log:
        for event in trace:
            # Get EventID from event (it might be in custom:EventID or EventID)
            event_id = None
            if f'custom:{event_id_column}' in event:
                event_id = str(event[f'custom:{event_id_column}'])
            elif event_id_column in event:
                event_id = str(event[event_id_column])

            if event_id and event_id in event_id_to_gt:
                event['OrgLabel'] = event_id_to_gt[event_id]
                merged_count += 1
            else:
                # If no ground truth found, use concept:name as fallback
                event['OrgLabel'] = event.get('concept:name', 'Unknown')
                missing_count += 1

    print(f"[GROUND_TRUTH] Merged {merged_count} events, {missing_count} events missing ground truth")

    return event_log

def create_single_log_dataset(log_path):
    """Create a dataset structure for a single real log"""
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} does not exist")
        return []

    # Convert CSV to XES if needed
    xes_log_path = convert_csv_to_xes_if_needed(log_path)

    log_name = os.path.splitext(os.path.basename(log_path))[0]

    # Get ground truth configuration from config
    has_ground_truth = label_config.get("has_ground_truth", False)
    ground_truth_path = label_config.get("ground_truth_path", None)
    ground_truth_activity_column = label_config.get("ground_truth_activity_column", "ground_truth_activity")
    event_id_column = label_config.get("event_id_column", "EventID")

    dataset = {
        "setting": (log_name, log_name),  # Use log_name for both (not "real_log")
        "event_log_path": xes_log_path,  # Use converted XES path
        "original_log_path": xes_log_path,  # Use same log as both imprecise and original
        "is_real_life_log": not has_ground_truth,  # False if we have ground truth (synthetic), True if real-life
        "ground_truth_path": ground_truth_path,  # Path to separate ground truth file
        "ground_truth_activity_column": ground_truth_activity_column,
        "event_id_column": event_id_column,
        # No xixi_log_path for real logs
        # No model_path - will be generated from log
    }

    return [dataset]

def main():
    # Check if DATA_PATH is a single log file or synthetic data folder
    if is_single_log_file(DATA_PATH):
        print(f"Processing single real log: {DATA_PATH}")
        complete_datasets = create_single_log_dataset(DATA_PATH)
        
        if not complete_datasets:
            return
            
    else:
        # Handle synthetic data (original behavior)
        directory = DATA_PATH
        print(f"Processing synthetic data from: {directory}")

        # DEBUGGING: Check if directory exists
        if not os.path.exists(directory):
            print(f"ERROR: Directory does not exist: {directory}")
            return

        # DEBUGGING: List all folders in the directory
        try:
            all_folders = os.listdir(directory)
            print(f"✓ Found {len(all_folders)} folders in data directory:")
            for folder in all_folders:
                print(f"  - {folder}")
        except Exception as e:
            print(f"ERROR: Could not list directory {directory}: {e}")
            return

        data = {}
        setting_ids = []
        folder_processed_count = 0

        for folder_name in all_folders:
            folder_path = os.path.join(directory, folder_name)

            # Skip if not a directory
            if not os.path.isdir(folder_path):
                print(f"⚠ Skipping {folder_name} (not a directory)")
                continue

            # Check for logs and models subdirectories
            logs_path = os.path.join(folder_path, "logs")
            models_path = os.path.join(folder_path, "models")

            if not os.path.exists(logs_path):
                print(f"⚠ Skipping {folder_name} (no 'logs' subdirectory)")
                continue

            if not os.path.exists(models_path):
                print(f"⚠ Folder {folder_name} has no 'models' subdirectory (models are optional)")

            folder_processed_count += 1
            print(f"\n📂 Processing folder: {folder_name}")

            log_folder_list = os.listdir(logs_path)
            print(f"  → Found {len(log_folder_list)} files in logs folder")

            for file_name in log_folder_list:
                setting_id = '_'.join(file_name.split("_")[:2])
                if setting_id not in setting_ids:
                    setting_ids.append((setting_id, folder_name))
                    data[(setting_id, folder_name)] = {}
                    print(f"     • New setting: {setting_id}")

        print(f"\n✓ Processed {folder_processed_count} folders, found {len(setting_ids)} unique settings")

        m = 0

        for folder_name in all_folders:
            folder_path = os.path.join(directory, folder_name)

            # Skip if not a directory or missing logs
            if not os.path.isdir(folder_path):
                continue

            logs_path = os.path.join(folder_path, "logs")
            models_path = os.path.join(folder_path, "models")

            if not os.path.exists(logs_path):
                continue

            log_folder_list = os.listdir(logs_path)
            model_folder_list = os.listdir(models_path) if os.path.exists(models_path) else []

            print(f"\n📝 Populating paths for folder: {folder_name}")

            for log_file_name in log_folder_list:
                setting = ('_'.join(log_file_name.split("_")[:2]), folder_name)
                if setting not in data:
                    print(f"  ⚠ Setting {setting} not in data dictionary, skipping")
                    continue

                data[setting]["setting"] = setting
                if "LogR" in log_file_name:
                    data[setting]["xixi_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)
                    print(f"  ✓ Found LogR (xixi): {log_file_name}")
                if "LogD" in log_file_name:
                    data[setting]["event_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)
                    print(f"  ✓ Found LogD (event): {log_file_name}")
                if "LogR" not in log_file_name and "LogD" not in log_file_name:
                    data[setting]["original_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)
                    print(f"  ✓ Found Log (original): {log_file_name}")

            for model_file_name in model_folder_list:
                m = m + 1
                # Extract prefix (e.g., "B" from "B_ModelGen...") and add "_1" to match log file naming
                setting = (model_file_name.split("_")[0] + "_1", folder_name)
                if setting in data.keys():
                    data[setting]["model_path"] = os.path.join(directory, folder_name, "models", model_file_name)
                    print(f"  ✓ Found model: {model_file_name}")

        # Filter complete datasets
        print(f"\n🔍 Filtering complete datasets (need event_log_path AND original_log_path)...")
        data2 = c.deepcopy(data)
        complete_datasets = []
        incomplete_datasets = []

        for d in data:
            has_event_log = "event_log_path" in data[d]
            has_original_log = "original_log_path" in data[d]
            num_keys = len(data[d])

            if num_keys >= 3 and has_event_log and has_original_log:
                complete_datasets.append(data[d])
                print(f"  ✅ COMPLETE: {d} - Keys: {list(data[d].keys())}")
            else:
                incomplete_datasets.append((d, data[d]))
                del data2[d]
                print(f"  ❌ INCOMPLETE: {d} - Keys: {list(data[d].keys())} (missing: {'' if has_event_log else 'event_log_path '}{'' if has_original_log else 'original_log_path'})")

        data = data2
        print(f"\n{'='*80}")
        print(f"📊 DATASET SUMMARY:")
        print(f"  Complete datasets: {len(complete_datasets)}")
        print(f"  Incomplete datasets: {len(incomplete_datasets)}")
        print(f"{'='*80}")

        if len(incomplete_datasets) > 0:
            print(f"\n⚠ WARNING: {len(incomplete_datasets)} datasets are incomplete.")
            print("  This usually means files are missing LogD or Log files.")
            print("  Check the logs folder structure in each folder.")

        # Apply filters if specified
        if FILTER_FOLDER or FILTER_LOG_TYPES:
            print(f"\n{'='*80}")
            print(f"🔍 APPLYING FILTERS:")
            print(f"  Before filtering: {len(complete_datasets)} datasets")

            filtered_datasets = []
            for dataset in complete_datasets:
                log_type, folder = dataset["setting"]

                # Check folder filter
                if FILTER_FOLDER and folder != FILTER_FOLDER:
                    continue

                # Check log type filter
                if FILTER_LOG_TYPES and log_type not in FILTER_LOG_TYPES:
                    continue

                filtered_datasets.append(dataset)
                print(f"  ✅ KEEP: {log_type} from {folder}")

            complete_datasets = filtered_datasets
            print(f"  After filtering: {len(complete_datasets)} datasets")
            print(f"{'='*80}")

            if len(complete_datasets) == 0:
                print("ERROR: No datasets match the filters. Check --folder and --logs parameters.")
                return

        # Apply per-folder limit if specified
        if PER_FOLDER_LIMIT is not None:
            print(f"\n{'='*80}")
            print(f"🔍 APPLYING PER-FOLDER DATASET LIMIT:")
            print(f"  Total complete datasets available: {len(complete_datasets)}")
            print(f"  Limiting to {PER_FOLDER_LIMIT} datasets per folder")

            # Group datasets by folder
            folder_datasets = {}
            for dataset in complete_datasets:
                log_type, folder = dataset["setting"]
                if folder not in folder_datasets:
                    folder_datasets[folder] = []
                folder_datasets[folder].append(dataset)

            # Take first N datasets from each folder
            limited_datasets = []
            for folder in sorted(folder_datasets.keys()):
                folder_data = folder_datasets[folder][:PER_FOLDER_LIMIT]
                limited_datasets.extend(folder_data)
                print(f"  {folder}: {len(folder_data)} datasets (out of {len(folder_datasets[folder])} available)")

            complete_datasets = limited_datasets
            print(f"\n  Total datasets after per-folder limit: {len(complete_datasets)}")
            print(f"{'='*80}")

        # Apply max datasets limit if specified (applied after per-folder limit)
        elif MAX_DATASETS is not None and len(complete_datasets) > MAX_DATASETS:
            print(f"\n{'='*80}")
            print(f"🔍 APPLYING DATASET LIMIT:")
            print(f"  Total complete datasets available: {len(complete_datasets)}")
            print(f"  Limiting to first {MAX_DATASETS} datasets")
            print(f"  (To process all, remove --max-datasets parameter)")

            # Show which datasets will be processed
            print(f"\n  Datasets to be processed:")
            for i, dataset in enumerate(complete_datasets[:MAX_DATASETS]):
                log_type, folder = dataset["setting"]
                print(f"    {i+1}. {log_type} from {folder}")

            complete_datasets = complete_datasets[:MAX_DATASETS]
            print(f"{'='*80}")

    # OUTPUT CONFIGURATION - Use config from DupliMend
    # PRIORITY: Environment variable > Config > Fallback
    # This allows PBS scripts to override config for scratch storage
    if 'BASELINE_RESULTS_DIR' in os.environ:
        RESULTS_BASE_DIR = os.environ.get('BASELINE_RESULTS_DIR')
        print(f"[CONFIG] Using results directory from environment: {RESULTS_BASE_DIR}")
    else:
        try:
            RESULTS_BASE_DIR = label_config.get("results_dir", "../results")
            print(f"[CONFIG] Using results directory from config: {RESULTS_BASE_DIR}")
        except:
            RESULTS_BASE_DIR = "../results"
            print(f"[CONFIG] Using fallback results directory: {RESULTS_BASE_DIR}")

    # Determine the log name for directory naming
    # Use the first dataset's folder name as the identifier
    if len(complete_datasets) > 0:
        log_name_identifier = complete_datasets[0]["setting"][1]  # Use folder name (e.g., "feb16-1625")
        print(f"[CONFIG] Using log name identifier: {log_name_identifier}")
    else:
        # Fallback to numbered identifier if no datasets
        log_name_identifier = str(experiment_nr_parameter)
        print(f"[CONFIG] No datasets found, using fallback identifier: {log_name_identifier}")

    # Get dataset name from environment variable (set by run_specific_folders.sh)
    dataset_name = os.environ.get('DATASET_NAME', 'default_dataset')

    # Create dataset subfolder if it doesn't exist
    dataset_results_dir = Path(RESULTS_BASE_DIR) / dataset_name
    dataset_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RESULTS] Results will be saved to: {dataset_results_dir}/")

    # Check if result files already exist using new naming convention
    # Files are named: {folder_name}_{log_name}_result_{start_data_set_size_parameter}.csv
    # Example: feb16-1625_A_1_result_-1.csv
    if complete_datasets:
        # Check for existing results for any dataset in this folder
        first_dataset = complete_datasets[0]
        log_name = first_dataset["setting"][0]  # e.g., "A_1" - index 0 is log name
        folder_name = first_dataset["setting"][1]  # e.g., "feb16-1625" - index 1 is folder name
        result_file_path = dataset_results_dir / f"{folder_name}_{log_name}_result_{start_data_set_size_parameter}.csv"

        if result_file_path.is_file():
            print(f'⚠️  Warning: Result file already exists: {result_file_path}')
            print(f'    Skipping to avoid overwriting existing results.')
            return

    # Note: CSV files with headers are now created and managed by test_epoch.py with batch writing

    # Process the batch with multiple threshold combinations
    if is_single_log_file(DATA_PATH):
        # For single log, process with different threshold combinations
        datasets_to_process = complete_datasets
    else:
        # For synthetic data - check if we should process ALL or just one batch
        # If start_data_set_size_parameter is -1, process ALL datasets (auto-loop through batches)
        # Otherwise, process the specified batch only
        if start_data_set_size_parameter == -1:
            print(f"\n📊 PROCESSING ALL DATASETS (AUTO-LOOP MODE):")
            print(f"  Total complete datasets: {len(complete_datasets)}")
            print(f"  Processing in mini-batches of {batch_size_parameter} for memory management")
            datasets_to_process = complete_datasets  # Process ALL
        else:
            # Original batching behavior for distributed processing
            batch_start = start_data_set_size_parameter * batch_size_parameter
            batch_end = min((start_data_set_size_parameter + 1) * batch_size_parameter, len(complete_datasets))
            print(f"\n📊 BATCH PROCESSING (SINGLE BATCH MODE):")
            print(f"  Total complete datasets: {len(complete_datasets)}")
            print(f"  Batch start index: {batch_start}")
            print(f"  Batch end index: {batch_end}")
            print(f"  Datasets in this batch: {batch_end - batch_start}")
            datasets_to_process = complete_datasets[batch_start:batch_end]

    # Process each dataset with error handling
    base_results = []
    successful_datasets = 0
    failed_datasets = 0
    start_time_all = time()

    for idx, d in enumerate(datasets_to_process, start=1):
        dataset_id = d.get("setting", ("unknown", "unknown"))
        print(f"\n{'='*80}")
        print(f"🔄 PROCESSING DATASET {idx}/{len(datasets_to_process)}")
        print(f"   Log: {dataset_id[0]}, Folder: {dataset_id[1]}")
        print(f"   Progress: {successful_datasets} successful, {failed_datasets} failed so far")
        elapsed_time = time() - start_time_all
        print(f"   Elapsed time: {elapsed_time/3600:.2f} hours")
        if successful_datasets > 0:
            avg_time_per_dataset = elapsed_time / successful_datasets
            remaining_datasets = len(datasets_to_process) - idx
            estimated_remaining_time = remaining_datasets * avg_time_per_dataset
            print(f"   Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")
        print(f"{'='*80}")

        try:
            dataset_start = time()
            result = inside_function(d, log_name_identifier)
            dataset_end = time()

            if result is not None:
                base_results.append(result)
                successful_datasets += 1
                print(f"✅ Dataset {idx} completed successfully in {dataset_end - dataset_start:.2f} seconds")
            else:
                failed_datasets += 1
                print(f"⚠️  Dataset {idx} returned None - skipped")

            # MEMORY MANAGEMENT: Force garbage collection and temp cleanup after each dataset
            import gc
            gc.collect()

            # Also clean up temp directories periodically
            if idx % 5 == 0:
                cleanup_temp_directories()
                print(f"🧹 Memory and temp file cleanup performed after {idx} datasets")

        except Exception as e:
            failed_datasets += 1
            print(f"❌ ERROR processing dataset {idx} ({dataset_id[0]}, {dataset_id[1]})")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()
            print(f"   Continuing to next dataset...")

            # Memory cleanup after error
            import gc
            gc.collect()
            continue

    end_time_all = time()
    total_time = end_time_all - start_time_all

    print(f"\n{'='*80}")
    print(f"📊 BATCH SUMMARY:")
    print(f"  Total datasets attempted: {len(datasets_to_process)}")
    print(f"  Successfully processed: {successful_datasets}")
    print(f"  Failed/skipped: {failed_datasets}")
    print(f"  Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if successful_datasets > 0:
        print(f"  Average time per dataset: {total_time/successful_datasets:.2f} seconds")
    print(f"={'='*80}\n")
    
    # NOTE: CSV writing is handled directly in test_epoch.py during threshold loops
    # The threshold combinations and results are written there, not here
    dataset_name = os.environ.get('DATASET_NAME', 'default_dataset')
    print(f"Processing completed. Results written directly by test_epoch.py threshold loops.")
    print(f"Check CSV files in: '{RESULTS_BASE_DIR}/{dataset_name}/'")
    print(f"                    Look for files matching pattern: *_result_{start_data_set_size_parameter}.csv")

def inside_function(paths, log_name_identifier):
    print('Inside called')
    print(f"Available paths: {list(paths.keys())}")
    print(f"xixi_log_path exists: {'xixi_log_path' in paths.keys()}")
    
    if "event_log_path" in paths.keys() and "original_log_path" in paths.keys():
        print('Processing dataset')

        # Load logs
        event_log = xes_import_factory.apply(paths["event_log_path"], parameters={
            xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})

        # Handle ground truth based on configuration
        if "ground_truth_path" in paths and paths["ground_truth_path"]:
            # Synthetic log with separate ground truth file
            print(f"[GROUND_TRUTH] Loading ground truth from separate file...")
            event_log = load_and_merge_ground_truth(
                event_log,
                paths["ground_truth_path"],
                paths.get("ground_truth_activity_column", "ground_truth_activity"),
                paths.get("event_id_column", "EventID")
            )
        elif paths.get("is_real_life_log", False):
            # Real-life log without ground truth - use concept:name as OrgLabel
            print("[GROUND_TRUTH] Real-life log - using concept:name as OrgLabel...")
            for trace in event_log:
                for event in trace:
                    if 'OrgLabel' not in event:
                        event['OrgLabel'] = event['concept:name']
            print(f"[GROUND_TRUTH] Added OrgLabel to {len(event_log)} traces")

        # Handle xixi_log - may not exist for real logs
        xixi_log = None
        if 'xixi_log_path' in paths.keys() and paths['xixi_log_path'] and os.path.exists(paths['xixi_log_path']):
            xixi_log = xes_import_factory.apply(paths["xixi_log_path"], parameters={
                xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})
        
        print(f"Original log path: {paths['original_log_path']}")
        original_event_log = xes_import_factory.apply(paths["original_log_path"], parameters={
            xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})

        # Add ground truth to original log too (same as event log)
        if "ground_truth_path" in paths and paths["ground_truth_path"]:
            print(f"[GROUND_TRUTH] Adding ground truth to original log...")
            original_event_log = load_and_merge_ground_truth(
                original_event_log,
                paths["ground_truth_path"],
                paths.get("ground_truth_activity_column", "ground_truth_activity"),
                paths.get("event_id_column", "EventID")
            )
        elif paths.get("is_real_life_log", False):
            for trace in original_event_log:
                for event in trace:
                    if 'OrgLabel' not in event:
                        event['OrgLabel'] = event['concept:name']

        # Handle model - generate from log if not provided
        if 'model_path' in paths.keys() and paths['model_path'] and os.path.exists(paths['model_path']):
            original_tree = ptml_importer.apply(paths["model_path"])
            original_net, original_initial_marking, original_final_marking = pt_converter.apply(original_tree)
        else:
            # Generate model from original log
            original_tree = inductive_miner.apply(original_event_log)
            original_net, original_initial_marking, original_final_marking = pt_converter.apply(original_tree)        
            print("Generated model from original log")

        time0 = time()
        print('log_name_identifier:', log_name_identifier)

        # Call test_epoch.run
        org_model_prec, precise_refined_log_prec, \
        imp_prec, xixi_prec, ref_log_prec, \
        ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_no_vertical_prec, \
        ref_log_all_prec, \
        ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_vertical_prec, \
        number_of_different_original_labels, \
        epoch_time, \
        time_needed_for_all_extensions, \
        time_for_greedy_mapping, time_for_semi_greedy_mapping, \
        num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi, num_of_new_labels_no_vertical, \
        mapping_quality, mapping_folding_quality, mapping_semi_quality, mapping_folding_semi_quality, \
        refined_log_fitness, refined_log_fscore, expected_entropy_clusters, expected_entropy_labels, nmi \
            = test_epoch.run(event_log, xixi_log, original_event_log, original_net, original_initial_marking,
                            original_final_marking, log_name_identifier, start_data_set_size_parameter,
                            paths["setting"][0], paths["setting"][1],
                            paths["event_log_path"],
                            use_adaptive_parameters=False,
                            is_real_life_log=paths.get("is_real_life_log", False),
                            log_size_parameter=log_size_parameter)
        
        time1 = time()
        print(f"Processing time: {time1 - time0}")

        print(f"Results for {paths['setting'][0]}, {paths['setting'][1]}:")
        print(f"  Org model precision: {org_model_prec}")
        print(f"  Imprecise log precision: {imp_prec}")
        print(f"  Refined log precision: {ref_log_prec}")
        print(f"  Xixi precision: {xixi_prec}")

        return paths["setting"][0], paths["setting"][1], \
               org_model_prec, precise_refined_log_prec, \
               imp_prec, xixi_prec, ref_log_prec, \
               ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_no_vertical_prec, \
               ref_log_all_prec, \
               ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_vertical_prec, \
               number_of_different_original_labels, \
               epoch_time, \
               time_needed_for_all_extensions, \
               time_for_greedy_mapping, time_for_semi_greedy_mapping, \
               num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi, num_of_new_labels_no_vertical, \
               mapping_quality, mapping_folding_quality, mapping_semi_quality, mapping_folding_semi_quality, \
               refined_log_fitness, refined_log_fscore, expected_entropy_clusters, expected_entropy_labels, nmi
    else:
        print("Skipping dataset - missing required paths")
        return None

if __name__ == "__main__":
    main()

# import copy as c
# import csv
# import os
# import sys
# from pathlib import Path
# from time import time

# from pm4py.objects.conversion.process_tree import converter as pt_converter
# from pm4py.objects.log.importer.xes import importer as xes_import_factory
# from pm4py.objects.process_tree.importer import importer as ptml_importer
# from pm4py.algo.discovery.inductive import algorithm as inductive_miner


# import test_epoch

# log_size_parameter = int(sys.argv[1])
# # number_of_cores = int(sys.argv[2])
# batch_size_parameter = int(sys.argv[2])  # max 610
# experiment_nr_parameter = int(sys.argv[3])  # max 610
# start_data_set_size_parameter = int(sys.argv[4])  # max 610


# # end_data_set_size_parameter = int(sys.argv[4])  # max 610
# import sys
# sys.setrecursionlimit(10000) 
# def main():
#     # directory = "xixi_files/noImprInLoop_default_OD" 23937<
#     # directory = "../../../data/noImprInLoop_default_OD"
#     # directory = "../../../data/test"
#     directory = "../data/noImprInLoop_default_OD"

#     data = {}
#     setting_ids = []
#     for folder_name in (os.listdir(directory)):
#         log_folder_list = os.listdir(os.path.join(directory, folder_name, "logs"))

#         for file_name in log_folder_list:
#             setting_id = '_'.join(file_name.split("_")[:2])
#             if setting_id not in setting_ids:
#                 setting_ids.append((setting_id, folder_name))
#                 data[(setting_id, folder_name)] = {}

#     m = 0

#     for folder_name in os.listdir(directory):
#         log_folder_list = os.listdir(os.path.join(directory, folder_name, "logs"))
#         model_folder_list = os.listdir(os.path.join(directory, folder_name, "models"))
#         for log_file_name in log_folder_list:
#             setting = ('_'.join(log_file_name.split("_")[:2]), folder_name)
#             data[setting]["setting"] = setting  # TODO change data dictionary to list
#             if "LogR" in log_file_name:
#                 data[setting]["xixi_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)
#             if "LogD" in log_file_name:
#                 # print('Test')
#                 data[setting]["event_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)
#             if "LogR" not in log_file_name and "LogD" not in log_file_name:
#                 # print('Ping')
#                 data[setting]["original_log_path"] = os.path.join(directory, folder_name, "logs", log_file_name)

#         for model_file_name in model_folder_list:
#             m = m + 1
#             # setting = (model_file_name[0] + "1", folder_name) #todo dangerous
#             setting = ('_'.join(model_file_name.split("_")[0]) + "1", folder_name)  # todo dangerous
#             if setting in data.keys():
#                 data[setting]["model_path"] = os.path.join(directory, folder_name, "models", model_file_name)

#     Path(f'{RESULTS_BASE_DIR}/exp_' + str(experiment_nr_parameter)).mkdir(parents=True, exist_ok=True)
#     Path(f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cd_' + str(experiment_nr_parameter)).mkdir(parents=True, exist_ok=True)
#     Path(f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cc_' + str(experiment_nr_parameter)).mkdir(parents=True, exist_ok=True)

#     # TODO: Add no of clusters identified
#     header = [
#         'Log', 'Folder', 'Original Labels',
#         'Original Model Precision', 'Original Log Simplicity', 'Original Log Generalization',
#         'Precise Log Precision ', 'Precise Log Simplicity', 'Precise Log Generalization',
#         'Unrefined Log Precision', 'Unrefined Log Simplicity', 'Unrefined Log Generalization',
#         'Xixi Log Precision', 'Xixi Log Simplicity', 'Xixi Log Generalization',
#         'Variant Threshold', 'Unfolding Threshold', 'Log Size', 'Refined Log Precision',
#         'Refined Log ARI', 'Refined Log Simplicity', 'Refined Log Generalization'
#     ]

#     if Path(f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cc_' + str(experiment_nr_parameter) + "/result_" + str(
#             start_data_set_size_parameter) + '.csv').is_file() or Path(
#         f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cd_' + str(experiment_nr_parameter) + "/result_" + str(
#             start_data_set_size_parameter) + '.csv').is_file():
#         print('Warning: File already exists!')
#         return

#     with open(f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cc_' + str(experiment_nr_parameter) + "/result_" + str(start_data_set_size_parameter) + '.csv', 'w', newline='') as csvfile:

#         fwriter = csv.writer(csvfile)
#         fwriter.writerow(header)

#     with open(f'{RESULTS_BASE_DIR}/exp_' + 'xixi_cd_' + str(experiment_nr_parameter) + "/result_" + str(start_data_set_size_parameter) + '.csv', 'w', newline='') as csvfile:
#         fwriter = csv.writer(csvfile)
#         fwriter.writerow(header)

#     # Start from the first cell below the headers.
#     row = 1
#     col = 0

#     # print(data)
#     print(len(data))
#     print("m ", m)
#     # data2 = c.deepcopy(data)
#     # for d in data:
#     #     if len(data[d]) != 5:  # print(d)
#     #         # print(d)
#     #         # print(data[d])
#     #         # print("ERRRRRRRRRRRRRRRRRRRROW")
#     #         # del data2[d]
#     #         pass

#     # ###########################
#     # # What does del data2[d] do?
#     # ##########################

#     # data = data2
#     # print("neue länge:", (len(data)))

#     # time1 = time()
#     # '''
#     # #from multiprocessing.pool import ThreadPool as ThreadPool

#     # from multiprocessing import Pool as ThreadPool
#     # pool = ThreadPool(number_of_cores)
#     # #results = pool.imap(inside_function, list(data.values())[:end_data_set_size_parameter])
#     # #    results = pool.map_async(inside_function, list(data.values())[:data_set_size_parameter]).get()


#     # #results = [pool.apply_async(inside_function, t) for t in used_data]
#     # #results = pool.map(inside_function, data.values())

#     # pool.close()
#     # pool.join()

#     # '''
#     # # print("position in files: ", list(data.values())[start_data_set_size_parameter:start_data_set_size_parameter+15])
#     # print(list(data.values())[start_data_set_size_parameter * batch_size_parameter:(start_data_set_size_parameter + 1) * batch_size_parameter])

#     # results = [inside_function(d) for d in list(data.values())[start_data_set_size_parameter * batch_size_parameter:(start_data_set_size_parameter + 1) * batch_size_parameter]]
#     data2 = c.deepcopy(data)
#     complete_datasets = []

#     for d in data:
#         if len(data[d]) >= 3 and "event_log_path" in data[d] and "original_log_path" in data[d]:
#             complete_datasets.append(data[d])
#         else:
#             del data2[d]

#     data = data2
#     print("Complete datasets found:", len(complete_datasets))

#     # Process the batch
#     batch_start = start_data_set_size_parameter * batch_size_parameter
#     batch_end = min((start_data_set_size_parameter + 1) * batch_size_parameter, len(complete_datasets))
#     print(f"Processing datasets {batch_start} to {batch_end}")

#     results = [inside_function(d) for d in complete_datasets[batch_start:batch_end]]

#     time2 = time()
#     print("overalltime: ", time2 - time1)

#     # print(list(results))
#     # Path(f'{RESULTS_BASE_DIR}/exp_' + str(experiment_nr_parameter)).mkdir(parents=True, exist_ok=True)
#     # with open(f'{RESULTS_BASE_DIR}/exp_' + str(experiment_nr_parameter) + "/result_" + str(start_data_set_size_parameter) + '.csv', 'w') as csvfile:
#     #     fwriter = csv.writer(csvfile)
#     #     fwriter.writerow(("log", "folder", \
#     #           "org_model_prec", "precise_refined_log_prec",\
#     #         "imp_prec", "xixi_prec", "ref_log_prec",\
#     #         "ref_log_comdec_prec", "ref_log_folding_prec", "ref_log_semi_prec", "ref_log_no_vertical_prec",\
#     #         "ref_log_all_prec",\
#     #         "ref_log_no_comdec_prec", "ref_log_no_folding_prec", "ref_log_no_semi_prec", "ref_log_vertical_prec",\
#     #         "number_of_different_original_labels", \
#     #         "epoch_time",\
#     #         "time_needed_for_all_extensions", \
#     #         "time_for_greedy_mapping", "time_for_semi_greedy_mapping", \
#     #         "num_of_new_labels", "num_of_new_labels_comdec", "num_of_new_labels_folding", "num_of_new_labels_semi", "num_of_new_labels_no_vertical", \
#     #         "mapping_quality", "mapping_folding_quality", "mapping_semi_quality", "mapping_folding_semi_quality"))
#     #     for x in results:
#     #         fwriter.writerow(x)


# def inside_function(paths):
#     print('Inside called')
#     print(f"Available paths: {list(paths.keys())}")
#     print(f"xixi_log_path exists: {'xixi_log_path' in paths.keys()}")
#     if 'xixi_log_path' in paths.keys():
#         print(f"xixi_log_path value: {paths['xixi_log_path']}")
#         print(f"File exists: {os.path.exists(paths['xixi_log_path'])}")
#     if "event_log_path" in paths.keys() and "original_log_path" in paths.keys():
#         print('in if')
#         # Change log_size to log_size_parameter in all these lines:
#         event_log = xes_import_factory.apply(paths["event_log_path"], parameters={
#             xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})

#         xixi_log = xes_import_factory.apply(paths["xixi_log_path"], parameters={
#             xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter}) if 'xixi_log_path' in paths.keys() and paths['xixi_log_path'] else None
        
#         print(paths["original_log_path"])
#         original_event_log = xes_import_factory.apply(paths["original_log_path"], parameters={
#             xes_import_factory.Variants.ITERPARSE.value.Parameters.MAX_TRACES: log_size_parameter})

#         if 'model_path' in paths.keys() and paths['model_path']:
#             original_tree = ptml_importer.apply(paths["model_path"])
#             original_net, original_initial_marking, original_final_marking = pt_converter.apply(original_tree)
#         else:
#             # Change this line:
#             original_tree = inductive_miner.apply(original_event_log)
#             original_net, original_initial_marking, original_final_marking = pt_converter.apply(original_tree)        
#             print(original_net)

#         time0 = time()
#         print('experiment_nr_parameter')
#         print(experiment_nr_parameter)

#         org_model_prec, precise_refined_log_prec, \
#         imp_prec, xixi_prec, ref_log_prec, \
#         ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_no_vertical_prec, \
#         ref_log_all_prec, \
#         ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_vertical_prec, \
#         number_of_different_original_labels, \
#         epoch_time, \
#         time_needed_for_all_extensions, \
#         time_for_greedy_mapping, time_for_semi_greedy_mapping, \
#         num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi, num_of_new_labels_no_vertical, \
#         mapping_quality, mapping_folding_quality, mapping_semi_quality, mapping_folding_semi_quality \
#             = test_epoch.run(event_log, xixi_log, original_event_log, original_net, original_initial_marking,
#                             original_final_marking, experiment_nr_parameter, start_data_set_size_parameter,
#                             paths["setting"][0], paths["setting"][1],
#                             paths["event_log_path"],  # Add this line
#                             use_adaptive_parameters=False)
#         time1 = time()
#         # print("time not adaptive: ", time1 - time0)
#         '''
#         org_model_prec, precise_refined_log_prec, \
#         imp_prec, xixi_prec, ref_log_prec, \
#         ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_postprocessing_prec, \
#         ref_log_all_prec, \
#         ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_no_postprocessing_prec, \
#         number_of_different_original_labels = test_epoch.run(event_log, xixi_log, original_event_log, original_net,
#                                                             original_initial_marking, original_final_marking, use_adaptive_parameters=True)
#         time2 = time()
#         print("time adaptive: ", time2 - time1)
#         '''
#         print(paths["setting"][0], paths["setting"][1], \
#               org_model_prec, precise_refined_log_prec, \
#               imp_prec, xixi_prec, ref_log_prec, \
#               ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_no_vertical_prec, \
#               ref_log_all_prec, \
#               ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_vertical_prec, \
#               number_of_different_original_labels, \
#               epoch_time, \
#               time_needed_for_all_extensions, \
#               time_for_greedy_mapping, time_for_semi_greedy_mapping, \
#               num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi,
#               num_of_new_labels_no_vertical, \
#               mapping_quality, mapping_folding_quality, mapping_semi_quality, mapping_folding_semi_quality
#               )

#         return paths["setting"][0], paths["setting"][1], \
#                org_model_prec, precise_refined_log_prec, \
#                imp_prec, xixi_prec, ref_log_prec, \
#                ref_log_comdec_prec, ref_log_folding_prec, ref_log_semi_prec, ref_log_no_vertical_prec, \
#                ref_log_all_prec, \
#                ref_log_no_comdec_prec, ref_log_no_folding_prec, ref_log_no_semi_prec, ref_log_vertical_prec, \
#                number_of_different_original_labels, \
#                epoch_time, \
#                time_needed_for_all_extensions, \
#                time_for_greedy_mapping, time_for_semi_greedy_mapping, \
#                num_of_new_labels, num_of_new_labels_comdec, num_of_new_labels_folding, num_of_new_labels_semi, num_of_new_labels_no_vertical, \
#                mapping_quality, mapping_folding_quality, mapping_semi_quality, mapping_folding_semi_quality


# main()
