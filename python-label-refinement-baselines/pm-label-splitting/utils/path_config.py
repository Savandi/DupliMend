"""
Centralized path configuration for PM-Label-Splitting
This module is imported by all other modules to avoid circular imports
"""
import os
import sys

# PRIORITY 1: Check for environment variables (set by shell script for WSL)
if 'PM_LABEL_OUTPUTS' in os.environ:
    OUTPUT_BASE_DIR = os.environ['PM_LABEL_OUTPUTS']
    RESULTS_BASE_DIR = os.environ.get('PM_LABEL_RESULTS', './results')
    BEST_RESULTS_DIR = os.environ.get('PM_LABEL_BEST_RESULTS', './best_results')
    print(f"✅ Using paths from environment variables (WSL mode)")
    print(f"   Outputs: {OUTPUT_BASE_DIR}")
    print(f"   Results: {RESULTS_BASE_DIR}")
else:
    # PRIORITY 2: Load from config file
    try:
        # Suppress config output
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        config_dir = os.path.abspath('../../config')
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)

        from config import evaluation_config

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        pm_config = evaluation_config.get("baseline_evaluation_config", {}).get("pm_label_splitting", {})

        OUTPUT_BASE_DIR = pm_config.get("output_dir", "./outputs")
        RESULTS_BASE_DIR = pm_config.get("results_dir", "./results")
        BEST_RESULTS_DIR = pm_config.get("best_results_dir", "./best_results")

        print(f"✅ Using DupliMend PM Label Splitting config")

    except (ImportError, Exception) as e:
        # PRIORITY 3: Fallback to relative paths
        print(f"⚠️  Could not import DupliMend config, using fallback paths")
        OUTPUT_BASE_DIR = "./outputs"
        RESULTS_BASE_DIR = "./results"
        BEST_RESULTS_DIR = "./best_results"

# Create dataset subfolder if DATASET_NAME is set
if 'DATASET_NAME' in os.environ:
    dataset_name = os.environ['DATASET_NAME']
    OUTPUT_BASE_DIR = os.path.join(OUTPUT_BASE_DIR, dataset_name)
    RESULTS_BASE_DIR = os.path.join(RESULTS_BASE_DIR, dataset_name)
    BEST_RESULTS_DIR = os.path.join(BEST_RESULTS_DIR, dataset_name)

    # Create directories
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    os.makedirs(BEST_RESULTS_DIR, exist_ok=True)

    print(f"✅ Using dataset subfolder: {dataset_name}")
