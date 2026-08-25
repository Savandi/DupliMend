import numpy as np
import os
import json
from collections import defaultdict, deque

# === ENVIRONMENT-BASED CONFIGURATION OVERRIDE ===
def parse_hidden_dims(hidden_dims_str):
    """Parse hidden dimensions from string format like '[128,64]' to list [128, 64]"""
    try:
        # Remove spaces and brackets, split by comma
        dims_str = hidden_dims_str.strip('[]').replace(' ', '')
        return [int(dim) for dim in dims_str.split(',') if dim]
    except:
        print(f"WARNING: Could not parse hidden_dims '{hidden_dims_str}', using default [128, 64]")
        return [128, 64]

def load_env_config():
    """Load configuration parameters from environment variables"""
    env_overrides = {}
    
    # Autoencoder parameters
    if 'LATENT_DIM' in os.environ:
        env_overrides['latent_dim'] = int(os.environ['LATENT_DIM'])
    if 'HIDDEN_DIMS' in os.environ:
        env_overrides['hidden_dims'] = parse_hidden_dims(os.environ['HIDDEN_DIMS'])
    if 'LEARNING_RATE' in os.environ:
        env_overrides['learning_rate'] = float(os.environ['LEARNING_RATE'])
    if 'MAX_EPOCHS' in os.environ:
        env_overrides['max_epochs'] = int(os.environ['MAX_EPOCHS'])
    if 'NOISE_STD' in os.environ:
        env_overrides['noise_std'] = float(os.environ['NOISE_STD'])
    if 'DROPOUT_RATE' in os.environ:
        env_overrides['dropout_rate'] = float(os.environ['DROPOUT_RATE'])
    if 'BATCH_SIZE' in os.environ:
        env_overrides['batch_size'] = int(os.environ['BATCH_SIZE'])
    if 'WEIGHT_DECAY' in os.environ:
        env_overrides['weight_decay'] = float(os.environ['WEIGHT_DECAY'])
    if 'SPARSITY_LAMBDA' in os.environ:
        env_overrides['sparsity_lambda'] = float(os.environ['SPARSITY_LAMBDA'])
    if 'EARLY_STOPPING_PATIENCE' in os.environ:
        env_overrides['early_stopping_patience'] = int(os.environ['EARLY_STOPPING_PATIENCE'])
        
    # Training parameters (both modes)
    if 'WARMUP_EVENTS' in os.environ:
        env_overrides['warmup_global_events'] = int(os.environ['WARMUP_EVENTS'])
    
    # Training approach detection
    if 'TRAINING_APPROACH' in os.environ:
        env_overrides['training_approach'] = os.environ['TRAINING_APPROACH']  # 'offline' or 'online'
    
    # Offline-specific training parameters (*)
    if 'LEARNING_RATE' in os.environ:
        env_overrides['learning_rate'] = float(os.environ['LEARNING_RATE'])
    if 'MAX_EPOCHS' in os.environ:
        env_overrides['max_epochs'] = int(os.environ['MAX_EPOCHS'])
    if 'EARLY_STOPPING_PATIENCE' in os.environ:
        env_overrides['early_stopping_patience'] = int(os.environ['EARLY_STOPPING_PATIENCE'])
        
    # Online-specific training parameters (*)
    if 'INCREMENTAL_LEARNING_RATE' in os.environ:
        env_overrides['incremental_learning_rate'] = float(os.environ['INCREMENTAL_LEARNING_RATE'])
    if 'INCREMENTAL_TRAIN_EPOCHS' in os.environ:
        env_overrides['incremental_train_epochs'] = int(os.environ['INCREMENTAL_TRAIN_EPOCHS'])
    if 'INCREMENTAL_TRAINING_INTERVAL' in os.environ:
        env_overrides['incremental_training_interval'] = int(os.environ['INCREMENTAL_TRAINING_INTERVAL'])
    if 'BUFFER_RETENTION_SIZE' in os.environ:
        env_overrides['buffer_retention_size'] = int(os.environ['BUFFER_RETENTION_SIZE'])
    
    # Clustering parameters (both modes need these for inference!)
    if 'CLUSTERING_THRESHOLD' in os.environ:
        env_overrides['clustering_threshold'] = float(os.environ['CLUSTERING_THRESHOLD'])
    if 'MIN_CLUSTER_WEIGHT' in os.environ:
        env_overrides['min_cluster_weight'] = int(os.environ['MIN_CLUSTER_WEIGHT'])
    if 'VARIANCE_THRESHOLD' in os.environ:
        env_overrides['variance_threshold'] = float(os.environ['VARIANCE_THRESHOLD'])
    if 'MERGE_THRESHOLD' in os.environ:
        env_overrides['merge_threshold'] = float(os.environ['MERGE_THRESHOLD'])
    if 'FADING_FACTOR' in os.environ:
        env_overrides['fading_factor'] = float(os.environ['FADING_FACTOR'])
    if 'CLUSTERING_ALGORITHM' in os.environ:
        env_overrides['clustering_algorithm'] = os.environ['CLUSTERING_ALGORITHM']
    if 'DISTANCE_METRIC' in os.environ:
        env_overrides['distance_metric'] = os.environ['DISTANCE_METRIC']
        
    # Memory management parameters (both modes)
    if 'TEMPORAL_DECAY_RATE' in os.environ:
        env_overrides['temporal_decay_rate'] = float(os.environ['TEMPORAL_DECAY_RATE'])
    if 'FORGETTING_FACTOR' in os.environ:
        env_overrides['forgetting_factor'] = float(os.environ['FORGETTING_FACTOR'])
    if 'DECAY_AFTER_EVENTS' in os.environ:
        env_overrides['decay_after_events'] = int(os.environ['DECAY_AFTER_EVENTS'])
    if 'MEMORY_UPDATE_INTERVAL' in os.environ:
        env_overrides['memory_update_interval'] = int(os.environ['MEMORY_UPDATE_INTERVAL'])
    if 'MAX_CENTROIDS_PER_ACTIVITY' in os.environ:
        env_overrides['max_centroids_per_activity'] = int(os.environ['MAX_CENTROIDS_PER_ACTIVITY'])
    if 'MEMORY_DECAY_FACTOR' in os.environ:
        env_overrides['memory_decay_factor'] = float(os.environ['MEMORY_DECAY_FACTOR'])
        
    # Drift detection parameters (both modes)  
    if 'ADWIN_DELTA' in os.environ:
        env_overrides['adwin_delta'] = float(os.environ['ADWIN_DELTA'])
    
    # Regularization parameters (both modes)
    if 'CLUSTER_REG_WEIGHT' in os.environ:
        env_overrides['cluster_reg_weight'] = float(os.environ['CLUSTER_REG_WEIGHT'])
    if 'MEMORY_REGULARIZATION_WEIGHT' in os.environ:
        env_overrides['memory_regularization_weight'] = float(os.environ['MEMORY_REGULARIZATION_WEIGHT'])
        
    # Control flow parameters
    if 'USE_CONTROL_FLOW' in os.environ:
        env_overrides['use_control_flow_features'] = os.environ['USE_CONTROL_FLOW'].lower() == 'true'
    if 'CONTROL_FLOW_CONTEXT_WINDOW' in os.environ:
        env_overrides['control_flow_context_window'] = int(os.environ['CONTROL_FLOW_CONTEXT_WINDOW'])
    if 'EMBEDDING_DIM' in os.environ:
        env_overrides['embedding_dim'] = int(os.environ['EMBEDDING_DIM'])
    if 'MAX_CONTROL_FLOW_PATTERNS' in os.environ:
        env_overrides['max_control_flow_patterns'] = int(os.environ['MAX_CONTROL_FLOW_PATTERNS'])
    if 'CONTROL_FLOW_FEATURE_BOOST' in os.environ:
        env_overrides['control_flow_feature_boost'] = int(os.environ['CONTROL_FLOW_FEATURE_BOOST'])

    # Evaluation/Ground truth configuration
    if 'GROUND_TRUTH_PATH' in os.environ:
        env_overrides['ground_truth_path'] = os.environ['GROUND_TRUTH_PATH']
    if 'GROUND_TRUTH_ACTIVITY_COLUMN' in os.environ:
        env_overrides['ground_truth_activity_column'] = os.environ['GROUND_TRUTH_ACTIVITY_COLUMN']
    if 'EVENT_ID_COLUMN' in os.environ:
        env_overrides['event_id_column'] = os.environ['EVENT_ID_COLUMN']
    if 'CONTROL_FLOW_COLUMN' in os.environ:
        env_overrides['control_flow_column'] = os.environ['CONTROL_FLOW_COLUMN']
    if 'CASE_ID_COLUMN' in os.environ:
        env_overrides['case_id_column'] = os.environ['CASE_ID_COLUMN']
    if 'DEFAULT_ACTIVITY' in os.environ:
        env_overrides['default_activity'] = os.environ['DEFAULT_ACTIVITY']

    # Output configuration
    if 'EXPERIMENT_OUTPUT_DIR' in os.environ:
        env_overrides['experiment_output_dir'] = os.environ['EXPERIMENT_OUTPUT_DIR']
    if 'EXPERIMENT_CONFIG_NAME' in os.environ:
        env_overrides['experiment_config_name'] = os.environ['EXPERIMENT_CONFIG_NAME']

    return env_overrides

# Load environment overrides
ENV_OVERRIDES = load_env_config()

# === PORTABLE PATH ROOTS ===
# All dataset and results locations are resolved relative to these roots so the
# repository runs on any machine. Override them with environment variables to point
# at a shared drive or scratch space; the defaults are repo-relative.
#
#   DUPLIMEND_DATA_DIR      raw/processed input logs        (default ./data)
#   DUPLIMEND_OUTPUT_DIR    run artefacts and tracking dirs (default ./run_output)
#   DUPLIMEND_RESULTS_DIR   evaluation results              (default ./evaluation_results)
#   DUPLIMEND_BASELINE_DATA folder-based BPM2016 synthetic logs
#                                                           (default $DUPLIMEND_DATA_DIR/noImprInLoop_default_OD)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get('DUPLIMEND_DATA_DIR', os.path.join(REPO_ROOT, 'data'))
OUTPUT_DIR = os.environ.get('DUPLIMEND_OUTPUT_DIR', 'run_output')
RESULTS_DIR = os.environ.get('DUPLIMEND_RESULTS_DIR', 'evaluation_results')
BASELINE_DATA_DIR = os.environ.get('DUPLIMEND_BASELINE_DATA',
                                   os.path.join(DATA_DIR, 'noImprInLoop_default_OD'))
SYNTHETIC_LOGS_DIR = os.path.join(REPO_ROOT, 'src', 'synthetic_logs')

# === TRAINING APPROACH DETECTION ===
def detect_training_approach():
    """Detect whether we're using offline or online training approach"""
    # Check explicit environment variable first
    if 'training_approach' in ENV_OVERRIDES:
        return ENV_OVERRIDES['training_approach']
    
    # Auto-detect based on environment
    if 'PBS_JOBID' in os.environ:
        # Running on HPC - likely offline batch training
        return 'offline'
    else:
        # Default to online for local runs
        return 'online'

TRAINING_APPROACH = detect_training_approach()

# Update training mode config based on approach
training_mode_string = "offline_mode" if TRAINING_APPROACH == 'offline' else "online_mode"

# Log configuration being used
print("=== DUPLIMEND CONFIGURATION ===")
print(f"Training Approach: {TRAINING_APPROACH}")
print(f"Training Mode: {training_mode_string}")

if ENV_OVERRIDES:
    print("=== ENVIRONMENT CONFIGURATION OVERRIDES ===")
    for key, value in ENV_OVERRIDES.items():
        print(f"{key}: {value}")
    print("=" * 47)

# --- Perspective Column Mapping ---
# control_flow_column = 'activity_label'
# timestamp_column = 'SYSCALL_timestamp'
# resource_column = None
# case_id_column = 'SYSCALL_pid'
# event_id_column = 'EventID'

# === Stream-processing column configuration ===
# These name the columns main.py reads while consuming the stream. They default
# to I-PALIA and are environment-driven, so a different log can be processed
# without editing this file.
#
#   I-PALIA          CONTROL_FLOW_COLUMN=concept:name   CASE_ID_COLUMN=case:concept:name
#                    TIMESTAMP_COLUMN=time:timestamp    RESOURCE_COLUMN=(unset)
#   Document Review  CONTROL_FLOW_COLUMN=Activity       CASE_ID_COLUMN=CaseID
#                    TIMESTAMP_COLUMN=Timestamp         RESOURCE_COLUMN=Resource
#   CybersecIoT      CONTROL_FLOW_COLUMN=activity_label CASE_ID_COLUMN=SYSCALL_pid
#                    TIMESTAMP_COLUMN=SYSCALL_timestamp RESOURCE_COLUMN=(unset)
control_flow_column = os.environ.get('CONTROL_FLOW_COLUMN', 'concept:name')
timestamp_column = os.environ.get('TIMESTAMP_COLUMN', 'time:timestamp')
resource_column = os.environ.get('RESOURCE_COLUMN') or None
case_id_column = os.environ.get('CASE_ID_COLUMN', 'case:concept:name')
event_id_column = os.environ.get('EVENT_ID_COLUMN', 'EventID')

excluded_columns = {
        case_id_column, event_id_column, timestamp_column, control_flow_column, 'PROCESS_comm', 'CUSTOM_libs', 'OrgLabel', 'org:resource', 'lifecycle:transition', 'Timestamp', 'Lifecycle'
        }

# case_id_column, event_id_column, timestamp_column, control_flow_column, 'MineName', 'MineAddress', 'MineDecLat', 'MineDecLon', 'MineAddressTown', 'MineAddressState', 'Region', 'MineCluster', 'MineAddressPostalCode'

training_mode_config = {
    "mode": training_mode_string,  # Uses detected approach: offline_mode or online_mode
    "checkpoint_interval": 5000,

    # I-PALIA has ~8400 events, use 4000 for warmup (about 50%)
    "warmup_global_events": ENV_OVERRIDES.get('warmup_global_events', 4000),
    "incremental_training_interval": ENV_OVERRIDES.get('incremental_training_interval', 1000),

    "min_events_for_incremental_training": 1,
    "min_events_for_dynamic_training": 10,

    # FOLDER-BASED CONFIGURATION (for large-scale experiments)
    "training_folder": os.environ.get('DUPLIMEND_TRAIN_FOLDER',
                                      os.path.join(DATA_DIR, 'processed_train_data')),
    "training_file_pattern": "*.csv",
    "test_folder": os.environ.get('DUPLIMEND_TEST_FOLDER',
                                  os.path.join(DATA_DIR, 'processed_test_data')),
    "test_file_pattern": "*.csv",

    # === DEFAULT: I-PALIA dataset for Quick Start ===
    "default_input_file": "src/synthetic_logs/ipalia.csv",

    # Model persistence - uses local output directory by default
    "save_models_after_training": True,
    "models_save_dir": ENV_OVERRIDES.get('experiment_output_dir', "run_output"),
    "load_pretrained_models": False,

    "feature_vectors_base_dir": ENV_OVERRIDES.get('experiment_output_dir', "run_output"),
    "models_base_dir": ENV_OVERRIDES.get('experiment_output_dir', "run_output"),
}

evaluation_config = {
    # Main output directory for DupliMend runs (uses local path by default)
    "results_base_dir": os.environ.get('DUPLIMEND_OUTPUT_DIR', 'run_output'),

    "multi_test_evaluation_config": {
        "default_tracking_base_dir": os.environ.get('DUPLIMEND_TRACKING_BASE_DIR', OUTPUT_DIR),
        "default_ground_truth_dir": os.environ.get('DUPLIMEND_GROUND_TRUTH_DIR',
                                                   os.path.join(DATA_DIR, 'processed_groundtruth_test_data')),
        "default_output_dir": os.environ.get('DUPLIMEND_MULTI_EVAL_OUTPUT_DIR', RESULTS_DIR),
        "default_activity": "openat_www-data",
        "event_id_column": "EventID",
        "control_flow_column": "activity_label",
        "control_flow_column_ground_truth": "ground_truth_activity_label",
        "case_id_column": "SYSCALL_pid"
    },

    "single_evaluation_config": {
        # Output directory for evaluation results (local by default)
        "default_output_dir": "evaluation_results",
        "default_event_vectors_path": None,  # Auto-detected from tracking dir
        "default_centroids_path": None,      # Auto-detected from tracking dir

        # Process Mining Configuration
        "inductive_miner_noise_threshold": 0.0,

        # === I-PALIA Dataset Configuration (Default for Quick Start) ===
        # Ground truth: ipalia_groundtruth.csv contains ground_truth_activity column
        # Activity of interest: "A" (the homonymous label to be refined into A_start, A_middle, A_end)
        "default_ground_truth_path": ENV_OVERRIDES.get('ground_truth_path', "src/synthetic_logs/ipalia_groundtruth.csv"),
        "default_activity": ENV_OVERRIDES.get('default_activity', "A"),
        "event_id_column": ENV_OVERRIDES.get('event_id_column', "EventID"),
        "control_flow_column": ENV_OVERRIDES.get('control_flow_column', "concept:name"),
        "control_flow_column_ground_truth": ENV_OVERRIDES.get('ground_truth_activity_column', "ground_truth_activity"),
        "case_id_column": ENV_OVERRIDES.get('case_id_column', "case:concept:name")

        # === Alternative: Document Review Process ===
        # "default_ground_truth_path": "src/synthetic_logs/document_review_process_groundtruth.csv",
        # "default_activity": "Submit application",
        # "control_flow_column": "Activity",
        # "control_flow_column_ground_truth": "ground_truth_activity",
        # "case_id_column": "CaseID"
    },
    
    "baseline_evaluation_config": {
        # Base directory for all baseline results
        "baseline_results_base_dir": os.environ.get('DUPLIMEND_BASELINE_RESULTS_DIR',
                                                    os.path.join(RESULTS_DIR, 'baselines')),

        # Label Refinement Baseline Configuration
        "label_refinement": {
            # Folder-based BPM2016 synthetic logs. Set DUPLIMEND_BASELINE_DATA to the
            # extracted "noImprInLoop_default_OD" directory.
            "data_path_synthetic": BASELINE_DATA_DIR,

            # Single file (CSV/XES) - WITH ground truth (EDIT THESE TO SWITCH DATASETS)
            # Example: document_review_process.csv or ipalia.csv
            "data_path_real": os.path.join(SYNTHETIC_LOGS_DIR, 'document_review_process.csv'),
            "csv_config_name": "document_review_config",  # For CSV: "ipalia_config" or "document_review_config" | For XES: set to None

            # Ground truth configuration (for synthetic logs with ground truth)
            "has_ground_truth": True,
            "ground_truth_path": os.path.join(SYNTHETIC_LOGS_DIR,
                                              'document_review_process_groundtruth.csv'),
            "ground_truth_activity_column": "ground_truth_activity",
            "event_id_column": "EventID",

            # Real-world log without ground truth (XES files) - UNCOMMENT AND CONFIGURE TO USE
            # "data_path_real": os.path.join(DATA_DIR, 'BPI_Challenge_2013_closed_problems.xes'),
            # "csv_config_name": None,  # Not needed for XES
            # "has_ground_truth": False,  # No ground truth = only process mining metrics

            # Output directories
            "output_dir": os.path.join(RESULTS_DIR, 'baselines', 'label_refinement', 'outputs'),
            "results_dir": os.path.join(RESULTS_DIR, 'baselines', 'label_refinement', 'results'),
            "best_results_dir": os.path.join(RESULTS_DIR, 'baselines', 'label_refinement', 'best_results')
        },

        # PM Label Splitting Baseline Configuration
        "pm_label_splitting": {
            # Data paths
            "data_path_synthetic": BASELINE_DATA_DIR,

            # Single file (CSV/XES) - WITH ground truth (EDIT THESE TO SWITCH DATASETS)
            # Example: document_review_process.csv or ipalia.csv
            "data_path_real": os.path.join(SYNTHETIC_LOGS_DIR, 'document_review_process.csv'),
            "csv_config_name": "document_review_config",  # For CSV: "ipalia_config" or "document_review_config" | For XES: set to None

            # Ground truth configuration (for synthetic logs with ground truth)
            "has_ground_truth": True,
            "ground_truth_path": os.path.join(SYNTHETIC_LOGS_DIR,
                                              'document_review_process_groundtruth.csv'),
            "ground_truth_activity_column": "ground_truth_activity",
            "event_id_column": "EventID",

            # Real-world log without ground truth (XES files) - UNCOMMENT AND CONFIGURE TO USE
            # "data_path_real": os.path.join(DATA_DIR, 'BPI_Challenge_2013_closed_problems.xes'),
            # "csv_config_name": None,  # Not needed for XES
            # "has_ground_truth": False,  # No ground truth = only process mining metrics

            # Output directories
            "output_dir": os.path.join(RESULTS_DIR, 'baselines', 'pm_label_splitting', 'outputs'),
            "results_dir": os.path.join(RESULTS_DIR, 'baselines', 'pm_label_splitting', 'results'),
            "best_results_dir": os.path.join(RESULTS_DIR, 'baselines', 'pm_label_splitting', 'best_results')
        }
    }
}

# --- Forgetting & Decay Control ---
forgetting_params = {
    "decay_after_events": ENV_OVERRIDES.get('decay_after_events', 10000),
    "removal_threshold_events": 100000,
    "frequency_decay_threshold": 2,
    "temporal_decay_rate": ENV_OVERRIDES.get('temporal_decay_rate', 0.04),
    "forgetting_factor": ENV_OVERRIDES.get('forgetting_factor', 0.85),
    "forgetting_threshold": 0.0004
    }

#Control Flow Parameters
use_control_flow_features = ENV_OVERRIDES.get('use_control_flow_features', True)
control_flow_context_window = ENV_OVERRIDES.get('control_flow_context_window', 7)  #7 and 2
control_flow_pattern_window_size =  2#2
max_control_flow_patterns = ENV_OVERRIDES.get('max_control_flow_patterns', 5)   
embedding_dim = ENV_OVERRIDES.get('embedding_dim', 8) #8
parallelization_threshold = 18
enable_self_loop_handling = True
unfolding_threshold = 0.2
max_categories = 2000


# Autoencoder parameters
autoencoder_params = {
    # Existing parameters with environment overrides
    "latent_dim": ENV_OVERRIDES.get('latent_dim', 32),
    "hidden_dims": ENV_OVERRIDES.get('hidden_dims', [128, 64]),
    "sparsity_lambda": ENV_OVERRIDES.get('sparsity_lambda', 0.001),
    "noise_std": ENV_OVERRIDES.get('noise_std', 0.1),
    
    # Enhanced training parameters with environment overrides
    "dropout_rate": ENV_OVERRIDES.get('dropout_rate', 0.1),
    "learning_rate": ENV_OVERRIDES.get('learning_rate', 0.001),
    "max_epochs": ENV_OVERRIDES.get('max_epochs', 150),  # Increased for early stopping
    "batch_size": ENV_OVERRIDES.get('batch_size', 32),
    "min_events_for_training": 1,
    
    # Early stopping
    "early_stopping_patience": ENV_OVERRIDES.get('early_stopping_patience', 15),
    "min_delta": 1e-6,
    
    # Enhanced Adam optimizer
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": ENV_OVERRIDES.get('weight_decay', 1e-5),
    "amsgrad": False,
    
    # Learning rate scheduling
    "lr_decay_factor": 0.7,
    "lr_patience": 5,
    "min_lr": 1e-7,
    
    # Training stability
    "grad_clip": 1.0,
    
    # TensorBoard settings
    "enable_tensorboard": True,
    
    # FULL BATCH MODE SETTINGS
    "use_full_batch_mode": True,  
    "max_batch_size_gb": 4.0,    

    # Incremental training (online mode)
    "incremental_learning_rate": ENV_OVERRIDES.get('incremental_learning_rate', 0.0005),
    "incremental_train_epochs": ENV_OVERRIDES.get('incremental_train_epochs', 2),
    "buffer_retention_size": ENV_OVERRIDES.get('buffer_retention_size', 100),

    "cluster_regularization": {
        "cluster_reg_weight": ENV_OVERRIDES.get('cluster_reg_weight', 0.1),
        "separation_weight": 1.0,
        "compactness_weight": 0.5,
        "consistency_weight": 0.3,
        "min_separation_distance": 0.1,
        "max_intra_cluster_variance": 0.05
    },
    "memory_regularization": {
        "enable_memory_regularization": True,
        "memory_regularization_weight": ENV_OVERRIDES.get('memory_regularization_weight', 0.1),
        "max_centroids_per_activity": 50,
        "centroid_stability_threshold": 10,
        "memory_update_interval": 3000,
        "memory_decay_factor": 0.95,
        "exemplar_count": 2,
        "similarity_threshold": 0.7,
    }
}

# --- Feature Selection (Enhancement) Parameters ---
feature_selection_params = {
    "control_flow_feature_boost": ENV_OVERRIDES.get('control_flow_feature_boost', 200),  # Boosting for control flow features 200
}

categorical_feature_params = {
        "categorical_columns": [col for col in [
        "SYSCALL_exit_hint",
        # # # "PROCESS_comm",
        "CUSTOM_openFiles",
        # "CUSTOM_libs",
        resource_column            
    ] if col is not None and col not in excluded_columns]
}

# --- Clustering Algorithm Configuration ---
clustering_params = {
    "algorithm": ENV_OVERRIDES.get('clustering_algorithm', "river_dbstream"),  # Options: "river_dbstream", "river_streamkmeans", "river_denstream", "river_clustream"
    "distance_metric": ENV_OVERRIDES.get('distance_metric', "cosine"),  # Options: "euclidean", "cosine", "manhattan"
    # DBStream parameters (both modes need these for inference!)
    "clustering_threshold": ENV_OVERRIDES.get('clustering_threshold', 0.01), #0.9 and 0.15 0.95
    "fading_factor": ENV_OVERRIDES.get('fading_factor', 0.001), #0.001

    # DenStream parameters
    "decaying_factor": 0.01,
    "beta": 0.5,
    "mu": 2.5,
    "epsilon": 0.5,
    "n_samples_init": 10,

    # CluStream parameters
    "max_micro_clusters": 200,
    "time_gap": 1000,
    "halflife": 0.1,
    "n_macro_clusters": 20,
    "split_perturbation": 0.05,

    # STREAMKMeans parameters
    "n_clusters": 5,
    "sigma": 1.0,

    # General parameters (both modes)
    "min_cluster_weight": ENV_OVERRIDES.get('min_cluster_weight', 20), #20 and 60
    "variance_threshold": ENV_OVERRIDES.get('variance_threshold', 1e-07), #0.0001 and 5e-07 and  0.000001 and  0.000011
    "merge_threshold": ENV_OVERRIDES.get('merge_threshold', 0.01), #0.5 and 0.7
    "decay_interval": 10000,

    "drift_detection": {
        # ADWIN parameters (adaptive window - no fixed sizes needed!)
        "adwin_delta": ENV_OVERRIDES.get('adwin_delta', 0.002),  # Sensitivity (smaller = more sensitive to drift)
    },
    "memory_management": {
        "update_interval": ENV_OVERRIDES.get('memory_update_interval', 100),  # Update memory every N events
        "max_centroids_per_activity": ENV_OVERRIDES.get('max_centroids_per_activity', 50),
        "centroid_stability_threshold": 10,
        "memory_decay_factor": ENV_OVERRIDES.get('memory_decay_factor', 0.95),
        "recent_embeddings_maxlen": 1000,
        "recent_embeddings_for_norm": 100,
        "min_recent_embeddings_for_memory": 10,
        "recent_embeddings_for_memory": 50,
        "min_embeddings_per_cluster_for_memory": 3,
    }
}
#
# clustering_params = {
#     "algorithm": "river_dbstream",  # Options: "river_dbstream", "river_streamkmeans", "river_denstream", "river_clustream"
#     "distance_metric": "cosine",  # Options: "euclidean", "cosine", "manhattan"
#     # DBStream parameters
#     "clustering_threshold":  0.9, #0.9 and 0.15 0.95
#     "fading_factor": 0.001, #0.001

#     # DenStream parameters
#     "decaying_factor": 0.01,
#     "beta": 0.5,
#     "mu": 2.5,
#     "epsilon": 0.5,
#     "n_samples_init": 10,

#     # CluStream parameters
#     "max_micro_clusters": 200,
#     "time_gap": 1000,
#     "halflife": 0.1,
#     "n_macro_clusters": 20,
#     "split_perturbation": 0.05,

#     # STREAMKMeans parameters
#     "n_clusters": 5,
#     "sigma": 1.0,

#     # General parameters
#     "min_cluster_weight":20, #20 and 60
#     "variance_threshold": 1.1e-05, #0.0001 and 5e-07 and  0.000001 and  0.000011
#     "merge_threshold": 0.2, #0.5 and 0.7
#     "decay_interval": 10000,

#     "drift_detection": {
#         "variance_drift_threshold": 0.1,
#         "inter_cluster_threshold": 0.05,
#         "embedding_norm_threshold": 0.15,
#         "drift_window_size": 1000,
#         "min_observations_for_drift": 100,
#         "check_interval": 5000
#     },
#     "memory_management": {
#         "update_interval": 2000,  # Update memory every N events
#         "max_centroids_per_activity": 25,
#         "centroid_stability_threshold": 20,
#         "memory_decay_factor": 0.95,
#         "recent_embeddings_maxlen": 1000,
#         "recent_embeddings_for_norm": 100,
#         "min_recent_embeddings_for_memory": 10,
#         "recent_embeddings_for_memory": 50,
#         "min_embeddings_per_cluster_for_memory": 3,
#     }
# }

# --- Misc ---
time_feature_columns = ["hour_bin", "day_of_week", "is_weekend", "week_of_month", "season", "month"]

# === CONFIGURATION VALIDATION AND LOGGING ===
def validate_config():
    """Validate configuration parameters"""
    issues = []
    
    # Validate autoencoder parameters
    if autoencoder_params['latent_dim'] <= 0:
        issues.append("latent_dim must be positive")
    if autoencoder_params['learning_rate'] <= 0:
        issues.append("learning_rate must be positive")
    if autoencoder_params['max_epochs'] <= 0:
        issues.append("max_epochs must be positive")
    if not (0 <= autoencoder_params['noise_std'] <= 1):
        issues.append("noise_std should be between 0 and 1")
    if not (0 <= autoencoder_params['dropout_rate'] <= 1):
        issues.append("dropout_rate should be between 0 and 1")
    
    # Validate control flow parameters
    if control_flow_context_window <= 0:
        issues.append("control_flow_context_window must be positive")
    if embedding_dim <= 0:
        issues.append("embedding_dim must be positive")
    
    if issues:
        print("CONFIG VALIDATION ERRORS:")
        for issue in issues:
            print(f"  - {issue}")
        raise ValueError("Configuration validation failed")
    else:
        print("[OK] Configuration validation passed")

def log_final_config():
    """Log the final configuration being used"""
    config_name = ENV_OVERRIDES.get('experiment_config_name', 'default')
    print(f"\n=== FINAL CONFIGURATION: {config_name.upper()} ===")
    
    print("AUTOENCODER:")
    for key, value in autoencoder_params.items():
        if isinstance(value, dict):
            continue  # Skip nested dicts for brevity
        print(f"  {key}: {value}")
    
    print("CONTROL FLOW:")
    print(f"  use_control_flow_features: {use_control_flow_features}")
    print(f"  control_flow_context_window: {control_flow_context_window}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  max_control_flow_patterns: {max_control_flow_patterns}")
    print(f"  control_flow_feature_boost: {feature_selection_params['control_flow_feature_boost']}")
    
    print("TRAINING:")
    print(f"  warmup_global_events: {training_mode_config['warmup_global_events']}")
    print(f"  mode: {training_mode_config['mode']}")
    
    if 'experiment_output_dir' in ENV_OVERRIDES:
        print(f"OUTPUT DIR: {ENV_OVERRIDES['experiment_output_dir']}")
    
    print("=" * 50)

# Run validation and logging
validate_config()
log_final_config()

# Save configuration to file if output directory is specified
if 'experiment_output_dir' in ENV_OVERRIDES:
    import os
    import json
    from datetime import datetime
    
    config_file = os.path.join(ENV_OVERRIDES['experiment_output_dir'], 'experiment_config.json')
    os.makedirs(ENV_OVERRIDES['experiment_output_dir'], exist_ok=True)
    
    config_data = {
        'experiment_name': ENV_OVERRIDES.get('experiment_config_name', 'default'),
        'timestamp': datetime.now().isoformat(),
        'autoencoder_params': {k: v for k, v in autoencoder_params.items() if not isinstance(v, dict)},
        'control_flow_params': {
            'use_control_flow_features': use_control_flow_features,
            'control_flow_context_window': control_flow_context_window,
            'embedding_dim': embedding_dim,
            'max_control_flow_patterns': max_control_flow_patterns,
            'control_flow_feature_boost': feature_selection_params['control_flow_feature_boost']
        },
        'training_params': {
            'warmup_global_events': training_mode_config['warmup_global_events'],
            'mode': training_mode_config['mode']
        },
        'environment_overrides': ENV_OVERRIDES
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"[OK] Configuration saved to: {config_file}")