import os
import random
import csv
import argparse
import psutil
import torch
import pickle
import subprocess
import json
import sys
import glob
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from src.duplimend_framework.utils.global_state import (
    event_embedding_mapping,
    directly_follows_graph,
    per_case_directly_follows_graph,
    case_activity_sequences
)
from config.config import *
from src.duplimend_framework.utils.control_flow_feature_utils import (
    forget_old_cases, update_case_sequence, extract_control_flow_features
)
from src.duplimend_framework.utils.logging_utils import log_traceability
from src.duplimend_framework.label_refinement import LabelRefiner
from src.duplimend_framework.utils.online_activity_embeddings import DynamicActivityEmbeddings
from src.duplimend_framework.utils.online_onehot_encoder import OnlineOneHotEncoder
from src.duplimend_framework.streaming_sparse_denoising_autoencoder import SparseDenoisingAutoencoder
from src.duplimend_framework.cluster_manager import ClusterManager
from src.duplimend_framework.utils.cluster_evolution_tracker import ClusterEvolutionTracker
from src.duplimend_framework.drift_retraining.cluster_aware_autoencoder import ClusterAwareAutoencoder

from src.duplimend_framework.feature_vector_builder import FeatureVectorBuilder
from src.duplimend_framework.utils.global_state import event_embedding_mapping
from sklearn.preprocessing import RobustScaler
from src.duplimend_framework.utils.config_saver import save_config_to_output
from src.evaluation.evaluate_single_test_file import run_evaluation

_tensorboard_process = None


def load_components_from_feature_vectors(feature_vectors_base_dir, folder_name):
    """Load activity embeddings, encoders, and global state from feature vectors directory."""
    feature_vectors_dir = os.path.join(feature_vectors_base_dir, f"feature_vectors_{folder_name}")

    components = {}

    embeddings_path = os.path.join(feature_vectors_dir, "activity_embeddings.pkl")
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            components['activity_embeddings'] = pickle.load(f)
    else:
        components['activity_embeddings'] = None

    encoders_path = os.path.join(feature_vectors_dir, "onehot_encoders.pkl")
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            components['onehot_encoders'] = pickle.load(f)
    else:
        components['onehot_encoders'] = {}

    global_state_path = os.path.join(feature_vectors_dir, "global_state.pkl")
    if os.path.exists(global_state_path):
        with open(global_state_path, 'rb') as f:
            components['global_state'] = pickle.load(f)
    else:
        components['global_state'] = None

    return components


def launch_tensorboard_clean_integration(log_dir, port=6007):
    """Launch TensorBoard using the clean launcher script."""
    import subprocess
    import sys
    import os

    script_path = os.path.join(os.path.dirname(__file__), "src", "launch_tensorboard.py")

    if not os.path.exists(script_path):
        return None

    try:
        process = subprocess.Popen([
            sys.executable, script_path, log_dir, str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception:
        return None


def save_checkpoint_state(checkpoint_event_count, total_processed, cluster_manager, cluster_tracker,
                          event_vectors_buffer, output_dir, original_file_path, file_name_prefix=""):
    """Save clustering state at a specific checkpoint during stream processing."""
    import json
    import os

    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{checkpoint_event_count}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    event_vectors_path = os.path.join(checkpoint_dir,
                                      f"event_feature_vectors_checkpoint_{checkpoint_event_count}.jsonl")
    with open(event_vectors_path, "w") as f:
        for event_vector_data in event_vectors_buffer:
            f.write(json.dumps(event_vector_data) + "\n")

    centroids_path = os.path.join(checkpoint_dir, f"centroids_checkpoint_{checkpoint_event_count}.json")
    checkpoint_name = f"{file_name_prefix}_checkpoint_{checkpoint_event_count}" if file_name_prefix else f"checkpoint_{checkpoint_event_count}"
    cluster_tracker.save_final_centroids(cluster_manager, centroids_path, checkpoint_name)

    refined_log_path = os.path.join(checkpoint_dir,
                                    f"refined_log_with_clusters_checkpoint_{checkpoint_event_count}.csv")

    create_refined_log_with_cluster_mapping(
        original_file_path=original_file_path,
        event_vectors_path=event_vectors_path,
        centroids_path=centroids_path,
        output_refined_log_path=refined_log_path
    )

    return checkpoint_dir


def create_refined_log_with_cluster_mapping(original_file_path, event_vectors_path, centroids_path,
                                            output_refined_log_path, activity_of_interest=None):
    """Create a refined log by mapping events to their nearest cluster centroids using cosine distance."""
    original_df = pd.read_csv(original_file_path)

    event_vectors = []
    if os.path.exists(event_vectors_path):
        with open(event_vectors_path, "r") as f:
            for line in f:
                if line.strip():
                    event_vectors.append(json.loads(line))

    with open(centroids_path, "r") as f:
        centroids_data = json.load(f)

    if "centroids_by_activity" in centroids_data:
        centroids_by_activity = centroids_data["centroids_by_activity"]
    else:
        centroids_by_activity = centroids_data

    event_id_to_vector = {}
    for ev in event_vectors:
        event_id = str(ev.get(event_id_column, ev.get("event_id", "")))
        if event_id:
            event_id_to_vector[event_id] = {
                "vector": np.array(ev["feature_vector"]),
                "activity": ev["activity"]
            }

    original_df[event_id_column] = original_df[event_id_column].astype(str)
    refined_activities = []

    for _, row in original_df.iterrows():
        event_id = str(row[event_id_column])
        activity_label = row[control_flow_column]

        if activity_of_interest and activity_label != activity_of_interest:
            refined_activities.append(activity_label)
            continue

        if (event_id in event_id_to_vector and
                activity_label in centroids_by_activity and
                centroids_by_activity[activity_label]):

            event_vector = event_id_to_vector[event_id]["vector"]
            activity_centroids = centroids_by_activity[activity_label]

            centroid_matrix = []
            centroid_ids = []

            for centroid_data in activity_centroids:
                if isinstance(centroid_data, dict) and 'centroid' in centroid_data:
                    centroid_matrix.append(centroid_data['centroid'])
                    centroid_ids.append(centroid_data.get('cluster_id', len(centroid_ids)))
                elif isinstance(centroid_data, list):
                    centroid_matrix.append(centroid_data)
                    centroid_ids.append(len(centroid_ids))

            if centroid_matrix:
                centroid_matrix = np.array(centroid_matrix)
                distances = cosine_distances([event_vector], centroid_matrix)[0]
                nearest_idx = np.argmin(distances)
                nearest_centroid_id = centroid_ids[nearest_idx]
                refined_label = f"{activity_label}_{nearest_centroid_id}"
                refined_activities.append(refined_label)
            else:
                refined_activities.append(activity_label)
        else:
            refined_activities.append(activity_label)

    original_df[f"refined_{control_flow_column}"] = refined_activities
    original_df.to_csv(output_refined_log_path, index=False)

    return output_refined_log_path


def reset_for_concept_drift(cluster_manager):
    """Reset specific data structures for concept drift handling while preserving learned mappings."""
    cluster_manager.reset_all_clustering_states()
    event_embedding_mapping.clear()


def get_config_values():
    """Extract specific configuration values for execution."""
    return {
        "control_flow_column": control_flow_column,
        "timestamp_column": timestamp_column,
        "resource_column": resource_column,
        "case_id_column": case_id_column,
        "event_id_column": event_id_column,
        "excluded_columns": list(excluded_columns),
        "training_mode_config": training_mode_config,
        "use_control_flow_features": use_control_flow_features,
        "control_flow_context_window": control_flow_context_window,
        "control_flow_pattern_window_size": control_flow_pattern_window_size,
        "max_control_flow_patterns": max_control_flow_patterns,
        "embedding_dim": embedding_dim,
        "parallelization_threshold": parallelization_threshold,
        "unfolding_threshold": unfolding_threshold,
        "autoencoder_params": autoencoder_params,
        "clustering_params": clustering_params,
        "feature_selection_params": feature_selection_params,
        "categorical_feature_params": categorical_feature_params,
        "forgetting_params": forgetting_params,
    }


def save_trained_models(autoencoders, scalers, models_dir, activity_embeddings, online_onehot_encoders):
    """Save all trained models and encoders."""
    os.makedirs(models_dir, exist_ok=True)

    for activity_label, autoencoder in autoencoders.items():
        model_path = os.path.join(models_dir, f"final_autoencoder_{activity_label}.pth")
        torch.save(autoencoder.state_dict(), model_path)

    for activity_label, scaler in scalers.items():
        scaler_path = os.path.join(models_dir, f"final_scaler_{activity_label}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    embeddings_path = os.path.join(models_dir, "activity_embeddings.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(activity_embeddings, f)

    encoders_path = os.path.join(models_dir, "onehot_encoders.pkl")
    with open(encoders_path, 'wb') as f:
        pickle.dump(online_onehot_encoders, f)

    metadata = {
        "trained_activities": list(autoencoders.keys()),
        "autoencoder_config": autoencoder_params,
        "training_timestamp": pd.Timestamp.now().isoformat()
    }
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_trained_models(models_base_dir, training_folder, force_cpu=False):
    """Load pre-trained models, activity embeddings, encoders, and global state."""
    folder_name = os.path.basename(training_folder.rstrip('/\\'))
    models_dir = os.path.join(models_base_dir, f"trained_models_{folder_name}")

    if not os.path.exists(models_dir):
        return {}, {}, None, {}

    autoencoders = {}
    scalers = {}
    activity_embeddings = None
    online_onehot_encoders = {}

    discovered_activities = set()
    for filename in os.listdir(models_dir):
        if filename.startswith("autoencoder_") and filename.endswith(".pth"):
            activity_label = filename[len("autoencoder_"):-len(".pth")]
            discovered_activities.add(activity_label)

    if discovered_activities:
        device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for activity_label in sorted(discovered_activities):
            try:
                autoencoder_path = os.path.join(models_dir, f"autoencoder_{activity_label}.pth")
                scaler_path = os.path.join(models_dir, f"scaler_{activity_label}.pkl")

                if not os.path.exists(autoencoder_path) or not os.path.exists(scaler_path):
                    continue

                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                if hasattr(scaler, 'n_features_in_'):
                    input_dim = scaler.n_features_in_
                elif hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    input_dim = len(scaler.scale_)
                else:
                    continue

                autoencoder = SparseDenoisingAutoencoder(
                    input_dim=input_dim,
                    latent_dim=autoencoder_params.get("latent_dim", 32),
                    hidden_dims=autoencoder_params.get("hidden_dims", [128, 64]),
                    sparsity_lambda=autoencoder_params.get("sparsity_lambda", 0.001),
                    noise_std=autoencoder_params.get("noise_std", 0.1)
                ).to(device)

                state_dict = torch.load(autoencoder_path, map_location=device)
                autoencoder.load_state_dict(state_dict)
                autoencoder.eval()

                autoencoders[activity_label] = autoencoder
                scalers[activity_label] = scaler

            except Exception:
                continue

    embeddings_path = os.path.join(models_dir, "activity_embeddings.pkl")
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, 'rb') as f:
                activity_embeddings = pickle.load(f)
        except Exception:
            activity_embeddings = None

    encoders_path = os.path.join(models_dir, "onehot_encoders.pkl")
    if os.path.exists(encoders_path):
        try:
            with open(encoders_path, 'rb') as f:
                online_onehot_encoders = pickle.load(f)
        except Exception:
            online_onehot_encoders = {}

    global_state_path = os.path.join(models_dir, "global_state.pkl")
    if os.path.exists(global_state_path):
        try:
            with open(global_state_path, 'rb') as f:
                global_state = pickle.load(f)

            from src.duplimend_framework.utils.global_state import (
                directly_follows_graph,
                per_case_directly_follows_graph,
                case_activity_sequences
            )

            if "directly_follows_graph" in global_state:
                directly_follows_graph.__dict__.update(global_state["directly_follows_graph"].__dict__)

            if "per_case_directly_follows_graph" in global_state:
                per_case_directly_follows_graph.clear()
                per_case_directly_follows_graph.update(global_state["per_case_directly_follows_graph"])

            if "case_activity_sequences" in global_state:
                case_activity_sequences.clear()
                case_activity_sequences.update(global_state["case_activity_sequences"])

        except Exception:
            pass

    return autoencoders, scalers, activity_embeddings, online_onehot_encoders


def get_files_from_folder(folder_path, file_pattern="*.csv"):
    """Get all files matching pattern from a folder."""
    if not os.path.exists(folder_path):
        return []

    pattern = os.path.join(folder_path, file_pattern)
    files = glob.glob(pattern)
    files.sort()
    return files


def process_test_file(test_file_path, autoencoders, scalers, activity_embeddings,
                      online_onehot_encoders, data_columns, cluster_manager, label_refiner,
                      output_dir, cluster_tracker, global_event_counter):
    """Process a single test file for inference and clustering."""
    from src.duplimend_framework.utils.global_state import (
        event_embedding_mapping,
        per_case_directly_follows_graph,
        case_activity_sequences
    )
    event_embedding_mapping.clear()
    per_case_directly_follows_graph.clear()
    case_activity_sequences.clear()

    if hasattr(directly_follows_graph, 'reset_timestamps_for_testing'):
        directly_follows_graph.reset_timestamps_for_testing(new_base_event=global_event_counter)
    else:
        if hasattr(directly_follows_graph, 'reset'):
            directly_follows_graph.reset()
        else:
            directly_follows_graph.__init__()

    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    test_output_dir = os.path.join(output_dir, f"test_results_{test_file_name}")
    os.makedirs(test_output_dir, exist_ok=True)

    event_feature_vector_path = os.path.join(test_output_dir, "event_feature_vectors.jsonl")
    event_feature_vector_file = open(event_feature_vector_path, "w")

    refined_log_path = os.path.join(test_output_dir, f"refined_{test_file_name}.csv")

    cluster_manager.reset_all_clustering_states()

    all_columns = get_csv_columns(test_file_path)
    input_columns = all_columns + ["refined_activity"]
    test_label_refiner = LabelRefiner(refined_log_path, input_columns)

    checkpoint_interval = training_mode_config.get("checkpoint_interval")
    test_event_counter = 0
    event_vectors_buffer = []

    processed_event_ids = set()
    skipped_events = 0
    unseen_activities = set()

    for event_idx, event_dict in enumerate(stream_csv_row_by_row(test_file_path)):
        try:
            event_id = event_dict.get(event_id_column)
            if not event_id or event_id in processed_event_ids:
                continue

            global_event_counter += 1
            test_event_counter += 1
            case_id = event_dict[case_id_column]
            activity_label = event_dict[control_flow_column]

            if use_control_flow_features:
                update_case_sequence(case_id, activity_label, event_dict[timestamp_column], global_event_counter)

                if case_id in case_activity_sequences and len(case_activity_sequences[case_id]) > 1:
                    prev_activity = case_activity_sequences[case_id][-2]['activity']
                    directly_follows_graph.add_transition(case_id, prev_activity, activity_label, global_event_counter)
                    per_case_directly_follows_graph[case_id][(prev_activity, activity_label)] += 1

                control_flow_features = extract_control_flow_features(case_id, activity_label)
            else:
                control_flow_features = None

            feature_vector = FeatureVectorBuilder.build(
                event_dict, data_columns, control_flow_features, online_onehot_encoders,
                activity_embeddings, excluded_columns, use_control_flow_features
            )
            feature_vector = np.array(feature_vector, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            if activity_label in autoencoders and activity_label in scalers:
                scaler = scalers[activity_label]
                autoencoder = autoencoders[activity_label]

                if isinstance(autoencoder, dict) and autoencoder.get("type") == "lazy_load":
                    model_path = autoencoder["path"]
                    input_dim = len(feature_vector)

                    autoencoder_loaded = SparseDenoisingAutoencoder(
                        input_dim=input_dim,
                        latent_dim=autoencoder_params["latent_dim"],
                        hidden_dims=autoencoder_params["hidden_dims"],
                        sparsity_lambda=autoencoder_params["sparsity_lambda"],
                        noise_std=autoencoder_params["noise_std"]
                    )
                    autoencoder_loaded.load_state_dict(torch.load(model_path, map_location='cpu'))
                    autoencoder_loaded.eval()
                    autoencoders[activity_label] = autoencoder_loaded
                    autoencoder = autoencoder_loaded

                feature_vector_scaled = scaler.transform([feature_vector])[0]
                x = torch.tensor(feature_vector_scaled, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    _, embedding = autoencoder(x)

                embedding_np = embedding.squeeze(0).cpu().numpy()
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding_np = embedding_np / norm

                event_embedding_mapping[event_id] = embedding_np

                cluster_id, details = cluster_manager.process_event(
                    event_id,
                    activity_label,
                    embedding_np,
                    [f"f_{i}" for i in range(len(embedding_np))],
                    timestamp=event_dict[timestamp_column]
                )

                event_vector_data = {
                    "event_id": event_id,
                    "activity": activity_label,
                    "timestamp": str(event_dict[timestamp_column]),
                    "feature_vector": embedding_np.tolist(),
                    "feature_names": [f"f_{i}" for i in range(len(embedding_np))],
                    "cluster_id": cluster_id,
                    "processing_method": "inference",
                    "test_file": test_file_name,
                    "test_event_position": test_event_counter,
                    "global_event_position": global_event_counter
                }

                event_feature_vector_file.write(json.dumps(event_vector_data) + "\n")
                event_vectors_buffer.append(event_vector_data)

                if test_event_counter % checkpoint_interval == 0:
                    save_checkpoint_state(
                        checkpoint_event_count=test_event_counter,
                        total_processed=global_event_counter,
                        cluster_manager=cluster_manager,
                        cluster_tracker=cluster_tracker,
                        event_vectors_buffer=event_vectors_buffer,
                        output_dir=test_output_dir,
                        original_file_path=test_file_path,
                        file_name_prefix=test_file_name
                    )

                refined_event = test_label_refiner.process_event(
                    event_dict, cluster_id, None,
                    {"split_occurred": details.get("is_new_cluster", False),
                     "merge_occurred": details.get("merge_occurred", False)}
                )

                test_label_refiner.append_event_to_csv(refined_event)
            else:
                unseen_activities.add(activity_label)
                skipped_events += 1

            processed_event_ids.add(event_id)

            if global_event_counter % forgetting_params["decay_after_events"] == 0:
                forget_old_cases(global_event_counter)
                directly_follows_graph.apply_forgetting(global_event_counter)

        except Exception:
            continue

    event_feature_vector_file.close()
    centroids_path = os.path.join(test_output_dir, f"final_centroids.json")
    cluster_tracker.save_final_centroids(cluster_manager, centroids_path, test_file_name)

    if test_event_counter > 0 and len(event_vectors_buffer) > 0:
        save_checkpoint_state(
            checkpoint_event_count=test_event_counter,
            total_processed=global_event_counter,
            cluster_manager=cluster_manager,
            cluster_tracker=cluster_tracker,
            event_vectors_buffer=event_vectors_buffer,
            output_dir=test_output_dir,
            original_file_path=test_file_path,
            file_name_prefix=f"{test_file_name}_final"
        )

    refined_log_with_clusters_path = os.path.join(test_output_dir, f"refined_log_with_clusters_{test_file_name}.csv")

    create_refined_log_with_cluster_mapping(
        original_file_path=test_file_path,
        event_vectors_path=event_feature_vector_path,
        centroids_path=centroids_path,
        output_refined_log_path=refined_log_with_clusters_path
    )

    return test_output_dir, global_event_counter


def log_resource_usage(event_idx, global_event_counter, output_dir):
    """Log resource usage to file."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent(interval=0.1)
    log_line = f"Event {event_idx} | Global {global_event_counter} | Memory: {mem_mb:.2f} MB | CPU: {cpu_percent:.1f}%"
    with open(os.path.join(output_dir, "resource_usage.log"), "a") as f:
        f.write(log_line + "\n")


def stream_csv_row_by_row(file_path, encoding='ISO-8859-1'):
    """Stream CSV file row by row with timestamp parsing."""
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        csv_reader = csv.DictReader(file)

        for row_dict in csv_reader:
            if timestamp_column in row_dict:
                try:
                    timestamp_value = row_dict[timestamp_column]
                    try:
                        row_dict[timestamp_column] = float(timestamp_value)
                        yield row_dict
                    except (ValueError, TypeError):
                        timestamp = pd.to_datetime(timestamp_value, errors='coerce', utc=True)
                        if pd.notna(timestamp):
                            row_dict[timestamp_column] = timestamp
                            yield row_dict
                except Exception:
                    continue
            else:
                yield row_dict


def get_csv_columns(file_path, encoding='ISO-8859-1'):
    """Get column names from CSV file."""
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader.fieldnames)


def incremental_autoencoder_training(activity_buffers, autoencoders, scalers, models_dir,
                                     training_round, is_retraining=False, cluster_manager=None,
                                     is_warmup_phase=False):
    """Enhanced with cluster-aware regularization, memory protection, and drift detection"""

    # Enhanced cluster regularization config with memory protection
    global epoch, epoch_step
    cluster_reg_config = autoencoder_params.get("cluster_regularization", {})

    # Memory regularization config for catastrophic forgetting protection
    memory_config = autoencoder_params.get("memory_regularization", {})

    # DISABLE REGULARIZATION DURING WARMUP
    if is_warmup_phase:
        print(f"[WARMUP] Disabling cluster-aware and memory-aware regularization during warmup phase")
        cluster_reg_config = None  # Disable cluster regularization
        memory_config = {"enable_memory_regularization": False}  # Disable memory regularization

    run_name = f"round_{training_round}" if not is_retraining else f"retrain_{training_round}"
    tb_log_dir = os.path.join(models_dir, "tensorboard_logs", run_name)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Create writer with specific directory
    writer = SummaryWriter(log_dir=tb_log_dir)

    print(f"[TENSORBOARD] Creating logs in: {tb_log_dir}")
    print(f"[TENSORBOARD] Full path structure: {os.path.abspath(tb_log_dir)}")

    # Enhanced training parameters
    train_epochs = autoencoder_params.get("incremental_train_epochs", 50)
    min_events_for_training = training_mode_config.get("min_events_for_incremental_training", 1)
    learning_rate = autoencoder_params.get("learning_rate", 0.001)
    incremental_lr = autoencoder_params.get("incremental_learning_rate", 0.0005)
    memory_update_interval = autoencoder_params.get("memory_update_interval", 10)

    # Early stopping parameters
    early_stopping_patience = autoencoder_params.get("early_stopping_patience", 10)
    min_delta = autoencoder_params.get("min_delta", 1e-6)

    # Learning rate scheduler parameters
    min_lr_threshold = autoencoder_params.get("min_lr", 1e-7)

    trained_activities = []
    global_step = 0

    # Log only once per training round (avoid duplicate graphs)
    graph_logged = False

    for activity_label, buffer in activity_buffers.items():
        if len(buffer) >= min_events_for_training:
            if is_warmup_phase:
                print(
                    f"[WARMUP] Training '{activity_label}' with {len(buffer)} events (basic mode - no regularization)")
            else:
                print(
                    f"[TRAINING] Training '{activity_label}' with {len(buffer)} events (enhanced mode - with regularization)")

            # Extract feature vectors and event dictionaries from buffer
            feature_vectors = []
            event_dicts = []
            for event_dict, fv in buffer:
                feature_vectors.append(fv)
                event_dicts.append(event_dict)

            # Handle scaler
            if activity_label not in scalers or is_retraining:
                scalers[activity_label] = RobustScaler()
                X_scaled = scalers[activity_label].fit_transform(np.stack(feature_vectors))
            else:
                X_scaled = scalers[activity_label].transform(np.stack(feature_vectors))

            input_dim = len(feature_vectors[0])

            # Create cluster-aware autoencoder with conditional regularization
            if activity_label not in autoencoders or is_retraining:
                autoencoder = ClusterAwareAutoencoder(
                    input_dim=input_dim,
                    latent_dim=autoencoder_params["latent_dim"],
                    hidden_dims=autoencoder_params["hidden_dims"],
                    sparsity_lambda=autoencoder_params["sparsity_lambda"],
                    noise_std=autoencoder_params["noise_std"],
                    cluster_regularization_config=cluster_reg_config,  # None during warmup
                    memory_config=memory_config  # Disabled during warmup
                )

                # Load existing memory state if available
                memory_state_path = os.path.join(models_dir, f"memory_state_{activity_label}.json")
                if os.path.exists(memory_state_path) and not is_retraining:
                    autoencoder.load_memory_state(memory_state_path)
                    print(f"[MEMORY] Loaded existing memory state for '{activity_label}'")

                autoencoders[activity_label] = autoencoder
                current_lr = learning_rate
            else:
                autoencoder = autoencoders[activity_label]

                # DYNAMICALLY ENABLE/DISABLE REGULARIZATION
                if is_warmup_phase:
                    autoencoder.enable_cluster_regularization = False
                    autoencoder.enable_memory_regularization = False
                    print(f"[WARMUP] Disabled regularization for existing autoencoder: {activity_label}")
                else:
                    autoencoder.enable_cluster_regularization = True
                    autoencoder.enable_memory_regularization = True
                    print(f"[POST-WARMUP] Enabled regularization for autoencoder: {activity_label}")

                # Convert to cluster-aware if not already and not in warmup
                if not hasattr(autoencoder, 'cluster_regularizer') and not is_warmup_phase:
                    cluster_autoencoder = ClusterAwareAutoencoder(
                        input_dim=input_dim,
                        latent_dim=autoencoder_params["latent_dim"],
                        hidden_dims=autoencoder_params["hidden_dims"],
                        sparsity_lambda=autoencoder_params["sparsity_lambda"],
                        noise_std=autoencoder_params["noise_std"],
                        cluster_regularization_config=cluster_reg_config,
                        memory_config=memory_config
                    )
                    # Copy weights from old autoencoder
                    cluster_autoencoder.load_state_dict(autoencoder.state_dict())

                    # Load memory state if available
                    memory_state_path = os.path.join(models_dir, f"memory_state_{activity_label}.json")
                    if os.path.exists(memory_state_path):
                        cluster_autoencoder.load_memory_state(memory_state_path)
                        print(f"[MEMORY] Loaded memory state for converted autoencoder '{activity_label}'")

                    autoencoders[activity_label] = cluster_autoencoder
                    autoencoder = cluster_autoencoder

                current_lr = incremental_lr

            # Update training state
            autoencoder.update_training_state(activity_label)

            # Convert to tensors
            X = torch.tensor(X_scaled, dtype=torch.float32)

            # GET CLUSTER ASSIGNMENTS - Enhanced version
            cluster_assignments = []
            if is_warmup_phase:
                cluster_assignments = [-1] * len(buffer)
                print(f"[WARMUP] Using dummy cluster assignments for '{activity_label}' (no clustering during warmup)")
            elif cluster_manager and activity_label in cluster_manager.clustering_adapters:
                adapter = cluster_manager.get_or_create_adapter(activity_label)

                for fv in feature_vectors:
                    try:
                        if hasattr(adapter, 'predict_cluster'):
                            cluster_id = adapter.predict_cluster(fv)
                        else:
                            significant_clusters = adapter.get_significant_clusters(cluster_manager.min_cluster_weight)
                            if significant_clusters:
                                best_cluster = -1
                                best_distance = float('inf')

                                for cid in significant_clusters:
                                    centroid = adapter.get_cluster_centroid(cid)
                                    if centroid is not None and len(centroid) == len(fv):
                                        distance = np.linalg.norm(fv - centroid)
                                        if distance < best_distance:
                                            best_distance = distance
                                            best_cluster = cid

                                cluster_id = best_cluster
                            else:
                                cluster_id = -1  # No clusters available yet

                        cluster_assignments.append(cluster_id)

                    except Exception as e:
                        print(f"[WARNING] Error getting cluster assignment: {e}")
                        cluster_assignments.append(-1)
            else:
                # NO CLUSTER DATA: Use dummy assignments
                cluster_assignments = [-1] * len(buffer)
                print(f"[INFO] No cluster data available for '{activity_label}', using dummy assignments")

            print(f"[TRAINING] Got {len(cluster_assignments)} cluster assignments for '{activity_label}'")
            print(f"[TRAINING] Unique clusters: {set(cluster_assignments)}")

            # Enhanced optimizer
            optimizer = torch.optim.Adam(
                autoencoder.parameters(),
                lr=current_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-5
            )

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.7,
                patience=5,
                min_lr=min_lr_threshold
            )

            # Early stopping variables
            best_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            autoencoder.train()

            # Log model architecture to TensorBoard (ONLY ONCE PER ROUND)
            if not graph_logged:
                try:
                    writer.add_graph(autoencoder, X[:1])
                    graph_logged = True
                    print(f"[TENSORBOARD] Model graph logged for round {training_round}")
                except Exception as e:
                    print(f"[WARNING] Could not log model graph to TensorBoard: {e}")

            # Log hyperparameters
            hparams = {
                'learning_rate': current_lr,
                'latent_dim': autoencoder_params["latent_dim"],
                'hidden_dims': str(autoencoder_params["hidden_dims"]),
                'sparsity_lambda': autoencoder_params["sparsity_lambda"],
                'noise_std': autoencoder_params["noise_std"],
                'activity': activity_label,
                'training_round': str(training_round),
                'buffer_size': len(buffer),
                'cluster_regularization_enabled': hasattr(autoencoder,
                                                          'cluster_regularizer') and autoencoder.enable_cluster_regularization,
                'memory_protection_enabled': hasattr(autoencoder,
                                                     'memory_manager') and autoencoder.enable_memory_regularization,
                'is_warmup_phase': is_warmup_phase
            }

            training_mode_desc = "basic (warmup)" if is_warmup_phase else "enhanced (with regularization)"
            print(f"[TRAINING] {activity_label}: Starting {training_mode_desc} training with LR={current_lr}")

            for epoch in range(train_epochs):
                optimizer.zero_grad()

                # Use cluster-aware forward pass with memory protection (or standard if warmup)
                if hasattr(autoencoder, 'forward_with_clusters') and not is_warmup_phase:
                    x_hat, z, losses = autoencoder.forward_with_clusters(
                        X, cluster_assignments, activity_label
                    )
                else:
                    # Standard forward pass during warmup
                    x_hat, z = autoencoder.forward(X)
                    reconstruction_loss = torch.nn.functional.mse_loss(x_hat, X)
                    sparsity_loss = autoencoder_params["sparsity_lambda"] * torch.mean(torch.abs(z))
                    losses = {
                        'reconstruction_loss': reconstruction_loss,
                        'sparsity_loss': sparsity_loss,
                        'total_loss': reconstruction_loss + sparsity_loss
                    }

                total_loss = losses['total_loss']
                total_loss.backward()

                # Calculate gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step(total_loss.item())

                # Enhanced TensorBoard logging
                epoch_step = global_step + epoch
                writer.add_scalar(f'{activity_label}/Total_Loss', total_loss.item(), epoch_step)
                writer.add_scalar(f'{activity_label}/Reconstruction_Loss', losses['reconstruction_loss'].item(),
                                  epoch_step)
                writer.add_scalar(f'{activity_label}/Sparsity_Loss', losses['sparsity_loss'].item(), epoch_step)
                writer.add_scalar(f'{activity_label}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch_step)
                writer.add_scalar(f'{activity_label}/Gradient_Norm', grad_norm.item(), epoch_step)

                # Log cluster regularization components if available
                if 'cluster_regularization_loss' in losses:
                    writer.add_scalar(f'{activity_label}/Cluster_Regularization_Loss',
                                      losses['cluster_regularization_loss'].item(), epoch_step)
                if 'separation_loss' in losses:
                    writer.add_scalar(f'{activity_label}/Separation_Loss', losses['separation_loss'], epoch_step)
                if 'compactness_loss' in losses:
                    writer.add_scalar(f'{activity_label}/Compactness_Loss', losses['compactness_loss'], epoch_step)
                if 'consistency_loss' in losses:
                    writer.add_scalar(f'{activity_label}/Consistency_Loss', losses['consistency_loss'], epoch_step)

                # Log memory regularization metrics if available
                if 'memory_regularization_loss' in losses:
                    writer.add_scalar(f'{activity_label}/Memory_Regularization_Loss',
                                      losses['memory_regularization_loss'].item(), epoch_step)
                if 'matched_points' in losses:
                    writer.add_scalar(f'{activity_label}/Memory_Matched_Points', losses['matched_points'], epoch_step)
                if 'match_ratio' in losses:
                    writer.add_scalar(f'{activity_label}/Memory_Match_Ratio', losses['match_ratio'], epoch_step)
                if 'stored_clusters' in losses:
                    writer.add_scalar(f'{activity_label}/Memory_Stored_Clusters', losses['stored_clusters'], epoch_step)

                # Log latent space statistics
                with torch.no_grad():
                    latent_mean = torch.mean(z).item()
                    latent_std = torch.std(z).item()
                    latent_sparsity = (torch.abs(z) < 1e-3).float().mean().item()

                    writer.add_scalar(f'{activity_label}/Latent_Mean', latent_mean, epoch_step)
                    writer.add_scalar(f'{activity_label}/Latent_Std', latent_std, epoch_step)
                    writer.add_scalar(f'{activity_label}/Latent_Sparsity', latent_sparsity, epoch_step)

                # Log weight distributions every 10 epochs
                if epoch % memory_update_interval == 0:
                    for name, param in autoencoder.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'{activity_label}/Weights/{name}', param.data, epoch_step)
                            writer.add_histogram(f'{activity_label}/Gradients/{name}', param.grad.data, epoch_step)

                # Check cluster quality periodically (only if not warmup and has cluster regularizer)
                if (epoch % memory_update_interval == 0 and len(set(cluster_assignments)) > 1 and
                        hasattr(autoencoder, 'check_cluster_quality') and not is_warmup_phase):
                    with torch.no_grad():
                        quality_result = autoencoder.check_cluster_quality(z, cluster_assignments, activity_label)
                        writer.add_scalar(f'{activity_label}/Cluster_Quality_OK',
                                          1.0 if quality_result['quality_ok'] else 0.0, epoch_step)

                        if not quality_result['quality_ok']:
                            print(
                                f"[WARNING] {activity_label} Epoch {epoch + 1}: Cluster quality degradation detected: {quality_result['degradation_signals']}")

                # Update memory periodically during training (only if not warmup)
                if (epoch % memory_update_interval == 0 and hasattr(autoencoder, 'memory_manager') and
                        not is_warmup_phase and autoencoder.enable_memory_regularization):
                    with torch.no_grad():
                        autoencoder.update_memory_from_clusters(
                            activity_label, z, cluster_assignments,
                            global_step + epoch
                        )

                # Early stopping logic
                if total_loss.item() < best_loss - min_delta:
                    best_loss = total_loss.item()
                    patience_counter = 0
                    best_model_state = autoencoder.state_dict().copy()
                    writer.add_scalar(f'{activity_label}/Best_Loss', best_loss, epoch_step)
                    if epoch % 5 == 0:
                        mode_str = "(warmup)" if is_warmup_phase else "(enhanced)"
                        print(
                            f"[TRAINING] {activity_label} {mode_str} Epoch {epoch + 1}: Loss={total_loss.item():.6f} (NEW BEST) LR={optimizer.param_groups[0]['lr']:.8f}")
                else:
                    patience_counter += 1
                    if epoch % memory_update_interval == 0:
                        mode_str = "(warmup)" if is_warmup_phase else "(enhanced)"
                        print(
                            f"[TRAINING] {activity_label} {mode_str} Epoch {epoch + 1}: Loss={total_loss.item():.6f} (patience={patience_counter}/{early_stopping_patience}) LR={optimizer.param_groups[0]['lr']:.8f}")

                writer.add_scalar(f'{activity_label}/Patience_Counter', patience_counter, epoch_step)

                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    print(
                        f"[TRAINING] {activity_label}: Early stopping at epoch {epoch + 1} (best_loss={best_loss:.6f})")
                    break

                # Check if learning rate is too low
                if optimizer.param_groups[0]['lr'] < min_lr_threshold * 1.1:
                    print(f"[TRAINING] {activity_label}: Learning rate too low, stopping at epoch {epoch + 1}")
                    break

            # Restore best model
            if best_model_state is not None:
                autoencoder.load_state_dict(best_model_state)
                print(f"[TRAINING] {activity_label}: Restored best model (loss={best_loss:.6f})")

            autoencoder.eval()
            final_epochs = epoch + 1

            # Save memory state after training (only if not warmup)
            if (hasattr(autoencoder, 'memory_manager') and not is_warmup_phase and
                    autoencoder.enable_memory_regularization):
                memory_state_path = os.path.join(models_dir, f"memory_state_{activity_label}.json")
                autoencoder.save_memory_state(memory_state_path)
                print(f"[MEMORY] Saved memory state for '{activity_label}' to: {memory_state_path}")

                # Log memory statistics
                memory_stats = autoencoder.get_memory_statistics()
                print(f"[MEMORY] Memory statistics for '{activity_label}': {memory_stats}")

                # Log memory stats to TensorBoard
                if memory_stats:
                    writer.add_scalar(f'{activity_label}/Memory_Total_Centroids',
                                      memory_stats.get('total_centroids', 0), epoch_step)
                    writer.add_scalar(f'{activity_label}/Memory_Total_Exemplars',
                                      memory_stats.get('total_exemplars', 0), epoch_step)

            # Log final hyperparameters with enhanced metrics
            try:
                final_metrics = {
                    'final_loss': best_loss,
                    'epochs_trained': final_epochs,
                    'early_stopped': patience_counter >= early_stopping_patience,
                    'final_lr': optimizer.param_groups[0]['lr'],
                    'warmup_phase': is_warmup_phase
                }

                # Add memory metrics if available
                if (hasattr(autoencoder, 'memory_manager') and not is_warmup_phase and
                        autoencoder.enable_memory_regularization):
                    memory_summary = autoencoder.get_memory_statistics()
                    if memory_summary:
                        final_metrics.update({
                            'memory_centroids': memory_summary.get('total_centroids', 0),
                            'memory_exemplars': memory_summary.get('total_exemplars', 0)
                        })

                writer.add_hparams(hparams, final_metrics)
            except Exception as e:
                print(f"[WARNING] Could not log hyperparameters to TensorBoard: {e}")

            trained_activities.append(activity_label)

            mode_str = "WARMUP" if is_warmup_phase else "ENHANCED"
            print(
                f"[TRAINING] {activity_label} {mode_str} COMPLETED - Final Loss: {best_loss:.6f}, Epochs: {final_epochs}/{train_epochs}")
            if hasattr(autoencoder, 'cluster_regularizer') and autoencoder.enable_cluster_regularization:
                print(f"[TRAINING] {activity_label} - Cluster regularization: ENABLED")
            else:
                print(f"[TRAINING] {activity_label} - Cluster regularization: DISABLED")
            if hasattr(autoencoder, 'memory_manager') and autoencoder.enable_memory_regularization:
                print(f"[TRAINING] {activity_label} - Memory protection: ENABLED")
            else:
                print(f"[TRAINING] {activity_label} - Memory protection: DISABLED")

            # Update global step for next activity
            global_step += final_epochs

    writer.close()
    print(f"[TENSORBOARD] Logs saved to: {tb_log_dir}")
    print(f"[TENSORBOARD] Parent directory for TensorBoard: {models_dir}")
    print(f"[TENSORBOARD] Refresh browser to see new data at http://localhost:6007")

    # Enhanced summary
    phase_desc = "warmup" if is_warmup_phase else "enhanced"
    print(f"\n[TRAINING-SUMMARY] {phase_desc.title()} training completed:")
    print(f"  - Activities trained: {len(trained_activities)}")
    print(f"  - Phase: {phase_desc}")
    print(
        f"  - Cluster-aware regularization: {'DISABLED (warmup)' if is_warmup_phase else ('ENABLED' if cluster_reg_config else 'DISABLED')}")
    print(
        f"  - Memory protection: {'DISABLED (warmup)' if is_warmup_phase else ('ENABLED' if memory_config.get('enable_memory_regularization') else 'DISABLED')}")
    print(f"  - Drift detection integration: {'AVAILABLE' if cluster_manager else 'NOT AVAILABLE'}")

    return trained_activities


def process_event_stream(file_path, activity_embeddings, online_onehot_encoders,
                         data_columns, cluster_manager, label_refiner, output_dir, cluster_tracker):
    """
    Process the entire event stream with the new incremental training approach and drift detection.
    """
    # Configuration
    warmup_global_events = training_mode_config.get("warmup_global_events", 10000)
    incremental_interval = training_mode_config.get("incremental_training_interval", 1000)
    retraining_warmup = training_mode_config.get("retraining_warmup_events", 10000)
    buffer_retention_size = autoencoder_params.get("buffer_retention_size", 100)

    drift_config = clustering_params.get("drift_detection", {})
    drift_check_interval = drift_config.get("check_interval", 1000)

    checkpoint_interval = training_mode_config.get("checkpoint_interval")
    inference_event_counter = 0  # Counter for inference events only
    event_vectors_buffer = []  # Buffer to store all event vectors for checkpoints

    # Initialize state
    autoencoders = {}
    scalers = {}
    activity_buffers = {}  # Store events per activity for training
    min_buffer_to_train = training_mode_config.get("min_events_for_dynamic_training", 10)
    global_event_counter = 0
    training_round = 0
    phase = "warmup"  # "warmup", "inference", or "retraining"
    last_retraining_event = 0

    # Per-activity retraining tracking (non-blocking)
    activity_retraining_status = {}  # Track which activities are being retrained
    activity_retraining_buffers = {}  # Separate buffers for retraining
    new_autoencoders = {}  # Store new autoencoders being trained
    new_scalers = {}  # Store new scalers being trained
    retraining_start_events = {}  # Track when retraining started per activity

    # Create models directory - FIX: Use file_path instead of training_folder
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    models_dir = os.path.join(output_dir, f"trained_models_{file_name}")
    os.makedirs(models_dir, exist_ok=True)

    # Setup output files
    event_feature_vector_path = os.path.join(output_dir, "event_feature_vectors.jsonl")
    event_feature_vector_file = open(event_feature_vector_path, "w")

    processed_event_ids = set()
    all_seen_activities = set()

    print(f"[PROCESSING] Starting event stream processing with checkpoint interval: {checkpoint_interval}")
    print(
        f"[PROCESSING] Warmup phase: {warmup_global_events} events with incremental training every {incremental_interval} events")
    print(f"[DRIFT] Drift detection enabled - checking every {drift_check_interval} events")

    for event_idx, event_dict in enumerate(stream_csv_row_by_row(file_path)):
        try:
            event_id = event_dict.get(event_id_column)
            if not event_id or event_id in processed_event_ids:
                continue

            global_event_counter += 1
            case_id = event_dict[case_id_column]
            activity_label = event_dict[control_flow_column]
            all_seen_activities.add(activity_label)

            # Always update control flow features (regardless of phase)
            if use_control_flow_features:
                update_case_sequence(case_id, activity_label, event_dict[timestamp_column], global_event_counter)

                if case_id in case_activity_sequences and len(case_activity_sequences[case_id]) > 1:
                    prev_activity = case_activity_sequences[case_id][-2]['activity']
                    directly_follows_graph.add_transition(case_id, prev_activity, activity_label, global_event_counter)
                    per_case_directly_follows_graph[case_id][(prev_activity, activity_label)] += 1

                control_flow_features = extract_control_flow_features(case_id, activity_label)
            else:
                control_flow_features = None

            # Build feature vector
            feature_vector = FeatureVectorBuilder.build(
                event_dict, data_columns, control_flow_features, online_onehot_encoders,
                activity_embeddings, excluded_columns, use_control_flow_features
            )
            feature_vector = np.array(feature_vector, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            # Determine current phase and handle accordingly
            if phase == "warmup" and global_event_counter <= warmup_global_events:
                # WARMUP PHASE: Collect data and train periodically WITHOUT regularization
                if activity_label not in activity_buffers:
                    activity_buffers[activity_label] = []
                activity_buffers[activity_label].append((event_dict.copy(), feature_vector))

                # Incremental training during warmup
                if global_event_counter % incremental_interval == 0:
                    training_round = global_event_counter // incremental_interval
                    print(
                        f"\n[WARMUP] Incremental training round {training_round} at global event {global_event_counter}")

                    trained_activities = incremental_autoencoder_training(
                        activity_buffers, autoencoders, scalers, models_dir,
                        f"warmup_{training_round}", is_retraining=False, cluster_manager=cluster_manager,
                        is_warmup_phase=True
                    )

                    # Manage buffer sizes
                    for act_label in activity_buffers:
                        if len(activity_buffers[act_label]) > buffer_retention_size:
                            activity_buffers[act_label] = activity_buffers[act_label][-buffer_retention_size:]

                    print(f"[WARMUP] Trained {len(trained_activities)} activities: {trained_activities}")

                # Transition to inference after warmup
                if global_event_counter == warmup_global_events:
                    print(
                        f"\n[PHASE TRANSITION] Warmup complete. Switching to inference phase with enhanced regularization at event {global_event_counter}")
                    phase = "inference"
                    activity_buffers.clear()

            elif phase == "inference" or (
                    phase == "retraining" and global_event_counter > last_retraining_event + retraining_warmup):
                # INFERENCE PHASE: Use trained autoencoders for clustering
                if activity_label in autoencoders and activity_label in scalers:
                    scaler = scalers[activity_label]
                    autoencoder = autoencoders[activity_label]

                    # Generate embedding
                    feature_vector_scaled = scaler.transform([feature_vector])[0]
                    x = torch.tensor(feature_vector_scaled, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        x_reconstructed, embedding = autoencoder(x)
                        # Calculate reconstruction error for drift detection
                        reconstruction_error = torch.nn.functional.mse_loss(x_reconstructed, x).item()

                    embedding_np = embedding.squeeze(0).cpu().numpy()
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding_np = embedding_np / norm

                    # Store embedding
                    event_embedding_mapping[event_id] = embedding_np

                    # Process with cluster manager - now passing reconstruction error and embeddings
                    cluster_id, details = cluster_manager.process_event(
                        event_id,
                        activity_label,
                        embedding_np,
                        [f"f_{i}" for i in range(len(embedding_np))],
                        timestamp=event_dict[timestamp_column],
                        reconstruction_error=reconstruction_error,
                        autoencoder_embedding=embedding_np
                    )

                    adapter = cluster_manager.get_or_create_adapter(activity_label)
                    significant_clusters = adapter.get_significant_clusters(cluster_manager.min_cluster_weight)

                    if significant_clusters and len(significant_clusters) > 0:
                        # Get current centroids
                        centroids = []
                        for cid in significant_clusters:
                            centroid = adapter.get_cluster_centroid(cid)
                            if centroid is not None:
                                centroids.append(centroid)

                        # Update drift detection
                        if centroids:
                            cluster_manager.update_drift_detection(
                                activity_label,
                                centroids,
                                [embedding_np],  # Current embedding
                                global_event_counter
                            )

                    inference_event_counter += 1

                    event_vector_data = {
                        "event_id": event_id,
                        "activity": activity_label,
                        "timestamp": str(event_dict[timestamp_column]),
                        "feature_vector": embedding_np.tolist(),
                        "feature_names": [f"f_{i}" for i in range(len(embedding_np))],
                        "cluster_id": cluster_id,
                        "processing_method": "inference",
                        "global_event_position": global_event_counter,
                        "inference_event_position": inference_event_counter,
                        "phase": phase
                    }

                    event_feature_vector_file.write(json.dumps(event_vector_data) + "\n")
                    event_vectors_buffer.append(event_vector_data)

                    # Save checkpoint at intervals
                    if inference_event_counter % checkpoint_interval == 0:
                        save_checkpoint_state(
                            checkpoint_event_count=inference_event_counter,
                            total_processed=global_event_counter,
                            cluster_manager=cluster_manager,
                            cluster_tracker=cluster_tracker,
                            event_vectors_buffer=event_vectors_buffer,
                            output_dir=output_dir,
                            original_file_path=file_path,
                            file_name_prefix=file_name
                        )

                    # Refine labels
                    refined_event = label_refiner.process_event(
                        event_dict, cluster_id, None,
                        {"split_occurred": details.get("is_new_cluster", False),
                         "merge_occurred": details.get("merge_occurred", False)}
                    )

                    label_refiner.append_event_to_csv(refined_event)
                else:
                    if activity_label not in activity_buffers:
                        activity_buffers[activity_label] = []
                    activity_buffers[activity_label].append((event_dict.copy(), feature_vector))

                    if len(activity_buffers[activity_label]) >= min_buffer_to_train:
                        print(
                            f"[DYNAMIC-TRAINING] Training new autoencoder for unseen activity '{activity_label}' with {len(activity_buffers[activity_label])} buffered events")
                        trained_activities = incremental_autoencoder_training(
                            {activity_label: activity_buffers[activity_label]},
                            autoencoders,
                            scalers,
                            models_dir=models_dir,
                            training_round="dynamic_inference",
                            is_retraining=False,
                            cluster_manager=cluster_manager
                        )
                        activity_buffers[activity_label] = []
                    else:
                        print(
                            f"[DYNAMIC-BUFFERING] Buffering event for unseen activity '{activity_label}' (buffer size = {len(activity_buffers[activity_label])})")

            elif phase == "retraining":
                # RETRAINING PHASE: Use enhanced regularization
                if activity_label not in activity_buffers:
                    activity_buffers[activity_label] = []
                activity_buffers[activity_label].append((event_dict.copy(), feature_vector))

                events_since_retraining = global_event_counter - last_retraining_event
                if events_since_retraining % incremental_interval == 0:
                    retraining_round = events_since_retraining // incremental_interval
                    print(
                        f"\n[RETRAINING] Incremental retraining round {retraining_round} at global event {global_event_counter}")

                    # Reset models on first retraining round
                    is_reset_round = (events_since_retraining == incremental_interval)

                    trained_activities = incremental_autoencoder_training(
                        activity_buffers, autoencoders, scalers, models_dir,
                        f"retrain_{retraining_round}", is_retraining=is_reset_round, cluster_manager=cluster_manager,
                        is_warmup_phase=False
                    )

                    # Manage buffer sizes
                    for act_label in activity_buffers:
                        if len(activity_buffers[act_label]) > buffer_retention_size:
                            activity_buffers[act_label] = activity_buffers[act_label][-buffer_retention_size:]

                    print(f"[RETRAINING] Trained {len(trained_activities)} activities: {trained_activities}")

                # Transition back to inference after retraining warmup
                if events_since_retraining == retraining_warmup:
                    print(
                        f"\n[PHASE TRANSITION] Retraining complete. Switching back to inference phase at event {global_event_counter}")
                    phase = "inference"
                    activity_buffers.clear()

            if phase == "inference":
                # Check if current activity needs retraining based on drift
                if activity_label in all_seen_activities:
                    # Get drift info from the cluster_manager's last process_event call
                    should_retrain, retrain_info = cluster_manager.should_trigger_retraining(activity_label)

                    if should_retrain and activity_label not in activity_retraining_status:
                        # Start retraining for this specific activity (non-blocking)
                        print(
                            f"\n[DRIFT-TRIGGERED] Starting background retraining for '{activity_label}': {retrain_info}")
                        print(f"[DRIFT-TRIGGERED] Continuing to use existing autoencoder until new one is ready")

                        # Mark activity as being retrained
                        activity_retraining_status[activity_label] = "collecting_data"
                        activity_retraining_buffers[activity_label] = []
                        retraining_start_events[activity_label] = global_event_counter

                        # Don't switch phase - keep processing with old model

                # Collect data for activities being retrained
                if activity_label in activity_retraining_status:
                    if activity_retraining_status[activity_label] == "collecting_data":
                        # Add to retraining buffer
                        activity_retraining_buffers[activity_label].append((event_dict.copy(), feature_vector))

                        # Check if we have enough data to start training new model
                        events_collected = len(activity_retraining_buffers[activity_label])
                        if events_collected >= min_buffer_to_train and events_collected % incremental_interval == 0:
                            print(
                                f"[BACKGROUND-RETRAIN] Training new model for '{activity_label}' with {events_collected} events")

                            # Train new model in background (using existing function)
                            new_autoencoders[activity_label] = autoencoders.get(activity_label)  # Start with copy
                            new_scalers[activity_label] = scalers.get(activity_label)

                            # Train the new model
                            trained = incremental_autoencoder_training(
                                {activity_label: activity_retraining_buffers[activity_label]},
                                new_autoencoders,
                                new_scalers,
                                models_dir,
                                f"drift_retrain_{activity_label}_{global_event_counter}",
                                is_retraining=True,
                                cluster_manager=cluster_manager,
                                is_warmup_phase=False
                            )

                            # Switch to new model if training succeeded
                            if activity_label in trained:
                                print(f"[MODEL-SWITCH] Switching to new model for '{activity_label}'")
                                autoencoders[activity_label] = new_autoencoders[activity_label]
                                scalers[activity_label] = new_scalers[activity_label]

                                # Clean up retraining status
                                del activity_retraining_status[activity_label]
                                del activity_retraining_buffers[activity_label]
                                del retraining_start_events[activity_label]
                                if activity_label in new_autoencoders:
                                    del new_autoencoders[activity_label]
                                if activity_label in new_scalers:
                                    del new_scalers[activity_label]

                                # Reset drift detector for this activity
                                cluster_manager.reset_detectors(activity_label)
                                print(f"[MODEL-SWITCH] Successfully switched to new model for '{activity_label}'")

            # Mark as processed
            processed_event_ids.add(event_id)

            # Periodic maintenance (always active)
            if global_event_counter % forgetting_params["decay_after_events"] == 0:
                forget_old_cases(global_event_counter)
                directly_follows_graph.apply_forgetting(global_event_counter)

            # Enhanced periodic logging with drift information
            if global_event_counter % 1000 == 0:
                print(f"[{phase.upper()}] Processed {global_event_counter} events")
                log_resource_usage(event_idx, global_event_counter, output_dir)

                if global_event_counter % (drift_check_interval * 2) == 0:
                    drift_summary = cluster_manager.get_drift_summary()
                    print(f"[DRIFT-SUMMARY] Monitoring {len(drift_summary.get('activities_monitored', []))} activities")
                    print(f"[DRIFT-SUMMARY] Current drift scores: {drift_summary.get('current_drift_scores', {})}")
                    print(f"[DRIFT-SUMMARY] Total drift alerts: {drift_summary.get('total_alerts', 0)}")

        except Exception as e:
            print(f"[ERROR] Error processing event {event_dict.get(event_id_column, 'Unknown')}: {e}")
            continue

    # Final training for any remaining events in buffers
    if activity_buffers:
        print(f"\n[FINAL] Processing remaining buffered events...")
        final_round = f"final_{global_event_counter}"
        trained_activities = incremental_autoencoder_training(
            activity_buffers, autoencoders, scalers, models_dir,
            final_round, is_retraining=False, cluster_manager=cluster_manager
        )
        print(f"[FINAL] Final training completed for {len(trained_activities)} activities")

    # Save final checkpoint with remaining inference events
    if inference_event_counter > 0 and len(event_vectors_buffer) > 0:
        print(f"\n[FINAL-CHECKPOINT] Saving final checkpoint with all {inference_event_counter} inference events")
        save_checkpoint_state(
            checkpoint_event_count=inference_event_counter,
            total_processed=global_event_counter,
            cluster_manager=cluster_manager,
            cluster_tracker=cluster_tracker,
            event_vectors_buffer=event_vectors_buffer,
            output_dir=output_dir,
            original_file_path=file_path,
            file_name_prefix=f"{file_name}_final"
        )

    event_feature_vector_file.close()
    if autoencoders and scalers:
        save_trained_models(autoencoders, scalers, models_dir, activity_embeddings, online_onehot_encoders)
        print(f"[MODELS] Saved {len(autoencoders)} trained models to {models_dir}")

    print(f"\n[COMPLETED] Event stream processing finished at global event {global_event_counter}")
    print(f"[COMPLETED] Total inference events: {inference_event_counter}")
    print(f"[DRIFT] Final drift summary: {cluster_manager.get_drift_summary()}")

    return all_seen_activities


def reset_global_state():
    """Reset all global state for fresh runs"""
    global event_embedding_mapping, directly_follows_graph
    global per_case_directly_follows_graph, case_activity_sequences

    from src.duplimend_framework.utils.global_state import (
        activity_positions,
        activity_instance_counts,
        activity_hash_mapping,
        self_loop_cases,
        case_context_freeze,
        case_last_seen_global,
        last_activity_per_case
    )

    # Clear all control flow structures (excluding patterns)
    event_embedding_mapping.clear()
    per_case_directly_follows_graph.clear()
    case_activity_sequences.clear()
    activity_positions.clear()
    activity_instance_counts.clear()
    activity_hash_mapping.clear()
    self_loop_cases.clear()
    case_context_freeze.clear()
    case_last_seen_global.clear()
    last_activity_per_case.clear()

    # Reset DFG properly
    if hasattr(directly_follows_graph, 'reset'):
        directly_follows_graph.reset()
    else:
        # Reinitialize if no reset method
        directly_follows_graph.__init__()

    print("[RESET] All control flow structures reset for fresh run")


def set_all_seeds(seed=42, force_cpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Skip CUDA operations in offline mode to avoid conflicts
    if not force_cpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif force_cpu:
        print(f"[CONFIG] Skipping CUDA initialization for offline mode")


def main():
    global all_seen_activities, data_columns, all_columns, tensorboard_log_dir
    parser = argparse.ArgumentParser(description='Run DupliMend pipeline with multi-file training and testing')
    parser.add_argument('--mode', type=str, choices=['online_mode', 'offline_mode'],
                        default=training_mode_config.get("mode", "online_mode"),
                        help='Processing mode')
    parser.add_argument('--input', type=str,
                        default=training_mode_config["default_input_file"],
                        help='Input file path (for online_mode)')
    parser.add_argument('--load-models', action='store_true',
                        default=training_mode_config.get("load_pretrained_models"),
                        help='Load pre-trained models instead of training')
    parser.add_argument('--models-dir', type=str,
                        help='Directory containing pre-trained models (for offline_mode with --load-models)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for refined log (online_mode mode)')
    args = parser.parse_args()

    reset_global_state()
    # Force CPU mode for offline processing to avoid CUDA conflicts
    force_cpu_mode = (args.mode == "offline_mode")
    set_all_seeds(42, force_cpu=force_cpu_mode)
    print(f"[CONFIG] Processing mode: {args.mode}")
    print(f"[CONFIG] Load pre-trained models: {args.load_models}")
    if force_cpu_mode:
        print(f"[CONFIG] CPU-only mode enabled for offline processing")

    # Create output directory
    output_dir = os.path.dirname(args.output) if args.output else "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    results_base_dir = evaluation_config.get("results_base_dir")

    # Initialize components
    activity_embeddings = DynamicActivityEmbeddings(embedding_dim=embedding_dim, seed=42)
    cluster_tracker = ClusterEvolutionTracker(output_dir=results_base_dir)
    if not args.output:
        args.output = os.path.join(cluster_tracker.output_dir, "refined_log.csv")
    config_values = get_config_values()
    cluster_tracker.save_config_snapshot(config_values)

    # Also save config using shared utility for consistency - save inside tracking folder
    save_config_to_output(cluster_tracker.output_dir, "main_execution", {"results_base_dir": results_base_dir, "tracking_dir": cluster_tracker.output_dir})
    cluster_manager = ClusterManager(config=clustering_params, tracker=cluster_tracker)

    # Initialize one-hot encoders
    online_onehot_encoders = {}
    categorical_columns = categorical_feature_params.get("categorical_columns", [])
    for col in categorical_columns:
        if col and col not in online_onehot_encoders and col not in excluded_columns:
            online_onehot_encoders[col] = OnlineOneHotEncoder(max_categories)
    if (resource_column and
            resource_column not in online_onehot_encoders and
            resource_column not in excluded_columns):
        online_onehot_encoders[resource_column] = OnlineOneHotEncoder(max_categories=9)

    try:
        if args.mode == "offline_mode":
            # Multi-file training and testing mode - USE SEPARATE PIPELINE

            if args.load_models:
                # Load pre-trained models for inference only
                training_folder = training_mode_config.get("training_folder")
                models_base_dir = training_mode_config.get("models_base_dir")

                if not training_folder:
                    print("[ERROR] No training_folder specified in config for model loading")
                    return

                print(f"[LOAD-MODELS] Loading pre-trained models...")
                print(f"  Training folder: {training_folder}")
                print(f"  Models base dir: {models_base_dir}")

                autoencoders, scalers, loaded_activity_embeddings, loaded_onehot_encoders = load_trained_models(
                    models_base_dir, training_folder, force_cpu=True
                )

                if not autoencoders:
                    print("[ERROR] No autoencoders loaded! Check model paths.")
                    return

                # Use loaded components if available, otherwise keep initialized ones
                if loaded_activity_embeddings:
                    activity_embeddings = loaded_activity_embeddings
                    print(
                        f"[LOAD] Using pre-trained activity embeddings: {len(activity_embeddings.embeddings)} activities")

                if loaded_onehot_encoders:
                    online_onehot_encoders.update(loaded_onehot_encoders)
                    print(f"[LOAD] Using pre-trained one-hot encoders: {len(loaded_onehot_encoders)} columns")

                print(f"[LOAD-MODELS] Successfully loaded:")
                print(f"  - {len(autoencoders)} autoencoders")
                print(f"  - {len(scalers)} scalers")
                print(f"  - Activity embeddings: {len(activity_embeddings.embeddings)} activities")
                print(f"  - One-hot encoders: {len(online_onehot_encoders)} columns")
                print(f"  - Global state (DFG): restored")

                all_seen_activities = set(autoencoders.keys())
                print(f"[CONFIG] Loaded models for {len(all_seen_activities)} activities")

            else:
                # Multi-file training mode without pre-trained models
                print(f"[ERROR] Multi-file training mode requires pre-trained models!")
                print(f"[ERROR] Please run the following steps:")
                print(f"[ERROR] 1. python feature_vector_file_generation.py")
                print(f"[ERROR] 2. python training_autoencoders.py")
                print(f"[ERROR] 3. python main.py --mode offline_mode --load-models")
                return

            # INFERENCE-ONLY TESTING PHASE FOR MULTI-FILE MODE
            test_folder = training_mode_config.get("test_folder")
            test_file_pattern = training_mode_config.get("test_file_pattern", "*.csv")

            if not test_folder:
                print("[ERROR] No test_folder specified in config for multi-file mode")
                return

            if not os.path.exists(test_folder):
                print(f"[ERROR] Test folder not found: {test_folder}")
                return

            test_files = get_files_from_folder(test_folder, test_file_pattern)

            if not test_files:
                print("[ERROR] No test files found in test folder")
                return

            # Get data columns from first test file
            first_test_file = test_files[0]
            all_columns = get_csv_columns(first_test_file)
            data_columns = [col for col in all_columns if col not in excluded_columns]

            # Initialize label refiner for overall results
            input_columns = all_columns + ["refined_activity"]
            label_refiner = LabelRefiner(args.output, input_columns)

            print(f"\n{'=' * 70}")
            print(f"STARTING INFERENCE ON {len(test_files)} TEST FILES")
            print(f"{'=' * 70}")

            # START TENSORBOARD EARLY with cluster_tracker output directory
            # Disable TensorBoard for offline mode to avoid package conflicts
            if autoencoder_params.get("enable_tensorboard", True) and args.mode != "offline_mode":
                try:
                    # Use the clean launcher instead
                    file_name = os.path.splitext(os.path.basename(args.input))[0] if args.input else "default"
                    models_dir = os.path.join(cluster_tracker.output_dir, f"trained_models_{file_name}")
                    tensorboard_log_dir = os.path.join(models_dir, "tensorboard_logs")

                    os.makedirs(tensorboard_log_dir, exist_ok=True)

                    tb_process = launch_tensorboard_clean_integration(tensorboard_log_dir, 6007)
                    if tb_process:
                        print(f"[TENSORBOARD] ✓ Successfully launched clean TensorBoard")
                        print(f"[TENSORBOARD] Log directory: {tensorboard_log_dir}")
                    else:
                        print(f"[TENSORBOARD] ✗ Failed to launch clean TensorBoard")
                        print(
                            f"[TENSORBOARD] Manual command: python launch_tensorboard_clean.py \"{tensorboard_log_dir}\" 6007")

                except Exception as e:
                    print(f"[WARNING] TensorBoard auto-launch failed: {e}")
                    print(f"[INFO] Manual command: python launch_tensorboard_clean.py \"{tensorboard_log_dir}\" 6007")
            else:
                print(f"[TENSORBOARD] Disabled for offline mode to avoid package conflicts")
            current_global_counter = 0

            # Process each test file for inference only
            for test_file in test_files:
                if os.path.exists(test_file):
                    test_result_dir, current_global_counter = process_test_file(
                        test_file, autoencoders, scalers, activity_embeddings,
                        online_onehot_encoders, data_columns, cluster_manager,
                        label_refiner, cluster_tracker.output_dir, cluster_tracker,
                        current_global_counter
                    )

                    # Create refined log for THIS specific test file immediately after processing
                    test_file_name = os.path.splitext(os.path.basename(test_file))[0]
                    event_vectors_path = os.path.join(test_result_dir, "event_feature_vectors.jsonl")
                    centroids_path = os.path.join(test_result_dir, "final_centroids.json")

                    # Individual test file refined log with cluster mapping
                    individual_refined_log_path = os.path.join(test_result_dir,
                                                               f"refined_log_with_clusters_{test_file_name}.csv")

                    if os.path.exists(event_vectors_path) and os.path.exists(centroids_path):
                        create_refined_log_with_cluster_mapping(
                            original_file_path=test_file,
                            event_vectors_path=event_vectors_path,
                            centroids_path=centroids_path,
                            output_refined_log_path=individual_refined_log_path
                        )
                        print(
                            f"[REFINED_LOG] Created individual refined log for {test_file_name}: {individual_refined_log_path}")
                    else:
                        print(f"[WARNING] Missing files for refined log creation for {test_file_name}")

                else:
                    print(f"[WARNING] Test file not found: {test_file}")

        else:
            # SINGLE FILE MODE 
            if not os.path.exists(args.input):
                print(f"[ERROR] Input file not found: {args.input}")
                return

            all_columns = get_csv_columns(args.input)
            data_columns = [col for col in all_columns if col not in excluded_columns]

            input_columns = all_columns + ["refined_activity"]
            label_refiner = LabelRefiner(args.output, input_columns)

            print(f"\n{'=' * 70}")
            print(f"STARTING SINGLE FILE PROCESSING")
            print(f"{'=' * 70}")

            # KEEP THE EXISTING process_event_stream FUNCTION FOR SINGLE FILE MODE
            all_seen_activities = process_event_stream(
                args.input, activity_embeddings, online_onehot_encoders,
                data_columns, cluster_manager, label_refiner, cluster_tracker.output_dir, cluster_tracker
            )

            input_file_name = os.path.splitext(os.path.basename(args.input))[0]
            centroids_path = os.path.join(cluster_tracker.output_dir, f"final_centroids.json")
            cluster_tracker.save_final_centroids(cluster_manager, centroids_path, input_file_name)

            # Create refined log with cluster mapping
            event_vectors_path = os.path.join(cluster_tracker.output_dir, "event_feature_vectors.jsonl")
            refined_log_path = os.path.join(cluster_tracker.output_dir,
                                            f"refined_log_with_clusters_{input_file_name}.csv")

            create_refined_log_with_cluster_mapping(
                original_file_path=args.input,
                event_vectors_path=event_vectors_path,
                centroids_path=centroids_path,
                output_refined_log_path=refined_log_path
            )

            # AUTO-RUN EVALUATION FOR ONLINE MODE
            if args.mode == "online_mode":
                print(f"\n{'=' * 70}")
                print(f"AUTOMATIC EVALUATION")
                print(f"{'=' * 70}")

                # Get evaluation config
                single_eval_config = evaluation_config.get("single_evaluation_config", {})
                ground_truth_path = single_eval_config.get("default_ground_truth_path")
                activity_of_interest = single_eval_config.get("default_activity")

                # Create evaluation output directory structure
                # evaluation_results/duplimend/tracking_XXXXXX/
                eval_base_dir = os.path.dirname(single_eval_config.get("default_output_dir", "./evaluation_results"))
                tracking_folder_name = os.path.basename(cluster_tracker.output_dir)
                eval_output_dir = os.path.join(eval_base_dir, "duplimend", tracking_folder_name)

                # Check if ground truth exists
                if ground_truth_path and os.path.exists(ground_truth_path):
                    try:
                        print(f"[EVALUATION] Running automatic evaluation...")
                        print(f"[EVALUATION] Ground truth: {ground_truth_path}")
                        print(f"[EVALUATION] Activity: {activity_of_interest}")
                        print(f"[EVALUATION] Output directory: {eval_output_dir}")

                        eval_results = run_evaluation(
                            tracking_dir_path=cluster_tracker.output_dir,
                            ground_truth_file_path=ground_truth_path,
                            activity=activity_of_interest,
                            output_dir=eval_output_dir,
                            event_id_col=single_eval_config.get("event_id_column"),
                            control_flow_col=single_eval_config.get("control_flow_column"),
                            control_flow_gt_col=single_eval_config.get("control_flow_column_ground_truth"),
                            case_id_col=single_eval_config.get("case_id_column")
                        )

                        if eval_results:
                            print(f"[EVALUATION] ✓ Evaluation completed successfully")
                            print(f"[EVALUATION] Results saved to: {eval_output_dir}")
                        else:
                            print(f"[EVALUATION] ✗ Evaluation completed with warnings")

                    except Exception as eval_error:
                        print(f"[EVALUATION] ✗ Evaluation failed: {eval_error}")
                        import traceback
                        traceback.print_exc()
                else:
                    if not ground_truth_path:
                        print(f"[EVALUATION] Skipping: No ground truth path configured")
                    else:
                        print(f"[EVALUATION] Skipping: Ground truth file not found at {ground_truth_path}")

    except Exception as e:
        print(f"[ERROR] Fatal error in processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Generate final reports
        cluster_tracker.generate_summary_report()

        print(f"\n[COMPLETED] Processing finished")
        print(f"[COMPLETED] Cluster tracking data: {cluster_tracker.output_dir}")

        # Print final cluster summary
        print("\n=== Final Cluster Summary ===")
        for activity in all_seen_activities if 'all_seen_activities' in locals() else []:
            macro_summary = cluster_manager.get_macro_cluster_summary(activity)
            print(f"Activity '{activity}': {len(macro_summary)} clusters")
            for macro in macro_summary:
                print(f"  - Cluster {macro['macro_id']}: {macro['samples']} samples")


if __name__ == "__main__":
    main()
