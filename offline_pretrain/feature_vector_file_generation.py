import os
import json
import glob
import pickle
from config.config import *
from src.duplimend_framework.utils.control_flow_feature_utils import (
    forget_old_cases, update_case_sequence, extract_control_flow_features
)
from src.duplimend_framework.utils.online_activity_embeddings import DynamicActivityEmbeddings
from src.duplimend_framework.utils.online_onehot_encoder import OnlineOneHotEncoder
from src.duplimend_framework.feature_vector_builder import FeatureVectorBuilder
from src.duplimend_framework.utils.global_state import (
    directly_follows_graph,
    per_case_directly_follows_graph,
    case_activity_sequences
)
from src.duplimend_framework.utils.config_saver import save_config_to_output
import csv
import pandas as pd


def stream_csv_row_by_row(file_path, encoding='utf-8'):
    """Stream CSV file row by row with robust error handling."""
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding, newline='', errors='replace') as file:
                csv_reader = csv.DictReader(file)
                for row_dict in csv_reader:
                    if not any(row_dict.values()):
                        continue

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
            return
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    raise Exception(f"Could not read {file_path} with any encoding")


def get_csv_columns(file_path, encoding='ISO-8859-1'):
    """Get CSV column names."""
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader.fieldnames)


def get_files_from_folder(folder_path, file_pattern="*.csv"):
    """Get all files matching pattern from a folder."""
    if not os.path.exists(folder_path):
        return []

    pattern = os.path.join(folder_path, file_pattern)
    files = glob.glob(pattern)
    files.sort()
    return files


def reset_global_state():
    """Reset all global state for fresh runs."""
    from src.duplimend_framework.utils.global_state import (
        event_embedding_mapping,
        per_case_directly_follows_graph,
        case_activity_sequences,
        activity_positions,
        activity_instance_counts,
        activity_hash_mapping,
        self_loop_cases,
        case_context_freeze,
        case_last_seen_global,
        last_activity_per_case
    )

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

    if hasattr(directly_follows_graph, 'reset'):
        directly_follows_graph.reset()
    else:
        directly_follows_graph.__init__()


def validate_csv_file(file_path):
    """Validate CSV file before processing."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False

        with open(file_path, 'rb') as f:
            if b'\x00' in f.read(1024):
                return False

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            header = f.readline()
            if not header.strip():
                return False

        return True
    except Exception:
        return False


def generate_feature_vector_files(training_folder, file_pattern, output_dir):
    """
    Generate feature vector files for each activity label from training folder.
    Builds control flow graphs sequentially across all training files.
    """
    reset_global_state()
    training_files = get_files_from_folder(training_folder, file_pattern)

    if not training_files:
        return None

    folder_name = os.path.basename(training_folder.rstrip('/\\'))
    feature_vectors_dir = os.path.join(output_dir, f"feature_vectors_{folder_name}")
    os.makedirs(feature_vectors_dir, exist_ok=True)

    activity_embeddings = DynamicActivityEmbeddings(embedding_dim=embedding_dim, seed=42)
    online_onehot_encoders = {}

    categorical_columns = categorical_feature_params.get("categorical_columns", [])
    for col in categorical_columns:
        if col and col not in online_onehot_encoders and col not in excluded_columns:
            online_onehot_encoders[col] = OnlineOneHotEncoder(max_categories)

    if (resource_column and
            resource_column not in online_onehot_encoders and
            resource_column not in excluded_columns):
        online_onehot_encoders[resource_column] = OnlineOneHotEncoder(max_categories=9)

    first_file = training_files[0]
    all_columns = get_csv_columns(first_file)
    data_columns = [col for col in all_columns if col not in excluded_columns]

    global_event_counter = 0
    all_seen_activities = set()
    activity_feature_files = {}
    activity_event_counters = {}

    try:
        for file_idx, file_path in enumerate(training_files):
            if not validate_csv_file(file_path):
                continue

            file_event_count = 0
            try:
                for event_idx, event_dict in enumerate(stream_csv_row_by_row(file_path)):
                    try:
                        event_id = event_dict.get(event_id_column)
                        if not event_id:
                            event_id = f"event_{global_event_counter}"

                        global_event_counter += 1
                        file_event_count += 1
                        case_id = event_dict[case_id_column]
                        activity_label = event_dict[control_flow_column]
                        all_seen_activities.add(activity_label)
                        activity_embeddings.update(activity_label)

                        if activity_label not in activity_event_counters:
                            activity_event_counters[activity_label] = 0

                        if use_control_flow_features:
                            timestamp_value = event_dict[timestamp_column]
                            if isinstance(timestamp_value, pd.Timestamp):
                                timestamp_numeric = timestamp_value.timestamp()
                            elif isinstance(timestamp_value, (int, float)):
                                timestamp_numeric = float(timestamp_value)
                            else:
                                try:
                                    timestamp_numeric = float(timestamp_value)
                                except (ValueError, TypeError):
                                    timestamp_numeric = global_event_counter

                            update_case_sequence(case_id, activity_label, timestamp_numeric, global_event_counter)

                            if case_id not in per_case_directly_follows_graph:
                                per_case_directly_follows_graph[case_id] = defaultdict(int)

                            if case_id in case_activity_sequences and len(case_activity_sequences[case_id]) > 1:
                                prev_activity = case_activity_sequences[case_id][-2]['activity']
                                directly_follows_graph.add_transition(case_id, prev_activity, activity_label,
                                                                      global_event_counter)
                                per_case_directly_follows_graph[case_id][(prev_activity, activity_label)] += 1

                            control_flow_features = extract_control_flow_features(case_id, activity_label)
                        else:
                            control_flow_features = None

                        for col in data_columns:
                            if col in online_onehot_encoders:
                                value = event_dict.get(col, None)
                                online_onehot_encoders[col].encode(value)

                        feature_vector = FeatureVectorBuilder.build(
                            event_dict, data_columns, control_flow_features, online_onehot_encoders,
                            activity_embeddings, excluded_columns, use_control_flow_features
                        )
                        feature_vector = np.array(feature_vector, dtype=np.float32)
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0)

                        if activity_label not in activity_feature_files:
                            feature_file_path = os.path.join(feature_vectors_dir, f"features_{activity_label}.jsonl")
                            activity_feature_files[activity_label] = open(feature_file_path, 'w')

                        activity_event_counters[activity_label] += 1

                        feature_data = {
                            "event_id": activity_event_counters[activity_label],
                            "activity": activity_label,
                            "feature_vector": feature_vector.tolist()
                        }

                        activity_feature_files[activity_label].write(json.dumps(feature_data) + "\n")

                        if global_event_counter % forgetting_params["decay_after_events"] == 0:
                            forget_old_cases(global_event_counter)
                            directly_follows_graph.apply_forgetting(global_event_counter)

                    except Exception:
                        continue

            except Exception:
                continue

    finally:
        for activity_label, file_handle in activity_feature_files.items():
            if file_handle:
                file_handle.close()

    embeddings_path = os.path.join(feature_vectors_dir, "activity_embeddings.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(activity_embeddings, f)

    encoders_path = os.path.join(feature_vectors_dir, "onehot_encoders.pkl")
    with open(encoders_path, 'wb') as f:
        pickle.dump(online_onehot_encoders, f)

    encoder_stats = {}
    for col, encoder in online_onehot_encoders.items():
        encoder_stats[col] = {
            "vocabulary_size": encoder.next_index,
            "max_categories": encoder.max_categories,
            "sample_values": list(encoder.value_to_index.keys())[:5]
        }

    if use_control_flow_features:
        global_state_path = os.path.join(feature_vectors_dir, "global_state.pkl")
        global_state = {
            "directly_follows_graph": directly_follows_graph,
            "per_case_directly_follows_graph": dict(per_case_directly_follows_graph),
            "case_activity_sequences": dict(case_activity_sequences)
        }
        with open(global_state_path, 'wb') as f:
            pickle.dump(global_state, f)

    metadata = {
        "total_events": global_event_counter,
        "total_files": len(training_files),
        "activities": sorted(list(all_seen_activities)),
        "activity_event_counts": activity_event_counters,
        "feature_vector_dim": len(feature_vector) if 'feature_vector' in locals() else None,
        "control_flow_enabled": use_control_flow_features,
        "training_files": [os.path.basename(f) for f in training_files],
        "generation_timestamp": pd.Timestamp.now().isoformat(),
        "data_columns": data_columns,
        "embedding_dim": embedding_dim,
        "categorical_columns": categorical_columns,
        "activity_embeddings_count": len(activity_embeddings.embeddings),
        "onehot_encoders_count": len(online_onehot_encoders),
        "encoder_stats": encoder_stats
    }

    metadata_path = os.path.join(feature_vectors_dir, "generation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return feature_vectors_dir, all_seen_activities, global_event_counter


def main():
    """Main function for feature vector file generation."""
    training_folder = training_mode_config.get("training_folder")
    training_file_pattern = training_mode_config.get("training_file_pattern", "*.csv")
    output_dir = training_mode_config.get("feature_vectors_base_dir", "output")

    if not training_folder:
        return

    additional_info = {
        "training_folder": training_folder,
        "training_file_pattern": training_file_pattern,
        "output_dir": output_dir
    }
    save_config_to_output(output_dir, "feature_generation", additional_info)

    if not os.path.exists(training_folder):
        return

    os.makedirs(output_dir, exist_ok=True)

    result = generate_feature_vector_files(
        training_folder, training_file_pattern, output_dir
    )


if __name__ == "__main__":
    main()
