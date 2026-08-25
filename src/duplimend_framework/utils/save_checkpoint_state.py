def save_checkpoint_state(checkpoint_event_count, total_processed, cluster_manager, cluster_tracker, 
                         event_vectors_buffer, output_dir, original_file_path, file_name_prefix=""):
    """
    Save clustering state at a specific checkpoint during stream processing
    
    Args:
        checkpoint_event_count: Current event count for this checkpoint
        total_processed: Total events processed so far
        cluster_manager: Current cluster manager state
        cluster_tracker: Cluster evolution tracker
        event_vectors_buffer: List of event vectors processed so far
        output_dir: Output directory for checkpoint files
        original_file_path: Path to original file being processed
        file_name_prefix: Prefix for checkpoint files (e.g., "test_file_name" for multi-file mode)
    """
    import json
    import os
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_{checkpoint_event_count}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    event_vectors_path = os.path.join(checkpoint_dir, f"event_feature_vectors_checkpoint_{checkpoint_event_count}.jsonl")
    with open(event_vectors_path, "w") as f:
        for event_vector_data in event_vectors_buffer:
            f.write(json.dumps(event_vector_data) + "\n")
    
    centroids_path = os.path.join(checkpoint_dir, f"centroids_checkpoint_{checkpoint_event_count}.json")
    checkpoint_name = f"{file_name_prefix}_checkpoint_{checkpoint_event_count}" if file_name_prefix else f"checkpoint_{checkpoint_event_count}"
    cluster_tracker.save_final_centroids(cluster_manager, centroids_path, checkpoint_name)
    
    refined_log_path = os.path.join(checkpoint_dir, f"refined_log_with_clusters_checkpoint_{checkpoint_event_count}.csv")
    
    create_refined_log_with_cluster_mapping(
        original_file_path=original_file_path,
        event_vectors_path=event_vectors_path,
        centroids_path=centroids_path,
        output_refined_log_path=refined_log_path
    )

    return checkpoint_dir