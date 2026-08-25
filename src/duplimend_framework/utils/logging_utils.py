import os
import logging

log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "traceability_log.txt")

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)

def log_traceability(action, label, details):
    """
    Logs traceability and auditability details.
    """
    logging.debug(f"{action.upper()} - {label}: {details}")

log_tags = {
    "dbstream_update": "DBSTREAM_UPDATE",
    "new_cluster": "NEW_CLUSTER",
    "cluster_update": "CLUSTER_UPDATE",
    "variability_and_thresholds": "VARIABILITY_AND_THRESHOLDS",
    "split_merge_result": "SPLIT_MERGE_RESULT",
    "cluster_pruning": "CLUSTER_PRUNING"
}