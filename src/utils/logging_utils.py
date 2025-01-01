import os
import logging

# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "traceability_log.txt")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_traceability(action, label, details):
    """
    Logs traceability and auditability details.
    """
    logging.info(f"{action.upper()} - {label}: {details}")