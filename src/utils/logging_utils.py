import os
import logging

# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "traceability_log.txt")

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'  #  Overwrites log file each run (change to 'a' to append)
)

def log_traceability(action, label, details):
    """
    Logs traceability and auditability details.
    """
    logging.debug(f"{action.upper()} - {label}: {details}")  #  Ensure using debug level
