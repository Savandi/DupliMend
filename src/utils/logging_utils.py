import os
import logging

# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "traceability_log.txt")

# Ensure log file exists before writing logs
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        f.write("")  # Create an empty file

# Configure logging to capture DEBUG-level logs and send to both file & console
logging.basicConfig(
    level=logging.DEBUG,  # ✅ Capture all logs including DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # ✅ Logs to file
        logging.StreamHandler()  # ✅ Logs to console (real-time debugging)
    ]
)

def log_traceability(action, label, details):
    """
    Logs traceability and auditability details.
    """
    logging.info(f"{action.upper()} - {label}: {details}")
