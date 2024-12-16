# logging_utils.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="traceability_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_traceability(action, activity_label, details):
    """
    Log traceability and auditability details.
    """
    timestamp = datetime.now().isoformat()
    logging.info(f"{action.upper()} - {activity_label}: {details}")
