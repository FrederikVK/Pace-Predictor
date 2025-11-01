# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv(override=True)

import os

# Local Imports
from utils import log_utils

# Logging
LOG_FILE = "logs/data.log"
logger = log_utils.get_logger(
    name="data_logger",
    logfile=LOG_FILE,
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LAST_FETCH_FILE = "data/local_db/meta/last_fetch.txt"

# Raw data
RAW_DIR = "data/local_db/raw"

# Processed data
PROCESSED_DIR = "data/local_db/processed"
PROCESSED_META_FILE = f"{PROCESSED_DIR}/running_meta.parquet"
PROCESSED_ACTIVTY_FILE = f"{PROCESSED_DIR}/running.parquet"

# ML Ready data
ML_READY_DIR = "data/local_db/ml_ready"
ML_READY_META_FILE = f"{ML_READY_DIR}/running_meta.parquet"
ML_READY_ACTIVITY_FILE = f"{ML_READY_DIR}/running.parquet"
ML_READY_TP_METRICS_FILE = f"{ML_READY_DIR}/running_to_metrics.parquet"
ML_READY_RESAMPLED_FILE = f"{ML_READY_DIR}/running_resampled.parquet"

# Garmin credentials from env
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

# Personal (physiological) variables
FTP = 15 / 3.6  # functional (lactate) threshold pace, 1 hour, 15km/h in m/s
