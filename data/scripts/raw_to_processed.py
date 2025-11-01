# Base imports
import json
import os

# Third-party imports
import pandas as pd
from fitparse import FitFile
from tqdm import tqdm

# Local imports
from data import config
from utils import log_utils


def fit_to_records(fit_path: str, activity_id: str) -> pd.DataFrame:
    """
    Convert a .fit file to a pandas DataFrame of activity records.

    Args:
        fit_path (str): Path to the .fit file.
        activity_id (str): The activity ID to associate with each record.

    Returns:
        pd.DataFrame: DataFrame containing all records from the .fit file, with activity_id column.
    """
    fitfile = FitFile(fit_path)
    records = []

    for record in fitfile.get_messages("record"):
        data = {d.name: d.value for d in record}  # type: ignore
        data["activity_id"] = activity_id
        records.append(data)

    return pd.DataFrame(records)


def json_to_records(json_path: str) -> pd.DataFrame:
    """
    Convert a .json metadata file to a pandas DataFrame.

    Args:
        json_path (str): Path to the .json file.

    Returns:
        pd.DataFrame: DataFrame containing the normalized JSON data.
                      Returns empty DataFrame on error.
    """
    try:
        with open(json_path, "r") as f:
            record = json.load(f)
        return pd.json_normalize(record)
    except ValueError as e:
        config.logger.error(f"Error reading {json_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def load_existing_ids(path: str, id_column: str = "activity_id") -> set:
    """
    Load a set of already-processed activity IDs from a parquet file.

    Args:
        path (str): Path to the parquet file.
        id_column (str): Name of the column containing activity IDs.

    Returns:
        set: Set of activity IDs (as strings) found in the file, or empty set if file missing or error.
    """
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            ids = set(df[id_column].astype(str).unique())
            return ids
        except Exception as e:
            config.logger.error(f"Failed to read {path}: {e}")
    return set()


def run_meta_pipeline(activity_dirs: set[str]) -> None:
    """
    Process new activity metadata (.json) files and append them to the processed meta parquet file.

    Args:
        activity_dirs (set[str]): Set of activity directory names to process.
    """
    if not activity_dirs:
        config.logger.info("No new meta activity directories to process.")
        return

    json_files = []
    for activity_dir in activity_dirs:
        activity_path = os.path.join(config.RAW_DIR, activity_dir)
        if not os.path.isdir(activity_path):
            continue
        for fname in os.listdir(activity_path):
            if fname.endswith(".json"):
                json_files.append((os.path.join(activity_path, fname), activity_dir))

    if not json_files:
        config.logger.info("No new .json files found for given activity directories.")
        return

    tqdm_logger = log_utils.TqdmLogger(config.logger)
    dfs = []
    for json_path, activity_dir in tqdm(
        json_files, desc="Processing .json files", file=tqdm_logger
    ):
        df = json_to_records(json_path)
        if not df.empty:
            if "activity_id" not in df.columns:
                df["activity_id"] = activity_dir
            dfs.append(df)

    if dfs:
        cols = [
            "activity_id",
            "activityName",
            "startTimeLocal",
            "activityType.typeKey",
            "distance",
            "duration",
            "averageSpeed",
            "maxSpeed",
            "averageHR",
            "maxHR",
            "elevationGain",
            "elevationLoss",
            "averageCadence",
            "startTimeGMT",
        ]
        new_df = pd.concat(dfs, ignore_index=True)[
            lambda x: [c for c in cols if c in x.columns]
        ]  # do not fail if col missing

        # Load existing data if available
        if os.path.exists(config.PROCESSED_META_FILE):
            existing_df = pd.read_parquet(config.PROCESSED_META_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        os.makedirs(os.path.dirname(config.PROCESSED_META_FILE), exist_ok=True)
        combined_df.to_parquet(config.PROCESSED_META_FILE)
        config.logger.info(
            f"Saved {len(new_df)} new rows ({len(combined_df)} total) to {config.PROCESSED_META_FILE}"
        )
    else:
        config.logger.warning("No valid .json data found to save.")


def run_fit_pipeline(activity_dirs: set[str]) -> None:
    """
    Process new activity .fit files and append them to the processed fit parquet file.

    Args:
        activity_dirs (set[str]): Set of activity directory names to process.
    """
    if not activity_dirs:
        config.logger.info("No new fit activity directories to process.")
        return

    fit_files = []
    for activity_dir in activity_dirs:
        activity_path = os.path.join(config.RAW_DIR, activity_dir)
        if not os.path.isdir(activity_path):
            continue
        for fname in os.listdir(activity_path):
            if fname.endswith(".fit"):
                fit_files.append((os.path.join(activity_path, fname), activity_dir))

    if not fit_files:
        config.logger.info("No new .fit files found for given activity directories.")
        return

    tqdm_logger = log_utils.TqdmLogger(config.logger)
    dfs = []
    for fit_path, activity_dir in tqdm(
        fit_files, desc="Processing .fit files", file=tqdm_logger
    ):
        try:
            df = fit_to_records(fit_path, activity_id=activity_dir)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            config.logger.error(f"Error reading {fit_path}: {e}")

    if dfs:
        cols = [
            "altitude",
            "cadence",
            "distance",
            "heart_rate",
            "speed",
            "timestamp",
            "activity_id",
        ]
        new_df = pd.concat(dfs, ignore_index=True)[
            lambda x: [c for c in cols if c in x.columns]
        ]  # do not fail if col missing

        # Load existing data if available
        if os.path.exists(config.PROCESSED_ACTIVTY_FILE):
            existing_df = pd.read_parquet(config.PROCESSED_ACTIVTY_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        os.makedirs(os.path.dirname(config.PROCESSED_ACTIVTY_FILE), exist_ok=True)
        combined_df.to_parquet(config.PROCESSED_ACTIVTY_FILE)
        config.logger.info(
            f"Saved {len(new_df)} new rows ({len(combined_df)} total) to {config.PROCESSED_ACTIVTY_FILE}"
        )
    else:
        config.logger.warning("No valid .fit data found to save.")


def process_garmin_files():
    """
    Main entry point for processing Garmin Connect raw data into processed parquet files.

    - Detects all activity directories in the raw data folder.
    - Loads already-processed activity IDs for both meta and fit data.
    - Determines which activities are new and need processing.
    - Runs the meta and fit processing pipelines.
    """
    # Get all activity directories (folder names)
    activity_dirs = {
        d
        for d in os.listdir(config.RAW_DIR)
        if os.path.isdir(os.path.join(config.RAW_DIR, d))
    }

    # Load already processed activity IDs from parquet outputs
    existing_meta_ids = load_existing_ids(config.PROCESSED_META_FILE)
    existing_fit_ids = load_existing_ids(config.PROCESSED_ACTIVTY_FILE)

    # Filter out activity directories that are already processed
    new_meta_dirs = activity_dirs - existing_meta_ids
    new_fit_dirs = activity_dirs - existing_fit_ids

    config.logger.info(f"Found {len(activity_dirs)} total activity directories")
    config.logger.info(
        f"Skipping {len(existing_meta_ids)} meta and {len(existing_fit_ids)} fit already processed"
    )
    config.logger.info(
        f"Processing {len(new_meta_dirs)} new meta and {len(new_fit_dirs)} new fit activities"
    )

    run_meta_pipeline(new_meta_dirs)
    run_fit_pipeline(new_fit_dirs)


def main():
    config.logger.info("***Starting Garmin Connect data processing...***")
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    process_garmin_files()
    config.logger.info("------------------------------------------------\n")
