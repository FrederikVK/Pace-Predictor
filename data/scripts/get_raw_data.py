# Base imports
import datetime
import json
import os
import zipfile

# Third-party imports
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
)

# Local imports
from data import config


def read_last_fetch() -> datetime.datetime:
    """
    Reads the last fetch timestamp from a file.

    Returns:
        datetime.datetime: The last fetch timestamp.
        If the file does not exist, returns a default date of January 1, 2000.
    """
    if not os.path.exists(config.LAST_FETCH_FILE):
        # Return some early default date if no file
        return datetime.datetime(2000, 1, 1)
    with open(config.LAST_FETCH_FILE, "r") as f:
        ts = f.read().strip()
    return datetime.datetime.strptime(ts, config.DATE_FORMAT)


def write_last_fetch(dt) -> None:
    """
    Writes the provided datetime as the last fetch timestamp to a file.

    Args:
        dt (datetime.datetime): The datetime to write as the last fetch timestamp.
    """
    with open(config.LAST_FETCH_FILE, "w") as f:
        f.write(dt.strftime(config.DATE_FORMAT))


def download_and_extract_activity(client: Garmin, activity: dict) -> None:
    """
    Downloads and extracts the activity data from Garmin Connect.

    Downloads the original activity zip file, extracts it into the raw directory,
    and saves the metadata JSON inside the same activity folder.

    Args:
        client (Garmin): An authenticated Garmin client.
        activity (dict): The activity metadata dictionary.
    """
    activity_id = activity["activityId"]
    data = client.download_activity(
        activity_id, dl_fmt=client.ActivityDownloadFormat.ORIGINAL
    )

    # Create activity folder (if not exists)
    activity_dir = os.path.join(config.RAW_DIR, str(activity_id))
    os.makedirs(activity_dir, exist_ok=True)

    zip_path = os.path.join(activity_dir, f"{activity_id}_original.zip")
    with open(zip_path, "wb") as f:
        f.write(data)

    # Extract zip files into the same folder (you already do this)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(activity_dir)

    # Save the metadata JSON inside the same activity folder
    metadata_path = os.path.join(activity_dir, f"{activity_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(activity, f, indent=2)


def extract_activites_from_garmin():
    """
    Main function to extract new activities from Garmin Connect.

    - Reads the last fetch timestamp.
    - Authenticates with Garmin Connect.
    - Fetches new activities since the last fetch.
    - Downloads and extracts each new activity.
    - Updates the last fetch timestamp after each activity.

    """
    since = read_last_fetch()
    config.logger.info(f"Last fetch was on: {since.strftime(config.DATE_FORMAT)}")

    try:
        client = Garmin(config.GARMIN_EMAIL, config.GARMIN_PASSWORD)
        client.login()
    except (GarminConnectAuthenticationError, GarminConnectConnectionError) as err:
        config.logger.error(f"Garmin Connect error: {err}")
        return

    all_new_activities = []
    start = 0
    max_per_page = 100

    config.logger.info("Fetching activities from Garmin Connect...")
    while True:
        batch = client.get_activities(start, max_per_page)
        if not batch:
            break

        # Filter only new activities
        new_activities = [a for a in batch if datetime.datetime.strptime(a["startTimeGMT"], config.DATE_FORMAT) > since]  # type: ignore
        if not new_activities:
            break  # no more new activities

        all_new_activities.extend(new_activities)
        start += max_per_page

    # Sort oldest to newest
    all_new_activities.sort(key=lambda a: a["startTimeGMT"])

    # Process and write after each
    for activity in all_new_activities:
        config.logger.info(
            f"""Downloading activity {activity["activityId"]} from {activity["startTimeGMT"]}"""
        )
        download_and_extract_activity(client, activity)

        act_time = datetime.datetime.strptime(
            activity["startTimeGMT"], config.DATE_FORMAT
        )
        write_last_fetch(act_time)

    config.logger.info(f"Downloaded {len(all_new_activities)} new activities.")
    try:
        config.logger.info(
            f"Last fetch timestamp updated to: {act_time.strftime(config.DATE_FORMAT)}"
        )
    except:
        config.logger.warning(
            f"Last fetch timestamp not updated, still: {since.strftime(config.DATE_FORMAT)}",
        )


def main():
    config.logger.info("***Starting Garmin Connect data extraction...***")
    os.makedirs(config.RAW_DIR, exist_ok=True)
    extract_activites_from_garmin()
    config.logger.info("------------------------------------------------\n")
