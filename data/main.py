from data import config
from data.scripts import get_raw_data, processed_to_ml_ready, raw_to_processed

if __name__ == "__main__":
    # Allow for missing .env or no credentials
    # Then just work with data already in GitHub
    if (config.GARMIN_EMAIL is not None) and (config.GARMIN_PASSWORD is not None):
        get_raw_data.main()
    else:
        config.logger.warning(
            "Skipping Garmin Connect data extraction; missing credentials"
        )
        config.logger.info("------------------------------------------------\n")

    raw_to_processed.main()
    processed_to_ml_ready.main()
