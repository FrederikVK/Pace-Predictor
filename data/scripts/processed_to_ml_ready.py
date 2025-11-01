# Base imports
import os

# Third-party imports
import pandas as pd
from tqdm import tqdm

# Local imports
from data import config
from utils import converters, log_utils


def prep_meta(meta) -> pd.DataFrame:
    """
    Prepare and transform the activity metadata for machine learning.

    - Loads processed metadata parquet file.
    - Filters for running activities.
    - Adds derived columns (year, month, week, distance in km, speed in km/h, pace, etc.).
    - Formats pace as both timedelta and string.

    Returns:
        pd.DataFrame: Transformed metadata for running activities.
    """
    cols = [
        "activity_id",
        "startTimeLocal",
        "startTimeGMT",
        "distance",
        "duration",
        "elevationGain",
        "elevationLoss",
        "averageHR",
        "maxHR",
        "averageSpeed",
        "maxSpeed",
    ]
    df = (
        meta.astype({"activity_id": "int"})
        .rename(columns={"activityType.typeKey": "activity_type"})
        .query("activity_type in ['running', 'track_running']")[cols]
        .astype(
            {
                "startTimeLocal": "datetime64[ns]",
                "startTimeGMT": "datetime64[ns]",
            }
        )
        .sort_values(["startTimeGMT"])  # type: ignore
        .assign(year=lambda x: x["startTimeGMT"].dt.year)
        .assign(month=lambda x: x["startTimeGMT"].dt.month)
        .assign(week=lambda x: x["startTimeGMT"].dt.isocalendar().week)
        .assign(distance_km=lambda x: x["distance"] / 1000.0)
        .assign(average_km_pr_hour=lambda x: converters.km_pr_hour(x["averageSpeed"]))
        .assign(max_km_pr_hour=lambda x: converters.km_pr_hour(x["maxSpeed"]))
        .assign(
            average_pace_min_sec=lambda x: converters.speed_to_pace(x["averageSpeed"])
        )
        .assign(max_pace_min_sec=lambda x: converters.speed_to_pace(x["maxSpeed"]))
        .assign(
            average_pace_min_sec_str=lambda x: converters.pace_to_str_series(
                x["average_pace_min_sec"]
            )
        )
        .assign(
            max_pace_min_sec_str=lambda x: converters.pace_to_str_series(
                x["max_pace_min_sec"]
            )
        )
    )
    return df


def prep_activities(activities: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """
    Prepare and transform the activity records for machine learning.

    - Loads processed activity records parquet file.
    - Filters for the given activity IDs.
    - Doubles cadence (to convert from single-leg to total steps).
    - Adds speed in km/h.

    Args:
        ids (list[int]): List of activity IDs to include.

    Returns:
        pd.DataFrame: Transformed activity records for the given IDs.
    """
    cols = [
        "activity_id",
        "timestamp",
        "altitude",
        "cadence",
        "distance",
        "speed",
        "heart_rate",
    ]
    df = (
        activities[cols]
        .astype(
            {
                "activity_id": int,
                "timestamp": "datetime64[ns]",
            }
        )
        .query("activity_id in @ids")
        .assign(cadence=lambda x: x["cadence"] * 2)
        .sort_values(["activity_id", "timestamp"])
        .assign(km_pr_hour=lambda x: converters.km_pr_hour(x["speed"]))
    )
    return df


def resample_and_smooth_activites(df: pd.DataFrame):

    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.fillna({"altitude": 0})  # fill out missing altitude
            .dropna(subset=["speed"])  # drop missing speed, will be interpolated later
            .query("speed*3.6 <= 25")  # filter out more than 25km/h
        )

    def resample_and_smooth(
        df, signals=["distance", "speed", "km_pr_hour", "heart_rate", "altitude"]
    ):
        id = df["activity_id"].unique()[0]

        return (
            df.sort_values(["timestamp"])
            .assign(
                segment=lambda x: x["timestamp"]
                .diff()
                .dt.total_seconds()
                .gt(30)
                .cumsum()
            )
            .set_index("timestamp")
            .groupby(["segment"])[signals]
            .apply(lambda x: x.resample("1s").interpolate("time"))
            .reset_index()
            .set_index("timestamp")
            .groupby(["segment"])[signals]
            .rolling("30s")
            .mean()
            .reset_index()
            .assign(activity_id=id)
        )

    res = []
    df = clean_data(df)
    tqdm_logger = log_utils.TqdmLogger(config.logger)
    for id in tqdm(sorted(df["activity_id"].unique()), file=tqdm_logger):
        res.append(resample_and_smooth(df.query("activity_id == @id")))
    res = pd.concat(res)
    return res


def tp_rTSS(
    df: pd.DataFrame,
    FTP: float = config.FTP,
):
    """https://www.trainingpeaks.com/learn/articles/running-training-stress-score-rtss-explained/"""

    df = (
        df.copy()
        # prep
        .sort_values(["activity_id", "timestamp"]).set_index("timestamp")
        # pace or gap (graded pace)
        .assign(gap_speed=lambda x: x["speed"])  # *x["GAP_factor"])
    )

    tss = (
        df
        # rolling window 30 sec
        .groupby("activity_id")["gap_speed"]
        .rolling(window=30, min_periods=1)
        .mean()
        .to_frame("speed_rolling")
        # raised to 4th power
        .assign(speed_rolling_4th=lambda x: x["speed_rolling"] ** 4)
        # compute normalized speed
        .reset_index()
        .groupby("activity_id")["speed_rolling_4th"]
        .mean()
        .to_frame("speed_rolling_4th_mean")
        .assign(normalized_speed=lambda x: x["speed_rolling_4th_mean"] ** 0.25)
        # Intensity Factor = Normalized Graded Pace / FTP
        .assign(IF=lambda x: x["normalized_speed"] / FTP)
        # Add necessary columns back
        .join(
            df.reset_index()
            .groupby("activity_id")
            .agg(
                duration_seconds=("gap_speed", "size"),
                avg_speed=("gap_speed", "mean"),
                avg_heart_rate=("heart_rate", "mean"),
                distance=("distance", "max"),
                timestamp=("timestamp", "min"),
            )
        )
        # run TSS
        .assign(
            rTSS=lambda x: ((x["duration_seconds"] * (x["IF"] ** 2) * 100) / (3600))
        )
        # Efficiency Factor
        .assign(EF=lambda x: x["normalized_speed"] / x["avg_heart_rate"])
        # Variability Index
        .assign(VI=lambda x: x["normalized_speed"] / x["avg_speed"])
        # final prep
        .reset_index()
        .assign(distance_km=lambda x: x["distance"] / 1000)
        .assign(avg_speed_km_pr_hour=lambda x: x["avg_speed"] * 3.6)
    )

    return tss


def make_processed_data_ml_ready():
    """
    Main entry point for preparing processed data for machine learning.

    - Prepares and saves metadata and activity records in ML-ready format.
    """
    config.logger.info("* Meta data *")
    raw_meta = pd.read_parquet(config.PROCESSED_META_FILE)
    meta = prep_meta(raw_meta)
    meta.to_parquet(config.ML_READY_META_FILE, index=False)
    config.logger.info(
        f"Saved {len(meta)} running activities to {config.ML_READY_META_FILE}"
    )

    config.logger.info("* Detailed activity data *")
    raw_activities = pd.read_parquet(config.PROCESSED_ACTIVTY_FILE)
    activities = prep_activities(raw_activities, ids=meta["activity_id"].tolist())
    activities.to_parquet(config.ML_READY_ACTIVITY_FILE, index=False)
    config.logger.info(
        f"""Saved {len(activities)} records for {activities["activity_id"].nunique()} running activities to {config.ML_READY_ACTIVITY_FILE}"""
    )

    config.logger.info("* Resampling and smoothing activity data *")
    resampled_activites = resample_and_smooth_activites(activities)
    resampled_activites.to_parquet(config.ML_READY_RESAMPLED_FILE, index=False)
    config.logger.info(
        f"""Saved {len(resampled_activites)} records for {resampled_activites["activity_id"].nunique()} resampled running activities to {config.ML_READY_RESAMPLED_FILE}"""
    )

    config.logger.info("* TrainingPeaks metrics *")
    tp_metrics = tp_rTSS(resampled_activites)
    tp_metrics.to_parquet(config.ML_READY_TP_METRICS_FILE, index=False)
    config.logger.info(
        f"Saved {len(tp_metrics)} TrainingPeaks metrics to {config.ML_READY_TP_METRICS_FILE}"
    )


def main():
    config.logger.info("***Preparing data for model...***")
    os.makedirs(config.ML_READY_DIR, exist_ok=True)
    make_processed_data_ml_ready()
    config.logger.info("------------------------------------------------\n")
