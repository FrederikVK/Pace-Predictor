from datetime import timedelta

import pandas as pd


def pace_to_str(pace):
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def pace_to_str_series(pace_timedelta: pd.Series) -> pd.Series:
    """
    Convert a timedelta pace (seconds per kilometer) to a string in 'MM:SS' format.

    Args:
        pace_timedelta (pd.Series): Pace values as timedelta.

    Returns:
        pd.Series: Pace values as strings in 'MM:SS' format.
    """
    mins = pace_timedelta.dt.components.minutes
    secs = pace_timedelta.dt.components.seconds
    return mins.astype(str) + ":" + secs.astype(str).str.zfill(2)


def format_seconds_to_hhmmss(seconds):
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def tuple_to_minutes(t):
    if len(t) == 2:
        minutes, seconds = t
        return minutes + seconds / 60
    elif len(t) == 3:
        hours, minutes, seconds = t
        return hours * 60 + minutes + seconds / 60
    else:
        raise ValueError("Unexpected tuple length")


def speed_to_pace(speed_mps: pd.Series) -> pd.Series:
    """
    Convert speed in meters per second to pace as a timedelta (seconds per kilometer).

    Args:
        speed_mps (pd.Series): Speed values in meters per second.

    Returns:
        pd.Series: Pace values as timedelta (seconds per kilometer).
    """
    pace_sec_per_km = 1000 / speed_mps  # seconds per km
    return pd.Series(pd.to_timedelta(pace_sec_per_km, unit="s"))


def km_pr_hour(speed_mps):
    """
    Convert speed from meters per second to kilometers per hour.

    Args:
        speed_mps (float or pd.Series): Speed in meters per second.

    Returns:
        float or pd.Series: Speed in kilometers per hour.
    """
    return 3.6 * speed_mps  # km per hour
