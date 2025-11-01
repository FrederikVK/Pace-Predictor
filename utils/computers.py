import numpy as np
import pandas as pd


def get_tl(s, tc):
    input_with_zero = pd.concat([pd.Series([0.0]), s], ignore_index=True)
    return input_with_zero.ewm(alpha=1 / tc, adjust=False).mean().values[1:]


def get_ctl(s):
    return get_tl(s, tc=42)


def get_atl(s):
    return get_tl(s, tc=7)


def riegel_equivalent_speed(time_min, dist_km, target_dist_km=5, k=1.06):
    duration = (
        time_min * (target_dist_km / dist_km) ** k
    )  # time at target distance, still in minutes
    return target_dist_km / (duration / 60)  # speed in km/h (since duration/60 = hours)
