from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd


def find_month_end_label_ids(
    df, start="2024-02-28", end=str(datetime.today().date()), date_col="timestamp"
):
    """
    Find the label_id for each month-end run between start and end dates (include end date).

    Returns:
        dict: {label_id: month_end_date}
    """
    month_ends = pd.date_range(
        start=pd.to_datetime(start), end=pd.to_datetime(end), freq="ME"
    )
    end = pd.to_datetime(end)
    if end not in month_ends:
        month_ends = month_ends.union([end])

    month_end_runs = {}
    for month_end in month_ends:
        time_diff = (df[date_col] - month_end).abs()
        closest_idx = time_diff.idxmin()
        label_id = df.loc[closest_idx, "label_id"]
        month_end_runs[label_id] = month_end.date()
    return month_end_runs


def add_label_id_to_df(df, start_offset=pd.Timedelta(days=7 * 6 + 1)):
    df = df.copy()
    start_dt = df["timestamp"].min() + start_offset
    df["label_id"] = 0
    idx = df["timestamp"] > start_dt
    df.loc[idx, "label_id"] = range(1, sum(idx) + 1)
    return df


def infer_optimal_avg_heart_rate(
    features: pd.DataFrame,
    distance_km: float,
) -> float:
    avg_heart_rate = (
        features.query("distance_km > @distance_km*0.8")
        .query("distance_km < @distance_km*1.2")["avg_heart_rate"]
        .max()
        # .sort_values(ascending=False)
        # .iloc[:3]
        # .mean()
    )
    return round(avg_heart_rate, 0)


def infer_CTL(
    features: pd.DataFrame,
) -> float:
    # current CTL
    return features["CTL"].iloc[-1]


def infer_Volume(
    features: pd.DataFrame,
) -> float:
    # current Volume
    return features["Volume"].iloc[-1]


def infer_FORM(
    features: pd.DataFrame,
    mode: str,
) -> float:
    if mode == "optimal":
        return features["FORM"].max()
    elif mode == "current":
        return features["FORM"].iloc[-1]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'optimal' or 'current'.")


def infer_EF_tl(
    features: pd.DataFrame,
) -> float:
    # current EF_TL
    return features["EF_tl"].iloc[-1]


def infer_VI(
    features: pd.DataFrame,
) -> float:
    # Theoretical optimal VI
    return 1.0


def convert_pred_to_float(pred: Union[float, np.ndarray, pd.Series]):
    if isinstance(pred, pd.Series):
        return float(pred.iloc[0])
    elif isinstance(pred, np.ndarray):
        return float(pred[0])
    else:
        return float(pred)
