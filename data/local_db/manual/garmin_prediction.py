import pandas as pd

from utils import converters


def get_garmin_pred():
    df = pd.DataFrame(
        {
            "5k": [
                (19, 53),
                (18, 55),
                (18, 38),
                (18, 46),
                (19, 7),
                (18, 51),
                (18, 46),
                (18, 35),
                (18, 28),
                (18, 21),
                (18, 26),
                (18, 26),
                (18, 26),
                (18, 26),
                (18, 28),
                (18, 51),
                (18, 50),
                (18, 41),
                (18, 36),
                (18, 26),
                (18, 4),
                (17, 59),
                (17, 52),
            ],
            "10k": [
                (43, 16),
                (40, 1),
                (39, 18),
                (39, 43),
                (40, 55),
                (39, 45),
                (39, 33),
                (39, 10),
                (38, 57),
                (38, 42),
                (38, 54),
                (38, 54),
                (38, 52),
                (38, 52),
                (38, 54),
                (39, 43),
                (39, 43),
                (39, 26),
                (39, 13),
                (38, 49),
                (38, 5),
                (37, 53),
                (37, 41),
            ],
            "half": [
                (1, 42, 21),
                (1, 30, 51),
                (1, 29, 6),
                (1, 31, 29),
                (1, 37, 6),
                (1, 29, 40),
                (1, 28, 11),
                (1, 27, 12),
                (1, 26, 43),
                (1, 26, 7),
                (1, 27, 10),
                (1, 27, 7),
                (1, 27, 7),
                (1, 26, 33),
                (1, 26, 36),
                (1, 29, 33),
                (1, 29, 2),
                (1, 27, 51),
                (1, 27, 22),
                (1, 26, 47),
                (1, 25, 31),
                (1, 25, 44),
                (1, 25, 37),
            ],
            "full": [
                (3, 52, 20),
                (3, 28, 33),
                (3, 26, 9),
                (3, 31, 39),
                (3, 43, 49),
                (3, 23, 47),
                (3, 11, 1),
                (3, 8, 40),
                (3, 7, 27),
                (3, 6, 3),
                (3, 11, 33),
                (3, 20, 15),
                (3, 17, 20),
                (3, 13, 31),
                (3, 13, 3),
                (3, 25, 4),
                (3, 23, 41),
                (3, 14, 19),
                (3, 12, 6),
                (3, 16, 26),
                (3, 14, 5),
                (3, 14, 37),
                (3, 14, 40),
            ],
        },
        index=[
            pd.to_datetime("2023-12-31"),
            pd.to_datetime("2024-1-31"),
            pd.to_datetime("2024-2-29"),
            pd.to_datetime("2024-3-31"),
            pd.to_datetime("2024-4-30"),
            pd.to_datetime("2024-5-31"),
            pd.to_datetime("2024-6-30"),
            pd.to_datetime("2024-7-31"),
            pd.to_datetime("2024-8-31"),
            pd.to_datetime("2024-9-30"),
            pd.to_datetime("2024-10-31"),
            pd.to_datetime("2024-11-30"),
            pd.to_datetime("2024-12-31"),
            pd.to_datetime("2025-1-31"),
            pd.to_datetime("2025-2-28"),
            pd.to_datetime("2025-3-31"),
            pd.to_datetime("2025-4-30"),
            pd.to_datetime("2025-5-31"),
            pd.to_datetime("2025-6-30"),
            pd.to_datetime("2025-7-31"),
            pd.to_datetime("2025-8-31"),
            pd.to_datetime("2025-9-30"),
            pd.to_datetime("2025-10-31"),
        ],
    )

    dist_km = {"5k": 5, "10k": 10, "half": 21.0975, "full": 42.195}

    cols = df.columns
    for col in cols:
        df[f"{col}_duration_min"] = df[col].apply(converters.tuple_to_minutes)

    df["5k_pace"] = df["5k_duration_min"] / dist_km["5k"]
    df["10k_pace"] = df["10k_duration_min"] / dist_km["10k"]
    df["half_pace"] = df["half_duration_min"] / dist_km["half"]
    df["full_pace"] = df["full_duration_min"] / dist_km["full"]

    df["5k_km_pr_hour"] = 60 / df["5k_pace"]
    df["10k_km_pr_hour"] = 60 / df["10k_pace"]
    df["half_km_pr_hour"] = 60 / df["half_pace"]
    df["full_km_pr_hour"] = 60 / df["full_pace"]

    df["5k_pace_str"] = df["5k_pace"].apply(converters.pace_to_str)
    df["10k_pace_str"] = df["10k_pace"].apply(converters.pace_to_str)
    df["half_pace_str"] = df["half_pace"].apply(converters.pace_to_str)
    df["full_pace_str"] = df["full_pace"].apply(converters.pace_to_str)

    return df


garmin_pred = get_garmin_pred()
