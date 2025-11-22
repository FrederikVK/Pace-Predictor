# Base Imports
import pandas as pd

from data import config

# Local Imports
from utils import converters


def get_races():
    runs = (
        pd.read_parquet(config.ML_READY_META_FILE)[
            [
                "activity_id",
                "startTimeGMT",
                "distance_km",
                "average_km_pr_hour",
                "averageHR",
                "elevationGain",
                "elevationLoss",
            ]
        ]
        .astype({"startTimeGMT": "datetime64[ns]"})
        .sort_values("startTimeGMT")
        .dropna()
    )

    ids_anot = {
        13106337936: "NA",
        17919944989: "Herlev run: full out, but snow and ice",
        16787124784: "TT",
        16579345006: "Training run: Could have run faster",
        17448078084: "Løberen Skovmaren: full out, but very hilly",
        17040830054: "Copenhagen Halfmarathon: Could have run faster",
        17158133480: "Berlin Marathon: full out, but too fast due to GPS errors",
        17021330573: "Track TT",
        19940086810: "Track TT",
        20809115831: "TT: Around Svanesøen, a bit muddy",
    }
    ids = list(ids_anot.keys())  # noqa
    races = runs.query("activity_id in @ids").copy()
    races["pace"] = 60 / races["average_km_pr_hour"]
    races["pace_str"] = races["pace"].apply(converters.pace_to_str)
    races["run_type"] = (
        races["distance_km"]
        .astype(int)
        .map({5: "5k", 10: "10k", 21: "Halfmarathon", 42: "Marathon"})
    )
    races["run_distance"] = races["run_type"].map(
        {"5k": 5, "10k": 10, "Halfmarathon": 21.097, "Marathon": 42.195}
    )
    races["run_duration_min"] = races["pace"] * races["run_distance"]
    races["run_duration"] = (races["run_duration_min"] * 60).apply(
        converters.format_seconds_to_hhmmss
    )
    races.index = pd.to_datetime(pd.to_datetime(races["startTimeGMT"]).dt.date).values

    # annotate races
    races["Description"] = races["activity_id"].map(ids_anot)

    return races


races = get_races()
