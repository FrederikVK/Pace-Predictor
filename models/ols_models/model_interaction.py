# Base imports
from typing import Optional

# Third-party imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from models import config
from models.utils import helpers

# Local imports
from models.utils.pipelines import NextStepPredictor, Trainer
from utils import computers

# Model-specific constants
FEATURES = [
    "distance_km",
    "avg_heart_rate",
    "avg_heart_rate_sq",
    "distance_km_x_avg_heart_rate",
    "Volume",
    # "CTL",
    # "FORM",
    # "EF_tl",
]
MODEL_NAME = "model_interaction"


class FeatureComputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, df, drop_first_n: Optional[int] = 50):
        check_is_fitted(self, "is_fitted_")

        feat = df.copy()

        # Feature time-frame (all days included)
        feat["date"] = pd.to_datetime(feat["timestamp"].dt.date)
        full_index = pd.date_range(feat["date"].min(), feat["date"].max(), freq="D")
        tss_by_day = (
            feat.groupby("date")
            .agg(
                rTSS=("rTSS", "sum"),
                EF=("EF", "max"),
                distance_km=("distance_km", "sum"),
            )
            .reindex(full_index)
            .fillna({"rTSS": 0})
            .assign(CTL=lambda x: computers.get_ctl(x["rTSS"]))
            .assign(ATL=lambda x: computers.get_atl(x["rTSS"]))
            .assign(EF_tl=lambda x: computers.get_atl(x["EF"]))
            .assign(FORM=lambda x: x["CTL"] - x["ATL"])
            .assign(
                Volume=lambda x: x["distance_km"]
                .fillna(0)
                .rolling(window=31, min_periods=1)
                .sum()
                / 31
            )
            .shift(1)  # ensure no future leakage -- rows one level down
            .reset_index()
            .rename(columns={"index": "date"})
        )
        feat = feat.merge(
            tss_by_day, on="date", how="left", validate="m:1", suffixes=("", "_tmp")
        )

        # To ensure no NaN
        if drop_first_n:
            feat = feat.iloc[drop_first_n:].copy()

        # new features
        feat["avg_heart_rate_sq"] = feat["avg_heart_rate"] ** 2
        feat["EF_tl_sq"] = feat["EF_tl"] ** 2
        feat["distance_km_x_avg_heart_rate"] = (
            feat["distance_km"] * feat["avg_heart_rate"]
        )
        feat["distance_km_x_CTL"] = feat["distance_km"] * feat["CTL"]

        return feat[FEATURES + ["activity_id"]]


class PacePredictorPipeline:
    def __init__(self, df, feature_names):
        self.df = helpers.add_label_id_to_df(df)
        self.max_label_id = self.df["label_id"].max()
        self.month_end_runs = helpers.find_month_end_label_ids(self.df.query("include"))
        self.label_ids = list(self.month_end_runs.keys())
        self.pipelines = {}
        self.label_id = None
        self.feature_names = feature_names

    def init_pipeline(self, label_id):
        self.label_id = label_id
        try:
            return self.pipelines[label_id]
        except Exception:
            feature_pipeline = Pipeline(
                steps=[
                    ("feature_computer", FeatureComputer()),
                ]
            )
            self.next_step_pred_pipeline = NextStepPredictor(
                feature_pipeline=feature_pipeline,
                model=LinearRegression(),
                label_id=label_id,
            )
            self.pipelines[label_id] = self.next_step_pred_pipeline
            return self.pipelines[label_id]

    def prep_input(
        self,
        distance_km: float,
        mode: str,
        avg_heart_rate: Optional[float] = None,
    ):
        features = self.next_step_pred_pipeline.feature_pipeline.transform(
            self.df.query("label_id <= @self.label_id")
        )

        if avg_heart_rate is None:
            avg_heart_rate = helpers.infer_optimal_avg_heart_rate(
                features=features,  # type: ignore
                distance_km=distance_km,  # type: ignore
            )

        if "Volume" in FEATURES:
            Volume = helpers.infer_Volume(features=features)  # type: ignore
        else:
            Volume = None

        if "CTL" in FEATURES:
            CTL = helpers.infer_CTL(features=features)  # type: ignore
        else:
            CTL = None

        if "FORM" in FEATURES:
            FORM = helpers.infer_FORM(features=features, mode=mode)  # type: ignore
        else:
            FORM = None

        if "EF_tl" in FEATURES:
            EF_tl = helpers.infer_EF_tl(features=features)  # type: ignore
        else:
            EF_tl = None

        return pd.DataFrame(
            {
                "avg_heart_rate": [avg_heart_rate],
                "distance_km": [distance_km],
                "avg_heart_rate_sq": [avg_heart_rate**2],
                "distance_km_x_avg_heart_rate": [distance_km * avg_heart_rate],
                "Volume": [Volume],
                "CTL": [CTL],
                "FORM": [FORM],
                "EF_tl": [EF_tl],
            },
        ).astype(float)[FEATURES]

    def predict(
        self,
        distance_km: float,
        mode: str,
        avg_heart_rate: Optional[float] = None,
    ) -> float:
        X = self.prep_input(
            distance_km=distance_km, mode=mode, avg_heart_rate=avg_heart_rate
        )
        return float(self.next_step_pred_pipeline.model.predict(X)[0])


def prep_data():
    from data import config

    df = pd.read_parquet(config.ML_READY_TP_METRICS_FILE)
    df = df.dropna(subset=["avg_heart_rate"])
    df["include"] = True  # include all
    return df


def train():
    config.logger.info(f"***Training {MODEL_NAME}...***")

    df = prep_data()

    # test it works
    pipeline = PacePredictorPipeline(df, feature_names=FEATURES)
    pipeline.init_pipeline(label_id=105)
    pipeline.next_step_pred_pipeline.fit(pipeline.df)
    config.logger.debug(
        pipeline.predict(distance_km=42.195, avg_heart_rate=150, mode="optimal")
    )
    config.logger.debug(
        pipeline.predict(distance_km=42.195, avg_heart_rate=150, mode="current")
    )

    # train
    trainer = Trainer(model_name=MODEL_NAME, pace_predictor_pipeline=pipeline)
    trainer.train(pipeline.df)
