# Base Imports
# Third-party Imports
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from models import config

# Local Imports
from models.utils import helpers
from utils import log_utils


class NextStepPredictor(BaseEstimator, RegressorMixin):
    """
    This step makes next step prediction for a given model.
    It will fit the model on all data up to the current label_id,
    and then predict the next step based on the current optimal race features
    """

    def __init__(
        self,
        feature_pipeline,
        model,
        label_id,
        label_id_col="label_id",
        target_col="avg_speed_km_pr_hour",
        weight_col=None,
    ):
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.label_id = label_id
        self.label_id_col = label_id_col
        self.target_col = target_col
        self.weight_col = weight_col

    def fit(self, df):
        train = df.query(f"{self.label_id_col} < @self.label_id").copy()
        train = self.feature_pipeline.fit_transform(train)

        act_ids = set(train["activity_id"]).intersection(
            df.query("include")["activity_id"]
        )

        y_train = df.query("activity_id in @act_ids")[self.target_col].sort_index()
        X_train = (
            train.query("activity_id in @act_ids")
            .sort_index()
            .drop(columns=["activity_id"], errors="ignore")
        )

        if self.weight_col is None:
            self.model.fit(
                X_train.drop(columns=[self.weight_col], errors="ignore"), y_train
            )
        else:
            self.model.fit(
                X_train.drop(columns=[self.weight_col], errors="ignore"),
                y_train,
                sample_weight=X_train[self.weight_col],
            )

        return self

    def predict(self, df) -> float:
        """
        Next step prediction based on the current label_id and the optimal race features.
        """
        test = df.query(f"{self.label_id_col} <= @self.label_id").copy()
        test = self.feature_pipeline.transform(test).iloc[-1:]
        X_test = test.drop(columns=["activity_id", self.weight_col], errors="ignore")
        return helpers.convert_pred_to_float(self.model.predict(X_test))

    def predict_in_sample(self, df) -> pd.Series:
        X = self.feature_pipeline.transform(df)
        idx = df.query("activity_id in @X['activity_id']")["timestamp"]
        X = X.drop(columns=["activity_id", self.weight_col], errors="ignore")
        return pd.Series(self.model.predict(X), index=idx)


class Trainer:
    def __init__(
        self,
        model_name: str,
        pace_predictor_pipeline,
    ):
        self.model_name = model_name
        self.pace_predictor_pipeline = pace_predictor_pipeline
        self.label_ids = pace_predictor_pipeline.label_ids
        self.month_end_runs = pace_predictor_pipeline.month_end_runs

    def train(self, df: pd.DataFrame):
        mlflow.set_experiment(self.model_name)

        tqdm_logger = log_utils.TqdmLogger(config.logger)
        for label_id in tqdm(self.label_ids, file=tqdm_logger):
            with mlflow.start_run(run_name=f"label_{label_id}") as run:

                # pipeline
                self.pace_predictor_pipeline.init_pipeline(label_id)
                self.pace_predictor_pipeline.next_step_pred_pipeline.fit(df)
                model = self.pace_predictor_pipeline.next_step_pred_pipeline.model

                # Log metadata
                mlflow.log_param("label_id", label_id)

                # Log coefficients
                try:
                    mlflow.log_param("intercept", model.intercept_)
                    for name, coef in zip(
                        self.pace_predictor_pipeline.feature_names, model.coef_
                    ):
                        mlflow.log_param(f"coef_{name}", coef)
                except:
                    pass

                # Pace predictions
                if label_id in self.month_end_runs:
                    suffix = str(self.month_end_runs[label_id])

                    pace_pred = {
                        f"5k.{suffix}": model.predict(
                            self.pace_predictor_pipeline.prep_input(
                                distance_km=5.0,
                                avg_heart_rate=config.MAX_AVG_HR_5K,
                                mode="optimal",
                            )
                        ),
                        f"10k.{suffix}": model.predict(
                            self.pace_predictor_pipeline.prep_input(
                                distance_km=10.0,
                                avg_heart_rate=config.MAX_AVG_HR_10K,
                                mode="optimal",
                            )
                        ),
                        f"halfmarathon.{suffix}": model.predict(
                            self.pace_predictor_pipeline.prep_input(
                                distance_km=21.097,
                                avg_heart_rate=config.MAX_AVG_HR_HALFMARATHON,
                                mode="optimal",
                            )
                        ),
                        f"marathon.{suffix}": model.predict(
                            self.pace_predictor_pipeline.prep_input(
                                distance_km=42.195,
                                avg_heart_rate=config.MAX_AVG_HR_MARATHON,
                                mode="optimal",
                            )
                        ),
                    }

                    mlflow.log_metrics(pace_pred)  # type: ignore

                # If last model
                if label_id == self.label_ids[-1]:

                    # fitted values - in sample
                    df_sub = df.query("include")
                    y_test = df_sub.set_index("timestamp")["avg_speed_km_pr_hour"]
                    y_pred = self.pace_predictor_pipeline.next_step_pred_pipeline.predict_in_sample(
                        df_sub
                    )
                    file_path = f"{config.MODEL_OUTPUT_DIR}{self.model_name}_fitted_values.parquet"
                    pd.concat(
                        [y_pred.to_frame("y_pred"), y_test.to_frame("y_test")], axis=1
                    ).dropna().to_parquet(file_path)

                    # Log and register pipeline, with current model
                    input_example = self.pace_predictor_pipeline.prep_input(
                        distance_km=5.0, avg_heart_rate=168, mode="optimal"
                    )
                    signature = infer_signature(
                        input_example, model.predict(input_example)
                    )
                    mlflow.sklearn.log_model(  # type: ignore
                        self.pace_predictor_pipeline,
                        name=self.model_name,
                        registered_model_name=self.model_name,
                        signature=signature,
                    )
