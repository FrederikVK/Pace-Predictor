import pandas as pd

from models import config


def get_data_from_experiment(experiment_name):
    experiment = config.mlflow_client.get_experiment_by_name(experiment_name)

    # Get all runs from experiment
    runs = config.mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],  # type: ignore
        order_by=["start_time DESC"],
    )

    # Extract model metrics, parameters and predictions
    df = (
        pd.DataFrame([run.data.params | run.data.metrics for run in runs])
        .astype(float)
        .astype({"label_id": int})
        .sort_values("label_id", ignore_index=True)
    )
    assert df["label_id"].is_unique, "Multiple runs with same label_id"

    # Problem with formatting of pace_cols, we need to flatten them
    pace_cols = [col for col in df.columns if "." in col]
    km_pr_hour_pred = pd.Series(
        {col: df[col].dropna().iloc[0] for col in pace_cols}
    ).reset_index()
    km_pr_hour_pred = (
        km_pr_hour_pred["index"]
        .str.split(".", expand=True)
        .rename(columns={0: "distance", 1: "date"})
        .assign(pace=km_pr_hour_pred[0])
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .groupby(["date", "distance"])["pace"]
        .mean()
        .unstack()
    )

    coef = df.drop(columns=pace_cols)
    return coef, km_pr_hour_pred


def main():
    export_km_hour_pr_pred = {}
    for model_name in config.MODEL_NAMES:
        coef, km_pr_hour_pred = get_data_from_experiment(
            experiment_name=model_name,
        )
        coef.to_parquet(f"{config.MODEL_OUTPUT_DIR}{model_name}_coef.parquet")
        export_km_hour_pr_pred[model_name] = km_pr_hour_pred

    res = []
    for key, val in export_km_hour_pr_pred.items():
        res.append(val.assign(model=key))
    (
        pd.concat(res)
        .rename(columns={"halfmarathon": "Halfmarathon", "marathon": "Marathon"})
        .to_parquet(f"{config.MODEL_OUTPUT_DIR}km_pr_hour_pred.parquet")
    )
