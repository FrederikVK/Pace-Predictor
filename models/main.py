# Base Imports
import os
import shutil

from models import config

# Local Imports
from models.ols_models import (
    model_interaction,
    model_ols_simple,
    model_tp_metrics,
    model_weight,
)
from models.utils import export


def reset_mlflow(path="mlruns"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/.trash", exist_ok=True)


if __name__ == "__main__":

    config.logger.info("***Starting model training (we drop previous models)...***")
    reset_mlflow("mlruns")  # no need for historical models for now

    # Train different models
    model_ols_simple.train()
    model_interaction.train()
    model_tp_metrics.train()
    model_weight.train()
    config.logger.info("------------------------------------------------\n")

    # Export output from models
    config.logger.info("***Export output from models...***")
    export.main()
    config.logger.info("------------------------------------------------\n")
