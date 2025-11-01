from mlflow.tracking import MlflowClient

# Local Imports
from utils import log_utils

# Logging
LOG_FILE = "logs/models.log"
logger = log_utils.get_logger(
    name="models_logger",
    logfile=LOG_FILE,
)

mlflow_client = MlflowClient()

MODEL_NAMES = [
    "model_ols_simple",
    "model_interaction",
    "model_tp_metrics",
    "model_weight",
]
MODEL_DESCRIPTION = {
    MODEL_NAMES[
        0
    ]: "Simple OLS with basic features (distance, avg_heart_rate, running volume).",
    MODEL_NAMES[
        1
    ]: f"{MODEL_NAMES[0]} with interaction terms (distance and avg_heart_rate, avg_heart_rate^2).",
    MODEL_NAMES[
        2
    ]: f"{MODEL_NAMES[1]} with domain specific running metrics as features (CTL, FORM, Efficiency Factor).",
    MODEL_NAMES[3]: f"{MODEL_NAMES[2]} with extra weighting on intensive runs.",
}

MODEL_OUTPUT_DIR = "models/output/"

# Physiological parameters (personal)
MAX_AVG_HR_5K = 167
MAX_AVG_HR_10K = 165
MAX_AVG_HR_HALFMARATHON = 162
MAX_AVG_HR_MARATHON = 152
MAX_HR = 184
