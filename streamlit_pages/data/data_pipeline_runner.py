# Base Imports
import datetime
import os
import subprocess

import numpy as np
import pandas as pd

# Third-party Imports
import streamlit as st

# Local Imports
from data import config
from utils import log_utils


def get_metrics() -> dict[str, int | datetime.date]:
    raw_activities = len(os.listdir(config.RAW_DIR))
    try:
        processed_acitvities = pd.read_parquet(config.PROCESSED_META_FILE)[
            "activity_id"
        ].nunique()
    except FileNotFoundError:
        processed_acitvities = 0

    try:
        df = pd.read_parquet(config.ML_READY_META_FILE)
        run_activities = df["activity_id"].nunique()
        start_end = df["startTimeLocal"].agg(["min", "max"]).astype("datetime64[ns]")
    except FileNotFoundError:
        run_activities = 0
        start_end = pd.Series([np.nan, np.nan], index=["min", "max"]).astype(
            "datetime64[ns]"
        )

    return {
        "raw_activities": raw_activities,
        "processed_activities": processed_acitvities,
        "run_activities": run_activities,
        "first_run": start_end["min"].date(),
        "last_run": start_end["max"].date(),
    }


# Create two columns
col1, col2, col3 = st.columns([2, 0.01, 1])  # col2 is just a hack

with col1:
    # Initialize session state
    if "pipeline_ran" not in st.session_state:
        st.session_state.pipeline_ran = False
    if "process" not in st.session_state:
        st.session_state.process = None

    if st.button("Run Pipeline"):
        # Run pipeline as subprocess
        process = subprocess.Popen(
            ["python", "-u", "-m", "data.main"],  # -u = unbuffered
            stdout=open(config.LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
        )

        log_utils.stream_log_until_done(process=process, log_file=config.LOG_FILE)

with col2:
    pass


with col3:
    st.markdown("### Data Summary")

    metrics = get_metrics()
    st.write(f"**Number of raw activities:** {metrics['raw_activities']}")
    st.write(f"**Number of processed activities:** {metrics['processed_activities']}")
    st.write(f"**Number of run activities:** {metrics['run_activities']}")
    st.write(f"**First run date:** {metrics['first_run']}")
    st.write(f"**Last run date:** {metrics['last_run']}")
