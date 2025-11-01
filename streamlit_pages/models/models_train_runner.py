# Base Imports
import subprocess

# Third-party Imports
import streamlit as st

# Local Imports
from models import config

with st.container():
    # Initialize session state
    if "pipeline_ran" not in st.session_state:
        st.session_state.pipeline_ran = False
    if "process" not in st.session_state:
        st.session_state.process = None

    if st.button("Run Pipeline"):

        # Run pipeline as subprocess
        st.session_state.process = subprocess.Popen(
            ["python", "-u", "-m", "models.main"],  # -u = unbuffered
            stdout=subprocess.DEVNULL,  # discard stdout
            stderr=subprocess.DEVNULL,  # discard stderr
        )

        # Show a spinner while waiting
        with st.spinner("Pipeline is running, please wait..."):
            st.session_state.process.wait()  # wait for completion

        st.success("Pipeline finished successfully!")
        st.session_state.pipeline_ran = True


st.markdown("# Training models")

# Build the markdown text for all models
model_text = "\n".join(
    f"- **{name}**: {desc}" for name, desc in config.MODEL_DESCRIPTION.items()
)

st.markdown(f"The Pipeline will train the following models:\n{model_text}")
