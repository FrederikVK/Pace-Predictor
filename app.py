import streamlit as st

st.set_page_config(
    page_title="Pace Predictor App",
    layout="wide",  # forces all pages to wide
    initial_sidebar_state="expanded",
)

# Introduction
introduction = st.Page(
    "streamlit_pages/introduction/introduction.py",
    title="Introduction",
    icon="🏃",
    default=True,
)

# Data
data_pipeline_runner = st.Page(
    "streamlit_pages/data/data_pipeline_runner.py",
    title="Data Pipeline Runner",
    icon="▶️",
)
data_visualization = st.Page(
    "streamlit_pages/data/data_visualization.py",
    title="Data Visualization",
    icon="📈",
)
training_load = st.Page(
    "streamlit_pages/data/training_load.py",
    title="Training Load",
    icon="📈",
)

# Models
models_train_runner = st.Page(
    "streamlit_pages/models/models_train_runner.py",
    title="Models Train Runner",
    icon="▶️",
)
models_inspection = st.Page(
    "streamlit_pages/models/inspection.py",
    title="Model Inspection",
    icon="🧠",
)

# Pace Predictor
pace_predictor = st.Page(
    "streamlit_pages/pace_predictor/pace_predictor.py",
    title="Pace Predictor",
    icon="⌚",
)

compare_with_garmin = st.Page(
    "streamlit_pages/pace_predictor/compare_with_garmin.py",
    title="Pace Predictor vs. Garmin",
    icon="⌚",
)

pg = st.navigation(
    {
        "Introduction": [introduction],
        "Data": [data_pipeline_runner, data_visualization, training_load],
        "Models": [
            models_train_runner,
            models_inspection,
        ],
        "Pace Predictor": [pace_predictor, compare_with_garmin],
    }
)

pg.run()
