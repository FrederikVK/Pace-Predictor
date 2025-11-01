import streamlit as st

st.markdown(
    """
# Welcome to Pace Predictor

Pace Predictor is a small hobby project designed to explore and understand running performance using personal Garmin activity data.

It lets you analyze your training, visualize progress, and compare your own Linear Regression pace models with Garmin's built-in Pace Predictor.

### What you can do

Use the sidebar to navigate between the different pages:
- *Run data pipeline*: Extract, clean and transform Garmin activity data
- *Summary Statistics*: View aggregated metrics and timelines of your activities
- *Visualizations*: Explore training patterns, trends, and relationships between key variables
- *Model Training*: Train and compare Linear Regression models on your data
- *Pace Prediction*: Estimate expected pace for any distance based on your recent performance
---

Select a page from the sidebar to get started!
"""
)
