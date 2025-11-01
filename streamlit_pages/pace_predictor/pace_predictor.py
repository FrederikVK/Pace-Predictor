# Base Imports
import copy
from typing import Optional

# Third-party Imports
import mlflow
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models import config

# Local Imports
from models.utils import helpers
from utils import converters


def load_model(model_name: str):
    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/latest")  # type: ignore
    model = pipeline.next_step_pred_pipeline.model  # type: ignore
    return model, pipeline


def validate_input(X):
    if X["avg_heart_rate"].isna().any():
        st.error("avg_heart_rate is required and could not be inferred from data")
        st.stop()


def retrain_model_and_predict_by_month(
    pipeline,
    mode: str,
    distance_km: float,
    avg_heart_rate: Optional[float] = None,
):
    pipeline = copy.deepcopy(pipeline)

    pace_predictions = {}
    for label_id, date in pipeline.month_end_runs.items():
        pipeline.init_pipeline(label_id=label_id)
        pipeline.next_step_pred_pipeline.fit(pipeline.df)
        model = pipeline.next_step_pred_pipeline.model

        X = pipeline.prep_input(
            distance_km=distance_km,
            avg_heart_rate=avg_heart_rate,
            mode=mode,
        )
        validate_input(X)
        pace_predictions[date] = helpers.convert_pred_to_float(model.predict(X))

    return pace_predictions


def pace_plot(model_pred: pd.Series, distance: float, title: str):
    fig = go.Figure()

    durations_seconds = []
    hover_texts = []
    for idx, speed in model_pred.items():
        if distance is not None and speed > 0:
            duration_sec = (distance / speed) * 3600
            pace = 60 / speed

            duration_str = converters.format_seconds_to_hhmmss(duration_sec)
            pace_str = converters.pace_to_str(pace)
        else:
            duration_sec = None
            duration_str = "N/A"
            pace_str = "N/A"

        durations_seconds.append(duration_sec if duration_sec is not None else 0)

        text = (
            f"Date: {idx.strftime('%Y-%m-%d')}<br>"  # type: ignore
            f"Distance: {distance} km<br>"
            f"Duration: {duration_str}<br>"
            f"Pace: {pace_str}<br>"
            f"Speed: {speed:.2f} km/h"
        )
        hover_texts.append(text)

    fig.add_trace(
        go.Scatter(
            x=model_pred.index,
            y=model_pred.values,
            mode="lines+markers",
            name="Speed (km/h)",
            line=dict(color="blue"),
            marker=dict(size=6),
            hoverinfo="text",
            text=hover_texts,
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=model_pred.index,
            y=durations_seconds,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            yaxis="y2",
        )
    )

    def margin_range(min_val, max_val, margin=0.02):
        span = max_val - min_val
        return [min_val - span * margin, max_val + span * margin]

    speed_min, speed_max = min(model_pred.values), max(model_pred.values)
    duration_min, duration_max = min(durations_seconds), max(durations_seconds)

    speed_range = margin_range(speed_min, speed_max)
    duration_range = margin_range(duration_min, duration_max)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(
            title="Speed (km/h)",
            range=speed_range,
            side="left",
            showline=True,
            showgrid=True,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Duration (hh:mm:ss)",
            overlaying="y",
            side="right",
            range=[duration_range[1], duration_range[0]],  # <-- reversed here
            tickmode="array",
            tickvals=[duration_max, (duration_min + duration_max) / 2, duration_min],
            ticktext=[
                converters.format_seconds_to_hhmmss(duration_max),
                converters.format_seconds_to_hhmmss((duration_min + duration_max) / 2),
                converters.format_seconds_to_hhmmss(duration_min),
            ],
            showline=True,
            showgrid=False,
            zeroline=False,
            showticklabels=True,
        ),
        showlegend=False,
        hovermode="closest",
    )

    return fig


st.title(f"Pace Prediction")
model_name = st.radio("Choose model", config.MODEL_NAMES, index=3)
model, pipeline = load_model(model_name)
st.markdown(
    f"""
Using the model: **{model_name}**
"""
)

# features
distance_km = st.number_input(
    "distance_km",
    value=5.0,
    min_value=0.8,
    max_value=42.195,
)
use_infer = st.checkbox("Infer avg_heart_rate from data", value=False)
if use_infer:
    avg_heart_rate = None
else:
    avg_heart_rate = st.number_input(
        "avg_heart_rate",
        value=config.MAX_AVG_HR_5K,
        min_value=100,
        max_value=config.MAX_HR,
        help="Set manually or check 'Infer from data'",
    )

if st.button("Predict"):
    # Prepare input for model as DataFrame
    X_opt = pipeline.prep_input(  # type: ignore
        distance_km=distance_km,
        avg_heart_rate=avg_heart_rate,
        mode="optimal",
    )
    X_current = pipeline.prep_input(  # type: ignore
        distance_km=distance_km,
        avg_heart_rate=avg_heart_rate,
        mode="current",
    )
    validate_input(X_opt)
    validate_input(X_current)

    # Make prediction
    def make_pred(X):
        speed = helpers.convert_pred_to_float(model.predict(X))
        duration_seconds = (distance_km / speed) * 3600
        pace = 60 / speed
        duration_str = converters.format_seconds_to_hhmmss(duration_seconds)
        pace_str = converters.pace_to_str(pace)
        return pd.DataFrame(
            {
                "Distance (km)": [distance_km],
                "Duration (hh:mm:ss)": [duration_str],
                "Pace (min/km)": [pace_str],
                "Speed (km/h)": [speed],
            }
        )

    st.markdown("### Input:")
    drop_cols = ["avg_heart_rate_sq", "distance_km_x_avg_heart_rate"]
    st.dataframe(
        pd.concat(
            [
                X_opt.T[0].to_frame("Optimal"),
                X_current.T[0].to_frame("Current"),
            ],
            axis=1,
        ).T.drop(columns=drop_cols, errors="ignore")
    )
    st.markdown("### Prediction:")
    st.dataframe(
        pd.concat(
            [
                make_pred(X_opt).T[0].to_frame("Optimal"),
                make_pred(X_current).T[0].to_frame("Current"),
            ],
            axis=1,
        ).T
    )


if st.button("Compute Pace Plot"):
    pace_predictions = retrain_model_and_predict_by_month(
        pipeline=pipeline,
        mode="optimal",
        distance_km=distance_km,
        avg_heart_rate=avg_heart_rate,
    )
    st.markdown("### Pace Prediction:")
    fig = pace_plot(
        pd.Series(pace_predictions), distance_km, f"Pace Prediction - {distance_km} km"
    )
    st.plotly_chart(fig, use_container_width=True)
