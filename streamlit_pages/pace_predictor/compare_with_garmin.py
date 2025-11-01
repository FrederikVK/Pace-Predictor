# Base Imports
import pandas as pd

# Third-party Imports
import plotly.graph_objects as go
import streamlit as st

# Local Imports
from data.local_db.manual import garmin_prediction, labelled_runs
from models import config
from utils import converters

garmin_pred = garmin_prediction.garmin_pred[
    ["5k_km_pr_hour", "10k_km_pr_hour", "half_km_pr_hour", "full_km_pr_hour"]
].rename(
    columns={
        "5k_km_pr_hour": "5k",
        "10k_km_pr_hour": "10k",
        "half_km_pr_hour": "Halfmarathon",
        "full_km_pr_hour": "Marathon",
    }
)
races = labelled_runs.races.iloc[1:].copy()[
    ["run_type", "average_km_pr_hour", "Description", "averageHR"]
]

# Your predefined dictionaries
colors = {"5k": "blue", "10k": "orange", "Halfmarathon": "green", "Marathon": "red"}

distances: dict[str, float | str] = {
    "5k": 5,
    "10k": 10,
    "Halfmarathon": 21.0975,
    "Marathon": 42.195,
}

model_pred = pd.read_parquet(f"{config.MODEL_OUTPUT_DIR}km_pr_hour_pred.parquet")


def pace_plot(model_pred, races, title):
    fig = go.Figure()

    # Garmin lines with detailed hover info
    for col in model_pred.columns:
        hover_texts = []
        for idx, speed in model_pred[col].items():
            distance = distances.get(col, None)

            if distance is not None and speed > 0:
                duration_seconds = (distance / speed) * 3600
                pace = 60 / speed

                duration_str = converters.format_seconds_to_hhmmss(duration_seconds)
                pace_str = converters.pace_to_str(pace)
            else:
                duration_str = "N/A"
                pace_str = "N/A"
                distance = "N/A"

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
                y=model_pred[col],
                mode="lines+markers",
                name=col,
                line=dict(color=colors[col]),
                marker=dict(size=6),
                hoverinfo="text",
                text=hover_texts,
            )
        )

    # Race points with Description in hover
    for run_type in races["run_type"].unique():
        filtered = races[races["run_type"] == run_type]

        hover_texts = []
        for idx, row in filtered.iterrows():
            speed = row["average_km_pr_hour"]
            hr = row["averageHR"]
            distance = distances.get(run_type, None)

            if distance is not None and speed > 0:
                duration_seconds = (distance / speed) * 3600
                pace = 60 / speed

                duration_str = converters.format_seconds_to_hhmmss(duration_seconds)
                pace_str = converters.pace_to_str(pace)
            else:
                duration_str = "N/A"
                pace_str = "N/A"
                distance = "N/A"

            description = row.get("Description", "")

            text = (
                f"Date: {idx.strftime('%Y-%m-%d')}<br>"
                f"Distance: {distance} km<br>"
                f"Duration: {duration_str}<br>"
                f"Pace: {pace_str}<br>"
                f"Speed: {speed:.2f} km/h<br>"
                f"HR: {hr} <br>"
                f"Description: {description}"
            )
            hover_texts.append(text)

        fig.add_trace(
            go.Scatter(
                x=filtered.index,
                y=filtered["average_km_pr_hour"],
                mode="markers",
                name=f"Race points: {run_type}",
                marker=dict(color=colors[run_type], size=10, symbol="circle-open"),
                hoverinfo="text",
                text=hover_texts,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="km/t",
        legend_title="Run Type",
        hovermode="closest",
        yaxis=dict(range=[10, 18]),
    )
    return fig


model_name = st.radio("Choose model", config.MODEL_NAMES, index=3)

fig1 = pace_plot(
    model_pred=model_pred.query("model == @model_name").drop(columns=["model"]),
    races=races,
    title="Own Pace Predictor",
)
fig2 = pace_plot(
    model_pred=garmin_pred.iloc[1:], races=races, title="Garmin Pace Predictor"
)
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
