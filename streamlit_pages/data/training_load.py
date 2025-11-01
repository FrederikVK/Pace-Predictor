# Base Imports
import pandas as pd

# Third-party Imports
import plotly.graph_objects as go
import streamlit as st

from data import config

# Local Imports
from utils import computers


@st.cache_data
def load_data():
    df = pd.read_parquet(config.ML_READY_TP_METRICS_FILE)
    df["date"] = df["timestamp"].dt.date
    return df


def get_tss(df):
    full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return (
        df.groupby("date")
        .agg(
            rTSS=("rTSS", "sum"),
            distance_km=("distance_km", "sum"),
        )
        .reindex(full_index)
        .fillna({"rTSS": 0})
        .assign(CTL=lambda x: computers.get_ctl(x["rTSS"]))
        .assign(ATL=lambda x: computers.get_atl(x["rTSS"]))
        .assign(FORM=lambda x: x["CTL"] - x["ATL"])
        .assign(
            Volume=lambda x: x["distance_km"]
            .fillna(0)
            .rolling(window=31, min_periods=1)
            .sum()
            / 31
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )


df = load_data()
tss = get_tss(df)


from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=("CTL / ATL / FORM", "Volume"),
)

# Top chart (CTL/ATL/FORM)
fig.add_trace(
    go.Scatter(x=tss["date"], y=tss["CTL"], name="CTL", line=dict(color="blue")),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=tss["date"], y=tss["ATL"], name="ATL", line=dict(color="red")),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=tss["date"], y=tss["FORM"], name="FORM", line=dict(color="green", dash="dot")
    ),
    row=1,
    col=1,
)

# Bottom chart (Volume as bars)
fig.add_trace(
    go.Bar(
        x=tss["date"],
        y=tss["Volume"],
        name="Volume (last 31 days, avg. km/day)",
        marker_color="gray",
    ),
    row=2,
    col=1,
)

# Layout
fig.update_layout(
    title="Performance Management Chart",
    hovermode="x unified",
    template="plotly_white",
    height=800,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
### Explanation of Metrics:
- **CTL (Chronic Training Load)**: Represents the long-term training load, calculated as
  an exponentially weighted moving average (EWMA) of the training stress score (TSS) with a time constant of 42 days.
- **ATL (Acute Training Load)**: Represents the short-term training load, calculated as
  an exponentially weighted moving average (EWMA) of TSS with a time constant of 7 days.
- **FORM**: The difference between CTL and ATL, indicating the athlete's readiness.
- **Volume**: The average distance covered per day over the last 31 days.
"""
)
