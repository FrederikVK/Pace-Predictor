# Base Imports
import pandas as pd

# Third-party Imports
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local Imports
from data import config


def histogram(df: pd.DataFrame, metric: str, bins: int = 30):
    """Plot an interactive histogram showing counts using Plotly."""
    data = df[metric].dropna()
    if data.empty:
        raise ValueError(f"No valid data in column '{metric}' to plot.")

    # Plot histogram with counts (not density)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            histnorm="",  # <- counts, not density
            marker_color="indianred",
            opacity=0.75,
            name="Histogram",
            hovertemplate=f"{metric}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Histogram of '{metric}' (Counts)",
        xaxis_title=metric,
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.05,
    )

    return fig


def time_plot(
    df: pd.DataFrame,
    metric: str,
    timespan: str,
):
    if metric in ["distance_km", "elevationGain", "rTSS"]:
        func = "sum"
    elif metric in ["averageHR", "average_km_pr_hour", "IF", "VI", "normalized_speed"]:
        func = "mean"
    elif metric in ["maxHR", "EF"]:
        func = "max"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    title = f"{metric} {func} over time: ({timespan})"
    if timespan == "day":
        kind = "points"
    else:
        kind = "bar"

    # Create time grouping column
    if timespan == "month":
        df["time_group"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        time_format = "%Y-%m"
    elif timespan == "week":
        df["time_group"] = pd.to_datetime(
            df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
            format="%G%V%u",
        )
        time_format = "%Y-W%V"
    elif timespan == "day":
        df["time_group"] = pd.to_datetime(df["timestamp"])
        time_format = "%Y-%m-%d"
    else:
        raise ValueError("Invalid timespan. Choose 'week', 'month', or 'day'.")

    grouped = (
        df.groupby("time_group")[metric]
        .agg(func)
        .reset_index()
        .sort_values("time_group")
    )

    grouped["label"] = grouped["time_group"].dt.strftime(time_format)

    # Plot
    if kind == "bar":
        fig = px.bar(
            grouped,
            x="time_group",
            y=metric,
            hover_data={"label": True, "time_group": False},
            title=title,
        )
    else:
        fig = px.scatter(
            grouped,
            x="time_group",
            y=metric,
            hover_data={"label": True, "time_group": False},
            title=title,
        )

    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>" + metric + ": %{y}<extra></extra>",
        customdata=grouped[["label"]].values,
    )

    fig.update_layout(
        xaxis_title=timespan.capitalize(),
        yaxis_title=metric,
        xaxis=dict(tickformat=time_format, tickangle=45),
    )

    return fig


def add_year_separators(fig, df, timespan):
    if timespan == "month":
        min_date = pd.to_datetime(df[["year", "month"]].assign(day=1)).min()
        max_date = pd.to_datetime(df[["year", "month"]].assign(day=1)).max()
    elif timespan == "week":
        min_date = pd.to_datetime(
            df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
            format="%G%V%u",
        ).min()
        max_date = pd.to_datetime(
            df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
            format="%G%V%u",
        ).max()
    elif timespan == "day":
        min_date = pd.to_datetime(df["timestamp"]).min()
        max_date = pd.to_datetime(df["timestamp"]).max()
    else:
        raise ValueError("Invalid timespan. Choose 'week', 'month', or 'day'.")

    years = range(min_date.year, max_date.year + 1)
    shapes = []
    for year in years:
        x_pos = pd.to_datetime(f"{year}-01-01")
        if min_date <= x_pos <= max_date:
            shapes.append(
                dict(
                    type="line",
                    x0=x_pos,
                    x1=x_pos,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="Black", width=2, dash="solid"),
                    layer="below",
                )
            )

    fig.update_layout(shapes=shapes)
    return fig


@st.cache_data
def load_data():
    derived = pd.read_parquet(config.ML_READY_TP_METRICS_FILE)
    derived["normalized_speed"] = (
        derived["normalized_speed"] * 3.6
    )  # convert m/s to km/h
    derived["year"] = derived["timestamp"].dt.year
    derived["month"] = derived["timestamp"].dt.month
    derived["week"] = derived["timestamp"].dt.isocalendar().week

    raw = pd.read_parquet(config.ML_READY_META_FILE)
    raw["timestamp"] = raw["startTimeGMT"]
    return raw, derived


raw, derived = load_data()

METRICS = {
    "distance_km (raw)": "distance_km",
    "elevationGain (raw)": "elevationGain",
    "averageHR (raw)": "averageHR",
    "maxHR (raw)": "maxHR",
    "average_km_pr_hour (raw)": "average_km_pr_hour",
    "normalized_speed (derived)": "normalized_speed",
    "rTSS (derived)": "rTSS",
    "IF (derived)": "IF",
    "EF (derived)": "EF",
    "VI (derived)": "VI",
}

METRICS_EQUATIONS = {
    "distance_km (raw)": r"""\\[6.5em]""",
    "elevationGain (raw)": r"""\\[6.5em]""",
    "averageHR (raw)": r"""\\[6.5em]""",
    "maxHR (raw)": r"""\\[6.5em]""",
    "average_km_pr_hour (raw)": r"""\\[6.5em]""",
    "normalized_speed (derived)": r"""
        \mathrm{normalized\ speed} = \left[ \frac{1}{N} \sum_{i=1}^N \left( \overline{v}_i \right)^4 \right]^{\frac14} \\
        \\[0.5em]
        \text{where } \overline{v}_i \text{ is the 30-second rolling average speed at time step } i
        \\[0em]
        \text{}
        """,
    "rTSS (derived)": r"""
        \mathrm{rTSS\ (running\ Training\ Stress\ Score)} = \frac{t \times \mathrm{IF}^2}{3600} \times 100 \\
        \\[0.5em]
        \text{where } t \text{ is duration in seconds and IF is Intensity Factor}
        \\[1.0em]
        \text{}
        """,
    "IF (derived)": r"""
        \mathrm{IF\ (Intensity\ Factor)} = \frac{\mathrm{normalized\ speed}}{\mathrm{FTS}} \\
        \\[1em]
        \text{where } \mathrm{FTS\ (Functional\ Threshold\ Speed)} \text{ is maximum sustainable speed for $\sim$ 1 hour}
        \\[1em]
        \text{}
        """,
    "EF (derived)": r"""
        \mathrm{EF\ (Efficiency\ Factor)} = \frac{\mathrm{normalized\ speed}}{\mathrm{HR}_{\mathrm{avg}}} \\
        \\[4em]
        """,
    "VI (derived)": r"""
        \mathrm{VI\ (Variability\ Index)} = \frac{\mathrm{normalized\ speed}}{\mathrm{average\ speed}} \\
        \\[4em]
        """,
}

# Layout
col1, col2 = st.columns(2)

with col1:
    metric = st.selectbox("Select a metric to plot:", list(METRICS.keys()))
    bins = st.slider("Select number of bins:", min_value=5, max_value=100, value=30)

    if "raw" in metric:
        df = raw
    elif "derived" in metric:
        df = derived
    else:
        raise

    fig = histogram(df, metric=METRICS[metric], bins=bins)
    st.plotly_chart(fig, use_container_width=True)

    st.latex(METRICS_EQUATIONS[metric])

with col2:
    timespan = st.radio("Select timespan:", ["day", "week", "month"], index=1)

    # to align the plots in UI
    st.markdown(
        "<br>" * 2, unsafe_allow_html=True
    )  # adds vertical space (x line breaks)

    fig = time_plot(df, metric=METRICS[metric], timespan=timespan)
    fig = add_year_separators(fig=fig, df=df, timespan=timespan)
    st.plotly_chart(fig, use_container_width=True)
