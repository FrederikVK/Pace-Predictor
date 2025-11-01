# Base Imports
# Third-party Imports
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local Imports
from models import config


def load_model(model_name):

    # Load the latest version of the registered model from MLflow Model Registry
    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/latest")  # type: ignore
    coef = pd.read_parquet(f"{config.MODEL_OUTPUT_DIR}{model_name}_coef.parquet")
    fitted = pd.read_parquet(
        f"{config.MODEL_OUTPUT_DIR}{model_name}_fitted_values.parquet"
    )

    coef_cols = [
        x for x in coef.columns if x.startswith("coef_") or x.startswith("intercept")
    ]
    coef = coef[coef_cols].copy()

    fitted["residuals"] = fitted["y_pred"] - fitted["y_test"]

    features = pipeline.next_step_pred_pipeline.feature_pipeline.transform(pipeline.df)  # type: ignore
    features["timestamp"] = features["activity_id"].map(
        pipeline.df.set_index("activity_id")["timestamp"]  # type: ignore
    )
    fitted = (
        fitted.reset_index()
        .merge(features, on="timestamp", validate="1:1", how="left")
        .set_index("timestamp")
        .sort_index()
    )

    return {
        "coef": coef,
        "fitted": fitted,
    }


st.title(f"Model inspection")

# Side-by-side layout
col1, col2 = st.columns([1, 5])

with col1:
    model_name = st.radio("Choose model", config.MODEL_NAMES, index=3)
res = load_model(model_name)

with col2:
    st.markdown(
        f"""
    **Estimated coefficients**
    """
    )
    try:
        coefs = res["coef"].iloc[-1:].astype(float).round(4).squeeze()  # Series

        val = coefs["intercept"]
        sign = "+" if val >= 0 else "-"
        terms = [f"{sign} {abs(val)}"]  # intercept

        for name, val in coefs.drop("intercept").items():
            name = name.split("coef_")[1]
            name = name.replace("_", "\_")
            name = name.replace(".sq", "^2")
            name = name.replace(".x.", " \cdot ")
            sign = "+" if val >= 0 else "-"
            terms.append(f"{sign} {abs(val)} \cdot {name}")

        # Insert line breaks after every 2 terms
        for i in range(2, len(terms), 4):
            terms[i] = "\\\\ " + terms[i]  # LaTeX line break

        equation = "y =  \\\\" + " ".join(terms)

        st.markdown(f"$$ {equation} $$")
    except:
        st.markdown("Not a regression model")

st.markdown(
    f"""
Using the model: **{model_name}**
"""
)

if model_name != "model_rf":
    st.title("Marginal Effect Plot")
    options = ["avg_heart_rate", "EF_tl", "distance_km", "CTL", "Volume", "FORM"]

    def check_in_index(var):
        return bool(coefs.index.str.contains(var).any())

    # Side-by-side layout
    col1, col2 = st.columns([1, 6])

    with col1:
        st.markdown(
            "<br><br><br>", unsafe_allow_html=True
        )  # adjust number of <br> as needed
        var = st.radio(
            "Choose variable", [x for x in options if check_in_index(x)], index=0
        )

    def plot_line(df, x_col, y_col, color_col=None, title="", labels=None):
        fig = px.line(df, x=x_col, y=y_col, color=color_col, labels=labels, title=title)
        return fig

    # --- avg_heart_rate effect ---
    if var == "avg_heart_rate":
        xmin, xmax = 100, 184
        x = np.linspace(xmin, xmax, 100)

        distances = {
            "5 km": 5,
            "10 km": 10,
            "Half marathon": 21.0975,
            "Marathon": 42.195,
        }

        coef_hr = coefs.get("coef_avg_heart_rate", 0)
        coef_hr_sq = coefs.get("coef_avg_heart_rate_sq", 0)
        coef_inter = coefs.get("coef_distance_km_x_avg_heart_rate", 0)

        data = []
        for label, dist in distances.items():
            y_hat = coef_hr * x + coef_hr_sq * (x**2) + coef_inter * dist * x
            data.extend(
                [
                    {"avg_heart_rate": xi, "fitted_y": yi, "distance": label}
                    for xi, yi in zip(x, y_hat)
                ]
            )

        df_plot = pd.DataFrame(data)
        labels = {
            "avg_heart_rate": "Avg Heart Rate (bpm)",
            "fitted_y": "Fitted y",
            "distance": "Distance",
        }
        fig = plot_line(
            df_plot,
            "avg_heart_rate",
            "fitted_y",
            "distance",
            "Effect of Heart Rate on Fitted y by distance_km",
            labels,
        )

    # --- Linear variables effect ---
    elif var in ["EF_tl", "CTL", "Volume", "FORM"]:
        xmin, xmax = res["fitted"][var].min(), res["fitted"][var].max()
        x = np.linspace(xmin, xmax, 100)
        coef = coefs.get(f"coef_{var}", 0)
        y_hat = coef * x

        df_plot = pd.DataFrame({var: x, "fitted_y": y_hat})
        labels = {var: var, "fitted_y": "Fitted y"}
        fig = plot_line(
            df_plot,
            var,
            "fitted_y",
            title=f"Effect of {var} on Fitted y",
            labels=labels,
        )

    # --- distance_km effect ---
    elif var == "distance_km":
        xmin, xmax = 2, 42
        x = np.linspace(xmin, xmax, 100)
        hr_values = [140, 150, 160, 170]

        coef_dist = coefs.get("coef_distance_km", 0)
        coef_inter = coefs.get("coef_distance_km_x_avg_heart_rate", 0)

        data = []
        for hr in hr_values:
            y_hat = coef_dist * x + coef_inter * x * hr
            data.extend(
                [
                    {"distance_km": xi, "fitted_y": yi, "avg_heart_rate": hr}
                    for xi, yi in zip(x, y_hat)
                ]
            )

        df_plot = pd.DataFrame(data)
        labels = {
            "distance_km": "Distance (km)",
            "fitted_y": "Fitted y",
            "avg_heart_rate": "Heart Rate",
        }
        fig = plot_line(
            df_plot,
            "distance_km",
            "fitted_y",
            "avg_heart_rate",
            "Effect of Distance on Fitted y by Heart Rate",
            labels,
        )

    with col2:
        st.plotly_chart(fig, use_container_width=True)


# Add hover text column
fitted = res["fitted"]

cols = [
    ("HR", "avg_heart_rate", 2),
    ("Distance km", "distance_km", 2),
    ("FORM", "FORM", 2),
    ("CTL", "CTL", 2),
    ("Volume", "Volume", 2),
    ("EF_tl", "EF_tl", 4),
    ("y_pred", "y_pred", 2),
    ("y_test", "y_test", 2),
]

parts = [
    f"{label}: " + fitted[col].round(decimals).astype(str)
    for label, col, decimals in cols
    if col in fitted.columns
]

st.title("Residual Plot")

# Combine hover info
fitted["hover_text"] = pd.DataFrame(parts).T.agg("<br>".join, axis=1)

# Define bins and colors
bin_config_all = {
    "distance_km_bin": {
        "bins": [0, 2, 7.5, 15, 21, np.inf],
        "labels": ["<2", "2-8", "8-15", "15-21", ">21"],
        "colors": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
        "col_name": "distance_bin",
        "source_col": "distance_km",
    },
    "avg_heart_rate_bin": {
        "bins": [0, 128, 144, 157, 167, np.inf],
        "labels": ["<128", "128-144", "144-157", "157-167", ">167"],
        "colors": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
        "col_name": "hr_bin",
        "source_col": "avg_heart_rate",
    },
    "w_bin": {
        "bins": [1, 2, np.inf],
        "labels": ["1", ">1"],
        "colors": ["#636EFA", "#EF553B"],
        "col_name": "w_bin",
        "source_col": "weight",
    },
}

# Only keep bins that exist in the dataframe
available_bins = [
    b for b in bin_config_all if bin_config_all[b]["source_col"] in fitted.columns
]
bin_choice = st.radio("Choose bin type", available_bins, index=0)

bin_config = bin_config_all[bin_choice]

# Create the bin column
fitted[bin_config["col_name"]] = pd.cut(
    fitted[bin_config["source_col"]],
    bins=bin_config["bins"],
    labels=bin_config["labels"],
    right=False,
)

# Map colors
fitted["color"] = (
    fitted[bin_config["col_name"]]
    .astype(str)
    .map(dict(zip(bin_config["labels"], bin_config["colors"])))
)

# Plot
fig = go.Figure()
for label, color in zip(bin_config["labels"], bin_config["colors"]):
    group = fitted[fitted[bin_config["col_name"]] == label]
    if not group.empty:
        fig.add_trace(
            go.Scatter(
                x=group.index,
                y=group["residuals"],
                mode="markers",
                marker=dict(color=color, size=10),
                name=label,
                text=group["hover_text"],
                hoverinfo="text",
            )
        )

fig.update_layout(
    title=f"Residuals (in-sample) Colored by {bin_choice}",
    xaxis_title="Date",
    yaxis_title="Residual",
    legend_title=bin_choice,
)

st.plotly_chart(fig, use_container_width=True)
