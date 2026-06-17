"""
📈 Model Comparison Leaderboard
=================================
Side‑by‑side comparison of every model with auto‑highlighted winners,
a radar chart, and a training‑time bar chart.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Comparison · Walmart M5", page_icon=None, layout="wide")

from dashboard.components.charts import (  # noqa: E402
    DATA_COLOURS,
    PALETTE,
    create_comparison_radar,
    create_leaderboard_table,
    create_metric_card,
    create_plotly_theme,
    section_header,
)

_THEME = create_plotly_theme()
PROJECT = pathlib.Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT / "results" / "metrics"


# ── load / generate metrics ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading evaluation metrics …")
def load_metrics() -> pd.DataFrame | None:
    """Try loading pre‑computed metrics from the results directory."""
    candidates = [
        METRICS_DIR / "comparison.csv",
        METRICS_DIR / "leaderboard.csv",
        PROJECT / "results" / "comparison.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    # Try JSON per‑model files
    if METRICS_DIR.is_dir():
        rows = []
        for f in sorted(METRICS_DIR.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                data.setdefault("Model", f.stem)
                rows.append(data)
        if rows:
            return pd.DataFrame(rows)
    return None


def _placeholder_metrics() -> pd.DataFrame:
    """Realistic placeholder metrics for demo rendering."""
    return pd.DataFrame([
        {"Model": "SARIMA",    "RMSE": 3.42, "MAE": 2.58, "MAPE": 18.7, "SMAPE": 16.1, "Training Time (s)": 124},
        {"Model": "Prophet",   "RMSE": 3.15, "MAE": 2.31, "MAPE": 15.9, "SMAPE": 14.3, "Training Time (s)":  87},
        {"Model": "LightGBM",  "RMSE": 2.48, "MAE": 1.76, "MAPE": 11.2, "SMAPE": 10.4, "Training Time (s)":  42},
        {"Model": "TFT",       "RMSE": 2.31, "MAE": 1.64, "MAPE": 10.5, "SMAPE":  9.8, "Training Time (s)": 310},
        {"Model": "Chronos-2", "RMSE": 2.72, "MAE": 1.98, "MAPE": 13.1, "SMAPE": 12.0, "Training Time (s)":  15},
    ])


metrics_df = load_metrics()
using_placeholder = metrics_df is None
if using_placeholder:
    metrics_df = _placeholder_metrics()

# ── title ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="background:linear-gradient(135deg,#667eea,#764ba2);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:800;">
        Model Comparison Leaderboard
    </h1>
    """,
    unsafe_allow_html=True,
)

if using_placeholder:
    st.info(
        "No pre‑computed metrics found in `results/metrics/`.  "
        "Displaying **placeholder data** — run the evaluation pipeline to "
        "generate real results.",
    )

# ── leaderboard table ──────────────────────────────────────────────────────
section_header("Leaderboard", "Best value in each metric highlighted in green")
st.markdown(create_leaderboard_table(metrics_df), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── winner summary cards ───────────────────────────────────────────────────
err_cols = [c for c in ["RMSE", "MAE", "MAPE", "SMAPE"] if c in metrics_df.columns]
if err_cols:
    card_cols = st.columns(len(err_cols))
    for col, metric in zip(card_cols, err_cols):
        best_idx = metrics_df[metric].idxmin()
        best_model = metrics_df.loc[best_idx, "Model"]
        best_val = metrics_df.loc[best_idx, metric]
        col.markdown(
            create_metric_card(
                f"Best {metric}",
                f"{best_val}",
                delta=best_model,
                icon="",
            ),
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── grouped bar chart ──────────────────────────────────────────────────────
section_header("Metrics Comparison", "Grouped bar chart across all models")

melt_cols = [c for c in metrics_df.columns if c not in ("Model", "Training Time (s)")]
melted = metrics_df.melt(id_vars="Model", value_vars=melt_cols,
                          var_name="Metric", value_name="Value")
fig_bar = px.bar(
    melted, x="Metric", y="Value", color="Model", barmode="group",
    color_discrete_sequence=DATA_COLOURS, template=_THEME,
    labels={"Value": "Score"},
)
fig_bar.update_layout(legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig_bar, use_container_width=True)

# ── radar chart ─────────────────────────────────────────────────────────────
section_header("Radar Chart", "Normalised view of each model's strengths")

# Normalise error metrics to 0‑1 (inverted so larger = better)
radar_data: dict[str, dict[str, float]] = {}
for _, row in metrics_df.iterrows():
    model = row["Model"]
    radar_data[model] = {}
    for m in err_cols:
        mn = metrics_df[m].min()
        mx = metrics_df[m].max()
        rng = mx - mn if mx != mn else 1.0
        radar_data[model][m] = round(1 - (row[m] - mn) / rng, 3)

fig_radar = create_comparison_radar(radar_data, metric_names=err_cols)
st.plotly_chart(fig_radar, use_container_width=True)

# ── training time ───────────────────────────────────────────────────────────
if "Training Time (s)" in metrics_df.columns:
    section_header("Training Time", "Wall‑clock seconds for a single training run")

    time_df = metrics_df[["Model", "Training Time (s)"]].sort_values("Training Time (s)")
    fig_time = px.bar(
        time_df, x="Training Time (s)", y="Model", orientation="h",
        color="Model", color_discrete_sequence=DATA_COLOURS,
        template=_THEME,
    )
    fig_time.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_time, use_container_width=True)

# ── footer ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 0.5rem 0;color:#4b5563;font-size:0.78rem;">
        Metrics are computed on the same hold‑out window for fair comparison.
    </div>
    """,
    unsafe_allow_html=True,
)
