"""
🔮 Demand Forecasts
====================
Visualise model predictions vs actuals, residuals, and per‑model
summary metrics.  Falls back to demo data when trained artefacts
don't exist yet.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forecasts · Walmart M5", page_icon=None, layout="wide")

from dashboard.components.charts import (  # noqa: E402
    PALETTE,
    create_forecast_chart,
    create_metric_card,
    create_plotly_theme,
    create_residuals_plot,
    section_header,
)

_THEME = create_plotly_theme()
PROJECT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT / "results"
MODELS = ["SARIMA", "Prophet", "LightGBM", "TFT", "Chronos-2"]


# ── helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading forecast results …")
def load_forecast(model: str) -> pd.DataFrame | None:
    """Try loading a forecast CSV saved by the training pipeline."""
    candidates = [
        RESULTS_DIR / "forecasts" / f"{model.lower()}_forecast.csv",
        RESULTS_DIR / "forecasts" / f"{model.lower()}_forecast.parquet",
        RESULTS_DIR / f"{model.lower()}_forecast.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
    return None


def _demo_forecast(model: str) -> pd.DataFrame:
    """Synthetic forecast data for first‑run demo."""
    rng = np.random.default_rng(hash(model) % 2**31)
    n = 28  # M5 evaluation period
    dates = pd.date_range("2016-05-23", periods=n, freq="D")
    base = rng.integers(120, 260)
    trend = np.linspace(0, rng.integers(5, 15), n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    actual = np.maximum(base + trend + seasonal + rng.normal(0, 10, n), 0)
    bias = rng.normal(0, 6, n)
    predicted = actual + bias
    ci_half = rng.uniform(12, 25, n)
    return pd.DataFrame({
        "date": dates,
        "actual": np.round(actual, 1),
        "predicted": np.round(predicted, 1),
        "ci_lower": np.round(predicted - ci_half, 1),
        "ci_upper": np.round(predicted + ci_half, 1),
    })


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    res = actual - predicted
    mae = np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res ** 2))
    mape = np.mean(np.abs(res / np.where(actual == 0, 1, actual))) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": f"{round(mape, 1)}%"}


# ── title ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="background:linear-gradient(135deg,#667eea,#764ba2);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:800;">
        Demand Forecasts
    </h1>
    """,
    unsafe_allow_html=True,
)

# ── sidebar controls ────────────────────────────────────────────────────────
st.sidebar.markdown("### Controls")
selected_model = st.sidebar.selectbox("Model", MODELS, index=2)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Select a model to view its forecast against actual values.  "
    "Confidence intervals (where available) are rendered as shaded bands."
)

# ── load or generate data ───────────────────────────────────────────────────
fc = load_forecast(selected_model)
using_demo = fc is None
if using_demo:
    fc = _demo_forecast(selected_model)
    st.warning(
        f"No saved forecast found for **{selected_model}**.  "
        "Showing **demo data**.  Run the training pipeline and save "
        "forecasts to `results/forecasts/<model>_forecast.csv`.",
    )

metrics = _compute_metrics(fc["actual"].values, fc["predicted"].values)

# ── metric cards ────────────────────────────────────────────────────────────
m_cols = st.columns(3)
for col, (name, val) in zip(m_cols, metrics.items()):
    icon = ""
    col.markdown(create_metric_card(name, str(val), icon=icon), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── tabs: Forecast / Residuals / Table ──────────────────────────────────────
tab_fc, tab_res, tab_tbl = st.tabs(["Forecast", "Residuals", "Data Table"])

with tab_fc:
    fig = create_forecast_chart(
        dates=fc["date"],
        actual=fc["actual"],
        predicted=fc["predicted"],
        ci_lower=fc.get("ci_lower"),
        ci_upper=fc.get("ci_upper"),
        title=f"{selected_model} — Forecast vs Actuals",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_res:
    residuals = fc["actual"] - fc["predicted"]
    fig_r = create_residuals_plot(fc["date"], residuals.values,
                                  title=f"{selected_model} — Residuals")
    st.plotly_chart(fig_r, use_container_width=True)

    with st.expander("ℹ️ Interpreting residuals"):
        st.markdown(
            "Residuals represent **Actual − Predicted**.  Ideally they should "
            "be centred around zero with no visible pattern.  Systematic bias "
            "(e.g. consistently positive) suggests the model is under‑predicting."
        )

with tab_tbl:
    display_df = fc.copy()
    display_df["residual"] = np.round(fc["actual"] - fc["predicted"], 2)
    st.dataframe(
        display_df.style.format({"actual": "{:.1f}", "predicted": "{:.1f}",
                                  "residual": "{:.2f}"}),
        use_container_width=True,
        height=460,
    )
