"""
📊 Exploratory Data Analysis
==============================
Interactive exploration of the M5 sales data with filters for store,
category, and date range.  All charts use the project's dark Plotly theme.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Walmart M5",
    page_icon=None,
    layout="wide",
)

from dashboard.components.charts import (  # noqa: E402
    DATA_COLOURS,
    PALETTE,
    create_plotly_theme,
    section_header,
)

_THEME = create_plotly_theme()
PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data"


# ── data loading (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading sales data …")
def load_sales() -> Optional[pd.DataFrame]:
    """Try to load the processed / raw sales data."""
    # Try several common paths produced by src/ modules
    candidates = [
        DATA_DIR / "processed" / "sales_long.parquet",
        DATA_DIR / "processed" / "sales_long.csv",
        DATA_DIR / "raw" / "sales_train_evaluation.csv",
        DATA_DIR / "sales_train_evaluation.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, parse_dates=["date"] if "date" in
                               pd.read_csv(p, nrows=0).columns else None)
    return None


@st.cache_data(show_spinner="Loading calendar …")
def load_calendar() -> Optional[pd.DataFrame]:
    for p in [DATA_DIR / "raw" / "calendar.csv", DATA_DIR / "calendar.csv"]:
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"])
    return None


def _demo_sales() -> pd.DataFrame:
    """Generate synthetic sales data for demo / first‑run."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2011-01-29", periods=1941, freq="D")
    stores = [f"CA_{i}" for i in range(1, 5)] + \
             [f"TX_{i}" for i in range(1, 4)] + \
             [f"WI_{i}" for i in range(1, 4)]
    categories = ["HOBBIES", "HOUSEHOLD", "FOODS"]
    rows = []
    for store in stores:
        for cat in categories:
            base = rng.integers(80, 250)
            trend = np.linspace(0, rng.integers(10, 40), len(dates))
            seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            noise = rng.normal(0, 8, len(dates))
            sales = np.maximum(base + trend + seasonal + noise, 0).astype(int)
            for d, s in zip(dates, sales):
                rows.append({"date": d, "store_id": store, "cat_id": cat, "sales": int(s)})
    return pd.DataFrame(rows)


# ── load data ───────────────────────────────────────────────────────────────
sales_df = load_sales()
if sales_df is None:
    st.info("No processed data found — showing **demo data**. "
            "Place your M5 data in `data/raw/` and run the preprocessing pipeline.")
    sales_df = _demo_sales()

# Normalise columns
if "date" not in sales_df.columns and "d" in sales_df.columns:
    cal = load_calendar()
    if cal is not None:
        sales_df = sales_df.merge(cal[["d", "date"]], on="d", how="left")

if "date" in sales_df.columns:
    sales_df["date"] = pd.to_datetime(sales_df["date"])

# ── sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")

store_options = sorted(sales_df["store_id"].unique()) if "store_id" in sales_df.columns else []
selected_stores = st.sidebar.multiselect("Store", store_options, default=store_options[:3])

cat_options = sorted(sales_df["cat_id"].unique()) if "cat_id" in sales_df.columns else []
selected_cats = st.sidebar.multiselect("Category", cat_options, default=cat_options)

if "date" in sales_df.columns:
    min_date = sales_df["date"].min().date()
    max_date = sales_df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                       min_value=min_date, max_value=max_date)
else:
    date_range = None

# Apply filters
filt = sales_df.copy()
if selected_stores and "store_id" in filt.columns:
    filt = filt[filt["store_id"].isin(selected_stores)]
if selected_cats and "cat_id" in filt.columns:
    filt = filt[filt["cat_id"].isin(selected_cats)]
if date_range and len(date_range) == 2 and "date" in filt.columns:
    filt = filt[(filt["date"] >= pd.Timestamp(date_range[0])) &
                (filt["date"] <= pd.Timestamp(date_range[1]))]

# ── title ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="background:linear-gradient(135deg,#bfa085,#8c6c53);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:800;">
        Exploratory Data Analysis
    </h1>
    """,
    unsafe_allow_html=True,
)

# ─── Section 1: Overall Sales Trends ────────────────────────────────────────
section_header("1 · Overall Sales Trend", "Daily total sales across selected filters")

if "date" in filt.columns:
    daily = filt.groupby("date")["sales"].sum().reset_index()
    fig1 = px.line(daily, x="date", y="sales", template=_THEME,
                   labels={"date": "Date", "sales": "Total Sales"})
    fig1.update_traces(line=dict(color=PALETTE["blue"], width=1.8))
    fig1.update_layout(hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Date column not available for trend chart.")

# ─── Section 2: Sales by Store ──────────────────────────────────────────────
section_header("2 · Sales by Store", "Aggregated sales per store for the selected period")

if "store_id" in filt.columns:
    store_agg = filt.groupby("store_id")["sales"].sum().reset_index().sort_values("sales", ascending=True)
    fig2 = px.bar(store_agg, x="sales", y="store_id", orientation="h",
                  color="store_id", color_discrete_sequence=DATA_COLOURS,
                  template=_THEME, labels={"sales": "Total Sales", "store_id": "Store"})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ─── Section 3: Sales by Category ───────────────────────────────────────────
section_header("3 · Sales by Category", "Proportion of total sales across product categories")

if "cat_id" in filt.columns:
    cat_agg = filt.groupby("cat_id")["sales"].sum().reset_index()
    fig3 = px.treemap(cat_agg, path=["cat_id"], values="sales",
                      color="sales", color_continuous_scale=["#4d3c32", "#8c6c53", "#bfa085"],
                      template=_THEME)
    fig3.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# ─── Section 4: Seasonality Decomposition ───────────────────────────────────
section_header("4 · Seasonality Decomposition", "Trend, seasonal, and residual components")

with st.expander("ℹ️ How does decomposition work?"):
    st.markdown(
        "We use **statsmodels** `seasonal_decompose` with a multiplicative model "
        "(period = 7 for weekly seasonality).  The decomposition separates the "
        "raw signal into **trend**, **seasonal**, and **residual** components."
    )

if "date" in filt.columns:
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        daily_ts = filt.groupby("date")["sales"].sum()
        daily_ts = daily_ts.asfreq("D").ffill()
        if len(daily_ts) >= 14:
            result = seasonal_decompose(daily_ts, model="additive", period=7)
            decomp_df = pd.DataFrame({
                "Trend": result.trend,
                "Seasonal": result.seasonal,
                "Residual": result.resid,
            }).dropna()

            tabs = st.tabs(["Trend", "Seasonal", "Residual"])
            for tab, col, clr in zip(tabs,
                                     ["Trend", "Seasonal", "Residual"],
                                     [PALETTE["blue"], PALETTE["teal"], "#ff6b6b"]):
                with tab:
                    fig_d = px.line(decomp_df, y=col, template=_THEME)
                    fig_d.update_traces(line_color=clr)
                    st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("Not enough data points for decomposition (need ≥ 14 days).")
    except ImportError:
        st.warning("Install `statsmodels` for seasonality decomposition.")

# ─── Section 5: Correlation Heatmap ─────────────────────────────────────────
section_header("5 · Correlation Heatmap", "Relationships between sales and external signals")

# Build a small correlation matrix from available numeric columns
numeric_cols = filt.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    corr = filt[numeric_cols].corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, "#222222"], [0.5, "#8c6c53"], [1, "#bfa085"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig5.update_layout(template=_THEME, height=480, title="Pearson Correlation")
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info(
        "Correlation heatmap requires at least two numeric columns "
        "(e.g. sales, sell_price, snap flags).  Run the feature engineering "
        "pipeline to enrich the dataset."
    )

# ── footer hint ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 0.5rem 0;color:#4b5563;font-size:0.78rem;">
        Use the sidebar filters to refine the analysis.
    </div>
    """,
    unsafe_allow_html=True,
)
