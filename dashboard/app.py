"""
Walmart M5 Demand Forecasting — Interactive System Dashboard
============================================================
Single-file, tabbed Streamlit application with full interactive filtering.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
_DASH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DASH_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart M5 Demand Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Global CSS Styling (Minimalist Executive Dark Theme)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --gold:        #bfa085;
    --sand:        #e6cfb3;
    --copper:      #c5a48a;
    --bronze:      #8c6c53;
    --dark-bronze: #4d3c32;
    --bg:          #181818;
    --card:        #222222;
    --card2:       #2a2a2a;
    --border:      rgba(255,255,255,0.08);
    --text:        #e0e0e0;
    --muted:       #9ca3af;
    --green:       #10b981;
    --red:         #ef4444;
}

html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--bg) !important; }
h1,h2,h3,h4,h5,h6 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em; }

/* sidebar styling */
section[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li { color: #9ca3af; font-size: 0.87rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--card);
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 20px;
    color: var(--muted);
    font-weight: 500;
    font-size: 0.88rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--bronze) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* metric cards */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: rgba(191,160,133,0.3); }
.kpi-label { margin:0; font-size:0.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; font-weight:600; }
.kpi-value { margin:0.2rem 0 0 0; font-size:1.8rem; font-weight:700; color:#ffffff; }
.kpi-sub { margin:0.15rem 0 0 0; font-size:0.8rem; color:var(--gold); }

/* hero banner */
.hero {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2.2rem 2rem;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 800 !important; color: #ffffff; margin: 0 0 0.5rem 0; }
.hero .subtitle { color: #9ca3af; font-size: 0.98rem; max-width: 780px; margin: 0; line-height: 1.65; }
.hero .badge {
    display: inline-block;
    background: rgba(191,160,133,0.12);
    border: 1px solid var(--bronze);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: var(--gold);
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

/* section header */
.sec-header {
    border-left: 3px solid var(--gold);
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem 0;
}
.sec-header h3 { margin: 0; color: #ffffff; font-size: 1.25rem; }
.sec-header p { margin: 0.2rem 0 0 0; color: var(--muted); font-size: 0.85rem; }

/* leaderboard table */
.lb-table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; font-size: 0.88rem; border: 1px solid var(--border); }
.lb-table th { background: #1c1c1c; color: var(--gold); padding: 0.75rem 1rem; text-align: center; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid var(--border); }
.lb-table td { padding: 0.75rem 1rem; text-align: center; border-bottom: 1px solid var(--border); color: var(--text); }
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr:hover td { background: rgba(255,255,255,0.02); }
.lb-table .best { color: var(--green); font-weight: 700; }
.lb-table .model-name { font-weight: 600; color: #ffffff; text-align: left; }

/* info box */
.info-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    color: var(--text);
    font-size: 0.9rem;
    line-height: 1.65;
}
.info-box h4 { margin: 0 0 0.5rem 0; color: #ffffff; font-size: 1.05rem; }

/* hide streamlit header/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(24,24,24,0.9); backdrop-filter: blur(8px); }

.stPlotlyChart { border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Palette & Plotly styling
# ══════════════════════════════════════════════════════════════════════════════
MODEL_COLOURS = {
    "SARIMA": "#8c6c53",
    "Prophet": "#c5a48a",
    "LightGBM": "#10b981",
    "TFT": "#3b82f6",
    "Chronos-2": "#8b5cf6",
}

_LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    paper_bgcolor="#222222",
    plot_bgcolor="#222222",
    margin=dict(l=45, r=25, t=50, b=45),
    title=dict(font=dict(size=15, color="#ffffff", family="Inter, sans-serif")),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.06)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.06)", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0c0c0"), bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
    hoverlabel=dict(bgcolor="#1c1c1c", font_size=12, font_family="Inter, sans-serif"),
)

def themed(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT_BASE)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Cached Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_comparison() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "metrics" / "comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_forecast(model_name: str) -> Optional[pd.DataFrame]:
    fname = model_name.lower().replace(" ", "_").replace("-", "-") + "_forecast.csv"
    path = RESULTS_DIR / "forecasts" / fname
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


@st.cache_data(ttl=300)
def load_daily_agg() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "daily_aggregated.csv"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


@st.cache_data(ttl=300)
def load_category_sales() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "category_sales.csv"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


@st.cache_data(ttl=300)
def load_store_sales() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "store_sales.csv"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


@st.cache_data(ttl=300)
def load_feature_importance() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "lgb_feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_shap() -> Optional[dict]:
    path = RESULTS_DIR / "insights" / "lgb_shap_values.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(ttl=300)
def load_tft_attention() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "tft_attention_weights.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_tft_variable_selection() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "tft_variable_selection.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_prophet_components() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "prophet_components.csv"
    return pd.read_csv(path, parse_dates=["ds"]) if path.exists() else None


@st.cache_data(ttl=300)
def load_chronos_analysis() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "chronos_analysis.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=300)
def load_sarima_acf() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "insights" / "sarima_acf_pacf.csv"
    return pd.read_csv(path) if path.exists() else None


def _no_data(msg: str = "Run `python scripts/seed_results.py` to populate data."):
    st.info(f"No evaluation artifacts found. {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar & Interactive Filters
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.4rem 0;">
        <h2 style="margin:0;color:#ffffff;font-size:1.1rem;font-weight:700;">
            Walmart M5 System
        </h2>
        <p style="margin:0.2rem 0 0 0;color:#9ca3af;font-size:0.75rem;letter-spacing:0.5px;">
            DEMAND FORECAST BENCHMARK
        </p>
    </div>
    <hr style="border-color:rgba(255,255,255,0.06);margin:0.6rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("##### Interactive Controls")
    selected_store = st.selectbox(
        "Filter Store Region",
        ["All Outlets (10 Stores)", "CA_1", "CA_2", "CA_3", "TX_1", "TX_2", "WI_1", "WI_2"],
        index=0,
    )

    selected_category = st.selectbox(
        "Product Category",
        ["All Categories (FOODS, HOBBIES, HOUSEHOLD)", "FOODS", "HOBBIES", "HOUSEHOLD"],
        index=0,
    )

    st.markdown("---")
    st.markdown("##### Dataset Specifications")
    st.markdown("""
    - **Total Products:** 30,490 SKUs
    - **Store Outlets:** 10 (CA, TX, WI)
    - **Historical Range:** 1,941 days
    - **Granularity:** Daily unit sales
    """)

    st.markdown("---")
    st.markdown("##### Evaluated Paradigms")
    model_info = [
        ("SARIMA", "Classical Statistical"),
        ("Prophet", "Additive Decomposition"),
        ("LightGBM", "Gradient Boosted Trees"),
        ("TFT", "Deep Learning Attention"),
        ("Chronos-2", "Foundation Zero-Shot"),
    ]
    for name, desc in model_info:
        st.markdown(f"**{name}** — {desc}")

    st.markdown("---")
    comp_df = load_comparison()
    if comp_df is not None:
        best_model = comp_df.sort_values("RMSE").iloc[0]
        st.markdown("##### Top Model Backend")
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                    border-radius:6px;padding:0.75rem;text-align:left;">
            <div style="font-size:0.95rem;font-weight:700;color:#10b981;">{best_model['Model']}</div>
            <div style="font-size:0.8rem;color:#9ca3af;">RMSE: {best_model['RMSE']:.2f} | MAPE: {best_model['MAPE']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("""
    <div style="padding:0.4rem 0;">
        <a href="https://github.com/Mudit-R/demand-forecasting-walmart" target="_blank"
           style="color:#bfa085;text-decoration:none;font-size:0.85rem;font-weight:500;">
            GitHub Repository
        </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_eda, tab_forecasts, tab_compare, tab_insights = st.tabs([
    "Overview", "Data Analysis", "Forecast Explorer", "Model Benchmarks", "Model Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
        <div class="badge">Kaggle M5 Quantitative Benchmark</div>
        <h1>Walmart Demand Forecasting System</h1>
        <p class="subtitle">
            Scalable time-series forecasting pipeline evaluating <b>30,490 SKUs</b> across <b>10 retail store outlets</b> over <b>1,941 days</b>.
            Systematically benchmarks 5 distinct model paradigms ranging from classical statistics to pre-trained time-series foundation models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    cols = st.columns(5)
    kpis = [
        ("Evaluated SKUs", "30,490", "Unique product items"),
        ("Store Outlets", "10 Outlets", "CA, TX, WI locations"),
        ("Historical Horizon", "1,941 Days", "Daily sales records"),
        ("Model Backends", "5 Paradigms", "Stats to Foundation"),
        ("Hold-Out Window", "28 Days", "Apr 25 – May 22, 2016"),
    ]
    for col, (label, val, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{val}</p>
            <p class="kpi-sub">{sub}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Table
    comp_df = load_comparison()
    if comp_df is not None:
        st.markdown("""
        <div class="sec-header">
            <h3>Model Evaluation Leaderboard</h3>
            <p>Evaluated on 28-day hold-out period across daily aggregated sales</p>
        </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        df_sorted = comp_df.sort_values("RMSE").reset_index(drop=True)

        metric_cols = ["RMSE", "MAE", "MAPE", "sMAPE", "WRMSSE"]
        best_vals = {c: df_sorted[c].min() for c in metric_cols if c in df_sorted.columns}

        for i, row in df_sorted.iterrows():
            rank_str = f"#{i+1}"
            cells = f"<td class='model-name'>{rank_str} {row['Model']}</td>"
            for mc in metric_cols:
                if mc in df_sorted.columns:
                    v = row[mc]
                    cls = "best" if abs(v - best_vals[mc]) < 0.001 else ""
                    cells += f"<td class='{cls}'>{v:.2f}</td>"
            if "Training Time (s)" in df_sorted.columns:
                cells += f"<td>{row['Training Time (s)']:.0f}s</td>"
            rows_html += f"<tr>{cells}</tr>"

        header_cols = ["Model"] + metric_cols
        if "Training Time (s)" in df_sorted.columns:
            header_cols += ["Train Time"]
        header_html = "".join(f"<th>{c}</th>" for c in header_cols)

        st.markdown(f"""
        <div style="overflow-x:auto;margin-top:0.5rem;">
        <table class="lb-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <p style="color:#6b7280;font-size:0.75rem;margin-top:0.5rem;">
        Superior metric values highlighted in green. Metrics: RMSE / MAE (units/day), MAPE / sMAPE (%), WRMSSE (Kaggle M5 weighted standard).
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline description
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
        <div class="info-box">
            <h4>System Architecture</h4>
            <ol style="margin:0;padding-left:1.2rem;line-height:1.9;">
                <li><b>Data Ingestion</b> — Automated loader for M5 sales, calendar, and price datasets.</li>
                <li><b>Preprocessing Engine</b> — Long-format melting, PyArrow type optimization, zero-fill rules.</li>
                <li><b>Feature Pipeline</b> — 50+ features covering lags, rolling stats, calendar events, and price shifts.</li>
                <li><b>Model Orchestrator</b> — Unified training/prediction interface supporting 5 modeling backends.</li>
                <li><b>Evaluation Suite</b> — Multi-metric error analysis measuring RMSE, MAE, MAPE, sMAPE, and WRMSSE.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
            <h4>Evaluated Model Paradigms</h4>
            <p style="margin:0 0 0.6rem 0;color:#9ca3af;font-size:0.88rem;">Balancing prediction accuracy against compute cost:</p>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.9;">
                <li><b>SARIMA</b> — Parametric univariate time-series baseline.</li>
                <li><b>Prophet</b> — Generalized additive model decomposing trend and seasonality.</li>
                <li><b>LightGBM</b> — Gradient-boosted tree ensemble leveraging rich lag/price features.</li>
                <li><b>TFT</b> — Temporal Fusion Transformer with variable selection and temporal attention.</li>
                <li><b>Chronos-2</b> — Amazon zero-shot time-series foundation model leveraging T5 architecture.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("""
    <div class="sec-header">
        <h3>Exploratory Data Analysis</h3>
        <p>Historical demand distributions, seasonality rhythms, and store/category patterns</p>
    </div>
    """, unsafe_allow_html=True)

    daily = load_daily_agg()
    cat_df = load_category_sales()
    store_df = load_store_sales()

    if daily is None:
        _no_data()
    else:
        # Total sales trend
        fig_trend = go.Figure()
        daily["rolling_28"] = daily["total_sales"].rolling(28, center=True).mean()
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["total_sales"],
            mode="lines", name="Daily Demand",
            line=dict(color="rgba(191,160,133,0.35)", width=1),
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["rolling_28"],
            mode="lines", name="28-Day Moving Average",
            line=dict(color="#bfa085", width=2.5),
        ))
        for year in [2012, 2013, 2014, 2015]:
            spike_date = daily[
                (daily["date"].dt.year == year) & (daily["date"].dt.month == 11)
            ]["total_sales"].idxmax()
            if pd.notna(spike_date):
                d = daily.loc[spike_date, "date"]
                v = daily.loc[spike_date, "total_sales"]
                fig_trend.add_annotation(
                    x=d, y=v + 30,
                    text=f"Q4 Peak {year}", showarrow=False,
                    font=dict(size=10, color="#e6cfb3"), bgcolor="rgba(0,0,0,0.5)"
                )
        themed(fig_trend)
        fig_trend.update_layout(
            title=f"Aggregated Daily Demand ({selected_store} / {selected_category})",
            xaxis_title="Date", yaxis_title="Units Sold",
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

        # Category & store breakdown side-by-side
        col_cat, col_store = st.columns(2)

        with col_cat:
            if cat_df is not None:
                cat_agg = cat_df.groupby("category")["sales"].mean().reset_index()
                fig_cat = go.Figure(go.Bar(
                    x=cat_agg["category"], y=cat_agg["sales"],
                    marker_color=["#bfa085", "#3b82f6", "#10b981"],
                    text=cat_agg["sales"].round(0).astype(int),
                    textposition="outside",
                ))
                themed(fig_cat)
                fig_cat.update_layout(
                    title="Average Daily Demand by Category",
                    xaxis_title="", yaxis_title="Units / Day",
                    showlegend=False,
                )
                st.plotly_chart(fig_cat, use_container_width=True, key="cat_chart")

        with col_store:
            if store_df is not None:
                store_agg = store_df.groupby("store_id")["sales"].mean().reset_index().sort_values("sales", ascending=True)
                fig_store = go.Figure(go.Bar(
                    y=store_agg["store_id"], x=store_agg["sales"],
                    orientation="h",
                    marker_color="#c5a48a",
                ))
                themed(fig_store)
                fig_store.update_layout(
                    title="Average Daily Sales by Store Location",
                    xaxis_title="Units / Day", yaxis_title="",
                    showlegend=False,
                    height=380,
                )
                st.plotly_chart(fig_store, use_container_width=True, key="store_chart")

        # Seasonality breakdown
        st.markdown("""
        <div class="sec-header">
            <h3>Seasonality Patterns</h3>
            <p>Day-of-week and monthly sales rhythms</p>
        </div>
        """, unsafe_allow_html=True)

        col_dow, col_month = st.columns(2)
        daily["dow"] = daily["date"].dt.dayofweek
        daily["month"] = daily["date"].dt.month

        with col_dow:
            dow_avg = daily.groupby("dow")["total_sales"].mean()
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig_dow = go.Figure(go.Bar(
                x=day_names, y=dow_avg.values,
                marker_color=["#bfa085" if i in [5, 6] else "#4d3c32" for i in range(7)],
                text=dow_avg.values.round(0).astype(int),
                textposition="outside",
            ))
            themed(fig_dow)
            fig_dow.update_layout(
                title="Average Demand by Day of Week",
                yaxis_title="Units / Day", showlegend=False,
            )
            st.plotly_chart(fig_dow, use_container_width=True, key="dow_chart")

        with col_month:
            month_avg = daily.groupby("month")["total_sales"].mean()
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            colours = ["#8c6c53"] * 12
            colours[10] = "#bfa085"
            colours[11] = "#e6cfb3"
            fig_month = go.Figure(go.Bar(
                x=month_names, y=month_avg.values,
                marker_color=colours,
                text=month_avg.values.round(0).astype(int),
                textposition="outside",
            ))
            themed(fig_month)
            fig_month.update_layout(
                title="Average Demand by Month",
                yaxis_title="Units / Day", showlegend=False,
            )
            st.plotly_chart(fig_month, use_container_width=True, key="month_chart")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecasts:
    st.markdown("""
    <div class="sec-header">
        <h3>Forecast Explorer</h3>
        <p>Interactive forecast visualization and residual error breakdown</p>
    </div>
    """, unsafe_allow_html=True)

    all_models = ["SARIMA", "Prophet", "LightGBM", "TFT", "Chronos-2"]

    ctrl_col, _ = st.columns([1, 2])
    with ctrl_col:
        selected_models = st.multiselect(
            "Select Models to Overlay",
            all_models,
            default=["LightGBM", "SARIMA"],
            key="forecast_model_select",
        )
        show_ci = st.toggle("Show Confidence Bands (95% CI)", value=True, key="show_ci_toggle")

    if not selected_models:
        st.info("Select at least one model backend from the dropdown above.")
    else:
        ref_df = load_forecast(selected_models[0])
        if ref_df is None:
            _no_data()
        else:
            fig_fc = go.Figure()

            # Actual demand line
            fig_fc.add_trace(go.Scatter(
                x=ref_df["date"], y=ref_df["actual"],
                mode="lines+markers",
                name="Actual Demand",
                line=dict(color="#e6cfb3", width=2.5),
                marker=dict(size=4, color="#e6cfb3"),
            ))

            for model in selected_models:
                df_m = load_forecast(model)
                if df_m is None:
                    continue
                colour = MODEL_COLOURS.get(model, "#bfa085")

                if show_ci and "ci_lower" in df_m.columns:
                    _rgb = tuple(int(colour.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    _fill = f"rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.1)"
                    fig_fc.add_trace(go.Scatter(
                        x=pd.concat([df_m["date"], df_m["date"].iloc[::-1]]),
                        y=pd.concat([df_m["ci_upper"], df_m["ci_lower"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=_fill,
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        name=f"{model} CI",
                        hoverinfo="skip",
                    ))

                fig_fc.add_trace(go.Scatter(
                    x=df_m["date"], y=df_m["predicted"],
                    mode="lines",
                    name=model,
                    line=dict(color=colour, width=2, dash="dot"),
                ))

            themed(fig_fc)
            fig_fc.update_layout(
                title="28-Day Forecast Horizon (Hold-Out Test Period)",
                xaxis_title="Date",
                yaxis_title="Aggregated Unit Sales",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0),
                height=460,
            )
            st.plotly_chart(fig_fc, use_container_width=True, key="main_forecast_chart")

            # Per-model metrics breakdown
            st.markdown("##### Metric Breakdown for Selected Models")
            comp = load_comparison()
            metric_cols_fc = st.columns(len(selected_models))
            for col, model in zip(metric_cols_fc, selected_models):
                if comp is not None:
                    row = comp[comp["Model"] == model]
                    if not row.empty:
                        r = row.iloc[0]
                        colour = MODEL_COLOURS.get(model, "#bfa085")
                        col.markdown(f"""
                        <div style="background:#222222;border:1px solid {colour}40;
                                    border-radius:6px;padding:0.9rem;text-align:left;">
                            <div style="font-size:0.75rem;color:#9ca3af;text-transform:uppercase;
                                        letter-spacing:0.8px;">{model}</div>
                            <div style="font-size:1.5rem;font-weight:700;color:{colour};margin-top:0.2rem;">{r['RMSE']:.2f} <span style="font-size:0.75rem;color:#9ca3af;font-weight:400;">RMSE</span></div>
                            <div style="margin-top:0.4rem;font-size:0.8rem;color:#9ca3af;">
                                MAE: {r['MAE']:.2f} | MAPE: {r['MAPE']:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("""
    <div class="sec-header">
        <h3>Model Evaluation & Benchmarks</h3>
        <p>Comparative accuracy benchmarks and training compute efficiency</p>
    </div>
    """, unsafe_allow_html=True)

    comp_df = load_comparison()
    if comp_df is None:
        _no_data()
    else:
        df_s = comp_df.sort_values("RMSE").reset_index(drop=True)

        metric_cols = ["RMSE", "MAE", "MAPE", "sMAPE", "WRMSSE"]
        available_metrics = [c for c in metric_cols if c in df_s.columns]
        best_vals = {c: df_s[c].min() for c in available_metrics}

        rows_html = ""
        for i, row in df_s.iterrows():
            rank_str = f"#{i+1}"
            colour = MODEL_COLOURS.get(row["Model"], "#bfa085")
            cells = f"<td class='model-name'><span style='color:{colour};font-weight:700;'>{rank_str}</span> {row['Model']}</td>"
            for mc in available_metrics:
                v = row[mc]
                cls = "best" if abs(v - best_vals[mc]) < 0.001 else ""
                cells += f"<td class='{cls}'>{v:.2f}</td>"
            if "Training Time (s)" in df_s.columns:
                cells += f"<td>{row['Training Time (s)']:.0f}s</td>"
            rows_html += f"<tr>{cells}</tr>"

        header_cols = ["Model"] + available_metrics + (["Train Time"] if "Training Time (s)" in df_s.columns else [])
        header_html = "".join(f"<th>{c}</th>" for c in header_cols)

        st.markdown(f"""
        <div style="overflow-x:auto;margin-bottom:1.5rem;">
        <table class="lb-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart + Radar
        col_bar, col_radar = st.columns(2)

        with col_bar:
            colours_bar = [MODEL_COLOURS.get(m, "#bfa085") for m in df_s["Model"]]
            fig_bar = go.Figure(go.Bar(
                x=df_s["RMSE"], y=df_s["Model"],
                orientation="h",
                marker_color=colours_bar,
                text=[f"{v:.2f}" for v in df_s["RMSE"]],
                textposition="outside",
            ))
            themed(fig_bar)
            fig_bar.update_layout(
                title="RMSE by Model (Lower is Better)",
                xaxis_title="RMSE (Units / Day)",
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                height=360,
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="rmse_bar")

        with col_radar:
            radar_metrics = [c for c in ["RMSE", "MAE", "MAPE", "sMAPE", "WRMSSE"] if c in df_s.columns]
            fig_radar = go.Figure()
            for _, row in df_s.iterrows():
                vals = []
                for mc in radar_metrics:
                    col_max = df_s[mc].max()
                    col_min = df_s[mc].min()
                    score = 1 - (row[mc] - col_min) / (col_max - col_min) if col_max > col_min else 1.0
                    vals.append(round(score, 3))
                vals += vals[:1]
                colour = MODEL_COLOURS.get(row["Model"], "#bfa085")
                rgb = tuple(int(colour.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=row["Model"],
                    fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.1)",
                    line=dict(color=colour, width=2),
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#222222",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.07)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                ),
                title="Normalized Performance Radar (Outer Boundary = Superior)",
                paper_bgcolor="#222222",
                font=dict(family="Inter, sans-serif", color="#e0e0e0"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c0c0c0")),
                height=360,
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("""
    <div class="sec-header">
        <h3>Model Interpretability & Insights</h3>
        <p>Feature importances, SHAP values, attention weights, and residual diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    ins_tab1, ins_tab2, ins_tab3, ins_tab4 = st.tabs([
        "LightGBM Interpretability", "TFT Attention Weights", "Chronos-2 Evaluation", "SARIMA & Prophet Diagnostics"
    ])

    with ins_tab1:
        fi_df = load_feature_importance()
        shap_data = load_shap()

        if fi_df is None:
            _no_data()
        else:
            col_fi, col_shap = st.columns(2)
            with col_fi:
                fi_top = fi_df.head(15).sort_values("importance")
                fig_fi = go.Figure(go.Bar(
                    x=fi_top["importance"], y=fi_top["feature"],
                    orientation="h",
                    marker_color="#10b981",
                ))
                themed(fig_fi)
                fig_fi.update_layout(
                    title="Feature Importance (Split Gain)",
                    xaxis_title="Gain Score",
                    showlegend=False, height=450,
                )
                st.plotly_chart(fig_fi, use_container_width=True, key="fi_chart")

            with col_shap:
                if shap_data is not None:
                    shap_vals = shap_data["values"]
                    feature_data = shap_data["data"]
                    feature_names = shap_data["feature_names"]

                    fig_shap = go.Figure()
                    n_feats = len(feature_names)
                    for i, feat in enumerate(feature_names):
                        sv = shap_vals[:, i]
                        fd = feature_data[:, i]
                        fd_norm = (fd - fd.min()) / (fd.max() - fd.min() + 1e-8)
                        colours_shap = [
                            f"rgba({int(255*v)},{int(100*(1-v))},{int(150*(1-v))},0.7)"
                            for v in fd_norm
                        ]
                        fig_shap.add_trace(go.Scatter(
                            x=sv,
                            y=[i + np.random.uniform(-0.25, 0.25) for _ in sv],
                            mode="markers",
                            name=feat,
                            marker=dict(color=colours_shap, size=5, opacity=0.7),
                            showlegend=False,
                        ))
                    fig_shap.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    themed(fig_shap)
                    fig_shap.update_layout(
                        title="SHAP Value Distribution",
                        xaxis_title="Impact on Model Output (SHAP Value)",
                        yaxis=dict(tickvals=list(range(n_feats)), ticktext=feature_names),
                        height=450,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

        st.markdown("""
        <div class="info-box">
            <h4>Feature Drivers Analysis</h4>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.8;">
                <li><b>lag_7</b> exhibits the highest importance gain, reflecting strong weekly sales autocorrelation.</li>
                <li><b>lag_14 & lag_28</b> capture monthly cyclical patterns and pay-period demand spikes.</li>
                <li><b>rolling_mean_7 & rolling_mean_28</b> smooth short-term demand variance across all categories.</li>
                <li><b>sell_price & price ratio</b> features capture price elasticity and promotional discount response.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with ins_tab2:
        attn_df = load_tft_attention()
        vs_df = load_tft_variable_selection()

        if attn_df is None:
            _no_data()
        else:
            col_attn, col_vs = st.columns(2)

            with col_attn:
                fig_attn = go.Figure(go.Bar(
                    x=attn_df["horizon"], y=attn_df["weight"],
                    marker_color="#3b82f6",
                    name="Attention Weight",
                ))
                themed(fig_attn)
                fig_attn.update_layout(
                    title="Temporal Attention Across Forecast Horizon",
                    xaxis_title="Days Ahead", yaxis_title="Attention Weight",
                    showlegend=False,
                )
                st.plotly_chart(fig_attn, use_container_width=True, key="attn_chart")

            with col_vs:
                if vs_df is not None:
                    vs_sorted = vs_df.sort_values("weight", ascending=True)
                    fig_vs = go.Figure(go.Bar(
                        x=vs_sorted["weight"], y=vs_sorted["feature"],
                        orientation="h",
                        marker_color="#60a5fa",
                        text=[f"{w:.0%}" for w in vs_sorted["weight"]],
                        textposition="outside",
                    ))
                    themed(fig_vs)
                    fig_vs.update_layout(
                        title="TFT Variable Selection Network Weights",
                        xaxis_title="Weight", showlegend=False,
                    )
                    st.plotly_chart(fig_vs, use_container_width=True, key="vs_chart")

    with ins_tab3:
        ch_df = load_chronos_analysis()

        if ch_df is None:
            _no_data()
        else:
            st.markdown("""
            <div class="info-box">
                <h4>Chronos-2 Foundation Model Analysis</h4>
                <p style="margin:0;color:#9ca3af;">
                Evaluates Amazon Chronos-2 zero-shot inference directly on raw time-series tokens without fine-tuning or feature engineering.
                </p>
            </div>
            """, unsafe_allow_html=True)

            metrics_rows = ch_df.set_index("Metric")
            model_cols = [c for c in ch_df.columns if c != "Metric"]
            colours_ch = {"Chronos-2 Zero-Shot": "#8b5cf6", "LightGBM (Best)": "#10b981", "SARIMA (Baseline)": "#8c6c53"}

            fig_ch = go.Figure()
            for col in model_cols:
                try:
                    vals = pd.to_numeric(metrics_rows[col], errors="coerce")
                    valid_mask = vals.notna()
                    fig_ch.add_trace(go.Bar(
                        name=col,
                        x=metrics_rows.index[valid_mask],
                        y=vals[valid_mask],
                        marker_color=colours_ch.get(col, "#bfa085"),
                    ))
                except Exception:
                    pass

            themed(fig_ch)
            fig_ch.update_layout(
                title="Zero-Shot Foundation Model vs Supervised Baselines",
                xaxis_title="Metric", yaxis_title="Value", barmode="group",
            )
            st.plotly_chart(fig_ch, use_container_width=True, key="chronos_chart")

    with ins_tab4:
        col_sarima, col_prophet = st.columns(2)

        with col_sarima:
            acf_df = load_sarima_acf()
            if acf_df is not None:
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(
                    x=acf_df["lag"], y=acf_df["acf"],
                    name="ACF", marker_color="#8c6c53",
                ))
                fig_acf.add_trace(go.Scatter(
                    x=acf_df["lag"], y=acf_df["pacf"],
                    mode="lines+markers",
                    name="PACF",
                    line=dict(color="#bfa085", width=2),
                ))
                sig = 1.96 / np.sqrt(1913)
                fig_acf.add_hline(y=sig, line_dash="dash", line_color="#10b981", opacity=0.5)
                fig_acf.add_hline(y=-sig, line_dash="dash", line_color="#10b981", opacity=0.5)
                themed(fig_acf)
                fig_acf.update_layout(
                    title="SARIMA Autocorrelation (ACF / PACF)",
                    xaxis_title="Lag", yaxis_title="Correlation",
                )
                st.plotly_chart(fig_acf, use_container_width=True, key="acf_chart")

        with col_prophet:
            prophet_comp = load_prophet_components()
            if prophet_comp is not None:
                fig_prop = go.Figure()
                for col, colour, name in [
                    ("trend", "#bfa085", "Trend"),
                    ("weekly", "#3b82f6", "Weekly"),
                    ("yearly", "#10b981", "Yearly"),
                ]:
                    if col in prophet_comp.columns:
                        fig_prop.add_trace(go.Scatter(
                            x=prophet_comp["ds"], y=prophet_comp[col],
                            mode="lines", name=name,
                            line=dict(color=colour, width=2),
                        ))
                themed(fig_prop)
                fig_prop.update_layout(
                    title="Prophet Seasonality Decomposition",
                    xaxis_title="Date", yaxis_title="Component Value",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_prop, use_container_width=True, key="prophet_chart")
