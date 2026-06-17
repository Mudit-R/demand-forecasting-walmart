"""
Walmart M5 Demand Forecasting — Streamlit Dashboard
====================================================
Main entry point.  Run with:

    streamlit run dashboard/app.py

Uses Streamlit's built‑in multi‑page support (pages/ directory).
"""

from __future__ import annotations

import streamlit as st

# ── page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Walmart M5 Demand Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts (Inter + Playfair Display) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap');

    /* ── root variables (NextChapter palette) ── */
    :root {
        --purple: #8c6c53;
        --blue: #bfa085;
        --teal: #e6cfb3;
        --indigo: #c5a48a;
        --deep-blue: #4d3c32;
        --bg: #222222;
        --card-bg: #282828;
        --text: #e0e0e0;
        --border: rgba(255,255,255,0.08);
    }

    /* ── global resets ── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: var(--bg);
    }

    /* ── headings ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 600 !important;
    }

    /* ── sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1c1c 0%, #222222 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c4c4d4;
        font-size: 0.88rem;
    }

    /* ── hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(34,34,34,0.85);
        backdrop-filter: blur(12px);
    }

    /* ── smooth transitions on interactive widgets ── */
    .stSelectbox, .stMultiSelect, .stSlider, .stDateInput {
        transition: all 0.2s ease;
    }

    /* ── metric cards row ── */
    .metric-row {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }
    .metric-row > div {
        flex: 1 1 200px;
    }

    /* ── quick‑link cards ── */
    .ql-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.6rem;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        cursor: default;
    }
    .ql-card:hover {
        transform: translateY(-4px);
        border-color: var(--blue);
        box-shadow: 0 8px 30px rgba(191,160,133,0.12);
    }
    .ql-card h4 {
        font-family: 'Playfair Display', Georgia, serif;
        margin: 0.6rem 0 0.4rem 0;
        color: #ffffff;
        font-size: 1.2rem;
    }
    .ql-card p {
        margin: 0;
        color: #9ca3af;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ── hero section (warm editorial sand) ── */
    .hero {
        background: #f4ebe1;
        border: 1px solid #bfa085;
        border-radius: 8px;
        padding: 3rem 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 2.6rem;
        font-weight: 700 !important;
        color: #222222;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: #4e4e4e;
        font-size: 1.1rem;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 0.5rem 0;">
            <h2 style="margin:0.3rem 0 0 0;
                        background:linear-gradient(135deg,#bfa085,#8c6c53);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        font-weight:800;">
                Walmart M5
            </h2>
            <p style="margin:0;color:#9ca3af;font-size:0.82rem;letter-spacing:1px;">
                DEMAND FORECASTING
            </p>
        </div>
        <hr style="border-color:rgba(255,255,255,0.06);margin:1rem 0;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Dataset Info")
    st.markdown(
        """
        - **Products:** 30,490  
        - **Stores:** 10 (3 states)  
        - **Time span:** 1,941 days  
        - **Granularity:** Daily unit sales  
        """
    )

    st.markdown("---")
    st.markdown("##### Models")
    st.markdown(
        """
        1. SARIMA  
        2. Prophet  
        3. LightGBM  
        4. Temporal Fusion Transformer  
        5. Chronos‑2 (zero‑shot)  
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0;">
            <a href="https://github.com/Mudit-R" target="_blank"
               style="color:#bfa085;text-decoration:none;font-weight:600;">
                GitHub — Mudit‑R
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── hero section ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Walmart M5 Demand Forecasting</h1>
        <p>
            End‑to‑end demand forecasting across <b>30 K+ products</b>,
            <b>10 stores</b>, and <b>1,941 days</b> of sales history —
            powered by five state‑of‑the‑art models from classical
            statistics to foundation‑model zero‑shot inference.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── key metrics ─────────────────────────────────────────────────────────────
from dashboard.components.charts import create_metric_card  # noqa: E402

cols = st.columns(4)
cards = [
    ("Total Products", "30,490", ""),
    ("Stores", "10", ""),
    ("Time Range", "1,941 days", ""),
    ("Models", "5", ""),
]
for col, (title, value, icon) in zip(cols, cards):
    col.markdown(create_metric_card(title, value, icon=icon), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── project description ────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: #282828;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1.8rem 2rem;
        margin-bottom: 2rem;
        line-height: 1.75;
        color: #c4c4d4;
        font-size: 0.95rem;
    ">
        <h4 style="color:#fff;margin-top:0;font-family:'Playfair Display', Georgia, serif;">About the Project</h4>
        This project tackles the
        <a href="https://www.kaggle.com/c/m5-forecasting-accuracy"
           target="_blank" style="color:#bfa085;">
            Kaggle M5 Forecasting — Accuracy
        </a>
        competition.  We benchmark <b>five diverse approaches</b>:
        <ol>
            <li><b>SARIMA</b> — classical seasonal ARIMA with grid‑search tuning</li>
            <li><b>Prophet</b> — Facebook's additive model with holiday & event regressors</li>
            <li><b>LightGBM</b> — gradient‑boosted trees on rich lag / rolling features</li>
            <li><b>Temporal Fusion Transformer (TFT)</b> — attention‑based deep forecasting via NeuralForecast</li>
            <li><b>Chronos‑2</b> — Amazon's foundation model for zero‑shot time‑series prediction</li>
        </ol>
        Each model is evaluated on the same hold‑out window with RMSE, MAE, MAPE,
        and SMAPE, and results are compared on the <em>Model Comparison</em> page.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── quick links ─────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='color:#fff;margin-bottom:0.8rem;'>Explore</h3>",
    unsafe_allow_html=True,
)

link_cols = st.columns(4)
pages = [
    ("", "EDA", "Dive into sales distributions, seasonality, and correlations."),
    ("", "Forecasts", "Visualise model predictions vs actuals with confidence bands."),
    ("", "Model Comparison", "Leaderboard & radar chart across all five models."),
    ("", "Insights", "Feature importance, SHAP, attention weights & more."),
]
for col, (icon, name, desc) in zip(link_cols, pages):
    col.markdown(
        f"""
        <div class="ql-card">
            <span style="font-size:2rem;">{icon}</span>
            <h4>{name}</h4>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── footer ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:3rem 0 1rem 0;color:#4b5563;font-size:0.78rem;">
        Built by <a href="https://github.com/Mudit-R" target="_blank"
        style="color:#bfa085;">Mudit‑R</a> &nbsp;·&nbsp;
        Streamlit + Plotly + LightGBM + NeuralForecast + Chronos‑2
    </div>
    """,
    unsafe_allow_html=True,
)
