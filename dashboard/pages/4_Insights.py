"""
🧠 Model Insights & Interpretability
======================================
Per‑model interpretability visualisations: feature importance, SHAP
values, attention weights, seasonality decomposition, ACF/PACF, and
zero‑shot analysis.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Insights · Walmart M5", page_icon=None, layout="wide")

from dashboard.components.charts import (  # noqa: E402
    DATA_COLOURS,
    PALETTE,
    create_metric_card,
    create_plotly_theme,
    section_header,
)

_THEME = create_plotly_theme()
PROJECT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

# ── title ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="background:linear-gradient(135deg,#bfa085,#8c6c53);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-weight:800;">
        Model Insights &amp; Interpretability
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color:#9ca3af;max-width:720px;line-height:1.6;">
    Explore what drives each model's predictions.  Select a model tab below to
    view feature importance, SHAP summaries, attention weights, seasonality
    components, or ACF / PACF diagnostics.
    </p>
    """,
    unsafe_allow_html=True,
)


# ── helper: styled placeholder ──────────────────────────────────────────────
def _placeholder(model: str, what: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:rgba(30,30,60,0.45);
            border:1px dashed rgba(102,126,234,0.35);
            border-radius:14px;
            padding:2.5rem;
            text-align:center;
            color:#9ca3af;
        ">
            <p style="font-size:1.2rem;margin:0;color:#ff6b6b;font-weight:600;">[Locked]</p>
            <p style="margin:0.5rem 0 0 0;font-size:0.95rem;">
                <b>{what}</b> for <b>{model}</b> not available yet.<br>
                Train the model and save artefacts to <code>models/</code>
                or <code>results/</code> to unlock this view.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── helper: safe pickle load ───────────────────────────────────────────────
def _load_pickle(path: pathlib.Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TABS — one per model family
# ─────────────────────────────────────────────────────────────────────────────
tab_lgb, tab_tft, tab_prophet, tab_sarima, tab_chronos = st.tabs(
    ["LightGBM", "TFT", "Prophet", "SARIMA", "Chronos‑2"]
)

# ── LightGBM ────────────────────────────────────────────────────────────────
with tab_lgb:
    section_header("LightGBM — Feature Importance", "Gain‑based importance from the trained booster")

    imp_path = RESULTS_DIR / "insights" / "lgb_feature_importance.csv"
    imp_df: Optional[pd.DataFrame] = None
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
    else:
        # Try loading model directly
        model_path = MODELS_DIR / "lightgbm_model.pkl"
        model = _load_pickle(model_path)
        if model is not None:
            try:
                imp_df = pd.DataFrame({
                    "feature": model.feature_name(),
                    "importance": model.feature_importance(importance_type="gain"),
                }).sort_values("importance", ascending=False).head(20)
            except Exception:
                pass

    if imp_df is not None and not imp_df.empty:
        imp_df = imp_df.sort_values("importance", ascending=True).tail(20)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                         color="importance",
                         color_continuous_scale=["#8c6c53", "#bfa085"],
                         template=_THEME, labels={"importance": "Gain", "feature": ""})
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        # Show demo importance
        demo_features = [
            "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", "sell_price",
            "day_of_week", "month", "snap_flag", "event_flag", "rolling_mean_28",
            "item_mean_sales", "store_mean_sales", "lag_14", "year", "dept_mean_sales",
        ]
        demo_imp = pd.DataFrame({
            "feature": demo_features,
            "importance": np.sort(np.random.default_rng(42).exponential(500, len(demo_features)))[::-1],
        }).sort_values("importance", ascending=True)
        st.info("Showing **demo** feature importance — train LightGBM to see real values.")
        fig_imp = px.bar(demo_imp, x="importance", y="feature", orientation="h",
                         color="importance",
                         color_continuous_scale=["#8c6c53", "#bfa085"],
                         template=_THEME, labels={"importance": "Gain", "feature": ""})
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    section_header("SHAP Summary", "Beeswarm plot of SHAP values for top features")

    shap_path = RESULTS_DIR / "insights" / "lgb_shap_values.pkl"
    shap_data = _load_pickle(shap_path)
    if shap_data is not None:
        try:
            import shap
            import matplotlib.pyplot as plt

            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_data["values"], shap_data.get("data"),
                              feature_names=shap_data.get("feature_names"),
                              show=False, plot_type="dot", max_display=15)
            ax.set_facecolor(PALETTE["bg"])
            fig_shap.patch.set_facecolor(PALETTE["bg"])
            st.pyplot(fig_shap)
        except Exception as exc:
            st.warning(f"Could not render SHAP plot: {exc}")
    else:
        _placeholder("LightGBM", "SHAP values")

# ── TFT ─────────────────────────────────────────────────────────────────────
with tab_tft:
    section_header("Temporal Fusion Transformer — Attention Weights",
                   "Interpretable multi‑horizon attention from the TFT model")

    attn_path = RESULTS_DIR / "insights" / "tft_attention_weights.csv"
    if attn_path.exists():
        attn = pd.read_csv(attn_path)
        if "horizon" in attn.columns and "weight" in attn.columns:
            fig_attn = px.bar(attn, x="horizon", y="weight",
                              color="weight",
                              color_continuous_scale=["#8c6c53", "#bfa085"],
                              template=_THEME,
                              labels={"horizon": "Forecast Horizon (days)", "weight": "Attention"})
            fig_attn.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.dataframe(attn, use_container_width=True)
    else:
        # Demo attention weights
        st.info("Showing **demo** attention weights — train TFT to see real values.")
        horizons = list(range(1, 29))
        weights = np.random.default_rng(7).dirichlet(np.ones(28))
        fig_attn = go.Figure(go.Bar(
            x=horizons, y=weights,
            marker=dict(
                color=weights,
                colorscale=[[0, "#8c6c53"], [1, "#bfa085"]],
            ),
        ))
        fig_attn.update_layout(
            template=_THEME,
            title="Attention Weights per Forecast Horizon",
            xaxis_title="Horizon (days)", yaxis_title="Attention Weight",
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    section_header("Variable Selection Weights",
                   "How much each input feature contributed to predictions")
    vs_path = RESULTS_DIR / "insights" / "tft_variable_selection.csv"
    if vs_path.exists():
        vs = pd.read_csv(vs_path).sort_values("weight", ascending=True).tail(15)
        fig_vs = px.bar(vs, x="weight", y="feature", orientation="h",
                        color="weight", color_continuous_scale=["#4d3c32", "#bfa085"],
                        template=_THEME, labels={"weight": "Selection Weight", "feature": ""})
        fig_vs.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_vs, use_container_width=True)
    else:
        _placeholder("TFT", "Variable selection weights")

# ── Prophet ─────────────────────────────────────────────────────────────────
with tab_prophet:
    section_header("Prophet — Seasonality Components",
                   "Weekly, yearly, and holiday effects learned by the model")

    prophet_comp = RESULTS_DIR / "insights" / "prophet_components.csv"
    if prophet_comp.exists():
        comp = pd.read_csv(prophet_comp, parse_dates=["ds"])
        comp_cols = [c for c in comp.columns if c not in ("ds",)]
        tabs_p = st.tabs([c.title() for c in comp_cols])
        for tab, col in zip(tabs_p, comp_cols):
            with tab:
                fig_p = px.line(comp, x="ds", y=col, template=_THEME,
                                labels={"ds": "Date", col: col.title()})
                fig_p.update_traces(line_color=PALETTE["teal"])
                st.plotly_chart(fig_p, use_container_width=True)
    else:
        # Demo weekly & yearly seasonality
        st.info("Showing **demo** seasonality — train Prophet to see real decomposition.")
        demo_tabs = st.tabs(["Weekly", "Yearly"])
        with demo_tabs[0]:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            vals = [0.8, 0.9, 1.0, 0.95, 1.2, 1.5, 1.3]
            fig_w = go.Figure(go.Scatter(x=days, y=vals, mode="lines+markers",
                                         line=dict(color=PALETTE["teal"], width=2.5),
                                         marker=dict(size=8)))
            fig_w.update_layout(template=_THEME, title="Weekly Seasonality",
                                yaxis_title="Effect Multiplier")
            st.plotly_chart(fig_w, use_container_width=True)
        with demo_tabs[1]:
            months = pd.date_range("2015-01-01", periods=365, freq="D")
            yearly = np.sin(2 * np.pi * np.arange(365) / 365) * 0.3 + 1
            fig_y = go.Figure(go.Scatter(x=months, y=yearly, mode="lines",
                                         line=dict(color=PALETTE["blue"], width=2)))
            fig_y.update_layout(template=_THEME, title="Yearly Seasonality",
                                yaxis_title="Effect Multiplier")
            st.plotly_chart(fig_y, use_container_width=True)

    st.markdown("---")
    section_header("Changepoints", "Structural breaks detected by Prophet")

    cp_path = RESULTS_DIR / "insights" / "prophet_changepoints.csv"
    if cp_path.exists():
        cp = pd.read_csv(cp_path, parse_dates=["ds"])
        st.dataframe(cp, use_container_width=True)
    else:
        _placeholder("Prophet", "Changepoint data")

# ── SARIMA ──────────────────────────────────────────────────────────────────
with tab_sarima:
    section_header("SARIMA — ACF & PACF", "Autocorrelation diagnostics of model residuals")

    acf_path = RESULTS_DIR / "insights" / "sarima_acf_pacf.csv"
    if acf_path.exists():
        acf_df = pd.read_csv(acf_path)
        acf_tabs = st.tabs(["ACF", "PACF"])
        for tab, col in zip(acf_tabs, ["acf", "pacf"]):
            with tab:
                if col in acf_df.columns:
                    fig_acf = go.Figure(go.Bar(
                        x=list(range(len(acf_df))), y=acf_df[col],
                        marker_color=PALETTE["blue"],
                    ))
                    # Significance band
                    n = len(acf_df)
                    ci = 1.96 / np.sqrt(n) if n > 0 else 0.2
                    fig_acf.add_hline(y=ci, line_dash="dash",
                                      line_color="rgba(255,255,255,0.3)")
                    fig_acf.add_hline(y=-ci, line_dash="dash",
                                      line_color="rgba(255,255,255,0.3)")
                    fig_acf.update_layout(template=_THEME,
                                          title=col.upper(),
                                          xaxis_title="Lag",
                                          yaxis_title="Correlation")
                    st.plotly_chart(fig_acf, use_container_width=True)
    else:
        # Demo ACF / PACF
        st.info("Showing **demo** ACF / PACF — train SARIMA to see real diagnostics.")
        rng = np.random.default_rng(99)
        lags = 40
        demo_acf = np.concatenate([[1.0], 0.7 * np.exp(-np.arange(1, lags) / 5) + rng.normal(0, 0.03, lags - 1)])
        demo_pacf = np.concatenate([[1.0], [0.65] + list(rng.normal(0, 0.06, lags - 2))])

        acf_tabs = st.tabs(["ACF", "PACF"])
        for tab, vals, name in zip(acf_tabs, [demo_acf, demo_pacf], ["ACF", "PACF"]):
            with tab:
                fig_acf = go.Figure(go.Bar(
                    x=list(range(len(vals))), y=vals,
                    marker_color=PALETTE["blue"],
                ))
                ci = 1.96 / np.sqrt(lags)
                fig_acf.add_hline(y=ci, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_acf.add_hline(y=-ci, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_acf.update_layout(template=_THEME, title=name,
                                      xaxis_title="Lag", yaxis_title="Correlation")
                st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("---")
    section_header("Model Summary", "SARIMA order and coefficient diagnostics")
    summary_path = RESULTS_DIR / "insights" / "sarima_summary.txt"
    if summary_path.exists():
        st.code(summary_path.read_text(), language="text")
    else:
        _placeholder("SARIMA", "Model summary statistics")

# ── Chronos‑2 ───────────────────────────────────────────────────────────────
with tab_chronos:
    section_header("Chronos‑2 — Zero‑Shot Performance Analysis",
                   "Foundation model inference without task‑specific training")

    st.markdown(
        """
        <div style="
            background:rgba(30,30,60,0.45);
            border:1px solid rgba(255,255,255,0.06);
            border-radius:14px;
            padding:1.5rem 1.8rem;
            margin-bottom:1.5rem;
            color:#c4c4d4;
            line-height:1.65;
        ">
        <b>Chronos‑2</b> is Amazon's pre‑trained time‑series foundation model.
        It performs <em>zero‑shot</em> forecasting — no fine‑tuning on M5 data.
        This tab compares its out‑of‑the‑box accuracy against domain‑tuned models
        to assess the viability of foundation models for retail demand forecasting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chronos_path = RESULTS_DIR / "insights" / "chronos_analysis.csv"
    if chronos_path.exists():
        ch = pd.read_csv(chronos_path)
        st.dataframe(ch, use_container_width=True)
    else:
        # Demo comparison cards
        st.info("Showing **demo** zero‑shot analysis — run Chronos‑2 inference to see real numbers.")

        c1, c2, c3 = st.columns(3)
        c1.markdown(create_metric_card("Zero‑Shot RMSE", "2.72", delta="vs LightGBM: +9.7%", icon=""),
                    unsafe_allow_html=True)
        c2.markdown(create_metric_card("Inference Time", "15 s", delta="6× faster than TFT", icon=""),
                    unsafe_allow_html=True)
        c3.markdown(create_metric_card("No Training Data", "✓", icon=""),
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Demo: per‑category comparison
        demo_cats = ["FOODS", "HOUSEHOLD", "HOBBIES"]
        demo_rmse_lgb = [2.31, 2.65, 2.48]
        demo_rmse_chr = [2.55, 2.89, 2.72]
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(name="LightGBM", x=demo_cats, y=demo_rmse_lgb,
                                marker_color=PALETTE["blue"]))
        fig_ch.add_trace(go.Bar(name="Chronos‑2", x=demo_cats, y=demo_rmse_chr,
                                marker_color=PALETTE["teal"]))
        fig_ch.update_layout(template=_THEME, barmode="group",
                             title="RMSE by Category: LightGBM vs Chronos‑2",
                             yaxis_title="RMSE")
        st.plotly_chart(fig_ch, use_container_width=True)

# ── page footer ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 0.5rem 0;color:#4b5563;font-size:0.78rem;">
        Interpretability artefacts are generated during model training.
        Re‑run the pipeline to refresh.
    </div>
    """,
    unsafe_allow_html=True,
)
