"""
Reusable chart & UI components for the Walmart M5 dashboard.

Provides:
    - Styled metric cards with glassmorphism CSS
    - A dark, vibrant Plotly template used across every chart
    - Standard forecast visualisation (actual vs predicted + CI)
    - Radar chart for multi‑model comparison
    - Colour‑highlighted leaderboard table
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── colour palette (NextChapter theme) ───────────────────────────────────────
PALETTE = {
    "purple": "#8c6c53",      # muted bronze
    "blue": "#bfa085",        # warm gold
    "teal": "#e6cfb3",        # pale sand
    "indigo": "#c5a48a",      # copper
    "deep_blue": "#4d3c32",   # dark bronze
    "bg": "#222222",          # warm dark charcoal
    "card_bg": "#282828",     # card charcoal
    "text": "#e0e0e0",
    "accent_gradient": "#bfa085",
}

DATA_COLOURS = [
    "#bfa085",  # gold
    "#e6cfb3",  # sand
    "#c5a48a",  # copper
    "#8c6c53",  # bronze
    "#4d3c32",  # dark bronze
    "#34d399",  # green
    "#ff6b6b",  # red
    "#764ba2",  # purple
]


# ─────────────────────────── metric card ────────────────────────────────────
def create_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "",
) -> str:
    """Return HTML for a single glassmorphism metric card."""
    delta_html = ""
    if delta is not None:
        is_positive = not delta.strip().startswith("-")
        colour = "#34d399" if is_positive else "#ff6b6b"
        arrow = "▲" if is_positive else "▼"
        delta_html = (
            f'<p style="margin:0;font-size:0.85rem;color:{colour};">'
            f"{arrow} {delta}</p>"
        )

    return f"""
    <div style="
        background: {PALETTE['card_bg']};
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    ">
        <p style="margin:0 0 0.3rem 0;font-size:1.5rem;">{icon}</p>
        <p style="margin:0;font-size:0.82rem;color:#9ca3af;
                  text-transform:uppercase;letter-spacing:1.2px;">
            {title}
        </p>
        <p style="margin:0.35rem 0 0 0;font-size:1.85rem;
                  font-weight:700;
                  font-family:'Playfair Display', Georgia, serif;
                  color:{PALETTE['blue']};">
            {value}
        </p>
        {delta_html}
    </div>
    """


# ──────────────────────── plotly template ───────────────────────────────────
def create_plotly_theme() -> go.layout.Template:
    """Create a consistent dark Plotly template with vibrant accent colours."""
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, sans-serif", color=PALETTE["text"]),
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor=PALETTE["bg"],
            colorway=DATA_COLOURS,
            title=dict(font=dict(size=18, color="#ffffff")),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.08)",
                showline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.08)",
                showline=False,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text"]),
            ),
            hoverlabel=dict(
                bgcolor="#1e1e3c",
                font_size=13,
                font_family="Inter, sans-serif",
            ),
            margin=dict(l=50, r=30, t=60, b=50),
        )
    )


_THEME = create_plotly_theme()


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the project‑wide dark theme to *fig* in‑place."""
    fig.update_layout(template=_THEME)
    return fig


# ──────────────────── forecast chart ────────────────────────────────────────
def create_forecast_chart(
    dates: Sequence,
    actual: Sequence[float],
    predicted: Sequence[float],
    ci_lower: Optional[Sequence[float]] = None,
    ci_upper: Optional[Sequence[float]] = None,
    title: str = "Forecast vs Actuals",
) -> go.Figure:
    """Standard forecast line chart with optional confidence interval band."""
    fig = go.Figure()

    # confidence interval
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=list(ci_upper) + list(ci_lower)[::-1],
            fill="toself",
            fillcolor="rgba(102,126,234,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True,
            name="95 % CI",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=list(dates), y=list(actual),
        mode="lines",
        name="Actual",
        line=dict(color=PALETTE["teal"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=list(dates), y=list(predicted),
        mode="lines",
        name="Predicted",
        line=dict(color=PALETTE["blue"], width=2, dash="dot"),
    ))

    _apply_theme(fig)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ──────────────────── radar / spider chart ──────────────────────────────────
def create_comparison_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[Sequence[str]] = None,
) -> go.Figure:
    """
    Radar chart comparing several models across multiple metrics.

    Parameters
    ----------
    metrics_dict : dict
        ``{model_name: {metric: value, ...}, ...}``
    metric_names : list[str], optional
        Explicit ordering of metrics on the radar axes.  Defaults to the
        union of all keys across models.
    """
    if metric_names is None:
        _names: set[str] = set()
        for m in metrics_dict.values():
            _names.update(m.keys())
        metric_names = sorted(_names)

    fig = go.Figure()

    for idx, (model, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        values += values[:1]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metric_names) + [metric_names[0]],
            fill="toself",
            name=model,
            fillcolor=f"rgba({_hex_to_rgb(DATA_COLOURS[idx % len(DATA_COLOURS)])},0.15)",
            line=dict(color=DATA_COLOURS[idx % len(DATA_COLOURS)], width=2),
        ))

    _apply_theme(fig)
    fig.update_layout(
        polar=dict(
            bgcolor=PALETTE["bg"],
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
        title="Model Comparison Radar",
        showlegend=True,
    )
    return fig


def _hex_to_rgb(hex_str: str) -> str:
    """Convert '#RRGGBB' → 'R,G,B'."""
    h = hex_str.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


# ──────────────────── leaderboard table ─────────────────────────────────────
def create_leaderboard_table(metrics_df: pd.DataFrame) -> str:
    """
    Return HTML for a styled leaderboard table.

    *metrics_df* must have a 'Model' column and numeric metric columns.
    The best (lowest) value in each metric column is highlighted green.
    """
    metric_cols = [c for c in metrics_df.columns if c != "Model"]

    # Determine the best (min) index per metric
    best: dict[str, int] = {}
    for col in metric_cols:
        series = pd.to_numeric(metrics_df[col], errors="coerce")
        if series.notna().any():
            best[col] = int(series.idxmin())

    header = "".join(f"<th>{c}</th>" for c in metrics_df.columns)
    rows = []
    for idx, row in metrics_df.iterrows():
        cells = []
        for col in metrics_df.columns:
            val = row[col]
            if col in best and best[col] == idx:
                cells.append(
                    f'<td style="color:#34d399;font-weight:700;">{val}</td>'
                )
            else:
                cells.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div style="overflow-x:auto;">
    <table style="
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        background:{PALETTE['card_bg']};
        backdrop-filter:blur(16px);
        border-radius:12px;
        overflow:hidden;
        font-size:0.92rem;
    ">
        <thead>
            <tr style="background:rgba(102,126,234,0.2);">
                {header}
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    </div>
    <style>
        table th, table td {{
            padding: 0.75rem 1rem;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        table tr:hover {{
            background: rgba(102,126,234,0.08);
        }}
    </style>
    """


# ──────────────────── small helper charts ───────────────────────────────────
def create_residuals_plot(
    dates: Sequence,
    residuals: Sequence[float],
    title: str = "Residuals",
) -> go.Figure:
    """Residual bar + zero‑line chart."""
    colours = [PALETTE["teal"] if r >= 0 else "#ff6b6b" for r in residuals]
    fig = go.Figure(go.Bar(
        x=list(dates), y=list(residuals),
        marker_color=colours,
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    _apply_theme(fig)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Error")
    return fig


def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header inside Streamlit."""
    sub = f'<p style="color:#9ca3af;margin:0;font-size:0.88rem;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin:2.5rem 0 1.2rem 0; border-left: 2px solid {PALETTE['blue']}; padding-left: 1rem;">
            <h3 style="margin:0;
                        font-family:'Playfair Display', Georgia, serif;
                        color:#ffffff;
                        font-size:1.4rem;
                        font-weight:600;">
                {title}
            </h3>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )
