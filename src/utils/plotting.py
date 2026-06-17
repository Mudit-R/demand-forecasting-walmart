"""
Plotly Visualization Utilities
==============================

Production-quality interactive charts for demand-forecasting analysis.
All functions return :class:`plotly.graph_objects.Figure` instances styled
with a consistent **dark theme** and vibrant accent palette.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Theme constants ─────────────────────────────────────────────────────────
_BG = "#0E1117"
_PAPER = "#0E1117"
_GRID = "#1E2530"
_TEXT = "#FAFAFA"
_FONT = "Inter, Segoe UI, sans-serif"

_PALETTE = [
    "#00D4FF",  # cyan
    "#FF6B6B",  # coral
    "#51CF66",  # green
    "#FCC419",  # yellow
    "#CC5DE8",  # purple
    "#FF922B",  # orange
    "#20C997",  # teal
    "#748FFC",  # indigo
]


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_TEXT, size=13),
        title_font_size=18,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=_GRID,
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 1.  Forecast chart
# ════════════════════════════════════════════════════════════════════════════


def plot_forecast(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    title: str = "Demand Forecast vs Actuals",
    dates: Optional[pd.DatetimeIndex] = None,
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> go.Figure:
    """Interactive forecast vs actuals line chart.

    Parameters
    ----------
    actual : array-like
        Ground-truth values.
    predicted : array-like
        Forecasted values (same length as *actual*).
    title : str
        Chart title.
    dates : pd.DatetimeIndex | None
        X-axis dates.  Falls back to integer index if ``None``.
    confidence_intervals : tuple[lower, upper] | None
        Lower and upper bounds for the prediction interval.

    Returns
    -------
    go.Figure
    """
    x = dates if dates is not None else np.arange(len(actual))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(actual),
            name="Actual",
            mode="lines",
            line=dict(color=_PALETTE[0], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(predicted),
            name="Forecast",
            mode="lines",
            line=dict(color=_PALETTE[1], width=2, dash="dash"),
        )
    )

    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255,107,107,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
    )
    return _apply_theme(fig)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Seasonality decomposition
# ════════════════════════════════════════════════════════════════════════════


def plot_seasonality(
    df: pd.DataFrame,
    column: str = "sales",
    date_col: str = "date",
    period: int = 7,
) -> go.Figure:
    """Visualise weekly and monthly seasonality patterns.

    Creates a 2×1 subplot:
    * **Top** — average *column* by day-of-week.
    * **Bottom** — average *column* by month.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *date_col* (datetime) and *column*.
    column : str
        Numeric column to decompose (default ``"sales"``).
    date_col : str
        Date column (default ``"date"``).
    period : int
        Ignored in the current implementation (reserved for STL).

    Returns
    -------
    go.Figure
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    by_dow = df.groupby(df[date_col].dt.dayofweek)[column].mean()
    by_month = df.groupby(df[date_col].dt.month)[column].mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Average Sales by Day of Week", "Average Sales by Month"],
        vertical_spacing=0.18,
    )

    fig.add_trace(
        go.Bar(
            x=[day_names[i] for i in by_dow.index],
            y=by_dow.values,
            marker_color=_PALETTE[0],
            name="Day of Week",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[month_names[i - 1] for i in by_month.index],
            y=by_month.values,
            marker_color=_PALETTE[2],
            name="Month",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title="Seasonality Analysis", height=600)
    return _apply_theme(fig)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Feature importance
# ════════════════════════════════════════════════════════════════════════════


def plot_feature_importance(
    importances: np.ndarray | list,
    feature_names: Sequence[str],
    top_n: int = 20,
    title: str = "Top Feature Importances",
) -> go.Figure:
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : array-like
        Importance scores (same length as *feature_names*).
    feature_names : sequence[str]
        Feature labels.
    top_n : int
        Number of features to display (default 20).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.nlargest(top_n, "importance").sort_values("importance")

    fig = go.Figure(
        go.Bar(
            x=fi["importance"],
            y=fi["feature"],
            orientation="h",
            marker=dict(
                color=fi["importance"],
                colorscale=[[0, _PALETTE[4]], [1, _PALETTE[0]]],
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="",
        height=max(400, top_n * 25),
    )
    return _apply_theme(fig)


# ════════════════════════════════════════════════════════════════════════════
# 4.  Model comparison
# ════════════════════════════════════════════════════════════════════════════


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    title: str = "Model Comparison",
) -> go.Figure:
    """Grouped bar chart comparing models across metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Rows = models (index), columns = metric names.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, metric in enumerate(metrics_df.columns):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                marker_color=_PALETTE[i % len(_PALETTE)],
            )
        )

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Model",
        yaxis_title="Metric Value",
        legend_title="Metric",
    )
    return _apply_theme(fig)


# ════════════════════════════════════════════════════════════════════════════
# 5.  Residual analysis
# ════════════════════════════════════════════════════════════════════════════


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
) -> go.Figure:
    """Residual distribution histogram + residuals-vs-fitted scatter.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Residual Distribution", "Residuals vs Fitted"],
        horizontal_spacing=0.12,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=50,
            marker_color=_PALETTE[0],
            opacity=0.8,
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Residuals vs fitted
    fig.add_trace(
        go.Scatter(
            x=np.asarray(y_pred),
            y=residuals,
            mode="markers",
            marker=dict(color=_PALETTE[2], size=4, opacity=0.6),
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=_PALETTE[1], row=1, col=2)

    fig.update_layout(title=title, height=400)
    fig.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Fitted Value", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    return _apply_theme(fig)


# ════════════════════════════════════════════════════════════════════════════
# 6.  Correlation heatmap
# ════════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Interactive correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    columns : list[str] | None
        Subset of numeric columns to include.  If ``None``, all numeric
        columns are used.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    if columns is not None:
        numeric = df[columns].select_dtypes(include="number")
    else:
        numeric = df.select_dtypes(include="number")

    corr = numeric.corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, _PALETTE[1]],   # negative → coral
                [0.5, _BG],           # zero → background
                [1.0, _PALETTE[0]],   # positive → cyan
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=10,
        )
    )

    size = max(400, len(corr.columns) * 35)
    fig.update_layout(
        title=title,
        width=size,
        height=size,
        xaxis=dict(tickangle=45),
    )
    return _apply_theme(fig)
