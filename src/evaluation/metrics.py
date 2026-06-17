"""
Forecast Evaluation Metrics
============================

Provides standard regression metrics plus the **WRMSSE** metric used in
the Walmart M5 competition.

All metric functions accept NumPy arrays and return plain floats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Individual metrics
# ════════════════════════════════════════════════════════════════════════════


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value (lower is better).
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE value (lower is better).
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Entries where ``y_true == 0`` are excluded to avoid division by zero.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAPE as a percentage (e.g. ``12.5`` means 12.5 %).
    """
    y_true, y_pred = _validate(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    .. math::
        \\text{sMAPE} = \\frac{100}{n} \\sum \\frac{|y - \\hat{y}|}
        {(|y| + |\\hat{y}|) / 2}

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        sMAPE as a percentage (0–200 scale; lower is better).
    """
    y_true, y_pred = _validate(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    train_series: Optional[np.ndarray] = None,
) -> float:
    """Weighted Root Mean Squared Scaled Error (M5 competition metric).

    For a single series the RMSSE is:

    .. math::
        \\text{RMSSE} = \\sqrt{\\frac{\\text{MSE}(y, \\hat{y})}
        {\\frac{1}{n-1} \\sum_{t=2}^{n} (y_t - y_{t-1})^2}}

    If *weights* are supplied the final metric is a weighted average of
    per-series RMSSE values.  When called on a single series (the common
    case in this project), the weight is implicitly 1.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values in the evaluation period.
    y_pred : np.ndarray
        Predicted values.
    weights : np.ndarray | None
        Per-series dollar-sales weights.  If ``None``, equal weight is used.
    train_series : np.ndarray | None
        Historical training values used to compute the scaling denominator.
        If ``None``, the denominator is estimated from ``y_true`` itself.

    Returns
    -------
    float
        WRMSSE value (lower is better).
    """
    y_true, y_pred = _validate(y_true, y_pred)

    # Scaling denominator
    if train_series is not None:
        diffs = np.diff(train_series.astype(np.float64))
    else:
        diffs = np.diff(y_true.astype(np.float64))

    scale = np.mean(diffs ** 2) if len(diffs) > 0 else 1.0
    scale = max(scale, 1e-8)  # guard against zero

    mse_val = np.mean((y_true - y_pred) ** 2)
    rmsse_val = np.sqrt(mse_val / scale)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()  # normalise
        return float(np.sum(rmsse_val * weights))

    return float(rmsse_val)


# ════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ════════════════════════════════════════════════════════════════════════════


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_series: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all standard metrics and return as a dictionary.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    train_series : np.ndarray | None
        Training history for WRMSSE scaling.

    Returns
    -------
    dict[str, float]
        ``{"RMSE": …, "MAE": …, "MAPE": …, "sMAPE": …, "WRMSSE": …}``
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
        "WRMSSE": round(wrmsse(y_true, y_pred, train_series=train_series), 4),
    }


def compare_models(
    results: Dict[str, Dict[str, float]],
    sort_by: str = "RMSE",
    ascending: bool = True,
) -> pd.DataFrame:
    """Build a comparison table from per-model metric dictionaries.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of ``model_name → {metric_name: value}``.
        Example::

            {
                "SARIMA":   {"RMSE": 2.1, "MAE": 1.5, …},
                "LightGBM": {"RMSE": 1.8, "MAE": 1.2, …},
            }

    sort_by : str
        Column to sort by (default ``"RMSE"``).
    ascending : bool
        Sort order (default ascending = best on top).

    Returns
    -------
    pd.DataFrame
        Indexed by model name, one column per metric.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    # Round for readability
    df = df.round(4)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Internal validation
# ════════════════════════════════════════════════════════════════════════════


def _validate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to float64 arrays and validate shapes."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Empty arrays — cannot compute metrics.")

    return y_true, y_pred
