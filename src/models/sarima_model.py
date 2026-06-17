"""
SARIMA Forecaster
=================

Fits a Seasonal ARIMA model via :class:`statsmodels.tsa.statespace.SARIMAX`
on a **single** item-store time series.

Default seasonal order assumes daily data with weekly seasonality:
``(1, 1, 1)(1, 1, 1, 7)``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation.metrics import evaluate_all
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA forecaster for univariate demand series.

    Parameters
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)`` — non-seasonal ARIMA order.
    seasonal_order : tuple[int, int, int, int]
        ``(P, D, Q, s)`` — seasonal component.  ``s=7`` for weekly
        seasonality on daily data.
    enforce_stationarity : bool
        Passed to SARIMAX (default ``False``).
    enforce_invertibility : bool
        Passed to SARIMAX (default ``False``).

    Examples
    --------
    >>> model = SARIMAForecaster()
    >>> model.fit(train_series)
    >>> forecasts = model.predict(horizon=28)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        super().__init__(name="SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Set after fit
        self._model_fit = None
        self._train_series: Optional[pd.Series] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame | pd.Series,
        target: str = "sales",
        auto_order: bool = False,
        **kwargs: Any,
    ) -> "SARIMAForecaster":
        """Fit SARIMAX on a single univariate time series.

        Parameters
        ----------
        train_df : pd.DataFrame | pd.Series
            If a DataFrame, must contain ``date`` and *target* columns.
            If a Series, the index should be datetime.
        target : str
            Column name for the sales values (default ``"sales"``).
        auto_order : bool
            If ``True``, perform a simple grid search over a small set of
            ``(p, d, q)`` values and select by AIC.
        **kwargs
            Extra keyword arguments forwarded to ``SARIMAX.fit()``.

        Returns
        -------
        SARIMAForecaster
            ``self``
        """
        # Coerce to a datetime-indexed Series
        series = self._to_series(train_df, target)
        self._train_series = series

        if auto_order:
            self.order = self._auto_detect_order(series)
            logger.info("Auto-detected order: %s", self.order)

        logger.info(
            "Fitting SARIMA%s×%s on %d observations …",
            self.order,
            self.seasonal_order,
            len(series),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self._model_fit = model.fit(disp=False, **kwargs)

        logger.info("SARIMA fitted — AIC=%.2f", self._model_fit.aic)
        self.is_fitted = True
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        return_ci: bool = True,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forecast future values with optional confidence intervals.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast (default 28).
        return_ci : bool
            Include lower/upper confidence interval columns.
        alpha : float
            Significance level for CI (default 0.05 → 95 % CI).

        Returns
        -------
        pd.DataFrame
            Columns: ``yhat`` (point forecast), and optionally
            ``yhat_lower``, ``yhat_upper``.
        """
        self._check_fitted()

        forecast = self._model_fit.get_forecast(steps=horizon, alpha=alpha)
        yhat = forecast.predicted_mean

        result = pd.DataFrame({"yhat": yhat.values}, index=yhat.index)
        result["yhat"] = result["yhat"].clip(lower=0)  # demand ≥ 0

        if return_ci:
            ci = forecast.conf_int(alpha=alpha)
            result["yhat_lower"] = ci.iloc[:, 0].clip(lower=0).values
            result["yhat_upper"] = ci.iloc[:, 1].clip(lower=0).values

        return result

    # ── evaluate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame | pd.Series,
        target: str = "sales",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compare forecasts to actual test values.

        Parameters
        ----------
        test_df : pd.DataFrame | pd.Series
            Ground truth for the forecast horizon.
        target : str
            Column name (if DataFrame) for actual values.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics.
        """
        self._check_fitted()
        y_true = self._to_series(test_df, target).values
        horizon = len(y_true)
        preds = self.predict(horizon=horizon, return_ci=False)
        y_pred = preds["yhat"].values

        metrics = evaluate_all(y_true, y_pred)
        logger.info("SARIMA evaluation: %s", metrics)
        return metrics

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model_fit is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @staticmethod
    def _to_series(data: pd.DataFrame | pd.Series, target: str) -> pd.Series:
        """Convert input to a datetime-indexed :class:`pd.Series`."""
        if isinstance(data, pd.Series):
            series = data.copy()
        elif isinstance(data, pd.DataFrame):
            if "date" in data.columns:
                series = data.set_index("date")[target]
            else:
                series = data[target]
        else:
            raise TypeError(f"Expected DataFrame or Series, got {type(data).__name__}")

        if not pd.api.types.is_datetime64_any_dtype(series.index):
            series.index = pd.to_datetime(series.index)
        series = series.asfreq("D")
        series = series.fillna(0)
        return series

    def _auto_detect_order(
        self,
        series: pd.Series,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
    ) -> Tuple[int, int, int]:
        """Simple AIC grid search over (p, d, q) space.

        This is a lightweight alternative to ``pmdarima.auto_arima``.  For
        production workloads consider using ``pmdarima`` directly.
        """
        best_aic = np.inf
        best_order = self.order

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(
                                series,
                                order=(p, d, q),
                                seasonal_order=self.seasonal_order,
                                enforce_stationarity=self.enforce_stationarity,
                                enforce_invertibility=self.enforce_invertibility,
                            )
                            fit = model.fit(disp=False, maxiter=50)
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                    except Exception:
                        continue

        logger.info("AIC grid search best order: %s (AIC=%.2f)", best_order, best_aic)
        return best_order
