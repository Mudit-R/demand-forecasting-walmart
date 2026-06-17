"""
Prophet Forecaster
==================

Wraps :class:`prophet.Prophet` for single-item-store demand forecasting
with:

* US public holidays
* Weekly + yearly seasonality (auto-scaled)
* Custom regressors for SNAP days, events, and promotions
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)

# Suppress noisy Prophet / cmdstanpy logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecaster for univariate demand series.

    Parameters
    ----------
    growth : str
        ``"linear"`` (default) or ``"logistic"``.
    yearly_seasonality : bool | int | str
        Enable yearly seasonality (default ``True``).
    weekly_seasonality : bool | int | str
        Enable weekly seasonality (default ``True``).
    daily_seasonality : bool
        Disable daily seasonality for daily-level data (default ``False``).
    changepoint_prior_scale : float
        Flexibility of trend changepoints (default 0.05).
    seasonality_prior_scale : float
        Strength of seasonality (default 10.0).
    holidays_prior_scale : float
        Strength of holiday effects (default 10.0).

    Examples
    --------
    >>> model = ProphetForecaster()
    >>> model.fit(train_df)
    >>> forecasts = model.predict(horizon=28)
    """

    def __init__(
        self,
        growth: str = "linear",
        yearly_seasonality: bool | int | str = True,
        weekly_seasonality: bool | int | str = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
    ) -> None:
        super().__init__(name="Prophet")
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale

        # State
        self._model = None
        self._regressor_cols: List[str] = []
        self._train_df: Optional[pd.DataFrame] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str = "sales",
        regressors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "ProphetForecaster":
        """Fit Prophet on a single item-store time series.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain ``date`` (or ``ds``) and *target* (or ``y``)
            columns.  Optional regressor columns can be passed.
        target : str
            Column name for the target variable (default ``"sales"``).
        regressors : list[str] | None
            Additional regressor column names to add (e.g.
            ``["snap", "has_event", "sell_price"]``).  Columns must exist
            in *train_df*.
        **kwargs
            Extra keyword arguments forwarded to ``Prophet.fit()``.

        Returns
        -------
        ProphetForecaster
            ``self``
        """
        from prophet import Prophet

        # Prepare Prophet-compatible DataFrame (ds, y)
        pdf = self._prepare_prophet_df(train_df, target)

        # Auto-detect available regressor columns
        candidate_regressors = regressors or [
            "snap",
            "has_event",
            "sell_price",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
        self._regressor_cols = [c for c in candidate_regressors if c in pdf.columns]

        logger.info(
            "Fitting Prophet on %d observations, regressors=%s …",
            len(pdf),
            self._regressor_cols,
        )

        # Build model
        self._model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
        )

        # US holidays
        self._model.add_country_holidays(country_name="US")

        # Custom regressors
        for col in self._regressor_cols:
            self._model.add_regressor(col)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(pdf, **kwargs)

        self._train_df = pdf
        self.is_fitted = True
        logger.info("Prophet fitted successfully.")
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        future_regressors: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate forecasts with uncertainty intervals.

        Parameters
        ----------
        horizon : int
            Number of days to forecast (default 28).
        future_regressors : pd.DataFrame | None
            DataFrame with future values of custom regressors.  Must have
            a ``ds`` column and one column per regressor added during fit.
            If ``None``, regressors are filled with their training-set mean.

        Returns
        -------
        pd.DataFrame
            Columns: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """
        self._check_fitted()

        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        future = future.tail(horizon).reset_index(drop=True)

        # Fill regressors
        if self._regressor_cols:
            if future_regressors is not None:
                for col in self._regressor_cols:
                    if col in future_regressors.columns:
                        future[col] = future_regressors[col].values[:horizon]
            else:
                # Use training-set mean as placeholder
                for col in self._regressor_cols:
                    future[col] = self._train_df[col].mean()

        forecast = self._model.predict(future)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["yhat"] = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        return result.reset_index(drop=True)

    # ── evaluate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame,
        target: str = "sales",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate Prophet forecasts against ground truth.

        Parameters
        ----------
        test_df : pd.DataFrame
            Must contain ``date`` (or ``ds``) and *target* (or ``y``).
        target : str
            Column name for actual values.

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        self._check_fitted()
        pdf = self._prepare_prophet_df(test_df, target)
        y_true = pdf["y"].values
        horizon = len(y_true)

        # Build future regressors from test data
        future_reg = pdf if self._regressor_cols else None
        preds = self.predict(horizon=horizon, future_regressors=future_reg)
        y_pred = preds["yhat"].values

        metrics = evaluate_all(y_true, y_pred)
        logger.info("Prophet evaluation: %s", metrics)
        return metrics

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @staticmethod
    def _prepare_prophet_df(
        df: pd.DataFrame,
        target: str = "sales",
    ) -> pd.DataFrame:
        """Rename columns to Prophet's expected ``ds`` / ``y`` schema."""
        pdf = df.copy()

        # Date column → ds
        if "date" in pdf.columns and "ds" not in pdf.columns:
            pdf = pdf.rename(columns={"date": "ds"})
        if "ds" not in pdf.columns:
            raise ValueError(
                "DataFrame must contain a 'date' or 'ds' column."
            )
        pdf["ds"] = pd.to_datetime(pdf["ds"])

        # Target column → y
        if target in pdf.columns and "y" not in pdf.columns:
            pdf = pdf.rename(columns={target: "y"})
        if "y" not in pdf.columns:
            raise ValueError(
                f"DataFrame must contain a '{target}' or 'y' column."
            )

        return pdf
