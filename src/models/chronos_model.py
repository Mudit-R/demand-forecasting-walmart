"""
Chronos-2 Zero-Shot Forecaster
==============================

Uses Amazon's pre-trained `Chronos-2 <https://github.com/amazon-science/chronos-forecasting>`_
foundation model for **zero-shot** time-series forecasting — no task-specific
training required.

Supported model sizes: ``tiny``, ``small``, ``base``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import evaluate_all
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)

# Map friendly size names → HuggingFace model IDs
_MODEL_MAP = {
    "tiny": "amazon/chronos-t5-tiny",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
}


class ChronosForecaster(BaseForecaster):
    """Zero-shot demand forecaster powered by Chronos-2.

    Because Chronos is a *pre-trained* foundation model, :meth:`fit` merely
    stores the reference time series — no gradient updates are performed.
    All the heavy lifting happens in :meth:`predict`, which calls the
    Chronos pipeline for probabilistic forecasting.

    Parameters
    ----------
    model_size : ``"tiny"`` | ``"small"`` | ``"base"``
        Which Chronos checkpoint to load (default ``"small"``).
    device : str
        PyTorch device string (default ``"cpu"``).  Use ``"cuda"`` for GPU.
    num_samples : int
        Number of forecast sample paths for probabilistic prediction
        (default 20).
    temperature : float | None
        Sampling temperature.  ``None`` uses the model default.

    Examples
    --------
    >>> model = ChronosForecaster(model_size="small")
    >>> model.fit(train_series)
    >>> preds = model.predict(horizon=28)
    """

    def __init__(
        self,
        model_size: Literal["tiny", "small", "base"] = "small",
        device: str = "cpu",
        num_samples: int = 20,
        temperature: Optional[float] = None,
    ) -> None:
        super().__init__(name="Chronos")
        if model_size not in _MODEL_MAP:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from {list(_MODEL_MAP.keys())}."
            )
        self.model_size = model_size
        self.model_id = _MODEL_MAP[model_size]
        self.device = device
        self.num_samples = num_samples
        self.temperature = temperature

        # State
        self._pipeline = None
        self._context: Optional[torch.Tensor] = None
        self._train_dates: Optional[pd.DatetimeIndex] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame | pd.Series,
        target: str = "sales",
        **kwargs: Any,
    ) -> "ChronosForecaster":
        """Store the reference time series and lazily load the Chronos pipeline.

        No model training occurs — Chronos is a zero-shot model.

        Parameters
        ----------
        train_df : pd.DataFrame | pd.Series
            If a DataFrame, must contain ``date`` and *target* columns.
            If a Series, the index should be datetime.
        target : str
            Column name for the sales values (default ``"sales"``).

        Returns
        -------
        ChronosForecaster
            ``self``
        """
        series, dates = self._extract_series(train_df, target)
        self._context = torch.tensor(series, dtype=torch.float32)
        self._train_dates = dates

        # Lazy-load the pipeline
        if self._pipeline is None:
            self._load_pipeline()

        self.is_fitted = True
        logger.info(
            "Chronos '%s' ready — context length=%d",
            self.model_size,
            len(series),
        )
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        return_samples: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate zero-shot probabilistic forecasts.

        Parameters
        ----------
        horizon : int
            Number of future time steps (default 28).
        return_samples : bool
            If ``True``, include individual sample paths in the output.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
            If *return_samples* is ``True``, additional columns
            ``sample_0`` … ``sample_{N-1}`` are appended.
        """
        self._check_fitted()

        logger.info("Chronos predicting %d steps (samples=%d) …", horizon, self.num_samples)

        predict_kwargs: Dict[str, Any] = {
            "context": self._context.unsqueeze(0),
            "prediction_length": horizon,
            "num_samples": self.num_samples,
        }
        if self.temperature is not None:
            predict_kwargs["temperature"] = self.temperature

        # samples shape: (1, num_samples, horizon)
        samples = self._pipeline.predict(**predict_kwargs)
        samples = samples.squeeze(0).numpy()  # (num_samples, horizon)

        # Point forecast = median; intervals = quantiles
        yhat = np.median(samples, axis=0)
        yhat_lower = np.quantile(samples, 0.025, axis=0)
        yhat_upper = np.quantile(samples, 0.975, axis=0)

        # Build date index
        if self._train_dates is not None and len(self._train_dates) > 0:
            last_date = self._train_dates[-1]
            dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        else:
            dates = pd.RangeIndex(horizon)

        result = pd.DataFrame(
            {
                "date": dates,
                "yhat": np.clip(yhat, 0, None),
                "yhat_lower": np.clip(yhat_lower, 0, None),
                "yhat_upper": np.clip(yhat_upper, 0, None),
            }
        )

        if return_samples:
            for i in range(samples.shape[0]):
                result[f"sample_{i}"] = np.clip(samples[i], 0, None)

        return result

    # ── evaluate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame | pd.Series,
        target: str = "sales",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate Chronos forecasts against actual values.

        Parameters
        ----------
        test_df : pd.DataFrame | pd.Series
            Ground truth for the forecast horizon.
        target : str
            Column name for actuals (if DataFrame).

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        self._check_fitted()
        y_true, _ = self._extract_series(test_df, target)
        horizon = len(y_true)

        preds = self.predict(horizon=horizon)
        y_pred = preds["yhat"].values

        metrics = evaluate_all(y_true, y_pred)
        logger.info("Chronos evaluation: %s", metrics)
        return metrics

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._context is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    def _load_pipeline(self) -> None:
        """Load the Chronos pipeline from HuggingFace."""
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos package not found. Install with:\n"
                "  pip install chronos-forecasting"
            ) from exc

        logger.info("Loading Chronos pipeline '%s' on %s …", self.model_id, self.device)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        logger.info("Chronos pipeline loaded.")

    @staticmethod
    def _extract_series(
        data: pd.DataFrame | pd.Series,
        target: str,
    ) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """Extract a NumPy array and optional date index from input data."""
        dates = None
        if isinstance(data, pd.Series):
            values = data.values.astype(np.float32)
            if pd.api.types.is_datetime64_any_dtype(data.index):
                dates = pd.DatetimeIndex(data.index)
        elif isinstance(data, pd.DataFrame):
            if "date" in data.columns:
                dates = pd.DatetimeIndex(pd.to_datetime(data["date"]))
            values = data[target].values.astype(np.float32)
        else:
            raise TypeError(f"Expected DataFrame or Series, got {type(data).__name__}")

        return values, dates
