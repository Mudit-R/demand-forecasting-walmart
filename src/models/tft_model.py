"""
Temporal Fusion Transformer (TFT) Forecaster
=============================================

Uses :class:`neuralforecast.models.TFT` from the *NeuralForecast* library.

The TFT handles:
* **Static covariates** — store, category (encoded as integers).
* **Known future inputs** — calendar features, events, SNAP flags.
* **Observed past inputs** — price, lagged sales.

NeuralForecast expects a DataFrame with columns ``unique_id``, ``ds``, ``y``
plus any exogenous features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer wrapper around NeuralForecast.

    Parameters
    ----------
    horizon : int
        Forecast horizon in time steps (default 28 days).
    input_size : int
        Number of historical time steps the model looks back (default 28).
    learning_rate : float
        Adam learning rate (default 1e-3).
    max_steps : int
        Maximum training iterations (default 500).
    hidden_size : int
        Hidden layer dimension (default 64).
    attention_head_size : int
        Number of attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    batch_size : int
        Training batch size (default 64).
    scaler_type : str
        NeuralForecast scaler (default ``"robust"``).

    Examples
    --------
    >>> model = TFTForecaster(horizon=28, max_steps=300)
    >>> model.fit(train_df)
    >>> preds = model.predict()
    """

    def __init__(
        self,
        horizon: int = 28,
        input_size: int = 28,
        learning_rate: float = 1e-3,
        max_steps: int = 500,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        batch_size: int = 64,
        scaler_type: str = "robust",
    ) -> None:
        super().__init__(name="TFT")
        self.horizon = horizon
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.scaler_type = scaler_type

        # State
        self._nf = None  # NeuralForecast instance
        self._train_df: Optional[pd.DataFrame] = None
        self._futr_exog_cols: List[str] = []
        self._hist_exog_cols: List[str] = []
        self._static_cols: List[str] = []

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str = "sales",
        static_features: Optional[List[str]] = None,
        futr_exog_cols: Optional[List[str]] = None,
        hist_exog_cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "TFTForecaster":
        """Train the TFT model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Long-format DataFrame with columns ``item_id``, ``store_id``,
            ``date``, *target*, and any exogenous features.
        target : str
            Target column name (default ``"sales"``).
        static_features : list[str] | None
            Columns for static covariates (e.g. ``["dept_id", "cat_id"]``).
        futr_exog_cols : list[str] | None
            Known-future exogenous columns (e.g. calendar features).
        hist_exog_cols : list[str] | None
            Historical-only exogenous columns (e.g. ``["sell_price"]``).
        **kwargs
            Extra arguments forwarded to ``NeuralForecast.fit()``.

        Returns
        -------
        TFTForecaster
            ``self``
        """
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT

        nf_df = self._prepare_nf_df(train_df, target)

        # Auto-detect exogenous columns if not provided
        self._futr_exog_cols = futr_exog_cols or self._detect_futr_exog(nf_df)
        self._hist_exog_cols = hist_exog_cols or self._detect_hist_exog(nf_df)
        self._static_cols = static_features or []

        logger.info(
            "Fitting TFT — horizon=%d, input_size=%d, max_steps=%d, "
            "futr_exog=%s, hist_exog=%s, static=%s",
            self.horizon,
            self.input_size,
            self.max_steps,
            self._futr_exog_cols,
            self._hist_exog_cols,
            self._static_cols,
        )

        tft_model = TFT(
            h=self.horizon,
            input_size=self.input_size,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            hidden_size=self.hidden_size,
            n_head=self.attention_head_size,
            dropout=self.dropout,
            batch_size=self.batch_size,
            scaler_type=self.scaler_type,
            futr_exog_list=self._futr_exog_cols if self._futr_exog_cols else None,
            hist_exog_list=self._hist_exog_cols if self._hist_exog_cols else None,
        )

        self._nf = NeuralForecast(
            models=[tft_model],
            freq="D",
        )

        # Handle static features
        static_df = None
        if self._static_cols:
            static_df = self._build_static_df(nf_df, self._static_cols)

        self._nf.fit(df=nf_df, static_df=static_df, **kwargs)
        self._train_df = nf_df
        self.is_fitted = True
        logger.info("TFT training complete.")
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        futr_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate TFT forecasts.

        Parameters
        ----------
        horizon : int
            Ignored (uses the horizon set at init).  Included for API
            compatibility.
        futr_df : pd.DataFrame | None
            Future exogenous features.  Must have ``unique_id``, ``ds``,
            and columns matching ``futr_exog_cols``.

        Returns
        -------
        pd.DataFrame
            Columns: ``unique_id``, ``ds``, ``yhat``.
        """
        self._check_fitted()

        preds = self._nf.predict(futr_df=futr_df, **kwargs)
        preds = preds.reset_index()

        # NeuralForecast names the prediction column "TFT"
        pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")]
        if pred_col:
            preds = preds.rename(columns={pred_col[0]: "yhat"})

        preds["yhat"] = preds["yhat"].clip(lower=0)
        return preds

    # ── evaluate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame,
        target: str = "sales",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate TFT forecasts against ground truth.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test DataFrame with ``date``, ``item_id``, ``store_id``, *target*.

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        self._check_fitted()

        nf_test = self._prepare_nf_df(test_df, target)
        preds = self.predict()

        # Merge predictions with actuals
        merged = nf_test.merge(preds[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="inner")
        y_true = merged["y"].values
        y_pred = merged["yhat"].values

        metrics = evaluate_all(y_true, y_pred)
        logger.info("TFT evaluation: %s", metrics)
        return metrics

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._nf is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @staticmethod
    def _prepare_nf_df(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
        """Convert to NeuralForecast format: ``unique_id``, ``ds``, ``y``."""
        nf_df = df.copy()

        # Create unique_id from item + store
        if "unique_id" not in nf_df.columns:
            nf_df["unique_id"] = (
                nf_df["item_id"].astype(str) + "_" + nf_df["store_id"].astype(str)
            )

        # Rename columns
        rename_map = {}
        if "date" in nf_df.columns:
            rename_map["date"] = "ds"
        if target in nf_df.columns and target != "y":
            rename_map[target] = "y"
        nf_df = nf_df.rename(columns=rename_map)

        nf_df["ds"] = pd.to_datetime(nf_df["ds"])
        nf_df["y"] = nf_df["y"].astype("float32")

        return nf_df

    @staticmethod
    def _detect_futr_exog(df: pd.DataFrame) -> List[str]:
        """Auto-detect known-future exogenous columns."""
        candidates = [
            "day_of_week",
            "month",
            "quarter",
            "is_weekend",
            "has_event",
            "snap",
            "day_of_month",
            "week_of_year",
            "is_month_start",
            "is_month_end",
        ]
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _detect_hist_exog(df: pd.DataFrame) -> List[str]:
        """Auto-detect historical-only exogenous columns."""
        candidates = ["sell_price", "price_change", "price_momentum", "price_norm"]
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _build_static_df(
        df: pd.DataFrame,
        static_cols: List[str],
    ) -> pd.DataFrame:
        """Build a static-features DataFrame (one row per ``unique_id``)."""
        static = df.groupby("unique_id")[static_cols].first().reset_index()
        # Encode categoricals as integers
        for col in static_cols:
            if not pd.api.types.is_numeric_dtype(static[col]):
                static[col] = static[col].astype("category").cat.codes
        return static
