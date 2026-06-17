"""
LightGBM Forecaster
====================

Gradient-boosted tree model using :mod:`lightgbm` for tabular demand
forecasting.

Key design choices
------------------
* **No shuffling** — preserves temporal order during training.
* **Time-based validation** — a trailing slice of the training set serves
  as the early-stopping validation set.
* **Recursive prediction** — future lags are filled iteratively so that
  multi-step forecasts remain autoregressive.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all
from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)

# Features that should never be used as model input
_DROP_COLS = [
    "sales",
    "date",
    "d",
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    "wm_yr_wk",
    "weekday",
    "wday",
    "event_name_1",
    "event_name_2",
    "event_type_1",
    "event_type_2",
]


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based demand forecaster with recursive prediction.

    Parameters
    ----------
    params : dict | None
        LightGBM training parameters.  Sensible defaults for time-series
        are used when ``None``.
    num_boost_round : int
        Maximum boosting iterations (default 1 000).
    early_stopping_rounds : int
        Stop training if validation metric does not improve for this many
        rounds (default 50).
    val_days : int
        Number of trailing training days used for early-stopping
        validation (default 28).

    Examples
    --------
    >>> model = LightGBMForecaster()
    >>> model.fit(train_df)
    >>> preds = model.predict(horizon=28, recent_data=train_df)
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": 42,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 1_000,
        early_stopping_rounds: int = 50,
        val_days: int = 28,
    ) -> None:
        super().__init__(name="LightGBM")
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.val_days = val_days

        # State
        self._model: Optional[lgb.Booster] = None
        self._feature_names: List[str] = []
        self._train_df: Optional[pd.DataFrame] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str = "sales",
        categorical_features: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "LightGBMForecaster":
        """Train LightGBM with time-ordered split for validation.

        Parameters
        ----------
        train_df : pd.DataFrame
            Feature-rich training DataFrame.  Must contain *target* column
            and a ``date`` column for the time-based split.
        target : str
            Name of the target column (default ``"sales"``).
        categorical_features : list[str] | None
            Columns to treat as LightGBM categoricals.
        **kwargs
            Extra arguments forwarded to ``lgb.train()``.

        Returns
        -------
        LightGBMForecaster
            ``self``
        """
        df = train_df.copy()
        df = df.dropna(subset=[target])

        # Determine feature columns
        self._feature_names = self._select_features(df, target)
        logger.info(
            "Training LightGBM on %d rows, %d features …",
            len(df),
            len(self._feature_names),
        )

        # Encode categoricals
        cat_feats = categorical_features or []
        for col in cat_feats:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("category")

        # Time-based validation split
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        val_cutoff = max_date - pd.Timedelta(days=self.val_days - 1)

        train_mask = df["date"] < val_cutoff
        val_mask = df["date"] >= val_cutoff

        X_train = df.loc[train_mask, self._feature_names]
        y_train = df.loc[train_mask, target]
        X_val = df.loc[val_mask, self._feature_names]
        y_val = df.loc[val_mask, target]

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ]

        self._model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
            **kwargs,
        )

        self._train_df = train_df.copy()
        self.is_fitted = True
        logger.info(
            "LightGBM fitted — best iteration=%d, best RMSE=%.4f",
            self._model.best_iteration,
            self._model.best_score.get("valid", {}).get("rmse", float("nan")),
        )
        return self

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        recent_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate multi-step forecasts using a recursive strategy.

        At each step the model predicts one day ahead; the prediction is
        then appended to the history so that lag features remain valid.

        Parameters
        ----------
        horizon : int
            Number of future days to forecast (default 28).
        recent_data : pd.DataFrame | None
            Recent historical data needed to compute lag / rolling features
            for the first forecast step.  Falls back to the training set.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``yhat``.
        """
        self._check_fitted()

        history = (recent_data if recent_data is not None else self._train_df).copy()
        history["date"] = pd.to_datetime(history["date"])
        last_date = history["date"].max()

        predictions = []
        for step in range(horizon):
            next_date = last_date + pd.Timedelta(days=step + 1)

            # Build a single-row feature vector from the tail of history
            feat_row = self._build_next_features(history, next_date)
            if feat_row is None:
                predictions.append(0.0)
                continue

            X = feat_row[self._feature_names]
            yhat = float(self._model.predict(X, num_iteration=self._model.best_iteration)[0])
            yhat = max(yhat, 0.0)
            predictions.append(yhat)

            # Append prediction to history for recursive lags
            new_row = feat_row.copy()
            new_row["sales"] = yhat
            new_row["date"] = next_date
            history = pd.concat([history, new_row], ignore_index=True)

        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.DataFrame({"date": dates, "yhat": predictions})

    # ── evaluate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame,
        target: str = "sales",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate LightGBM predictions on a test set.

        Parameters
        ----------
        test_df : pd.DataFrame
            Feature-rich test DataFrame with *target* column.
        target : str
            Name of the target column.

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        self._check_fitted()

        X_test = test_df[self._feature_names]
        y_true = test_df[target].values
        y_pred = self._model.predict(X_test, num_iteration=self._model.best_iteration)
        y_pred = np.clip(y_pred, 0, None)

        metrics = evaluate_all(y_true, y_pred)
        logger.info("LightGBM evaluation: %s", metrics)
        return metrics

    # ── Feature importance ──────────────────────────────────────────────────

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame.

        Parameters
        ----------
        importance_type : str
            ``"gain"`` (default) or ``"split"``.
        top_n : int | None
            Return only the top-*n* features.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.
        """
        self._check_fitted()
        importance = self._model.feature_importance(importance_type=importance_type)
        fi = pd.DataFrame(
            {"feature": self._feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        if top_n is not None:
            fi = fi.head(top_n)
        return fi.reset_index(drop=True)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @staticmethod
    def _select_features(df: pd.DataFrame, target: str) -> List[str]:
        """Return numeric columns suitable for training."""
        exclude = set(_DROP_COLS) | {target}
        features = [
            c
            for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        return features

    def _build_next_features(
        self,
        history: pd.DataFrame,
        next_date: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Construct a single-row feature DataFrame for *next_date*.

        Uses the tail of *history* to derive lag / rolling features.
        """
        try:
            row = {}
            # Calendar features
            row["day_of_week"] = next_date.dayofweek
            row["day_of_month"] = next_date.day
            row["week_of_year"] = next_date.isocalendar().week
            row["month"] = next_date.month
            row["quarter"] = next_date.quarter
            row["year"] = next_date.year
            row["is_weekend"] = int(next_date.dayofweek >= 5)
            row["is_month_start"] = int(next_date.is_month_start)
            row["is_month_end"] = int(next_date.is_month_end)

            # Lag features from history
            sales_history = history["sales"].values
            for lag in [7, 14, 28, 90, 365]:
                col_name = f"lag_{lag}"
                if col_name in self._feature_names:
                    idx = len(sales_history) - lag
                    row[col_name] = float(sales_history[idx]) if idx >= 0 else 0.0

            # Rolling features
            for w in [7, 14, 28]:
                tail = sales_history[-w:] if len(sales_history) >= w else sales_history
                if len(tail) > 0:
                    row[f"rolling_mean_{w}"] = float(np.mean(tail))
                    row[f"rolling_std_{w}"] = float(np.std(tail))
                    row[f"rolling_min_{w}"] = float(np.min(tail))
                    row[f"rolling_max_{w}"] = float(np.max(tail))

            # Fill any remaining required features with 0
            for feat in self._feature_names:
                if feat not in row:
                    row[feat] = 0.0

            return pd.DataFrame([row])
        except Exception as exc:
            logger.warning("Failed to build features for %s: %s", next_date, exc)
            return None
