"""
Base Forecaster Interface
=========================

Defines the abstract contract that every M5 forecaster must implement,
plus concrete serialization helpers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all demand-forecasting models.

    Subclasses **must** implement :meth:`fit`, :meth:`predict`, and
    :meth:`evaluate`.  Serialization via :mod:`joblib` is provided for free.

    Attributes
    ----------
    name : str
        Human-readable model name (e.g. ``"SARIMA"``, ``"LightGBM"``).
    is_fitted : bool
        Whether the model has been trained.
    """

    def __init__(self, name: str = "BaseForecaster") -> None:
        self._name = name
        self.is_fitted: bool = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Return the human-readable model identifier."""
        return self._name

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> "BaseForecaster":
        """Train the model on historical data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.  Expected schema varies by subclass.
        **kwargs
            Model-specific keyword arguments.

        Returns
        -------
        BaseForecaster
            ``self``, for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int = 28, **kwargs: Any) -> pd.DataFrame:
        """Generate point forecasts (and optionally intervals).

        Parameters
        ----------
        horizon : int
            Number of future time steps to forecast (default 28 days).
        **kwargs
            Additional arguments (e.g. ``return_ci=True``).

        Returns
        -------
        pd.DataFrame
            Forecasts with at least a ``yhat`` column.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        test_df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate forecasts against ground truth.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data containing actual values.
        **kwargs
            Additional evaluation options.

        Returns
        -------
        dict[str, float]
            Mapping of metric name → value (e.g. ``{"RMSE": 2.34}``).
        """
        ...

    # ── Serialization ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model to disk using :mod:`joblib`.

        Parameters
        ----------
        path : str | Path
            Destination file path (e.g. ``"models/sarima.joblib"``).

        Returns
        -------
        Path
            Absolute path to the saved file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Cannot save {self.name}: model has not been fitted. "
                "Call .fit() first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("%s saved to %s", self.name, path.resolve())
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Load a previously saved forecaster from disk.

        Parameters
        ----------
        path : str | Path
            Path to the ``.joblib`` file.

        Returns
        -------
        BaseForecaster
            The deserialized model instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        TypeError
            If the loaded object is not a :class:`BaseForecaster` subclass.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at {path.resolve()}")

        model = joblib.load(path)

        if not isinstance(model, BaseForecaster):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected a "
                f"BaseForecaster subclass."
            )

        logger.info("Loaded %s from %s", model.name, path.resolve())
        return model

    # ── Dunder helpers ──────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return f"<{self.__class__.__name__}(name={self.name!r}, fitted={self.is_fitted})>"
