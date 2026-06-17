"""
Feature Engineering for M5 Demand Forecasting
==============================================

Builds ML-ready features from the preprocessed long-format DataFrame.

Feature groups
--------------
* **Lag features** — past sales at fixed offsets (7, 14, 28, 90, 365 days).
* **Rolling features** — rolling mean / std / min / max over windows.
* **Calendar features** — day-of-week, month, quarter, is_weekend, etc.
* **Price features** — price change %, momentum, relative price vs category.
* **Event features** — one-hot encoded events and SNAP indicators.

All group-level operations use ``(item_id, store_id)`` as the group key
so that features never leak across products or stores.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Group key used throughout
_GRP = ["item_id", "store_id"]


class FeatureEngineer:
    """Build tabular features for tree-based and neural models.

    All methods accept and return a DataFrame, making it easy to chain
    them or use them individually in notebooks.

    Examples
    --------
    >>> fe = FeatureEngineer()
    >>> df = fe.build_features(preprocessed_df)
    """

    # ── Lag features ────────────────────────────────────────────────────────

    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        lags: Optional[List[int]] = None,
        target: str = "sales",
    ) -> pd.DataFrame:
        """Add lagged values of *target* per item-store group.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``item_id``, ``store_id``, ``date``, and *target*.
        lags : list[int]
            Lag offsets in days.  Defaults to ``[7, 14, 28, 90, 365]``.
        target : str
            Column to lag (default ``"sales"``).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with new columns ``lag_{L}`` for each lag.
        """
        if lags is None:
            lags = [7, 14, 28, 90, 365]

        logger.info("Adding lag features: %s", lags)
        df = df.sort_values([*_GRP, "date"]).copy()

        grouped = df.groupby(_GRP, observed=True)[target]
        for lag in lags:
            col_name = f"lag_{lag}"
            df[col_name] = grouped.shift(lag).astype("float32")

        return df

    # ── Rolling features ────────────────────────────────────────────────────

    @staticmethod
    def add_rolling_features(
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        target: str = "sales",
    ) -> pd.DataFrame:
        """Add rolling statistics of *target* per item-store group.

        For each window size the following columns are created:
        ``rolling_mean_{W}``, ``rolling_std_{W}``, ``rolling_min_{W}``,
        ``rolling_max_{W}``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``item_id``, ``store_id``, ``date``, and *target*.
        windows : list[int]
            Rolling window sizes in days.  Defaults to ``[7, 14, 28]``.
        target : str
            Column to compute rolling stats on.

        Returns
        -------
        pd.DataFrame
            DataFrame with new rolling-statistic columns.
        """
        if windows is None:
            windows = [7, 14, 28]

        logger.info("Adding rolling features: windows=%s", windows)
        df = df.sort_values([*_GRP, "date"]).copy()

        grouped = df.groupby(_GRP, observed=True)[target]
        for w in windows:
            rolled = grouped.transform(
                lambda s, _w=w: s.shift(1).rolling(window=_w, min_periods=1).mean()
            )
            df[f"rolling_mean_{w}"] = rolled.astype("float32")

            rolled_std = grouped.transform(
                lambda s, _w=w: s.shift(1).rolling(window=_w, min_periods=1).std()
            )
            df[f"rolling_std_{w}"] = rolled_std.astype("float32")

            rolled_min = grouped.transform(
                lambda s, _w=w: s.shift(1).rolling(window=_w, min_periods=1).min()
            )
            df[f"rolling_min_{w}"] = rolled_min.astype("float32")

            rolled_max = grouped.transform(
                lambda s, _w=w: s.shift(1).rolling(window=_w, min_periods=1).max()
            )
            df[f"rolling_max_{w}"] = rolled_max.astype("float32")

        return df

    # ── Calendar features ───────────────────────────────────────────────────

    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive temporal features from the ``date`` column.

        New columns
        -----------
        ``day_of_week``, ``day_of_month``, ``week_of_year``, ``month``,
        ``quarter``, ``year``, ``is_weekend``, ``is_month_start``,
        ``is_month_end``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``date`` column (datetime64).

        Returns
        -------
        pd.DataFrame
            DataFrame with additional calendar columns.
        """
        logger.info("Adding calendar features …")
        dt = pd.to_datetime(df["date"])

        df["day_of_week"] = dt.dt.dayofweek.astype("int8")  # 0=Mon … 6=Sun
        df["day_of_month"] = dt.dt.day.astype("int8")
        df["week_of_year"] = dt.dt.isocalendar().week.astype("int8")
        df["month"] = dt.dt.month.astype("int8")
        df["quarter"] = dt.dt.quarter.astype("int8")
        df["year"] = dt.dt.year.astype("int16")
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
        df["is_month_start"] = dt.dt.is_month_start.astype("int8")
        df["is_month_end"] = dt.dt.is_month_end.astype("int8")

        return df

    # ── Price features ──────────────────────────────────────────────────────

    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive price-based features per item-store group.

        New columns
        -----------
        * ``price_change`` — week-over-week percentage change.
        * ``price_momentum`` — rolling 4-week mean of price change.
        * ``price_norm`` — price normalised to the item-store mean
          (relative price index).
        * ``price_std`` — rolling 4-week standard deviation of price.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``item_id``, ``store_id``, ``sell_price``.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional price-feature columns.
        """
        logger.info("Adding price features …")
        if "sell_price" not in df.columns:
            logger.warning("'sell_price' column not found — skipping price features.")
            return df

        grouped = df.groupby(_GRP, observed=True)["sell_price"]

        # Week-over-week percentage change
        df["price_change"] = grouped.pct_change().astype("float32")
        df["price_change"] = df["price_change"].fillna(0.0)

        # Momentum: rolling mean of price change (4-week ≈ 28 days)
        df["price_momentum"] = (
            df.groupby(_GRP, observed=True)["price_change"]
            .transform(lambda s: s.rolling(window=28, min_periods=1).mean())
            .astype("float32")
        )

        # Normalised price (relative to item-store average)
        item_store_mean = grouped.transform("mean")
        df["price_norm"] = (df["sell_price"] / item_store_mean.replace(0, np.nan)).astype("float32")
        df["price_norm"] = df["price_norm"].fillna(1.0)

        # Rolling price volatility
        df["price_std"] = (
            grouped.transform(lambda s: s.rolling(window=28, min_periods=1).std())
            .astype("float32")
        )
        df["price_std"] = df["price_std"].fillna(0.0)

        return df

    # ── Event features ──────────────────────────────────────────────────────

    @staticmethod
    def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode event columns and ensure SNAP flags are numeric.

        New columns
        -----------
        * ``has_event`` — binary flag: any event today.
        * One-hot columns for each unique ``event_type_1`` value.
        * ``snap`` — combined SNAP indicator for the item's state.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain event and SNAP columns from the calendar merge.

        Returns
        -------
        pd.DataFrame
            DataFrame with event indicator columns.
        """
        logger.info("Adding event features …")

        # Binary: has any event
        if "event_name_1" in df.columns:
            df["has_event"] = (
                (df["event_name_1"] != "NoEvent") & (df["event_name_1"].notna())
            ).astype("int8")
        else:
            df["has_event"] = 0

        # One-hot event types
        if "event_type_1" in df.columns:
            event_dummies = pd.get_dummies(
                df["event_type_1"],
                prefix="event_type",
                dtype="int8",
            )
            # Drop 'NoEvent' dummy if present
            no_event_cols = [c for c in event_dummies.columns if "NoEvent" in c]
            event_dummies = event_dummies.drop(columns=no_event_cols, errors="ignore")
            df = pd.concat([df, event_dummies], axis=1)

        # SNAP — map to the correct state column per row
        snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
        if all(c in df.columns for c in snap_cols) and "state_id" in df.columns:
            state_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

            def _snap_for_row(row: pd.Series) -> int:
                col = state_map.get(str(row.get("state_id", "")))
                if col and col in row.index:
                    return int(bool(row[col]))
                return 0

            df["snap"] = df.apply(_snap_for_row, axis=1).astype("int8")
        elif any(c in df.columns for c in snap_cols):
            # Fallback: just ensure they are int
            for col in snap_cols:
                if col in df.columns:
                    df[col] = df[col].astype("int8")

        return df

    # ── Full pipeline ───────────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature-engineering steps in sequence.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed long-format DataFrame (output of
            :class:`~src.data.preprocessor.WalmartM5Preprocessor`).

        Returns
        -------
        pd.DataFrame
            Feature-rich DataFrame ready for model training.
        """
        logger.info("═══ Starting feature engineering pipeline ═══")
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_calendar_features(df)
        df = self.add_price_features(df)
        df = self.add_event_features(df)
        logger.info(
            "═══ Feature engineering complete — %d columns ═══",
            len(df.columns),
        )
        return df
