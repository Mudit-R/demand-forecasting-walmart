"""
Walmart M5 Preprocessor
=======================

Transforms raw M5 CSV files into a clean, long-format DataFrame ready for
feature engineering and model training.

Pipeline steps
--------------
1. Melt wide sales matrix → long format  (item_id, store_id, d, sales)
2. Merge with calendar  (adds date, events, SNAP flags)
3. Merge with prices    (adds weekly sell_price)
4. Handle missing values (forward-fill prices, zero-sales logic)
5. Optionally filter to top-N items by total volume
6. Time-based train / test split (last 28 days → test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import WalmartM5Loader

logger = logging.getLogger(__name__)


class WalmartM5Preprocessor:
    """End-to-end preprocessing for the M5 dataset.

    All heavy methods are intentionally **static / class-level** so they can
    be called individually during experimentation without instantiating the
    full pipeline.

    Examples
    --------
    >>> pp = WalmartM5Preprocessor()
    >>> train, test = pp.run_pipeline("data/raw", top_n=50, test_days=28)
    """

    # ── 1. Melt ─────────────────────────────────────────────────────────────

    @staticmethod
    def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
        """Convert the wide sales matrix to long format.

        Parameters
        ----------
        sales_df : pd.DataFrame
            Raw sales DataFrame with columns
            ``[id, item_id, dept_id, cat_id, store_id, state_id, d_1 … d_1941]``.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns
            ``[item_id, dept_id, cat_id, store_id, state_id, d, sales]``.
        """
        id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        # Drop the 'id' column if present (it's just a concat of other IDs)
        if "id" in sales_df.columns:
            sales_df = sales_df.drop(columns=["id"])

        day_cols = [c for c in sales_df.columns if c.startswith("d_")]
        logger.info(
            "Melting %s items × %s days → long format …",
            f"{len(sales_df):,}",
            f"{len(day_cols):,}",
        )

        long = sales_df.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )
        long["sales"] = long["sales"].astype("int32")
        logger.info("Melted shape: %s", long.shape)
        return long

    # ── 2. Merge ─────────────────────────────────────────────────────────────

    @staticmethod
    def merge_all(
        sales_long: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join long-format sales with calendar and price data.

        Parameters
        ----------
        sales_long : pd.DataFrame
            Output of :meth:`melt_sales`.
        calendar : pd.DataFrame
            Calendar DataFrame (must contain columns ``d``, ``date``,
            ``wm_yr_wk``, events, SNAP flags).
        prices : pd.DataFrame
            Weekly sell-price DataFrame (``store_id``, ``item_id``,
            ``wm_yr_wk``, ``sell_price``).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame sorted by ``[item_id, store_id, date]``.
        """
        logger.info("Merging sales with calendar …")
        df = sales_long.merge(calendar, on="d", how="left")

        logger.info("Merging with prices …")
        # Ensure matching types for wm_yr_wk
        for frame in [df, prices]:
            if "wm_yr_wk" in frame.columns:
                frame["wm_yr_wk"] = frame["wm_yr_wk"].astype("int32")

        # Convert category to string for merge keys to avoid dtype mismatch
        merge_keys = ["store_id", "item_id", "wm_yr_wk"]
        for key in ["store_id", "item_id"]:
            if hasattr(df[key], "cat"):
                df[key] = df[key].astype(str)
            if hasattr(prices[key], "cat"):
                prices[key] = prices[key].astype(str)

        df = df.merge(prices, on=merge_keys, how="left")
        df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

        logger.info("Merged shape: %s", df.shape)
        return df

    # ── 3. Missing-value handling ────────────────────────────────────────────

    @staticmethod
    def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the merged DataFrame.

        Strategy
        --------
        * **sell_price** — forward-fill within each (item, store) group, then
          backward-fill any remaining leading NaNs, then fill with 0.
        * **event columns** — fill with ``"NoEvent"`` / ``"None"``.
        * **sales** — NaN treated as 0 (product not yet on shelf).

        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame from :meth:`merge_all`.

        Returns
        -------
        pd.DataFrame
            DataFrame with no remaining NaN values.
        """
        logger.info("Handling missing values …")
        n_before = df.isna().sum().sum()

        # Forward-fill prices per item-store
        if "sell_price" in df.columns:
            df["sell_price"] = (
                df.groupby(["item_id", "store_id"])["sell_price"]
                .transform(lambda s: s.ffill().bfill())
            )
            df["sell_price"] = df["sell_price"].fillna(0.0)

        # Event columns
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            if col in df.columns:
                df[col] = df[col].fillna("NoEvent")

        # Sales
        if "sales" in df.columns:
            df["sales"] = df["sales"].fillna(0)

        n_after = df.isna().sum().sum()
        logger.info("Missing values: %s → %s", f"{n_before:,}", f"{n_after:,}")
        return df

    # ── 4. Top-N filtering ──────────────────────────────────────────────────

    @staticmethod
    def filter_top_items(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """Keep only the *n* items with the highest total sales volume.

        This dramatically reduces dataset size for faster experimentation.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``item_id`` and ``sales`` columns.
        n : int
            Number of top items to keep (default 50).

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only rows for the top-*n* items.
        """
        total_sales = (
            df.groupby("item_id")["sales"]
            .sum()
            .nlargest(n)
            .index
        )
        filtered = df[df["item_id"].isin(total_sales)].reset_index(drop=True)
        logger.info(
            "Filtered to top %d items: %s → %s rows",
            n,
            f"{len(df):,}",
            f"{len(filtered):,}",
        )
        return filtered

    # ── 5. Train / test split ────────────────────────────────────────────────

    @staticmethod
    def create_train_test_split(
        df: pd.DataFrame,
        test_days: int = 28,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based split: the last *test_days* days form the test set.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``date`` column (datetime).
        test_days : int
            Number of trailing days for the test set (default 28).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, test_df)``
        """
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column for time-based split.")

        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=test_days - 1)

        train = df[df["date"] < cutoff].reset_index(drop=True)
        test = df[df["date"] >= cutoff].reset_index(drop=True)

        logger.info(
            "Train/test split — cutoff: %s | train: %s rows | test: %s rows",
            cutoff.date(),
            f"{len(train):,}",
            f"{len(test):,}",
        )
        return train, test

    # ── 6. Full pipeline ────────────────────────────────────────────────────

    def run_pipeline(
        self,
        data_dir: str | Path = "data/raw",
        top_n: Optional[int] = 50,
        test_days: int = 28,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full preprocessing pipeline.

        Parameters
        ----------
        data_dir : str | Path
            Path to directory with raw CSVs.
        top_n : int | None
            If set, keep only the top-*n* items by total sales.
            Pass ``None`` to keep all items.
        test_days : int
            Number of trailing days for the test set.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, test_df)`` — clean, long-format DataFrames ready
            for feature engineering.
        """
        logger.info("═══ Starting M5 preprocessing pipeline ═══")

        # Load
        loader = WalmartM5Loader(data_dir)
        sales, calendar, prices = loader.load_all()

        # Melt
        sales_long = self.melt_sales(sales)

        # Merge
        df = self.merge_all(sales_long, calendar, prices)

        # Missing values
        df = self.handle_missing(df)

        # Filter
        if top_n is not None:
            df = self.filter_top_items(df, n=top_n)

        # Split
        train, test = self.create_train_test_split(df, test_days=test_days)

        logger.info("═══ Pipeline complete ═══")
        return train, test
