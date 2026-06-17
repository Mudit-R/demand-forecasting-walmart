"""
Walmart M5 Dataset Loader
=========================

Handles loading of the three core M5 competition files:
- sales_train_evaluation.csv  (30,490 items × 1,941 days)
- calendar.csv                (event & SNAP metadata per day)
- sell_prices.csv             (weekly prices per item-store)

Memory is reduced via categorical dtypes for IDs and float32 for prices.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── dtype maps for memory optimization ──────────────────────────────────────
_SALES_CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
_PRICES_CAT_COLS = ["store_id", "item_id"]
_CALENDAR_CAT_COLS = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "weekday",
]


class WalmartM5Loader:
    """Load and validate the three Walmart M5 competition CSV files.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the raw CSV files.  Defaults to ``data/raw``
        relative to the current working directory.

    Examples
    --------
    >>> loader = WalmartM5Loader("data/raw")
    >>> sales, calendar, prices = loader.load_all()
    """

    SALES_FILE = "sales_train_evaluation.csv"
    CALENDAR_FILE = "calendar.csv"
    PRICES_FILE = "sell_prices.csv"
    KAGGLE_DATASET = "walmart-m5-forecasting-accuracy"

    def __init__(self, data_dir: str | Path = "data/raw") -> None:
        self.data_dir = Path(data_dir)

    # ── public API ──────────────────────────────────────────────────────────

    def load_sales(self) -> pd.DataFrame:
        """Load ``sales_train_evaluation.csv`` with optimized dtypes.

        Returns
        -------
        pd.DataFrame
            Shape ≈ (30 490, 1 946).  ID columns are ``category`` dtype;
            daily sales columns (``d_1`` … ``d_1941``) are ``int16``.

        Raises
        ------
        FileNotFoundError
            If the file is missing from *data_dir*.
        """
        path = self._validate_file(self.SALES_FILE)
        logger.info("Loading sales data from %s …", path)

        # Build dtype dict: category for IDs, int16 for daily columns
        day_cols = {f"d_{i}": "int16" for i in range(1, 1942)}
        cat_dtypes = {c: "category" for c in _SALES_CAT_COLS}
        dtypes = {**cat_dtypes, **day_cols}

        df = pd.read_csv(path, dtype=dtypes)
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("Sales loaded: %s rows, %.1f MB", f"{len(df):,}", mem_mb)
        return df

    def load_calendar(self) -> pd.DataFrame:
        """Load ``calendar.csv`` with parsed dates and category dtypes.

        Returns
        -------
        pd.DataFrame
            1 969 rows.  ``date`` is ``datetime64``; event/weekday columns
            are ``category`` dtype.
        """
        path = self._validate_file(self.CALENDAR_FILE)
        logger.info("Loading calendar data from %s …", path)

        df = pd.read_csv(
            path,
            parse_dates=["date"],
            dtype={c: "category" for c in _CALENDAR_CAT_COLS},
        )
        # SNAP columns to bool
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        logger.info("Calendar loaded: %s rows", f"{len(df):,}")
        return df

    def load_prices(self) -> pd.DataFrame:
        """Load ``sell_prices.csv`` with float32 prices and category IDs.

        Returns
        -------
        pd.DataFrame
            ~6.8 M rows.  ``sell_price`` is ``float32``.
        """
        path = self._validate_file(self.PRICES_FILE)
        logger.info("Loading price data from %s …", path)

        dtypes = {c: "category" for c in _PRICES_CAT_COLS}
        dtypes["sell_price"] = "float32"
        dtypes["wm_yr_wk"] = "int16"

        df = pd.read_csv(path, dtype=dtypes)
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("Prices loaded: %s rows, %.1f MB", f"{len(df):,}", mem_mb)
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets at once.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(sales, calendar, prices)``
        """
        return self.load_sales(), self.load_calendar(), self.load_prices()

    def download_dataset(
        self,
        competition: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """Download the M5 dataset from Kaggle using the CLI.

        Requires the ``kaggle`` Python package and a valid
        ``~/.kaggle/kaggle.json`` API token.

        Parameters
        ----------
        competition : str, optional
            Kaggle competition slug.  Defaults to
            ``walmart-m5-forecasting-accuracy``.
        force : bool
            Re-download even if files already exist.

        Returns
        -------
        Path
            Directory where the files were extracted.

        Raises
        ------
        RuntimeError
            If the ``kaggle`` CLI is not installed or the download fails.
        """
        comp = competition or self.KAGGLE_DATASET
        dest = self.data_dir
        dest.mkdir(parents=True, exist_ok=True)

        # Skip download when all files are present
        required = [self.SALES_FILE, self.CALENDAR_FILE, self.PRICES_FILE]
        if not force and all((dest / f).exists() for f in required):
            logger.info("All data files already present in %s — skipping download.", dest)
            return dest

        logger.info("Downloading M5 data from Kaggle competition '%s' …", comp)

        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            comp,
            "-p",
            str(dest),
            "--force" if force else "",
        ]
        cmd = [c for c in cmd if c]  # drop empty strings

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Kaggle download output:\n%s", result.stdout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "kaggle CLI not found. Install with: pip install kaggle"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Kaggle download failed:\n{exc.stderr}"
            ) from exc

        # Unzip if a zip file was downloaded
        zip_path = dest / f"{comp}.zip"
        if zip_path.exists():
            import zipfile

            logger.info("Extracting %s …", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()
            logger.info("Extraction complete.")

        return dest

    # ── internals ───────────────────────────────────────────────────────────

    def _validate_file(self, filename: str) -> Path:
        """Return the full path to *filename*, raising if it does not exist."""
        path = self.data_dir / filename
        if not path.exists():
            available = [f.name for f in self.data_dir.iterdir()] if self.data_dir.exists() else []
            raise FileNotFoundError(
                f"'{filename}' not found in {self.data_dir.resolve()}.\n"
                f"Available files: {available}\n"
                f"Run loader.download_dataset() or manually place the M5 CSVs "
                f"in '{self.data_dir.resolve()}'."
            )
        return path
