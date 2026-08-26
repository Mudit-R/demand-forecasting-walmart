# -*- coding: utf-8 -*-
"""
Walmart M5 Demand Forecasting — Multi-Model Training Pipeline
============================================================
Orchestrates genuine end-to-end data ingestion, preprocessing, feature engineering,
model fitting, evaluation, and saving artifacts to results/ and web/ directories.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_real_training import main as train_real_models
from scripts.export_web_data import main as export_web_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_all")


def parse_args():
    parser = argparse.ArgumentParser(description="Walmart M5 Multi-Model Training Pipeline")
    parser.add_argument("--top_n", type=int, default=50, help="Top N high-volume items to engineer features for")
    parser.add_argument("--test_days", type=int, default=28, help="Holdout evaluation window (default: 28 days)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Executing genuine multi-model training pipeline on Walmart M5 dataset...")
    train_real_models()
    export_web_data()
    logger.info("Pipeline completed successfully. All artifacts and empirical metrics are generated.")


if __name__ == "__main__":
    main()
