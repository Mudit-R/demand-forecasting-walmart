# -*- coding: utf-8 -*-
"""
Walmart M5 Real Model Training & Evaluation Pipeline
=====================================================
Executes genuine end-to-end training and evaluation on the Walmart M5 dataset:
1. Ingests raw sales_train_evaluation.csv, calendar.csv, sell_prices.csv
2. Preprocesses and builds temporal/lag/rolling/price features
3. Trains LightGBM, SARIMAX, Prophet, Chronos-2, and TFT models
4. Computes exact empirical out-of-sample evaluation metrics (RMSE, MAE, MAPE, sMAPE, WRMSSE)
5. Extracts real TreeSHAP values, feature importances, changepoints, and ACF diagnostics
6. Saves real model artifacts to models/, results/, and web/data.js
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_real_training import main as train_real_models
from scripts.export_web_data import main as export_web_data

def main():
    print("Executing genuine end-to-end M5 training and evaluation pipeline...")
    train_real_models()
    export_web_data()
    print("All models trained and real empirical results updated successfully!")

if __name__ == "__main__":
    main()
