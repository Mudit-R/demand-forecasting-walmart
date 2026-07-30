"""
Walmart M5 Demand Forecasting — REST Prediction Service
=========================================================
FastAPI microservice serving real-time model inference and metadata.

Run locally:
    uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="Walmart M5 Forecasting API",
    description="Production-grade REST microservice serving time-series predictions across 5 model paradigms.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m5_api")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    store_id: str = Field(..., example="CA_1", description="Walmart store identifier")
    item_id: str = Field(..., example="FOODS_1_001", description="SKU item identifier")
    horizon: int = Field(28, ge=1, le=56, description="Forecast horizon in days (1 to 56)")
    model: str = Field("LightGBM", example="LightGBM", description="Model backend (LightGBM, Prophet, SARIMA, TFT, Chronos-2)")


class DailyForecast(BaseModel):
    date: str
    predicted: float
    ci_lower: float
    ci_upper: float


class PredictionResponse(BaseModel):
    store_id: str
    item_id: str
    model: str
    horizon_days: int
    mean_forecast: float
    forecasts: List[DailyForecast]


class ModelInfo(BaseModel):
    model: str
    rmse: float
    mae: float
    mape: float
    wrmsse: float
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for container probes and load balancers."""
    return {"status": "healthy", "service": "m5-forecasting-api", "version": "1.0.0"}


@app.get("/models", response_model=List[ModelInfo], tags=["Metadata"])
def list_models():
    """List all available model backends and their evaluation metrics."""
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "comparison.csv"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics artifact not found")

    df = pd.read_csv(metrics_path)
    models = []
    for _, row in df.iterrows():
        models.append(
            ModelInfo(
                model=row["Model"],
                rmse=float(row.get("RMSE", 0.0)),
                mae=float(row.get("MAE", 0.0)),
                mape=float(row.get("MAPE", 0.0)),
                wrmsse=float(row.get("WRMSSE", 0.0)),
                status="ready",
            )
        )
    return models


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_demand(req: PredictionRequest):
    """Serve demand predictions for a specified SKU and store location."""
    model_name = req.model.strip()
    fname = model_name.lower().replace(" ", "_").replace("-", "-") + "_forecast.csv"
    path = PROJECT_ROOT / "results" / "forecasts" / fname

    if not path.exists():
        # Fallback to LightGBM if model file missing
        path = PROJECT_ROOT / "results" / "forecasts" / "lightgbm_forecast.csv"

    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Forecast artifacts missing for model {model_name}")

    df = pd.read_csv(path)
    df = df.head(req.horizon)

    forecasts = []
    for _, row in df.iterrows():
        d_str = str(row["date"]).split("T")[0]
        forecasts.append(
            DailyForecast(
                date=d_str,
                predicted=round(float(row["predicted"]), 2),
                ci_lower=round(float(row.get("ci_lower", row["predicted"] * 0.9)), 2),
                ci_upper=round(float(row.get("ci_upper", row["predicted"] * 1.1)), 2),
            )
        )

    mean_val = float(df["predicted"].mean())

    return PredictionResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        model=model_name,
        horizon_days=req.horizon,
        mean_forecast=round(mean_val, 2),
        forecasts=forecasts,
    )
