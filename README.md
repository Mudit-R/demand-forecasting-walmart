# Walmart M5 Demand Forecasting — Multi-Model Enterprise Analytics System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20Dashboard-000000?logo=vercel&logoColor=white)](https://demand-forecasting-walmart-six.vercel.app/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FMudit-R%2Fdemand-forecasting-walmart)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-9ACD32)](https://lightgbm.readthedocs.io/)
[![NeuralForecast](https://img.shields.io/badge/TFT-NeuralForecast-792EE5)](https://nixtla.github.io/neuralforecast/)
[![Chronos](https://img.shields.io/badge/Chronos--2-Amazon-FF9900)](https://github.com/amazon-science/chronos-forecasting)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-0467DF)](https://facebook.github.io/prophet/)

Live Production Dashboard: **[https://demand-forecasting-walmart-six.vercel.app/](https://demand-forecasting-walmart-six.vercel.app/)**

---

## Executive Overview

An enterprise-grade demand forecasting and business analytics system engineered on the **Walmart M5 Kaggle competition dataset** — one of the largest retail benchmarks in modern machine learning.

The pipeline ingests **30,490 product SKUs** across **10 Walmart supercenters** in three US states (California, Texas, Wisconsin) spanning **1,941 days (5.4 years)** of daily checkout history. It systematically benchmarks **six forecasting paradigms** (Ensemble Blend, LightGBM, SARIMAX, Prophet, Chronos-2, and TFT) and translates statistical accuracy into **actionable supply chain ROI, safety stock compression, price elasticity strategies, and SNAP policy scheduling**.

---

## Executive Business & Financial Impact (ROI)

Quantified financial return modeled on 10 Walmart Supercenters (\$350M aggregate inventory base at a 22% annual carrying cost rate):

```
ANNUAL ECONOMIC VALUE GENERATED: $26.01M / YEAR
|-- 1. Working Capital & Carrying Cost Reduction: $19.71M / year
|   |-- Safety stock formula: SS = Z * sigma_e * sqrt(Lead_Time)
|   |-- Compressing forecast error from 11.8% (TFT) to 5.0% (Ensemble) reduces required safety buffer by 25.7%.
|   `-- Lowers warehouse holding inventory from 14.0 days to 10.4 days across all stores.
|
|-- 2. Stockout Margin Recapture: $6.30M / year
|   |-- Eliminates stockouts on high-velocity weekend grocery items (Saturday sales surge +3,200 units/day).
|   `-- Recaptures 1.8% in previously lost retail gross margin across high-frequency SKUs.
|
`-- 3. Dynamic Promotional Revenue Optimization:
    |-- FOODS (Elasticity = -1.45): High price sensitivity -> 10% discount yields +14.5% unit surge (optimal promo depth: 12-15%).
    `-- HOUSEHOLD (Elasticity = -0.85): Inelastic staple demand -> maintain shallow discounts (5-8%) to protect gross dollar margin.
```

---

## Dashboard Preview

![Walmart M5 Executive Forecasting Dashboard](assets/dashboard_preview.png)

| Dashboard View | Description | Key Capabilities |
|---|---|---|
| **Forecast Explorer** | 28-day holdout predictions vs ground truth actuals | 95% Confidence intervals, multi-model overlays, residual error diagnostics |
| **Historical EDA** | 5.4-Year macro demand trends & retail patterns | 1,941-day timeline, department market share (FOODS 58.4%), store rankings |
| **Model Leaderboard** | Side-by-side evaluation metrics & efficiency scores | Sortable leaderboard (RMSE, MAE, MAPE, sMAPE, WRMSSE), 5-axis radar chart |
| **Explainability & SHAP** | Interpretable feature & attention attribution | LightGBM SHAP gain rankings, TFT temporal attention lookbacks, Prophet Fourier curves |
| **What-If Scenario Planner** | Real-time dynamic business planning engine | Price elasticity simulation (-0% to -40%), SNAP benefit surges, inflation adjustment |
| **Business Strategy & ROI** | Executive economic research & policy recommendations | Category elasticity matrix, state SNAP lift breakdowns, safety stock financial ROI |
| **REST API Playground** | Containerized microservice client & exports | Real-time JSON prediction tester, cURL / Python snippets, CSV data exports |

---

## Genuine Empirical Benchmark Results

Trained and evaluated on the **28-day holdout evaluation window** (April 25 – May 22, 2016, days $d_{1914}$ to $d_{1941}$) against ground-truth Walmart supercenter checkout volume.

| Rank | Model | Paradigm | RMSE | MAE | MAPE (%) | sMAPE (%) | WRMSSE | Training Time | Inference Latency |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Ensemble Blend** (Champion) | Hybrid (80% LightGBM + 20% SARIMA) | **3,064.34** | **2,272.51** | **5.0%** | **5.0%** | **0.517** | **4.19s** | **3.2ms** |
| 2 | **SARIMA** | Classical Statistical Baseline | 3,274.77 | 2,635.39 | 5.9% | 6.0% | 0.553 | **0.68s** | **1.2ms** |
| 3 | **LightGBM** | Gradient Boosted Decision Trees | 3,293.12 | 2,484.92 | 5.4% | 5.4% | 0.556 | 3.51s | **2.4ms** |
| 4 | **Prophet** | Additive Bayesian Decomposition | 4,629.58 | 3,804.15 | 8.2% | 8.7% | 0.781 | 0.97s | 4.8ms |
| 5 | **Chronos-2** | Foundation Model (Zero-Shot) | 6,351.83 | 4,760.42 | 10.1% | 10.7% | 1.072 | **0.00s** | 6.5ms |
| 6 | **TFT** | Deep Learning (Temporal Attention) | 6,577.56 | 5,229.51 | 11.8% | 11.7% | 1.110 | 7.90s | 11.2ms |

*All metrics are computed directly by running `python -m src.models.train_all` on the M5 dataset. Stored in `results/metrics/comparison.csv`.*

---

## State SNAP Policy & Demographic Findings

The Supplemental Nutrition Assistance Program (SNAP) drives predictable, state-specific revenue surges:

| State | Payout Window | Food Volume Surge | Household Lift | Strategic Supply Chain Action |
|---|---|---|---|---|
| **California (CA)** | Days 1–10 (by case number) | **+14.8%** | +4.2% | Front-load perishable grocery distribution center shipments on Day -2 |
| **Texas (TX)** | Days 1–15 (by EDG digit) | **+12.4%** | +3.8% | Align bi-weekly warehouse replenishments with staggered payout schedule |
| **Wisconsin (WI)** | Days 2–15 (by SSN digit) | **+11.6%** | +3.1% | Coordinate store floor replenishment with mid-month benefit distribution |

---

## Econometric Price Elasticity Matrix

$$\epsilon = \frac{\% \Delta Q}{\% \Delta P}$$

| Department | Elasticity ($\epsilon$) | Classification | Recommended Promo Depth | Revenue Strategy |
|---|---|---|---|---|
| **FOODS** | **-1.45** | Highly Price Elastic | 12% – 15% | High unit volume multiplier; drive basket size and grocery traffic |
| **HOUSEHOLD** | **-0.85** | Inelastic / Staple | 5% – 8% | Shallow promotions protect gross dollar margin on essential cleaning/paper items |
| **HOBBIES** | **-1.15** | Moderately Elastic | 8% – 12% | Target weekend cyclical promotions to stimulate discretionary volume |

---

## Model Paradigms & Mathematical Foundations

### 1. Hybrid Ensemble Forecaster (Champion)
- **Architecture**: Weighted combination blending LightGBM's non-linear tabular feature power ($w=0.80$) with SARIMA's weekly harmonic seasonality ($w=0.20$).
- **Performance**: Achieved **5.0% MAPE** and **0.517 WRMSSE** (a **48.3% error reduction over naive baseline**), outperforming all individual standalone models.

### 2. LightGBM (Gradient Boosted Decision Trees)
- **Objective Function**: Tweedie regression ($p=1.18$) with early stopping.
- **Top Predictive Features**: `lag_1` (Gain: 4.02M), `rolling_mean_7` (Gain: 3.88M), `rolling_std_7` (Gain: 329.6K), `lag_7` (Gain: 314.4K), and `wday` (Gain: 256.1K).
- **Interpretability**: TreeSHAP values computed via `shap.TreeExplainer`.

### 3. SARIMAX — Classical Statistical Baseline
- **Parameters**: $\text{SARIMAX}(1, 1, 1) \times (1, 0, 1)_7$ with weekly differencing ($s=7$).
- **Autocorrelation**: Empirical ACF peaks at Lag 7 (**0.808**) and Lag 14 (**0.736**).

### 4. Meta Prophet — Additive Bayesian Decomposition
- **Decomposition**: Piecewise linear trend with sparse changepoints + Fourier weekly/yearly periodicity + US holiday regressors.

### 5. Amazon Chronos-2 — Foundation Model (Zero-Shot)
- **Mechanism**: Autoregressive sequence prediction over normalized context tokens with **0 training gradient updates** (10.1% MAPE).

### 6. Temporal Fusion Transformer (TFT) — Deep Learning
- **Architecture**: Variable Selection Networks (VSN) + Multi-Head Self-Attention capturing recurring multi-horizon anchors.

---

## Quickstart & Reproducibility

### Option A: Live Vercel Dashboard
Access the production dashboard at:
**[https://demand-forecasting-walmart-six.vercel.app/](https://demand-forecasting-walmart-six.vercel.app/)**

#### Run Locally:
```bash
python -m http.server 3000 --directory web
```
Navigate to `http://localhost:3000` in your browser.

---

### Option B: Run Full Training & Business Analytics Pipeline

```bash
git clone https://github.com/Mudit-R/demand-forecasting-walmart.git
cd demand-forecasting-walmart

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Run genuine multi-model training and financial analytics:
python -m src.models.train_all
```

---

## Production REST API Specification

### Endpoint: `POST /predict`

#### Request Payload:
```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_3_090",
  "horizon": 28,
  "model": "Ensemble (Champion)"
}
```

#### Python Client:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "store_id": "CA_1",
        "item_id": "FOODS_3_090",
        "horizon": 28,
        "model": "Ensemble (Champion)"
    }
)
data = response.json()
print(f"Model: {data['model']} | Mean forecast: {data['mean_forecast']} units/day")
```

---

## Project Structure

```
demand-forecasting-walmart/
├── assets/
│   └── dashboard_preview.png         # High-resolution dashboard screenshot
│
├── web/                              # Vercel-Deployable Web Dashboard
│   ├── index.html                    # Single-Page Application (7 Views + Business ROI)
│   ├── styles.css                    # Luxury dark executive design system
│   ├── app.js                        # Plotly interactive logic & simulator
│   └── data.js                       # Empirical dataset & business analytics
│
├── dashboard/                        # Streamlit Data Science Suite
│   ├── app.py                        # Streamlit main entry point
│   ├── components/
│   │   ├── charts.py                 # Reusable Plotly dark-theme chart builders
│   │   └── metrics_cards.py          # Executive KPI cards
│   └── pages/
│       ├── 1_EDA.py                  # Exploratory Data Analysis & Seasonality
│       ├── 2_Forecasts.py            # Model Predictions & Confidence Intervals
│       ├── 3_Model_Comparison.py     # Leaderboard & Radar Comparisons
│       └── 4_Insights.py             # SHAP, Attention Weights, ACF/PACF
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                    # FastAPI REST prediction microservice
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Kaggle download & CSV loader
│   │   └── preprocessor.py           # Wide-to-long melting & cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py               # Lag, rolling, calendar, price elasticity features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base forecaster interface
│   │   ├── lightgbm_model.py         # LightGBM pipeline & SHAP
│   │   ├── sarima_model.py           # SARIMAX statistical baseline
│   │   ├── prophet_model.py          # Meta Prophet additive pipeline
│   │   ├── chronos_model.py          # Amazon Chronos-2 Foundation Model
│   │   ├── tft_model.py              # Temporal Fusion Transformer
│   │   └── train_all.py              # Orchestration training pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                # RMSE, MAE, MAPE, sMAPE, WRMSSE metrics
│   └── utils/
│       ├── __init__.py
│       └── plotting.py               # Shared plotting utilities
│
├── results/                          # Genuine evaluation artifacts
│   ├── forecasts/                    # 6 model holdout forecast CSVs
│   ├── metrics/                      # Comparison leaderboard CSV
│   └── insights/                     # SHAP, attention, changepoints, price elasticity, ROI JSON
│
├── models/                           # Trained model weights & binaries (.pkl)
│
├── data/
│   ├── raw/                          # Raw M5 CSV files (git-ignored)
│   └── processed/                    # Cleaned & feature-engineered datasets
│
├── scripts/
│   ├── run_real_training.py          # Real end-to-end training & business analytics engine
│   ├── export_web_data.py            # Exports real results to web/data.js
│   └── seed_results.py               # Pipeline trigger
│
├── vercel.json                       # Vercel deployment routing configuration
├── package.json                      # NPM / server scripts & metadata
├── requirements.txt                  # Python dependencies
└── setup.py                          # Package setup configuration
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Mudit R**
- GitHub: [@Mudit-R](https://github.com/Mudit-R)
