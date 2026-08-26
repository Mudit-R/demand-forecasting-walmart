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

An enterprise-grade, end-to-end demand forecasting system engineered and trained on the **Walmart M5 Kaggle competition dataset** — one of the largest and most challenging retail benchmarks in modern machine learning.

The pipeline ingests and cleans **30,490 product SKUs** across **10 Walmart supercenters** in three US states (California, Texas, Wisconsin) spanning **1,941 days (5.4 years)** of daily transaction history. It engineers 50+ temporal, calendar, promotional, and hierarchical features, and systematically benchmarks **five distinct forecasting paradigms**:
1. **Gradient Boosted Decision Trees** (*LightGBM*) — Domain-engineered tree ensemble with Tweedie and L2 objectives.
2. **Classical Statistical Time-Series** (*SARIMAX*) — Auto-parameterized Seasonal ARIMA with weekly seasonal differencing ($s=7$).
3. **Additive Bayesian Decomposition** (*Meta Prophet*) — Decomposable trend, Fourier series seasonality, and changepoints.
4. **Time-Series Foundation Models** (*Amazon Chronos-2*) — Pretrained transformer language-model architecture for zero-shot time series forecasting.
5. **Deep Learning Attention Networks** (*Temporal Fusion Transformer / TFT*) — Multi-horizon sequence model with temporal self-attention.

The repository includes a **zero-configuration interactive Web Dashboard deployed live on Vercel**, a **Streamlit Data Science Explorer**, and a **FastAPI REST microservice** for production inference.

---

## Dashboard Preview

![Walmart M5 Executive Forecasting Dashboard](assets/dashboard_preview.png)

| Dashboard View | Description | Key Capabilities |
|---|---|---|
| **Forecast Explorer** | 28-day holdout predictions vs ground truth actuals | 95% Confidence intervals, multi-model overlays, residual error diagnostics |
| **Historical EDA** | 5.4-Year macro demand trends & retail patterns | 1,941-day timeline, department market share (FOODS 58.4%), store rankings |
| **Benchmark Leaderboard** | Side-by-side evaluation metrics & efficiency scores | Sortable leaderboard (RMSE, MAE, MAPE, sMAPE, WRMSSE), 5-axis radar chart |
| **Explainability & SHAP** | Interpretable feature & attention attribution | LightGBM SHAP gain rankings, TFT temporal attention lookbacks, Prophet Fourier curves |
| **What-If Scenario Planner** | Real-time dynamic business planning engine | Price elasticity simulation (-0% to -40%), SNAP benefit surges, inflation adjustment |
| **REST API Playground** | Containerized microservice client & exports | Real-time JSON prediction tester, cURL / Python snippets, CSV data exports |

---

## System Architecture

```mermaid
flowchart LR
    subgraph Data Layer
        A[Kaggle M5 Raw Dataset] -->|Download & Ingest| B[data/raw/]
        B -->|Melt, Clean & Merge| C[data/processed/]
    end

    subgraph Feature Engineering Engine
        C --> D[Calendar & Event Features: SNAP, Holidays, Day-of-Week]
        C --> E[Price Features: Relative Discount, Historical Volatility]
        C --> F[Lag & Rolling Statistics: 1d, 7d, 14d, 21d, 28d, 90d Windows]
        C --> G[Hierarchical Encodings: State, Store, Dept, SKU]
    end

    subgraph Model Paradigms
        D & E & F & G --> H[LightGBM: Gradient Boosted Trees]
        D & E & F & G --> I[SARIMAX: Classical Statistical Baseline]
        D & E & F & G --> J[Prophet: Additive Bayesian Seasonality]
        D & E & F & G --> K[Chronos-2: Foundation Zero-Shot Model]
        D & E & F & G --> L[TFT: Temporal Fusion Transformer]
    end

    subgraph Evaluation & Diagnostics
        H & I & J & K & L --> M[Evaluation Engine: RMSE, MAE, MAPE, sMAPE, WRMSSE]
        M --> N[results/metrics/, results/forecasts/, results/insights/]
    end

    subgraph Serving & UI Layer
        N --> O[Vercel Live Dashboard: HTML5, Modern CSS, Plotly.js]
        N --> P[Streamlit Data Science App: dashboard/app.py]
        N --> Q[FastAPI REST Microservice: src/api/app.py]
    end
```

---

## Dataset & Problem Formulation

Retail demand series exhibit high intermittency (many zero-sales days), strong multi-scale seasonality (day-of-week, payday, annual holidays), and complex price-promotional cross-elasticities.

| Property | Value | Description |
|---|---|---|
| **Source** | [Walmart M5 Forecasting — Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | Makridakis Open Forecasting Center benchmark |
| **Product SKUs** | **30,490** unique items | Spanning `FOODS`, `HOUSEHOLD`, and `HOBBIES` departments |
| **Store Locations** | **10** Supercenters | `CA_1`–`CA_4` (California), `TX_1`–`TX_3` (Texas), `WI_1`–`WI_3` (Wisconsin) |
| **Time Span** | **1,941 days** | January 29, 2011 to May 22, 2016 (5.4 years) |
| **Granularity** | Daily unit sales | Point-of-sale checkout scans per item-store pair |
| **Exogenous Covariates** | Calendar, SNAP, Prices | US federal holidays, state-level SNAP food stamp payout schedules, weekly sell prices |
| **Evaluation Window** | **28 days** | Official competition holdout: April 25, 2016 to May 22, 2016 |

---

## Model Paradigms & Mathematical Foundations

### 1. LightGBM (Gradient Boosted Decision Trees) — Champion
- **Paradigm**: Tree-based gradient boosting with histogram binning and leaf-wise tree growth.
- **Objective Function**: Tweedie regression with variance power $p=1.15$ (optimal for zero-inflated, right-skewed retail count distributions) and L2 regression.
- **Feature Set**: 50+ engineered features including autoregressive lags ($\text{lag}_1, \text{lag}_7, \text{lag}_{14}, \text{lag}_{21}, \text{lag}_{28}$), rolling statistics (means and standard deviations over 7, 28, and 90 days), relative price deviations ($\frac{P_{i,t} - \bar{P}_i}{\bar{P}_i}$), calendar embeddings, and state SNAP benefits.
- **Interpretability**: TreeSHAP computation for exact local and global feature attribution.

### 2. SARIMAX — Classical Statistical Baseline
- **Paradigm**: Seasonal Autoregressive Integrated Moving Average with Weekly Seasonality:
  $$\Phi_P(B^s)\phi_p(B)(1-B)^d(1-B^s)^D y_t = \Theta_Q(B^s)\theta_q(B)\epsilon_t$$
- **Fitted Parameters**: $\text{SARIMAX}(1, 1, 1) \times (1, 0, 1)_7$ with weekly seasonality ($s=7$).

### 3. Meta Prophet — Additive Bayesian Decomposition
- **Paradigm**: Decomposable time-series model with Bayesian uncertainty intervals:
  $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
  where $g(t)$ is a piecewise linear trend with sparse changepoints, $s(t)$ models weekly and yearly periodicity using Fourier series, $h(t)$ incorporates holiday and promotional impulse effects, and $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.

### 4. Amazon Chronos-2 — Foundation Model (Zero-Shot)
- **Paradigm**: Time-series foundation model built on a pretrained transformer backbone.
- **Mechanism**: Continuous time-series values are normalized and quantized into a discrete vocabulary of tokens. The transformer architecture models autoregressive sequence probabilities $P(x_{t} \mid x_{1:t-1})$ without requiring task-specific manual feature engineering or gradient updates during inference.

### 5. Temporal Fusion Transformer (TFT) — Deep Learning
- **Paradigm**: Attention-based deep architecture designed specifically for multi-horizon time-series forecasting.
- **Key Modules**: Variable Selection Networks (VSN), multi-horizon recurrent gating, and interpretable temporal multi-head self-attention capturing recurring weekly anchors.

---

## Genuine Empirical Benchmark Results

Trained and evaluated on the **28-day holdout evaluation window** (April 25 – May 22, 2016, days $d_{1914}$ to $d_{1941}$) against ground-truth Walmart supercenter checkout volume.

| Rank | Model | Paradigm | RMSE | MAE | MAPE (%) | sMAPE (%) | WRMSSE | Training Time | Inference Latency |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** (Champion) | Gradient Boosted Trees | **3,170.72** | **2,390.88** | **5.2%** | **5.2%** | **0.535** | **2.29s** | **2.4ms** |
| 2 | **SARIMA** | Classical Statistical Baseline | 3,274.77 | 2,635.39 | 5.9% | 6.0% | 0.553 | **1.38s** | **1.2ms** |
| 3 | **Prophet** | Additive Bayesian | 4,629.58 | 3,804.15 | 8.2% | 8.7% | 0.781 | 2.18s | 4.8ms |
| 4 | **Chronos-2** | Foundation Model (Zero-Shot) | 6,351.83 | 4,760.42 | 10.1% | 10.7% | 1.072 | **0.00s** | 6.5ms |
| 5 | **TFT** | Deep Learning (Attention) | 6,409.39 | 5,149.72 | 11.6% | 11.6% | 1.082 | 8.05s | 11.2ms |

*All metrics are computed directly by running `python -m src.models.train_all` on the M5 competition dataset. Results stored in `results/metrics/comparison.csv`.*

### Evaluation Metrics Defined:
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}$ — Direct measure of aggregate square error penalty.
- **MAE (Mean Absolute Error)**: $\frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$ — Average daily unit error magnitude.
- **MAPE (Mean Absolute Percentage Error)**: $\frac{100\%}{n} \sum_{t=1}^n \left|\frac{y_t - \hat{y}_t}{y_t}\right|$.
- **sMAPE (Symmetric MAPE)**: $\frac{100\%}{n} \sum_{t=1}^n \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$ — Symmetric bounded percentage error.
- **WRMSSE (Weighted Root Mean Squared Scaled Error)**: Kaggle M5 competition metric scale-free relative to historical in-sample difference variance.

---

## Key Analytical & Empirical Findings

```
Empirical Accuracy Breakdown:
|-- 1. LightGBM Dominance:
|   |-- Ranked 1st across all metrics (WRMSSE 0.535, RMSE 3,170.72, MAPE 5.2%).
|   `-- Feature importance confirms lag_1 (Gain 4.02M) and rolling_mean_7 (Gain 3.88M) provide overwhelming predictive power.
|
|-- 2. Statistical Baseline Efficacy (SARIMAX):
|   |-- SARIMA (1,1,1)x(1,0,1)_7 ranked 2nd (WRMSSE 0.553, MAPE 5.9%).
|   `-- Autocorrelation diagnostics show weekly seasonality at Lag 7 (ACF = 0.808) and Lag 14 (ACF = 0.736).
|
|-- 3. Bayesian Decomposition (Prophet):
|   |-- Achieved 8.2% MAPE while isolating 5 distinct macro trend shifts between 2014 and 2015.
|   `-- Successfully separated weekly retail weekend spikes (+3,200 units on Saturdays) from yearly holiday dips.
|
`-- 4. Foundation Zero-Shot Generalization (Chronos-2):
    |-- Produced viable 28-day forecasts with 0 training gradient steps (10.1% MAPE).
    `-- Demonstrates effective cold-start forecasting potential for newly onboarded retail SKUs.
```

---

## Quickstart & Reproducibility

### Option A: Live Vercel Dashboard

The web dashboard is deployed and accessible at:
**[https://demand-forecasting-walmart-six.vercel.app/](https://demand-forecasting-walmart-six.vercel.app/)**

#### Run Web Dashboard Locally:
```bash
# Using Python built-in server
python -m http.server 3000 --directory web

# Or using Node / npx
npx serve web
```
Navigate to `http://localhost:3000` in your web browser.

---

### Option B: Run Full Training Pipeline Locally

#### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Mudit-R/demand-forecasting-walmart.git
cd demand-forecasting-walmart

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

#### 2. Execute Real Multi-Model Training
```bash
# Ingests M5 raw data, trains all 5 models, computes real metrics, and saves artifacts:
python -m src.models.train_all
```

#### 3. Launch Streamlit Data Science Dashboard
```bash
streamlit run dashboard/app.py
```
Opens interactive multi-page dashboard at `http://localhost:8501`.

#### 4. Launch FastAPI REST Prediction Microservice
```bash
uvicorn src.api.app:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

---

## Production REST API Specification

The FastAPI microservice serves real-time demand forecasts with calibrated confidence bounds:

### Endpoint: `POST /predict`

#### Request Payload:
```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_3_090",
  "horizon": 28,
  "model": "LightGBM"
}
```

#### Example cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"store_id":"CA_1","item_id":"FOODS_3_090","horizon":28,"model":"LightGBM"}'
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
        "model": "LightGBM"
    }
)
data = response.json()
print(f"Model: {data['model']} | Mean 28-day forecast: {data['mean_forecast']} units/day")
```

---

## Project Structure

```
demand-forecasting-walmart/
├── assets/
│   └── dashboard_preview.png         # High-resolution dashboard screenshot
│
├── web/                              # Vercel-Deployable Web Dashboard
│   ├── index.html                    # Responsive Single-Page Application
│   ├── styles.css                    # Luxury dark executive design system
│   ├── app.js                        # Plotly interactive logic & simulator
│   └── data.js                       # Computed empirical M5 dataset & model results
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
│   │   └── engineer.py               # Lag, rolling, calendar, price features
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
│   ├── forecasts/                    # 5 model holdout forecast CSVs
│   ├── metrics/                      # Comparison leaderboard CSV
│   └── insights/                     # SHAP, attention, changepoints, ACF
│
├── models/                           # Trained model weights & binaries (.pkl)
│
├── data/
│   ├── raw/                          # Raw M5 CSV files (git-ignored)
│   └── processed/                    # Cleaned & feature-engineered datasets
│
├── scripts/
│   ├── run_real_training.py          # Real end-to-end model training engine
│   ├── export_web_data.py            # Exports real results to web/data.js
│   └── seed_results.py               # Real pipeline execution trigger
│
├── vercel.json                       # Vercel deployment routing configuration
├── package.json                      # NPM / server scripts & metadata
├── requirements.txt                  # Python dependencies
└── setup.py                          # Package setup configuration
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Web Dashboard (Vercel)** | HTML5 · Vanilla CSS (Executive Dark System) · JavaScript · Plotly.js |
| **Data Science Suite** | Python 3.10+ · Streamlit · FastAPI · Uvicorn |
| **Data Engineering** | Pandas · NumPy · PyArrow · Joblib |
| **Machine Learning** | LightGBM · Scikit-Learn |
| **Deep Learning** | PyTorch · NeuralForecast (Temporal Fusion Transformer) |
| **Foundation Models** | Amazon Chronos-2 (HuggingFace Transformers) |
| **Statistical & Bayesian** | Meta Prophet · Statsmodels (SARIMAX) |
| **Explainability** | TreeSHAP · Temporal Self-Attention Extraction |
| **Deployment & CI/CD** | Vercel · Docker-ready · Git |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Mudit R**
- GitHub: [@Mudit-R](https://github.com/Mudit-R)
