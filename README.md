# 📈 Quant-DL: Cross-Stock Deep Learning & Machine Learning Predictive Engine for Indian Equities (NSE)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-38BDF8?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-3.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost / LightGBM / CatBoost](https://img.shields.io/badge/GBDT-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-10B981?style=flat)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-94A3B8?style=flat)](LICENSE)

An institutional-grade quantitative finance research platform and interactive web application investigating whether **recurrent deep learning models** (Simple RNN, LSTM, Bi-LSTM, GRU) and **gradient-boosted tree ensembles** (XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree) can learn **transferable temporal dynamics and cross-sectional alpha** across 125+ equities listed on the National Stock Exchange of India (NSE).

---

## 🏛️ Key Platform Highlights

- **Universe Coverage:** 125+ liquid National Stock Exchange (NSE) tickers across 11 sectors and 3 market-cap tiers (Mega, Mid, and Small-cap).
- **Historical Span:** 2015 to August 2026 (~11.5 years of daily adjusted OHLCV data, 300,000+ daily bars).
- **Scale-Free Feature Representation:** 31 dimensionless technical, momentum, volatility, and benchmark-relative features eliminating market-cap and price-level bias.
- **Strict Anti-Leakage Protocol:** StandardScaler and transformations fitted strictly on the 2015–2021 training block; temporal lookahead and cross-stock sequence overlaps are mathematically prevented.
- **Dual Out-of-Sample Evaluation:** Evaluated on both temporal out-of-sample data (2024–2026 for seen stocks) and cross-sectional held-out stocks (26 tickers never seen during training).
- **Full-Featured 6-Module Institutional Dashboard:** Interactive terminal offering fundamental intelligence & forensics, technical charting, ML tree ensembles, recurrent deep learning forecasts, empirical results leaderboards, and automated 9-page institutional PDF research dossiers.

---

## 🖥️ System Architecture & Web Application Modules

Launch the interactive dashboard with a single command:
```bash
streamlit run app.py
```

The application provides a unified financial terminal interface structured into **6 specialized modules**:

```
Quant-DL Application
│
├── 🏠 Landing Page (app.py)
│    └── Research methodology, temporal partitions, mathematical formulations, and master generalization metrics.
│
├── 🏛️ 01. Fundamentals Terminal (pages/01_Fundamentals_Terminal.py)
│    └── 10-year historical statements, DuPont 5-stage ROE breakdown, DCF Monte Carlo fair values, and forensic checks (Piotroski & Altman).
│
├── 📈 02. Technical Terminal (pages/02_Technical_Terminal.py)
│    └── Interactive Candlesticks, 8 strategy presets, technical regime score, pivot levels, and multi-indicator confluence matrix.
│
├── 🌲 03. Machine Learning Terminal (pages/03_ML_Terminal.py)
│    └── Tabular factor regression via LightGBM, XGBoost, CatBoost, Random Forest, and Decision Trees with feature importance attribution.
│
├── 🧠 04. Deep Learning Terminal (pages/04_DL_Terminal.py)
│    └── Multi-horizon neural sequence modeling across GRU, LSTM, Bi-LSTM, and Simple RNN with confidence intervals.
│
├── 🏆 05. Model Results & Scorecards (pages/05_Results.py)
│    └── Empirical benchmark leaderboard, dynamic model comparison bar charts with metric selectors, and 9 publication research figures.
│
└── 📑 06. Institutional Report Generator (pages/06_Report_Generation.py)
     └── One-click vector PDF generation creating an exhaustive 9-page institutional research memorandum with translucent finance watermark.
```

---

## 📊 Empirical Generalization Results

### 1. Deep Learning Out-of-Sample Performance (2024–2026 Test Period)

| Model | Partition | RMSE | MAE | Directional Accuracy | Information Coeff. (IC) | Strategy Sharpe | Strategy Sortino |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GRU** | **Unseen Stocks** | **0.0500** | **0.0353** | **52.14%** | **+0.1207** | **0.5725** | **0.9669** |
| **Bi-LSTM** | **Unseen Stocks** | 0.0502 | 0.0354 | **52.22%** | +0.0467 | **0.5892** | **0.9955** |
| **LSTM** | **Unseen Stocks** | 0.0501 | 0.0354 | **52.15%** | +0.0901 | 0.5738 | 0.9690 |
| **Simple RNN** | **Unseen Stocks** | 0.0504 | 0.0356 | 50.10% | +0.0348 | 0.3686 | 0.6247 |

> **Key Deep Learning Insight:** The Gated Recurrent Unit (GRU) achieved an Information Coefficient of **+0.1207** and a positive $R^2$ on held-out stocks never exposed during training, confirming that recurrent gating mechanisms learn genuine transferable temporal features across asset boundaries.

### 2. Machine Learning Benchmark Comparison

| Model | Seen Stocks DA | Unseen Stocks DA | Information Coeff. (IC) | Strategy Sharpe | Strategy Sortino | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 52.29% | **52.01%** | **+0.1669** | **0.6070** | **1.0113** | ~3.95s |
| **LightGBM** | **52.38%** | 51.88% | +0.1015 | 0.5936 | 0.9766 | **~0.76s** |
| **CatBoost** | 52.06% | 51.65% | +0.1366 | 0.5813 | 0.9682 | ~0.99s |
| **Random Forest** | 51.58% | 51.38% | +0.0869 | 0.4652 | 0.7567 | ~29.5s |
| **Decision Tree** | 51.39% | 51.21% | -0.0254 | 0.4328 | 0.6992 | ~1.90s |

---

## 🎯 Interactive Master Leaderboard Features

The **Results & Scorecards Terminal** (`pages/05_Results.py`) features an interactive benchmarking engine:
- **Dynamic Family Auto-Selection:** 
  - Selecting **"Deep Learning (Recurrent)"** automatically scopes the comparison multiselect to only the 4 DL models (`BiLSTM`, `GRU`, `LSTM`, `Simple RNN`).
  - Selecting **"Machine Learning (Ensemble)"** automatically scopes the multiselect to only the 5 ML models (`CatBoost`, `Decision Tree`, `LightGBM`, `Random Forest`, `XGBoost`).
  - Selecting **"All Model Types"** unifies all 9 architectures side-by-side.
- **Selectable Evaluation Metric Bar Charts:**
  Switch comparison bar charts across 8 metrics: Directional Accuracy (%), Information Coefficient (IC), Spearman Rank IC, Strategy Sharpe, Strategy Sortino, RMSE, MAE, and $R^2$.
- **Baseline Threshold Overlays:**
  Automatic reference lines showing the **50% Uninformative Random Walk Baseline** for Directional Accuracy, and the **0.0 Zero Alpha Baseline** for IC and Sharpe ratios.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python:** Version `3.10`, `3.11`, or `3.12` recommended.
- **Operating System:** Linux, macOS, or Windows (WSL2 recommended for Windows).

### 2. Clone the Repository
```bash
git clone https://github.com/MichaelF102/Stock_Market_Analysis_And_Prediction.git
cd Stock_Market_Analysis_And_Prediction
```

### 3. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Usage Guide

### 1. Running the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Use the sidebar or landing page navigation cards to switch between models, analyze company fundamentals, inspect technical chart indicators, view model scorecards, or generate institutional PDF research reports.

---

### 2. Retraining Machine Learning Models
To train the 5 Machine Learning algorithms (XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree) on the preprocessed feature dataset:
```bash
python train_ml_models.py
```
This script will:
- Load the 300,000+ observation parquet dataset (`data/processed/features.parquet`).
- Split temporally into Train (2015–2021), Validation (2022–2023), and Test (2024–2026).
- Train all 5 algorithms with early stopping and regularization.
- Save trained serialized model artifacts into `models/`.
- Export benchmark metrics to `results/ml_benchmark_metrics.csv` and feature importance rankings.

---

### 3. Running the End-to-End Deep Learning Pipeline
To run the full research pipeline (raw data verification, feature engineering, 60-day sequence tensor construction, deep neural network training, and metric evaluation):
```bash
python run_pipeline.py
```
*Optional stages can be executed individually:*
```bash
# Download raw equity & NIFTY 50 benchmark data
python -c "from src.data_loader import fetch_all_data; fetch_all_data()"

# Build 31 scale-free technical and macro features
python -c "from src.features import generate_features; generate_features()"

# Generate 3D sequence tensors (B, 60, 31)
python -c "from src.sequences import build_sequences; build_sequences()"

# Train Simple RNN, LSTM, and GRU models
python -c "from src.models import train_all_models; train_all_models()"
```

---

### 4. Running the Interactive Jupyter Notebooks
If you prefer an interactive notebook environment for research, experimentation, and paper figures:
```bash
# Launch JupyterLab or Notebook
jupyter lab
```
Open [`Indian_Equity_ML_and_DL_Training_Pipeline.ipynb`](file:///home/michaelfernandes/Desktop/Projects/Quant-DL/Indian_Equity_ML_and_DL_Training_Pipeline.ipynb) to execute each section step-by-step:
1. Environment verification & hardware checks.
2. Market dataset inspection & sector distributions.
3. Feature engineering & mathematical definitions.
4. Sequence construction & anti-leakage scaling.
5. Deep Learning training curves (Loss, MAE).
6. Machine Learning training & GBDT hyperparameter setup.
7. Out-of-sample seen vs unseen evaluation.
8. Feature importance and interpretability ranking.

---

### 5. Generating Institutional PDF Research Reports
1. In the running Streamlit web app, navigate to **06 Report Generation** in the sidebar.
2. Select any Indian equity ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`).
3. Review the live breakdown across Fundamental health, Technical regimes, and ML/DL forecasts.
4. Click **🚀 Generate Institutional PDF Report**.
5. The system dynamically renders a publication-grade, vector PDF complete with:
   - Executive dossier & composite moat scores.
   - 25-point Graham-Dodd investment checklist.
   - 10-year historical statements and DuPont 5-stage ROE breakdown.
   - Algorithmic technical summary with key support/resistance levels.
   - Deep Learning and Machine Learning consensus forecasts with confidence intervals.
   - Scenario analysis (Bull/Base/Bear) and fractional Kelly Criterion position sizing.

---

## 📁 Project Directory Structure

```text
Stock_Market_Analysis_And_Prediction/
├── app.py                                         # Streamlit Application Entrypoint & Landing Page
├── requirements.txt                               # Pinned Project Dependencies
├── sidebar.py                                     # Global Navigation & Model Selection Controls
├── run_pipeline.py                                # End-to-End DL Pipeline Runner
├── train_ml_models.py                             # ML Model Training & Evaluation Script
├── Indian_Equity_ML_and_DL_Training_Pipeline.ipynb # Complete ML & DL Pipeline Notebook
│
├── pages/                                         # Multi-Page Streamlit Dashboards
│   ├── 01_Fundamentals_Terminal.py                # 10-Year Financial Statements & Forensic Valuation
│   ├── 02_Technical_Terminal.py                   # High-Frequency Candlestick & Technical Regimes
│   ├── 03_ML_Terminal.py                          # GBDT & Random Forest Prediction Page
│   ├── 04_DL_Terminal.py                          # Deep Learning Forecasting Terminal
│   ├── 05_Results.py                              # Empirical Leaderboard & Metric Bar Charts
│   └── 06_Report_Generation.py                   # 9-Page Institutional PDF Dossier Engine
│
├── src/                                           # Core Quantitative Pipeline Modules
│   ├── config.py                                  # Paths, Dates, Splits & Hyperparameters
│   ├── universe.py                                # NSE 125+ Ticker Definitions & Sectors
│   ├── data_loader.py                             # yfinance Asynchronous Downloader
│   ├── features.py                                # 31 Scale-Free Feature Engineering Logic
│   ├── sequences.py                               # 60-Day Sliding Window Sequence Generator
│   ├── models.py                                  # Keras Simple RNN, LSTM, GRU Architectures
│   ├── evaluation.py                              # Financial Metrics (IC, Sharpe, Sortino, DA)
│   └── visualization.py                           # Matplotlib & Seaborn Publication Plots
│
├── utils/                                         # Helper Utilities & Report Generation
│   ├── helper.py                                  # Data Fetching & Financial Calculations
│   ├── report_generator.py                        # 9-Page Vector ReportLab PDF Generator
│   └── sidebar.py                                 # Financial Terminal Sidebar Controls
│
├── assets/                                        # Assets & Translucent Theme Watermarks
│   └── finance_bg_translucent.png                 # Institutional Background Watermark
│
├── data/                                          # Data Storage Layer
│   ├── raw/                                       # Parquet Files of Historical OHLCV
│   ├── processed/                                 # Parquet Engineered Features Dataset
│   └── sequences/                                 # Compressed .npz Sequence Tensors
│
├── models/                                        # Serialized Model Checkpoints & Scalers
│   ├── scaler.pkl                                 # StandardScaler fitted strictly on Train set
│   ├── gru.keras / lstm.keras / simplernn.keras   # Trained Keras Recurrent Models
│   ├── lightgbm.joblib                            # Trained LightGBM Model
│   ├── xgboost.joblib                             # Trained XGBoost Model
│   └── catboost.joblib                            # Trained CatBoost Model
│
└── results/                                       # Empirical Evaluation Artifacts
    ├── model_comparison.csv                       # DL Master Benchmark Results
    ├── ml_benchmark_metrics.csv                   # ML Benchmark Results
    ├── ml_feature_importances.parquet             # Quantitative Feature Rankings
    ├── cap_group_performance.csv                  # Market Cap Stratification Metrics
    ├── sector_performance.csv                     # Sectoral Performance Metrics
    ├── unseen_stock_performance.csv               # Pure Transferability Metrics
    └── plots/                                     # Publication-Grade Research Figures
```

---

## 🔒 Rigor & Methodological Safeguards

1. **Stationary Transformation:** All raw price levels ($Close_t$) are transformed into log returns, ratio spreads, or bounded oscillators to avoid unit-root instability.
2. **Strict Chronological Splitting:** The time horizon is partitioned strictly into Training (2015–2021), Validation (2022–2023), and Testing (2024–2026). No future data informs training.
3. **No Lookahead Data Contamination:** All normalization parameters ($\mu, \sigma$) are computed exclusively on the 2015–2021 training block and applied statically to validation and test sequences.
4. **Segmented Sequence Windows:** Sequences are constructed stock-by-stock, guaranteeing that a rolling 60-day window never crosses from one stock to another.

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for more information.

If you find this research or codebase useful for your academic work or quantitative trading research, please cite:
```bibtex
@misc{quant_dl_indian_equities,
  author = {Michael Fernandes},
  title = {Stock Price Analysis and Prediction Using Deep Learning and Machine Learning: Cross-Stock Generalization on Indian Equities},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MichaelF102/Stock_Market_Analysis_And_Prediction}}
}
```
