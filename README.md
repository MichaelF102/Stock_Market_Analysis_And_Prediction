# 📈 Quant-DL: Cross-Stock Deep Learning & Machine Learning Predictive Engine for Indian Equities (NSE)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-38BDF8?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-3.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost / LightGBM / CatBoost](https://img.shields.io/badge/GBDT-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-10B981?style=flat)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-94A3B8?style=flat)](LICENSE)

An institutional-grade quantitative finance research platform and interactive web application investigating whether **recurrent deep learning models** (Simple RNN, LSTM, Bi-LSTM, GRU) and **gradient-boosted tree ensembles** (XGBoost, LightGBM, CatBoost, Random Forest) can learn **transferable temporal dynamics and cross-sectional alpha** across 125+ equities listed on the National Stock Exchange of India (NSE).

---

## 🏛️ Key Platform Highlights

- **Universe Coverage:** 125+ liquid National Stock Exchange (NSE) tickers across 11 sectors and 3 market-cap tiers (Mega, Mid, and Small-cap).
- **Historical Span:** 2015 to August 2026 (~11.5 years of daily adjusted OHLCV data, 300,000+ daily bars).
- **Scale-Free Feature Representation:** 31 dimensionless technical, momentum, volatility, and benchmark-relative features eliminating market-cap and price-level bias.
- **Strict Anti-Leakage Protocol:** StandardScaler and transformations fitted strictly on the 2015–2021 training block; temporal lookahead and cross-stock sequence overlaps are mathematically prevented.
- **Dual Out-of-Sample Evaluation:** Evaluated on both temporal out-of-sample data (2024–2026 for seen stocks) and cross-sectional held-out stocks (26 tickers never seen during training).
- **Full-Featured 5-Module Dashboard:** Interactive terminal offering live deep learning forecasts, technical charting, multi-year fundamental analysis, tree ensemble consensus, and automated 9-page institutional PDF research dossiers.

---

## 🖥️ System Architecture & Web Application Modules

Launch the interactive dashboard with a single command:
```bash
streamlit run app.py
```

The application provides a unified financial terminal interface structured into 5 specialized modules:

```
Quant-DL Application
│
├── 🏠 Landing Page (app.py)
│    └── Research methodology, temporal partitions, mathematical formulations, and master generalization metrics.
│
├── 🧠 01. Deep Learning Prediction (pages/01_DL_Prediction.py)
│    └── Live multi-day forward projections using GRU, LSTM, Bi-LSTM, and Simple RNN with confidence intervals.
│
├── 📊 02. Algorithmic Technical Terminal (pages/02_Technical_Terminal.py)
│    └── High-frequency Candlestick charts, EMAs (20/50/200), RSI, MACD, Bollinger Bands, ATR, SuperTrend, and Volume profiling.
│
├── 📑 03. Institutional Fundamental Analysis (pages/03_Fundamentals.py)
│    └── Comprehensive 5-year balance sheet & P&L statements, DuPont ROE decomposition, solvency, and valuation multiples.
│
├── 🌲 04. Machine Learning Ensembles (pages/04_ML_Prediction.py)
│    └── Tabular factor regression via LightGBM, XGBoost, CatBoost, Random Forest, and Decision Trees with feature importance attribution.
│
└── 📥 05. Institutional Report Generator (pages/05_Report_Generation.py)
     └── One-click vector PDF generation creating an exhaustive 9-page institutional research report with translucent finance watermark.
```

---

## 📊 Empirical Generalization Results

### 1. Deep Learning Out-of-Sample Performance (2024–2026 Test Period)

| Model | Partition | RMSE | MAE | Directional Accuracy | Information Coeff. (IC) | Strategy Sharpe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GRU** | **Unseen Stocks** | **0.0500** | **0.0353** | **52.14%** | **0.1207** | **0.5725** |
| **Bi-LSTM** | **Unseen Stocks** | 0.0502 | 0.0354 | **52.22%** | 0.0467 | **0.5892** |
| **LSTM** | **Unseen Stocks** | 0.0501 | 0.0354 | **52.15%** | 0.0901 | 0.5738 |
| **Simple RNN** | **Unseen Stocks** | 0.0504 | 0.0356 | 50.10% | 0.0348 | 0.3686 |

> **Key Research Finding:** The Gated Recurrent Unit (GRU) achieved an Information Coefficient of **0.1207** and a positive $R^2$ on held-out stocks that were never seen during model training, confirming genuine cross-stock temporal feature transferability.

### 2. Machine Learning Benchmark Comparison

| Model | Seen Stocks DA | Unseen Stocks DA | Information Coeff. (IC) | Strategy Sharpe | Strategy Sortino | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 52.29% | **52.01%** | **0.1669** | **0.6070** | **1.0113** | ~3.95s |
| **LightGBM** | **52.38%** | 51.88% | 0.1015 | 0.5936 | 0.9766 | **~0.76s** |
| **CatBoost** | 52.06% | 51.65% | 0.1366 | 0.5813 | 0.9682 | ~0.99s |
| **Random Forest** | 51.58% | 51.38% | 0.0869 | 0.4652 | 0.7567 | ~29.5s |
| **Decision Tree** | 51.39% | 51.21% | -0.0254 | 0.4328 | 0.6992 | ~1.90s |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python:** Version `3.10`, `3.11`, or `3.12` recommended.
- **Operating System:** Linux, macOS, or Windows (WSL2 recommended for Windows).

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Quant-DL.git
cd Quant-DL
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
Open your browser at `http://localhost:8501`. Use the sidebar to switch between models, analyze company fundamentals, inspect technical chart indicators, or generate institutional PDF research reports.

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
# Generate the unified training notebook (ML + DL)
python build_unified_training_notebook.py

# Launch JupyterLab
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
1. In the running Streamlit web app, click **05 Report Generation** in the sidebar.
2. Select any Indian equity ticker (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`).
3. Review the live breakdown across Fundamental health, Technical regimes, and ML/DL forecasts.
4. Click **Generate & Download 9-Page Institutional PDF Report**.
5. The system dynamically renders a publication-grade, vector PDF complete with:
   - Executive dossier & composite moat scores.
   - 25-point Graham-Dodd investment checklist.
   - 5-year balance sheet, income statement, and cash flow analysis.
   - Algorithmic technical summary with key support/resistance levels.
   - Deep Learning and Machine Learning consensus forecasts with confidence intervals.
   - Subtle institutional watermark on every page without obscuring tabular data.

---

## 📁 Project Directory Structure

```text
Quant-DL/
├── app.py                                         # Streamlit Application Entrypoint & Main Page
├── requirements.txt                               # Pinned Project Dependencies
├── sidebar.py                                     # Global Navigation & Model Selection Controls
├── run_pipeline.py                                # End-to-End DL Pipeline Runner
├── train_ml_models.py                             # ML Model Training & Evaluation Script
├── build_unified_training_notebook.py             # Script to build unified research notebook
├── Indian_Equity_ML_and_DL_Training_Pipeline.ipynb # Complete ML & DL Pipeline Notebook
│
├── pages/                                         # Multi-Page Streamlit Dashboards
│   ├── 01_DL_Prediction.py                        # Deep Learning Forecasting Terminal
│   ├── 02_Technical_Terminal.py                   # High-Frequency Candlestick & Technicals
│   ├── 03_Fundamentals.py                         # 5-Year Financial Statements & Valuation
│   ├── 04_ML_Prediction.py                        # GBDT & Random Forest Prediction Page
│   └── 05_Report_Generation.py                   # 9-Page Institutional PDF Dossier Engine
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
│   ├── lightgbm_model.joblib                      # Trained LightGBM Model
│   ├── xgboost_model.joblib                       # Trained XGBoost Model
│   └── catboost_model.joblib                      # Trained CatBoost Model
│
└── results/                                       # Empirical Evaluation Artifacts
    ├── model_comparison.csv                       # DL Master Benchmark Results
    ├── ml_benchmark_metrics.csv                   # ML Benchmark Results
    ├── ml_feature_importances.parquet             # Quantitative Feature Rankings
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
  journal = {GitHub Repository}
}
```
