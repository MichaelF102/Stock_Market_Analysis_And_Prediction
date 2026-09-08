"""
Generates 'Indian_Equity_Machine_Learning_Research.ipynb' covering classical and gradient-boosted
tree ensemble machine learning models for cross-stock Indian equity return forecasting.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "Indian_Equity_Machine_Learning_Research.ipynb"


def generate_ml_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source_text: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        })

    def add_code(source_code: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_code.strip().split("\n")]
        })

    # =========================================================================
    # Section 01: Project Overview
    # =========================================================================
    add_md("""# Cross-Stock Machine Learning on Indian Equities
## Benchmarking Tree & Gradient Boosted Ensembles
**MSc Big Data Analytics / Quantitative Machine Learning Research**

---

### Abstract & Research Objectives
This research notebook establishes a standardized quantitative benchmark comparing **5 Machine Learning Tree & Gradient Boosted Regressors** across a cross-section of Indian equities on the National Stock Exchange (NSE):
1. **Decision Tree Regressor (CART)**
2. **Random Forest Regressor (Bagging Ensembles)**
3. **LightGBM Regressor (Histogram-based GBDT)**
4. **XGBoost Regressor (Exact/Approximate Split GBDT)**
5. **CatBoost Regressor (Ordered Boosting & Oblivious Trees)**

We investigate whether tree-based ensembles and non-linear inductive biases can capture cross-stock factor interactions and out-perform linear and recurrent deep learning baselines on out-of-sample unseen market periods and unseen assets.
""")

    # =========================================================================
    # Section 02: Mathematical Frameworks
    # =========================================================================
    add_md("""## 1. Mathematical Formulations of Evaluated Models

### A. Decision Tree Regressor (CART)
Recursively partitions feature space into $M$ orthogonal hyper-rectangles $R_1, \\dots, R_M$ minimizing sum of squared errors:
$$\\min_{j, s} \\left[ \\sum_{\\mathbf{x}_i \\in R_1(j,s)} (y_i - \\bar{y}_{R_1})^2 + \\sum_{\\mathbf{x}_i \\in R_2(j,s)} (y_i - \\bar{y}_{R_2})^2 \\right]$$

### B. Random Forest Regressor (Breiman, 2001)
A bootstrap aggregating (bagging) ensemble of $B$ de-correlated decision trees with random feature subsampling ($m = \\sqrt{p}$ or $p/3$):
$$\\hat{f}_{\\text{RF}}(\\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B T_b(\\mathbf{x})$$

### C. LightGBM (Ke et al., 2017)
Utilizes **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)** with leaf-wise (best-first) tree growth to optimize training speed and cache efficiency on dense financial tabular data.

### D. XGBoost (Chen & Guestrin, 2016)
Optimizes a second-order Taylor expansion of the objective function with explicit tree complexity penalties:
$$\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^N \\left[ g_i f_t(\\mathbf{x}_i) + \\frac{1}{2} h_i f_t^2(\\mathbf{x}_i) \\right] + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^T w_j^2$$

### E. CatBoost (Prokhorenkova et al., 2018)
Implements **Ordered Boosting** to overcome prediction shift (target leakage) inherent in standard gradient boosting algorithms, utilizing symmetric oblivious decision trees as base learners.
""")

    # =========================================================================
    # Section 03: Environment Setup & Imports
    # =========================================================================
    add_md("""## 2. Environment Setup & Library Imports""")
    add_code("""import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Set Plot Styles
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
""")

    # =========================================================================
    # Section 04: Data Loading & Split Inspection
    # =========================================================================
    add_md("""## 3. Dataset Loading & Leakage-Free Partitions""")
    add_code("""# Load precomputed scale-free features
features_path = Path("data/processed/features.parquet")
if not features_path.exists():
    features_path = Path("../data/processed/features.parquet")

df = pd.read_parquet(features_path)
df["Date"] = pd.to_datetime(df["Date"])
print(f"Loaded Features: {df.shape[0]:,} records, {df.shape[1]} columns, {df['Ticker'].nunique()} unique tickers.")

# Feature columns (Scale-free returns, RSI, Volatility, MACD, Moving Average Ratios, NIFTY Context)
exclude_cols = ["Date", "Ticker", "Sector", "Target_5D", "Forward_Return_7D", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
target_col = "Target_5D" if "Target_5D" in df.columns else "Forward_Return_7D"

print(f"Total Predictive Quantitative Features: {len(feature_cols)}")
""")

    add_code("""# Temporal Splits & Seen/Unseen Universe
train_mask = (df["Date"] <= "2021-12-31")
val_mask = (df["Date"] > "2021-12-31") & (df["Date"] <= "2023-12-31")
test_mask = (df["Date"] >= "2024-01-01") & (df["Date"] <= "2026-08-31")

train_df = df[train_mask].dropna(subset=feature_cols + [target_col])
val_df = df[val_mask].dropna(subset=feature_cols + [target_col])
test_df = df[test_mask].dropna(subset=feature_cols + [target_col])

print(f"Train Partition (2015–2021):     {len(train_df):,} samples")
print(f"Validation Partition (2022–2023): {len(val_df):,} samples")
print(f"Test Partition (2024–2026):       {len(test_df):,} samples")

# Standardize Feature Space
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
y_train = train_df[target_col].values

X_val = scaler.transform(val_df[feature_cols].values)
y_val = val_df[target_col].values

X_test = scaler.transform(test_df[feature_cols].values)
y_test = test_df[target_col].values
""")

    # =========================================================================
    # Section 05: Model Training & Hyperparameter Setup
    # =========================================================================
    add_md("""## 4. Machine Learning Model Training & Optimization""")
    add_code("""models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_split=50, min_samples_leaf=25, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=40, min_samples_leaf=20, n_jobs=-1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
    "CatBoost": cb.CatBoostRegressor(iterations=150, learning_rate=0.04, depth=6, l2_leaf_reg=3.0, random_seed=42, verbose=False, thread_count=-1)
}

trained_models = {}
feature_importances = {}
evaluation_records = []

for name, model in models.items():
    t0 = time.time()
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    trained_models[name] = model
    
    # Feature Importances
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    elif hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()
    else:
        imp = np.zeros(len(feature_cols))
        
    df_imp = pd.DataFrame({"Feature": feature_cols, "Importance": imp}).sort_values(by="Importance", ascending=False)
    feature_importances[name] = df_imp
    
    # Out-of-Sample Predictions
    preds = model.predict(X_test)
    
    # Quantitative Metrics
    da = (np.sign(preds) == np.sign(y_test)).mean() * 100.0
    ic = np.corrcoef(preds, y_test)[0, 1]
    rank_ic = pd.Series(preds).corr(pd.Series(y_test), method='spearman')
    
    # Strategy Sharpe
    signals = np.sign(preds)
    strat_ret = signals * y_test
    sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-8)) * np.sqrt(252 / 5)
    
    # Downside Sortino
    neg_rets = strat_ret[strat_ret < 0]
    downside_std = neg_rets.std() if len(neg_rets) > 0 else strat_ret.std()
    sortino = (strat_ret.mean() / (downside_std + 1e-8)) * np.sqrt(252 / 5)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    evaluation_records.append({
        "Model": name,
        "Directional Accuracy (%)": da,
        "Information Coefficient (IC)": ic,
        "Rank IC": rank_ic,
        "Strategy Sharpe": sharpe,
        "Strategy Sortino": sortino,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Train Time (s)": round(elapsed, 2)
    })
    print(f"  ✓ {name}: DA = {da:.2f}% | IC = {ic:+.4f} | Sharpe = {sharpe:.3f} | Time = {elapsed:.2f}s")
""")

    # =========================================================================
    # Section 06: Out-of-Sample Comparative Scorecard
    # =========================================================================
    add_md("""## 5. Out-of-Sample Empirical Scorecard & Benchmark Analysis""")
    add_code("""results_df = pd.DataFrame(evaluation_records).sort_values(by="Information Coefficient (IC)", ascending=False)
display(results_df.style.background_gradient(cmap="viridis", subset=["Directional Accuracy (%)", "Information Coefficient (IC)", "Strategy Sharpe"]))
""")

    # =========================================================================
    # Section 07: Feature Importance Analysis
    # =========================================================================
    add_md("""## 6. Global Feature Importance & Factor Attribution""")
    add_code("""# Top 15 Features for LightGBM, XGBoost, and Random Forest
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

top_models = ["Random Forest", "LightGBM", "XGBoost"]
colors = ["#38BDF8", "#00E676", "#F59E0B"]

for idx, m_name in enumerate(top_models):
    df_top = feature_importances[m_name].head(12)
    sns.barplot(data=df_top, x="Importance", y="Feature", ax=axes[idx], color=colors[idx])
    axes[idx].set_title(f"{m_name} Top 12 Predictive Factors", fontweight='bold')
    axes[idx].set_xlabel("Relative Importance")

plt.tight_layout()
plt.show()
""")

    # =========================================================================
    # Section 08: Model Serialization
    # =========================================================================
    add_md("""## 7. Model Serialization & Streamlit Deployment Artifacts""")
    add_code("""models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

# Save individual models
for name, model in trained_models.items():
    clean_name = name.lower().replace(" ", "_")
    joblib.dump(model, models_dir / f"{clean_name}.joblib")

# Save consolidated dictionary
joblib.dump(trained_models, models_dir / "ml_models.joblib")
joblib.dump(scaler, models_dir / "scaler.pkl")

# Save evaluation results
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_dir / "ml_benchmark_metrics.csv", index=False)
print("All 5 ML models and evaluation scorecards successfully saved to models/ and results/.")
""")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    generate_ml_notebook()
