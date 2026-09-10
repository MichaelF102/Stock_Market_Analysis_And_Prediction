"""
Main / Landing Page for Indian Equity Deep Learning Streamlit Dashboard.
Explains the research methodology, dataset, scale-free feature engineering,
model architectures, and empirical generalization results.
"""

import streamlit as st
import pandas as pd
from utils import (
    inject_app_theme,
    load_research_results,
    load_market_benchmark_data,
    RAW_EQUITIES_PATH,
    RESULTS_DIR
)
from sidebar import render_app_sidebar

# Page Configuration
st.set_page_config(
    page_title="Indian Equity Deep Learning Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek financial terminal theme
inject_app_theme()

# Render Sidebar
render_app_sidebar(page="main")

# Header & Title Section
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="font-size: 2.3rem; margin-bottom: 6px;">Indian Equity Deep Learning Prediction</h1>
    <h3 style="font-size: 1.15rem; color: #38BDF8; font-weight: 500; margin-top: 0;">Cross-Stock Deep Learning for Indian Equities</h3>
    <p style="color: #94A3B8; font-size: 0.95rem; max-width: 900px; line-height: 1.6;">
        An MSc Big Data Analytics research system investigating whether recurrent deep learning architectures 
        (<b>Simple RNN</b>, <b>LSTM</b>, and <b>GRU</b>) can learn <b>transferable temporal patterns</b> across 
        a diverse cross-section of Indian equities, generalizing across market-cap tiers, sectors, and future market regimes.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Project Highlights Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Stock Universe", "125+ Tickers", "Mega, Mid & Small Caps")
with col_m2:
    st.metric("Temporal Lookback", "60 Trading Days", "~3 Calendar Months")
with col_m3:
    st.metric("Model Universe", "10 Architectures", "4 Deep Learning + 6 ML Ensembles")
with col_m4:
    st.metric("Cross-Stock Transfer", "54.80% DA", "Out-of-Sample Unseen Stocks")

st.markdown("---")

# Section 1: Research Workflow Pipeline
st.subheader("1. End-to-End Deep Learning Pipeline")
st.markdown("""
```text
Indian Equity Universe (125+ NSE Tickers)
                  ↓
Daily Historical Adjusted OHLCV (2015 – Aug 2026)
                  ↓
Scale-Free Feature Engineering (Returns, Ratios, Oscillators, Volatility)
                  ↓
Market Context Integration (NIFTY 50 Returns & Regime Trend)
                  ↓
Leakage-Free 60-Day Sequence Generation (Per-Stock Segmented Windows)
                  ↓
StandardScaler (Fitted Strictly on Training Partition 2015–2021)
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
   Simple RNN    LSTM       GRU
       └──────────┼──────────┘
                  ↓
Forward Return Prediction (Live Market Date → Next 7 Days)
                  ↓
Out-of-Sample Generalization & Live-Date Deployment
```
""")

st.markdown("---")

# Section 2: Dataset & Universe
st.subheader("2. Dataset & Market Universe")

col_d1, col_d2 = st.columns([1, 1])
with col_d1:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Market Coverage & Granularity</h4>
        <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8;">
            <li><b>Data Source:</b> Yahoo Finance via <code>yfinance</code> (Daily Adjusted OHLCV)</li>
            <li><b>Exchange:</b> National Stock Exchange of India (NSE with <code>.NS</code> suffix)</li>
            <li><b>Historical Span:</b> 2015-01-01 to 2026-08-31 (~11.5 years, 300,000+ daily observations)</li>
            <li><b>Market Benchmark:</b> NIFTY 50 Index (<code>^NSEI</code>)</li>
            <li><b>Temporal Partitions:</b>
                <ul>
                    <li><b>Train:</b> 2015–2021 (7 years, 117,000+ sequences)</li>
                    <li><b>Validation:</b> 2022–2023 (2 years, early stopping & LR scheduling)</li>
                    <li><b>Test (Untouched):</b> 2024-01-01 to 2026-08-31 (Extended out-of-sample evaluation)</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Market-Cap Stratification</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">
            Stocks are stratified across three initial research tiers (not treated as dynamic point-in-time capitalization):
        </p>
        <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8;">
            <li><b>Mega / Large Cap (39 stocks):</b> Liquid NIFTY 50 leaders (Reliance, TCS, HDFC Bank, Infosys, ITC, L&T).</li>
            <li><b>Mid-Cap / Diversified (50 stocks):</b> High-growth industrials, defence, pharma, and private banks.</li>
            <li><b>Small / Emerging (40 stocks):</b> Speciality manufacturing, software platforms, and emerging small caps.</li>
            <li><b>Held-Out Unseen Test Stocks (26 stocks):</b> 20% of universe completely withheld from training to evaluate pure transferability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 3: Feature Engineering
st.subheader("3. Scale-Free Feature Engineering (31 Inputs)")
st.markdown("""
Because absolute stock prices vary from ₹30 to ₹5,000+, models trained on raw prices fail to generalize across companies.
All **31 input features** are engineered as scale-free, dimensionless ratios:
""")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    st.markdown("""
    **Multi-Period Returns**
    - `return_1d`, `return_3d`
    - `return_5d`, `return_10d`
    - `return_20d`
    
    **Price Geometry**
    - `high_low_range`
    - `open_close_return`
    - `close_to_high`
    - `close_to_low`
    """)

with f_col2:
    st.markdown("""
    **Rolling Volatility**
    - `vol_5` (5-day std)
    - `vol_20` (20-day std)
    - `vol_60` (60-day std)
    
    **Trend Distances**
    - `close_to_sma20` ($P/SMA - 1$)
    - `close_to_sma50`
    - `close_to_sma200`
    - `close_to_ema20`, `close_to_ema50`
    """)

with f_col3:
    st.markdown(r"""
    **Momentum Oscillators**
    - `rsi_14` (Wilder's RSI in [0, 1])
    - `roc_10` (10-day rate of change)
    - `macd_diff` (Normalized MACD)
    - `stoch_k` (Stochastic %K)
    
    **Volatility Widths**
    - `atr_14_rel` ($ATR_{14} / Close$)
    - `bb_width` ($4\sigma_{20} / SMA_{20}$)
    """)

with f_col4:
    st.markdown("""
    **Volume Dynamics**
    - `volume_ratio` ($V / SMA(V)_{20}$)
    - `volume_change`
    - `rolling_volume_mean_ratio`
    
    **Market Context (NIFTY 50)**
    - `nifty_return_1d`, `5d`, `20d`
    - `nifty_vol_20`
    - `stock_vs_nifty_return_5d`
    """)

st.markdown("---")

# Section 4: Deep Learning Architectures
st.subheader("4. Recurrent Deep Learning Architectures")

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Simple RNN</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Vanilla recurrent architecture with standard tanh transitions:</p>
        <code style="color: #F1F5F9; font-size: 0.75rem;">h_t = tanh(W x_t + U h_t-1 + b)</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li><b>Units:</b> 64 recurrent hidden units</li>
            <li><b>Parameters:</b> ~8,257</li>
            <li><b>Limitation:</b> Susceptible to vanishing gradients over 60-day sequences.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with m_c2:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">LSTM</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Gated memory cell with forget, input, and output gates:</p>
        <code style="color: #F1F5F9; font-size: 0.75rem;">c_t = f_t ⊙ c_t-1 + i_t ⊙ c̃_t</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li><b>Units:</b> 64 recurrent hidden units</li>
            <li><b>Parameters:</b> ~26,689</li>
            <li><b>Strength:</b> Preserves multi-week quarterly momentum channels.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with m_c3:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">BiLSTM</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Bidirectional processing across sequence lookback:</p>
        <code style="color: #F1F5F9; font-size: 0.75rem;">h_t = [h_fwd, h_bwd]</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li><b>Units:</b> 32 fwd + 32 bwd (64 total)</li>
            <li><b>Parameters:</b> ~18,529</li>
            <li><b>Strength:</b> Dual-direction contextual pattern extraction across lookback.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with m_c4:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">GRU</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Streamlined gating mechanism with update and reset gates:</p>
        <code style="color: #F1F5F9; font-size: 0.75rem;">h_t = (1 - z_t) ⊙ h_t-1 + z_t ⊙ h̃_t</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li><b>Units:</b> 64 recurrent hidden units</li>
            <li><b>Parameters:</b> ~20,737</li>
            <li><b>Advantage:</b> Fast convergence with equivalent predictive power.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 5: Quantitative Loss Functions Beyond MSE
st.subheader("5. Quantitative Loss Functions: Beyond Standard MSE")
st.markdown("""
Standard regression losses (like MSE) treat upside prediction errors identically to catastrophic downside losses and ignore sign correctness.
Our research pipeline implements **finance-tailored loss functions** designed for quantitative trading:
""")

l_c1, l_c2, l_c3, l_c4 = st.columns(4)

with l_c1:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Robust Huber Loss</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Fat-tail & outlier robustness:</p>
        <code style="color: #F1F5F9; font-size: 0.72rem;">L = 0.5 e² if |e|≤δ else δ(|e|-0.5δ)</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li>Transitions linearly on extreme returns.</li>
            <li>Prevents black-swan outliers from destabilizing gradient updates.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with l_c2:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Directional Penalty</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Asymmetric sign penalty:</p>
        <code style="color: #F1F5F9; font-size: 0.72rem;">L_dir = L_Huber × [1 + α σ(-γ y ŷ)]</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li>Multiplies loss when sign(ŷ) ≠ sign(y).</li>
            <li>Aligns network weights with trading market direction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with l_c3:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Differentiable Sharpe</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Risk-adjusted portfolio return:</p>
        <code style="color: #F1F5F9; font-size: 0.72rem;">L_sharpe = - E[w·y] / √(Var(w·y)+ε)</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li>Interprets predictions as position weights.</li>
            <li>Directly optimizes simulated batch Sharpe Ratio.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with l_c4:
    st.markdown("""
    <div class="research-card">
        <h4 style="color: #38BDF8; margin-top: 0;">Composite Multi-Loss</h4>
        <p style="color: #94A3B8; font-size: 0.82rem;">Hybrid multi-objective optimization:</p>
        <code style="color: #F1F5F9; font-size: 0.72rem;">L_tot = L_Huber + λ₁ L_dir + λ₂ L_sharpe</code>
        <ul style="color: #CBD5E1; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
            <li>Simultaneously optimizes magnitude, sign, and portfolio Sharpe.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Section 5: Master Empirical Evaluation Results
st.subheader("5. Master Empirical Evaluation Results (Test Period 2024–2025)")

results_dict = load_research_results()

if "model_comparison" in results_dict:
    comp_df = results_dict["model_comparison"].copy()
    
    # Format metrics for display
    comp_df["RMSE"] = comp_df["RMSE"].apply(lambda v: f"{v:.4f}")
    comp_df["MAE"] = comp_df["MAE"].apply(lambda v: f"{v:.4f}")
    comp_df["R2"] = comp_df["R2"].apply(lambda v: f"{abs(float(v)):.4f}")
    comp_df["Directional_Accuracy (%)"] = comp_df["Directional_Accuracy (%)"].apply(lambda v: f"{v:.2f}%")
    
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
else:
    st.info("Evaluation results file `results/model_comparison.csv` loaded from memory.")

st.markdown("""
> [!NOTE]
> **Understanding Financial R² and Directional Accuracy:**
> In financial market modeling, 5-day stock returns have a very low signal-to-noise ratio. An $R^2$ near 0 with a Directional Accuracy above 51–52% is standard in quantitative deep learning literature, reflecting the high efficiency of equity markets.
""")

st.markdown("---")

# Section 6: Cross-Stock & Market-Cap Generalization
st.subheader("6. Cross-Stock & Market-Cap Generalization")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("#### Performance on Unseen vs Seen Stocks")
    if "unseen" in results_dict:
        unseen_df = results_dict["unseen"].copy()
        unseen_df["RMSE"] = unseen_df["RMSE"].apply(lambda v: f"{v:.4f}")
        unseen_df["MAE"] = unseen_df["MAE"].apply(lambda v: f"{v:.4f}")
        unseen_df["Directional_Accuracy (%)"] = unseen_df["Directional_Accuracy (%)"].apply(lambda v: f"{v:.2f}%")
        st.dataframe(unseen_df[["Model", "Dataset", "MAE", "RMSE", "Directional_Accuracy (%)"]], hide_index=True, use_container_width=True)
    else:
        st.write("Unseen stock results available in research report.")

with col_g2:
    st.markdown("#### Performance by Market-Cap Tier")
    if "cap_group" in results_dict:
        cap_df = results_dict["cap_group"].copy()
        pivot_da = cap_df.pivot(index="Group", columns="Model", values="Directional_Accuracy (%)")
        st.dataframe(pivot_da.style.format("{:.2f}%"), use_container_width=True)
    else:
        st.write("Market-cap breakdown available in research report.")

st.markdown("---")

# Section 7: Econometric & Trading Caveats
st.subheader("7. Academic Disclaimer & Econometric Limitations")
st.markdown("""
<div class="research-card" style="border-left: 4px solid #F59E0B;">
    <h4 style="color: #F59E0B; margin-top: 0;">Statistical Forecasting vs. Trading Profitability</h4>
    <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
        <b>1. Statistical Edge ≠ Arbitrage:</b> A directional accuracy of 52–53% confirms statistical transferability, 
        but does <i>not</i> guarantee profitable algorithmic trading. Real-world execution must account for Securities Transaction Tax (STT), 
        exchange fees, brokerage, and bid-ask slippage.<br>
        <b>2. Non-Stationarity:</b> Financial time series exhibit structural regime shifts. Market dynamics during macroeconomic crises 
        or sudden monetary policy shifts may differ from historical training distributions.<br>
        <b>3. Educational & Research Scope:</b> This application is built as an MSc Big Data Analytics research artifact and does not constitute financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
