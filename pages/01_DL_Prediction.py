"""
Deep Learning Prediction Page for Indian Equity Forecasting.
Generates single-stock 7-day forward return predictions and estimated future prices
using trained Simple RNN, LSTM, BiLSTM, and GRU models.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from utils import (
    inject_app_theme,
    load_trained_models,
    load_feature_scaler,
    load_market_benchmark_data,
    fetch_stock_historical_data,
    process_features_and_predict,
    create_price_forecast_chart,
    create_models_comparison_chart,
    LOOKBACK,
    FORECAST_HORIZON
)
from sidebar import render_app_sidebar

# Page Setup
st.set_page_config(
    page_title="Deep Learning Forecast | Quant-DL",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
inject_app_theme()

# Render Sidebar with Prediction Controls
controls = render_app_sidebar(page="prediction")

# Main Page Header
st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="font-size: 2.1rem; margin-bottom: 4px;">Deep Learning Live Return Forecast (Next 7 Days)</h1>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0;">
        Generate model-based forward return forecasts from the <b>live market date</b> to the <b>next 7 days</b> using recurrent deep learning representations (<b>RNN</b>, <b>LSTM</b>, <b>BiLSTM</b>, <b>GRU</b>).
    </p>
</div>
""", unsafe_allow_html=True)

ticker = controls.get("ticker")
company_name = controls.get("company_name", ticker)
selected_model_name = controls.get("selected_model", "GRU")
compare_all = controls.get("compare_all", False)

if not ticker:
    st.info("👈 Please select a stock from the sidebar to begin forecasting.")
    st.stop()

# Load Core Artifacts
models_dict = load_trained_models()
scaler = load_feature_scaler()
nifty_df = load_market_benchmark_data()

if not models_dict:
    st.error("No trained models could be loaded. Please ensure models are present in `models/`.")
    st.stop()

if scaler is None:
    st.error("Prediction scaler not found. Model requires the training-fitted `StandardScaler`.")
    st.stop()

# Fetch Stock Historical Data (Up to Live Date)
with st.spinner(f"Loading live market data for {ticker}..."):
    stock_df = fetch_stock_historical_data(ticker)

if stock_df.empty or len(stock_df) < LOOKBACK:
    st.error(
        f"Insufficient historical observations for **{ticker}** ({len(stock_df)} observations). "
        f"At least {LOOKBACK} trading days are required to build a valid sequence."
    )
    st.stop()

# Stock Overview Header Card
start_date_str = stock_df["Date"].min().strftime("%Y-%m-%d")
live_date_str = stock_df["Date"].max().strftime("%Y-%m-%d")
latest_close_val = float(stock_df["Close"].iloc[-1])

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.markdown(f"**Stock:** `{ticker}`")
    st.caption(f"{company_name}")
with col_h2:
    st.markdown(f"**Historical Span:** `{start_date_str}` to `{live_date_str}`")
    st.caption(f"Total Observations: **{len(stock_df):,}** days")
with col_h3:
    st.markdown(f"**Live Market Close:** `₹{latest_close_val:,.2f}`")
    st.caption(f"As of {live_date_str}")
with col_h4:
    st.markdown(f"**Active Model:** `{selected_model_name}`")
    st.caption("Lookback: 60 Days | Horizon: Next 7 Days")

st.markdown("---")

# Execute Prediction Pipeline
try:
    if selected_model_name not in models_dict:
        st.error(f"Selected model '{selected_model_name}' is not currently available.")
        st.stop()

    active_model = models_dict[selected_model_name]
    
    # Run exact scale-free feature inference pipeline
    pred_results = process_features_and_predict(
        stock_df=stock_df,
        nifty_df=nifty_df,
        model=active_model,
        scaler=scaler,
        model_name=selected_model_name
    )

    forecast_date_str = pred_results["forecast_target_date"].strftime("%Y-%m-%d")

    # Display Prominent Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric("Selected Model", pred_results["model_name"])
    with m2:
        st.metric("Live Close Price", f"₹{pred_results['latest_close']:,.2f}")
    with m3:
        st.metric(
            "Predicted 7-Day Return",
            f"{pred_results['pred_return_pct']:+.2f}%",
            delta=f"{pred_results['pred_return_pct']:+.2f}%"
        )
    with m4:
        diff_price = pred_results["estimated_price"] - pred_results["latest_close"]
        st.metric(
            "Estimated 7-Day Price",
            f"₹{pred_results['estimated_price']:,.2f}",
            delta=f"₹{diff_price:+,.2f}"
        )
    with m5:
        st.markdown("**Directional Outlook**")
        st.markdown(
            f'<div style="margin-top: 8px;"><span class="{pred_results["badge_class"]}">{pred_results["direction"]}</span></div>'
            f'<div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Target: {forecast_date_str}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Price Chart + Forecast
    st.subheader(f"Live Price & Next 7-Day Forecast Projection ({selected_model_name})")
    fig = create_price_forecast_chart(
        history_df=pred_results["clean_history_df"],
        latest_date=pred_results["latest_date"],
        latest_close=pred_results["latest_close"],
        pred_return=pred_results["pred_return"],
        est_price=pred_results["estimated_price"],
        model_name=selected_model_name,
        lookback_days=160
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model Comparison Section (if enabled)
    if compare_all:
        st.markdown("---")
        st.subheader("Model Comparison: RNN vs. LSTM vs. BiLSTM vs. GRU (Next 7 Days)")
        
        comparison_list = []
        for m_name in ["Simple RNN", "LSTM", "BiLSTM", "GRU"]:
            if m_name in models_dict:
                m_obj = models_dict[m_name]
                res = process_features_and_predict(
                    stock_df, nifty_df, m_obj, scaler, m_name
                )
                comparison_list.append(res)
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            comp_table_df = pd.DataFrame([
                {
                    "Architecture": r["model_name"],
                    "Predicted 7D Return": f"{r['pred_return_pct']:+.2f}%",
                    "Estimated Price in 7D": f"₹{r['estimated_price']:,.2f}",
                    "Target Date": r["forecast_target_date"].strftime("%Y-%m-%d"),
                    "Direction": r["direction"]
                }
                for r in comparison_list
            ])
            st.dataframe(comp_table_df, hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 10px;">
                <b>Consensus Note:</b> Comparing multiple architectures provides robustness checks. 
                Gated models (LSTM and GRU) typically offer superior stability across multi-week trends.
            </div>
            """, unsafe_allow_html=True)

        with col_t2:
            bar_fig = create_models_comparison_chart(comparison_list)
            st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # Research Interpretation & Model Details Columns
    col_info1, col_info2 = st.columns([1, 1])

    with col_info1:
        st.markdown("#### Research Interpretation")
        st.markdown(f"""
        <div class="research-card">
            <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6; margin-bottom: 8px;">
                The <b>{selected_model_name}</b> model estimates a forward return of 
                <b>{pred_results['pred_return_pct']:+.2f}%</b> for <b>{company_name}</b> across the next 7 days.
            </p>
            <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6; margin-bottom: 8px;">
                From the live market close of <b>₹{pred_results['latest_close']:,.2f}</b> on <b>{live_date_str}</b>, 
                this corresponds to an estimated target level of <b>₹{pred_results['estimated_price']:,.2f}</b> 
                for <b>{forecast_date_str}</b> (t+7 days).
            </p>
            <p style="color: #64748B; font-size: 0.8rem; line-height: 1.5; margin-top: 12px;">
                <i>Disclaimer: The displayed forecast is a statistical point prediction derived from scale-free technical indicators. 
                It is intended solely for academic research and does not constitute investment advice or a guaranteed price target.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("#### Model Architecture Metadata")
        try:
            param_count = active_model.count_params()
        except Exception:
            param_count = "N/A"

        st.markdown(f"""
        <div class="research-card">
            <table style="width: 100%; font-size: 0.88rem; color: #CBD5E1;">
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Model Architecture:</b></td><td>{selected_model_name}</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Input Sequence:</b></td><td>60 Trading Days</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Input Features:</b></td><td>31 Scale-Free Indicators</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Forecast Horizon:</b></td><td>Live Date → Next 7 Days</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Target Type:</b></td><td>Forward Return (ŷ = P_t+7d / P_t - 1)</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Trainable Parameters:</b></td><td>{param_count:,}</td></tr>
                <tr><td style="padding: 6px 0; color: #94A3B8;"><b>Scaling Standard:</b></td><td>StandardScaler (Train 2015–2021)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Expandable Feature Vector Transparency
    with st.expander("🔍 Show Latest 31 Model Features (Pre-Scaled Human Readable Inputs)"):
        st.markdown("The table below details the exact 31 scale-free feature values extracted at the latest time $t$ before normalization:")
        
        feat_items = list(pred_results["latest_feature_vector"].items())
        half = len(feat_items) // 2 + 1
        c_f1, c_f2 = st.columns(2)
        
        with c_f1:
            df_f1 = pd.DataFrame(feat_items[:half], columns=["Feature Name", "Pre-Scaled Value"])
            df_f1["Pre-Scaled Value"] = df_f1["Pre-Scaled Value"].apply(lambda v: f"{v:+.4f}" if isinstance(v, (int, float)) else str(v))
            st.dataframe(df_f1, hide_index=True, use_container_width=True)
            
        with c_f2:
            df_f2 = pd.DataFrame(feat_items[half:], columns=["Feature Name", "Pre-Scaled Value"])
            df_f2["Pre-Scaled Value"] = df_f2["Pre-Scaled Value"].apply(lambda v: f"{v:+.4f}" if isinstance(v, (int, float)) else str(v))
            st.dataframe(df_f2, hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Error executing prediction pipeline for {ticker}: {e}")
    st.exception(e)
