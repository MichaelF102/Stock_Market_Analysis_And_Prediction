"""
Machine Learning Forecast Terminal for Indian Equities.
Deploys and benchmarks 5 Machine Learning Regressors:
1. LightGBM Regressor (Histogram-based GBDT)
2. XGBoost Regressor (Exact/Approximate Split GBDT)
3. CatBoost Regressor (Ordered Boosting & Oblivious Trees)
4. Random Forest Regressor (Bagging Ensembles)
5. Decision Tree Regressor (CART)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from utils import (
    inject_app_theme,
    load_trained_ml_models,
    load_feature_scaler,
    load_market_benchmark_data,
    fetch_stock_historical_data,
    process_features_and_predict_ml,
    create_price_forecast_chart,
    create_models_comparison_chart,
    load_ml_benchmark_metrics,
    LOOKBACK,
    FORECAST_HORIZON
)
from src.ml_models import extract_feature_importances, get_ml_model_definitions
from sidebar import render_app_sidebar

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="ML Forecast & Benchmarks | Quant-DL",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
inject_app_theme()

# Render Sidebar with ML Prediction Controls
controls = render_app_sidebar(page="ml_prediction")

# Main Page Header
st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="font-size: 2.1rem; margin-bottom: 4px;">Machine Learning Return Forecast & Ensembles</h1>
    <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0;">
        Generate model-based forward return forecasts from the <b>live market date</b> to the <b>next 7 days</b> using bagging and gradient-boosted tree ensembles (<b>LightGBM</b>, <b>XGBoost</b>, <b>CatBoost</b>, <b>Random Forest</b>, <b>Decision Tree</b>).
    </p>
</div>
""", unsafe_allow_html=True)

ticker = controls.get("ticker")
company_name = controls.get("company_name", ticker)
selected_model_name = controls.get("selected_model", "LightGBM")
compare_all = controls.get("compare_all", False)

if not ticker:
    st.info("👈 Please select a stock from the sidebar to begin forecasting.")
    st.stop()

# Load Core Artifacts
models_dict = load_trained_ml_models()
scaler = load_feature_scaler()
nifty_df = load_market_benchmark_data()

# Fallback: if models are not yet trained on disk, initialize standard definitions
if not models_dict:
    st.warning("⚠️ Serialized ML models not detected on disk. Using initialized reference models. Run `python train_ml_models.py` to train full checkpoint weights.")
    models_dict = get_ml_model_definitions()

if scaler is None:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

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
    st.markdown(f"**Active ML Model:** `{selected_model_name}`")
    st.caption("Lookback: 60 Days | Horizon: Next 7 Days")

st.markdown("---")

# Execute Prediction Pipeline
try:
    if selected_model_name not in models_dict:
        # Fallback to LightGBM or first available
        active_model_name = list(models_dict.keys())[0]
        active_model = models_dict[active_model_name]
    else:
        active_model_name = selected_model_name
        active_model = models_dict[selected_model_name]
    
    # Run exact scale-free feature inference pipeline
    pred_results = process_features_and_predict_ml(
        stock_df=stock_df,
        nifty_df=nifty_df,
        model=active_model,
        scaler=scaler,
        model_name=active_model_name
    )

    forecast_date_str = pred_results["forecast_target_date"].strftime("%Y-%m-%d")

    # Display Prominent Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric("Selected Algorithm", pred_results["model_name"])
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
    st.subheader(f"Live Price & Next 7-Day Forecast Projection ({active_model_name})")
    fig = create_price_forecast_chart(
        history_df=pred_results["clean_history_df"],
        latest_date=pred_results["latest_date"],
        latest_close=pred_results["latest_close"],
        pred_return=pred_results["pred_return"],
        est_price=pred_results["estimated_price"],
        model_name=active_model_name,
        lookback_days=160
    )
    st.plotly_chart(fig, use_container_width=True)

    # Multi-Model Comparison Section
    if compare_all or st.checkbox("Show 5 ML Regressors Comparison", value=True):
        st.markdown("---")
        st.subheader("🌲 All 5 Machine Learning Regressors Comparison (Next 7 Days)")
        
        all_ml_names = ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Decision Tree"]
        comparison_list = []
        for m_name in all_ml_names:
            if m_name in models_dict:
                m_obj = models_dict[m_name]
                try:
                    res = process_features_and_predict_ml(
                        stock_df, nifty_df, m_obj, scaler, m_name
                    )
                    comparison_list.append(res)
                except Exception:
                    pass
        
        if comparison_list:
            col_t1, col_t2 = st.columns([1.1, 1.1])
            with col_t1:
                comp_table_df = pd.DataFrame([
                    {
                        "Algorithm": r["model_name"],
                        "Predicted 7D Return": f"{r['pred_return_pct']:+.2f}%",
                        "Estimated Price": f"₹{r['estimated_price']:,.2f}",
                        "Direction": r["direction"]
                    }
                    for r in comparison_list
                ])
                st.dataframe(comp_table_df, hide_index=True, use_container_width=True)
                
                # Ensemble Consensus Average
                mean_ret = np.mean([r["pred_return_pct"] for r in comparison_list])
                mean_price = latest_close_val * (1.0 + mean_ret / 100.0)
                st.markdown(f"""
                <div class="research-card" style="padding:14px 18px; margin-top:10px;">
                    <span style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">5-MODEL ENSEMBLE CONSENSUS</span>
                    <div style="font-size:1.3rem; font-weight:800; color:{'#00E676' if mean_ret>=0 else '#EF4444'}; font-family:'JetBrains Mono'; margin-top:4px;">
                        {mean_ret:+.2f}% &nbsp;·&nbsp; Target: ₹{mean_price:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_t2:
                bar_fig = create_models_comparison_chart(comparison_list)
                st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance & Quantitative Scorecard
    col_feat, col_bench = st.columns([1, 1])

    with col_feat:
        st.markdown(f"#### 🔍 Feature Importance & Attribution ({active_model_name})")
        
        feature_names = list(pred_results["latest_feature_vector"].keys())
        df_imp = extract_feature_importances(active_model, feature_names)
        
        if not df_imp.empty:
            top10 = df_imp.head(10)
            fig_imp = px.bar(
                top10,
                x="Normalized_Importance",
                y="Feature",
                orientation="h",
                color="Normalized_Importance",
                color_continuous_scale="Viridis",
                title=f"Top 10 Driving Features for {active_model_name}"
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(color="#94A3B8"),
                height=340,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance weights are computed across tree ensemble nodes.")

    with col_bench:
        st.markdown("#### 🏆 Out-of-Sample Empirical Scorecard (2024–2026)")
        
        bench_df = load_ml_benchmark_metrics()
        unseen_bench = bench_df[bench_df["Dataset"].str.contains("Unseen")][
            ["Model", "Directional_Accuracy (%)", "Information_Coefficient (IC)", "Strategy_Sharpe", "Strategy_Sortino"]
        ].sort_values(by="Information_Coefficient (IC)", ascending=False)
        
        st.dataframe(unseen_bench, hide_index=True, use_container_width=True)
        st.caption("Out-of-sample benchmark metrics evaluated strictly on untouched test partitions (2024–2026).")

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
