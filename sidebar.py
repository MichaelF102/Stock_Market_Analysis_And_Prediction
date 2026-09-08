"""
Sidebar Module for Indian Equity Deep Learning Dashboard.
Provides unified sidebar navigation, stock selection, model selection, and execution controls.
"""

from typing import Dict, Any, Tuple
import streamlit as st
import pandas as pd
from utils import load_stock_universe_metadata


def render_app_sidebar(page: str = "main") -> Dict[str, Any]:
    """
    Renders the unified sidebar for the Indian Equity Deep Learning Dashboard.
    
    Parameters:
    - page: "main" or "prediction"
    
    Returns:
    - Dictionary with user selections (selected_stock, selected_model, compare_all).
    """
    st.sidebar.markdown("""
    <div style="padding: 10px 0px 20px 0px; text-align: center;">
        <h2 style="color: #38BDF8; margin: 0; font-size: 1.3rem;">Quant-DL Engine</h2>
        <p style="color: #64748B; margin: 4px 0 0 0; font-size: 0.8rem;">Indian Equities Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

    sidebar_controls = {
        "ticker": None,
        "company_name": None,
        "sector": None,
        "selected_model": "GRU",
        "compare_all": False,
        "predict_btn": False
    }

    if page in ["prediction", "ml_prediction"]:
        st.sidebar.markdown("### 🔍 Stock Selection")

        stocks_df = load_stock_universe_metadata()
        if stocks_df.empty:
            st.sidebar.error("Stock universe metadata unavailable.")
            return sidebar_controls

        # Optional Sector Filtering
        sectors = ["All Sectors"] + sorted([s for s in stocks_df["Sector"].dropna().unique() if s != "—"])
        selected_sector = st.sidebar.selectbox("Filter by Sector", sectors, index=0)

        if selected_sector != "All Sectors":
            filtered_df = stocks_df[stocks_df["Sector"] == selected_sector]
        else:
            filtered_df = stocks_df

        st.sidebar.caption(f"Available Universe: **{len(filtered_df)}** stocks")

        # Stock / Company Dropdown
        stock_options = list(filtered_df["Display_Label"].unique())
        
        # Default to RELIANCE.NS if available
        default_idx = 0
        for i, opt in enumerate(stock_options):
            if "RELIANCE.NS" in opt:
                default_idx = i
                break

        selected_label = st.sidebar.selectbox(
            "Select Stock",
            stock_options,
            index=default_idx,
            help="Choose an Indian stock from NSE"
        )

        selected_row = filtered_df[filtered_df["Display_Label"] == selected_label].iloc[0]
        sidebar_controls["ticker"] = selected_row["Ticker"]
        sidebar_controls["company_name"] = selected_row["Company_Name"]
        sidebar_controls["sector"] = selected_row.get("Sector", "Unknown")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 Model Selection")

        # Model Architecture Selection
        if page == "ml_prediction":
            model_options = ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Decision Tree"]
            selected_model = st.sidebar.selectbox(
                "Machine Learning Model",
                model_options,
                index=0,
                help="Select the ML model algorithm"
            )
            compare_all = st.sidebar.checkbox(
                "Compare All 5 ML Models",
                value=False,
                help="Evaluate all 5 Machine Learning algorithms simultaneously"
            )
        else:
            model_options = ["GRU", "LSTM", "BiLSTM", "Simple RNN"]
            selected_model = st.sidebar.selectbox(
                "Architecture",
                model_options,
                index=0,
                help="Select the deep learning model architecture"
            )
            compare_all = st.sidebar.checkbox(
                "Compare All Models (RNN vs LSTM vs BiLSTM vs GRU)",
                value=False,
                help="Evaluate all four architectures simultaneously on the selected stock"
            )

        sidebar_controls["selected_model"] = selected_model
        sidebar_controls["compare_all"] = compare_all

        # Predict Button
        st.sidebar.markdown("")
        sidebar_controls["predict_btn"] = st.sidebar.button("⚡ Generate Prediction", use_container_width=True)

    else:
        # Main Landing Page Sidebar Info
        st.sidebar.markdown("### 📋 Navigation")
        st.sidebar.info(
            "Use the page menu at the top or navigate to **Prediction** to forecast returns for any NSE stock."
        )

    # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
        <div><b>Data Window:</b> 2015 – Aug 2026</div>
        <div><b>Lookback:</b> 60 Trading Days</div>
        <div><b>Forecast:</b> Live Date → Next 7 Days</div>
        <div style="margin-top: 8px;">MSc Big Data Analytics Project</div>
    </div>
    """, unsafe_allow_html=True)

    return sidebar_controls
