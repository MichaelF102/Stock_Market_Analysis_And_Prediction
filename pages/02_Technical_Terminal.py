import streamlit as st
import pandas as pd
import numpy as np
from utils.sidebar import render_sidebar
from utils.helper import load_data, get_ticker_info, inject_custom_theme, _fmt_num, drop_holiday_nans
from technical.registry import INDICATOR_REGISTRY
from technical.scoring import calculate_technical_score
from technical.signals import generate_signals_for_dataframe, check_crossover_above, check_crossover_below
from charts.price import create_price_chart
from charts.oscillators import create_oscillator_charts
from charts.volume import create_volume_chart
from config.presets import render_strategy_explainer

st.set_page_config(page_title="Technical Terminal", layout="wide")
inject_custom_theme()

ticker, company, exchange, period, interval, region = render_sidebar()

df = load_data(ticker, period=period, interval=interval)
df = drop_holiday_nans(df)

if df.empty:
    st.error(f"No price data found for {ticker}.")
    st.stop()

info = get_ticker_info(ticker)

close_val = float(df["Close"].iloc[-1])
prev_close_val = float(df["Close"].iloc[-2]) if len(df) > 1 else close_val
calc_change = close_val - prev_close_val
calc_pct_change = (calc_change / prev_close_val * 100.0) if prev_close_val > 0 else 0.0

curr_price = float(info.get("currentPrice")) if (info.get("currentPrice") is not None and not pd.isna(info.get("currentPrice"))) else close_val
change = float(info.get("change")) if (info.get("change") is not None and not pd.isna(info.get("change"))) else calc_change
pct_change = float(info.get("percentChange")) if (info.get("percentChange") is not None and not pd.isna(info.get("percentChange"))) else calc_pct_change
currency = info.get("currency") or ("INR" if region == "India" else "USD")
curr_symbol = "₹" if currency == "INR" else "$"

day_low = float(df["Low"].iloc[-1])
day_high = float(df["High"].iloc[-1])
high_52 = float(info.get("high52")) if (info.get("high52") is not None and not pd.isna(info.get("high52"))) else float(df["High"].max())
low_52 = float(info.get("low52")) if (info.get("low52") is not None and not pd.isna(info.get("low52"))) else float(df["Low"].min())

latest_vol = int(df["Volume"].iloc[-1]) if pd.notna(df["Volume"].iloc[-1]) else 0
avg_vol_20 = int(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 and pd.notna(df["Volume"].rolling(20).mean().iloc[-1]) else latest_vol

# 1. INSTITUTIONAL INSTRUMENT HEADER
col_head, col_stats = st.columns([1.5, 2])

with col_head:
    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<h2 style='margin:0; font-size:1.8rem; font-weight:800; color:#FFFFFF;'>{company}</h2>"
        f"<span style='color:#38BDF8; font-weight:600; font-size:1.0rem;'>{ticker} · {exchange} ({region})</span>"
        f"</div>"
        f"<div style='font-size:2.2rem; font-weight:800; font-family:\"JetBrains Mono\", monospace; color:#FFFFFF; margin-bottom:4px;'>"
        f"{curr_symbol}{curr_price:,.2f} &nbsp;"
        f"<span style='font-size:1.2rem; color:{'#00E676' if change>=0 else '#FF5252'}; font-weight:700;'>"
        f"{change:+.2f} ({pct_change:+.2f}%)</span>"
        f"</div>",
        unsafe_allow_html=True
    )

with col_stats:
    st.markdown(
        f"<div class='glass-card' style='padding:16px 20px; margin-bottom:0;'>"
        f"<div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; font-size:0.85rem;'>"
        f"<div><div style='color:#94A3B8;'>Day Range</div><div style='font-weight:700;'>{curr_symbol}{day_low:,.2f} — {curr_symbol}{day_high:,.2f}</div></div>"
        f"<div><div style='color:#94A3B8;'>52W Range</div><div style='font-weight:700;'>{curr_symbol}{low_52:,.2f} — {curr_symbol}{high_52:,.2f}</div></div>"
        f"<div><div style='color:#94A3B8;'>Latest Vol</div><div style='font-weight:700;'>{_fmt_num(latest_vol)}</div></div>"
        f"<div><div style='color:#94A3B8;'>20D Avg Vol</div><div style='font-weight:700;'>{_fmt_num(avg_vol_20)}</div></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# Pre-calculate essential indicators for Technical Regime & Signals
core_inds = {}
core_inds["SMA20"] = INDICATOR_REGISTRY["SMA"]["func"](df, period=20)
core_inds["SMA50"] = INDICATOR_REGISTRY["SMA"]["func"](df, period=50)
core_inds["SMA200"] = INDICATOR_REGISTRY["SMA"]["func"](df, period=200) if len(df) >= 200 else core_inds["SMA50"]
core_inds["RSI"] = INDICATOR_REGISTRY["RSI"]["func"](df, period=14)
core_inds["MACD"] = INDICATOR_REGISTRY["MACD"]["func"](df, fast=12, slow=26, signal=9)
core_inds["ADX"] = INDICATOR_REGISTRY["ADX"]["func"](df, period=14)
core_inds["Bollinger_Bands"] = INDICATOR_REGISTRY["Bollinger_Bands"]["func"](df, period=20, std_dev=2.0)
core_inds["RVOL"] = INDICATOR_REGISTRY["RVOL"]["func"](df, period=20)

regime_data = calculate_technical_score(df, core_inds)

# 2. COMPACT TECHNICAL REGIME BAR
st.markdown(
    f"<div class='glass-card' style='padding:16px 24px; margin-bottom:20px; border-left: 4px solid #38BDF8;'>"
    f"<div style='font-size:0.8rem; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>TECHNICAL REGIME SUMMARY</div>"
    f"<div style='display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; text-align:center; font-size:0.85rem;'>"
    f"<div><div style='color:#94A3B8;'>TREND</div><div><span class='badge {'badge-emerald' if regime_data['breakdown']['Trend']['score']>=15 else 'badge-rose'}'>{'BULLISH' if regime_data['breakdown']['Trend']['score']>=15 else 'BEARISH'}</span></div></div>"
    f"<div><div style='color:#94A3B8;'>MOMENTUM</div><div><span class='badge {'badge-emerald' if regime_data['breakdown']['Momentum']['score']>=13 else 'badge-rose'}'>{'POSITIVE' if regime_data['breakdown']['Momentum']['score']>=13 else 'NEGATIVE'}</span></div></div>"
    f"<div><div style='color:#94A3B8;'>STRENGTH</div><div><span class='badge {'badge-cyan' if regime_data['breakdown']['Trend Strength']['score']>=10 else 'badge-amber'}'>{'STRONG' if regime_data['breakdown']['Trend Strength']['score']>=10 else 'WEAK'}</span></div></div>"
    f"<div><div style='color:#94A3B8;'>VOLATILITY</div><div><span class='badge {'badge-cyan' if regime_data['breakdown']['Volatility'].get('state')=='NORMAL' else 'badge-amber'}'>{regime_data['breakdown']['Volatility'].get('state', 'NORMAL')}</span></div></div>"
    f"<div><div style='color:#94A3B8;'>VOLUME</div><div><span class='badge {'badge-emerald' if regime_data['breakdown']['Volume']['score']>=8 else 'badge-amber'}'>{'CONFIRMED' if regime_data['breakdown']['Volume']['score']>=8 else 'LOW VOL'}</span></div></div>"
    f"<div><div style='color:#94A3B8;'>REGIME / BIAS</div><div><span class='badge {'badge-emerald' if 'Bull' in regime_data['regime'] else 'badge-rose' if 'Bear' in regime_data['regime'] else 'badge-cyan'}'>{regime_data['regime']}</span></div></div>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)

# 3. TERMINAL CONTROLS & PRESET TOOLBAR
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1.2, 1, 1])

with ctrl1:
    terminal_mode = st.radio("Terminal Mode", ["Standard", "Advanced"], index=0, horizontal=True)

with ctrl2:
    preset_sel = st.selectbox(
        "Analysis Preset",
        ["Standard", "Swing Trading", "Trend Following", "Breakout", "Mean Reversion", "Momentum", "Volatility", "Volume Analysis", "Custom"],
        index=0
    )

with ctrl3:
    show_events = st.checkbox("Show Event Markers", value=True)

with ctrl4:
    show_tech_levels = st.checkbox("Show Support & Resistance", value=True)

# Preset Indicator Mappings
PRESET_MAPPINGS = {
    "Standard": {"overlays": ["SMA", "Bollinger_Bands"], "panels": ["RSI", "MACD"]},
    "Swing Trading": {"overlays": ["SMA", "Bollinger_Bands"], "panels": ["RSI", "MACD", "ADX", "ATR"]},
    "Trend Following": {"overlays": ["SMA", "EMA", "SuperTrend"], "panels": ["MACD", "ADX"]},
    "Breakout": {"overlays": ["Donchian_Channels", "SuperTrend"], "panels": ["ADX", "RVOL", "ATR"]},
    "Mean Reversion": {"overlays": ["Bollinger_Bands"], "panels": ["RSI", "Stochastic", "ATR"]},
    "Momentum": {"overlays": ["EMA"], "panels": ["RSI", "MACD", "ROC", "Stochastic"]},
    "Volatility": {"overlays": ["Bollinger_Bands", "Keltner_Channels"], "panels": ["ATR", "RVOL"]},
    "Volume Analysis": {"overlays": ["VWAP"], "panels": ["OBV", "CMF", "RVOL", "MFI"]},
    "Custom": {"overlays": ["SMA"], "panels": ["RSI"]}
}

default_overlays = PRESET_MAPPINGS[preset_sel]["overlays"]
default_panels = PRESET_MAPPINGS[preset_sel]["panels"]

# Indicator categorization lists
ALL_OVERLAYS = ["SMA", "EMA", "WMA", "HMA", "VWMA", "Bollinger_Bands", "Keltner_Channels", "Donchian_Channels", "VWAP", "SuperTrend", "Ichimoku", "Parabolic_SAR"]
ALL_PANELS = ["RSI", "MACD", "ROC", "Stochastic", "Williams_R", "ATR", "ADX", "Aroon", "Vortex", "MFI", "CMF", "OBV", "RVOL", "PVT", "Volume_Osc", "Elder_Force", "Ease_of_Movement", "CMO", "TRIX"]

if terminal_mode == "Standard":
    AVAILABLE_OVERLAYS = ["SMA", "EMA", "Bollinger_Bands", "VWAP", "SuperTrend"]
    AVAILABLE_PANELS = ["RSI", "MACD", "ADX", "ATR", "Stochastic", "RVOL"]
else:
    AVAILABLE_OVERLAYS = ALL_OVERLAYS
    AVAILABLE_PANELS = ALL_PANELS

# 4. CATEGORIZED INDICATOR SELECTOR
st.markdown("### 🛠️ Terminal Indicator Composition")
exp1, exp2 = st.columns(2)

with exp1:
    selected_overlays = st.multiselect(
        "Price Axis Overlays (Plotted directly on Candlestick Chart)",
        options=AVAILABLE_OVERLAYS,
        default=[o for o in default_overlays if o in AVAILABLE_OVERLAYS]
    )

with exp2:
    selected_panels = st.multiselect(
        "Analytical Subpanels (Separate Oscillator & Volatility Panels)",
        options=AVAILABLE_PANELS,
        default=[p for p in default_panels if p in AVAILABLE_PANELS]
    )

# Compute Overlay & Panel Series
calculated_overlays = {}
calculated_panels = {}

for key in selected_overlays:
    info_reg = INDICATOR_REGISTRY[key]
    params = {p: s["default"] for p, s in info_reg["parameters"].items()}
    calculated_overlays[info_reg["name"]] = info_reg["func"](df, **params)

for key in selected_panels:
    info_reg = INDICATOR_REGISTRY[key]
    params = {p: s["default"] for p, s in info_reg["parameters"].items()}
    calculated_panels[info_reg["name"]] = info_reg["func"](df, **params)

# 5. TECHNICAL SUPPORT & RESISTANCE LEVEL CALCULATIONS
sr_levels = None
if show_tech_levels:
    recent_highs = df["High"].tail(50).max()
    recent_lows = df["Low"].tail(50).min()
    pivot = (recent_highs + recent_lows + curr_price) / 3
    r1 = (2 * pivot) - recent_lows
    s1 = (2 * pivot) - recent_highs
    r2 = pivot + (recent_highs - recent_lows)
    s2 = pivot - (recent_highs - recent_lows)
    
    sr_levels = {
        "Resistance 2": r2,
        "Resistance 1": r1,
        "Support 1": s1,
        "Support 2": s2
    }

# 6. EVENT MARKERS EXTRACTION
event_markers = []
if show_events:
    # Golden / Death Cross
    if len(df) >= 50:
        sma20 = core_inds["SMA20"]
        sma50 = core_inds["SMA50"]
        c_up = check_crossover_above(sma20, sma50)
        c_dn = check_crossover_below(sma20, sma50)
        
        for i in range(len(df)-20, len(df)):
            if c_up.iloc[i]:
                event_markers.append({
                    "date": df.index[i], "price": df["Low"].iloc[i] * 0.98,
                    "text": "Golden Cross ▲", "symbol": "triangle-up", "color": "#00E676", "position": "bottom center"
                })
            elif c_dn.iloc[i]:
                event_markers.append({
                    "date": df.index[i], "price": df["High"].iloc[i] * 1.02,
                    "text": "Death Cross ▼", "symbol": "triangle-down", "color": "#FF5252", "position": "top center"
                })
                
    # MACD Crosses
    macd_df = core_inds["MACD"]
    mc_up = check_crossover_above(macd_df["MACD"], macd_df["Signal"])
    mc_dn = check_crossover_below(macd_df["MACD"], macd_df["Signal"])
    for i in range(len(df)-10, len(df)):
        if mc_up.iloc[i]:
            event_markers.append({
                "date": df.index[i], "price": df["Low"].iloc[i] * 0.99,
                "text": "MACD Bull ▲", "symbol": "circle", "color": "#38BDF8", "position": "bottom center"
            })
        elif mc_dn.iloc[i]:
            event_markers.append({
                "date": df.index[i], "price": df["High"].iloc[i] * 1.01,
                "text": "MACD Bear ▼", "symbol": "circle", "color": "#FFA726", "position": "top center"
            })

st.markdown("---")

# 7. MAIN PRICE & OVERLAY CHART
st.subheader("📈 Price & Volume Interactive Analysis")
st.markdown("<div style='font-size:0.85rem; color:#94A3B8; font-weight:600; margin-bottom:4px;'>Chart Horizon Picker</div>", unsafe_allow_html=True)

fig_price = create_price_chart(
    df,
    ticker=f"{company} ({ticker})",
    overlays=calculated_overlays,
    show_events=show_events,
    event_markers=event_markers,
    support_resistance=sr_levels
)
st.plotly_chart(fig_price, use_container_width=True)

# 8. VOLUME PANEL
fig_vol = create_volume_chart(df)
st.plotly_chart(fig_vol, use_container_width=True)

# 9. OSCILLATOR & ANALYTICAL SUBPANELS
if calculated_panels:
    fig_osc = create_oscillator_charts(df, calculated_panels)
    if fig_osc:
        st.plotly_chart(fig_osc, use_container_width=True)

st.markdown("---")

# 10. CURRENT TECHNICAL STATE TABLE & TECHNICAL LEVELS
col_tbl, col_confluence = st.columns([1.5, 1])

with col_tbl:
    st.subheader("📋 Current Technical State")
    all_computed = {**calculated_overlays, **calculated_panels, **core_inds}
    sig_table = generate_signals_for_dataframe(df, all_computed)
    if not sig_table.empty:
        st.dataframe(sig_table[["Indicator", "Value", "State", "Event", "Direction"]], use_container_width=True, hide_index=True)
    else:
        st.info("No active indicator state computed.")

with col_confluence:
    st.subheader("🎯 Technical Confluence & Levels")
    
    if sr_levels:
        st.markdown("**Key Pivot Levels**")
        st.markdown(
            f"- **Resistance 2**: `{curr_symbol}{sr_levels['Resistance 2']:,.2f}`\n"
            f"- **Resistance 1**: `{curr_symbol}{sr_levels['Resistance 1']:,.2f}`\n"
            f"- **Current Close**: `{curr_symbol}{curr_price:,.2f}`\n"
            f"- **Support 1**: `{curr_symbol}{sr_levels['Support 1']:,.2f}`\n"
            f"- **Support 2**: `{curr_symbol}{sr_levels['Support 2']:,.2f}`\n"
        )
        
    st.markdown("**Transparent Confluence Summary**")
    bull_count = len(sig_table[sig_table["Direction"] == "Bullish"]) if not sig_table.empty else 0
    bear_count = len(sig_table[sig_table["Direction"] == "Bearish"]) if not sig_table.empty else 0
    neut_count = len(sig_table[sig_table["Direction"] == "Neutral"]) if not sig_table.empty else 0
    
    st.markdown(
        f"- 🟢 **Bullish Signals**: `{bull_count}`\n"
        f"- 🔴 **Bearish Signals**: `{bear_count}`\n"
        f"- 🟡 **Neutral Signals**: `{neut_count}`\n"
    )

st.markdown("---")

# 11. STRATEGY EXPLAINER GUIDE
render_strategy_explainer(preset_sel)

# 12. DATA QUALITY & METADATA FOOTER
st.markdown(
    f"<div class='glass-card' style='padding:12px 20px; font-size:0.8rem; color:#94A3B8; display:flex; justify-content:space-between; margin-top:20px;'>"
    f"<div><b>Data Source</b>: Yahoo Finance &nbsp;|&nbsp; <b>Interval</b>: {interval} &nbsp;|&nbsp; <b>Period</b>: {period}</div>"
    f"<div><b>Bars Loaded</b>: {len(df):,} &nbsp;|&nbsp; <b>Last Bar Date</b>: {df.index[-1].strftime('%Y-%m-%d')}</div>"
    f"</div>",
    unsafe_allow_html=True
)
