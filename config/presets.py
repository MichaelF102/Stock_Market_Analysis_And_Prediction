"""
Trading Strategy Presets and Strategy Explainer Guide for Technical Terminal.
"""

import streamlit as st

STRATEGY_DETAILS = {
    "Standard": {
        "title": "Standard Technical Baseline",
        "description": "A balanced multi-indicator setup combining trend (SMA) with volatility bands (Bollinger Bands) and momentum oscillators (RSI, MACD).",
        "indicators": "SMA(20), SMA(50), Bollinger Bands, RSI(14), MACD(12,26,9)",
        "entry_rules": "Look for price trading above 50-day SMA with RSI recovering from oversold (30-40) and bullish MACD histogram expansion.",
        "exit_rules": "Exit on upper Bollinger Band test or bearish MACD divergence.",
        "risk_management": "Set stop-loss below the 20-day SMA or lower Bollinger Band."
    },
    "Swing Trading": {
        "title": "Multi-Day Swing Trading Setup",
        "description": "Captures multi-day momentum swings across support/resistance levels with trend strength confirmation.",
        "indicators": "SMA(20,50), Bollinger Bands, RSI(14), MACD, ADX(14), ATR(14)",
        "entry_rules": "Enter when ADX > 25 (strong trend) and RSI pulls back to 45-50 support during an established uptrend.",
        "exit_rules": "Take profit near major pivot resistance or when RSI crosses above 70.",
        "risk_management": "Use a 2.0x ATR trailing stop-loss from entry price."
    },
    "Trend Following": {
        "title": "Trend Following & Momentum Capture",
        "description": "Aligns trades strictly with higher-timeframe market trends using moving average ribbons and SuperTrend.",
        "indicators": "SMA(20,50,200), EMA(20), SuperTrend(10,3), MACD, ADX(14)",
        "entry_rules": "Go long when price is above SMA(200), SuperTrend is green/bullish, and MACD line is above zero.",
        "exit_rules": "Exit when price closes below the SuperTrend baseline or SMA(50).",
        "risk_management": "Trail stop along the SuperTrend stop line."
    },
    "Breakout": {
        "title": "Volatility & Channel Breakout Strategy",
        "description": "Detects high-momentum breakouts from consolidation zones using Donchian channels and volume expansion.",
        "indicators": "Donchian Channels(20), SuperTrend, ADX(14), RVOL(20), ATR(14)",
        "entry_rules": "Enter when price breaks 20-day Donchian High with Relative Volume (RVOL) > 1.5x and rising ADX.",
        "exit_rules": "Exit when price touches the Donchian Middle Band or after 5-10 bars.",
        "risk_management": "Stop-loss placed at the breakout candle low or middle Donchian band."
    },
    "Mean Reversion": {
        "title": "Mean Reversion & Extreme Exhaustion",
        "description": "Exploits temporary price overextensions back toward equilibrium moving averages.",
        "indicators": "Bollinger Bands(20,2), RSI(14), Stochastic(14,3,3), ATR(14)",
        "entry_rules": "Buy when price pierces the lower Bollinger Band with RSI < 30 and Stochastic %K crossing above %D from oversold (<20).",
        "exit_rules": "Exit when price reverts to the 20-period SMA midline or upper band.",
        "risk_management": "Hard stop-loss 1.5x ATR below the swing low."
    },
    "Momentum": {
        "title": "High-Velocity Momentum Strategy",
        "description": "Targets accelerated price moves with multiple momentum oscillator alignments.",
        "indicators": "EMA(20,50), RSI(14), MACD, ROC(10), Stochastic(14,3,3)",
        "entry_rules": "Long when 10-day Rate of Change (ROC) > 0, MACD histogram is expanding, and RSI is between 55 and 70.",
        "exit_rules": "Exit on momentum exhaustion (RSI divergence or ROC rolling over).",
        "risk_management": "Place stop-loss below the 20-day EMA."
    },
    "Volatility": {
        "title": "Volatility Squeeze & Expansion Setup",
        "description": "Identifies low-volatility squeezes (Bollinger Bands inside Keltner Channels) preceding explosive breakouts.",
        "indicators": "Bollinger Bands(20,2), Keltner Channels(20,1.5,10), ATR(14), RVOL(20)",
        "entry_rules": "Wait for Bollinger Bands to compress inside Keltner Channels, then enter in the direction of the subsequent channel expansion.",
        "exit_rules": "Exit when ATR begins contracting or price crosses opposite Keltner channel.",
        "risk_management": "Stop-loss at the midpoint of the squeeze channel."
    },
    "Volume Analysis": {
        "title": "Institutional Volume & Money Flow Analysis",
        "description": "Tracks smart-money institutional accumulation and distribution via volume-weighted indicators.",
        "indicators": "VWAP, OBV, CMF(20), RVOL(20), MFI(14)",
        "entry_rules": "Buy when price holds above VWAP with Chaikin Money Flow (CMF) > +0.10 and On-Balance Volume (OBV) making new highs.",
        "exit_rules": "Exit when price falls below VWAP or CMF turns negative (< -0.05).",
        "risk_management": "Initial stop-loss set below the VWAP support band."
    },
    "Custom": {
        "title": "Custom Strategy Configuration",
        "description": "User-defined technical indicator configuration for tailored technical analysis.",
        "indicators": "Configured by user via the indicator selection panels.",
        "entry_rules": "Follow the custom rule set aligned with your trading plan.",
        "exit_rules": "Adhere to predetermined target levels and pivot resistances.",
        "risk_management": "Maintain strict position sizing and predefined risk-reward ratio (> 1:2)."
    }
}


def render_strategy_explainer(preset_name: str = "Standard"):
    """Renders an institutional strategy explanation card in Streamlit."""
    strat = STRATEGY_DETAILS.get(preset_name, STRATEGY_DETAILS["Standard"])
    
    st.markdown(
        f"<div class='glass-card' style='padding:20px; border-left: 4px solid #10B981; margin-top:20px;'>"
        f"<h4 style='color:#10B981; margin:0 0 8px 0;'>📘 Strategy Guide: {strat['title']}</h4>"
        f"<p style='color:#CBD5E1; font-size:0.9rem; margin-bottom:12px;'>{strat['description']}</p>"
        f"<div style='display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; font-size:0.85rem;'>"
        f"<div><b style='color:#38BDF8;'>Key Indicators:</b><br><span style='color:#94A3B8;'>{strat['indicators']}</span></div>"
        f"<div><b style='color:#38BDF8;'>Entry Trigger:</b><br><span style='color:#94A3B8;'>{strat['entry_rules']}</span></div>"
        f"<div><b style='color:#38BDF8;'>Exit Rules:</b><br><span style='color:#94A3B8;'>{strat['exit_rules']}</span></div>"
        f"<div><b style='color:#38BDF8;'>Risk Management:</b><br><span style='color:#94A3B8;'>{strat['risk_management']}</span></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )
