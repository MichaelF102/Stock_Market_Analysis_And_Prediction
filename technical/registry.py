"""
Centralized Indicator Registry.
Defines metadata, parameter schemas, output series, and chart types for all indicators.
"""

from technical.trend import calculate_sma, calculate_ema, calculate_wma, calculate_hma, calculate_vwma
from technical.momentum import calculate_rsi, calculate_macd, calculate_roc, calculate_stochastic, calculate_williams_r
from technical.volatility import calculate_atr, calculate_bollinger_bands, calculate_keltner_channels, calculate_donchian_channels
from technical.volume import calculate_obv, calculate_cmf, calculate_adl
from technical.trend_strength import calculate_adx, calculate_aroon, calculate_vortex
from technical.money_flow import (
    calculate_mfi, calculate_vwap, calculate_chaikin_ad_oscillator, calculate_rvol,
    calculate_pvt, calculate_volume_oscillator, calculate_volume_roc,
    calculate_elder_force_index, calculate_ease_of_movement, calculate_nvi, calculate_pvi
)
from technical.trend_path import (
    calculate_supertrend, calculate_ichimoku, calculate_parabolic_sar,
    calculate_zigzag, calculate_fractals, calculate_cmo, calculate_trix
)

INDICATOR_REGISTRY = {
    # --- TREND ---
    "SMA": {
        "name": "Simple Moving Average",
        "category": "Trend",
        "description": "Calculates arithmetic mean of close prices over specified period.",
        "chart_type": "overlay",
        "func": calculate_sma,
        "parameters": {"period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1, "presets": [5, 10, 20, 50, 100, 200]}},
        "outputs": ["SMA"],
        "signal_type": "crossover"
    },
    "EMA": {
        "name": "Exponential Moving Average",
        "category": "Trend",
        "description": "Gives greater weight to recent price data.",
        "chart_type": "overlay",
        "func": calculate_ema,
        "parameters": {"period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1, "presets": [9, 21, 50, 100, 200]}},
        "outputs": ["EMA"],
        "signal_type": "crossover"
    },
    "WMA": {
        "name": "Weighted Moving Average",
        "category": "Trend",
        "description": "Applies linearly decaying weights to prices over period.",
        "chart_type": "overlay",
        "func": calculate_wma,
        "parameters": {"period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1, "presets": [10, 20, 50]}},
        "outputs": ["WMA"],
        "signal_type": "crossover"
    },
    "HMA": {
        "name": "Hull Moving Average",
        "category": "Trend",
        "description": "Reduces lag while increasing responsiveness and smoothness.",
        "chart_type": "overlay",
        "func": calculate_hma,
        "parameters": {"period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1, "presets": [9, 14, 20]}},
        "outputs": ["HMA"],
        "signal_type": "crossover"
    },
    "VWMA": {
        "name": "Volume Weighted Moving Average",
        "category": "Trend",
        "description": "Weights prices by volume traded in each period.",
        "chart_type": "overlay",
        "func": calculate_vwma,
        "parameters": {"period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1, "presets": [10, 20, 50]}},
        "outputs": ["VWMA"],
        "signal_type": "crossover"
    },

    # --- MOMENTUM ---
    "RSI": {
        "name": "Relative Strength Index",
        "category": "Momentum",
        "description": "Measures speed and change of price movements between 0 and 100.",
        "chart_type": "oscillator",
        "func": calculate_rsi,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [7, 14, 21]}},
        "outputs": ["RSI"],
        "signal_type": "threshold",
        "thresholds": {"oversold": 30, "overbought": 70}
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "category": "Momentum",
        "description": "Trend-following momentum indicator showing relationship between two EMAs.",
        "chart_type": "oscillator",
        "func": calculate_macd,
        "parameters": {
            "fast": {"type": "int", "default": 12, "min": 2, "max": 50, "step": 1, "presets": [12]},
            "slow": {"type": "int", "default": 26, "min": 5, "max": 100, "step": 1, "presets": [26]},
            "signal": {"type": "int", "default": 9, "min": 2, "max": 50, "step": 1, "presets": [9]}
        },
        "outputs": ["MACD", "Signal", "Hist"],
        "signal_type": "crossover"
    },
    "ROC": {
        "name": "Rate of Change",
        "category": "Momentum",
        "description": "Pure momentum oscillator measuring percentage price change.",
        "chart_type": "oscillator",
        "func": calculate_roc,
        "parameters": {"period": {"type": "int", "default": 12, "min": 1, "max": 100, "step": 1, "presets": [9, 12, 21]}},
        "outputs": ["ROC"],
        "signal_type": "threshold",
        "thresholds": {"zero": 0}
    },
    "Stochastic": {
        "name": "Stochastic Oscillator",
        "category": "Momentum",
        "description": "Compares close price to high-low range over period (%K, %D).",
        "chart_type": "oscillator",
        "func": calculate_stochastic,
        "parameters": {
            "k_period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]},
            "d_period": {"type": "int", "default": 3, "min": 1, "max": 20, "step": 1, "presets": [3]},
            "slowing": {"type": "int", "default": 3, "min": 1, "max": 20, "step": 1, "presets": [3]}
        },
        "outputs": ["Stoch_K", "Stoch_D"],
        "signal_type": "threshold",
        "thresholds": {"oversold": 20, "overbought": 80}
    },
    "Williams_R": {
        "name": "Williams %R",
        "category": "Momentum",
        "description": "Momentum indicator reflecting level of close relative to high-low range.",
        "chart_type": "oscillator",
        "func": calculate_williams_r,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["Williams_R"],
        "signal_type": "threshold",
        "thresholds": {"oversold": -80, "overbought": -20}
    },

    # --- VOLATILITY ---
    "ATR": {
        "name": "Average True Range",
        "category": "Volatility",
        "description": "Market volatility indicator measuring true high-low range.",
        "chart_type": "oscillator",
        "func": calculate_atr,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["ATR"],
        "signal_type": "state"
    },
    "Bollinger_Bands": {
        "name": "Bollinger Bands",
        "category": "Volatility",
        "description": "Volatility envelopes around SMA based on standard deviation.",
        "chart_type": "overlay",
        "func": calculate_bollinger_bands,
        "parameters": {
            "period": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 1, "presets": [20]},
            "std_dev": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1, "presets": [2.0]}
        },
        "outputs": ["BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_PctB"],
        "signal_type": "state"
    },
    "Keltner_Channels": {
        "name": "Keltner Channels",
        "category": "Volatility",
        "description": "Volatility channels based on EMA and Average True Range.",
        "chart_type": "overlay",
        "func": calculate_keltner_channels,
        "parameters": {
            "period": {"type": "int", "default": 20, "min": 5, "max": 50, "step": 1, "presets": [20]},
            "atr_period": {"type": "int", "default": 10, "min": 2, "max": 50, "step": 1, "presets": [10]},
            "multiplier": {"type": "float", "default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1, "presets": [2.0]}
        },
        "outputs": ["KC_Upper", "KC_Middle", "KC_Lower"],
        "signal_type": "state"
    },
    "Donchian_Channels": {
        "name": "Donchian Channels",
        "category": "Volatility",
        "description": "Channels formed by highest high and lowest low over N periods.",
        "chart_type": "overlay",
        "func": calculate_donchian_channels,
        "parameters": {"period": {"type": "int", "default": 20, "min": 5, "max": 100, "step": 1, "presets": [20]}},
        "outputs": ["DC_Upper", "DC_Middle", "DC_Lower"],
        "signal_type": "state"
    },

    # --- VOLUME ---
    "OBV": {
        "name": "On-Balance Volume",
        "category": "Volume",
        "description": "Cumulative volume indicator correlating volume flow to price change.",
        "chart_type": "volume",
        "func": calculate_obv,
        "parameters": {},
        "outputs": ["OBV"],
        "signal_type": "state"
    },
    "CMF": {
        "name": "Chaikin Money Flow",
        "category": "Volume",
        "description": "Measures Accumulation/Distribution volume over specified window.",
        "chart_type": "volume",
        "func": calculate_cmf,
        "parameters": {"period": {"type": "int", "default": 20, "min": 5, "max": 50, "step": 1, "presets": [20]}},
        "outputs": ["CMF"],
        "signal_type": "threshold",
        "thresholds": {"zero": 0}
    },
    "ADL": {
        "name": "Accumulation/Distribution Line",
        "category": "Volume",
        "description": "Cumulative measure of volume weighted by high-low close position.",
        "chart_type": "volume",
        "func": calculate_adl,
        "parameters": {},
        "outputs": ["ADL"],
        "signal_type": "state"
    },

    # --- TREND STRENGTH ---
    "ADX": {
        "name": "Average Directional Index",
        "category": "Trend Strength",
        "description": "Quantifies trend strength regardless of trend direction.",
        "chart_type": "oscillator",
        "func": calculate_adx,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["ADX", "Plus_DI", "Minus_DI"],
        "signal_type": "threshold",
        "thresholds": {"strong": 25}
    },
    "Aroon": {
        "name": "Aroon Indicator",
        "category": "Trend Strength",
        "description": "Identifies when trends start/end and strength of trend (Up/Down).",
        "chart_type": "oscillator",
        "func": calculate_aroon,
        "parameters": {"period": {"type": "int", "default": 25, "min": 5, "max": 50, "step": 1, "presets": [25]}},
        "outputs": ["Aroon_Up", "Aroon_Down", "Aroon_Osc"],
        "signal_type": "crossover"
    },
    "Vortex": {
        "name": "Vortex Indicator",
        "category": "Trend Strength",
        "description": "Two oscillators identifying start of positive or negative trend (VI+, VI-).",
        "chart_type": "oscillator",
        "func": calculate_vortex,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["Vortex_Plus", "Vortex_Minus"],
        "signal_type": "crossover"
    },

    # --- MONEY FLOW ---
    "MFI": {
        "name": "Money Flow Index",
        "category": "Money Flow",
        "description": "Volume-weighted RSI measuring money inflow and outflow.",
        "chart_type": "oscillator",
        "func": calculate_mfi,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["MFI"],
        "signal_type": "threshold",
        "thresholds": {"oversold": 20, "overbought": 80}
    },
    "VWAP": {
        "name": "Volume Weighted Average Price",
        "category": "Money Flow",
        "description": "Trading benchmark showing ratio of value traded to total volume.",
        "chart_type": "overlay",
        "func": calculate_vwap,
        "parameters": {},
        "outputs": ["VWAP"],
        "signal_type": "crossover"
    },
    "Chaikin_Osc": {
        "name": "Chaikin A/D Oscillator",
        "category": "Money Flow",
        "description": "MACD applied to Accumulation/Distribution Line.",
        "chart_type": "volume",
        "func": calculate_chaikin_ad_oscillator,
        "parameters": {
            "fast": {"type": "int", "default": 3, "min": 2, "max": 20, "step": 1, "presets": [3]},
            "slow": {"type": "int", "default": 10, "min": 5, "max": 50, "step": 1, "presets": [10]}
        },
        "outputs": ["Chaikin_Osc"],
        "signal_type": "threshold"
    },
    "RVOL": {
        "name": "Relative Volume",
        "category": "Money Flow",
        "description": "Ratio of current volume to average volume over N periods.",
        "chart_type": "volume",
        "func": calculate_rvol,
        "parameters": {"period": {"type": "int", "default": 20, "min": 5, "max": 50, "step": 1, "presets": [20]}},
        "outputs": ["RVOL"],
        "signal_type": "threshold",
        "thresholds": {"elevated": 1.5, "spike": 2.5}
    },
    "PVT": {
        "name": "Price Volume Trend",
        "category": "Money Flow",
        "description": "Cumulative volume based on percentage price changes.",
        "chart_type": "volume",
        "func": calculate_pvt,
        "parameters": {},
        "outputs": ["PVT"],
        "signal_type": "state"
    },
    "Volume_Osc": {
        "name": "Volume Oscillator",
        "category": "Money Flow",
        "description": "Difference between two volume moving averages as a percentage.",
        "chart_type": "volume",
        "func": calculate_volume_oscillator,
        "parameters": {
            "fast": {"type": "int", "default": 14, "min": 2, "max": 30, "step": 1, "presets": [14]},
            "slow": {"type": "int", "default": 28, "min": 10, "max": 60, "step": 1, "presets": [28]}
        },
        "outputs": ["Volume_Osc"],
        "signal_type": "threshold"
    },
    "Elder_Force": {
        "name": "Elder Force Index",
        "category": "Money Flow",
        "description": "Combines price movement direction, extent, and volume.",
        "chart_type": "volume",
        "func": calculate_elder_force_index,
        "parameters": {"period": {"type": "int", "default": 13, "min": 2, "max": 50, "step": 1, "presets": [13]}},
        "outputs": ["Elder_Force"],
        "signal_type": "threshold"
    },

    # --- TREND PATH / ADVANCED ---
    "SuperTrend": {
        "name": "SuperTrend",
        "category": "Trend Path",
        "description": "Trend-following indicator using ATR to plot dynamic trailing stops.",
        "chart_type": "overlay",
        "func": calculate_supertrend,
        "parameters": {
            "period": {"type": "int", "default": 10, "min": 2, "max": 50, "step": 1, "presets": [10]},
            "multiplier": {"type": "float", "default": 3.0, "min": 1.0, "max": 5.0, "step": 0.5, "presets": [3.0]}
        },
        "outputs": ["SuperTrend", "SuperTrend_Direction"],
        "signal_type": "state"
    },
    "Ichimoku": {
        "name": "Ichimoku Cloud",
        "category": "Trend Path",
        "description": "Comprehensive system defining support/resistance, trend, and momentum.",
        "chart_type": "overlay",
        "func": calculate_ichimoku,
        "parameters": {
            "tenkan_period": {"type": "int", "default": 9, "min": 5, "max": 30, "step": 1, "presets": [9]},
            "kijun_period": {"type": "int", "default": 26, "min": 10, "max": 60, "step": 1, "presets": [26]},
            "senkou_b_period": {"type": "int", "default": 52, "min": 20, "max": 120, "step": 1, "presets": [52]}
        },
        "outputs": ["Tenkan_Sen", "Kijun_Sen", "Senkou_Span_A", "Senkou_Span_B", "Chikou_Span"],
        "signal_type": "state"
    },
    "Parabolic_SAR": {
        "name": "Parabolic SAR",
        "category": "Trend Path",
        "description": "Plots trailing stop and reversal price points.",
        "chart_type": "overlay",
        "func": calculate_parabolic_sar,
        "parameters": {
            "af_step": {"type": "float", "default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005, "presets": [0.02]},
            "af_max": {"type": "float", "default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05, "presets": [0.2]}
        },
        "outputs": ["Parabolic_SAR"],
        "signal_type": "state"
    },
    "ZigZag": {
        "name": "ZigZag Indicator",
        "category": "Trend Path",
        "description": "Filters out noise to show swing highs and swing lows.",
        "chart_type": "overlay",
        "func": calculate_zigzag,
        "parameters": {"deviation": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5, "presets": [5.0]}},
        "outputs": ["ZigZag"],
        "signal_type": "state"
    },
    "CMO": {
        "name": "Chande Momentum Oscillator",
        "category": "Trend Path",
        "description": "Calculates momentum on both up and down days between -100 and +100.",
        "chart_type": "oscillator",
        "func": calculate_cmo,
        "parameters": {"period": {"type": "int", "default": 14, "min": 2, "max": 50, "step": 1, "presets": [14]}},
        "outputs": ["CMO"],
        "signal_type": "threshold",
        "thresholds": {"oversold": -50, "overbought": 50}
    },
    "TRIX": {
        "name": "TRIX Oscillator",
        "category": "Trend Path",
        "description": "Triple-smoothed exponential moving average oscillator.",
        "chart_type": "oscillator",
        "func": calculate_trix,
        "parameters": {
            "period": {"type": "int", "default": 15, "min": 5, "max": 50, "step": 1, "presets": [15]},
            "signal_period": {"type": "int", "default": 9, "min": 2, "max": 30, "step": 1, "presets": [9]}
        },
        "outputs": ["TRIX", "Signal"],
        "signal_type": "crossover"
    }
}

CATEGORIES = ["Trend", "Momentum", "Volatility", "Volume", "Trend Strength", "Money Flow", "Trend Path"]

def get_indicator(key: str) -> dict:
    return INDICATOR_REGISTRY.get(key, None)

def get_indicators_by_category(category: str) -> dict:
    return {k: v for k, v in INDICATOR_REGISTRY.items() if v["category"] == category}

def get_all_categories() -> list:
    return CATEGORIES
