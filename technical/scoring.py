import pandas as pd
import numpy as np

def calculate_technical_score(df: pd.DataFrame, indicator_results: dict) -> dict:
    """
    Computes transparent, robust Technical Score (0-100) broken down into 5 components:
    - Trend (30%): Evaluates multi-timeframe moving averages (SMA20, SMA50, SMA200), SuperTrend, and golden/death cross alignment.
    - Momentum (25%): RSI momentum regimes and MACD line/signal crossover structure.
    - Trend Strength (20%): ADX directional index and +DI / -DI dominance.
    - Volume (15%): Relative Volume (RVOL) participation and Chaikin Money Flow (CMF).
    - Volatility (10%): Bollinger Bands %B positioning and volatility expansion/squeeze.
    
    Returns dictionary with overall score, category breakdown scores, and regime classification.
    """
    if df is None or df.empty or len(df) < 5:
        return {
            "overall": 50.0,
            "breakdown": {
                "Trend": {"score": 15.0, "max": 30},
                "Momentum": {"score": 12.5, "max": 25},
                "Trend Strength": {"score": 10.0, "max": 20},
                "Volume": {"score": 7.5, "max": 15},
                "Volatility": {"score": 5.0, "max": 10, "state": "NORMAL"}
            },
            "regime": "Neutral / Consolidation"
        }
        
    close = float(df["Close"].iloc[-1])
    
    # --------------------------------------------------
    # 1. Trend Score (Max 30)
    # --------------------------------------------------
    # Extract Moving Averages from indicator_results or directly calculate
    def _get_ind_val(key_patterns):
        for k, v in indicator_results.items():
            for pat in key_patterns:
                if pat.lower() in k.lower():
                    if isinstance(v, pd.Series) and not v.dropna().empty:
                        return float(v.dropna().iloc[-1])
                    elif isinstance(v, pd.DataFrame) and not v.empty:
                        return float(v.iloc[:, 0].dropna().iloc[-1])
        return None

    sma20_val = _get_ind_val(["SMA20", "SMA_20", "EMA20"])
    if sma20_val is None and len(df) >= 20:
        sma20_val = float(df["Close"].rolling(20).mean().iloc[-1])
        
    sma50_val = _get_ind_val(["SMA50", "SMA_50", "EMA50"])
    if sma50_val is None and len(df) >= 50:
        sma50_val = float(df["Close"].rolling(50).mean().iloc[-1])
        
    sma200_val = _get_ind_val(["SMA200", "SMA_200", "EMA200"])
    if sma200_val is None and len(df) >= 200:
        sma200_val = float(df["Close"].rolling(200).mean().iloc[-1])

    trend_score = 0.0
    
    # Short-term trend: Price > 20 SMA (8 pts)
    if sma20_val is not None:
        if close > sma20_val:
            trend_score += 8.0
    else:
        trend_score += 4.0

    # Medium-term trend: Price > 50 SMA (8 pts)
    if sma50_val is not None:
        if close > sma50_val:
            trend_score += 8.0
    else:
        trend_score += 4.0

    # Long-term trend: Price > 200 SMA (8 pts)
    if sma200_val is not None:
        if close > sma200_val:
            trend_score += 8.0
    elif sma20_val is not None and sma50_val is not None:
        if close > sma50_val:
            trend_score += 8.0
    else:
        trend_score += 4.0

    # Moving Average Alignment: 20 SMA > 50 SMA (6 pts)
    if sma20_val is not None and sma50_val is not None:
        if sma20_val >= sma50_val:
            trend_score += 6.0
    else:
        trend_score += 3.0

    # SuperTrend Check (Bonus / Confirmation)
    if "SuperTrend" in indicator_results:
        st_df = indicator_results["SuperTrend"]
        if isinstance(st_df, pd.DataFrame) and "SuperTrend_Direction" in st_df.columns:
            st_dir = st_df["SuperTrend_Direction"].iloc[-1]
            if st_dir == 1:
                trend_score = min(30.0, trend_score + 3.0)
            elif st_dir == -1:
                trend_score = max(0.0, trend_score - 3.0)

    trend_score_norm = round(min(30.0, max(0.0, trend_score)), 1)

    # --------------------------------------------------
    # 2. Momentum Score (Max 25)
    # --------------------------------------------------
    mom_score = 0.0
    
    # RSI (14)
    rsi_val = None
    if "RSI" in indicator_results:
        rsi_s = indicator_results["RSI"]
        if isinstance(rsi_s, (pd.Series, pd.DataFrame)) and not rsi_s.empty:
            rsi_val = float(rsi_s.iloc[-1] if isinstance(rsi_s, pd.Series) else rsi_s.iloc[:, 0].iloc[-1])
            
    if rsi_val is None and len(df) >= 14:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])

    if rsi_val is not None and not np.isnan(rsi_val):
        if 50.0 <= rsi_val <= 70.0:
            mom_score += 10.0  # Optimal Bullish Momentum
        elif rsi_val > 70.0:
            mom_score += 8.0   # Strong Momentum (Overbought)
        elif 40.0 <= rsi_val < 50.0:
            mom_score += 5.0   # Mild Consolidation
        elif rsi_val < 30.0:
            mom_score += 4.0   # Oversold
        else:
            mom_score += 2.0   # Bearish Momentum
    else:
        mom_score += 6.0

    # MACD Line & Signal
    m_val, s_val = None, None
    if "MACD" in indicator_results:
        macd_res = indicator_results["MACD"]
        if isinstance(macd_res, pd.DataFrame):
            if "MACD" in macd_res and "Signal" in macd_res:
                m_val = float(macd_res["MACD"].iloc[-1])
                s_val = float(macd_res["Signal"].iloc[-1])
            elif "MACD" in macd_res:
                m_val = float(macd_res["MACD"].iloc[-1])
                
    if m_val is None and len(df) >= 26:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        m_val = float(macd_line.iloc[-1])
        s_val = float(signal_line.iloc[-1])

    if m_val is not None and s_val is not None and not (np.isnan(m_val) or np.isnan(s_val)):
        if m_val > s_val:
            mom_score += 10.0  # Bullish Crossover
        if m_val > 0:
            mom_score += 5.0   # Above Zero Line Expansion
    else:
        mom_score += 6.5

    mom_score_norm = round(min(25.0, max(0.0, mom_score)), 1)

    # --------------------------------------------------
    # 3. Trend Strength Score (Max 20)
    # --------------------------------------------------
    strength_score = 0.0
    adx_val, p_di, m_di = None, None, None
    
    if "ADX" in indicator_results:
        adx_df = indicator_results["ADX"]
        if isinstance(adx_df, pd.DataFrame):
            if "ADX" in adx_df.columns:
                adx_val = float(adx_df["ADX"].iloc[-1])
            if "Plus_DI" in adx_df.columns:
                p_di = float(adx_df["Plus_DI"].iloc[-1])
            if "Minus_DI" in adx_df.columns:
                m_di = float(adx_df["Minus_DI"].iloc[-1])

    if adx_val is not None and not np.isnan(adx_val):
        if adx_val >= 25.0:
            strength_score += 10.0  # Trending Regime
        elif adx_val >= 20.0:
            strength_score += 6.0   # Developing Trend
        else:
            strength_score += 3.0   # Choppy / Ranging
    else:
        strength_score += 5.0

    if p_di is not None and m_di is not None and not (np.isnan(p_di) or np.isnan(m_di)):
        if p_di > m_di:
            strength_score += 10.0  # Bullish Dominance
        else:
            strength_score += 2.0   # Bearish Dominance
    else:
        strength_score += 5.0

    strength_score_norm = round(min(20.0, max(0.0, strength_score)), 1)

    # --------------------------------------------------
    # 4. Volume Score (Max 15)
    # --------------------------------------------------
    vol_score = 0.0
    rvol_val = None
    if "RVOL" in indicator_results:
        rvol_s = indicator_results["RVOL"]
        if isinstance(rvol_s, (pd.Series, pd.DataFrame)) and not rvol_s.empty:
            rvol_val = float(rvol_s.iloc[-1] if isinstance(rvol_s, pd.Series) else rvol_s.iloc[:, 0].iloc[-1])

    if rvol_val is None and "Volume" in df.columns and len(df) >= 20:
        vol_sma = df["Volume"].rolling(20).mean().iloc[-1]
        if vol_sma > 0:
            rvol_val = float(df["Volume"].iloc[-1] / vol_sma)

    if rvol_val is not None and not np.isnan(rvol_val):
        if rvol_val >= 1.5:
            vol_score += 10.0  # Strong Volume Expansion
        elif rvol_val >= 1.0:
            vol_score += 7.0   # Average Volume
        else:
            vol_score += 4.0   # Low Volume
    else:
        vol_score += 5.0

    # CMF or Volume Trend
    cmf_val = _get_ind_val(["CMF", "Chaikin"])
    if cmf_val is not None and not np.isnan(cmf_val):
        if cmf_val > 0:
            vol_score += 5.0
        else:
            vol_score += 1.0
    else:
        vol_score += 3.0

    vol_score_norm = round(min(15.0, max(0.0, vol_score)), 1)

    # --------------------------------------------------
    # 5. Volatility Score (Max 10)
    # --------------------------------------------------
    volatility_score = 5.0
    vol_state = "NORMAL"
    
    if "Bollinger_Bands" in indicator_results:
        bb_df = indicator_results["Bollinger_Bands"]
        if isinstance(bb_df, pd.DataFrame) and "BB_PctB" in bb_df.columns:
            pct_b = float(bb_df["BB_PctB"].iloc[-1])
            if not np.isnan(pct_b):
                if 0.2 <= pct_b <= 0.8:
                    volatility_score = 8.0
                    vol_state = "NORMAL"
                elif pct_b > 0.8:
                    volatility_score = 6.0
                    vol_state = "EXPANDING"
                else:
                    volatility_score = 4.0
                    vol_state = "COMPRESSED"
                    
    volatility_score_norm = round(min(10.0, max(0.0, volatility_score)), 1)

    # Aggregate Overall Technical Score
    overall_score = round(
        trend_score_norm + mom_score_norm + strength_score_norm + vol_score_norm + volatility_score_norm, 1
    )

    # Regime Classification
    regime = classify_market_regime(trend_score_norm, mom_score_norm, strength_score_norm, overall_score)

    return {
        "overall": overall_score,
        "breakdown": {
            "Trend": {"score": trend_score_norm, "max": 30},
            "Momentum": {"score": mom_score_norm, "max": 25},
            "Trend Strength": {"score": strength_score_norm, "max": 20},
            "Volume": {"score": vol_score_norm, "max": 15},
            "Volatility": {"score": volatility_score_norm, "max": 10, "state": vol_state}
        },
        "regime": regime
    }

def classify_market_regime(trend_score: float, mom_score: float, strength_score: float, overall: float) -> str:
    if overall >= 75 and trend_score >= 20:
        return "Strong Bull Trend"
    elif overall >= 55 and trend_score >= 15:
        return "Bullish Trend"
    elif overall <= 30 and trend_score <= 10:
        return "Strong Bear Trend"
    elif overall <= 45 and trend_score < 15:
        return "Bearish Trend"
    elif strength_score < 8:
        return "Low Volatility / Consolidation"
    else:
        return "Neutral / Range"
