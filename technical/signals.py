import pandas as pd
import numpy as np

def check_crossover_above(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Returns boolean series where series1 crosses above series2"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def check_crossover_below(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Returns boolean series where series1 crosses below series2"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

def generate_signals_for_dataframe(df: pd.DataFrame, indicator_results: dict) -> pd.DataFrame:
    """
    Generates standardized signals with state, event, direction, strength, and evidence columns.
    """
    if df.empty:
        return pd.DataFrame(columns=["Indicator", "Value", "State", "Event", "Direction", "Strength", "Evidence"])
        
    last_idx = df.index[-1]
    curr_close = float(df["Close"].iloc[-1])
    
    signals = []
    
    # 1. Moving Averages
    for name in ["SMA", "EMA", "WMA", "HMA", "VWMA"]:
        if name in indicator_results:
            ser = indicator_results[name]
            val = float(ser.iloc[-1]) if isinstance(ser, pd.Series) else float(ser.iloc[:, 0].iloc[-1])
            
            if not pd.isna(val):
                state = "Bullish" if curr_close > val else "Bearish"
                direction = state
                strength = "Moderate"
                evidence = f"Close ({curr_close:,.2f}) > {name} ({val:,.2f})" if curr_close > val else f"Close ({curr_close:,.2f}) < {name} ({val:,.2f})"
                
                # Check crossover
                event = "—"
                if len(df) > 1:
                    c_up = check_crossover_above(df["Close"], ser if isinstance(ser, pd.Series) else ser.iloc[:, 0])
                    c_dn = check_crossover_below(df["Close"], ser if isinstance(ser, pd.Series) else ser.iloc[:, 0])
                    if c_up.iloc[-1]:
                        event = "Crossed Above ▲"
                        strength = "Strong"
                    elif c_dn.iloc[-1]:
                        event = "Crossed Below ▼"
                        strength = "Strong"
                        
                signals.append({
                    "Indicator": name,
                    "Value": f"{curr_close:,.2f} vs {val:,.2f}",
                    "State": state,
                    "Event": event,
                    "Direction": direction,
                    "Strength": strength,
                    "Evidence": evidence
                })

    # 2. RSI
    if "RSI" in indicator_results or "RSI 14" in indicator_results:
        rsi_ser = indicator_results.get("RSI", indicator_results.get("RSI 14"))
        val = float(rsi_ser.iloc[-1])
        if not pd.isna(val):
            if val >= 70:
                state = "Overbought"
                direction = "Bearish"
                strength = "Strong"
                evidence = f"RSI ({val:.1f}) >= 70 (Overbought Region)"
            elif val <= 30:
                state = "Oversold"
                direction = "Bullish"
                strength = "Strong"
                evidence = f"RSI ({val:.1f}) <= 30 (Oversold Region)"
            elif val >= 50:
                state = "Positive Momentum"
                direction = "Bullish"
                strength = "Moderate"
                evidence = f"RSI ({val:.1f}) > 50 (Bullish Half)"
            else:
                state = "Negative Momentum"
                direction = "Bearish"
                strength = "Moderate"
                evidence = f"RSI ({val:.1f}) < 50 (Bearish Half)"
                
            event = "—"
            if len(df) > 1:
                if check_crossover_above(rsi_ser, pd.Series(50, index=df.index)).iloc[-1]:
                    event = "Crossed Above 50 ▲"
                elif check_crossover_below(rsi_ser, pd.Series(50, index=df.index)).iloc[-1]:
                    event = "Crossed Below 50 ▼"
                    
            signals.append({
                "Indicator": "RSI (14)",
                "Value": f"{val:.1f}",
                "State": state,
                "Event": event,
                "Direction": direction,
                "Strength": strength,
                "Evidence": evidence
            })

    # 3. MACD
    if "MACD" in indicator_results or "MACD_DF" in indicator_results:
        macd_obj = indicator_results.get("MACD_DF", indicator_results.get("MACD"))
        if isinstance(macd_obj, pd.DataFrame) and "MACD" in macd_obj and "Signal" in macd_obj:
            m_val = float(macd_obj["MACD"].iloc[-1])
            s_val = float(macd_obj["Signal"].iloc[-1])
            
            state = "Bullish" if m_val > s_val else "Bearish"
            direction = state
            strength = "Strong" if abs(m_val - s_val) > 0.5 else "Moderate"
            evidence = f"MACD ({m_val:.2f}) > Signal ({s_val:.2f})" if m_val > s_val else f"MACD ({m_val:.2f}) < Signal ({s_val:.2f})"
            
            event = "—"
            if len(df) > 1:
                if check_crossover_above(macd_obj["MACD"], macd_obj["Signal"]).iloc[-1]:
                    event = "Bullish Cross ▲"
                    strength = "Strong"
                elif check_crossover_below(macd_obj["MACD"], macd_obj["Signal"]).iloc[-1]:
                    event = "Bearish Cross ▼"
                    strength = "Strong"
                    
            signals.append({
                "Indicator": "MACD (12,26,9)",
                "Value": f"{m_val:.2f} / {s_val:.2f}",
                "State": state,
                "Event": event,
                "Direction": direction,
                "Strength": strength,
                "Evidence": evidence
            })

    # 4. ADX
    if "ADX" in indicator_results or "ADX 14" in indicator_results:
        adx_obj = indicator_results.get("ADX", indicator_results.get("ADX 14"))
        adx_val = float(adx_obj["ADX"].iloc[-1]) if isinstance(adx_obj, pd.DataFrame) else float(adx_obj.iloc[-1])
        
        if not pd.isna(adx_val):
            if adx_val >= 25:
                state = "Strong Trend"
                direction = "Bullish"
                strength = "Strong"
                evidence = f"ADX ({adx_val:.1f}) >= 25 (Active Trend Presence)"
            elif adx_val >= 20:
                state = "Moderate Trend"
                direction = "Neutral"
                strength = "Moderate"
                evidence = f"ADX ({adx_val:.1f}) between 20-25 (Developing Trend)"
            else:
                state = "Weak / Ranging"
                direction = "Neutral"
                strength = "Weak"
                evidence = f"ADX ({adx_val:.1f}) < 20 (No Active Directional Trend)"
                
            signals.append({
                "Indicator": "ADX (14)",
                "Value": f"{adx_val:.1f}",
                "State": state,
                "Event": "—",
                "Direction": direction,
                "Strength": strength,
                "Evidence": evidence
            })

    # 5. RVOL
    if "RVOL" in indicator_results or "RVOL 20" in indicator_results:
        rvol_ser = indicator_results.get("RVOL", indicator_results.get("RVOL 20"))
        rvol_val = float(rvol_ser.iloc[-1])
        if not pd.isna(rvol_val):
            if rvol_val >= 1.5:
                state = "High Volume"
                direction = "Bullish"
                strength = "Strong"
                evidence = f"RVOL ({rvol_val:.2f}x) >= 1.5x (Volume Expansion)"
            elif rvol_val >= 0.8:
                state = "Average Volume"
                direction = "Neutral"
                strength = "Moderate"
                evidence = f"RVOL ({rvol_val:.2f}x) Normal Trading Volume"
            else:
                state = "Low Volume"
                direction = "Neutral"
                strength = "Weak"
                evidence = f"RVOL ({rvol_val:.2f}x) < 0.8x (Below Average Volume)"
                
            signals.append({
                "Indicator": "RVOL (20)",
                "Value": f"{rvol_val:.2f}x",
                "State": state,
                "Event": "—",
                "Direction": direction,
                "Strength": strength,
                "Evidence": evidence
            })

    # 6. SuperTrend
    if "SuperTrend" in indicator_results:
        st_df = indicator_results["SuperTrend"]
        if "SuperTrend_Direction" in st_df:
            st_dir = st_df["SuperTrend_Direction"].iloc[-1]
            st_val = float(st_df["SuperTrend"].iloc[-1])
            state = "Bullish" if st_dir == 1 else "Bearish"
            direction = state
            strength = "Strong"
            evidence = f"SuperTrend Bullish ({st_val:,.2f})" if st_dir == 1 else f"SuperTrend Bearish ({st_val:,.2f})"
            
            event = "—"
            if len(df) > 1:
                prev_dir = st_df["SuperTrend_Direction"].iloc[-2]
                if prev_dir == -1 and st_dir == 1:
                    event = "Flipped Bullish ▲"
                elif prev_dir == 1 and st_dir == -1:
                    event = "Flipped Bearish ▼"
                    
            signals.append({
                "Indicator": "SuperTrend (10,3)",
                "Value": f"{st_val:,.2f}",
                "State": state,
                "Event": event,
                "Direction": direction,
                "Strength": strength,
                "Evidence": evidence
            })

    return pd.DataFrame(signals)
