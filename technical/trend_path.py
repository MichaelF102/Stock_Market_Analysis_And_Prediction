import pandas as pd
import numpy as np
from technical.volatility import calculate_atr

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend Indicator"""
    atr = calculate_atr(df, period=period)
    hl2 = (df["High"] + df["Low"]) / 2
    
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index) # 1 for Bullish, -1 for Bearish
    
    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df["Close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df["Close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        prev_dir = direction.iloc[i-1]
        if prev_dir == 1:
            if df["Close"].iloc[i] <= final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if df["Close"].iloc[i] >= final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
                
    return pd.DataFrame({
        "SuperTrend": supertrend,
        "SuperTrend_Direction": direction
    }, index=df.index)

def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Ichimoku Kinko Hyo"""
    tenkan = (df["High"].rolling(window=tenkan_period).max() + df["Low"].rolling(window=tenkan_period).min()) / 2
    kijun = (df["High"].rolling(window=kijun_period).max() + df["Low"].rolling(window=kijun_period).min()) / 2
    
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = ((df["High"].rolling(window=senkou_b_period).max() + df["Low"].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
    chikou = df["Close"].shift(-displacement)
    
    return pd.DataFrame({
        "Tenkan_Sen": tenkan,
        "Kijun_Sen": kijun,
        "Senkou_Span_A": senkou_a,
        "Senkou_Span_B": senkou_b,
        "Chikou_Span": chikou
    }, index=df.index)

def calculate_parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR"""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)
    
    psar = np.zeros(n)
    if n < 2:
        return pd.Series(psar, index=df.index)
        
    bull = True
    af = af_step
    ep = high[0]
    psar[0] = low[0]
    
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
                    
    return pd.Series(psar, index=df.index)

def calculate_zigzag(df: pd.DataFrame, deviation: float = 5.0) -> pd.Series:
    """ZigZag Indicator based on percentage reversal threshold"""
    close = df["Close"].values
    n = len(df)
    zigzag = np.full(n, np.nan)
    if n == 0:
        return pd.Series(zigzag, index=df.index)
        
    trend = 0  # 1 for up, -1 for down
    last_pivot = close[0]
    last_pivot_idx = 0
    zigzag[0] = close[0]
    
    for i in range(1, n):
        pct_change = ((close[i] - last_pivot) / last_pivot) * 100
        if trend == 0:
            if abs(pct_change) >= deviation:
                trend = 1 if pct_change > 0 else -1
                last_pivot = close[i]
                last_pivot_idx = i
                zigzag[i] = close[i]
        elif trend == 1:
            if close[i] > last_pivot:
                zigzag[last_pivot_idx] = np.nan
                last_pivot = close[i]
                last_pivot_idx = i
                zigzag[i] = close[i]
            elif pct_change <= -deviation:
                trend = -1
                last_pivot = close[i]
                last_pivot_idx = i
                zigzag[i] = close[i]
        elif trend == -1:
            if close[i] < last_pivot:
                zigzag[last_pivot_idx] = np.nan
                last_pivot = close[i]
                last_pivot_idx = i
                zigzag[i] = close[i]
            elif pct_change >= deviation:
                trend = 1
                last_pivot = close[i]
                last_pivot_idx = i
                zigzag[i] = close[i]
                
    return pd.Series(zigzag, index=df.index)

def calculate_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """Williams Fractals (High / Low Williams 5-bar pattern)"""
    high = df["High"]
    low = df["Low"]
    
    fractal_high = (
        (high > high.shift(1)) & 
        (high > high.shift(2)) & 
        (high > high.shift(-1)) & 
        (high > high.shift(-2))
    )
    
    fractal_low = (
        (low < low.shift(1)) & 
        (low < low.shift(2)) & 
        (low < low.shift(-1)) & 
        (low < low.shift(-2))
    )
    
    return pd.DataFrame({
        "Fractal_High": np.where(fractal_high, high, np.nan),
        "Fractal_Low": np.where(fractal_low, low, np.nan)
    }, index=df.index)

def calculate_cmo(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (CMO)"""
    delta = df["Close"].diff()
    gains = delta.where(delta > 0, 0.0).rolling(window=period).sum()
    losses = (-delta.where(delta < 0, 0.0)).rolling(window=period).sum()
    
    cmo = 100 * ((gains - losses) / (gains + losses).replace(0, np.nan))
    return cmo

def calculate_trix(df: pd.DataFrame, period: int = 15, signal_period: int = 9) -> pd.DataFrame:
    """TRIX (Triple Exponential Moving Average Oscillator)"""
    ema1 = df["Close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan)) * 10000
    trix_signal = trix.ewm(span=signal_period, adjust=False).mean()
    
    return pd.DataFrame({
        "TRIX": trix,
        "Signal": trix_signal
    }, index=df.index)
