import pandas as pd
import numpy as np
from technical.volatility import calculate_atr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX, +DI, -DI)"""
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr = calculate_atr(df, period=period)
    
    pos_di = 100 * (pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    neg_di = 100 * (pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    
    dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return pd.DataFrame({
        "ADX": adx,
        "Plus_DI": pos_di,
        "Minus_DI": neg_di
    }, index=df.index)

def calculate_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Aroon Indicator (Up, Down, Oscillator)"""
    aroon_up = df["High"].rolling(window=period + 1).apply(
        lambda x: ((period - (period - 1 - np.argmax(x))) / period) * 100, raw=True
    )
    aroon_down = df["Low"].rolling(window=period + 1).apply(
        lambda x: ((period - (period - 1 - np.argmin(x))) / period) * 100, raw=True
    )
    aroon_osc = aroon_up - aroon_down
    
    return pd.DataFrame({
        "Aroon_Up": aroon_up,
        "Aroon_Down": aroon_down,
        "Aroon_Osc": aroon_osc
    }, index=df.index)

def calculate_vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Vortex Indicator (VI+, VI-)"""
    vm_plus = (df["High"] - df["Low"].shift(1)).abs()
    vm_minus = (df["Low"] - df["High"].shift(1)).abs()
    
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    trn = tr.rolling(window=period).sum()
    vortex_plus = vm_plus.rolling(window=period).sum() / trn.replace(0, np.nan)
    vortex_minus = vm_minus.rolling(window=period).sum() / trn.replace(0, np.nan)
    
    return pd.DataFrame({
        "Vortex_Plus": vortex_plus,
        "Vortex_Minus": vortex_minus
    }, index=df.index)
