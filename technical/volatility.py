import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """Bollinger Bands (Upper, Middle, Lower, Width, PctB)"""
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    width = ((upper - lower) / middle.replace(0, np.nan)) * 100
    pct_b = (df[column] - lower) / (upper - lower).replace(0, np.nan)
    
    return pd.DataFrame({
        "BB_Upper": upper,
        "BB_Middle": middle,
        "BB_Lower": lower,
        "BB_Width": width,
        "BB_PctB": pct_b
    }, index=df.index)

def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channels"""
    middle = df["Close"].ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df, period=atr_period)
    
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)
    
    return pd.DataFrame({
        "KC_Upper": upper,
        "KC_Middle": middle,
        "KC_Lower": lower
    }, index=df.index)

def calculate_donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channels"""
    upper = df["High"].rolling(window=period).max()
    lower = df["Low"].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return pd.DataFrame({
        "DC_Upper": upper,
        "DC_Middle": middle,
        "DC_Lower": lower
    }, index=df.index)
