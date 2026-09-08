import pandas as pd
import numpy as np

def calculate_sma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Simple Moving Average"""
    return df[column].rolling(window=period).mean()

def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Exponential Moving Average"""
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_wma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return df[column].rolling(window=period).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

def calculate_hma(df: pd.DataFrame, period: int = 20, column: str = "Close") -> pd.Series:
    """Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = calculate_wma(df, period=half_period, column=column)
    wma_full = calculate_wma(df, period=period, column=column)
    
    raw_hma = 2 * wma_half - wma_full
    temp_df = pd.DataFrame({"raw": raw_hma})
    return calculate_wma(temp_df, period=sqrt_period, column="raw")

def calculate_vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Weighted Moving Average"""
    pv = df["Close"] * df["Volume"]
    pv_sum = pv.rolling(window=period).sum()
    vol_sum = df["Volume"].rolling(window=period).sum()
    return pv_sum / vol_sum.replace(0, np.nan)
