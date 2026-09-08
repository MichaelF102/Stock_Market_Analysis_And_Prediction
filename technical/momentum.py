import pandas as pd
import numpy as np

def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Use Wilder's smoothing
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close") -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD)"""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Hist": histogram
    }, index=df.index)

def calculate_roc(df: pd.DataFrame, period: int = 12, column: str = "Close") -> pd.Series:
    """Rate of Change (ROC)"""
    return ((df[column] - df[column].shift(period)) / df[column].shift(period).replace(0, np.nan)) * 100

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)"""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    
    raw_k = 100 * ((df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan))
    stoch_k = raw_k.rolling(window=slowing).mean()
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        "Stoch_K": stoch_k,
        "Stoch_D": stoch_d
    }, index=df.index)

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R"""
    high_max = df["High"].rolling(window=period).max()
    low_min = df["Low"].rolling(window=period).min()
    
    w_r = -100 * ((high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan))
    return w_r
