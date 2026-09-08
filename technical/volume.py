import pandas as pd
import numpy as np

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)"""
    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    return obv

def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow (CMF)"""
    high_low = df["High"] - df["Low"]
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low.replace(0, np.nan)
    mf_volume = mf_multiplier.fillna(0) * df["Volume"]
    
    cmf = mf_volume.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum().replace(0, np.nan)
    return cmf

def calculate_adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line (ADL)"""
    high_low = df["High"] - df["Low"]
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low.replace(0, np.nan)
    mf_volume = mf_multiplier.fillna(0) * df["Volume"]
    return mf_volume.cumsum()
