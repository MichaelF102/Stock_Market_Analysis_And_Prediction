import pandas as pd
import numpy as np
from technical.trend import calculate_ema, calculate_vwma

def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MFI)"""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    
    tp_diff = typical_price.diff()
    pos_flow = np.where(tp_diff > 0, raw_money_flow, 0.0)
    neg_flow = np.where(tp_diff < 0, raw_money_flow, 0.0)
    
    pos_mf = pd.Series(pos_flow, index=df.index).rolling(window=period).sum()
    neg_mf = pd.Series(neg_flow, index=df.index).rolling(window=period).sum()
    
    money_ratio = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.fillna(50)

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (VWAP)"""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical_price * df["Volume"]
    return tp_vol.cumsum() / df["Volume"].cumsum().replace(0, np.nan)

def calculate_chaikin_ad_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Accumulation/Distribution Oscillator"""
    high_low = df["High"] - df["Low"]
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low.replace(0, np.nan)
    mf_volume = mf_multiplier.fillna(0) * df["Volume"]
    adl = mf_volume.cumsum()
    
    fast_ema = adl.ewm(span=fast, adjust=False).mean()
    slow_ema = adl.ewm(span=slow, adjust=False).mean()
    return fast_ema - slow_ema

def calculate_rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative Volume (RVOL)"""
    avg_vol = df["Volume"].rolling(window=period).mean()
    return df["Volume"] / avg_vol.replace(0, np.nan)

def calculate_pvt(df: pd.DataFrame) -> pd.Series:
    """Price Volume Trend (PVT)"""
    pct_change = df["Close"].pct_change().fillna(0)
    pvt = (pct_change * df["Volume"]).cumsum()
    return pvt

def calculate_volume_oscillator(df: pd.DataFrame, fast: int = 14, slow: int = 28) -> pd.Series:
    """Volume Oscillator (%)"""
    fast_vol = df["Volume"].ewm(span=fast, adjust=False).mean()
    slow_vol = df["Volume"].ewm(span=slow, adjust=False).mean()
    return ((fast_vol - slow_vol) / slow_vol.replace(0, np.nan)) * 100

def calculate_volume_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Volume Rate of Change (%)"""
    return ((df["Volume"] - df["Volume"].shift(period)) / df["Volume"].shift(period).replace(0, np.nan)) * 100

def calculate_elder_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Elder's Force Index (EFI)"""
    raw_efi = df["Close"].diff() * df["Volume"]
    return raw_efi.ewm(span=period, adjust=False).mean()

def calculate_ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ease of Movement (EOM)"""
    distance_moved = ((df["High"] + df["Low"]) / 2) - ((df["High"].shift(1) + df["Low"].shift(1)) / 2)
    box_ratio = (df["Volume"] / 100000000) / (df["High"] - df["Low"]).replace(0, np.nan)
    eom_1period = distance_moved / box_ratio.replace(0, np.nan)
    return eom_1period.rolling(window=period).mean()

def calculate_nvi(df: pd.DataFrame) -> pd.Series:
    """Negative Volume Index (NVI)"""
    close_pct = df["Close"].pct_change().fillna(0)
    vol_diff = df["Volume"].diff().fillna(0)
    
    nvi = pd.Series(1000.0, index=df.index)
    for i in range(1, len(df)):
        if vol_diff.iloc[i] < 0:
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + close_pct.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]
    return nvi

def calculate_pvi(df: pd.DataFrame) -> pd.Series:
    """Positive Volume Index (PVI)"""
    close_pct = df["Close"].pct_change().fillna(0)
    vol_diff = df["Volume"].diff().fillna(0)
    
    pvi = pd.Series(1000.0, index=df.index)
    for i in range(1, len(df)):
        if vol_diff.iloc[i] > 0:
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + close_pct.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]
    return pvi
