import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0).flatten()
    loss = np.where(delta < 0, -delta, 0.0).flatten()
    
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain_ema / (loss_ema + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['RSI14'] = rsi(df['Close'], 14)
    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'] = macd_line
    df['MACDsig'] = signal_line
    df['MACDhist'] = hist
    df['LogRet'] = np.log(df['Close']/df['Close'].shift(1))
    df = df.dropna()
    return df
