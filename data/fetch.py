"""
Download OHLCV data from yfinance and compute technical indicator features.
Returns a cleaned DataFrame with columns used directly by the trading env.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    return (ema_fast - ema_slow) / close  # normalised by price


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV and compute features.

    Returns columns:
        close, log_return, sma_10, sma_30, volatility, rsi, macd
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} [{start}, {end}]")

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"].astype(float)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["sma_10"] = df["close"].rolling(10).mean() / df["close"] - 1
    df["sma_30"] = df["close"].rolling(30).mean() / df["close"] - 1
    df["volatility"] = df["log_return"].rolling(20).std()

    df["rsi"] = _rsi(df["close"]) / 100.0  # scale to [0, 1]
    df["macd"] = _macd(df["close"])

    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    df = fetch("AAPL", "2015-01-01", "2022-12-31")
    print(df.tail())
    print("Shape:", df.shape)
