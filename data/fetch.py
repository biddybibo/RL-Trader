"""
Market data + feature pipeline.

Prices come from yfinance and are cached on disk (data/cache/*.csv), so repeat runs
don't re-download and results are reproducible offline.

Features are computed on the *full* cached history and only then sliced to
[start, end]. Every feature is trailing (uses data up to and including day t), so
this adds no look-ahead, and it means a 6-month test window keeps all of its days
and gets real 52-week ranges instead of losing its first month to indicator warm-up.

Feature columns (14, consumed by TradingEnv, which applies scaling):
  Price / momentum (6): log_return, sma_10, sma_30, volatility, rsi, macd
  Volume (2):           volume_ratio, atr_norm
  Range / calendar (4): hw_proximity, day_sin, day_cos, month_sin
  Macro (2):            vix_norm, rate_norm
Raw columns kept for the env and charts: close, high, low, volume
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_MAX_AGE_S = 3 * 24 * 3600
HISTORY_START = "2000-01-01"

FEATURE_COLUMNS = [
    "log_return", "sma_10", "sma_30", "volatility", "rsi", "macd",
    "volume_ratio", "atr_norm",
    "hw_proximity", "day_sin", "day_cos", "month_sin",
    "vix_norm", "rate_norm",
]

_mem_cache: dict[str, pd.DataFrame] = {}


# ── Raw price download with disk cache ────────────────────────────────

def _cache_path(symbol: str) -> str:
    safe = symbol.replace("^", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _download(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=HISTORY_START, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def load_prices(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Full daily OHLCV history for a symbol, from disk cache when fresh."""
    if symbol in _mem_cache and not refresh:
        return _mem_cache[symbol]
    path = _cache_path(symbol)
    fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_MAX_AGE_S
    if fresh and not refresh:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
    else:
        try:
            df = _download(symbol)
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(path)
        except Exception:
            if os.path.exists(path):  # offline: fall back to a stale cache
                df = pd.read_csv(path, index_col="date", parse_dates=True)
            else:
                raise
    _mem_cache[symbol] = df
    return df


# ── Indicators ────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return (close.ewm(span=fast, min_periods=fast).mean()
            - close.ewm(span=slow, min_periods=slow).mean()) / close


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(px: pd.DataFrame, vix: pd.Series | None, tnx: pd.Series | None) -> pd.DataFrame:
    df = pd.DataFrame(index=px.index)
    df["close"] = px["close"]
    df["high"] = px["high"]
    df["low"] = px["low"]
    df["volume"] = px["volume"]

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["sma_10"] = df["close"].rolling(10).mean() / df["close"] - 1
    df["sma_30"] = df["close"].rolling(30).mean() / df["close"] - 1
    df["volatility"] = df["log_return"].rolling(20).std()
    df["rsi"] = _rsi(df["close"]) / 100.0
    df["macd"] = _macd(df["close"])

    avg_vol_20 = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (avg_vol_20 + 1)
    df["atr_norm"] = _atr(df["high"], df["low"], df["close"]) / df["close"]

    high_52 = df["close"].rolling(252, min_periods=60).max()
    low_52 = df["close"].rolling(252, min_periods=60).min()
    df["hw_proximity"] = (df["close"] - low_52) / (high_52 - low_52 + 1e-8)

    df["day_sin"] = np.sin(2 * np.pi * df.index.day_of_week / 5)
    df["day_cos"] = np.cos(2 * np.pi * df.index.day_of_week / 5)
    df["month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)

    # Macro series: forward-fill only (never back-fill, which would leak the future)
    if vix is not None and not vix.empty:
        df["vix_norm"] = vix.reindex(df.index).ffill() / 30.0
    else:
        df["vix_norm"] = 20.0 / 30.0
    if tnx is not None and not tnx.empty:
        df["rate_norm"] = tnx.reindex(df.index).ffill() / 10.0
    else:
        df["rate_norm"] = 0.30

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def load_history(ticker: str) -> pd.DataFrame:
    """Full feature frame for a ticker over all cached history."""
    key = f"features::{ticker}"
    if key in _mem_cache:
        return _mem_cache[key]
    px = load_prices(ticker)
    try:
        vix = load_prices("^VIX")["close"]
    except Exception:
        vix = None
    try:
        tnx = load_prices("^TNX")["close"]
    except Exception:
        tnx = None
    df = build_features(px, vix, tnx)
    _mem_cache[key] = df
    return df


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Feature frame for [start, end] (inclusive), with indicators warmed up on prior history."""
    df = load_history(ticker)
    out = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if out.empty:
        raise ValueError(f"No data for {ticker} in [{start}, {end}]")
    return out.copy()


if __name__ == "__main__":
    d = fetch("AAPL", "2020-01-01", "2022-12-31")
    print(d.tail())
    print("Shape:", d.shape)
