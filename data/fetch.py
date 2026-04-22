"""
Download OHLCV + macro data from yfinance and compute features.

Observation columns (17 total — consumed by TradingEnv._obs):
  Price / momentum (6): log_return, sma_10, sma_30, volatility, rsi, macd
  Volume (2):           volume_ratio, atr_norm
  Range / calendar (4): hw_proximity, day_sin, day_cos, month_sin
  Macro (2):            vix_norm, rate_norm
  Raw (3):              close, high, low  (used by env internals, not in obs)
"""
import numpy as np
import pandas as pd
import yfinance as yf

# Macro series fetched once and reused across calls in the same process
_macro_cache: dict[str, pd.Series] = {}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return (close.ewm(span=fast, min_periods=fast).mean()
            - close.ewm(span=slow, min_periods=slow).mean()) / close


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _fetch_macro(symbol: str, start: str, end: str) -> pd.Series:
    """Download a single macro time-series, return a date-indexed Series."""
    key = f"{symbol}_{start}_{end}"
    if key in _macro_cache:
        return _macro_cache[key]
    try:
        raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("empty")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index.get_level_values(0)
                                     if isinstance(close.index, pd.MultiIndex)
                                     else close.index)
        _macro_cache[key] = close.astype(float)
        return _macro_cache[key]
    except Exception:
        return pd.Series(dtype=float)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker} [{start}, {end}]")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=pd.to_datetime(raw.index))
    df["close"]  = raw["Close"].astype(float)
    df["high"]   = raw["High"].astype(float)
    df["low"]    = raw["Low"].astype(float)
    df["volume"] = raw["Volume"].astype(float)

    # ── Price / momentum ─────────────────────────────────────────────
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["sma_10"]     = df["close"].rolling(10).mean() / df["close"] - 1
    df["sma_30"]     = df["close"].rolling(30).mean() / df["close"] - 1
    df["volatility"] = df["log_return"].rolling(20).std()
    df["rsi"]        = _rsi(df["close"]) / 100.0
    df["macd"]       = _macd(df["close"])

    # ── Volume ───────────────────────────────────────────────────────
    avg_vol_20           = df["volume"].rolling(20).mean()
    df["volume_ratio"]   = df["volume"] / (avg_vol_20 + 1)  # >1 = above-avg activity
    df["atr_norm"]       = _atr(df["high"], df["low"], df["close"]) / df["close"]

    # ── 52-week range position ────────────────────────────────────────
    high_52 = df["close"].rolling(252, min_periods=30).max()
    low_52  = df["close"].rolling(252, min_periods=30).min()
    df["hw_proximity"] = (df["close"] - low_52) / (high_52 - low_52 + 1e-8)

    # ── Calendar (sin/cos to avoid ordinal bias) ─────────────────────
    df["day_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["day_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)

    # ── Macro: VIX (fear) and 10Y rate (rate regime) ─────────────────
    vix  = _fetch_macro("^VIX", start, end)
    tnx  = _fetch_macro("^TNX", start, end)

    if not vix.empty:
        df["vix_norm"]  = vix.reindex(df.index, method="ffill").fillna(20.0) / 30.0
    else:
        df["vix_norm"]  = 0.67  # ~20 VIX / 30

    if not tnx.empty:
        df["rate_norm"] = tnx.reindex(df.index, method="ffill").fillna(3.0) / 10.0
    else:
        df["rate_norm"] = 0.30  # ~3% / 10

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = fetch("AAPL", "2020-01-01", "2022-12-31")
    print(df.tail())
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
