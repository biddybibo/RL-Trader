"""
Performance metrics shared by training, evaluation, walk-forward and the API.

All functions take a 1-D array of portfolio values (one per trading day) so every
part of the project computes Sharpe, drawdown, etc. the same way.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def daily_returns(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.zeros(0)
    return v[1:] / v[:-1] - 1.0


def sharpe(values: np.ndarray, risk_free: float = 0.0) -> float:
    r = daily_returns(values) - risk_free / TRADING_DAYS
    if len(r) < 2 or r.std() < 1e-12:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * r.mean() / r.std())


def sortino(values: np.ndarray, risk_free: float = 0.0) -> float:
    r = daily_returns(values) - risk_free / TRADING_DAYS
    if len(r) < 2:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    if dd < 1e-12:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * r.mean() / dd)


def max_drawdown(values: np.ndarray) -> float:
    """Most negative peak-to-trough move, as a fraction (e.g. -0.23)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1.0).min())


def total_return(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(v[-1] / v[0] - 1.0)


def cagr(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or v[0] <= 0:
        return 0.0
    years = (len(v) - 1) / TRADING_DAYS
    return float((v[-1] / v[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def calmar(values: np.ndarray) -> float:
    mdd = abs(max_drawdown(values))
    return float(cagr(values) / mdd) if mdd > 1e-9 else 0.0


def summarize(values: np.ndarray, allocations: np.ndarray | None = None) -> dict:
    out = {
        "total_return": round(total_return(values), 5),
        "cagr":         round(cagr(values), 5),
        "sharpe":       round(sharpe(values), 4),
        "sortino":      round(sortino(values), 4),
        "max_drawdown": round(max_drawdown(values), 5),
        "calmar":       round(calmar(values), 4),
    }
    if allocations is not None and len(allocations) > 0:
        a = np.asarray(allocations, dtype=float)
        out["avg_exposure"] = round(float(a.mean()), 4)
        out["turnover"] = round(float(np.abs(np.diff(a)).sum()) if len(a) > 1 else 0.0, 4)
    return out
