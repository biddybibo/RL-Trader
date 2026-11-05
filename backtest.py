"""
Deterministic backtests of the agent and baselines on a feature frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics
from env.trading_env import TradingEnv


def run_agent(agent, df: pd.DataFrame) -> dict:
    """Replay the policy mean over the whole frame. Returns daily values/allocations."""
    env = TradingEnv(df)
    obs, _ = env.reset()
    hidden = agent.init_hidden()
    while True:
        action, _, _, hidden = agent.select_action(obs, hidden, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return {"values": np.array(env.values), "allocations": np.array(env.allocations)}


def run_allocations(df: pd.DataFrame, allocations: np.ndarray) -> np.ndarray:
    """Portfolio values for a fixed allocation path, with the env's costs."""
    env = TradingEnv(df)
    env.reset()
    for a in allocations[: env.n_steps - 1]:
        env.step(np.array([a]))
    return np.array(env.values)


def sma_crossover_allocations(ticker: str, df: pd.DataFrame, fast=50, slow=200) -> np.ndarray:
    from data.fetch import load_history
    close = load_history(ticker)["close"]
    signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    return signal.reindex(df.index).fillna(0.0).to_numpy()[:-1]


def random_agent_sharpes(df: pd.DataFrame, n: int = 200, seed: int = 0) -> np.ndarray:
    """Sharpe ratios of random agents with persistent (AR(1)) allocations."""
    rng = np.random.default_rng(seed)
    steps = len(df) - 1
    out = np.empty(n)
    for k in range(n):
        a, path = rng.uniform(), np.empty(steps)
        for i in range(steps):
            a = float(np.clip(0.9 * a + 0.1 * rng.uniform(), 0, 1)) if rng.uniform() > 0.05 else rng.uniform()
            path[i] = a
        out[k] = metrics.sharpe(run_allocations(df, path))
    return out


def evaluate_ticker(agent, ticker: str, df: pd.DataFrame, n_random: int = 200) -> dict:
    res = run_agent(agent, df)
    prices = df["close"].to_numpy()
    bnh = 10_000.0 * prices / prices[0]
    sma = run_allocations(df, sma_crossover_allocations(ticker, df))
    agent_m = metrics.summarize(res["values"], res["allocations"])
    out = {
        "ticker": ticker, "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
        "agent": agent_m, "buy_and_hold": metrics.summarize(bnh), "sma_50_200": metrics.summarize(sma),
    }
    if n_random:
        rs = random_agent_sharpes(df, n_random)
        out["random"] = {"sharpe_p50": round(float(np.median(rs)), 4),
                         "sharpe_p95": round(float(np.percentile(rs, 95)), 4),
                         "agent_percentile": round(float((rs < agent_m["sharpe"]).mean() * 100), 1)}
    return out
