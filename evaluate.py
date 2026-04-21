"""
Backtest the trained PPO agent on 2023-2024 held-out data.

Metrics reported:
  - Total return
  - Annualised Sharpe ratio
  - Maximum drawdown
  - Comparison vs. buy-and-hold benchmark

Usage:
    python evaluate.py [--model checkpoints/ppo_final.pt]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless; swap to TkAgg/Qt5Agg if you want a window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.fetch import fetch
from env.trading_env import TradingEnv
from agent.ppo import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/ppo_final.pt")
    p.add_argument("--out", default="backtest.png")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def sharpe(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def max_drawdown(portfolio_values: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - peak) / (peak + 1e-8)
    return float(dd.min())


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

def run_backtest(agent: PPOAgent, env: TradingEnv, df):
    obs, _ = env.reset()
    portfolio_values = [env.initial_cash]
    allocations = []

    while True:
        action, _, _ = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        allocations.append(float(np.clip(action[0], 0.0, 1.0)))
        if terminated or truncated:
            break

    return np.array(portfolio_values), np.array(allocations)


def buy_and_hold(df, initial_cash: float = 10_000.0) -> np.ndarray:
    prices = df["close"].values
    shares = initial_cash / prices[0]
    return shares * prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Fetching evaluation data (AAPL 2023-2024)…")
    df = fetch("AAPL", "2023-01-01", "2024-12-31")
    print(f"  {len(df)} trading days")

    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=args.device)

    if os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print(f"[warn] Model file not found ({args.model}). Running untrained agent for demo.")

    ppo_values, allocs = run_backtest(agent, env, df)
    bnh_values = buy_and_hold(df)

    # Align lengths (env steps n-1 times so may differ by 1)
    n = min(len(ppo_values), len(bnh_values))
    ppo_values = ppo_values[:n]
    bnh_values = bnh_values[:n]
    dates = df.index[:n]

    ppo_returns = np.diff(np.log(ppo_values + 1e-8))
    bnh_returns = np.diff(np.log(bnh_values + 1e-8))

    ppo_total_ret = (ppo_values[-1] / ppo_values[0] - 1) * 100
    bnh_total_ret = (bnh_values[-1] / bnh_values[0] - 1) * 100

    print("\n=== Backtest Metrics (2023-2024) ===")
    print(f"{'Metric':<22} {'PPO Agent':>12} {'Buy-and-Hold':>14}")
    print("-" * 50)
    print(f"{'Total Return (%)':22} {ppo_total_ret:>12.2f} {bnh_total_ret:>14.2f}")
    print(f"{'Sharpe Ratio':22} {sharpe(ppo_returns):>12.3f} {sharpe(bnh_returns):>14.3f}")
    print(f"{'Max Drawdown (%)':22} {max_drawdown(ppo_values)*100:>12.2f} {max_drawdown(bnh_values)*100:>14.2f}")
    print(f"{'Final Value ($)':22} {ppo_values[-1]:>12.2f} {bnh_values[-1]:>14.2f}")

    # ---- Plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dates, ppo_values, label="PPO Agent", color="royalblue", linewidth=1.5)
    ax1.plot(dates, bnh_values, label="Buy & Hold", color="orange", linewidth=1.5, linestyle="--")
    ax1.set_title("PPO RL Trading Agent vs Buy-and-Hold (AAPL 2023-2024)", fontsize=13)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    n_alloc = min(len(allocs), len(dates) - 1)
    ax2.fill_between(dates[:n_alloc], allocs[:n_alloc], alpha=0.5, color="royalblue", label="Stock allocation")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Allocation")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nChart saved -> {args.out}")


if __name__ == "__main__":
    main()
