"""
Walk-forward evaluation — measures whether the agent generalizes or memorizes.

Uses expanding windows:
  Train on 2015 → growing end date, test on the following 6-month period.
  If test Sharpe improves as training data grows → agent is learning real patterns.
  If test Sharpe is flat or random            → agent is memorizing.

Results saved to checkpoints/{ticker}_walkforward.json
and served by the dashboard at /api/walkforward/{ticker}.

Usage:
    python walkforward.py --ticker AAPL --steps 20000
"""
import argparse
import json
import os
import time

import numpy as np

WINDOWS = [
    ("2015-01-01", "2016-12-31", "2017-01-01", "2017-06-30", "2017 H1"),
    ("2015-01-01", "2017-06-30", "2017-07-01", "2017-12-31", "2017 H2"),
    ("2015-01-01", "2017-12-31", "2018-01-01", "2018-06-30", "2018 H1"),
    ("2015-01-01", "2018-06-30", "2018-07-01", "2018-12-31", "2018 H2"),
    ("2015-01-01", "2018-12-31", "2019-01-01", "2019-06-30", "2019 H1"),
    ("2015-01-01", "2019-06-30", "2019-07-01", "2019-12-31", "2019 H2"),
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-06-30", "2020 H1"),
    ("2015-01-01", "2020-06-30", "2020-07-01", "2020-12-31", "2020 H2"),
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-06-30", "2021 H1"),
    ("2015-01-01", "2021-06-30", "2021-07-01", "2021-12-31", "2021 H2"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-06-30", "2022 H1"),
    ("2015-01-01", "2022-06-30", "2022-07-01", "2022-12-31", "2022 H2"),
]


def _train_window(agent, df, steps: int):
    from env.trading_env import TradingEnv
    agent.buffer.reset()
    env = TradingEnv(df)
    obs, _ = env.reset(seed=42)
    hidden = agent.init_hidden()
    for step in range(1, steps + 1):
        action, log_prob, value, hidden = agent.select_action(obs, hidden)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        hx = hidden[0].squeeze().cpu().numpy()
        cx = hidden[1].squeeze().cpu().numpy()
        agent.buffer.add(obs, action, reward, float(done), value, log_prob, hx, cx)
        obs = next_obs
        if done:
            obs, _ = env.reset()
            hidden = agent.init_hidden()
        if step % agent.rollout_steps == 0:
            _, _, last_value, _ = agent.select_action(obs, hidden)
            agent.update(last_value)


def _eval_window(agent, df):
    from env.trading_env import TradingEnv
    env = TradingEnv(df)
    obs, _ = env.reset()
    hidden = agent.init_hidden()
    portfolio_values = [env.initial_cash]
    while True:
        action, _, _, hidden = agent.select_action(obs, hidden)
        obs, _, terminated, truncated, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        if terminated or truncated:
            break

    pv   = np.array(portfolio_values)
    rets = np.diff(pv) / (pv[:-1] + 1e-8)
    sharpe     = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0
    total_ret  = float((pv[-1] / pv[0] - 1) * 100)
    peak       = np.maximum.accumulate(pv)
    max_dd     = float(((pv - peak) / (peak + 1e-8)).min() * 100)

    prices     = df["close"].values
    bnh        = (env.initial_cash / prices[0]) * prices
    bnh_rets   = np.diff(bnh) / bnh[:-1]
    bnh_sharpe = float(bnh_rets.mean() / (bnh_rets.std() + 1e-9) * np.sqrt(252))

    return {
        "sharpe":       round(sharpe, 3),
        "total_return": round(total_ret, 2),
        "max_drawdown": round(max_dd, 2),
        "bnh_sharpe":   round(bnh_sharpe, 3),
    }


def _write_results(results: list, ticker: str, steps_per_window: int, out_dir: str):
    test_sharpes  = [w["test_sharpe"]  for w in results]
    train_sharpes = [w["train_sharpe"] for w in results]
    avg_test  = round(float(np.mean(test_sharpes)),  3)
    gen_gap   = round(float(np.mean(train_sharpes)) - avg_test, 3)

    n = len(test_sharpes)
    xs = np.arange(n, dtype=float)
    slope = float(np.polyfit(xs, test_sharpes, 1)[0]) if n > 1 else 0.0

    output = {
        "ticker":             ticker,
        "steps_per_window":   steps_per_window,
        "windows":            results,
        "avg_test_sharpe":    avg_test,
        "generalization_gap": gen_gap,
        "trend_slope":        round(slope, 4),
        "updated":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(out_dir, f"{ticker}_walkforward.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return output


def run(ticker: str, steps_per_window: int, out_dir: str = "checkpoints",
        status: dict | None = None):
    from data.fetch import fetch
    from agent.ppo import PPOAgent
    from env.trading_env import TradingEnv as _TradingEnv

    os.makedirs(out_dir, exist_ok=True)
    # Determine obs_dim dynamically from a sample env
    _sample_df  = fetch(ticker, "2018-01-01", "2018-06-30")
    _sample_env = _TradingEnv(_sample_df)
    obs_dim     = _sample_env.observation_space.shape[0]
    agent = PPOAgent(obs_dim=obs_dim, act_dim=1, rollout_steps=2048)
    results = []

    for i, (tr_s, tr_e, te_s, te_e, label) in enumerate(WINDOWS):
        t0 = time.time()
        print(f"[{i+1}/{len(WINDOWS)}] {label}  train={tr_s}..{tr_e}  test={te_s}..{te_e}")

        if status is not None:
            status["progress"] = i

        df_train = fetch(ticker, tr_s, tr_e)
        _train_window(agent, df_train, steps_per_window)
        train_m = _eval_window(agent, df_train)

        df_test = fetch(ticker, te_s, te_e)
        test_m  = _eval_window(agent, df_test)

        results.append({
            "label":         label,
            "train_sharpe":  train_m["sharpe"],
            "test_sharpe":   test_m["sharpe"],
            "test_return":   test_m["total_return"],
            "test_drawdown": test_m["max_drawdown"],
            "bnh_sharpe":    test_m["bnh_sharpe"],
        })
        print(f"        train_sharpe={train_m['sharpe']:+.3f}  "
              f"test_sharpe={test_m['sharpe']:+.3f}  "
              f"bnh={test_m['bnh_sharpe']:+.3f}  ({time.time()-t0:.0f}s)")

        # Write after each window so the dashboard shows partial results
        output = _write_results(results, ticker, steps_per_window, out_dir)

    avg_test = output["avg_test_sharpe"]
    gen_gap  = output["generalization_gap"]
    slope    = output["trend_slope"]
    trend_str = "IMPROVING" if slope > 0.02 else ("DECLINING" if slope < -0.02 else "STABLE")
    path = os.path.join(out_dir, f"{ticker}_walkforward.json")

    print(f"\n=== Walk-Forward Summary ({ticker}) ===")
    print(f"Avg test Sharpe:      {avg_test:+.3f}")
    print(f"Generalization gap:   {gen_gap:+.3f}  (lower = less overfitting)")
    print(f"Trend:                {trend_str}  (slope={slope:+.4f})")
    print(f"Results saved ->      {path}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--steps",  type=int, default=20_000,
                   help="Training steps per window (20k=fast, 100k=thorough)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    args = p.parse_args()
    run(args.ticker, args.steps, args.ckpt_dir)
