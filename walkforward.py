"""
Walk-forward evaluation: does the agent generalise or memorise?

Expanding windows: train on 2015 → growing end date (continuing the same agent),
then backtest deterministically on the next 6 months.

Results: checkpoints/{ticker}_walkforward.json (served at /api/walkforward/{ticker}).

Usage:
    python walkforward.py --ticker AAPL --steps 20000
"""
import argparse
import json
import os
import time

import numpy as np

import metrics

WINDOWS = [
    (f"2015-01-01", end, ts, te, label)
    for end, ts, te, label in [
        ("2016-12-31", "2017-01-01", "2017-06-30", "2017 H1"),
        ("2017-06-30", "2017-07-01", "2017-12-31", "2017 H2"),
        ("2017-12-31", "2018-01-01", "2018-06-30", "2018 H1"),
        ("2018-06-30", "2018-07-01", "2018-12-31", "2018 H2"),
        ("2018-12-31", "2019-01-01", "2019-06-30", "2019 H1"),
        ("2019-06-30", "2019-07-01", "2019-12-31", "2019 H2"),
        ("2019-12-31", "2020-01-01", "2020-06-30", "2020 H1"),
        ("2020-06-30", "2020-07-01", "2020-12-31", "2020 H2"),
        ("2020-12-31", "2021-01-01", "2021-06-30", "2021 H1"),
        ("2021-06-30", "2021-07-01", "2021-12-31", "2021 H2"),
        ("2021-12-31", "2022-01-01", "2022-06-30", "2022 H1"),
        ("2022-06-30", "2022-07-01", "2022-12-31", "2022 H2"),
    ]
]


def _write_results(results, ticker, steps_per_window, out_dir):
    test = [w["test_sharpe"] for w in results]
    train = [w["train_sharpe"] for w in results]
    slope = float(np.polyfit(np.arange(len(test)), test, 1)[0]) if len(test) > 1 else 0.0
    output = {
        "ticker": ticker, "steps_per_window": steps_per_window, "windows": results,
        "avg_test_sharpe": round(float(np.mean(test)), 3),
        "avg_bnh_sharpe": round(float(np.mean([w["bnh_sharpe"] for w in results])), 3),
        "generalization_gap": round(float(np.mean(train) - np.mean(test)), 3),
        "trend_slope": round(slope, 4),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, f"{ticker}_walkforward.json"), "w") as f:
        json.dump(output, f, indent=2)
    return output


def run(ticker: str, steps_per_window: int, out_dir: str = "checkpoints", status: dict | None = None,
        seed: int = 42):
    from agent.ppo import PPOAgent
    from backtest import run_agent
    from data.fetch import fetch
    from env.trading_env import OBS_DIM
    from trainer import TrainConfig, Trainer

    os.makedirs(out_dir, exist_ok=True)
    agent = PPOAgent(OBS_DIM, 1, rollout_steps=2048)
    results, output = [], {}

    for i, (tr_s, tr_e, te_s, te_e, label) in enumerate(WINDOWS):
        t0 = time.time()
        if status is not None:
            status["progress"] = i
        df_train, df_test = fetch(ticker, tr_s, tr_e), fetch(ticker, te_s, te_e)
        cfg = TrainConfig(total_steps=steps_per_window, seed=seed + i, anneal_lr=False)
        Trainer({ticker: df_train}, cfg, ckpt_dir=None, agent=agent).run()

        train_v = run_agent(agent, df_train)["values"]
        test = run_agent(agent, df_test)
        prices = df_test["close"].to_numpy()
        tm = metrics.summarize(test["values"], test["allocations"])
        results.append({
            "label": label,
            "train_sharpe": round(metrics.sharpe(train_v), 3),
            "test_sharpe": round(tm["sharpe"], 3),
            "test_return": round(tm["total_return"] * 100, 2),
            "test_drawdown": round(tm["max_drawdown"] * 100, 2),
            "test_exposure": tm["avg_exposure"],
            "bnh_sharpe": round(metrics.sharpe(prices), 3),
        })
        r = results[-1]
        print(f"[{i+1}/{len(WINDOWS)}] {label}: train {r['train_sharpe']:+.2f}  test {r['test_sharpe']:+.2f}  "
              f"B&H {r['bnh_sharpe']:+.2f}  ({time.time()-t0:.0f}s)")
        output = _write_results(results, ticker, steps_per_window, out_dir)

    s = output["trend_slope"]
    print(f"\n{ticker}: avg test Sharpe {output['avg_test_sharpe']:+.3f} (B&H {output['avg_bnh_sharpe']:+.3f}), "
          f"gap {output['generalization_gap']:+.3f}, trend {'IMPROVING' if s > .02 else 'DECLINING' if s < -.02 else 'STABLE'}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--ckpt-dir", default="checkpoints")
    a = p.parse_args()
    run(a.ticker, a.steps, a.ckpt_dir)
