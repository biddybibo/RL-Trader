"""
Scorecard: deterministic backtests on held-out data, against baselines.

  * Out-of-time:   training tickers on 2023-2024
  * Out-of-sample: tickers never seen in training, on 2023-2024
  * Baselines:     buy & hold, SMA 50/200 crossover, 200 random agents
  * Several --model paths (e.g. one per seed) are aggregated with mean ± std.

Usage:
    python evaluate.py --model checkpoints/multi_latest.pt
    python evaluate.py --model checkpoints/seed1_latest.pt checkpoints/seed2_latest.pt --plot
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from agent.ppo import PPOAgent
from backtest import evaluate_ticker, run_agent
from data.fetch import fetch
from trainer import TRAIN_BASKET

HELD_OUT = ["JPM", "XOM", "KO", "UNH", "HD", "PG", "CAT", "DIS", "WMT", "INTC"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["checkpoints/multi_latest.pt"])
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--tickers", nargs="*", default=None, help="Default: training basket + held-out set")
    p.add_argument("--n-random", type=int, default=200)
    p.add_argument("--out", default="results/scorecard.json")
    p.add_argument("--plot", action="store_true", help="Save an equity-curve PNG per model for SPY")
    return p.parse_args()


def _agg(rows: list[dict], who: str, key: str):
    vals = [r[who][key] for r in rows if who in r]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def main():
    args = parse_args()
    tickers = args.tickers or (TRAIN_BASKET + HELD_OUT)
    data = {}
    for t in tickers:
        try:
            data[t] = fetch(t, args.start, args.end)
        except Exception as e:
            print(f"[skip] {t}: {e}")

    per_model = []
    for path in args.model:
        agent, meta = PPOAgent.from_checkpoint(path)
        rows = [evaluate_ticker(agent, t, df, args.n_random) | {"held_out": t not in TRAIN_BASKET}
                for t, df in data.items()]
        per_model.append({"model": path, "lifetime_steps": meta.get("lifetime_steps"), "rows": rows})
        if args.plot and "SPY" in data:
            _plot(agent, data["SPY"], os.path.splitext(path)[0] + "_SPY.png")

    # ── Print table (first model per ticker, then aggregate across models) ──
    print(f"\nScorecard {args.start} → {args.end}  ({len(args.model)} model(s))\n")
    hdr = f"{'Ticker':<7}{'Set':<10}{'Agent SR':>9}{'B&H SR':>8}{'SMA SR':>8}{'Agent DD':>10}{'B&H DD':>8}{'Rand pct':>9}{'Expo':>6}"
    print(hdr); print("-" * len(hdr))
    for i, t in enumerate(data):
        rs = [m["rows"][i] for m in per_model]
        a = np.mean([r["agent"]["sharpe"] for r in rs])
        dd = np.mean([r["agent"]["max_drawdown"] for r in rs])
        pct = np.mean([r.get("random", {}).get("agent_percentile", np.nan) for r in rs])
        ex = np.mean([r["agent"]["avg_exposure"] for r in rs])
        r0 = rs[0]
        print(f"{t:<7}{'held-out' if r0['held_out'] else 'train':<10}{a:>9.2f}{r0['buy_and_hold']['sharpe']:>8.2f}"
              f"{r0['sma_50_200']['sharpe']:>8.2f}{dd*100:>9.1f}%{r0['buy_and_hold']['max_drawdown']*100:>7.1f}%"
              f"{pct:>8.0f}%{ex:>6.2f}")

    summary = {}
    for group, flag in (("held_out", True), ("train_tickers", False)):
        rows = [r for m in per_model for r in m["rows"] if r["held_out"] == flag]
        if not rows:
            continue
        summary[group] = {
            who: {k: dict(zip(("mean", "std"), map(lambda x: round(x, 4), _agg(rows, who, k))))
                  for k in ("sharpe", "max_drawdown", "total_return")}
            for who in ("agent", "buy_and_hold", "sma_50_200")
        }
        beats = np.mean([r["agent"]["sharpe"] > r["buy_and_hold"]["sharpe"] for r in rows])
        summary[group]["agent_beats_bnh_sharpe_frac"] = round(float(beats), 3)
        print(f"\n{group}: agent Sharpe {summary[group]['agent']['sharpe']['mean']:+.2f} "
              f"± {summary[group]['agent']['sharpe']['std']:.2f} | B&H {summary[group]['buy_and_hold']['sharpe']['mean']:+.2f} "
              f"| SMA {summary[group]['sma_50_200']['sharpe']['mean']:+.2f} | beats B&H on {beats*100:.0f}% of tickers")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"period": [args.start, args.end], "summary": summary, "models": per_model,
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nSaved -> {args.out}")


def _plot(agent, df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = run_agent(agent, df)
    prices = df["close"].to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.plot(df.index, res["values"], label="Agent", color="#F0A63C")
    a1.plot(df.index, 10_000 * prices / prices[0], label="Buy & hold", color="#7D8A9C", ls="--")
    a1.legend(); a1.grid(alpha=.3); a1.set_ylabel("Portfolio ($)")
    a2.fill_between(df.index[:-1], res["allocations"], color="#F0A63C", alpha=.5)
    a2.set_ylim(0, 1); a2.set_ylabel("Allocation"); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)
    print(f"Chart -> {out}")


if __name__ == "__main__":
    main()
