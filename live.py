"""
Live desk: run the agents on the latest market data and publish tomorrow's calls.

    python live.py                    # after the US close, nightly (GitHub Actions)

For each agent (models/frozen.pt, and models/adaptive.pt if present) and each
watch-list ticker, the agent replays deterministically from the first trading day
of the current year to the latest close, carrying its memory and its own paper
account, exactly as in training. The final step is the call for the next session.

Outputs (web/public/data/live/):
  today.json   latest calls with the top drivers (integrated gradients)
  ledger.json  append-only log of every published call: what was said, and when
  record.json  forward-test paper record built only from published calls

The ledger is the source of truth: past entries are never rewritten, even if the
data provider later revises a price. The record applies each published allocation
to the following session's close-to-close return, minus the env's trading costs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np

import data.fetch as fetch_mod
from agent.ppo import PPOAgent
from env.trading_env import OBS_NAMES, OBS_VERSION, TradingEnv
from export import integrated_gradients, replay

WATCHLIST = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV",
             "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
AGENTS = {"frozen": "models/frozen.pt", "adaptive": "models/adaptive.pt"}
OUT = "web/public/data/live"
COST = TradingEnv.TRANSACTION_COST


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj, compact=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":") if compact else None, indent=None if compact else 1)
    os.replace(tmp, path)


def latest_calls(agent: PPOAgent, ticker: str, anchor: str):
    hist = fetch_mod.load_history(ticker)
    df = hist[hist.index >= anchor]
    if len(df) < 25:                       # early January: keep at least ~a month of memory
        df = hist.iloc[-25:]
    r = replay(agent, df)
    attr, base, _ = integrated_gradients(agent.net, r["obs"][-1:], r["h_prev"][-1:], r["c_prev"][-1:])
    order = np.argsort(-np.abs(attr[0]))[:5]
    alloc = r["alloc"]
    return {
        "ticker": ticker,
        "as_of": str(df.index[-1].date()),
        "close": round(float(df["close"].iloc[-1]), 4),
        "day_return": round(float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1), 5),
        "allocation": round(float(alloc[-1]), 4),
        "previous": round(float(alloc[-2]), 4),
        "drivers": [{"input": OBS_NAMES[k], "effect": round(float(attr[0][k]), 4)} for k in order],
        "baseline": round(float(base[0]), 4),
        "spark": [round(float(v), 4) for v in alloc[-30:]],
    }


def build_record(ledger: list[dict]) -> dict:
    """Paper record from published calls only. A call made at the close of day d
    earns the next recorded close for that ticker, i.e. it's settled one session later."""
    out = {}
    for agent in sorted({e["agent"] for e in ledger}):
        series = {}
        for t in sorted({e["ticker"] for e in ledger if e["agent"] == agent}):
            calls = sorted((e for e in ledger if e["agent"] == agent and e["ticker"] == t), key=lambda e: e["as_of"])
            eq, bh, prev_a, rows = 1.0, 1.0, 0.0, []
            for a, b in zip(calls, calls[1:]):
                ret = b["close"] / a["close"] - 1
                eq *= 1 + a["allocation"] * ret - COST * abs(a["allocation"] - prev_a)
                bh *= 1 + ret
                prev_a = a["allocation"]
                rows.append({"date": b["as_of"], "agent": round(eq, 6), "bnh": round(bh, 6)})
            series[t] = rows
        # equal-weight across tickers, aligned on settled dates
        dates = sorted({r["date"] for rows in series.values() for r in rows})
        agg = []
        for d in dates:
            pts = [next((r for r in rows if r["date"] == d), None) for rows in series.values()]
            pts = [p for p in pts if p]
            agg.append({"date": d, "agent": round(float(np.mean([p["agent"] for p in pts])), 6),
                        "bnh": round(float(np.mean([p["bnh"] for p in pts])), 6), "n": len(pts)})
        out[agent] = {"tickers": series, "equal_weight": agg,
                      "first_call": min(e["as_of"] for e in ledger if e["agent"] == agent),
                      "settled_days": len(dates)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=WATCHLIST)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    fetch_mod.CACHE_MAX_AGE_S = 0   # always download today's bars (falls back to cache if offline)
    os.makedirs(args.out, exist_ok=True)
    ledger_path = os.path.join(args.out, "ledger.json")
    ledger = load_json(ledger_path, [])
    seen = {(e["agent"], e["ticker"], e["as_of"]) for e in ledger}
    now = dt.datetime.now(dt.timezone.utc)
    anchor = f"{now.year}-01-01"

    today = {"generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "anchor": anchor, "agents": {}}
    for name, path in AGENTS.items():
        if not os.path.exists(path):
            continue
        agent, meta = PPOAgent.from_checkpoint(path)
        agent.net.eval()
        if meta.get("obs_version") != OBS_VERSION:
            raise SystemExit(f"{path}: obs_version mismatch")
        calls = []
        for t in args.tickers:
            try:
                c = latest_calls(agent, t, anchor)
            except Exception as e:
                print(f"[live] {name} {t}: skipped ({e})")
                continue
            calls.append(c)
            key = (name, t, c["as_of"])
            if key not in seen:
                ledger.append({"agent": name, "ticker": t, "as_of": c["as_of"], "close": c["close"],
                               "allocation": c["allocation"], "published": today["generated"]})
                seen.add(key)
            print(f"[live] {name:8s} {t:5s} {c['as_of']}  {c['allocation']:.0%} in stock (prev {c['previous']:.0%})")
        today["agents"][name] = {"model": os.path.basename(path), "trained_through": meta.get("train_end"),
                                 "updated": meta.get("updated"), "calls": calls}

    ledger.sort(key=lambda e: (e["as_of"], e["agent"], e["ticker"]))
    save_json(ledger_path, ledger, compact=True)
    save_json(os.path.join(args.out, "today.json"), today)
    save_json(os.path.join(args.out, "record.json"), build_record(ledger), compact=True)
    print(f"[live] ledger: {len(ledger)} calls")


if __name__ == "__main__":
    main()
