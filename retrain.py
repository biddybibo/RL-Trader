"""
Monthly continual learning for the adaptive agent.

    python retrain.py --steps 200000

Warm-starts from models/adaptive.pt (or models/frozen.pt the first time), keeps
training on a rolling window that ends a year ago, picks the best of the *newly trained*
checkpoints by Sharpe on the most recent year (it always moves on, even if the
starting point scored higher: that's the experiment), and writes it back to
models/adaptive.pt. The frozen
agent never changes, so the live desk compares "stop learning" with "keep learning".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile

from agent.ppo import PPOAgent
from trainer import TrainConfig, Trainer, load_basket

FROZEN, ADAPTIVE, HISTORY = "models/frozen.pt", "models/adaptive.pt", "models/adaptive_history.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--train-years", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    today = dt.date.today()
    val_start = today - dt.timedelta(days=365)
    train_start = val_start.replace(year=val_start.year - args.train_years)
    src = ADAPTIVE if os.path.exists(ADAPTIVE) else FROZEN
    agent, meta = PPOAgent.from_checkpoint(src)
    print(f"[retrain] warm start from {src}; train {train_start} → {val_start}, validate {val_start} → {today}")

    cfg = TrainConfig(total_steps=args.steps, reward_mode="mv", risk_aversion=5.0, seed=args.seed,
                      gamma=agent.gamma, ckpt_every=25_000)
    train = load_basket(start=str(train_start), end=str(val_start))
    val = load_basket(start=str(val_start), end=str(today), log=lambda *_: None)
    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(train, cfg, ckpt_dir=tmp, run_name="adaptive", resume=False, agent=agent, val_datasets=val)
        tr.lifetime_steps = int(meta.get("lifetime_steps", 0))
        tr.run()
        best, best_meta = PPOAgent.from_checkpoint(os.path.join(tmp, "adaptive_best.pt"))

    os.makedirs("models", exist_ok=True)
    best_meta.update(train_start=str(train_start), train_end=str(val_start), val_end=str(today),
                     warm_start=os.path.basename(src), updated=str(today))
    best.save(ADAPTIVE, **best_meta)
    hist = json.load(open(HISTORY)) if os.path.exists(HISTORY) else []
    hist.append({"date": str(today), "val_sharpe": round(float(best_meta.get("val_sharpe", 0)), 4),
                 "lifetime_steps": best_meta.get("lifetime_steps"), "train_end": str(val_start)})
    with open(HISTORY, "w") as f:
        json.dump(hist, f, indent=1)
    print(f"[retrain] saved {ADAPTIVE}: val Sharpe {best_meta.get('val_sharpe', 0):+.3f}")


if __name__ == "__main__":
    main()
