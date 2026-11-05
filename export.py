"""
Export static artifacts for the web showcase (the frontend's data contract).

    python export.py --model checkpoints/level_s1_latest.pt --replay-run level_s1

Writes to web/public/data/:
  manifest.json         obs spec, feature scaling, model info, ticker index, scorecard
  traces/{TICKER}.json  daily replay: price, allocation, equity vs B&H, critic value,
                        scaled observation, integrated-gradient attribution per input,
                        2-D projection of the LSTM memory
  replays.json          the same window replayed by checkpoints across training
  model.onnx            single-step policy: (obs, h, c) -> (allocation, value, h, c)

Attribution note: integrated gradients explain today's allocation with respect to
today's observation while holding the LSTM memory fixed. They don't credit
information carried in memory from earlier days.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from agent.ppo import PPOAgent
from data.fetch import fetch
from env.trading_env import FEATURE_SCALING, OBS_DIM, OBS_NAMES, OBS_VERSION, TradingEnv
from evaluate import HELD_OUT
from trainer import TRAIN_BASKET
import metrics

IG_STEPS = 64
R = lambda a, d=4: np.round(np.asarray(a, dtype=np.float64), d).tolist()


def replay(agent: PPOAgent, df):
    """Deterministic replay recording everything the frontend shows per day."""
    env = TradingEnv(df)
    obs, _ = env.reset()
    hidden = agent.init_hidden()
    rows = {"obs": [], "h_prev": [], "c_prev": [], "h": [], "value": [], "alloc": []}
    while True:
        rows["obs"].append(obs.copy())
        rows["h_prev"].append(hidden[0].view(-1).numpy().copy())
        rows["c_prev"].append(hidden[1].view(-1).numpy().copy())
        action, _, value, hidden = agent.select_action(obs, hidden, deterministic=True)
        rows["h"].append(hidden[0].view(-1).numpy().copy())
        rows["value"].append(value)
        rows["alloc"].append(float(action[0]))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    out = {k: np.array(v) for k, v in rows.items()}
    out["values"] = np.array(env.values)
    return out


def integrated_gradients(net, obs, h_prev, c_prev, steps=IG_STEPS):
    """IG of the allocation w.r.t. each input, baseline = all-zero observation (the
    'neutral' scaled state), memory held fixed. Returns (T, OBS_DIM) and the
    completeness gap max|sum(attr) - (f(x) - f(0))|."""
    T = len(obs)
    alphas = (torch.arange(1, steps + 1, dtype=torch.float32) / steps).view(1, steps, 1)
    x = torch.as_tensor(obs, dtype=torch.float32).view(T, 1, -1)
    path = (alphas * x).reshape(T * steps, 1, -1).requires_grad_(True)
    h = torch.as_tensor(h_prev, dtype=torch.float32).repeat_interleave(steps, 0).unsqueeze(0)
    c = torch.as_tensor(c_prev, dtype=torch.float32).repeat_interleave(steps, 0).unsqueeze(0)
    mean, _, _, _ = net(path, (h.contiguous(), c.contiguous()))
    grads, = torch.autograd.grad(mean.sum(), path)
    attr = (grads.view(T, steps, -1).mean(1) * x.view(T, -1)).detach().numpy()
    with torch.no_grad():
        hh = torch.as_tensor(h_prev, dtype=torch.float32).unsqueeze(0)
        cc = torch.as_tensor(c_prev, dtype=torch.float32).unsqueeze(0)
        fx = net(x, (hh, cc))[0].view(-1).numpy()
        f0 = net(torch.zeros_like(x), (hh, cc))[0].view(-1).numpy()
    gap = float(np.abs(attr.sum(1) - (fx - f0)).max())
    return attr, f0, gap


def export_onnx(agent: PPOAgent, path: str):
    net = agent.net.eval()

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, obs, h, c):
            mean, _, value, (h2, c2) = self.net(obs, (h, c))
            return mean.view(1), value.view(1), h2, c2

    hd = net.hidden_dim
    args = (torch.zeros(1, 1, OBS_DIM), torch.zeros(1, 1, hd), torch.zeros(1, 1, hd))
    torch.onnx.export(Step(), args, path, input_names=["obs", "h", "c"],
                      output_names=["allocation", "value", "h_out", "c_out"],
                      opset_version=17, dynamo=False)
    # Parity check against PyTorch on random inputs, over a short rollout
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    rng = np.random.default_rng(0)
    h = c = np.zeros((1, 1, hd), np.float32)
    ht = (torch.zeros(1, 1, hd), torch.zeros(1, 1, hd))
    worst = 0.0
    for _ in range(50):
        o = rng.normal(size=(1, 1, OBS_DIM)).astype(np.float32)
        a, v, h, c = sess.run(None, {"obs": o, "h": h, "c": c})
        with torch.no_grad():
            m, _, vt, ht = net(torch.as_tensor(o), ht)
        worst = max(worst, abs(float(a[0]) - float(m)), abs(float(v[0]) - float(vt)))
    return worst


def fit_projection(hiddens: np.ndarray):
    """PCA to 2-D, fitted once over all exported tickers so maps are comparable."""
    mu = hiddens.mean(0)
    _, s, vt = np.linalg.svd(hiddens - mu, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    return mu, vt[:2], var[:2]


def checkpoint_replays(run: str, ckpt_dir: str, ticker: str, start: str, end: str, n: int = 10):
    files = glob.glob(os.path.join(ckpt_dir, f"{run}_[0-9]*.pt"))
    files = sorted(files, key=lambda f: int(re.search(r"_(\d+)\.pt$", f).group(1)))
    latest = os.path.join(ckpt_dir, f"{run}_latest.pt")
    if not files:
        return None
    idx = np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int))
    picks = [files[i] for i in idx]
    df = fetch(ticker, start, end)
    prices = df["close"].to_numpy()
    out = {"ticker": ticker, "dates": [str(d.date()) for d in df.index],
           "close": R(prices, 2), "bnh": R(10_000 * prices / prices[0], 2), "checkpoints": []}
    for f in picks + ([latest] if os.path.exists(latest) else []):
        agent, meta = PPOAgent.from_checkpoint(f)
        r = replay(agent, df)
        s = metrics.summarize(r["values"], r["alloc"])
        out["checkpoints"].append({"steps": int(meta.get("lifetime_steps", 0)),
                                   "allocation": R(r["alloc"], 3), "value": R(r["values"], 2),
                                   "sharpe": round(s["sharpe"], 3), "max_drawdown": round(s["max_drawdown"], 4),
                                   "total_return": round(s["total_return"], 4)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="web/public/data")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--tickers", nargs="*", default=TRAIN_BASKET + HELD_OUT)
    p.add_argument("--scorecard", default="results/scorecard.json")
    p.add_argument("--replay-run", default=None, help="Checkpoint prefix for the learning-replay exhibit")
    p.add_argument("--replay-ticker", default="SPY")
    p.add_argument("--replay-start", default="2020-01-01")
    p.add_argument("--replay-end", default="2020-12-31")
    p.add_argument("--ckpt-dir", default="checkpoints")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
    agent, meta = PPOAgent.from_checkpoint(args.model)
    agent.net.eval()
    if meta.get("obs_version") != OBS_VERSION:
        raise SystemExit(f"model obs_version {meta.get('obs_version')} != {OBS_VERSION}")

    runs = {}
    for t in args.tickers:
        try:
            df = fetch(t, args.start, args.end)
        except Exception as e:
            print(f"[export] {t}: skipped ({e})")
            continue
        runs[t] = (df, replay(agent, df))
    mu, comps, var = fit_projection(np.concatenate([r["h"] for _, r in runs.values()]))

    index = []
    for t, (df, r) in runs.items():
        attr, base, gap = integrated_gradients(agent.net, r["obs"], r["h_prev"], r["c_prev"])
        proj = (r["h"] - mu) @ comps.T
        prices = df["close"].to_numpy()
        bnh = 10_000 * prices / prices[0]
        s_agent = metrics.summarize(r["values"], r["alloc"])
        s_bnh = metrics.summarize(bnh)
        group = "train" if t in TRAIN_BASKET else "held_out"
        trace = {
            "ticker": t, "group": group, "start": args.start, "end": args.end,
            "dates": [str(d.date()) for d in df.index],
            "close": R(prices, 2), "equity": R(r["values"], 2), "bnh": R(bnh, 2),
            # decision on day i is made at close of dates[i], applied to day i+1
            "allocation": R(r["alloc"], 4), "critic_value": R(r["value"], 3),
            "obs": R(r["obs"], 3), "attribution": R(attr, 5), "attribution_baseline": R(base, 4),
            "memory_xy": R(proj, 3),
            "metrics": {"agent": s_agent, "bnh": s_bnh},
        }
        with open(os.path.join(args.out, "traces", f"{t}.json"), "w") as f:
            json.dump(trace, f, separators=(",", ":"))
        index.append({"ticker": t, "group": group, "sharpe": round(s_agent["sharpe"], 3),
                      "bnh_sharpe": round(s_bnh["sharpe"], 3),
                      "max_drawdown": round(s_agent["max_drawdown"], 4),
                      "bnh_max_drawdown": round(s_bnh["max_drawdown"], 4),
                      "avg_exposure": round(s_agent["avg_exposure"], 3)})
        print(f"[export] {t:5s} {group:8s} sharpe {s_agent['sharpe']:+.2f} vs {s_bnh['sharpe']:+.2f} | IG gap {gap:.1e}")

    parity = export_onnx(agent, os.path.join(args.out, "model.onnx"))
    print(f"[export] model.onnx parity vs torch: {parity:.1e}")

    if args.replay_run:
        rep = checkpoint_replays(args.replay_run, args.ckpt_dir, args.replay_ticker,
                                 args.replay_start, args.replay_end)
        if rep:
            with open(os.path.join(args.out, "replays.json"), "w") as f:
                json.dump(rep, f, separators=(",", ":"))
            print(f"[export] replays.json: {len(rep['checkpoints'])} checkpoints on {args.replay_ticker}")

    scorecard = None
    if args.scorecard and os.path.exists(args.scorecard):
        with open(args.scorecard) as f:
            scorecard = json.load(f)
    manifest = {
        "obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "obs_names": OBS_NAMES,
        "feature_scaling": FEATURE_SCALING, "obs_clip": 5.0,
        "model": {"path": "model.onnx", "hidden_dim": agent.net.hidden_dim,
                  "source": os.path.basename(args.model), "config": agent.config,
                  "train_meta": meta, "onnx_parity": parity},
        "memory_projection": {"method": "pca", "explained_variance": R(var, 4)},
        "attribution": {"method": "integrated_gradients", "steps": IG_STEPS,
                        "baseline": "zero scaled observation", "memory": "held fixed"},
        "window": {"start": args.start, "end": args.end}, "tickers": index,
        "scorecard": scorecard,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=float)
    print(f"[export] wrote {args.out}/manifest.json ({len(index)} tickers)")


if __name__ == "__main__":
    main()
