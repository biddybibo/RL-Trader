"""
Shared training loop used by train.py (CLI), main.py (dashboard) and walkforward.py.

Each episode samples a ticker from the basket and a random 1-year window, so the
agent can't memorise one price path. Time-limit truncation is bootstrapped with
the critic's value of the next state instead of being treated as a terminal state.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from agent.ppo import PPOAgent
from env.trading_env import OBS_DIM, OBS_VERSION, TradingEnv
import metrics

TRAIN_BASKET = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "NVDA", "AMZN"]
TRAIN_START, TRAIN_END = "2010-01-01", "2022-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"


@dataclass
class TrainConfig:
    total_steps: int = 500_000
    rollout_steps: int = 2048
    lr: float = 3e-4
    gamma: float = 0.9
    gae_lambda: float = 0.9
    n_epochs: int = 10
    episode_length: int = 252
    reward_scale: float = 100.0
    reward_mode: str = "level"
    risk_aversion: float = 1.0
    anneal_lr: bool = True
    ckpt_every: int = 50_000
    seed: int = 42
    device: str = "cpu"


@dataclass
class Hooks:
    on_step: Callable | None = None        # (trainer, step, env, ticker, action, reward) every `step_every`
    on_episode: Callable | None = None     # (trainer, summary dict)
    on_rollout: Callable | None = None     # (trainer, step, stats dict)
    step_every: int = 10


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, datasets: dict, cfg: TrainConfig, ckpt_dir: str | None = "checkpoints",
                 run_name: str = "multi", resume: bool = True, agent: PPOAgent | None = None,
                 hooks: Hooks | None = None, val_datasets: dict | None = None):
        if not datasets:
            raise ValueError("No training data")
        self.cfg, self.hooks = cfg, hooks or Hooks()
        self.ckpt_dir, self.run_name = ckpt_dir, run_name
        self.rng = np.random.default_rng(cfg.seed)
        seed_everything(cfg.seed)
        self.envs = {t: TradingEnv(df, episode_length=cfg.episode_length, random_start=True,
                                 reward_mode=cfg.reward_mode,
                                 risk_aversion=cfg.risk_aversion)
                     for t, df in datasets.items()}
        self.tickers = list(self.envs)
        self.val_datasets = val_datasets or {}
        self.best_val = -np.inf
        self.val_history: list[dict] = []
        self.lifetime_steps = 0
        self.episodes: list[dict] = []
        self.step = 0

        if agent is not None:
            self.agent = agent
        else:
            self.agent = None
            if resume and ckpt_dir and os.path.exists(self.ckpt_path):
                try:
                    self.agent, meta = PPOAgent.from_checkpoint(self.ckpt_path, cfg.device)
                    if meta.get("obs_version") != OBS_VERSION:
                        raise ValueError("observation version mismatch")
                    self.lifetime_steps = int(meta.get("lifetime_steps", 0))
                    print(f"[train] resumed {self.ckpt_path} ({self.lifetime_steps:,} lifetime steps)")
                except Exception as e:
                    print(f"[train] could not resume ({e}); starting fresh")
                    self.agent = None
            if self.agent is None:
                self.agent = PPOAgent(OBS_DIM, 1, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                                      n_epochs=cfg.n_epochs, rollout_steps=cfg.rollout_steps, device=cfg.device)

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.ckpt_dir, f"{self.run_name}_latest.pt")

    def save(self, milestone: bool = False):
        if not self.ckpt_dir:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        meta = {"lifetime_steps": self.lifetime_steps + self.step, "obs_version": OBS_VERSION,
                "train_config": asdict(self.cfg), "tickers": self.tickers,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.agent.save(self.ckpt_path, **meta)
        if milestone:
            self.agent.save(os.path.join(self.ckpt_dir, f"{self.run_name}_{meta['lifetime_steps']}.pt"), **meta)
        with open(os.path.join(self.ckpt_dir, f"{self.run_name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def validate(self) -> float | None:
        """Mean deterministic Sharpe over the validation window; saves {run}_best.pt on improvement."""
        if not self.val_datasets:
            return None
        from backtest import run_agent
        self.agent.net.eval()
        score = float(np.mean([metrics.sharpe(run_agent(self.agent, df)["values"])
                               for df in self.val_datasets.values()]))
        self.agent.net.train()
        steps = self.lifetime_steps + self.step
        self.val_history.append({"steps": steps, "val_sharpe": round(score, 4)})
        if score > self.best_val and self.ckpt_dir:
            self.best_val = score
            self.agent.save(os.path.join(self.ckpt_dir, f"{self.run_name}_best.pt"),
                            lifetime_steps=steps, obs_version=OBS_VERSION, val_sharpe=score,
                            train_config=asdict(self.cfg), tickers=self.tickers)
        if self.ckpt_dir:
            with open(os.path.join(self.ckpt_dir, f"{self.run_name}_val.json"), "w") as f:
                json.dump({"best": self.best_val, "history": self.val_history}, f, indent=1)
        return score

    def _new_episode(self):
        ticker = self.tickers[int(self.rng.integers(len(self.tickers)))]
        env = self.envs[ticker]
        obs, _ = env.reset(seed=int(self.rng.integers(2**31)))
        return ticker, env, obs, self.agent.init_hidden()

    def _summarize(self, ticker, env) -> dict:
        prices = env.episode_prices()
        bnh = env.initial_cash * prices / prices[0]
        s = metrics.summarize(np.array(env.values), np.array(env.allocations))
        s.update(ticker=ticker, steps=self.step, start=str(env.dates[env.start].date()),
                 end=str(env.dates[env.t].date()), bnh_return=round(metrics.total_return(bnh), 5),
                 bnh_sharpe=round(metrics.sharpe(bnh), 4))
        return s

    def run(self, stop_event: threading.Event | None = None,
            run_event: threading.Event | None = None) -> dict:
        """Train for cfg.total_steps. `run_event` cleared = paused; `stop_event` set = stop early."""
        cfg, agent, hooks = self.cfg, self.agent, self.hooks
        buf = agent.buffer
        ticker, env, obs, hidden = self._new_episode()
        ep_start = True
        next_ckpt = cfg.ckpt_every
        self.step = 0

        while self.step < cfg.total_steps:
            if run_event is not None and not run_event.is_set():
                run_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            self.step += 1
            h_in = hidden[0].view(-1).cpu().numpy()
            c_in = hidden[1].view(-1).cpu().numpy()
            action, log_prob, value, hidden = agent.select_action(obs, hidden)
            next_obs, reward, terminated, truncated, info = env.step(action)
            r = reward * cfg.reward_scale
            if truncated and not terminated:
                r += agent.gamma * agent.value(next_obs, hidden)
            done = terminated or truncated
            buf.add(obs, action, r, float(done), value, log_prob, h_in, c_in, float(ep_start))
            ep_start = False
            obs = next_obs

            if hooks.on_step and self.step % hooks.step_every == 0:
                hooks.on_step(self, self.step, env, ticker, action, reward)

            if done:
                summary = self._summarize(ticker, env)
                self.episodes.append(summary)
                if hooks.on_episode:
                    hooks.on_episode(self, summary)
                ticker, env, obs, hidden = self._new_episode()
                ep_start = True

            if buf.full:
                if cfg.anneal_lr:
                    agent.set_lr_fraction(max(0.1, 1.0 - self.step / cfg.total_steps))
                stats = agent.update(agent.value(obs, hidden) if not ep_start else 0.0)
                if hooks.on_rollout:
                    hooks.on_rollout(self, self.step, stats)

            if self.step >= next_ckpt:
                self.save(milestone=True)
                v = self.validate()
                if v is not None:
                    print(f"[val] step {self.lifetime_steps + self.step:,}: val Sharpe {v:+.3f} (best {self.best_val:+.3f})", flush=True)
                next_ckpt += cfg.ckpt_every

        self.save()
        return {"steps": self.step, "lifetime_steps": self.lifetime_steps + self.step,
                "episodes": len(self.episodes)}


def load_basket(tickers=TRAIN_BASKET, start=TRAIN_START, end=TRAIN_END, log=print) -> dict:
    from data.fetch import fetch
    out = {}
    for t in tickers:
        try:
            out[t] = fetch(t, start, end)
            log(f"[data] {t}: {len(out[t])} days")
        except Exception as e:
            log(f"[data] {t}: skipped ({e})")
    return out
