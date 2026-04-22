import asyncio
import json as json_lib
import os
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Tickers the agent trains on simultaneously (multi-asset generalisation)
TRAIN_BASKET = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "NVDA", "AMZN"]
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2022-12-31"


# ── Shared agent state ────────────────────────────────────────────────

class AgentState:
    def __init__(self):
        self.is_training          = False
        self.is_evaluating        = False
        self.episode              = 0
        self.total_steps          = 0
        self.total_steps_target   = 500_000
        self.current_loss         = 0.0
        self.portfolio_value      = 10_000.0
        self.cash                 = 10_000.0
        self.shares_held          = 0.0
        self.current_position     = 0.0
        self.ticker               = "AAPL"
        self.sharpe_ratio         = 0.0
        self.max_drawdown         = 0.0
        self.total_return         = 0.0
        self.portfolio_history: deque = deque(maxlen=300)
        self.action_history: deque   = deque(maxlen=300)
        self.reward_history: deque   = deque(maxlen=300)
        self.loss_history: deque     = deque(maxlen=200)
        self.trade_log: deque        = deque(maxlen=50)
        self._price_history: deque   = deque(maxlen=300)
        # per-rollout efficiency histories (one entry per PPO update)
        self.eff_loss: deque         = deque(maxlen=1000)
        self.eff_sharpe: deque       = deque(maxlen=1000)
        self.eff_return: deque       = deque(maxlen=1000)
        self.eff_win_rate: deque     = deque(maxlen=1000)
        self.eff_steps: deque        = deque(maxlen=1000)
        self.win_rate                = 0.0
        self.rollout_count           = 0
        self.steps_per_sec           = 0.0
        # additional risk metrics
        self.sortino_ratio           = 0.0
        self.calmar_ratio            = 0.0
        self.win_loss_ratio          = 0.0
        self.avg_turnover            = 0.0
        self.current_ep_ticker       = ""
        self._lock                   = threading.Lock()

agent_state = AgentState()
_training_thread: threading.Thread | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _save_meta(path: str, lifetime_steps: int, ticker: str):
    import json
    with open(path, "w") as f:
        json.dump({"lifetime_steps": lifetime_steps, "ticker": ticker, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def _load_lifetime_steps(ticker: str) -> int:
    import json
    meta_path = f"checkpoints/{ticker}_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("lifetime_steps", 0)
    return 0


# ── Real PPO training (runs in background thread) ─────────────────────

def _compute_metrics(pv_arr: np.ndarray):
    """Return (sharpe, sortino, calmar, max_dd, total_ret) from portfolio value array."""
    if len(pv_arr) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    rets     = np.diff(pv_arr) / (pv_arr[:-1] + 1e-8)
    mean_r   = rets.mean()
    std_r    = rets.std() + 1e-9
    downside = rets[rets < 0]
    down_std = downside.std() + 1e-9 if len(downside) > 1 else std_r

    sharpe   = float(mean_r / std_r * np.sqrt(252))
    sortino  = float(mean_r / down_std * np.sqrt(252))
    peak     = np.maximum.accumulate(pv_arr)
    max_dd   = float(((pv_arr - peak) / (peak + 1e-8)).min() * 100)
    total_r  = float((pv_arr[-1] / pv_arr[0] - 1) * 100)
    calmar   = total_r / (abs(max_dd) + 1e-8)
    return round(sharpe, 2), round(sortino, 2), round(calmar, 2), round(max_dd, 2), round(total_r, 2)


def _run_ppo_training(s: AgentState):
    import json
    import torch
    from data.fetch import fetch
    from env.trading_env import TradingEnv
    from agent.ppo import PPOAgent

    # Pin PyTorch to 1 thread — avoids core contention with FastAPI's thread pool
    torch.set_num_threads(1)

    # ── Pre-load entire multi-asset basket ───────────────────────────
    print("[train] Loading multi-asset basket...")
    ticker_data: dict[str, object] = {}
    for t in TRAIN_BASKET:
        try:
            ticker_data[t] = fetch(t, TRAIN_START, TRAIN_END)
            print(f"[train]   {t}: {len(ticker_data[t])} days")
        except Exception as e:
            print(f"[train]   {t}: skipped ({e})")

    if not ticker_data:
        print("[train] No data loaded — aborting")
        s.is_training = False
        return

    # ── Init agent from any env ───────────────────────────────────────
    sample_df  = next(iter(ticker_data.values()))
    sample_env = TradingEnv(sample_df)
    obs_dim    = sample_env.observation_space.shape[0]
    act_dim    = sample_env.action_space.shape[0]
    agent      = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, rollout_steps=2048)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path  = "checkpoints/multi_latest.pt"
    meta_path  = "checkpoints/multi_meta.json"
    lifetime_steps = 0

    if os.path.exists(ckpt_path):
        try:
            agent.load(ckpt_path)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    lifetime_steps = json.load(f).get("lifetime_steps", 0)
            print(f"[train] Resumed multi-asset ({lifetime_steps:,} lifetime steps)")
        except Exception:
            print("[train] Checkpoint incompatible, starting fresh")

    # ── First episode setup ───────────────────────────────────────────
    def _new_env():
        t  = random.choice(list(ticker_data.keys()))
        s.current_ep_ticker = t
        return TradingEnv(ticker_data[t]), t

    env, cur_ticker = _new_env()
    obs, _   = env.reset(seed=42)
    hidden   = agent.init_hidden()
    next_ckpt = 50_000
    ep_turnover_acc: list[float] = []

    # Pre-cache df lengths to avoid recomputing min() inside the hot loop
    df_last_idx: dict[str, int] = {t: len(df) - 1 for t, df in ticker_data.items()}
    cur_df_last = df_last_idx[cur_ticker]

    # Batch state writes every N steps — broadcast only reads 10×/sec
    _STATE_WRITE_EVERY = 10
    _local_pv:    float = 10_000.0
    _local_cash:  float = 10_000.0
    _local_alloc: float = 0.0
    _local_rew:   float = 0.0
    _local_price: float = 0.0

    for step in range(1, s.total_steps_target + 1):
        if not s.is_training:
            break

        action, log_prob, value, hidden = agent.select_action(obs, hidden)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        hx = hidden[0].squeeze().cpu().numpy()
        cx = hidden[1].squeeze().cpu().numpy()
        agent.buffer.add(obs, action, reward, float(done), value, log_prob, hx, cx)
        obs   = next_obs
        alloc = float(np.clip(action[0], 0.0, 1.0))

        # Keep locals hot — deques and shared state only updated every N steps
        _local_pv    = env.portfolio_value
        _local_cash  = env.cash
        _local_alloc = alloc
        _local_rew   = float(reward)
        price_idx    = min(env.t, cur_df_last)
        _local_price = float(ticker_data[cur_ticker].iloc[price_idx]["close"])

        if step % _STATE_WRITE_EVERY == 0:
            s.total_steps      = step
            s.portfolio_value  = round(_local_pv, 2)
            s.cash             = round(_local_cash, 2)
            s.shares_held      = env.shares
            s.current_position = _local_alloc
            s.portfolio_history.append(round(_local_pv, 2))
            s.action_history.append(round(_local_alloc, 3))
            s.reward_history.append(round(_local_rew, 5))
            s._price_history.append(round(_local_price, 2))

        if done:
            s.episode += 1
            ep_turnover_acc.append(env.episode_turnover)
            env, cur_ticker = _new_env()
            obs, _ = env.reset()
            hidden = agent.init_hidden()
            cur_df_last = df_last_idx[cur_ticker]

        # ── PPO update ────────────────────────────────────────────────
        if step % agent.rollout_steps == 0:
            _, _, last_value, _ = agent.select_action(obs, hidden)
            agent.update(last_value)
            s.current_loss = round(agent.last_loss, 4)
            s.loss_history.append(s.current_loss)

            pv_arr = np.array(list(s.portfolio_history))
            sharpe, sortino, calmar, max_dd, total_r = _compute_metrics(pv_arr)
            s.sharpe_ratio  = sharpe
            s.sortino_ratio = sortino
            s.calmar_ratio  = calmar
            s.max_drawdown  = max_dd
            s.total_return  = total_r

            rw_arr   = np.array(list(s.reward_history))
            wins_arr = rw_arr[rw_arr > 0]
            loss_arr = rw_arr[rw_arr < 0]
            s.win_rate       = round(len(wins_arr) / max(len(rw_arr), 1) * 100, 1)
            avg_win  = float(wins_arr.mean())       if len(wins_arr) else 0.0
            avg_loss = float(np.abs(loss_arr).mean()) if len(loss_arr) else 1e-8
            s.win_loss_ratio = round(avg_win / (avg_loss + 1e-9), 2)

            if ep_turnover_acc:
                s.avg_turnover = round(float(np.mean(ep_turnover_acc[-20:])), 2)

            s.rollout_count += 1
            s.eff_loss.append(s.current_loss)
            s.eff_sharpe.append(s.sharpe_ratio)
            s.eff_return.append(s.total_return)
            s.eff_win_rate.append(s.win_rate)
            s.eff_steps.append(step)

            action_type = "BUY" if alloc > 0.6 else ("SELL" if alloc < 0.2 else "HOLD")
            s.trade_log.appendleft({
                "time":   time.strftime("%H:%M:%S"),
                "action": action_type,
                "price":  round(float(ticker_data[cur_ticker].iloc[price_idx]["close"]), 2),
                "alloc":  round(alloc, 2),
                "pnl":    round(float(reward * 10_000), 2),
            })

        if step >= next_ckpt:
            agent.save(ckpt_path)
            _save_meta(meta_path, lifetime_steps + step, "multi")
            print(f"[train] checkpoint @ {lifetime_steps + step:,} steps")
            next_ckpt += 50_000

    agent.save(ckpt_path)
    _save_meta(meta_path, lifetime_steps + s.total_steps_target, "multi")
    print(f"[train] done — lifetime {lifetime_steps + s.total_steps_target:,} steps")
    s.is_training = False


# ── WebSocket manager ─────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()


# ── Broadcast loop (4 ticks/sec) ──────────────────────────────────────

_perf_last_steps: int   = 0
_perf_last_time:  float = 0.0


async def broadcast_loop():
    global _perf_last_steps, _perf_last_time
    _perf_last_time = asyncio.get_event_loop().time()
    while True:
        if agent_state.is_training and manager.active:
            s = agent_state

            now = asyncio.get_event_loop().time()
            dt  = now - _perf_last_time
            if dt >= 1.0:
                s.steps_per_sec   = round((s.total_steps - _perf_last_steps) / dt)
                _perf_last_steps  = s.total_steps
                _perf_last_time   = now

            await manager.broadcast({
                "type":               "tick",
                "total_steps":        s.total_steps,
                "total_steps_target": s.total_steps_target,
                "episode":            s.episode,
                "loss":               s.current_loss,
                "portfolio_value":    round(s.portfolio_value, 2),
                "cash":               round(s.cash, 2),
                "shares_held":        round(s.shares_held, 4),
                "position":           round(s.current_position, 3),
                "sharpe":             s.sharpe_ratio,
                "sortino":            s.sortino_ratio,
                "calmar":             s.calmar_ratio,
                "max_drawdown":       s.max_drawdown,
                "total_return":       s.total_return,
                "win_loss_ratio":     s.win_loss_ratio,
                "avg_turnover":       s.avg_turnover,
                "price":              s._price_history[-1] if s._price_history else 182.0,
                "reward":             s.reward_history[-1] if s.reward_history else 0.0,
                "trade_log":          list(s.trade_log)[:10],
                "is_training":        s.is_training,
                "win_rate":           s.win_rate,
                "rollout_count":      s.rollout_count,
                "steps_per_sec":      s.steps_per_sec,
                "current_ep_ticker":  s.current_ep_ticker,
            })
        elif not agent_state.is_training and manager.active:
            await manager.broadcast({"type": "training_complete", "steps": agent_state.total_steps})

        await asyncio.sleep(0.1)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(broadcast_loop())
    yield


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="RL Trader API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    s = agent_state
    return {
        "is_training":        s.is_training,
        "is_evaluating":      s.is_evaluating,
        "total_steps":        s.total_steps,
        "total_steps_target": s.total_steps_target,
        "episode":            s.episode,
        "ticker":             s.ticker,
        "portfolio_value":    round(s.portfolio_value, 2),
        "sharpe":             s.sharpe_ratio,
        "max_drawdown":       s.max_drawdown,
        "total_return":       s.total_return,
        "portfolio_history":  list(s.portfolio_history),
        "action_history":     list(s.action_history),
        "loss_history":       list(s.loss_history),
        "price_history":      list(s._price_history),
        "trade_log":          list(s.trade_log),
        "lifetime_steps":     _load_lifetime_steps(s.ticker),
        "eff_loss":           list(s.eff_loss),
        "eff_sharpe":         list(s.eff_sharpe),
        "eff_return":         list(s.eff_return),
        "eff_win_rate":       list(s.eff_win_rate),
        "eff_steps":          list(s.eff_steps),
        "win_rate":           s.win_rate,
        "rollout_count":      s.rollout_count,
        "steps_per_sec":      s.steps_per_sec,
        "sortino":            s.sortino_ratio,
        "calmar":             s.calmar_ratio,
        "win_loss_ratio":     s.win_loss_ratio,
        "avg_turnover":       s.avg_turnover,
        "current_ep_ticker":  s.current_ep_ticker,
    }


class TrainConfig(BaseModel):
    ticker: str = "AAPL"
    total_steps: int = 500_000


@app.post("/api/train/start")
def start_training(config: TrainConfig):
    global _training_thread
    s = agent_state
    if s.is_training:
        return {"ok": False, "message": "Already training"}

    s.ticker             = config.ticker
    s.total_steps_target = config.total_steps
    s.total_steps        = 0
    s.episode            = 0
    s.current_loss       = 0.0
    s.portfolio_value    = 10_000.0
    s.cash               = 10_000.0
    s.shares_held        = 0.0
    s.current_position   = 0.0
    s.sharpe_ratio       = 0.0
    s.max_drawdown       = 0.0
    s.total_return       = 0.0
    s.portfolio_history.clear()
    s.action_history.clear()
    s.loss_history.clear()
    s.reward_history.clear()
    s._price_history.clear()
    s.trade_log.clear()
    s.eff_loss.clear()
    s.eff_sharpe.clear()
    s.eff_return.clear()
    s.eff_win_rate.clear()
    s.eff_steps.clear()
    s.win_rate           = 0.0
    s.rollout_count      = 0
    s.steps_per_sec      = 0.0
    s.sortino_ratio      = 0.0
    s.calmar_ratio       = 0.0
    s.win_loss_ratio     = 0.0
    s.avg_turnover       = 0.0
    s.current_ep_ticker  = ""
    s.is_training        = True

    _training_thread = threading.Thread(target=_run_ppo_training, args=(s,), daemon=True)
    _training_thread.start()
    return {"ok": True, "message": f"Training started on {config.ticker}"}


@app.post("/api/train/stop")
def stop_training():
    agent_state.is_training = False
    return {"ok": True}


@app.post("/api/train/pause")
def pause_training():
    agent_state.is_training = not agent_state.is_training
    return {"ok": True, "is_training": agent_state.is_training}


@app.get("/api/history")
def get_history():
    s = agent_state
    return {
        "portfolio_history": list(s.portfolio_history),
        "action_history":    list(s.action_history),
        "loss_history":      list(s.loss_history),
        "price_history":     list(s._price_history),
    }


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    s = agent_state
    await ws.send_json({
        "type":              "init",
        "is_training":       s.is_training,
        "portfolio_history": list(s.portfolio_history),
        "loss_history":      list(s.loss_history),
        "price_history":     list(s._price_history),
        "trade_log":         list(s.trade_log),
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Walk-forward endpoints ────────────────────────────────────────────

_wf_thread: threading.Thread | None = None
_wf_status: dict = {"running": False, "ticker": "", "progress": 0, "total": 12, "error": ""}


@app.get("/api/walkforward/{ticker}")
def get_walkforward(ticker: str):
    path = f"checkpoints/{ticker}_walkforward.json"
    if os.path.exists(path):
        with open(path) as f:
            return json_lib.load(f)
    return {"ticker": ticker, "windows": [], "avg_test_sharpe": 0,
            "generalization_gap": 0, "trend_slope": 0}


@app.get("/api/walkforward/status/current")
def get_wf_status():
    return _wf_status


class WalkForwardConfig(BaseModel):
    ticker: str = "AAPL"
    steps_per_window: int = 20_000


@app.post("/api/walkforward/run")
def run_walkforward_endpoint(config: WalkForwardConfig):
    global _wf_thread
    if _wf_thread and _wf_thread.is_alive():
        return {"ok": False, "message": "Walk-forward already running"}
    _wf_thread = threading.Thread(
        target=_run_walkforward_bg,
        args=(config.ticker, config.steps_per_window),
        daemon=True,
    )
    _wf_thread.start()
    return {"ok": True, "message": f"Walk-forward started for {config.ticker}"}


def _run_walkforward_bg(ticker: str, steps_per_window: int):
    import traceback
    from walkforward import run
    global _wf_status
    _wf_status = {"running": True, "ticker": ticker, "progress": 0, "total": 12, "error": ""}
    try:
        run(ticker, steps_per_window, status=_wf_status)
    except Exception as e:
        err = traceback.format_exc()
        print(f"[walkforward] ERROR:\n{err}")
        _wf_status["error"] = str(e)
    finally:
        _wf_status["running"] = False


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
