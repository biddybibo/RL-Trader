import asyncio
import json as json_lib
import os
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
from trainer import TRAIN_BASKET, TRAIN_START, TRAIN_END


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
        self.is_paused               = False
        self.bnh_sharpe              = 0.0
        self.train_stats: dict       = {}
        self._live_price             = 0.0
        self._lock                   = threading.Lock()

agent_state = AgentState()
_training_thread: threading.Thread | None = None


# ── Training (runs in a background thread via the shared Trainer) ─────

_stop_event = threading.Event()
_run_event = threading.Event()   # cleared = paused


def _load_lifetime_steps(run: str = "multi") -> int:
    meta_path = f"checkpoints/{run}_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json_lib.load(f).get("lifetime_steps", 0)
    return 0


def _on_step(tr, step, env, ticker, action, reward):
    s = agent_state
    s.total_steps = step
    s.portfolio_value = round(env.portfolio_value, 2)
    s.cash = round(env.cash, 2)
    s.shares_held = env.shares
    s.current_position = round(float(env.position), 3)
    s.current_ep_ticker = ticker
    s._live_price = float(env.close[env.t])
    s.reward_history.append(round(float(reward), 5))
    if len(env.allocations) >= 2:
        delta = env.allocations[-1] - env.allocations[-2]
        if abs(delta) >= 0.25:
            s.trade_log.appendleft({
                "time": str(env.dates[env.t - 1].date()), "ticker": ticker,
                "action": "BUY" if delta > 0 else "SELL",
                "price": round(float(env.close[env.t - 1]), 2),
                "alloc": round(env.allocations[-1], 2),
                "pnl": round(float(env.values[-1] - env.values[-2]), 2),
            })


def _on_episode(tr, summary):
    s = agent_state
    s.episode += 1
    env = tr.envs[summary["ticker"]]
    # Chart = the last completed episode, agent vs buy-and-hold on the same days
    s.portfolio_history.clear(); s.portfolio_history.extend(round(v, 2) for v in env.values[-300:])
    s._price_history.clear(); s._price_history.extend(round(float(p), 2) for p in env.episode_prices()[-300:])
    s.action_history.clear(); s.action_history.extend(round(a, 3) for a in env.allocations[-300:])
    recent = tr.episodes[-20:]
    mean = lambda k: float(np.mean([e[k] for e in recent]))
    s.sharpe_ratio = round(mean("sharpe"), 2)
    s.sortino_ratio = round(mean("sortino"), 2)
    s.calmar_ratio = round(mean("calmar"), 2)
    s.max_drawdown = round(mean("max_drawdown") * 100, 2)
    s.total_return = round(mean("total_return") * 100, 2)
    s.avg_turnover = round(mean("turnover"), 2)
    s.win_rate = round(float(np.mean([e["sharpe"] > e["bnh_sharpe"] for e in recent])) * 100, 1)  # % episodes beating B&H
    s.bnh_sharpe = round(mean("bnh_sharpe"), 2)


def _on_rollout(tr, step, stats):
    s = agent_state
    s.current_loss = round(stats["loss"], 4)
    s.loss_history.append(s.current_loss)
    s.train_stats = stats
    s.rollout_count += 1
    rw = np.array(list(s.reward_history))
    wins, losses = rw[rw > 0], rw[rw < 0]
    s.win_loss_ratio = round(float(wins.mean()) / (float(np.abs(losses).mean()) + 1e-9), 2) if len(wins) and len(losses) else 0.0
    s.eff_loss.append(s.current_loss)
    s.eff_sharpe.append(s.sharpe_ratio)
    s.eff_return.append(s.total_return)
    s.eff_win_rate.append(s.win_rate)
    s.eff_steps.append(step)


def _run_ppo_training(s: AgentState):
    import torch
    from trainer import Hooks, TrainConfig, Trainer, load_basket

    torch.set_num_threads(1)  # avoid contention with FastAPI's thread pool
    try:
        data = load_basket(TRAIN_BASKET, TRAIN_START, TRAIN_END)
        trainer = Trainer(data, TrainConfig(total_steps=s.total_steps_target), "checkpoints", "multi",
                          hooks=Hooks(on_step=_on_step, on_episode=_on_episode, on_rollout=_on_rollout))
        out = trainer.run(stop_event=_stop_event, run_event=_run_event)
        print(f"[train] done: {out['steps']:,} steps, lifetime {out['lifetime_steps']:,}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[train] failed: {e}")
    finally:
        s.is_training = False
        s.is_paused = False


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
                "price":              round(s._live_price, 2),
                "reward":             s.reward_history[-1] if s.reward_history else 0.0,
                "trade_log":          list(s.trade_log)[:10],
                "is_training":        s.is_training,
                "win_rate":           s.win_rate,
                "rollout_count":      s.rollout_count,
                "steps_per_sec":      s.steps_per_sec,
                "current_ep_ticker":  s.current_ep_ticker,
                "is_paused":          s.is_paused,
                "bnh_sharpe":         s.bnh_sharpe,
                "train_stats":        s.train_stats,
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
        "lifetime_steps":     _load_lifetime_steps(),
        "is_paused":          s.is_paused,
        "bnh_sharpe":         s.bnh_sharpe,
        "train_stats":        s.train_stats,
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
    if s.is_training or (_training_thread and _training_thread.is_alive()):
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
    s.is_paused          = False
    _stop_event.clear()
    _run_event.set()

    _training_thread = threading.Thread(target=_run_ppo_training, args=(s,), daemon=True)
    _training_thread.start()
    return {"ok": True, "message": f"Training started on {config.ticker}"}


@app.post("/api/train/stop")
def stop_training():
    _stop_event.set()
    _run_event.set()  # unblock if paused so the thread can save and exit
    return {"ok": True}


@app.post("/api/train/pause")
def pause_training():
    s = agent_state
    if not s.is_training:
        return {"ok": False, "message": "Not training"}
    if _run_event.is_set():
        _run_event.clear(); s.is_paused = True
    else:
        _run_event.set(); s.is_paused = False
    return {"ok": True, "is_training": s.is_training, "is_paused": s.is_paused}


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
