import asyncio
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

def _run_ppo_training(s: AgentState):
    from data.fetch import fetch
    from env.trading_env import TradingEnv
    from agent.ppo import PPOAgent

    print(f"[train] Fetching {s.ticker} data...")
    df = fetch(s.ticker, "2015-01-01", "2022-12-31")
    print(f"[train] {len(df)} trading days loaded")

    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, rollout_steps=2048)

    os.makedirs("checkpoints", exist_ok=True)

    # Resume from latest checkpoint if one exists
    ckpt_path = f"checkpoints/{s.ticker}_latest.pt"
    meta_path = f"checkpoints/{s.ticker}_meta.json"
    lifetime_steps = 0
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            lifetime_steps = meta.get("lifetime_steps", 0)
        print(f"[train] Resumed {s.ticker} from {ckpt_path} ({lifetime_steps:,} lifetime steps)")
    else:
        print(f"[train] No checkpoint found, starting fresh for {s.ticker}")

    obs, _ = env.reset(seed=42)
    ep_reward = 0.0
    next_ckpt = 50_000

    for step in range(1, s.total_steps_target + 1):
        if not s.is_training:
            break

        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, float(done), value, log_prob)
        ep_reward += reward
        obs = next_obs

        alloc = float(np.clip(action[0], 0.0, 1.0))

        # Update shared state (GIL protects simple assignments)
        s.total_steps     = step
        s.portfolio_value = env.portfolio_value
        s.cash            = env.cash
        s.shares_held     = env.shares
        s.current_position = alloc
        s.portfolio_history.append(round(env.portfolio_value, 2))
        s.action_history.append(round(alloc, 3))
        s.reward_history.append(round(float(reward), 5))

        price_idx = min(env.t, len(df) - 1)
        s._price_history.append(round(float(df.iloc[price_idx]["close"]), 2))

        if done:
            s.episode += 1
            ep_reward  = 0.0
            obs, _     = env.reset()

        # PPO update every rollout
        if step % agent.rollout_steps == 0:
            _, _, last_value = agent.select_action(obs)
            agent.update(last_value)
            s.current_loss = round(agent.last_loss, 4)
            s.loss_history.append(s.current_loss)

            pv_arr = np.array(list(s.portfolio_history))
            if len(pv_arr) > 10:
                rets = np.diff(pv_arr) / (pv_arr[:-1] + 1e-8)
                s.sharpe_ratio = round(float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)), 2)
                peak = np.maximum.accumulate(pv_arr)
                s.max_drawdown = round(float(((pv_arr - peak) / (peak + 1e-8)).min() * 100), 2)
                s.total_return = round(float((pv_arr[-1] / pv_arr[0] - 1) * 100), 2)

            action_type = "BUY" if alloc > 0.6 else ("SELL" if alloc < 0.2 else "HOLD")
            s.trade_log.appendleft({
                "time":   time.strftime("%H:%M:%S"),
                "action": action_type,
                "price":  round(float(df.iloc[price_idx]["close"]), 2),
                "alloc":  round(alloc, 2),
                "pnl":    round(float(reward * 10_000), 2),
            })

        if step >= next_ckpt:
            agent.save(f"checkpoints/{s.ticker}_{lifetime_steps + step}.pt")
            agent.save(ckpt_path)  # always overwrite latest
            _save_meta(meta_path, lifetime_steps + step, s.ticker)
            print(f"[train] checkpoint -> {s.ticker} lifetime {lifetime_steps + step:,} steps")
            next_ckpt += 50_000

    agent.save(ckpt_path)
    _save_meta(meta_path, lifetime_steps + s.total_steps_target, s.ticker)
    print(f"[train] done -> {s.ticker} lifetime {lifetime_steps + s.total_steps_target:,} steps")
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

async def broadcast_loop():
    while True:
        if agent_state.is_training and manager.active:
            s = agent_state
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
                "max_drawdown":       s.max_drawdown,
                "total_return":       s.total_return,
                "price":              s._price_history[-1] if s._price_history else 182.0,
                "reward":             s.reward_history[-1] if s.reward_history else 0.0,
                "trade_log":          list(s.trade_log)[:10],
                "is_training":        s.is_training,
            })
        elif not agent_state.is_training and manager.active:
            await manager.broadcast({"type": "training_complete", "steps": agent_state.total_steps})

        await asyncio.sleep(0.25)


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
