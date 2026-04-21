# RL Trader

A reinforcement learning stock trading agent trained with **Proximal Policy Optimization (PPO)** and an **LSTM ActorCritic** network. The agent learns to trade a single stock by observing technical indicators and managing a simulated portfolio — with a real-time React dashboard to watch it learn.

---

## What It Does

The agent starts with $10,000 and decides every trading day what percentage of its portfolio to hold in a given stock (0% = all cash, 100% = fully invested). It learns purely from experience: getting rewarded for portfolio gains and penalized for losses and transaction costs.

Unlike a simple rule-based bot, this agent:
- **Remembers** market context across days via an LSTM hidden state
- **Learns market regimes** — it can distinguish trending vs. volatile periods
- **Compounds intelligence** — each training session resumes from the last checkpoint, so the agent only ever gets smarter

---

## Architecture

```
Observation (9 floats per timestep)
    ├── position          current stock allocation [0, 1]
    ├── cash_ratio        cash / initial cash
    ├── unrealized_pnl    (portfolio - initial) / initial
    ├── log_return        previous day log return
    ├── sma_10            (SMA10 - price) / price
    ├── sma_30            (SMA30 - price) / price
    ├── volatility        20-day rolling std of log returns
    ├── rsi               RSI / 100, scaled to [0, 1]
    └── macd              (EMA12 - EMA26) / price

LSTM ActorCritic Network
    ├── Encoder:    Linear(9 → 128) + Tanh
    ├── LSTM:       128 → 128  (carries memory across timesteps)
    ├── Actor head: Linear(128 → 1) + Tanh  →  target allocation [-1, 1]
    └── Critic head: Linear(128 → 1)         →  state value estimate

PPO Training
    ├── Clipped surrogate objective  (ε = 0.2)
    ├── GAE-lambda advantage estimation  (γ = 0.99, λ = 0.95)
    ├── Entropy bonus  (coef = 0.01)
    ├── Value function loss  (coef = 0.5)
    ├── Truncated BPTT through sequences of length 32
    └── Adam optimizer  (lr = 3e-4)
```

---

## Project Structure

```
rl-trader/
├── agent/
│   └── ppo.py              LSTM ActorCritic, RolloutBuffer, PPOAgent
├── data/
│   └── fetch.py            yfinance pipeline + RSI, MACD, SMA features
├── env/
│   └── trading_env.py      Custom Gymnasium environment
├── frontend/               React + Vite dashboard
│   └── src/
│       ├── App.tsx          Main dashboard layout
│       ├── components/
│       │   └── Sparkline.tsx  Canvas sparkline charts
│       ├── hooks/
│       │   └── useTraderWS.ts  WebSocket hook with auto-reconnect
│       └── lib/
│           └── api.ts       REST API calls
├── main.py                 FastAPI backend + WebSocket broadcast
├── train.py                CLI training loop (headless)
├── evaluate.py             Backtest on held-out data with charts
└── requirements.txt
```

---

## Quickstart

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Start the backend

```bash
python main.py
```

Runs at `http://localhost:8000`

### 4. Start the frontend (new terminal)

```bash
cd frontend
npx vite
```

Open `http://localhost:5173` in your browser.

### 5. Train

Click **Start Training** in the dashboard. Pick a ticker and step count, then watch the agent learn in real time.

---

## Training Modes

### Dashboard (visual)

Start `main.py` and the frontend, then use the UI. Best for monitoring and exploration.

```
Start Training  →  runs real PPO in a background thread, streams to browser
Pause           →  freezes training (resume with Pause again)
Stop            →  saves checkpoint immediately and halts
```

### CLI (headless, faster)

```bash
python train.py --steps 500000 --ckpt-dir checkpoints
```

Options:
| Flag | Default | Description |
|---|---|---|
| `--steps` | 500,000 | Total environment steps |
| `--ckpt-dir` | `checkpoints/` | Where to save `.pt` files |
| `--rollout-steps` | 2,048 | Steps per PPO update |
| `--lr` | 3e-4 | Learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--seed` | 42 | Random seed |

---

## Checkpoints & Persistent Memory

Every training session automatically saves and resumes. Nothing is ever lost.

```
checkpoints/
  AAPL_latest.pt      ← always the most recent weights for AAPL
  AAPL_meta.json      ← lifetime step count + timestamp
  AAPL_50000.pt       ← milestone snapshots every 50k steps
  AAPL_100000.pt
  MSFT_latest.pt      ← separate brain per ticker
  MSFT_meta.json
  ...
```

- Each ticker maintains its **own independent model**
- Hitting **Start Training** loads `{TICKER}_latest.pt` automatically
- The footer shows **lifetime steps** — cumulative across all sessions
- If you switch tickers, the previous ticker's progress is untouched

---

## Evaluation (Backtest)

Run the agent on 2023–2024 held-out AAPL data and compare against buy-and-hold:

```bash
python evaluate.py --model checkpoints/AAPL_latest.pt
```

Outputs:
- Terminal table: Total Return, Sharpe Ratio, Max Drawdown, Final Value
- `backtest.png`: Portfolio value chart vs. buy-and-hold + allocation subplot

Example output after full training:

```
=== Backtest Metrics (2023-2024) ===
Metric                    PPO Agent   Buy-and-Hold
--------------------------------------------------
Total Return (%)              ...          66.10
Sharpe Ratio                  ...          1.289
Max Drawdown (%)              ...         -16.61
Final Value ($)               ...       16609.89
```

---

## Dashboard Panels

| Panel | What it shows |
|---|---|
| **Portfolio Value** | Agent's simulated account value from $10,000 start |
| **AAPL Price** | Real closing prices from training data |
| **Key Metrics** | Sharpe, Max Drawdown, Cash, Shares, Loss, Reward |
| **Agent Position** | Current allocation bar: SELL ← HOLD → BUY |
| **Training Loss** | PPO loss curve over time (should trend down) |
| **Trade Log** | BUY/SELL/HOLD decisions with price, allocation, PnL |
| **Progress bar** | Steps completed / target + episode count |
| **Lifetime steps** | Total steps trained across all sessions (footer) |

---

## How the Agent Gets Smarter Over Time

The LSTM gives the agent **memory**. Each step, instead of just seeing 9 numbers, it sees 9 numbers *plus a hidden memory vector* that encodes everything it has observed so far in the episode. This allows it to learn:

- Multi-day trends ("price has been rising for 3 days")
- Volatility regimes ("we're in a high-risk period, reduce exposure")
- Recovery patterns ("we just bounced off a low, this might continue")

The more sessions you run, the more patterns the agent has been exposed to. The checkpoint system ensures every session builds on the last — the agent never forgets what it learned.

---

## Supported Tickers

AAPL · MSFT · GOOGL · TSLA · SPY · NVDA · AMZN

Each ticker trains on **2015–2022** data and is evaluated on **2023–2024**.

---

## Requirements

```
Python >= 3.10
Node >= 18

gymnasium>=0.29.0
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.36
matplotlib>=3.7.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
```

---

## Roadmap

- [ ] Multi-stock observations (cross-asset correlations)
- [ ] Automatic rolling data window (train on latest N years)
- [ ] Curriculum learning (easy markets → volatile markets)
- [ ] GPU training support
- [ ] Export trained agent as ONNX for deployment
