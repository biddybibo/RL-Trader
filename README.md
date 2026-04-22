# RL Trader

A reinforcement learning stock trading agent trained with **Proximal Policy Optimization (PPO)** and an **LSTM ActorCritic** network. The agent trains across a basket of 7 tickers simultaneously, observes 17 market + macro features, and uses a Sortino-shaped reward to optimize risk-adjusted returns — with a real-time React dashboard to watch it learn.

---

## What It Does

The agent starts with $10,000 and decides every trading day what percentage of its portfolio to hold in a given stock (0% = all cash, 100% = fully invested). It learns purely from experience across multiple tickers, so it learns generalizable market structure rather than ticker-specific patterns.

Unlike a simple rule-based bot, this agent:
- **Remembers** market context across days via an LSTM hidden state
- **Trains on 7 tickers simultaneously** — learns market structure, not ticker quirks
- **Optimizes risk-adjusted returns** — Sortino-shaped reward penalizes downside volatility and drawdowns
- **Sees macro context** — VIX and 10Y yield tell the agent what kind of market it's in
- **Compounds intelligence** — each session resumes from the last checkpoint

---

## Architecture

```
Observation (17 floats per timestep)
  Portfolio state (3):
    ├── position          current stock allocation [0, 1]
    ├── cash_ratio        cash / initial cash
    └── unrealized_pnl   (portfolio - initial) / initial

  Price / momentum (6):
    ├── log_return        previous day log return
    ├── sma_10            (SMA10 - price) / price
    ├── sma_30            (SMA30 - price) / price
    ├── volatility        20-day rolling std of log returns
    ├── rsi               RSI / 100
    └── macd              (EMA12 - EMA26) / price

  Volume (2):
    ├── volume_ratio      today volume / 20-day avg volume
    └── atr_norm          ATR(14) / price

  Range / calendar (4):
    ├── hw_proximity      (price - 52w_low) / (52w_high - 52w_low)
    ├── day_sin           sin(2π × day_of_week / 5)
    ├── day_cos           cos(2π × day_of_week / 5)
    └── month_sin         sin(2π × (month-1) / 12)

  Macro (2):
    ├── vix_norm          VIX / 30  (fear regime)
    └── rate_norm         10Y yield / 10  (rate environment)

LSTM ActorCritic Network
    ├── Encoder:     Linear(17 → 128) + Tanh
    ├── LSTM:        128 → 128  (carries memory across timesteps)
    ├── Actor head:  Linear(128 → 1) + Tanh  →  target allocation [-1, 1]
    └── Critic head: Linear(128 → 1)          →  state value estimate

PPO Training
    ├── Clipped surrogate objective  (ε = 0.2)
    ├── GAE-lambda advantage estimation  (γ = 0.99, λ = 0.95)
    ├── Entropy bonus  (coef = 0.01)
    ├── Value function loss  (coef = 0.5)
    ├── Truncated BPTT through sequences of length 32
    └── Adam optimizer  (lr = 3e-4)

Reward (Sortino-shaped)
    ├── Base:             log portfolio return
    ├── Drawdown penalty: −0.05 × |drawdown from peak|
    └── Downside penalty: −0.20 × |loss|  (losses penalized 20% more than gains)

Transaction costs
    ├── Base cost:        0.1% per unit |Δposition|
    └── Market impact:    scales with illiquidity (low volume = higher cost)
```

---

## Project Structure

```
rl-trader/
├── agent/
│   └── ppo.py                LSTM ActorCritic, RolloutBuffer, PPOAgent
├── data/
│   └── fetch.py              yfinance pipeline — OHLCV + volume + macro features
├── env/
│   └── trading_env.py        Custom Gymnasium environment (17-dim obs, Sortino reward)
├── frontend/
│   └── src/
│       ├── App.tsx            Main dashboard layout (3-zone Framer Motion)
│       ├── components/
│       │   ├── StatCard.tsx        Spring-animated stat cell
│       │   ├── DualLineChart.tsx   Portfolio vs buy-and-hold canvas chart
│       │   ├── SharpeTrend.tsx     Loss + Sharpe per-rollout canvas charts
│       │   ├── Sparkline.tsx       Generic canvas sparkline
│       │   └── WalkForwardChart.tsx Walk-forward bar chart
│       ├── hooks/
│       │   └── useTraderWS.ts  WebSocket hook with auto-reconnect
│       └── lib/
│           └── api.ts          REST API calls
├── main.py                   FastAPI backend + WebSocket broadcast
├── walkforward.py            Walk-forward generalization analysis
├── train.py                  CLI training loop (headless)
├── evaluate.py               Backtest on held-out data with charts
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

Click **Start Training** in the dashboard. The agent will preload all 7 tickers, then train across them simultaneously. Watch it learn in real time.

---

## Multi-Asset Training

The agent trains on a fixed basket of 7 tickers, randomly switching to a new ticker at the start of each episode:

```
AAPL · MSFT · GOOGL · TSLA · SPY · NVDA · AMZN
Training data: 2010–2022
```

This prevents overfitting to any single ticker's patterns. The dashboard shows which ticker the current episode is training on (↻ ticker indicator in the Risk Metrics card).

---

## Training Modes

### Dashboard (visual)

Start `main.py` and the frontend, then use the UI.

```
Start Training  →  preloads all tickers, runs PPO across basket
Pause           →  freezes training (resume with Pause again)
Stop            →  saves checkpoint immediately and halts
```

### CLI (headless, faster)

```bash
python train.py --steps 500000 --ckpt-dir checkpoints
```

---

## Checkpoints

```
checkpoints/
  multi_latest.pt     ← most recent multi-asset weights
  multi_meta.json     ← lifetime step count + timestamp
  multi_50000.pt      ← milestone snapshots every 50k steps
  AAPL_walkforward.json  ← walk-forward analysis results
```

---

## Walk-Forward Analysis

Measures whether the agent genuinely generalizes or just memorizes training data.

Uses **expanding windows**: train on 2015 → growing end date, test on the immediately following 6-month period. If test Sharpe improves as training data grows → the agent is learning real patterns.

```bash
python walkforward.py --ticker AAPL --steps 20000
```

Results are saved to `checkpoints/AAPL_walkforward.json` and visualized in the dashboard.

| Output | Meaning |
|---|---|
| **Avg Test Sharpe** | Mean Sharpe across all 12 test windows |
| **Generalization Gap** | Train Sharpe − Test Sharpe (lower = less overfitting) |
| **Trend** | Is test Sharpe improving as training data grows? |

---

## Dashboard Layout

Built with **Framer Motion** for smooth spring-animated number transitions and panel reveals.

### Zone 1 — Stats Rail
Eight animated live metrics across the top:

| Stat | Description |
|---|---|
| **Steps** | Training steps completed / total target |
| **Episode** | Current episode number + active ticker |
| **Portfolio** | Current value (spring-animated) + total return % |
| **Sharpe** | Risk-adjusted return (green ≥ 1.0) |
| **Sortino** | Downside-only risk ratio (green ≥ 1.5) |
| **Max DD** | Peak-to-trough drawdown |
| **Win Rate** | Fraction of profitable trades + rollout count |
| **Steps/sec** | Training throughput + lifetime steps |

### Zone 2 — Charts (side by side)
- **Left (3fr)**: Portfolio vs Buy-and-Hold — canvas dual-line chart, both series normalised to the same start value so relative performance is immediately visible
- **Right (2fr)**: Training Trend — stacked Loss/rollout + Sharpe/rollout charts, updated every PPO rollout

### Position Bar
Slim animated bar between charts and drawer showing current agent allocation (SELL ↔ HOLD ↔ BUY) with spring physics.

### Zone 3 — Collapsible Drawer
Click to expand; toggle between two tabs:
- **Walk-Forward** — bar chart + Avg Test Sharpe / Gen Gap / Trend summary + Run Analysis button
- **Trade Log** — BUY/SELL/HOLD decisions with price, allocation, PnL (newest first)

### Risk Metrics explained

| Metric | Good value | What it measures |
|---|---|---|
| **Sharpe** | > 1.0 | Return per unit of total volatility |
| **Sortino** | > 1.5 | Return per unit of *downside* volatility |
| **Calmar** | > 0.5 | Return / max drawdown (capital efficiency) |
| **Max DD** | < −20% | Worst peak-to-trough loss |
| **W/L Ratio** | > 1.5 | Avg win size / avg loss size |
| **Turnover** | < 5x | Avg position changes per episode (lower = cheaper) |

---

## Evaluation (Backtest)

```bash
python evaluate.py --model checkpoints/multi_latest.pt
```

Runs the agent on 2023–2024 held-out data and compares against buy-and-hold.

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
