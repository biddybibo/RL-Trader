# RL Trader — Project Roadmap

A realistic progression from single-stock prototype to live portfolio trading agent.
Each phase builds directly on the last. Do not skip phases.

---

## Phase 1 — Single Stock Foundation
**Status: COMPLETE**
**Tickers trained: AAPL, MSFT, GOOGL**

- [x] Custom Gymnasium trading environment
- [x] PPO algorithm with clipped surrogate objective
- [x] LSTM ActorCritic with sequential market memory
- [x] GAE-lambda advantage estimation
- [x] Technical indicators: RSI, MACD, SMA, volatility
- [x] 0.1% transaction cost modeling
- [x] Checkpoint system — agent remembers across sessions
- [x] Real-time React dashboard with WebSocket streaming
- [x] Backtest evaluation: Sharpe ratio, max drawdown, vs buy-and-hold

**What the agent can do right now:**
Learns to time entries and exits on a single stock using price patterns.
Has memory across days. Saves and compounds across training sessions.

**Honest limitation:**
It has memorized 2015–2022 AAPL/MSFT/GOOGL. It has not learned to
generalize. It would likely struggle on a stock it has never seen.

---

## Phase 2 — Generalization & Robustness
**Status: NOT STARTED**
**Estimated training: 5M–10M steps per ticker**

### Goals
- Agent stops memorizing and starts pattern-matching
- Survives market regimes it was not trained on

### What to build
- [ ] Expand training universe to 20+ tickers (add TSLA, NVDA, AMZN, JPM, BAC, XOM, etc.)
- [ ] Train each ticker to 5M+ lifetime steps
- [ ] Add market regime features to observation:
      - VIX (fear index)
      - 10-year Treasury yield
      - SPY daily return (broad market context)
- [ ] Improve reward function:
      - Penalize large drawdowns explicitly
      - Add Sharpe-based reward shaping
- [ ] Add dropout to LSTM layers to reduce overfitting
- [ ] Evaluate on 2023–2024 across all tickers, not just AAPL

### Success criteria
Agent achieves Sharpe > 0.8 on at least 5 out of 10 held-out tickers
it was NOT trained on.

---

## Phase 3 — Multi-Asset Portfolio
**Status: NOT STARTED**
**This is the hardest phase technically**

### Goals
- Agent manages a basket of stocks simultaneously
- Learns correlations between assets
- Allocates capital across positions, not just one stock

### What to build
- [ ] Redesign observation space:
      - Stack N stocks × M features into a 2D observation
      - Add cross-asset correlation matrix as feature
- [ ] Redesign action space:
      - Output N allocation weights that sum to 1.0 (softmax)
      - Support cash as an explicit allocation target
- [ ] Add portfolio-level constraints:
      - Max position size per stock (e.g., 30%)
      - Max sector concentration
      - Minimum cash buffer (e.g., 5%)
- [ ] Upgrade reward to portfolio-level log return
- [ ] Retrain from scratch on multi-asset environment
- [ ] Add attention mechanism so agent weighs which stocks matter most

### Success criteria
Agent manages a 5-stock portfolio with Sharpe > 1.0 on 2023–2024
with max drawdown under 20%.

---

## Phase 4 — Live Paper Trading (Alpaca API)
**Status: NOT STARTED**
**No real money. Real prices. Real conditions.**

### Goals
- Agent trades against live market data, not historical replay
- Proves it works outside of training distribution
- Builds confidence before real capital

### What to build
- [ ] Connect Alpaca API (free tier, paper trading account)
      - Real-time price feed via WebSocket
      - Paper order execution (fake money, real fills)
- [ ] Replace yfinance historical replay with live data stream
- [ ] Add latency handling (market open/close, weekends, holidays)
- [ ] Build live monitoring dashboard:
      - Real P&L vs benchmark
      - Open positions table
      - Daily performance log
- [ ] Add kill switch: auto-pause if drawdown exceeds 10% in a day
- [ ] Run paper trading for minimum 3 months before Phase 5

### Success criteria
3 months of paper trading with:
- Sharpe > 1.0 annualized
- Max daily drawdown < 5%
- Positive total return vs SPY

---

## Phase 5 — Live Trading with Real Capital
**Status: NOT STARTED**
**Only attempt this after Phase 4 success criteria are met**

### Goals
- Deploy agent with real money at small scale
- Strict risk controls at all times

### What to build
- [ ] Start with $1,000–$5,000 maximum capital
- [ ] Hard risk limits enforced in code, not just model output:
      - Daily loss limit: stop trading if down >2% in a day
      - Position limit: never >20% in any single stock
      - Drawdown circuit breaker: pause if down >10% from peak
- [ ] Full audit log of every trade with reasoning
- [ ] Weekly performance review process
- [ ] Gradual capital scaling only if 3-month rolling Sharpe > 1.2

### What this is NOT
This is not "set it and forget it." The agent must be monitored daily.
Markets change. Models drift. Regimes shift. Human oversight is required.

---

## Long-Term Vision (12–24 months out)

| Capability | What it requires |
|---|---|
| Intraday trading (1m/5m bars) | Much larger model, GPU training, co-location |
| Options & derivatives | Entirely different action space and pricing model |
| Macro/news signal integration | NLP pipeline, sentiment model, earnings calendar |
| Fully autonomous portfolio | Proven 12+ month live track record first |

---

## Key Numbers to Hit Before Moving On

| Phase | Milestone | Metric |
|---|---|---|
| Phase 1 → 2 | 10+ tickers trained | 5M steps each |
| Phase 2 → 3 | Generalization proven | Sharpe > 0.8 on unseen tickers |
| Phase 3 → 4 | Portfolio works in backtest | Sharpe > 1.0, DD < 20% |
| Phase 4 → 5 | Paper trading validated | 3 months, Sharpe > 1.0 live |
| Phase 5 scaling | Real capital growth | 12 months, Sharpe > 1.2 |

---

## Current Stats (as of April 2026)

| Ticker | Lifetime Steps | Status |
|--------|---------------|--------|
| AAPL   | 500,000       | Phase 1 complete |
| MSFT   | 500,000       | Phase 1 complete |
| GOOGL  | 500,000       | Phase 1 complete |

**Next action:** Continue training all three to 5M steps,
then begin adding VIX + yield features for Phase 2.
