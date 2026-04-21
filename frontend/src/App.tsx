import { useEffect, useRef, useState, useCallback } from "react";
import { useTraderWS } from "./hooks/useTraderWS";
import type { TradeEntry } from "./hooks/useTraderWS";
import { getStatus, startTraining, stopTraining, pauseTraining } from "./lib/api";
import { Sparkline } from "./components/Sparkline";
import "./index.css";

// ── Types ─────────────────────────────────────────────────────────────
interface AppState {
  isTraining: boolean;
  totalSteps: number;
  totalStepsTarget: number;
  episode: number;
  loss: number;
  portfolioValue: number;
  cash: number;
  sharesHeld: number;
  position: number;
  sharpe: number;
  maxDrawdown: number;
  totalReturn: number;
  price: number;
  reward: number;
  portfolioHistory: number[];
  lossHistory: number[];
  priceHistory: number[];
  tradeLog: TradeEntry[];
  lifetimeSteps: number;
}

const DEFAULT_STATE: AppState = {
  isTraining: false, totalSteps: 0, totalStepsTarget: 500000,
  episode: 0, loss: 0, portfolioValue: 10000, cash: 10000,
  sharesHeld: 0, position: 0, sharpe: 0, maxDrawdown: 0,
  totalReturn: 0, price: 182, reward: 0,
  portfolioHistory: [], lossHistory: [], priceHistory: [], tradeLog: [],
  lifetimeSteps: 0,
};

// ── Helpers ───────────────────────────────────────────────────────────
const fmt$ = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 }).format(n);

const fmtPct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;

function PositionMeter({ value }: { value: number }) {
  const pct = ((value + 1) / 2) * 100;
  const color = value > 0.15 ? "#00ff9d" : value < -0.15 ? "#ff4d6d" : "#f0b429";
  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginBottom: 3 }}>
        <span>SELL</span><span>HOLD</span><span>BUY</span>
      </div>
      <div style={{ position: "relative", height: 6, background: "#1a1a1a", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0,
          width: `${pct}%`, background: color,
          borderRadius: 3, transition: "width 0.3s ease, background 0.3s ease",
        }} />
        <div style={{
          position: "absolute", top: "50%", left: "50%",
          transform: "translate(-50%,-50%)", width: 1, height: 10, background: "#333",
        }} />
      </div>
      <div style={{ textAlign: "center", fontSize: 11, color, marginTop: 3, fontFamily: "monospace" }}>
        {value >= 0 ? "+" : ""}{(value * 100).toFixed(1)}%
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState<AppState>(DEFAULT_STATE);
  const [ticker, setTicker] = useState("AAPL");
  const [stepsInput, setStepsInput] = useState("500000");
  const [flash, setFlash] = useState<"buy" | "sell" | null>(null);
  const prevPos = useRef(0);
  const { connected, lastTick } = useTraderWS("ws://localhost:8000/ws");

  // Load initial state
  useEffect(() => {
    getStatus().then((d) => {
      setState((s) => ({
        ...s,
        isTraining:       d.is_training,
        totalSteps:       d.total_steps,
        totalStepsTarget: d.total_steps_target,
        episode:          d.episode,
        portfolioValue:   d.portfolio_value,
        sharpe:           d.sharpe,
        maxDrawdown:      d.max_drawdown,
        totalReturn:      d.total_return,
        portfolioHistory: d.portfolio_history || [],
        lossHistory:      d.loss_history || [],
        priceHistory:     d.price_history || [],
        tradeLog:         d.trade_log || [],
        lifetimeSteps:    d.lifetime_steps || 0,
      }));
    }).catch(() => {});
  }, []);

  // Handle WS ticks
  useEffect(() => {
    if (!lastTick) return;
    const t = lastTick;

    if (t.type === "init") {
      setState((s) => ({
        ...s,
        isTraining:       t.is_training ?? s.isTraining,
        portfolioHistory: t.portfolio_history ?? s.portfolioHistory,
        lossHistory:      t.loss_history ?? s.lossHistory,
        priceHistory:     t.price_history ?? s.priceHistory,
        tradeLog:         t.trade_log ?? s.tradeLog,
      }));
      return;
    }

    if (t.type === "training_complete") {
      setState((s) => ({ ...s, isTraining: false }));
      return;
    }

    if (t.type === "tick") {
      const pos = t.position ?? 0;
      if (Math.abs(pos - prevPos.current) > 0.3) {
        setFlash(pos > prevPos.current ? "buy" : "sell");
        setTimeout(() => setFlash(null), 600);
      }
      prevPos.current = pos;

      setState((s) => ({
        ...s,
        isTraining:       t.is_training !== undefined ? t.is_training : s.isTraining,
        totalSteps:       t.total_steps ?? s.totalSteps,
        totalStepsTarget: t.total_steps_target ?? s.totalStepsTarget,
        episode:          t.episode ?? s.episode,
        loss:             t.loss ?? s.loss,
        portfolioValue:   t.portfolio_value ?? s.portfolioValue,
        cash:             t.cash ?? s.cash,
        sharesHeld:       t.shares_held ?? s.sharesHeld,
        position:         pos,
        sharpe:           t.sharpe ?? s.sharpe,
        maxDrawdown:      t.max_drawdown ?? s.maxDrawdown,
        totalReturn:      t.total_return ?? s.totalReturn,
        price:            t.price ?? s.price,
        reward:           t.reward ?? s.reward,
        tradeLog:         t.trade_log ?? s.tradeLog,
        portfolioHistory: s.portfolioHistory.length === 0 && t.portfolio_value
          ? [t.portfolio_value]
          : [...s.portfolioHistory.slice(-299), t.portfolio_value ?? s.portfolioValue],
        lossHistory: t.loss
          ? [...s.lossHistory.slice(-199), t.loss]
          : s.lossHistory,
        priceHistory: t.price
          ? [...s.priceHistory.slice(-299), t.price]
          : s.priceHistory,
      }));
    }
  }, [lastTick]);

  const handleStart = useCallback(async () => {
    const steps = parseInt(stepsInput) || 500000;
    await startTraining(ticker, steps);
    setState((s) => ({ ...s, isTraining: true }));
  }, [ticker, stepsInput]);

  const handleStop = useCallback(async () => {
    await stopTraining();
    setState((s) => ({ ...s, isTraining: false }));
  }, []);

  const handlePause = useCallback(async () => {
    const res = await pauseTraining();
    setState((s) => ({ ...s, isTraining: res.is_training }));
  }, []);

  const progress = Math.min((state.totalSteps / state.totalStepsTarget) * 100, 100);
  const returnPositive = state.totalReturn >= 0;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon" />
            <span className="logo-text">RL<span className="logo-accent">trader</span></span>
          </div>
          <div className={`ws-badge ${connected ? "ws-connected" : "ws-disconnected"}`}>
            <span className="ws-dot" />
            {connected ? "live" : "reconnecting"}
          </div>
        </div>
        <div className="header-controls">
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="select-input"
            disabled={state.isTraining}
          >
            {["AAPL","MSFT","GOOGL","TSLA","SPY","NVDA","AMZN"].map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <input
            type="number"
            value={stepsInput}
            onChange={(e) => setStepsInput(e.target.value)}
            className="steps-input"
            disabled={state.isTraining}
            placeholder="Steps"
          />
          {!state.isTraining ? (
            <button className="btn btn-start" onClick={handleStart}>
              <span className="btn-icon">▶</span> Start Training
            </button>
          ) : (
            <>
              <button className="btn btn-pause" onClick={handlePause}>⏸ Pause</button>
              <button className="btn btn-stop" onClick={handleStop}>■ Stop</button>
            </>
          )}
        </div>
      </header>

      {/* ── Progress bar ── */}
      <div className="progress-bar-wrap">
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="progress-labels">
          <span>{state.totalSteps.toLocaleString()} steps</span>
          <span>{progress.toFixed(1)}%</span>
          <span>ep {state.episode}</span>
        </div>
      </div>

      {/* ── Main grid ── */}
      <main className="grid">

        {/* ── Portfolio card ── */}
        <div className={`card card-portfolio ${flash === "buy" ? "flash-buy" : flash === "sell" ? "flash-sell" : ""}`}>
          <div className="card-label">Portfolio Value</div>
          <div className="card-value pv-main">{fmt$(state.portfolioValue)}</div>
          <div className={`card-sub ${returnPositive ? "pos" : "neg"}`}>
            {fmtPct(state.totalReturn)} total return
          </div>
          <div style={{ marginTop: 12 }}>
            <Sparkline
              data={state.portfolioHistory}
              width={320} height={60}
              color={returnPositive ? "#00ff9d" : "#ff4d6d"}
              fillColor={returnPositive ? "rgba(0,255,157,0.07)" : "rgba(255,77,109,0.07)"}
            />
          </div>
        </div>

        {/* ── Price card ── */}
        <div className="card card-price">
          <div className="card-label">{ticker} Price</div>
          <div className="card-value">${state.price.toFixed(2)}</div>
          <div style={{ marginTop: 12 }}>
            <Sparkline
              data={state.priceHistory}
              width={220} height={50}
              color="#f0b429"
              fillColor="rgba(240,180,41,0.07)"
            />
          </div>
        </div>

        {/* ── Metrics ── */}
        <div className="card card-metrics">
          <div className="card-label">Key Metrics</div>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Sharpe</span>
              <span className={`metric-val ${state.sharpe >= 1 ? "pos" : state.sharpe < 0 ? "neg" : "neutral"}`}>
                {state.sharpe.toFixed(2)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Max DD</span>
              <span className="metric-val neg">{state.maxDrawdown.toFixed(1)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Cash</span>
              <span className="metric-val">{fmt$(state.cash)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Shares</span>
              <span className="metric-val">{state.sharesHeld.toFixed(3)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Loss</span>
              <span className="metric-val neutral">{state.loss.toFixed(4)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Reward</span>
              <span className={`metric-val ${state.reward >= 0 ? "pos" : "neg"}`}>
                {state.reward.toFixed(5)}
              </span>
            </div>
          </div>
        </div>

        {/* ── Agent position ── */}
        <div className="card card-position">
          <div className="card-label">Agent Position</div>
          <PositionMeter value={state.position} />
          <div style={{ marginTop: 14 }}>
            <Sparkline
              data={state.portfolioHistory.map((_, i) =>
                i < (state as any).actionHistory?.length
                  ? (state as any).actionHistory[i]
                  : state.position
              )}
              width={220} height={40}
              color="#a78bfa"
              fillColor="rgba(167,139,250,0.08)"
            />
          </div>
        </div>

        {/* ── Loss chart ── */}
        <div className="card card-loss">
          <div className="card-label">Training Loss</div>
          {state.lossHistory.length > 1 ? (
            <Sparkline
              data={state.lossHistory}
              width={320} height={70}
              color="#60a5fa"
              fillColor="rgba(96,165,250,0.07)"
            />
          ) : (
            <div className="empty-chart">waiting for updates…</div>
          )}
        </div>

        {/* ── Trade log ── */}
        <div className="card card-trades">
          <div className="card-label">Trade Log</div>
          <div className="trade-list">
            {state.tradeLog.length === 0 && (
              <div className="empty-chart">no trades yet</div>
            )}
            {state.tradeLog.map((t, i) => (
              <div key={i} className={`trade-row trade-${t.action.toLowerCase()}`}>
                <span className="trade-time">{t.time}</span>
                <span className={`trade-action trade-badge-${t.action.toLowerCase()}`}>{t.action}</span>
                <span className="trade-price">${t.price.toFixed(2)}</span>
                <span className="trade-alloc">{(t.alloc * 100).toFixed(0)}%</span>
                <span className={`trade-pnl ${t.pnl >= 0 ? "pos" : "neg"}`}>
                  {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

      </main>

      {/* ── Status bar ── */}
      <footer className="statusbar">
        <span>PPO · ActorCritic · GAE-λ</span>
        <span style={{ color: "var(--purple)", fontFamily: "var(--mono)", fontSize: 10 }}>
          lifetime: {state.lifetimeSteps.toLocaleString()} steps
        </span>
        <span className={state.isTraining ? "status-training" : "status-idle"}>
          {state.isTraining ? "● training" : "○ idle"}
        </span>
        <span>clip ε=0.2 · γ=0.99 · λ=0.95</span>
      </footer>
    </div>
  );
}
