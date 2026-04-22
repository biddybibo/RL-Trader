import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTraderWS } from "./hooks/useTraderWS";
import type { TradeEntry } from "./hooks/useTraderWS";
import { getStatus, startTraining, stopTraining, pauseTraining, getWalkForward, runWalkForward } from "./lib/api";
import { StatCard } from "./components/StatCard";
import { DualLineChart } from "./components/DualLineChart";
import { SharpeTrend } from "./components/SharpeTrend";
import { WalkForwardChart } from "./components/WalkForwardChart";
import type { WFWindow } from "./components/WalkForwardChart";
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
  position: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  maxDrawdown: number;
  totalReturn: number;
  winLossRatio: number;
  avgTurnover: number;
  price: number;
  reward: number;
  portfolioHistory: number[];
  lossHistory: number[];
  priceHistory: number[];
  tradeLog: TradeEntry[];
  lifetimeSteps: number;
  currentEpTicker: string;
}

const DEFAULT_STATE: AppState = {
  isTraining: false, totalSteps: 0, totalStepsTarget: 500000,
  episode: 0, loss: 0, portfolioValue: 10000, cash: 10000,
  position: 0, sharpe: 0, sortino: 0, calmar: 0,
  maxDrawdown: 0, totalReturn: 0, winLossRatio: 0, avgTurnover: 0,
  price: 0, reward: 0,
  portfolioHistory: [], lossHistory: [], priceHistory: [], tradeLog: [],
  lifetimeSteps: 0, currentEpTicker: "",
};

// ── Helpers ───────────────────────────────────────────────────────────
const fmt$ = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 0 }).format(n);

const fmtPct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
const fmtFixed2 = (n: number) => n.toFixed(2);

// ── Main App ──────────────────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState<AppState>(DEFAULT_STATE);
  const [ticker, setTicker] = useState("AAPL");
  const [stepsInput, setStepsInput] = useState("500000");
  const [winRate, setWinRate] = useState(0);
  const [stepsPerSec, setStepsPerSec] = useState(0);
  const [rolloutCount, setRolloutCount] = useState(0);
  const [effLoss, setEffLoss] = useState<number[]>([]);
  const [effSharpe, setEffSharpe] = useState<number[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<"wf" | "trades">("wf");
  const [wfWindows, setWfWindows] = useState<WFWindow[]>([]);
  const [wfSummary, setWfSummary] = useState({ avg_test_sharpe: 0, generalization_gap: 0, trend_slope: 0 });
  const [wfRunning, setWfRunning] = useState(false);

  const prevRolloutCount = useRef(0);
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
        sharpe:           d.sharpe       ?? s.sharpe,
        sortino:          d.sortino      ?? s.sortino,
        calmar:           d.calmar       ?? s.calmar,
        maxDrawdown:      d.max_drawdown,
        totalReturn:      d.total_return,
        winLossRatio:     d.win_loss_ratio ?? s.winLossRatio,
        avgTurnover:      d.avg_turnover   ?? s.avgTurnover,
        portfolioHistory: d.portfolio_history || [],
        lossHistory:      d.loss_history     || [],
        priceHistory:     d.price_history    || [],
        tradeLog:         d.trade_log        || [],
        lifetimeSteps:    d.lifetime_steps   || 0,
        currentEpTicker:  d.current_ep_ticker ?? "",
      }));
      if (d.eff_loss?.length)   setEffLoss(d.eff_loss);
      if (d.eff_sharpe?.length) setEffSharpe(d.eff_sharpe);
      if (d.win_rate !== undefined)      setWinRate(d.win_rate);
      if (d.steps_per_sec !== undefined) setStepsPerSec(d.steps_per_sec);
      if (d.rollout_count !== undefined) {
        setRolloutCount(d.rollout_count);
        prevRolloutCount.current = d.rollout_count;
      }
    }).catch(() => {});
  }, []);

  // Load walk-forward
  useEffect(() => {
    getWalkForward(ticker).then((d) => {
      setWfWindows(d.windows || []);
      setWfSummary({
        avg_test_sharpe:    d.avg_test_sharpe    || 0,
        generalization_gap: d.generalization_gap || 0,
        trend_slope:        d.trend_slope        || 0,
      });
    }).catch(() => {});
  }, [ticker]);

  const handleRunWalkForward = useCallback(async () => {
    setWfRunning(true);
    await runWalkForward(ticker, 20_000);
    const poll = setInterval(async () => {
      const [d, s] = await Promise.all([
        getWalkForward(ticker),
        fetch("http://localhost:8000/api/walkforward/status/current").then((r) => r.json()),
      ]);
      if (d.windows?.length > 0) {
        setWfWindows(d.windows);
        setWfSummary({ avg_test_sharpe: d.avg_test_sharpe, generalization_gap: d.generalization_gap, trend_slope: d.trend_slope });
      }
      if (!s.running) {
        clearInterval(poll);
        setWfRunning(false);
        if (s.error) { alert(`Walk-forward failed:\n\n${s.error}`); return; }
        getWalkForward(ticker).then((final) => {
          if (final.windows?.length > 0) {
            setWfWindows(final.windows);
            setWfSummary({ avg_test_sharpe: final.avg_test_sharpe, generalization_gap: final.generalization_gap, trend_slope: final.trend_slope });
          }
        }).catch(() => {});
      }
    }, 5000);
  }, [ticker]);

  // WebSocket ticks
  useEffect(() => {
    if (!lastTick) return;
    const t = lastTick;

    if (t.type === "init") {
      setState((s) => ({
        ...s,
        isTraining:       t.is_training      ?? s.isTraining,
        portfolioHistory: t.portfolio_history ?? s.portfolioHistory,
        lossHistory:      t.loss_history      ?? s.lossHistory,
        priceHistory:     t.price_history     ?? s.priceHistory,
        tradeLog:         t.trade_log         ?? s.tradeLog,
      }));
      return;
    }

    if (t.type === "training_complete") {
      setState((s) => ({ ...s, isTraining: false }));
      return;
    }

    if (t.type === "tick") {
      const pos = t.position ?? 0;
      if (Math.abs(pos - prevPos.current) > 0.5) prevPos.current = pos;

      if (t.win_rate      !== undefined) setWinRate(t.win_rate);
      if (t.steps_per_sec !== undefined) setStepsPerSec(t.steps_per_sec);

      const rc = t.rollout_count;
      if (rc !== undefined) {
        setRolloutCount(rc);
        if (rc > prevRolloutCount.current) {
          prevRolloutCount.current = rc;
          const loss   = t.loss;
          const sharpe = t.sharpe;
          if (loss   !== undefined) setEffLoss(h   => [...h.slice(-299), loss]);
          if (sharpe !== undefined) setEffSharpe(h => [...h.slice(-299), sharpe]);
        }
      }

      setState((s) => ({
        ...s,
        isTraining:       t.is_training       !== undefined ? t.is_training : s.isTraining,
        totalSteps:       t.total_steps        ?? s.totalSteps,
        totalStepsTarget: t.total_steps_target ?? s.totalStepsTarget,
        episode:          t.episode            ?? s.episode,
        loss:             t.loss               ?? s.loss,
        portfolioValue:   t.portfolio_value    ?? s.portfolioValue,
        cash:             t.cash               ?? s.cash,
        position:         pos,
        sharpe:           t.sharpe             ?? s.sharpe,
        sortino:          t.sortino            ?? s.sortino,
        calmar:           t.calmar             ?? s.calmar,
        maxDrawdown:      t.max_drawdown       ?? s.maxDrawdown,
        totalReturn:      t.total_return       ?? s.totalReturn,
        winLossRatio:     t.win_loss_ratio     ?? s.winLossRatio,
        avgTurnover:      t.avg_turnover       ?? s.avgTurnover,
        price:            t.price              ?? s.price,
        reward:           t.reward             ?? s.reward,
        currentEpTicker:  t.current_ep_ticker  ?? s.currentEpTicker,
        tradeLog:         t.trade_log          ?? s.tradeLog,
        portfolioHistory: t.portfolio_value !== undefined
          ? [...s.portfolioHistory.slice(-299), t.portfolio_value]
          : s.portfolioHistory,
        priceHistory: t.price !== undefined
          ? [...s.priceHistory.slice(-299), t.price]
          : s.priceHistory,
        lossHistory: t.loss !== undefined
          ? [...s.lossHistory.slice(-199), t.loss]
          : s.lossHistory,
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

  const progress = Math.min((state.totalSteps / (state.totalStepsTarget || 1)) * 100, 100);
  const returnPos = state.totalReturn >= 0;

  // Position bar fill (0 = all sell, 50 = hold, 100 = all buy)
  const posPct = ((state.position + 1) / 2) * 100;
  const posColor = state.position > 0.15 ? "var(--green)" : state.position < -0.15 ? "var(--red)" : "var(--yellow)";

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="logo">
          <div className="logo-dot" />
          <span className="logo-text">RL<span>trader</span></span>
        </div>

        <div className={`ws-pill ${connected ? "live" : "dead"}`}>
          <span className="ws-dot" />
          {connected ? "live" : "reconnecting"}
        </div>

        <div className="controls">
          <select
            className="ticker-select"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            disabled={state.isTraining}
          >
            {["AAPL","MSFT","GOOGL","TSLA","SPY","NVDA","AMZN"].map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>

          <input
            className="steps-input"
            type="number"
            value={stepsInput}
            onChange={(e) => setStepsInput(e.target.value)}
            disabled={state.isTraining}
            placeholder="Steps"
          />

          <AnimatePresence mode="wait">
            {!state.isTraining ? (
              <motion.button
                key="start"
                className="btn btn-start"
                onClick={handleStart}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.15 }}
              >
                ▶ Start
              </motion.button>
            ) : (
              <motion.div
                key="controls"
                style={{ display: "flex", gap: 6 }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <button className="btn btn-pause" onClick={handlePause}>⏸ Pause</button>
                <button className="btn btn-stop"  onClick={handleStop}>■ Stop</button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* ── Progress strip ── */}
      <div className="progress-strip">
        <motion.div
          className="progress-fill"
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        />
      </div>

      {/* ── Zone 1: Stats rail ── */}
      <div className="stats-rail">
        <StatCard
          label="Steps"
          value={state.totalSteps}
          format={(n) => (n / 1000).toFixed(0) + "k"}
          sub={`${progress.toFixed(1)}%`}
        />
        <StatCard
          label="Episode"
          value={state.episode}
          format={(n) => n.toFixed(0)}
          sub={state.currentEpTicker || undefined}
        />
        <StatCard
          label="Portfolio"
          value={state.portfolioValue}
          format={fmt$}
          colorClass={returnPos ? "pos" : "neg"}
          sub={fmtPct(state.totalReturn)}
        />
        <StatCard
          label="Sharpe"
          value={state.sharpe}
          format={fmtFixed2}
          colorClass={state.sharpe >= 1 ? "pos" : state.sharpe < 0 ? "neg" : ""}
        />
        <StatCard
          label="Sortino"
          value={state.sortino}
          format={fmtFixed2}
          colorClass={state.sortino >= 1.5 ? "pos" : state.sortino < 0 ? "neg" : ""}
        />
        <StatCard
          label="Max DD"
          value={state.maxDrawdown}
          format={(n) => n.toFixed(1) + "%"}
          colorClass="neg"
        />
        <StatCard
          label="Win Rate"
          value={winRate * 100}
          format={(n) => n.toFixed(1) + "%"}
          colorClass={winRate >= 0.5 ? "pos" : "neg"}
          sub={`${rolloutCount} rollouts`}
        />
        <StatCard
          label="Steps/sec"
          value={stepsPerSec}
          format={(n) => n.toFixed(0)}
          colorClass="blue"
          sub={`${(state.lifetimeSteps / 1000).toFixed(0)}k lifetime`}
        />
      </div>

      {/* ── Zone 2: Main charts ── */}
      <div className="charts-row">

        {/* Left: Portfolio vs buy-and-hold */}
        <div className="chart-panel">
          <div className="chart-header">
            <span className="chart-title">Portfolio vs Buy-and-Hold</span>
            <span className="chart-meta">
              <span><span className="legend-dot" style={{ background: "var(--green)" }} />agent</span>
              <span><span className="legend-dot" style={{ background: "#f0b42966" }} />price</span>
              <span style={{ color: returnPos ? "var(--green)" : "var(--red)" }}>
                {fmtPct(state.totalReturn)}
              </span>
            </span>
          </div>
          <DualLineChart portfolio={state.portfolioHistory} price={state.priceHistory} />
        </div>

        {/* Right: Training trend (loss + sharpe per rollout) */}
        <div className="chart-panel">
          <div className="chart-header">
            <span className="chart-title">Training Trend</span>
            <span className="chart-meta">
              <span><span className="legend-dot" style={{ background: "var(--blue)" }} />loss</span>
              <span><span className="legend-dot" style={{ background: "var(--green)" }} />sharpe</span>
              <span style={{ color: "var(--muted)" }}>{rolloutCount} rollouts</span>
            </span>
          </div>
          <SharpeTrend sharpeHistory={effSharpe} lossHistory={effLoss} />
        </div>

      </div>

      {/* ── Position bar (slim) ── */}
      <div style={{ flexShrink: 0, padding: "6px 20px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 12, background: "var(--surface)" }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted-hi)", textTransform: "uppercase", letterSpacing: "0.1em", width: 64, flexShrink: 0 }}>Agent Pos</span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted)" }}>SELL</span>
        <div style={{ flex: 1, position: "relative", height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
          <motion.div
            style={{ position: "absolute", left: 0, top: 0, bottom: 0, borderRadius: 2, background: posColor }}
            animate={{ width: `${posPct}%` }}
            transition={{ type: "spring", stiffness: 60, damping: 15 }}
          />
          <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--border-hi)" }} />
        </div>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted)" }}>BUY</span>
        <motion.span
          key={state.position}
          style={{ fontFamily: "var(--mono)", fontSize: 11, color: posColor, width: 48, textAlign: "right" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {state.position >= 0 ? "+" : ""}{(state.position * 100).toFixed(1)}%
        </motion.span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted-hi)", marginLeft: 12 }}>
          W/L {state.winLossRatio.toFixed(2)} · Turnover {state.avgTurnover.toFixed(1)}x · Calmar {state.calmar.toFixed(2)} · Loss {state.loss.toFixed(4)}
        </span>
      </div>

      {/* ── Zone 3: Collapsible drawer ── */}
      <div className="drawer">
        <div className="drawer-toggle" onClick={() => setDrawerOpen((o) => !o)}>
          <div className="drawer-toggle-left">
            <span className="drawer-title">Analysis</span>
            <div className="drawer-tabs" onClick={(e) => e.stopPropagation()}>
              <button
                className={`drawer-tab ${drawerTab === "wf" ? "active" : ""}`}
                onClick={() => { setDrawerTab("wf"); setDrawerOpen(true); }}
              >
                Walk-Forward
              </button>
              <button
                className={`drawer-tab ${drawerTab === "trades" ? "active" : ""}`}
                onClick={() => { setDrawerTab("trades"); setDrawerOpen(true); }}
              >
                Trade Log
              </button>
            </div>
          </div>
          <span className={`drawer-chevron ${drawerOpen ? "open" : ""}`}>▲</span>
        </div>

        <AnimatePresence initial={false}>
          {drawerOpen && (
            <motion.div
              key="drawer-body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 220, opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              style={{ overflow: "hidden" }}
            >
              <div className="drawer-body">

                {/* Left section */}
                <div className="drawer-section" style={{ height: 220 }}>
                  {drawerTab === "wf" ? (
                    <>
                      <div className="wf-stat-row">
                        {wfWindows.length > 0 && (<>
                          <div className="wf-stat">
                            <span className="wf-stat-label">Avg Test Sharpe</span>
                            <span className={`wf-stat-val ${wfSummary.avg_test_sharpe >= 0.5 ? "pos" : wfSummary.avg_test_sharpe < 0 ? "neg" : "neutral"}`}>
                              {wfSummary.avg_test_sharpe >= 0 ? "+" : ""}{wfSummary.avg_test_sharpe.toFixed(3)}
                            </span>
                          </div>
                          <div className="wf-stat">
                            <span className="wf-stat-label">Gen. Gap</span>
                            <span className={`wf-stat-val ${wfSummary.generalization_gap < 0.3 ? "pos" : "neg"}`}>
                              {wfSummary.generalization_gap.toFixed(3)}
                            </span>
                          </div>
                          <div className="wf-stat">
                            <span className="wf-stat-label">Trend</span>
                            <span className={`wf-stat-val ${wfSummary.trend_slope > 0.02 ? "pos" : wfSummary.trend_slope < -0.02 ? "neg" : "neutral"}`}>
                              {wfSummary.trend_slope > 0.02 ? "↑ IMPROVING" : wfSummary.trend_slope < -0.02 ? "↓ DECLINING" : "→ STABLE"}
                            </span>
                          </div>
                        </>)}
                        <button
                          className={`btn btn-sm wf-run-btn ${wfRunning ? "btn-pause" : "btn-start"}`}
                          onClick={handleRunWalkForward}
                          disabled={wfRunning || state.isTraining}
                        >
                          {wfRunning ? "⏳ Analyzing…" : "⚡ Run Analysis"}
                        </button>
                      </div>
                      <div style={{ flex: 1, minHeight: 0 }}>
                        <WalkForwardChart windows={wfWindows} height={148} />
                      </div>
                    </>
                  ) : (
                    <div className="trade-list">
                      {state.tradeLog.length === 0 && (
                        <div className="chart-empty">no trades yet</div>
                      )}
                      {[...state.tradeLog].reverse().map((t, i) => (
                        <div key={i} className="trade-row">
                          <span className="trade-time">{t.time}</span>
                          <span className={`trade-badge ${t.action.toLowerCase()}`}>{t.action}</span>
                          <span className="trade-price">${t.price.toFixed(2)}</span>
                          <span className="trade-alloc">{(t.alloc * 100).toFixed(0)}%</span>
                          <span className={`trade-pnl ${t.pnl >= 0 ? "pos" : "neg"}`}>
                            {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Right: mini-metrics */}
                <div className="drawer-section" style={{ height: 220, gap: 10 }}>
                  <span className="section-label">Episode Snapshot</span>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {[
                      ["Cash",    `$${state.cash.toFixed(0)}`,           "var(--text)"],
                      ["Price",   `$${state.price.toFixed(2)}`,           "var(--yellow)"],
                      ["Reward",  state.reward.toFixed(5),                state.reward >= 0 ? "var(--green)" : "var(--red)"],
                      ["Calmar",  state.calmar.toFixed(2),                state.calmar >= 0.5 ? "var(--green)" : "var(--muted-hi)"],
                      ["W/L",     state.winLossRatio.toFixed(2),          state.winLossRatio >= 1.5 ? "var(--green)" : "var(--muted-hi)"],
                      ["Turnover",state.avgTurnover.toFixed(1) + "x",     state.avgTurnover < 5 ? "var(--green)" : "var(--red)"],
                      ["Episode", state.episode.toFixed(0),               "var(--purple)"],
                      ["Ticker",  state.currentEpTicker || "—",           "var(--purple)"],
                    ].map(([label, val, color]) => (
                      <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--muted-hi)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</span>
                        <span style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700, color }}>{val}</span>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </div>
  );
}
