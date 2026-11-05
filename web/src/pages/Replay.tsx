import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ReplayChart from "../components/ReplayChart";
import Attribution from "../components/Attribution";
import MemoryMap from "../components/MemoryMap";
import type { Manifest, Trace } from "../lib/data";
import { fmtDate, num, pct, pctAbs, useData } from "../lib/data";

const SPEEDS = [10, 30, 90]; // trading days per second

export default function Replay() {
  const { ticker = "SPY" } = useParams();
  const nav = useNavigate();
  const { data: manifest } = useData<Manifest>("manifest.json");
  const { data: trace, error } = useData<Trace>(`traces/${ticker}.json`);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const raf = useRef(0);

  useEffect(() => { if (trace) { setIndex(trace.dates.length - 1); setPlaying(false); } }, [trace]);

  const n = trace?.dates.length ?? 0;
  useEffect(() => {
    if (!playing || !n) return;
    let last = performance.now(), acc = 0;
    const tick = (now: number) => {
      acc += ((now - last) / 1000) * SPEEDS[speed]; last = now;
      if (acc >= 1) {
        const k = Math.floor(acc); acc -= k;
        setIndex((i) => { const j = Math.min(n - 1, i + k); if (j >= n - 1) setPlaying(false); return j; });
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, speed, n]);

  const toggle = useCallback(() => {
    setPlaying((p) => { if (!p && index >= n - 1) setIndex(0); return !p; });
  }, [index, n]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, select, textarea")) return;
      if (e.key === " ") { e.preventDefault(); toggle(); }
      else if (e.key === "ArrowRight") { setPlaying(false); setIndex((i) => Math.min(n - 1, i + (e.shiftKey ? 21 : 1))); }
      else if (e.key === "ArrowLeft") { setPlaying(false); setIndex((i) => Math.max(0, i - (e.shiftKey ? 21 : 1))); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, n]);

  const groups = manifest ? (["train", "held_out"] as const).map((g) => ({
    g, label: g === "train" ? "Trained on" : "Never seen",
    items: manifest.tickers.filter((t) => t.group === g),
  })) : [];

  // decision for the day under the playhead (the last day has no decision)
  const d = trace ? Math.min(index, trace.allocation.length - 1) : 0;
  const delta = trace && d > 0 ? trace.allocation[d] - trace.allocation[d - 1] : 0;

  return (
    <div className="wrap">
      <header className="page-head">
        <span className="eyebrow">Replay · {manifest ? `${manifest.window.start.slice(0, 4)}–${manifest.window.end.slice(0, 4)}, out of sample` : "…"}</span>
        <h1>Watch the agent trade, one day at a time</h1>
        <p>Every day it chooses how much of the portfolio to hold in stock. Press play, or drag across the chart, to see what it held, what drove each decision, and where its memory was.</p>
      </header>

      <div className="grid" style={{ gap: 10, marginBottom: 18 }}>
        {groups.map(({ g, label, items }) => (
          <div className="chips" key={g} role="group" aria-label={label}>
            <span className="chip-group-label">{label}</span>
            {items.map((t) => (
              <button key={t.ticker} className="chip" aria-pressed={t.ticker === ticker}
                onClick={() => nav(`/replay/${t.ticker}`)}>{t.ticker}</button>
            ))}
          </div>
        ))}
      </div>

      {error && <div className="card">Couldn't load {ticker}: {error}</div>}
      {!trace && !error && <div className="skeleton" />}
      {trace && (
        <div className="grid">
          <div className="stats">
            <Stat label="Sharpe ratio" v={num(trace.metrics.agent.sharpe)} vs={num(trace.metrics.bnh.sharpe)} />
            <Stat label="Worst drawdown" v={pct(trace.metrics.agent.max_drawdown)} vs={pct(trace.metrics.bnh.max_drawdown)} />
            <Stat label="Total return" v={pct(trace.metrics.agent.total_return)} vs={pct(trace.metrics.bnh.total_return)} />
            <Stat label="Avg. in stock" v={pctAbs(trace.metrics.agent.avg_exposure ?? 0)} vs="100%" />
          </div>

          <section className="card">
            <div className="card-head">
              <div className="legend">
                <span><i className="swatch" style={{ background: "var(--s-agent)" }} />Agent</span>
                <span><i className="swatch" style={{ background: "var(--s-bench)" }} />Buy &amp; hold</span>
                <span><i className="swatch block" style={{ background: "var(--s-agent)", opacity: 0.4 }} />Share of portfolio in stock</span>
              </div>
              <span className="mono muted" style={{ fontSize: ".85rem" }}>{fmtDate(trace.dates[index])}</span>
            </div>
            <ReplayChart trace={trace} index={index} onIndex={(i) => { setPlaying(false); setIndex(i); }} />
            <div className="controls" style={{ marginTop: 12 }}>
              <button className="btn primary" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
                {playing ? "❚❚ Pause" : index >= n - 1 ? "▶ Replay" : "▶ Play"}
              </button>
              <input className="scrub" type="range" min={0} max={n - 1} value={index} aria-label="Day"
                onChange={(e) => { setPlaying(false); setIndex(+e.target.value); }} />
              <div className="chips" role="group" aria-label="Speed">
                {SPEEDS.map((s, k) => (
                  <button key={s} className="chip" aria-pressed={k === speed} onClick={() => setSpeed(k)}>{["1×", "3×", "9×"][k]}</button>
                ))}
              </div>
              <span className="muted" style={{ fontSize: ".78rem" }}>Space to play · ←/→ step a day · Shift for a month</span>
            </div>
          </section>

          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 420px), 1fr))" }}>
            <section className="card">
              <div className="card-head">
                <h2>Why it held {pctAbs(trace.allocation[d])}</h2>
                <span className="mono muted" style={{ fontSize: ".82rem" }}>
                  {fmtDate(trace.dates[d])} · {delta >= 0 ? "+" : "−"}{Math.abs(delta * 100).toFixed(1)}pt vs day before
                </span>
              </div>
              {manifest && (
                <Attribution names={manifest.obs_names} attr={trace.attribution[d]} obs={trace.obs[d]}
                  baseline={trace.attribution_baseline[d]} allocation={trace.allocation[d]} />
              )}
              <p className="muted" style={{ fontSize: ".78rem", marginTop: 8 }}>
                Integrated gradients on today's inputs, holding the agent's memory fixed. Memory of earlier days isn't credited here.
              </p>
            </section>
            <section className="card">
              <div className="card-head">
                <h2>Where its memory was</h2>
                {manifest && <span className="mono muted" style={{ fontSize: ".78rem" }}>
                  PCA of {manifest.model.hidden_dim}-d LSTM state · {pctAbs(manifest.memory_projection.explained_variance.reduce((a, b) => a + b, 0))} of variance
                </span>}
              </div>
              <MemoryMap xy={trace.memory_xy} allocation={trace.allocation} dates={trace.dates}
                index={d} onIndex={(i) => { setPlaying(false); setIndex(i); }} />
            </section>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, v, vs }: { label: string; v: string; vs: string }) {
  return (
    <div className="stat">
      <div className="eyebrow">{label}</div>
      <div className="v">{v}</div>
      <div className="vs">buy &amp; hold {vs}</div>
    </div>
  );
}
