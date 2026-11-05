import { useEffect, useMemo, useRef, useState } from "react";
import type { Manifest } from "../lib/data";
import { pctAbs, useData } from "../lib/data";
import { allocate } from "../lib/policy";
import { useWidth } from "../lib/chart";

type Scaling = Record<string, [number, number]>;
type Knobs = { vix: number; trend: number; vol: number; rsi: number; drawdown: number; position: number };

const KNOBS: { key: keyof Knobs; label: string; min: number; max: number; step: number; fmt: (v: number) => string }[] = [
  { key: "vix", label: "VIX", min: 10, max: 80, step: 1, fmt: (v) => v.toFixed(0) },
  { key: "trend", label: "Price vs 30-day avg", min: -15, max: 15, step: 0.5, fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}%` },
  { key: "vol", label: "Daily volatility", min: 0.5, max: 6, step: 0.1, fmt: (v) => `${v.toFixed(1)}%` },
  { key: "rsi", label: "RSI", min: 10, max: 90, step: 1, fmt: (v) => v.toFixed(0) },
  { key: "drawdown", label: "Agent's drawdown", min: -30, max: 0, step: 1, fmt: (v) => `${v.toFixed(0)}%` },
  { key: "position", label: "Currently in stock", min: 0, max: 100, step: 5, fmt: (v) => `${v.toFixed(0)}%` },
];
const DEFAULTS: Knobs = { vix: 18, trend: 2, vol: 1.2, rsi: 55, drawdown: 0, position: 25 };
const GRID = 28, VIX_RANGE: [number, number] = [10, 80], TREND_RANGE: [number, number] = [-15, 15];

/** Build a scaled observation from human-scale knobs; everything else sits at its
 *  training-set centre (scaled 0), i.e. an otherwise typical day. */
function makeObs(names: string[], sc: Scaling, k: Knobs): Float32Array {
  const raw: Record<string, number> = {
    vix_norm: k.vix / 30,
    sma_30: 1 / (1 + k.trend / 100) - 1,          // avg / close − 1
    sma_10: 1 / (1 + k.trend / 200) - 1,
    volatility: k.vol / 100,
    atr_norm: (k.vol / 100) * 1.35,
    rsi: k.rsi / 100,
  };
  const o = new Float32Array(names.length);
  names.forEach((n, i) => {
    let v = 0;
    if (n === "position") v = k.position / 100;
    else if (n === "drawdown") v = k.drawdown / 100 / 0.1;
    else if (n in raw && sc[n]) v = (raw[n] - sc[n][0]) / sc[n][1];
    o[i] = Math.max(-5, Math.min(5, v));
  });
  return o;
}

export default function Lab() {
  const { data: m } = useData<Manifest>("manifest.json");
  const [k, setK] = useState<Knobs>(DEFAULTS);
  const [warm, setWarm] = useState(true);
  const [alloc, setAlloc] = useState<number | null>(null);
  const [grid, setGrid] = useState<Float32Array | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const seq = useRef(0);
  const sc = m?.feature_scaling as Scaling | undefined;
  const warmup = warm ? 20 : 0;

  useEffect(() => {
    if (!m || !sc) return;
    const id = ++seq.current;
    allocate([makeObs(m.obs_names, sc, k)], m.model.hidden_dim, warmup)
      .then((a) => id === seq.current && setAlloc(a[0]), (e) => setErr(String(e)));
  }, [m, sc, k, warmup]);

  // the map only depends on the knobs that aren't its axes
  const { vix: _v, trend: _t, ...rest } = k;
  const restKey = JSON.stringify(rest);
  useEffect(() => {
    if (!m || !sc) return;
    const obs: Float32Array[] = [];
    for (let yi = 0; yi < GRID; yi++) for (let xi = 0; xi < GRID; xi++) {
      const vix = VIX_RANGE[0] + (xi / (GRID - 1)) * (VIX_RANGE[1] - VIX_RANGE[0]);
      const trend = TREND_RANGE[1] - (yi / (GRID - 1)) * (TREND_RANGE[1] - TREND_RANGE[0]);
      obs.push(makeObs(m.obs_names, sc, { ...k, vix, trend }));
    }
    let live = true;
    allocate(obs, m.model.hidden_dim, warmup).then((a) => live && setGrid(a), (e) => setErr(String(e)));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m, sc, restKey, warmup]);

  return (
    <div className="wrap">
      <header className="page-head">
        <span className="eyebrow">Policy lab · the real model, running in your browser</span>
        <h1>Set the market. See what it would hold.</h1>
        <p>Move the sliders and the trained network answers instantly. The map shows its answer for every combination of VIX and trend at once, given the other settings.</p>
      </header>
      {err && <div className="card">Couldn't run the model: {err}</div>}
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 380px), 1fr))" }}>
        <section className="card">
          <div className="lab-readout">
            <div>
              <div className="eyebrow">Would hold in stock</div>
              <div className="lab-big mono">{alloc === null ? "…" : pctAbs(alloc)}</div>
            </div>
            <div className="meter"><i style={{ width: `${(alloc ?? 0) * 100}%` }} /></div>
          </div>
          <div className="grid" style={{ gap: 12, marginTop: 18 }}>
            {KNOBS.map((d) => (
              <label key={d.key} className="knob">
                <span>{d.label}</span>
                <input type="range" min={d.min} max={d.max} step={d.step} value={k[d.key]}
                  onChange={(e) => setK({ ...k, [d.key]: +e.target.value })} />
                <output className="mono">{d.fmt(k[d.key])}</output>
              </label>
            ))}
          </div>
          <div className="controls" style={{ marginTop: 16, justifyContent: "space-between" }}>
            <div className="chips" role="group" aria-label="Memory">
              <span className="chip-group-label">Memory</span>
              <button className="chip" aria-pressed={warm} onClick={() => setWarm(true)}>20 days like this</button>
              <button className="chip" aria-pressed={!warm} onClick={() => setWarm(false)}>Blank</button>
            </div>
            <button className="btn" onClick={() => setK(DEFAULTS)}>Reset</button>
          </div>
          <p className="muted" style={{ fontSize: ".78rem", marginTop: 12 }}>
            Other inputs sit at a typical day's values. "20 days like this" feeds the same conditions for 20 days first, so the LSTM's memory has settled.
          </p>
        </section>
        <section className="card">
          <div className="card-head">
            <h2>Every VIX × trend combination</h2>
            <span className="legend"><span><i className="swatch block" style={{ background: "var(--s-agent)", opacity: 0.12 }} />0%</span>
              <span><i className="swatch block" style={{ background: "var(--s-agent)" }} />100% in stock</span></span>
          </div>
          <Heatmap grid={grid} vix={k.vix} trend={k.trend} onPick={(vix, trend) => setK({ ...k, vix, trend })} />
        </section>
      </div>
      <div className="callout" style={{ marginTop: 16 }}>
        <p>Remember the scorecard: this policy did not beat buy &amp; hold out of sample. The lab shows <em>what</em> it learned, not that it's right.</p>
      </div>
    </div>
  );
}

function Heatmap({ grid, vix, trend, onPick }: { grid: Float32Array | null; vix: number; trend: number; onPick: (v: number, t: number) => void }) {
  const [ref, width] = useWidth<HTMLDivElement>(400);
  const [hover, setHover] = useState<{ x: number; y: number; a: number } | null>(null);
  const L = 44, B = 30, size = Math.min(width - L - 8, 460), cell = size / GRID;
  const toV = (xi: number) => VIX_RANGE[0] + (xi / (GRID - 1)) * (VIX_RANGE[1] - VIX_RANGE[0]);
  const toT = (yi: number) => TREND_RANGE[1] - (yi / (GRID - 1)) * (TREND_RANGE[1] - TREND_RANGE[0]);
  const cx = L + ((vix - VIX_RANGE[0]) / (VIX_RANGE[1] - VIX_RANGE[0])) * (size - cell) + cell / 2;
  const cy = ((TREND_RANGE[1] - trend) / (TREND_RANGE[1] - TREND_RANGE[0])) * (size - cell) + cell / 2;
  const cells = useMemo(() => {
    if (!grid) return null;
    const out = [];
    for (let yi = 0; yi < GRID; yi++) for (let xi = 0; xi < GRID; xi++) {
      const a = grid[yi * GRID + xi];
      out.push(<rect key={`${xi}-${yi}`} x={L + xi * cell} y={yi * cell} width={cell + 0.5} height={cell + 0.5}
        style={{ fill: `color-mix(in srgb, var(--s-agent) ${(6 + 94 * a).toFixed(1)}%, var(--surface))`, cursor: "pointer" }}
        onPointerEnter={() => setHover({ x: toV(xi), y: toT(yi), a })}
        onClick={() => onPick(Math.round(toV(xi)), Math.round(toT(yi) * 2) / 2)} />);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid, cell]);
  return (
    <div ref={ref} className="chart" onPointerLeave={() => setHover(null)}>
      {!grid ? <div className="skeleton" style={{ height: size }} /> : (
        <svg width={width} height={size + B} role="img" aria-label="Allocation heatmap by VIX and trend">
          {cells}
          <circle cx={cx} cy={cy} r={6} fill="none" stroke="var(--fg)" strokeWidth={2} />
          <g className="axis">
            {[10, 30, 50, 70].map((v) => <text key={v} x={L + ((v - 10) / 70) * (size - cell) + cell / 2} y={size + 14} textAnchor="middle">{v}</text>)}
            <text x={L + size / 2} y={size + 28} textAnchor="middle">VIX →</text>
            {[15, 5, -5, -15].map((t) => <text key={t} x={L - 6} y={((15 - t) / 30) * (size - cell) + cell / 2 + 3.5} textAnchor="end">{t > 0 ? "+" : ""}{t}%</text>)}
          </g>
        </svg>
      )}
      <div className="legend mono" style={{ minHeight: 20 }}>
        {hover ? `VIX ${hover.x.toFixed(0)} · trend ${hover.y >= 0 ? "+" : ""}${hover.y.toFixed(1)}% → ${pctAbs(hover.a)} in stock` : "Trend (price vs 30-day avg) ↑ · click to set"}
      </div>
    </div>
  );
}
