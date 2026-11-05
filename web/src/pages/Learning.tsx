import { useMemo, useState } from "react";
import ReplayChart from "../components/ReplayChart";
import type { LearningCurves, Replays, Trace } from "../lib/data";
import { num, pct, useData } from "../lib/data";
import { extent, linePath, linear, pad, ticks, useWidth } from "../lib/chart";

export default function Learning() {
  const { data: lc } = useData<LearningCurves>("learning_curves.json");
  const { data: rp } = useData<Replays>("replays.json");
  return (
    <div className="wrap">
      <header className="page-head">
        <span className="eyebrow">How it learned</span>
        <h1>It got better at the past, not the future</h1>
        <p>Training used 2010–2020. Every 50,000 steps the agent was tested on 2021–22, which it never trained on, and the best checkpoint was kept. The gap between the two lines is overfitting.</p>
      </header>
      <div className="grid">
        {lc ? <Curves lc={lc} /> : <div className="skeleton" />}
        {rp ? <CheckpointReplay rp={rp} /> : <div className="skeleton" />}
      </div>
    </div>
  );
}

function Curves({ lc }: { lc: LearningCurves }) {
  const [kappa, setKappa] = useState(5);
  const [seed, setSeed] = useState(3);
  const [ref, width] = useWidth<HTMLDivElement>();
  const runs = lc.runs.filter((r) => r.risk_aversion === kappa);
  const run = runs.find((r) => r.seed === seed) ?? runs[0];
  const H = 300, M = { l: 40, r: 96, t: 12, b: 24 };

  const g = useMemo(() => {
    const maxStep = Math.max(...lc.runs.flatMap((r) => r.train.map((p) => p.steps)));
    const x = linear([0, maxStep], [M.l, width - M.r]);
    const ydom = pad(extent(...lc.runs.flatMap((r) => [r.train.map((p) => p.sharpe), r.val.map((p) => p.val_sharpe)])));
    const y = linear(ydom, [H - M.b, M.t]);
    return { x, y, yt: ticks(ydom, 5), xt: ticks([0, maxStep], 5) };
  }, [lc, width, M.l, M.r, M.b, M.t]);

  const best = run.val.reduce((a, b) => (b.val_sharpe > a.val_sharpe ? b : a));
  const lastT = run.train[run.train.length - 1], lastV = run.val[run.val.length - 1];
  return (
    <section className="card">
      <div className="card-head">
        <div className="legend">
          <span><i className="swatch" style={{ background: "var(--s-agent)" }} />Training years (2010–20)</span>
          <span><i className="swatch" style={{ background: "var(--s-bench)" }} />Validation years (2021–22)</span>
        </div>
        <div className="chips">
          <span className="chip-group-label">Risk aversion</span>
          {[1, 5].map((k) => <button key={k} className="chip" aria-pressed={k === kappa} onClick={() => setKappa(k)}>κ={k}</button>)}
          <span className="chip-group-label" style={{ marginLeft: 8 }}>Seed</span>
          {runs.map((r) => <button key={r.seed} className="chip" aria-pressed={r.seed === run.seed} onClick={() => setSeed(r.seed)}>{r.seed}</button>)}
        </div>
      </div>
      <div ref={ref} className="chart">
        <svg width={width} height={H} role="img" aria-label="Sharpe ratio on training versus validation years over training steps">
          <g className="axis">
            {g.yt.map((t) => <g key={t}><line className="gridline" x1={M.l} x2={width - M.r} y1={g.y(t)} y2={g.y(t)} />
              <text x={M.l - 8} y={g.y(t) + 3.5} textAnchor="end">{t.toFixed(1)}</text></g>)}
            {g.xt.map((t) => <text key={t} x={g.x(t)} y={H - 6} textAnchor="middle">{t ? `${t / 1e6 >= 1 ? t / 1e6 + "M" : t / 1e3 + "k"}` : "0"}</text>)}
            <line x1={M.l} x2={width - M.r} y1={g.y(0)} y2={g.y(0)} stroke="var(--muted)" strokeOpacity={0.5} />
          </g>
          {runs.filter((r) => r !== run).map((r) => (
            <path key={r.run} d={linePath(r.val.map((p) => g.x(p.steps)), r.val.map((p) => g.y(p.val_sharpe)))}
              fill="none" stroke="var(--s-bench)" strokeOpacity={0.2} strokeWidth={1.5} />
          ))}
          <path d={linePath(run.train.map((p) => g.x(p.steps)), run.train.map((p) => g.y(p.sharpe)))} fill="none" stroke="var(--s-agent)" strokeWidth={2} />
          <path d={linePath(run.val.map((p) => g.x(p.steps)), run.val.map((p) => g.y(p.val_sharpe)))} fill="none" stroke="var(--s-bench)" strokeWidth={2} />
          <circle cx={g.x(best.steps)} cy={g.y(best.val_sharpe)} r={5} fill="var(--s-bench)" stroke="var(--surface)" strokeWidth={2} />
          <text x={g.x(best.steps)} y={g.y(best.val_sharpe) - 10} textAnchor="middle" fill="var(--fg)" fontSize={11}>kept ({num(best.val_sharpe)})</text>
          <text x={width - M.r + 8} y={g.y(lastT.sharpe) + 4} fill="var(--fg)" fontSize={12}>Training</text>
          <text x={width - M.r + 8} y={g.y(lastV.val_sharpe) + 4} fill="var(--fg)" fontSize={12}>Validation</text>
        </svg>
      </div>
      <p className="muted" style={{ fontSize: ".8rem", marginTop: 8 }}>Sharpe ratio. Training line: mean over recent one-year training episodes. Faint lines: validation for the other seeds.</p>
    </section>
  );
}

function CheckpointReplay({ rp }: { rp: Replays }) {
  const cps = useMemo(() => rp.checkpoints.filter((c, k, a) => a.findIndex((d) => d.steps === c.steps) === k), [rp]);
  const [k, setK] = useState(cps.length - 1);
  const cp = cps[k];
  const trace = useMemo(() => ({
    ticker: rp.ticker, dates: rp.dates, close: rp.close, bnh: rp.bnh, equity: cp.value, allocation: cp.allocation,
  }) as unknown as Trace, [rp, cp]);
  return (
    <section className="card">
      <div className="card-head">
        <h2>The same year, replayed by the agent at each stage of training</h2>
        <span className="mono muted" style={{ fontSize: ".82rem" }}>{rp.ticker} · {rp.dates[0].slice(0, 4)} bear market</span>
      </div>
      <div className="controls" style={{ marginBottom: 12 }}>
        <span className="eyebrow">After</span>
        <input className="scrub" type="range" min={0} max={cps.length - 1} value={k} onChange={(e) => setK(+e.target.value)} aria-label="Training checkpoint" />
        <span className="mono" style={{ minWidth: 150 }}>{cp.steps.toLocaleString()} steps</span>
      </div>
      <div className="stats" style={{ marginBottom: 12 }}>
        <div className="stat"><div className="eyebrow">Sharpe</div><div className="v">{num(cp.sharpe)}</div></div>
        <div className="stat"><div className="eyebrow">Return</div><div className="v">{pct(cp.total_return)}</div><div className="vs">buy &amp; hold {pct(rp.bnh[rp.bnh.length - 1] / rp.bnh[0] - 1)}</div></div>
        <div className="stat"><div className="eyebrow">Worst drawdown</div><div className="v">{pct(cp.max_drawdown)}</div></div>
        <div className="stat"><div className="eyebrow">Checkpoint</div><div className="v">{k + 1}/{cps.length}</div></div>
      </div>
      <ReplayChart trace={trace} index={rp.dates.length - 1} onIndex={() => {}} />
    </section>
  );
}
