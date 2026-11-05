import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { OBS_LABELS, fmtDate, pct, pctAbs, useData } from "../lib/data";
import { extent, linePath, linear, pad, ticks, useWidth } from "../lib/chart";

type Call = {
  ticker: string; as_of: string; close: number; day_return: number; allocation: number; previous: number;
  drivers: { input: string; effect: number }[]; spark: number[];
};
type AgentCalls = { model: string; trained_through?: string; updated?: string; calls: Call[] };
type Today = { generated: string; anchor: string; agents: Record<string, AgentCalls> };
type Point = { date: string; agent: number; bnh: number; n?: number };
type Record_ = Record<string, { equal_weight: Point[]; first_call: string; settled_days: number }>;
type LedgerRow = { agent: string; ticker: string; as_of: string; close: number; allocation: number; published: string };

const AGENT_META: Record<string, { label: string; color: string; blurb: string }> = {
  frozen: { label: "Frozen", color: "var(--s-agent)", blurb: "Trained on 2010–2020, chosen on 2021–22, never updated. The agent from the replay and scorecard." },
  adaptive: { label: "Adaptive", color: "var(--s-adapt)", blurb: "Same agent, but it keeps learning: retrained every month on the latest ten years and picked on the most recent one." },
};

export default function Live() {
  const { data: today, error } = useData<Today>("live/today.json");
  const { data: record } = useData<Record_>("live/record.json");
  const { data: ledger } = useData<LedgerRow[]>("live/ledger.json");
  const agents = today ? Object.keys(today.agents) : [];
  const asOf = today?.agents.frozen?.calls[0]?.as_of;

  const rows = useMemo(() => {
    if (!today) return [];
    const base = today.agents[agents[0]]?.calls ?? [];
    return base.map((c) => ({ ticker: c.ticker, close: c.close, day: c.day_return,
      by: Object.fromEntries(agents.map((a) => [a, today.agents[a].calls.find((x) => x.ticker === c.ticker)])) as Record<string, Call | undefined> }));
  }, [today, agents]);

  return (
    <div className="wrap">
      <header className="page-head">
        <span className="eyebrow"><span className="live-dot" /> Live desk · {asOf ? `calls from the close of ${fmtDate(asOf)}` : "…"}</span>
        <h1>What the agent would hold tomorrow</h1>
        <p>Every weekday after the US close, both agents read the latest market data and publish an allocation for the next session. Calls are logged before the outcome is known and never edited, so the record below can't be overfit.</p>
      </header>

      {error && <div className="card">Live data isn't available yet: {error}</div>}
      {today && (
        <div className="grid">
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 380px), 1fr))" }}>
            {agents.map((a) => (
              <div className="card" key={a} style={{ display: "grid", gap: 6 }}>
                <div className="legend"><span><i className="swatch" style={{ background: AGENT_META[a]?.color, width: 18 }} /><strong style={{ color: "var(--fg)" }}>{AGENT_META[a]?.label ?? a} agent</strong></span></div>
                <p className="muted" style={{ fontSize: ".9rem" }}>{AGENT_META[a]?.blurb}</p>
                <p className="mono muted" style={{ fontSize: ".76rem" }}>
                  trained through {today.agents[a].trained_through ?? "—"} · model updated {today.agents[a].updated ?? "—"}
                </p>
              </div>
            ))}
          </div>

          <section className="card">
            <div className="card-head">
              <h2>Tonight's calls</h2>
              <span className="mono muted" style={{ fontSize: ".78rem" }}>share of each position held in stock for the next session · published {today.generated.replace("T", " ").slice(0, 16)} UTC</span>
            </div>
            <div className="calls">
              {rows.map((r) => <CallCard key={r.ticker} r={r} agents={agents} />)}
            </div>
          </section>

          <RecordCard record={record} />

          <section className="card">
            <div className="card-head">
              <h2>The ledger</h2>
              <a className="mono" href={`${import.meta.env.BASE_URL}data/live/ledger.json`} style={{ fontSize: ".8rem", color: "var(--agent)" }}>Download all {ledger?.length ?? ""} calls (JSON)</a>
            </div>
            <div className="table-scroll" style={{ maxHeight: 360, overflowY: "auto" }}>
              <table>
                <thead><tr><th>Close of</th><th>Agent</th><th>Ticker</th><th>In stock</th><th>Close</th><th>Published (UTC)</th></tr></thead>
                <tbody>
                  {ledger && [...ledger].reverse().slice(0, 120).map((e) => (
                    <tr key={`${e.agent}${e.ticker}${e.as_of}`}>
                      <td>{e.as_of}</td>
                      <td style={{ fontFamily: "var(--font-body)" }}>{AGENT_META[e.agent]?.label ?? e.agent}</td>
                      <td>{e.ticker}</td><td>{pctAbs(e.allocation)}</td><td>{e.close.toFixed(2)}</td>
                      <td className="muted">{e.published.replace("T", " ").slice(0, 16)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="callout">
            <p>A forward test, not advice. The <Link to="/scorecard">backtest scorecard</Link> says this agent cuts risk but hasn't beaten buy &amp; hold; the live record is where that gets tested for real. The site is rebuilt automatically each weekday evening.</p>
          </div>
        </div>
      )}
      {!today && !error && <div className="skeleton" />}
    </div>
  );
}

function CallCard({ r, agents }: { r: { ticker: string; close: number; day: number; by: Record<string, Call | undefined> }; agents: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <button className="call" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
      <div className="call-head">
        <span className="mono" style={{ fontWeight: 500 }}>{r.ticker}</span>
        <span className="mono muted" style={{ fontSize: ".78rem" }}>{r.close.toFixed(2)} <span className={r.day >= 0 ? "pos" : "neg"}>{pct(r.day)}</span></span>
      </div>
      {agents.map((a) => {
        const c = r.by[a];
        if (!c) return null;
        const d = c.allocation - c.previous;
        return (
          <div key={a} className="call-row">
            <span className="muted" style={{ fontSize: ".72rem" }}>{AGENT_META[a]?.label ?? a}</span>
            <span className="meter"><i style={{ width: `${c.allocation * 100}%`, background: AGENT_META[a]?.color }} /></span>
            <span className="mono" style={{ fontSize: ".85rem", textAlign: "right" }}>{pctAbs(c.allocation)}</span>
            <span className={`mono ${Math.abs(d) < 0.005 ? "muted" : d > 0 ? "pos" : "neg"}`} style={{ fontSize: ".72rem", textAlign: "right" }}>
              {Math.abs(d) < 0.005 ? "=" : `${d > 0 ? "▲" : "▼"}${Math.abs(d * 100).toFixed(0)}`}
            </span>
          </div>
        );
      })}
      {open && agents.map((a) => r.by[a] && (
        <div key={a} className="call-drivers">
          <span className="eyebrow">{AGENT_META[a]?.label} · why</span>
          {r.by[a]!.drivers.slice(0, 3).map((x) => (
            <span key={x.input} className="muted" style={{ fontSize: ".78rem" }}>
              <span className={x.effect >= 0 ? "pos" : "neg"}>{x.effect >= 0 ? "▶ stock" : "◀ cash"}</span> {OBS_LABELS[x.input] ?? x.input} {Math.abs(x.effect * 100).toFixed(1)}pt
            </span>
          ))}
        </div>
      ))}
    </button>
  );
}

function RecordCard({ record }: { record?: Record_ }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const series = record ? Object.entries(record) : [];
  const settled = series.length ? Math.max(...series.map(([, v]) => v.settled_days)) : 0;
  const first = series.length ? series.map(([, v]) => v.first_call).sort()[0] : undefined;
  const H = 240, M = { l: 44, r: 92, t: 10, b: 22 };
  const g = useMemo(() => {
    if (!settled) return null;
    const dates = [...new Set(series.flatMap(([, v]) => v.equal_weight.map((p) => p.date)))].sort();
    const all = ["", ...dates];                                            // "" = day of first call, value 1
    const x = linear([0, all.length - 1], [M.l, width - M.r]);
    const val = (pts: Point[], k: "agent" | "bnh") => all.map((d) => (d === "" ? 1 : pts.find((p) => p.date === d)?.[k] ?? NaN));
    const lines = series.map(([a, v]) => ({ a, ys: val(v.equal_weight, "agent") }));
    const bnh = val(series[0][1].equal_weight, "bnh");
    const ydom = pad(extent(...lines.map((l) => l.ys.filter(Number.isFinite)), bnh.filter(Number.isFinite)));
    const y = linear(ydom, [H - M.b, M.t]);
    return { x, y, all, lines, bnh, yt: ticks(ydom, 4) };
  }, [record, width, settled]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="card">
      <div className="card-head">
        <h2>Forward record</h2>
        <span className="mono muted" style={{ fontSize: ".78rem" }}>{settled} settled session{settled === 1 ? "" : "s"} · equal-weight across the watch-list · after costs</span>
      </div>
      <div ref={ref} className="chart">
        {!g ? (
          <div className="empty">
            <p><strong>The clock started {first ? fmtDate(first) : "today"}.</strong> Each call settles at the next close, so the first point appears after tomorrow's session. Check back: this line can only be drawn one day at a time.</p>
          </div>
        ) : (
          <svg width={width} height={H} role="img" aria-label="Forward paper record of the agents versus buy and hold">
            <g className="axis">
              {g.yt.map((t) => <g key={t}><line className="gridline" x1={M.l} x2={width - M.r} y1={g.y(t)} y2={g.y(t)} />
                <text x={M.l - 8} y={g.y(t) + 3.5} textAnchor="end">{pct(t - 1, 1)}</text></g>)}
              <text x={M.l} y={H - 4}>{first}</text>
              <text x={width - M.r} y={H - 4} textAnchor="end">{g.all[g.all.length - 1]}</text>
            </g>
            <path d={linePath(g.all.map((_, i) => g.x(i)), g.bnh.map(g.y))} fill="none" stroke="var(--s-bench)" strokeWidth={2} />
            {g.lines.map((l) => <path key={l.a} d={linePath(g.all.map((_, i) => g.x(i)), l.ys.map(g.y))} fill="none" stroke={AGENT_META[l.a]?.color} strokeWidth={2} />)}
            {[...g.lines.map((l) => ({ label: AGENT_META[l.a]?.label ?? l.a, v: l.ys[l.ys.length - 1] })), { label: "Buy & hold", v: g.bnh[g.bnh.length - 1] }]
              .map((d) => <text key={d.label} x={width - M.r + 8} y={g.y(d.v) + 4} fill="var(--fg)" fontSize={12}>{d.label} {pct(d.v - 1, 1)}</text>)}
          </svg>
        )}
      </div>
    </section>
  );
}
