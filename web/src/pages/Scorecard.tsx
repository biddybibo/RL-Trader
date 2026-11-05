import { Link } from "react-router-dom";
import type { GroupSummary, Manifest, TickerIndex } from "../lib/data";
import { num, pct, pctAbs, useData } from "../lib/data";
import { linear, useWidth } from "../lib/chart";

export default function Scorecard() {
  const { data: m } = useData<Manifest>("manifest.json");
  const sc = m?.scorecard;
  return (
    <div className="wrap">
      <header className="page-head">
        <span className="eyebrow">Scorecard · {sc ? `${sc.period[0]} → ${sc.period[1]}` : "…"}</span>
        <h1>Did it beat buying and holding? No.</h1>
        <p>Scored on two years it never trained on or was selected on, across {sc?.models.length ?? 5} independently trained seeds, against buy &amp; hold, a 50/200-day moving-average crossover and random agents.</p>
      </header>

      {!m && <div className="skeleton" />}
      {sc && (
        <div className="grid">
          <div className="callout">
            <p><strong>The honest result:</strong> on stocks it never saw, the agent's Sharpe ratio is about the same as buy &amp; hold's, or a little worse. It roughly halves the worst drawdown, but gives up most of the return to do it.</p>
            <p className="muted">Twenty seeds across three reward designs told the same story, and an early single-seed run that looked like a win didn't replicate. See <Link to="/learning">how it learned</Link> for why: it fits the training years far better than it generalises.</p>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 460px), 1fr))" }}>
            <SummaryCard title="Never-seen stocks" s={sc.summary.held_out} />
            <SummaryCard title="Training stocks, future years" s={sc.summary.train_tickers} />
          </div>
          <section className="card">
            <div className="card-head">
              <h2>Per stock, exported model</h2>
              <div className="legend">
                <span><i className="swatch block" style={{ background: "var(--s-agent)" }} />Agent</span>
                <span><i className="swatch block" style={{ background: "var(--s-bench)" }} />Buy &amp; hold</span>
              </div>
            </div>
            <PerTicker rows={m.tickers} />
            <p className="muted" style={{ fontSize: ".8rem", marginTop: 10 }}>Model {m.model.source}, chosen by its 2021–22 validation Sharpe. Click a stock to replay it.</p>
          </section>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ title, s }: { title: string; s: GroupSummary }) {
  const rows: [string, GroupSummary["agent"]][] = [["Agent (mean ± sd over seeds)", s.agent], ["Buy & hold", s.buy_and_hold], ["SMA 50/200", s.sma_50_200]];
  return (
    <section className="card">
      <div className="card-head"><h2>{title}</h2>
        <span className="mono muted" style={{ fontSize: ".82rem" }}>agent beats B&amp;H on {pctAbs(s.agent_beats_bnh_sharpe_frac)}</span></div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Strategy</th><th>Sharpe</th><th>Worst DD</th><th>Return</th></tr></thead>
          <tbody>
            {rows.map(([name, r], k) => (
              <tr key={name}>
                <td style={{ fontFamily: "var(--font-body)" }}>{name}</td>
                <td>{num(r.sharpe.mean)}{k === 0 && <span className="muted"> ±{r.sharpe.std.toFixed(2)}</span>}</td>
                <td>{pct(r.max_drawdown.mean, 0)}</td>
                <td>{pct(r.total_return.mean, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PerTicker({ rows }: { rows: TickerIndex[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const lo = Math.min(0, ...rows.flatMap((r) => [r.sharpe, r.bnh_sharpe]));
  const hi = Math.max(...rows.flatMap((r) => [r.sharpe, r.bnh_sharpe]));
  const L = 64, x = linear([lo, hi], [L, width - 8]), RH = 26;
  const sorted = [...rows].sort((a, b) => (a.group === b.group ? b.sharpe - b.bnh_sharpe - (a.sharpe - a.bnh_sharpe) : a.group === "held_out" ? -1 : 1));
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={sorted.length * RH + 24} role="img" aria-label="Sharpe ratio per stock, agent versus buy and hold">
        <g className="axis">
          {[...new Set([Math.ceil(lo), 0, 1, 2, Math.floor(hi)])].filter((t) => t >= lo && t <= hi).map((t) => (
            <g key={t}><line className="gridline" x1={x(t)} x2={x(t)} y1={0} y2={sorted.length * RH} />
              <text x={x(t)} y={sorted.length * RH + 16} textAnchor="middle">{t}</text></g>
          ))}
        </g>
        {sorted.map((r, k) => {
          const y = k * RH;
          return (
            <Link to={`/replay/${r.ticker}`} key={r.ticker}>
              <g style={{ cursor: "pointer" }}>
                <title>{`${r.ticker}: Sharpe ${r.sharpe.toFixed(2)} vs ${r.bnh_sharpe.toFixed(2)}; worst drawdown ${pct(r.max_drawdown, 0)} vs ${pct(r.bnh_max_drawdown, 0)}`}</title>
                <rect x={0} y={y} width={width} height={RH} fill="transparent" />
                <text x={0} y={y + 16} fill="var(--fg)" fontSize={12} fontFamily="var(--font-data)">{r.ticker}</text>
                <text x={36} y={y + 16} fill="var(--muted)" fontSize={9.5} fontFamily="var(--font-data)">{r.group === "held_out" ? "new" : "train"}</text>
                <rect x={Math.min(x(0), x(r.sharpe))} y={y + 5} width={Math.abs(x(r.sharpe) - x(0))} height={7} rx={2} fill="var(--s-agent)" />
                <rect x={Math.min(x(0), x(r.bnh_sharpe))} y={y + 14} width={Math.abs(x(r.bnh_sharpe) - x(0))} height={7} rx={2} fill="var(--s-bench)" />
              </g>
            </Link>
          );
        })}
      </svg>
    </div>
  );
}
