import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import ReplayChart from "../components/ReplayChart";
import type { Manifest, Trace } from "../lib/data";
import { OBS_LABELS, fmtDate, num, pct, pctAbs, useData } from "../lib/data";

const HERO = "JPM";
const DAYS_PER_SEC = 24;

const GROUPS: [string, string[]][] = [
  ["Its own account", ["position", "drawdown", "return so far"]],
  ["Price & momentum", ["log_return", "sma_10", "sma_30", "volatility", "rsi", "macd"]],
  ["Volume & range", ["volume_ratio", "atr", "52w_high"]],
  ["Calendar & macro", ["day_sin", "day_cos", "month_sin", "vix", "10y_rate"]],
];

const EXHIBITS = [
  { to: "/live", n: "00", title: "Live desk", tag: "Every weekday", body: "Tomorrow's allocation for 16 tickers, from a frozen agent and one that keeps learning, logged before the market opens." },
  { to: `/replay/${HERO}`, n: "01", title: "Full replay", tag: "17 stocks", body: "Every out-of-sample day for any stock: what it held, which inputs drove it, where its memory was." },
  { to: "/lab", n: "02", title: "Policy lab", tag: "Live model", body: "The trained network runs in your browser. Set VIX, trend and volatility and watch its answer change." },
  { to: "/scorecard", n: "03", title: "Scorecard", tag: "5 seeds", body: "Ten stocks it never saw, against buy & hold, a moving-average rule and random agents." },
  { to: "/learning", n: "04", title: "How it learned", tag: "Overfitting", body: "Training vs validation over a million steps, and one bear market replayed at every stage of training." },
];

export default function Home() {
  const { data: m } = useData<Manifest>("manifest.json");
  const { data: t } = useData<Trace>(`traces/${HERO}.json`);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!t || !playing) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setI(t.dates.length - 1); setPlaying(false); return; }
    const id = setInterval(() => setI((x) => (x + 1) % t.dates.length), 1000 / DAYS_PER_SEC);
    return () => clearInterval(id);
  }, [t, playing]);

  const s = m?.scorecard?.summary.held_out;
  const d = t ? Math.min(i, t.allocation.length - 1) : 0;
  const drivers = t && m ? m.obs_names.map((n, k) => ({ n, a: t.attribution[d][k] }))
    .sort((p, q) => Math.abs(q.a) - Math.abs(p.a)).slice(0, 4) : [];
  const maxAbs = Math.max(0.02, ...drivers.map((x) => Math.abs(x.a)));

  return (
    <>
      <header className="hero wrap">
        <div className="hero-top">
          <div>
            <div className="hero-eyebrow eyebrow"><span>PPO</span><span>LSTM memory</span><span>17 inputs</span><span>Daily bars</span></div>
            <h1>A trading agent that <em>shows its work.</em></h1>
          </div>
          <div className="hero-copy">
            <p>RL Trader learned how much stock to hold from eleven years of market data. This site replays its decisions on days and stocks it never trained on, shows what drove each one, and reports honestly how it did.</p>
            <div className="controls">
              <Link className="btn primary lg" to="/live">Tonight's live calls</Link>
              <Link className="btn lg" to={`/replay/${HERO}`}>Open the full replay</Link>
            </div>
          </div>
        </div>

        <div className="instrument">
          <div className="inst-main">
            <div className="card-head" style={{ marginBottom: 6 }}>
              <span className="inst-title">{HERO} · 2023–24 · never seen in training</span>
              <div className="legend">
                <span><i className="swatch" style={{ background: "var(--s-agent)" }} />Agent</span>
                <span><i className="swatch" style={{ background: "var(--s-bench)" }} />Buy &amp; hold</span>
              </div>
            </div>
            {t ? <ReplayChart trace={t} index={i} onIndex={(x) => { setPlaying(false); setI(x); }} /> : <div className="skeleton" />}
            <div className="controls" style={{ marginTop: 8 }}>
              <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
              <span className="muted" style={{ fontSize: ".78rem" }}>Drag across the chart to scrub</span>
            </div>
          </div>
          <aside className="readout" aria-live="off">
            {t && m ? (
              <>
                <div><div className="eyebrow">Decision at close of</div><div className="mono" style={{ fontSize: "1.02rem" }}>{fmtDate(t.dates[d])}</div></div>
                <div>
                  <div className="ro-alloc"><b>{pctAbs(t.allocation[d])}</b><span>in {HERO}</span></div>
                  <div className="meter" style={{ marginTop: 10 }}><i style={{ width: `${t.allocation[d] * 100}%` }} /></div>
                </div>
                <dl className="kv">
                  <dt>Close</dt><dd>${t.close[d].toFixed(2)}</dd>
                  <dt>Agent</dt><dd>{pct(t.equity[d] / t.equity[0] - 1)}</dd>
                  <dt>Buy &amp; hold</dt><dd>{pct(t.bnh[d] / t.bnh[0] - 1)}</dd>
                </dl>
                <div>
                  <div className="eyebrow" style={{ marginBottom: 8 }}>Biggest influences</div>
                  {drivers.map((x) => {
                    const w = (Math.abs(x.a) / maxAbs) * 50;
                    return (
                      <div className="why-row" key={x.n}>
                        <label>{OBS_LABELS[x.n] ?? x.n}</label>
                        <span className="attr-track"><span className="attr-bar" style={{ left: x.a >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: x.a >= 0 ? "var(--up)" : "var(--down)" }} /></span>
                      </div>
                    );
                  })}
                  <div className="why-scale"><span>◀ cash</span><span>stock ▶</span></div>
                </div>
              </>
            ) : <div className="skeleton" style={{ minHeight: 200 }} />}
          </aside>
        </div>

        {s && (
          <>
            <div className="strip">
              <div><div className="v">{num(s.agent.sharpe.mean)} <small>vs {num(s.buy_and_hold.sharpe.mean)}</small></div><div className="l">Sharpe, agent vs buy &amp; hold</div></div>
              <div><div className="v">{pct(s.agent.max_drawdown.mean, 0)} <small>vs {pct(s.buy_and_hold.max_drawdown.mean, 0)}</small></div><div className="l">Worst drawdown</div></div>
              <div><div className="v">{pct(s.agent.total_return.mean, 0)} <small>vs {pct(s.buy_and_hold.total_return.mean, 0)}</small></div><div className="l">Two-year return</div></div>
              <div><div className="v">{pctAbs(s.agent_beats_bnh_sharpe_frac)}</div><div className="l">Stocks where it beat B&amp;H</div></div>
              <div><div className="v">5 × 10</div><div className="l">Seeds × unseen stocks</div></div>
            </div>
            <p className="muted" style={{ fontSize: ".8rem", marginTop: 10 }}>
              Averages over five seeds on ten never-seen stocks, 2023–24. Short version: it cuts risk but doesn't beat the market. <Link to="/scorecard">Full scorecard →</Link>
            </p>
          </>
        )}
      </header>

      <section className="block wrap">
        <div className="sec-head">
          <h2>How it decides, once per trading day</h2>
          <p>No hand-written rules. The agent sees a snapshot of the market and its own account, carries memory of previous days, and picks how much of the portfolio to hold.</p>
        </div>
        <div className="pipeline">
          <div className="stage">
            <span className="n">01 · Observe</span>
            <h3>17 numbers about today</h3>
            <div className="groups">
              {GROUPS.map(([g, fs]) => (
                <div className="group" key={g}><b>{g}</b>{fs.map((f) => <span className="feat" key={f}>{f}</span>)}</div>
              ))}
            </div>
          </div>
          <div className="stage">
            <span className="n">02 · Decide</span>
            <h3>Memory, then a target allocation</h3>
            <p>An LSTM keeps a running summary of recent days, so one quiet session after a selloff doesn't look like a calm market.</p>
            <div className="net"><span>17 inputs</span><em>→</em><span>Linear 128</span><em>→</em><span className="hl">LSTM 128</span><em>→</em><span>0–100% in stock</span></div>
            <p>A second head estimates how good the situation is; PPO learns from the gap between that estimate and what happened.</p>
          </div>
          <div className="stage">
            <span className="n">03 · Learn</span>
            <h3>Rewarded for growth, charged for risk</h3>
            <div className="formula">r<sub>t</sub> = ℓ<sub>t</sub> − ½ · <span className="c">κ</span> · ℓ<sub>t</sub>²<br /><span className="muted">ℓ<sub>t</sub> = log(V<sub>t</sub> / V<sub>t−1</sub>), κ = 5</span></div>
            <p>Trained on AAPL, MSFT, GOOGL, TSLA, SPY, NVDA and AMZN, 2010–2020. The checkpoint was picked on 2021–22 and tested on 2023–24. Every trade pays 0.1% plus slippage.</p>
          </div>
        </div>
      </section>

      <section className="block wrap">
        <div className="sec-head">
          <h2>Five ways to look inside</h2>
          <p>Everything here is generated from the trained agent and runs as a static site. The lab runs the network itself, in your browser.</p>
        </div>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
          {EXHIBITS.map((e, k) => (
            <motion.div key={e.to} initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.06 * k, duration: 0.4 }}>
              <Link to={e.to} className="card exhibit">
                <div className="card-head" style={{ marginBottom: 0 }}><span className="eyebrow">{e.n}</span><span className="tag">{e.tag}</span></div>
                <h3>{e.title}</h3>
                <p className="muted">{e.body}</p>
                <span className="go">Open →</span>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>
    </>
  );
}
