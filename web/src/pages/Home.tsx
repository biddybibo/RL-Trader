import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const EXHIBITS = [
  { to: "/replay/SPY", eyebrow: "01 · Replay", title: "Watch it trade", body: "Two out-of-sample years, day by day: what it held, which inputs drove each decision, and where its memory was." },
  { to: "/scorecard", eyebrow: "02 · Scorecard", title: "Did it beat the market?", body: "Five seeds, ten stocks it never saw, three baselines. The answer is honest, and it's no." },
  { to: "/learning", eyebrow: "03 · Learning", title: "How it learned", body: "Training vs validation over a million steps, and the same bear market replayed by the agent at every stage." },
];

export default function Home() {
  return (
    <div className="wrap">
      <section style={{ padding: "40px 0 36px", maxWidth: 820 }}>
        <span className="eyebrow">Recurrent PPO · 17 inputs · one decision a day</span>
        <h1 style={{ fontSize: "clamp(2.4rem, 5.4vw, 4.2rem)", lineHeight: 1.04, margin: "14px 0 18px" }}>
          An RL trading agent, <span style={{ color: "var(--agent)" }}>taken apart</span> in public.
        </h1>
        <p className="muted" style={{ fontSize: "1.12rem", maxWidth: 640 }}>
          A PPO agent with LSTM memory decides each day how much of its portfolio to hold in stock. Here you can replay every
          decision, see what drove it, and judge the results against the honest baselines.
        </p>
        <div className="controls" style={{ marginTop: 24 }}>
          <Link to="/replay/SPY" className="btn primary" style={{ textDecoration: "none" }}>▶ Watch the replay</Link>
          <Link to="/scorecard" className="btn" style={{ textDecoration: "none" }}>See the scorecard</Link>
        </div>
      </section>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 280px), 1fr))" }}>
        {EXHIBITS.map((e, k) => (
          <motion.div key={e.to} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06 * k, duration: 0.4 }}>
            <Link to={e.to} className="card" style={{ display: "grid", gap: 8, textDecoration: "none", height: "100%" }}>
              <span className="eyebrow">{e.eyebrow}</span>
              <h2 style={{ fontSize: "1.3rem" }}>{e.title}</h2>
              <p className="muted">{e.body}</p>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
