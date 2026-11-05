import { motion } from "framer-motion";
import { OBS_LABELS, pctAbs } from "../lib/data";

/** Integrated-gradient attribution for one decision: which inputs pushed the
 *  allocation toward stock (right) or toward cash (left), holding memory fixed. */
export default function Attribution({ names, attr, obs, baseline, allocation, top = 8 }: {
  names: string[]; attr: number[]; obs: number[]; baseline: number; allocation: number; top?: number;
}) {
  const rows = names.map((n, k) => ({ n, a: attr[k], o: obs[k] }))
    .sort((p, q) => Math.abs(q.a) - Math.abs(p.a)).slice(0, top);
  const maxAbs = Math.max(0.02, ...rows.map((r) => Math.abs(r.a)));
  const rest = attr.reduce((s, v) => s + v, 0) - rows.reduce((s, r) => s + r.a, 0);

  return (
    <div>
      <div className="legend" style={{ marginBottom: 10 }}>
        <span><i className="swatch block" style={{ background: "var(--down)" }} />◀ toward cash</span>
        <span><i className="swatch block" style={{ background: "var(--up)" }} />toward stock ▶</span>
      </div>
      {rows.map((r) => {
        const w = (Math.abs(r.a) / maxAbs) * 50;
        return (
          <div className="attr-row" key={r.n} title={`input value (scaled): ${r.o.toFixed(2)}`}>
            <span className="name">{OBS_LABELS[r.n] ?? r.n}</span>
            <span className="attr-track">
              <motion.span className="attr-bar" initial={false}
                animate={{ left: r.a >= 0 ? "50%" : `${50 - w}%`, width: `${w}%` }}
                transition={{ type: "spring", stiffness: 380, damping: 34 }}
                style={{ background: r.a >= 0 ? "var(--up)" : "var(--down)" }} />
            </span>
            <span className="val">{r.a >= 0 ? "+" : "−"}{Math.abs(r.a * 100).toFixed(1)}pt</span>
          </div>
        );
      })}
      <p className="muted" style={{ fontSize: ".8rem", marginTop: 10 }}>
        Starting point (neutral market) {pctAbs(baseline)} · top inputs shown, others net {rest >= 0 ? "+" : "−"}
        {Math.abs(rest * 100).toFixed(1)}pt · decision {pctAbs(allocation)} in stock.
      </p>
    </div>
  );
}
