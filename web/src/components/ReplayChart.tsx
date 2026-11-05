import { useMemo, useRef, useState } from "react";
import type { Trace } from "../lib/data";
import { fmtDate, pct, pctAbs } from "../lib/data";
import { areaPath, extent, linePath, linear, monthLabel, monthTicks, pad, ticks, useWidth } from "../lib/chart";

const M = { l: 52, r: 86, t: 10 };
const EQ_H = 250, GAP = 26, AL_H = 84, AXIS = 22;

/** Equity (agent vs buy & hold) over the allocation band, sharing one time axis.
 *  The agent's line is revealed up to `index`; drag or click to scrub. */
export default function ReplayChart({ trace, index, onIndex }: {
  trace: Trace; index: number; onIndex: (i: number) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const dragging = useRef(false);
  const n = trace.dates.length;
  const H = M.t + EQ_H + GAP + AL_H + AXIS;
  const alTop = M.t + EQ_H + GAP;

  const geo = useMemo(() => {
    const x = linear([0, n - 1], [M.l, width - M.r]);
    const yDom = pad(extent(trace.equity, trace.bnh));
    const y = linear(yDom, [M.t + EQ_H, M.t]);
    const ya = linear([0, 1], [alTop + AL_H, alTop]);
    const xs = trace.dates.map((_, i) => x(i));
    // allocation[i] is held from day i to day i+1: draw it as a step
    const ax: number[] = [], ay: number[] = [];
    trace.allocation.forEach((a, i) => { ax.push(x(i), x(i + 1)); ay.push(ya(a), ya(a)); });
    return {
      x, y, ya, xs, yTicks: ticks(yDom, 4), mTicks: monthTicks(trace.dates, width < 560 ? 4 : 8),
      bnh: linePath(xs, trace.bnh.map(y)), agent: linePath(xs, trace.equity.map(y)),
      alloc: areaPath(ax, ay, ya(0)), allocLine: linePath(ax, ay),
    };
  }, [trace, width, n, alTop]);

  const cut = geo.xs[index];
  const toIndex = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.round(geo.x.invert(clientX - r.left))));
  };
  const shown = hover ?? index;
  const tipLeft = Math.min(geo.xs[shown] + 12, width - 190);
  const lastAgent = trace.equity[n - 1], lastBnh = trace.bnh[n - 1];
  const labelY = (v: number) => geo.y(v);
  let la = labelY(lastAgent), lb = labelY(lastBnh);
  if (Math.abs(la - lb) < 16) { const mid = (la + lb) / 2; la = mid + (lastAgent >= lastBnh ? -8 : 8); lb = mid + (lastAgent >= lastBnh ? 8 : -8); }

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={H} role="img"
        aria-label={`${trace.ticker}: agent equity versus buy and hold, with daily stock allocation`}
        style={{ touchAction: "pan-y", cursor: "crosshair" }}
        onPointerDown={(e) => { dragging.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); onIndex(toIndex(e.clientX)); }}
        onPointerMove={(e) => { const i = toIndex(e.clientX); setHover(i); if (dragging.current) onIndex(i); }}
        onPointerUp={() => { dragging.current = false; }}
        onPointerLeave={() => { setHover(null); dragging.current = false; }}>
        <g className="axis">
          {geo.yTicks.map((t) => (
            <g key={t}>
              <line className="gridline" x1={M.l} x2={width - M.r} y1={geo.y(t)} y2={geo.y(t)} />
              <text x={M.l - 8} y={geo.y(t) + 3.5} textAnchor="end">${(t / 1000).toFixed(t % 1000 ? 1 : 0)}k</text>
            </g>
          ))}
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line className="gridline" x1={M.l} x2={width - M.r} y1={geo.ya(t)} y2={geo.ya(t)} />
              <text x={M.l - 8} y={geo.ya(t) + 3.5} textAnchor="end">{t * 100}%</text>
            </g>
          ))}
          {geo.mTicks.map((i) => (
            <text key={i} x={geo.xs[i]} y={H - 6} textAnchor="middle">{monthLabel(trace.dates[i])}</text>
          ))}
          <text x={M.l} y={alTop - 8} style={{ textTransform: "uppercase", letterSpacing: ".06em" }}>In stock</text>
        </g>

        <defs>
          <clipPath id="past"><rect x={0} y={0} width={cut} height={H} /></clipPath>
          <clipPath id="future"><rect x={cut} y={0} width={Math.max(0, width - cut)} height={H} /></clipPath>
        </defs>

        <path d={geo.bnh} fill="none" stroke="var(--s-bench)" strokeWidth={2} strokeLinejoin="round" />
        <g clipPath="url(#future)" opacity={0.22}>
          <path d={geo.agent} fill="none" stroke="var(--s-agent)" strokeWidth={2} />
          <path d={geo.alloc} fill="var(--s-agent)" opacity={0.35} />
        </g>
        <g clipPath="url(#past)">
          <path d={geo.agent} fill="none" stroke="var(--s-agent)" strokeWidth={2} strokeLinejoin="round" />
          <path d={geo.alloc} fill="var(--s-agent)" opacity={0.35} />
          <path d={geo.allocLine} fill="none" stroke="var(--s-agent)" strokeWidth={1.5} />
        </g>

        {/* direct labels */}
        <text x={width - M.r + 8} y={lb + 4} fill="var(--fg)" fontSize={12}>Buy &amp; hold</text>
        <text x={width - M.r + 8} y={la + 4} fill="var(--fg)" fontSize={12} fontWeight={600}>Agent</text>

        {/* playhead */}
        <line x1={cut} x2={cut} y1={M.t} y2={alTop + AL_H} stroke="var(--fg)" strokeOpacity={0.5} strokeWidth={1} />
        <circle cx={cut} cy={geo.y(trace.equity[index])} r={4.5} fill="var(--s-agent)" stroke="var(--surface)" strokeWidth={2} />
        <circle cx={cut} cy={geo.y(trace.bnh[index])} r={4} fill="var(--s-bench)" stroke="var(--surface)" strokeWidth={2} />
        {hover !== null && hover !== index && (
          <line x1={geo.xs[hover]} x2={geo.xs[hover]} y1={M.t} y2={alTop + AL_H} stroke="var(--muted)" strokeDasharray="3 3" />
        )}
      </svg>

      {hover !== null && (
        <div className="tooltip" style={{ left: tipLeft, top: 8 }}>
          <div className="eyebrow" style={{ marginBottom: 4 }}>{fmtDate(trace.dates[shown])}</div>
          <div className="row"><span>Agent</span><span>{pct(trace.equity[shown] / trace.equity[0] - 1)}</span></div>
          <div className="row"><span>Buy &amp; hold</span><span>{pct(trace.bnh[shown] / trace.bnh[0] - 1)}</span></div>
          {shown < trace.allocation.length && (
            <div className="row"><span>In stock</span><span>{pctAbs(trace.allocation[shown])}</span></div>
          )}
          <div className="row muted"><span>Price</span><span>${trace.close[shown].toFixed(2)}</span></div>
        </div>
      )}
    </div>
  );
}
