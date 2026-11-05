import { useMemo, useState } from "react";
import { extent, linePath, linear, pad, useWidth } from "../lib/chart";
import { fmtDate, pctAbs } from "../lib/data";

const H = 280, P = 14, TRAIL = 25;

/** The LSTM's 128-d memory after each day, projected to 2-D (PCA). Dots are
 *  shaded by how much stock the agent held; the trail shows the last few weeks. */
export default function MemoryMap({ xy, allocation, dates, index, onIndex }: {
  xy: [number, number][]; allocation: number[]; dates: string[]; index: number; onIndex: (i: number) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>(360);
  const [hover, setHover] = useState<number | null>(null);
  const geo = useMemo(() => {
    const x = linear(pad(extent(xy.map((p) => p[0]))), [P, width - P]);
    const y = linear(pad(extent(xy.map((p) => p[1]))), [H - P, P]);
    return { pts: xy.map(([a, b]) => [x(a), y(b)] as [number, number]) };
  }, [xy, width]);

  const i = Math.min(index, geo.pts.length - 1);
  const from = Math.max(0, i - TRAIL);
  const trail = geo.pts.slice(from, i + 1);
  const shown = hover ?? i;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={H} role="img" aria-label="Agent memory map, two-dimensional projection">
        <rect x={0} y={0} width={width} height={H} fill="none" stroke="var(--grid)" rx={8} />
        {geo.pts.map(([px, py], k) => (
          <circle key={k} cx={px} cy={py} r={k === hover ? 5 : 3}
            fill="var(--s-agent)" fillOpacity={0.12 + 0.88 * allocation[k]}
            stroke={k === hover ? "var(--fg)" : "none"}
            onPointerEnter={() => setHover(k)} onPointerLeave={() => setHover(null)}
            onClick={() => onIndex(k)} style={{ cursor: "pointer" }} />
        ))}
        <path d={linePath(trail.map((p) => p[0]), trail.map((p) => p[1]))} fill="none" stroke="var(--fg)" strokeOpacity={0.55} strokeWidth={1.5} />
        <circle cx={geo.pts[i][0]} cy={geo.pts[i][1]} r={6.5} fill="var(--s-agent)" stroke="var(--surface)" strokeWidth={2.5} />
      </svg>
      <div className="legend" style={{ marginTop: 8, justifyContent: "space-between" }}>
        <span>
          <i className="swatch block" style={{ background: "var(--s-agent)", opacity: 0.15 }} /> mostly cash
          <i className="swatch block" style={{ background: "var(--s-agent)", marginLeft: 8 }} /> mostly stock
        </span>
        <span className="mono">{fmtDate(dates[shown])} · {pctAbs(allocation[shown])}</span>
      </div>
    </div>
  );
}
