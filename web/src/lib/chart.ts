import { useLayoutEffect, useRef, useState } from "react";

export type Scale = ((v: number) => number) & { domain: [number, number]; range: [number, number]; invert: (p: number) => number };

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = ((v: number) => r0 + (v - d0) * k) as Scale;
  f.domain = domain; f.range = range;
  f.invert = (p: number) => (k === 0 ? d0 : d0 + (p - r0) / k);
  return f;
}

export function extent(...arrays: number[][]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

export function pad([lo, hi]: [number, number], frac = 0.06): [number, number] {
  const p = (hi - lo || 1) * frac;
  return [lo - p, hi + p];
}

/** "Nice" ticks across a domain. */
export function ticks([lo, hi]: [number, number], count = 5): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / count, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function linePath(xs: number[], ys: number[]): string {
  let d = "";
  for (let i = 0; i < xs.length; i++) d += `${i ? "L" : "M"}${xs[i].toFixed(1)},${ys[i].toFixed(1)}`;
  return d;
}

export function areaPath(xs: number[], ys: number[], y0: number): string {
  if (!xs.length) return "";
  return `${linePath(xs, ys)}L${xs[xs.length - 1].toFixed(1)},${y0}L${xs[0].toFixed(1)},${y0}Z`;
}

/** Month-start indices for a date axis, thinned to roughly `max` labels. */
export function monthTicks(dates: string[], max = 8): number[] {
  const idx: number[] = [];
  for (let i = 1; i < dates.length; i++) if (dates[i].slice(5, 7) !== dates[i - 1].slice(5, 7)) idx.push(i);
  const every = Math.max(1, Math.ceil(idx.length / max));
  return idx.filter((_, k) => k % every === 0);
}

export const monthLabel = (s: string) => {
  const d = new Date(s + "T00:00:00");
  return d.getMonth() === 0 ? String(d.getFullYear()) : d.toLocaleDateString("en-US", { month: "short" });
};

/** Width of a container, tracked with ResizeObserver. */
export function useWidth<T extends HTMLElement>(fallback = 800) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.max(240, Math.floor(el.clientWidth)));
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}
