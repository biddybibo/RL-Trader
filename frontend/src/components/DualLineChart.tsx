import { useEffect, useRef } from "react";

interface DualLineChartProps {
  portfolio: number[];
  price: number[];
  width?: number;
  height?: number;
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  data: number[],
  min: number,
  max: number,
  w: number,
  h: number,
  color: string,
  fill?: string,
) {
  if (data.length < 2) return;
  const pad = max - min || 1;
  const x = (i: number) => (i / (data.length - 1)) * w;
  const y = (v: number) => h - ((v - min) / pad) * h;

  ctx.beginPath();
  ctx.moveTo(x(0), y(data[0]));
  for (let i = 1; i < data.length; i++) ctx.lineTo(x(i), y(data[i]));

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();

  if (fill) {
    ctx.lineTo(x(data.length - 1), h);
    ctx.lineTo(x(0), h);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }
}

export function DualLineChart({ portfolio, price }: DualLineChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (portfolio.length < 2 && price.length < 2) {
      ctx.fillStyle = "#404060";
      ctx.font = "11px 'Space Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("waiting for data…", w / 2, h / 2);
      return;
    }

    // Normalise price to start at same value as portfolio
    const pStart = portfolio[0] ?? 10000;
    const prStart = price[0] ?? 1;
    const priceNorm = price.map((p) => (p / prStart) * pStart);

    const allVals = [...portfolio, ...priceNorm];
    const minV = Math.min(...allVals);
    const maxV = Math.max(...allVals);

    drawLine(ctx, priceNorm, minV, maxV, w, h, "#f0b42966", "rgba(240,180,41,0.04)");
    drawLine(ctx, portfolio, minV, maxV, w, h, "#00d68f", "rgba(0,214,143,0.07)");

    // Baseline at start
    const baseY = h - ((pStart - minV) / (maxV - minV || 1)) * h;
    ctx.setLineDash([3, 5]);
    ctx.strokeStyle = "#252548";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, baseY);
    ctx.lineTo(w, baseY);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [portfolio, price]);

  return (
    <div ref={wrapRef} className="chart-canvas-wrap">
      <canvas ref={canvasRef} />
    </div>
  );
}
