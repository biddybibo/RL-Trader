import { useEffect, useRef } from "react";

interface SharpeTrendProps {
  sharpeHistory: number[];
  lossHistory: number[];
}

function miniChart(
  ctx: CanvasRenderingContext2D,
  data: number[],
  x0: number,
  y0: number,
  w: number,
  h: number,
  color: string,
  label: string,
) {
  if (data.length < 2) return;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const pad = max - min || 0.001;
  const px = (i: number) => x0 + (i / (data.length - 1)) * w;
  const py = (v: number) => y0 + h - ((v - min) / pad) * h;

  ctx.beginPath();
  ctx.moveTo(px(0), py(data[0]));
  for (let i = 1; i < data.length; i++) ctx.lineTo(px(i), py(data[i]));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Fill under
  ctx.lineTo(px(data.length - 1), y0 + h);
  ctx.lineTo(px(0), y0 + h);
  ctx.closePath();
  ctx.fillStyle = color + "18";
  ctx.fill();

  // Label
  ctx.fillStyle = "#6060a0";
  ctx.font = "9px 'Space Mono', monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, x0 + 4, y0 + 12);

  // Latest value
  const last = data[data.length - 1];
  ctx.fillStyle = color;
  ctx.font = "bold 11px 'Space Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillText(last.toFixed(2), x0 + w - 4, y0 + 14);
}

export function SharpeTrend({ sharpeHistory, lossHistory }: SharpeTrendProps) {
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

    if (sharpeHistory.length < 2 && lossHistory.length < 2) {
      ctx.fillStyle = "#404060";
      ctx.font = "11px 'Space Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("waiting for rollouts…", w / 2, h / 2);
      return;
    }

    const gap = 8;
    const half = h / 2 - gap / 2;

    miniChart(ctx, lossHistory,    0, 0,            w, half, "#60a5fa", "LOSS");
    miniChart(ctx, sharpeHistory,  0, half + gap,   w, half, "#00d68f", "SHARPE");
  }, [sharpeHistory, lossHistory]);

  return (
    <div ref={wrapRef} className="chart-canvas-wrap">
      <canvas ref={canvasRef} />
    </div>
  );
}
