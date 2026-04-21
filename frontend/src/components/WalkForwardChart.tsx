import { useEffect, useRef } from "react";

export interface WFWindow {
  label: string;
  train_sharpe: number;
  test_sharpe: number;
  test_return: number;
  test_drawdown: number;
  bnh_sharpe: number;
}

interface Props {
  windows: WFWindow[];
  width?: number;
  height?: number;
}

export function WalkForwardChart({ windows, height = 180 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || windows.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.offsetWidth;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const PAD = { top: 15, right: 20, bottom: 48, left: 48 };
    const cw = width - PAD.left - PAD.right;
    const ch = height - PAD.top - PAD.bottom;

    ctx.clearRect(0, 0, width, height);

    // Y range
    const allVals = windows.flatMap(w => [w.train_sharpe, w.test_sharpe, w.bnh_sharpe]);
    const yMin = Math.min(-0.3, ...allVals) - 0.15;
    const yMax = Math.max(1.5, ...allVals) + 0.2;
    const toY = (v: number) => PAD.top + ch - ((v - yMin) / (yMax - yMin)) * ch;

    // Grid lines
    const gridVals = [-0.5, 0, 0.5, 1.0, 1.5];
    gridVals.forEach(v => {
      if (v < yMin || v > yMax) return;
      const y = toY(v);
      ctx.beginPath();
      ctx.strokeStyle = v === 0 ? "#333" : "#1e1e1e";
      ctx.lineWidth = v === 0 ? 1.5 : 1;
      ctx.setLineDash(v === 0 ? [] : [3, 4]);
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + cw, y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#444";
      ctx.font = `9px 'Space Mono', monospace`;
      ctx.textAlign = "right";
      ctx.fillText(v.toFixed(1), PAD.left - 6, y + 3);
    });

    // Y axis label
    ctx.save();
    ctx.translate(10, PAD.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#444";
    ctx.font = "9px 'Space Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Sharpe", 0, 0);
    ctx.restore();

    const n = windows.length;
    const groupW = cw / n;
    const barW = groupW * 0.28;

    // Buy-and-hold average line
    const avgBnh = windows.reduce((s, w) => s + w.bnh_sharpe, 0) / n;
    const bnhY = toY(avgBnh);
    ctx.beginPath();
    ctx.strokeStyle = "#f0b429";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.moveTo(PAD.left, bnhY);
    ctx.lineTo(PAD.left + cw, bnhY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Trend line on test_sharpe
    const testVals = windows.map(w => w.test_sharpe);
    const xs = testVals.map((_, i) => i);
    const xMean = xs.reduce((a, b) => a + b, 0) / n;
    const yMeanT = testVals.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    xs.forEach((x, i) => { num += (x - xMean) * (testVals[i] - yMeanT); den += (x - xMean) ** 2; });
    const slope = den === 0 ? 0 : num / den;
    const intercept = yMeanT - slope * xMean;

    const trendColor = slope > 0.02 ? "#00ff9d" : slope < -0.02 ? "#ff4d6d" : "#f0b429";
    ctx.beginPath();
    ctx.strokeStyle = trendColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.moveTo(PAD.left, toY(intercept));
    ctx.lineTo(PAD.left + cw, toY(slope * (n - 1) + intercept));
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Bars
    windows.forEach((w, i) => {
      const cx = PAD.left + (i + 0.5) * groupW;

      // Train bar (gray)
      const trainTop = toY(Math.max(0, w.train_sharpe));
      const trainBot = toY(Math.min(0, w.train_sharpe));
      ctx.fillStyle = "#2a2a2a";
      ctx.strokeStyle = "#3a3a3a";
      ctx.lineWidth = 1;
      ctx.fillRect(cx - barW - 2, trainTop, barW, trainBot - trainTop);
      ctx.strokeRect(cx - barW - 2, trainTop, barW, trainBot - trainTop);

      // Test bar (green/red)
      const testColor = w.test_sharpe >= 0 ? "#00ff9d" : "#ff4d6d";
      const testTop = toY(Math.max(0, w.test_sharpe));
      const testBot = toY(Math.min(0, w.test_sharpe));
      ctx.fillStyle = testColor;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(cx + 2, testTop, barW, testBot - testTop);
      ctx.globalAlpha = 1;

      // X label (rotated)
      ctx.save();
      ctx.translate(cx, height - PAD.bottom + 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = "#555";
      ctx.font = "8px 'Space Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(w.label, 0, 0);
      ctx.restore();
    });

    // Legend
    const legendX = PAD.left + cw - 10;
    const legendY = PAD.top + 6;
    const items: [string, string][] = [
      ["#2a2a2a", "Train"],
      ["#00ff9d", "Test"],
      ["#f0b429", "B&H avg"],
      [trendColor, "Trend"],
    ];
    items.forEach(([color, label], i) => {
      const lx = legendX - i * 72;
      ctx.fillStyle = color;
      ctx.fillRect(lx - 26, legendY, 10, 8);
      ctx.fillStyle = "#555";
      ctx.font = "8px 'Space Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(label, lx - 14, legendY + 7);
    });

  }, [windows, height]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      {windows.length === 0
        ? <div className="empty-chart">run walk-forward analysis to see results</div>
        : <canvas ref={canvasRef} style={{ display: "block", width: "100%" }} />
      }
    </div>
  );
}
