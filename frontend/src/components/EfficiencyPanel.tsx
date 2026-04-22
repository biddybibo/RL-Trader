import { useEffect, useRef } from "react";

interface MiniChartProps {
  data: number[];
  label: string;
  value: string;
  color: string;
  fillColor: string;
  zeroLine?: boolean;
}

function MiniChart({ data, label, value, color, fillColor, zeroLine }: MiniChartProps) {
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas    = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || data.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W   = container.offsetWidth;
    const H   = 72;
    canvas.width        = W * dpr;
    canvas.height       = H * dpr;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const PAD = { t: 4, r: 4, b: 4, l: 4 };
    const cw = W - PAD.l - PAD.r;
    const ch = H - PAD.t - PAD.b;

    const min   = Math.min(...data);
    const max   = Math.max(...data);
    const range = max - min || 1;
    const toX   = (i: number) => PAD.l + (i / (data.length - 1)) * cw;
    const toY   = (v: number) => PAD.t + ch - ((v - min) / range) * ch;

    // Zero line
    if (zeroLine && min < 0 && max > 0) {
      const zy = toY(0);
      ctx.beginPath();
      ctx.strokeStyle = "#333";
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 4]);
      ctx.moveTo(PAD.l, zy);
      ctx.lineTo(PAD.l + cw, zy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Fill
    ctx.beginPath();
    ctx.moveTo(toX(0), H);
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.lineTo(toX(data.length - 1), H);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(v));
      else         ctx.lineTo(toX(i), toY(v));
    });
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = "round";
    ctx.stroke();

    // Endpoint dot
    const lx = toX(data.length - 1);
    const ly = toY(data[data.length - 1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [data, color, fillColor, zeroLine]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <span style={{ fontSize: 9, color: "var(--muted)", fontFamily: "var(--mono)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {label}
        </span>
        <span style={{ fontSize: 13, fontFamily: "var(--mono)", color }}>
          {value}
        </span>
      </div>
      {data.length < 2
        ? <div style={{ height: 72, display: "flex", alignItems: "center", justifyContent: "center", color: "#333", fontSize: 10, fontFamily: "var(--mono)" }}>
            waiting for rollouts…
          </div>
        : <canvas ref={canvasRef} style={{ display: "block", width: "100%" }} />
      }
    </div>
  );
}

interface Props {
  effLoss:    number[];
  effSharpe:  number[];
  effReturn:  number[];
  effWinRate: number[];
  effSteps:   number[];
  winRate:    number;
  stepsPerSec: number;
  rolloutCount: number;
}

export function EfficiencyPanel({
  effLoss, effSharpe, effReturn, effWinRate,
  winRate, stepsPerSec, rolloutCount,
}: Props) {
  const lastLoss    = effLoss.at(-1)    ?? 0;
  const lastSharpe  = effSharpe.at(-1)  ?? 0;
  const lastReturn  = effReturn.at(-1)  ?? 0;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 14, flexWrap: "wrap" }}>
        <div className="card-label" style={{ margin: 0 }}>Training Efficiency</div>
        <div style={{ display: "flex", gap: 20, marginLeft: "auto", flexWrap: "wrap" }}>
          <div className="wf-stat">
            <span className="wf-stat-label">Rollouts</span>
            <span className="wf-stat-val neutral">{rolloutCount.toLocaleString()}</span>
          </div>
          <div className="wf-stat">
            <span className="wf-stat-label">Steps/sec</span>
            <span className="wf-stat-val neutral">{stepsPerSec.toLocaleString()}</span>
          </div>
          <div className="wf-stat">
            <span className="wf-stat-label">Win Rate</span>
            <span className={`wf-stat-val ${winRate >= 55 ? "pos" : winRate < 45 ? "neg" : "neutral"}`}>
              {winRate.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px 24px" }}>
        <MiniChart
          data={effLoss}
          label="Loss / rollout"
          value={lastLoss.toFixed(4)}
          color="#60a5fa"
          fillColor="rgba(96,165,250,0.08)"
        />
        <MiniChart
          data={effSharpe}
          label="Sharpe / rollout"
          value={`${lastSharpe >= 0 ? "+" : ""}${lastSharpe.toFixed(2)}`}
          color={lastSharpe >= 1 ? "#00ff9d" : lastSharpe < 0 ? "#ff4d6d" : "#f0b429"}
          fillColor={lastSharpe >= 0 ? "rgba(0,255,157,0.07)" : "rgba(255,77,109,0.07)"}
          zeroLine
        />
        <MiniChart
          data={effReturn}
          label="Return % / rollout"
          value={`${lastReturn >= 0 ? "+" : ""}${lastReturn.toFixed(1)}%`}
          color={lastReturn >= 0 ? "#00ff9d" : "#ff4d6d"}
          fillColor={lastReturn >= 0 ? "rgba(0,255,157,0.07)" : "rgba(255,77,109,0.07)"}
          zeroLine
        />
        <MiniChart
          data={effWinRate}
          label="Win Rate % / rollout"
          value={`${winRate.toFixed(1)}%`}
          color="#a78bfa"
          fillColor="rgba(167,139,250,0.08)"
        />
      </div>
    </div>
  );
}
