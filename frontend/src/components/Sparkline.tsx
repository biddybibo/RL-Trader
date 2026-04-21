import { useEffect, useRef } from "react";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  lineWidth?: number;
}

export function Sparkline({
  data,
  width = 200,
  height = 50,
  color = "#00ff9d",
  fillColor = "rgba(0,255,157,0.08)",
  lineWidth = 1.5,
}: SparklineProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width  = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const toX = (i: number) => (i / (data.length - 1)) * width;
    const toY = (v: number) => height - ((v - min) / range) * (height - 4) - 2;

    // Fill
    ctx.beginPath();
    ctx.moveTo(toX(0), height);
    data.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.lineTo(toX(data.length - 1), height);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(v));
      else ctx.lineTo(toX(i), toY(v));
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Last point dot
    const lastX = toX(data.length - 1);
    const lastY = toY(data[data.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [data, width, height, color, fillColor, lineWidth]);

  return <canvas ref={ref} style={{ display: "block" }} />;
}
