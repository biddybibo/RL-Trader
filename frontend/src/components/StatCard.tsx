import { useEffect, useRef } from "react";
import { useMotionValue, useSpring, useTransform, motion } from "framer-motion";

interface StatCardProps {
  label: string;
  value: number;
  format?: (n: number) => string;
  colorClass?: string;
  sub?: string;
}

export function StatCard({ label, value, format, colorClass = "", sub }: StatCardProps) {
  const mv = useMotionValue(value);
  const spring = useSpring(mv, { stiffness: 80, damping: 18 });
  const display = useTransform(spring, (n) => (format ? format(n) : n.toFixed(2)));
  const prevRef = useRef(value);

  useEffect(() => {
    if (value !== prevRef.current) {
      mv.set(value);
      prevRef.current = value;
    }
  }, [value, mv]);

  return (
    <div className="stat-cell">
      <span className="stat-label">{label}</span>
      <motion.span className={`stat-value ${colorClass}`}>{display}</motion.span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}
