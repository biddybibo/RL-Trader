import { useEffect, useState } from "react";

export type Metrics = {
  total_return: number; cagr: number; sharpe: number; sortino: number;
  max_drawdown: number; calmar: number; avg_exposure?: number; turnover?: number;
};

export type TickerIndex = {
  ticker: string; group: "train" | "held_out"; sharpe: number; bnh_sharpe: number;
  max_drawdown: number; bnh_max_drawdown: number; avg_exposure: number;
};

type Stat = { mean: number; std: number };
export type GroupSummary = {
  agent: { sharpe: Stat; max_drawdown: Stat; total_return: Stat };
  buy_and_hold: { sharpe: Stat; max_drawdown: Stat; total_return: Stat };
  sma_50_200: { sharpe: Stat; max_drawdown: Stat; total_return: Stat };
  agent_beats_bnh_sharpe_frac: number;
};

export type Manifest = {
  obs_names: string[];
  model: { source: string; hidden_dim: number; train_meta: Record<string, unknown> };
  memory_projection: { explained_variance: number[] };
  window: { start: string; end: string };
  tickers: TickerIndex[];
  scorecard: { period: string[]; summary: Record<string, GroupSummary>; models: { model: string }[] } | null;
};

export type Trace = {
  ticker: string; group: "train" | "held_out"; dates: string[];
  close: number[]; equity: number[]; bnh: number[];
  allocation: number[]; critic_value: number[];
  obs: number[][]; attribution: number[][]; attribution_baseline: number[];
  memory_xy: [number, number][];
  metrics: { agent: Metrics; bnh: Metrics };
};

export type Replays = {
  ticker: string; dates: string[]; close: number[]; bnh: number[];
  checkpoints: { steps: number; allocation: number[]; value: number[]; sharpe: number; max_drawdown: number; total_return: number }[];
};

export type LearningCurves = {
  runs: { run: string; risk_aversion: number; seed: number;
    train: { steps: number; sharpe: number; bnh_sharpe: number }[];
    val: { steps: number; val_sharpe: number }[]; best_val: number }[];
};

const cache = new Map<string, Promise<unknown>>();
const BASE = `${import.meta.env.BASE_URL}data/`;

export function load<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(BASE + path).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(path) as Promise<T>;
}

export function useData<T>(path: string | null) {
  const [state, setState] = useState<{ data?: T; error?: string; path?: string }>({});
  useEffect(() => {
    if (!path) return;
    let live = true;
    load<T>(path).then((data) => live && setState({ data, path }), (e) => live && setState({ error: String(e), path }));
    return () => { live = false; };
  }, [path]);
  return state.path === path ? state : {};
}

/** Human labels for the 17 observation inputs. */
export const OBS_LABELS: Record<string, string> = {
  position: "Current position", drawdown: "Drawdown", episode_return: "Return so far",
  log_return: "Today's return", sma_10: "Price vs 10-day avg", sma_30: "Price vs 30-day avg",
  volatility: "Volatility", rsi: "RSI", macd: "MACD", volume_ratio: "Volume vs normal",
  atr_norm: "Daily range (ATR)", hw_proximity: "Near 52-week high", day_sin: "Weekday (sin)",
  day_cos: "Weekday (cos)", month_sin: "Month (sin)", vix_norm: "VIX", rate_norm: "10-yr rate",
};

export const pct = (x: number, d = 1) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;
export const pctAbs = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
export const num = (x: number, d = 2) => `${x >= 0 ? "+" : ""}${x.toFixed(d)}`;
export const fmtDate = (s: string) =>
  new Date(s + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
