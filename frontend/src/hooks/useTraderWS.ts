import { useEffect, useRef, useState, useCallback } from "react";

export type TickData = {
  type: "tick" | "init" | "training_complete";
  total_steps?: number;
  total_steps_target?: number;
  episode?: number;
  loss?: number;
  portfolio_value?: number;
  cash?: number;
  shares_held?: number;
  position?: number;
  sharpe?: number;
  max_drawdown?: number;
  total_return?: number;
  price?: number;
  reward?: number;
  trade_log?: TradeEntry[];
  portfolio_history?: number[];
  loss_history?: number[];
  price_history?: number[];
  is_training?: boolean;
};

export type TradeEntry = {
  time: string;
  action: "BUY" | "SELL" | "HOLD";
  price: number;
  alloc: number;
  pnl: number;
};

export function useTraderWS(url: string) {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastTick, setLastTick] = useState<TickData | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    const socket = new WebSocket(url);

    socket.onopen = () => {
      setConnected(true);
      clearTimeout(reconnectTimer.current);
    };

    socket.onmessage = (e) => {
      try {
        const data: TickData = JSON.parse(e.data);
        setLastTick(data);
      } catch {}
    };

    socket.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    socket.onerror = () => socket.close();
    ws.current = socket;
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  return { connected, lastTick };
}
