const BASE = "http://localhost:8000";

export async function getStatus() {
  const r = await fetch(`${BASE}/api/status`);
  return r.json();
}

export async function startTraining(ticker: string, totalSteps: number) {
  const r = await fetch(`${BASE}/api/train/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, total_steps: totalSteps }),
  });
  return r.json();
}

export async function stopTraining() {
  const r = await fetch(`${BASE}/api/train/stop`, { method: "POST" });
  return r.json();
}

export async function pauseTraining() {
  const r = await fetch(`${BASE}/api/train/pause`, { method: "POST" });
  return r.json();
}

export async function getWalkForward(ticker: string) {
  const r = await fetch(`${BASE}/api/walkforward/${ticker}`);
  return r.json();
}

export async function getWalkForwardStatus() {
  const r = await fetch(`${BASE}/api/walkforward/status/current`);
  return r.json();
}

export async function runWalkForward(ticker: string, stepsPerWindow = 20_000) {
  const r = await fetch(`${BASE}/api/walkforward/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, steps_per_window: stepsPerWindow }),
  });
  return r.json();
}
