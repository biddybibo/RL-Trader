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
