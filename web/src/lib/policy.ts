/** The trained policy, running in the browser via onnxruntime-web.
 *  model.onnx: (obs [B,1,17], h [1,B,H], c [1,B,H]) -> (allocation [B], value [B], h_out, c_out) */
import type { InferenceSession, Tensor } from "onnxruntime-web";

const ORT_VERSION = "1.22.0";
let session: Promise<{ s: InferenceSession; T: typeof Tensor }> | null = null;

function getSession() {
  if (!session) {
    session = (async () => {
      const ort = await import("onnxruntime-web/wasm");
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      ort.env.wasm.numThreads = 1;
      const s = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}data/model.onnx`);
      return { s, T: ort.Tensor };
    })();
  }
  return session;
}

/** Allocation for each observation row. `warmup` repeats the same observation that many
 *  days first, so the LSTM memory reflects "these conditions have persisted for a while". */
let queue: Promise<unknown> = Promise.resolve();

/** The wasm backend can't run overlapping calls on one session, so calls are serialised. */
export function allocate(obs: Float32Array[], hidden = 128, warmup = 0): Promise<Float32Array> {
  const run = queue.then(() => runOnce(obs, hidden, warmup));
  queue = run.catch(() => undefined);
  return run;
}

async function runOnce(obs: Float32Array[], hidden: number, warmup: number): Promise<Float32Array> {
  const { s, T } = await getSession();
  const B = obs.length, D = obs[0].length;
  const flat = new Float32Array(B * D);
  obs.forEach((o, i) => flat.set(o, i * D));
  const x = new T("float32", flat, [B, 1, D]);
  let h: Tensor = new T("float32", new Float32Array(B * hidden), [1, B, hidden]);
  let c: Tensor = new T("float32", new Float32Array(B * hidden), [1, B, hidden]);
  let out = await s.run({ obs: x, h, c });
  for (let k = 0; k < warmup; k++) {
    h = out.h_out as Tensor; c = out.c_out as Tensor;
    out = await s.run({ obs: x, h, c });
  }
  return out.allocation.data as Float32Array;
}
