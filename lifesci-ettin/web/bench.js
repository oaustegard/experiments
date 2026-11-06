// Step time of web/distill.js at training sizes, on random weights (the time does not depend on them).
import { DistillTrainer } from "./distill.js";

export async function bench(device, _get, { shapes = [[128, 128], [64, 256]], V = 50368, steps = 6, log = () => {} } = {}) {
  const maxR = Math.max(...shapes.map(([b, s]) => b * s)), maxB = Math.max(...shapes.map(([b]) => b));
  const tr = await DistillTrainer.create(device, { V, maxRows: maxR, maxSeqs: maxB });
  const p = new Float32Array(tr.nParams);
  let seed = 1;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296 - 0.5) * 0.04;
  for (let i = 0; i < p.length; i++) p[i] = rnd();
  for (const x of tr.layout) if (x.name.endsWith("norm")) p.fill(1, x.offset, x.offset + x.shape[0]);
  tr.setParams(p);
  const out = [];
  for (const [B, S] of shapes) {
    const ids = new Int32Array(B * S), lens = new Int32Array(B), tgt = new Float32Array(B * 768);
    for (let i = 0; i < ids.length; i++) ids[i] = Math.floor(Math.abs(rnd()) * 25 * V) % V;
    for (let b = 0; b < B; b++) lens[b] = S - (b % 7) * Math.floor(S / 16);
    for (let i = 0; i < tgt.length; i++) tgt[i] = rnd();
    const ms = [];
    for (let s = 0; s < steps; s++) {
      const t0 = performance.now();
      tr.loadBatch({ ids, B, S, lens, tgt });
      tr.forwardBackward({ slot: s });
      tr.adamStep({ lr: 1e-4, step: s + 1 });
      await tr.flush();
      ms.push(performance.now() - t0);
    }
    const loss = await tr.read(tr.LOSS, steps);
    const sorted = ms.slice(1).sort((a, b) => a - b);
    out.push({ B, S, ms, medianMs: sorted[Math.floor(sorted.length / 2)], loss: Array.from(loss) });
    log(JSON.stringify(out.at(-1)));
  }
  return { V, nParams: tr.nParams, shapes: out };
}
