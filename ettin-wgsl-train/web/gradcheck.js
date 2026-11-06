// Gradcheck of web/train.js against PyTorch: loss, every gradient tensor, and one AdamW step, on the
// batch in fixtures/gradcheck (run/reference.py gradcheck). Runs in a page (web/check.html) or a GPU
// Bridge job; `get(path)` fetches a file relative to the experiment directory.
import { Trainer } from "./train.js";

export async function gradcheck(device, get) {
  const meta = JSON.parse(new TextDecoder().decode(await get("fixtures/gradcheck/meta.json")));
  const bin = async (n, T = Float32Array) => new T((await get(`fixtures/gradcheck/${n}.bin`)).buffer.slice(0));
  const man = JSON.parse(new TextDecoder().decode(await get("model/manifest.json")));
  const weights = await get("model/weights.bin");
  const t0 = performance.now();
  const tr = await Trainer.create(device, { manifest: man, weights, maxRows: meta.B * meta.S, maxSeqs: meta.B, maxPool: meta.nP, maxSeg: meta.nSeg });
  const initMs = performance.now() - t0;
  if (tr.nParams !== meta.nParams) throw new Error(`layout mismatch: ${tr.nParams} vs ${meta.nParams}`);
  meta.layout.forEach((x, i) => { if (x.name !== tr.layout[i].name || x.offset !== tr.layout[i].offset) throw new Error(`layout ${x.name}`); });
  const p0 = await bin("params0"), gref = await bin("grads"), p1ref = await bin("params1");
  tr.setParams(p0);
  tr.loadBatch({ ids: await bin("ids", Int32Array), B: meta.B, S: meta.S, lens: await bin("lens", Int32Array), pos: await bin("pos", Int32Array),
    seg: await bin("seg", Int32Array), gold: await bin("gold") });
  const t1 = performance.now();
  tr.forwardBackward({ T: meta.T, slot: 0 });
  await tr.flush();
  const stepMs = performance.now() - t1;
  const loss = (await tr.read(tr.LOSS, 1))[0];
  const g = await tr.read(tr.GRD);
  const a = meta.adam;
  tr.adamStep({ lr: a.lr, lrHead: a.lr_head, wd: a.wd, step: a.step, clip: a.clip });
  const p1 = await tr.read(tr.PRM);
  const norm = (await tr.read(tr.NRM, 1))[0];
  const rel = (x, y, o, n) => {
    let num = 0, den = 0;
    for (let i = o; i < o + n; i++) { num += (x[i] - y[i]) ** 2; den += y[i] ** 2; }
    return den ? Math.sqrt(num / den) : Math.sqrt(num);
  };
  const tensors = meta.layout.map((x, i) => {
    const n = x.shape.reduce((u, v) => u * v, 1);
    return { name: x.name, n, gradRelErr: rel(g, gref, x.offset, n) };
  });
  // Tensors whose true gradient is zero (the head's last bias and LayerNorm bias: a constant added to
  // every option's logit cancels in the segment softmax) have only float noise as reference; they are
  // reported by absolute size and left out of the relative comparisons.
  const total = Math.sqrt(gref.reduce((u, v) => u + v * v, 0));
  for (const t of tensors) {
    const x = meta.layout.find((y) => y.name === t.name);
    let a = 0, b = 0;
    for (let i = x.offset; i < x.offset + t.n; i++) { a += gref[i] ** 2; b += g[i] ** 2; }
    t.refNorm = Math.sqrt(a); t.norm = Math.sqrt(b); t.zeroGrad = t.refNorm < 1e-6 * total;
  }
  const live = tensors.filter((t) => !t.zeroGrad);
  // the AdamW update, compared as p1 - p0 so p0 does not swamp the difference
  const du = [], dr = [];
  let flips = 0;
  for (const t of live) {
    const x = meta.layout.find((y) => y.name === t.name);
    for (let i = x.offset; i < x.offset + t.n; i++) {
      du.push(p1[i] - p0[i]); dr.push(p1ref[i] - p0[i]);
      if (Math.sign(p1[i] - p0[i]) !== Math.sign(p1ref[i] - p0[i])) flips++;
    }
  }
  return {
    loss, lossRef: meta.loss, lossRelErr: Math.abs(loss - meta.loss) / Math.abs(meta.loss),
    gradNorm: norm, gradNormRef: meta.grad_norm,
    maxGradRelErr: Math.max(...live.map((t) => t.gradRelErr)), allGradRelErr: rel(g, gref, 0, g.length),
    zeroGradTensors: tensors.filter((t) => t.zeroGrad).map(({ name, refNorm, norm }) => ({ name, refNorm, norm })),
    updateRelErr: rel(du, dr, 0, du.length), updateSignFlips: flips, updateN: du.length, tensors, initMs, stepMs,
    batch: { B: meta.B, S: meta.S, lens: meta.lens },
    grads: g,   // raw, for a float64 comparison (run/reference.py f64)
  };
}
