// Gradcheck of web/distill.js against PyTorch (run/reference.py gradcheck): loss, every gradient
// tensor, one AdamW step. `get(path)` fetches a file relative to this experiment directory.
import { DistillTrainer } from "./distill.js";

// opts.fixture: "gradcheck" (cosine loss) or "gradcheck_nce" (contrastive, run/reference.py nce)
export async function gradcheck(device, get, opts = {}) {
  const dir = opts.fixture || "gradcheck";
  const meta = JSON.parse(new TextDecoder().decode(await get(`fixtures/${dir}/meta.json`)));
  const bin = async (n, T = Float32Array) => new T((await get(`fixtures/${dir}/${n}.bin`)).slice().buffer);
  const t0 = performance.now();
  const tr = await DistillTrainer.create(device, { V: meta.V, maxRows: meta.B * meta.S, maxSeqs: meta.B });
  const initMs = performance.now() - t0;
  if (tr.nParams !== meta.nParams) throw new Error(`layout mismatch: ${tr.nParams} vs ${meta.nParams}`);
  meta.layout.forEach((x, i) => { if (x.name !== tr.layout[i].name || x.offset !== tr.layout[i].offset) throw new Error(`layout ${x.name}`); });
  const p0 = await bin("params0"), gref = await bin("grads"), p1ref = await bin("params1");
  tr.setParams(p0);
  tr.loadBatch({ ids: await bin("ids", Int32Array), B: meta.B, S: meta.S, lens: await bin("lens", Int32Array), tgt: meta.loss_kind === "nce" ? null : await bin("tgt") });
  await tr.flush();
  const t1 = performance.now();
  tr.forwardBackward({ slot: 0, loss: meta.loss_kind || "cos", scale: meta.scale });
  await tr.flush();
  const stepMs = performance.now() - t1;
  const loss = (await tr.read(tr.LOSS, 1))[0];
  const g = await tr.read(tr.GRD);
  const a = meta.adam;
  tr.adamStep({ lr: a.lr, wd: a.wd, step: a.step, clip: a.clip });
  const p1 = await tr.read(tr.PRM);
  const norm = (await tr.read(tr.NRM, 1))[0];
  const rel = (x, y, o, n) => { let num = 0, den = 0; for (let i = o; i < o + n; i++) { num += (x[i] - y[i]) ** 2; den += y[i] ** 2; } return den ? Math.sqrt(num / den) : Math.sqrt(num); };
  let total = 0;
  for (const v of gref) total += v * v;
  total = Math.sqrt(total);
  const tensors = meta.layout.map((x) => {
    const n = x.shape.reduce((u, v) => u * v, 1);
    let rn = 0;
    for (let i = x.offset; i < x.offset + n; i++) rn += gref[i] ** 2;
    return { name: x.name, n, gradRelErr: rel(g, gref, x.offset, n), refNorm: Math.sqrt(rn) };
  });
  const live = tensors.filter((t) => t.refNorm > 1e-6 * total);
  let num = 0, den = 0, flips = 0, nU = 0;
  for (let i = 0; i < p1.length; i++) {
    const du = p1[i] - p0[i], dr = p1ref[i] - p0[i];
    num += (du - dr) ** 2; den += dr * dr; nU++;
    if (Math.sign(du) !== Math.sign(dr)) flips++;
  }
  return { loss, lossRef: meta.loss, lossRelErr: Math.abs(loss - meta.loss) / Math.abs(meta.loss), gradNorm: norm, gradNormRef: meta.grad_norm,
    maxGradRelErr: Math.max(...live.map((t) => t.gradRelErr)), worst: live.slice().sort((a, b) => b.gradRelErr - a.gradRelErr).slice(0, 5),
    allGradRelErr: rel(g, gref, 0, g.length), updateRelErr: Math.sqrt(num / den), updateSignFlips: flips, updateN: nU, initMs, stepMs,
    batch: { B: meta.B, S: meta.S, lens: meta.lens }, grads: g };
}
