// The full training run of run/reference.py traj, on WebGPU: same initial adapter, same batch order,
// same schedule. Records the loss of every step and val / test / reply_thread metrics after each epoch.
// `get(path)` returns a file's bytes: fixtures/traj/* and model/* (the packed page model).
import { Trainer } from "./train.js";

const dec = new TextDecoder();

// smoke: train on the 2 shortest batches of epoch 1 and evaluate one short batch per set (a plumbing check)
export async function trajectory(device, get, { smoke = false, epochs = 0, log = () => {}, progress = () => {} } = {}) {
  const meta = JSON.parse(dec.decode(await get("fixtures/traj/meta.json")));
  const man = JSON.parse(dec.decode(await get("model/manifest.json")));
  const weights = await get("model/weights.bin");
  const arr = async (n, T) => new T((await get(`fixtures/traj/${n}.bin`)).slice().buffer);
  const sets = {};
  for (const name of Object.keys(meta.sets)) {
    const lens = await arr(`${name}_lens`, Int32Array), off = new Int32Array(lens.length + 1);
    for (let i = 0; i < lens.length; i++) off[i + 1] = off[i] + lens[i];
    sets[name] = { lens, off, ids: await arr(`${name}_ids`, Int32Array), pos: await arr(`${name}_pos`, Int32Array), gold: await arr(`${name}_gold`, Float32Array) };
  }
  const K = 2;   // options per example (noul: true, false)
  const collate = (set, idx) => {
    const B = idx.length, S = Math.max(...idx.map((i) => set.lens[i]));
    const ids = new Int32Array(B * S).fill(meta.pad), lens = new Int32Array(B), pos = new Int32Array(B * K), gold = new Float32Array(B * K);
    idx.forEach((e, b) => {
      ids.set(set.ids.subarray(set.off[e], set.off[e + 1]), b * S);
      lens[b] = set.lens[e];
      for (let k = 0; k < K; k++) { pos[b * K + k] = b * S + set.pos[e * K + k]; gold[b * K + k] = set.gold[e * K + k]; }
    });
    return { ids, B, S, lens, pos, seg: new Int32Array(B).fill(K), gold };
  };
  const all = [...meta.order.flatMap((ep) => ep.map((c) => ["train", c])), ...Object.entries(meta.evalOrder).flatMap(([n, cs]) => cs.map((c) => [n, c]))];
  const maxRows = Math.max(...all.map(([n, c]) => c.length * Math.max(...c.map((i) => sets[n].lens[i]))));
  const t0 = performance.now();
  const tr = await Trainer.create(device, { manifest: man, weights, maxRows, maxSeqs: meta.bs, maxPool: meta.bs * K, maxSeg: meta.bs });
  const initMs = performance.now() - t0;
  tr.setParams(await arr("params0", Float32Array));
  log(`trainer ready: ${tr.nParams} params, maxRows ${maxRows}, ${initMs.toFixed(0)} ms`);

  const { total, warmup: warm } = meta;
  const lam = (s) => Math.min(1, (s + 1) / warm) * Math.max(0, (total - s) / Math.max(1, total - warm));
  const nEp = epochs || meta.epochs;
  const losses = [], epochLog = [];
  let s = 0;
  const tTrain = performance.now();
  for (let ep = 0; ep < nEp; ep++) {
    const te = performance.now();
    const S = (set, c) => Math.max(...c.map((i) => sets[set].lens[i]));
    const order = smoke ? [...meta.order[ep]].sort((a, b) => S("train", a) - S("train", b)).slice(0, 2) : meta.order[ep];
    const first = s;
    for (const ch of order) {
      tr.loadBatch(collate(sets.train, ch));
      tr.forwardBackward({ T: meta.T, slot: s % 4096 });
      tr.adamStep({ lr: meta.lr * lam(s), lrHead: meta.lr_head * lam(s), wd: meta.wd, step: s + 1, clip: meta.clip });
      await tr.flush(s % 16 === 15);   // queue order keeps each step's uploads behind the previous submit
      s++;
      progress((ep + (s - first) / order.length) / nEp, `epoch ${ep + 1} step ${s - first}/${order.length}`);
    }
    await tr.flush(true);
    const l = await tr.read(tr.LOSS, 4096);
    for (let i = first; i < s; i++) losses.push(l[i % 4096]);
    const trainMs = performance.now() - te;
    const ev = {};
    for (const name of ["val", "test", "reply_thread"]) {
      const set = sets[name], n = set.lens.length, probs = new Float32Array(n).fill(NaN);
      const chunks = smoke ? meta.evalOrder[name].slice(0, 1) : meta.evalOrder[name];
      for (const ch of chunks) {
        tr.loadBatch(collate(set, ch));
        tr.forward();
        const z = await tr.read(tr.LOGIT, ch.length * K);
        ch.forEach((e, b) => {
          const a = z[b * K] / meta.T, c = z[b * K + 1] / meta.T, m = Math.max(a, c);
          probs[e] = Math.exp(a - m) / (Math.exp(a - m) + Math.exp(c - m));
        });
      }
      ev[name] = metrics(probs, set.gold, K);
      if (ep === nEp - 1) ev[name].probs = probs;
    }
    epochLog.push({ epoch: ep, trainMs, evalMs: performance.now() - te - trainMs, meanLoss: mean(losses.slice(first)), ...ev });
    log(`epoch ${ep}: loss ${mean(losses.slice(first)).toFixed(4)}, train ${(trainMs / 1000).toFixed(1)} s, test acc ${ev.test.accuracy?.toFixed(4)}`);
  }
  // per-step time with a sync after each step, on a fixed mid-length batch (after training, so it does not disturb it)
  const mid = smoke ? meta.evalOrder.val[0] : meta.order[0][Math.floor(meta.order[0].length / 2)];
  const bt = collate(smoke ? sets.val : sets.train, mid), stepMs = [];
  for (let i = 0; i < (smoke ? 1 : 5); i++) {
    const t = performance.now();
    tr.loadBatch(bt);
    tr.forwardBackward({ T: meta.T, slot: 4095 });
    await tr.flush(true);
    stepMs.push(performance.now() - t);
  }
  return { losses: Float32Array.from(losses), epochs: epochLog, initMs, trainWallMs: performance.now() - tTrain, steps: s,
    syncStepMs: stepMs, syncStepBatch: { B: bt.B, S: bt.S }, maxRows };
}

const mean = (a) => a.reduce((u, v) => u + v, 0) / a.length;

function metrics(probs, gold, K) {
  let n = 0, ok = 0, pos = 0, tp = 0, neg = 0, fp = 0, mae = 0;
  probs.forEach((p, i) => {
    if (Number.isNaN(p)) return;
    const g = gold[i * K];
    n++; mae += Math.abs(p - g);
    if ((p > 0.5) === (g > 0.5)) ok++;
    if (g > 0.5) { pos++; if (p > 0.5) tp++; } else { neg++; if (p > 0.5) fp++; }
  });
  return { n, accuracy: ok / n, recall: pos ? tp / pos : null, false_pos_rate: neg ? fp / neg : null, mean_abs_err: mae / n };
}
