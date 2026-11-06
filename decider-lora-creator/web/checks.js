// The creator's correctness checks, shared by check.html (any browser) and bridge_lora.js (a real GPU).
//   const R = await runChecks({ device, models, code, fetchBytes, n, test, epochs, log });
// models: the adapter-library models/ URL; code: this directory's URL. Each check returns { ok, ... }:
//   prompt_ids_match   the prompt builder against the token ids of the training code (fixtures.json)
//   head_vs_numpy_ref  head-only logits against adapter-library's NumPy reference (everyday/fixtures.json)
//   head_train_export  a short head-only run on the demo; its export reloaded in a fresh SpikeEngine
//   lora_forward       the WGSL trainer's logits against the head-only path on the same start (Wo_mlp zeroed)
//   lora_roundtrip     loadSpike then exportSpike returns the same weights (to fp16)
//   lora_train_export  a short LoRA run (loss falls); its export reloaded in a fresh SpikeEngine
//   head_resume, lora_resume  a run paused mid-epoch, its checkpoint structured-cloned (as IndexedDB stores
//                      it), the backend reset, then resumed, ends where the uninterrupted run ends
import * as C from "./creator.js";
import { HeadBackend, LoraBackend, truncateRank } from "./backends.js";
import { SpikeEngine } from "../../adapter-library/web/engine.js";

const maxAbs = (a, b) => { let m = 0; a.forEach((z, i) => z.forEach((x, j) => { m = Math.max(m, Math.abs(x - b[i][j])); })); return m; };
const argmaxSame = (a, b) => a.filter((z, i) => Array.from(z).indexOf(Math.max(...z)) === Array.from(b[i]).indexOf(Math.max(...b[i]))).length;

export async function runChecks({ device, models, code, fetchBytes = null, n = 16, test = 6, epochs = 3, lora = true, trainer = null, log = () => {} }) {
  const FB = fetchBytes ?? (async (u) => new Uint8Array(await fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); })));
  const J = async (u) => JSON.parse(new TextDecoder().decode(await FB(u)));
  const R = {};
  const put = (k, v) => { R[k] = v; log(k, v); };
  const BASE = models + "base31/";   // the library spikes and their fixtures are on the ±31 base since 2026-11-06
  const [manifest, wbytes, tok] = await Promise.all([J(BASE + "manifest.json"), FB(BASE + "weights.bin"), C.loadTokenizer(BASE + "tokenizer.json")]);
  const weights = () => wbytes.buffer.slice(wbytes.byteOffset, wbytes.byteOffset + wbytes.byteLength);
  const spike = async (set, id) => ({ sm: await J(models + `${set}/spikes/${id}.json`), bytes: await FB(models + `${set}/spikes/${id}.bin`) });

  // 1. prompt ids
  const demo = await J(code + "demo/invoice_processing.json"), fx = await J(code + "fixtures.json");
  const exs = C.examples(demo.decision, demo.cases.slice(0, 4), tok.build);
  const bad = fx.cases.filter((f, i) => { const e = exs[i]; return !(e && e.q === f.q && e.ids.join() === f.ids.join() && e.pos.join() === f.pos.join()); }).length;
  put("prompt_ids_match", { ok: bad === 0 && exs.length === fx.cases.length, sequences: fx.cases.length, mismatched: bad });

  // 2. head-only path vs the NumPy reference
  const efx = await J(models + "everyday/fixtures.json");
  const be = await HeadBackend.create(device, manifest, weights(), { init: await spike("everyday", efx.cases[0].spike) });
  let worst = 0, arg = 0;
  for (const c of efx.cases) {
    const s = await spike("everyday", c.spike);
    await be.loadSpike(s.sm, s.bytes);
    const [z] = await be.score([{ ids: c.ids, pos: c.pos }]);
    c.logits.forEach((x, i) => { worst = Math.max(worst, Math.abs(x - z[i])); });
    arg += argmaxSame([Float32Array.from(c.logits)], [z]);
  }
  put("head_vs_numpy_ref", { ok: worst < 1e-3 && arg === efx.cases.length, cases: efx.cases.length, max_abs_logit_diff: worst, argmax_same: arg });

  // the demo split and the library start (mean of the other three workflow spikes, rank 8)
  const lib = await Promise.all(demo.init.map((id) => spike("workflows", id)));
  const merged = truncateRank(C.mergeSpikes(lib.map((s) => C.unpackSpike(s.sm, s.bytes))), 8);
  const init = C.packSpike(merged, { id: "library-merge" });
  const split = C.splitCases(demo.cases, { seed: 0, maxTrain: n, maxTest: test, valFrac: 0.2 });
  const data = { train: C.examples(demo.decision, split.train, tok.build), val: C.examples(demo.decision, split.val, tok.build), test: C.examples(demo.decision, split.test, tok.build) };
  const eng2 = await SpikeEngine.create(device, manifest, weights(), { maxTokens: 4096, maxSeqs: 16, maxPos: 2048 });
  const reload = async (sp, exs) => { eng2.setSpike(sp.sm, sp.bytes); const out = []; for (const e of exs) out.push((await eng2.score([{ ids: e.ids, pos: e.pos }]))[0]); return out; };
  const trainRun = async (b, lr) => {
    const ep = [];
    const res = await C.train(b, data, { epochs, lr, bs: 16, seed: 0 }, (e) => { if (e.kind === "epoch") ep.push({ epoch: e.epoch, trainLoss: e.trainLoss, valKL: e.val?.kl, ms: Math.round(e.ms) }); });
    const zb = await b.score(data.test), z2 = await reload(res.spike, data.test);
    return { ep, res, d: maxAbs(zb, z2), same: argmaxSame(zb, z2), lossDown: ep.length > 2 && ep[ep.length - 1].trainLoss < ep[1].trainLoss };
  };

  // 3. head-only training and export
  await be.loadSpike(init.sm, init.bytes);
  const h = await trainRun(be, 8e-4);
  put("head_train_export", { ok: h.lossDown && h.d < 0.05 && h.same === data.test.length, epochs: h.ep, test_acc: h.res.test?.accuracy,
    export_reload_max_logit_diff: h.d, argmax_same: `${h.same}/${data.test.length}` });
  if (!lora) return R;

  // 4-6. the WGSL trainer
  const maxRows = 16 * Math.max(...[...data.train, ...data.val, ...data.test].map((e) => e.ids.length));
  const lb = await LoraBackend.create(device, manifest, weights(), { module: trainer ?? new URL("../../ettin-wgsl-train/web/train.js", code).href, pad: tok.pad, maxSeqs: 16, maxRows });
  const noMlp = { layers: merged.layers.map((L) => ({ ...L, Wo_mlp: { ...L.Wo_mlp, A: new Float32Array(L.Wo_mlp.A.length), B: new Float32Array(L.Wo_mlp.B.length) } })), head: merged.head };
  const init0 = C.packSpike(noMlp, { id: "library-merge-no-wo_mlp" });
  await lb.loadSpike(init0.sm, init0.bytes); await be.loadSpike(init0.sm, init0.bytes);
  const zl = await lb.score(data.test), zh = await be.score(data.test);
  put("lora_forward", { ok: maxAbs(zl, zh) < 0.02 && argmaxSame(zl, zh) === data.test.length, max_abs_logit_diff: maxAbs(zl, zh), argmax_same: `${argmaxSame(zl, zh)}/${zl.length}` });
  const back = C.unpackSpike(...Object.values(await lb.exportSpike()));
  let rt = 0;
  noMlp.layers.forEach((L, i) => ["Wqkv", "Wo", "Wi"].forEach((k) => { for (const t of ["A", "B"]) L[k][t].forEach((x, j) => { rt = Math.max(rt, Math.abs(x - back.layers[i][k][t][j]) / (Math.abs(x) + 1e-3)); }); }));
  put("lora_roundtrip", { ok: rt < 2e-3, max_rel_err: rt });
  await lb.loadSpike(init.sm, init.bytes);
  const l = await trainRun(lb, 5e-4);
  put("lora_train_export", { ok: l.lossDown && l.d < 0.05 && l.same === data.test.length, epochs: l.ep, test_acc: l.res.test?.accuracy, wo_mlp_dropped: lb.dropped,
    export_reload_max_logit_diff: l.d, argmax_same: `${l.same}/${data.test.length}` });
  // 7. pause and resume on the real backends
  const resumeCheck = async (b, lr, stopAt) => {
    const opts = { epochs, lr, bs: 16, seed: 0 };
    await b.loadSpike(init.sm, init.bytes);
    const U = await C.train(b, data, opts);
    const zU = await b.score(data.test);
    await b.loadSpike(init.sm, init.bytes);
    let seen = 0;
    const P = await C.train(b, data, opts, (e) => { if (e.kind === "step") seen = e.step; }, () => seen === stopAt);
    if (!P.paused) return { ok: false, error: "did not pause" };
    const cp = structuredClone(P.checkpoint);
    await b.loadSpike(init.sm, init.bytes);   // as if the page had reloaded: the backend holds the start again
    const Rz = await C.train(b, data, opts, () => {}, () => false, { resume: cp });
    const zR = await b.score(data.test), d = maxAbs(zU, zR);
    return { ok: d < 1e-4 && argmaxSame(zU, zR) === zU.length && Rz.bestEpoch === U.bestEpoch, pausedAt: { epoch: cp.ep, step: cp.step },
             max_abs_logit_diff: d, best_epoch: [U.bestEpoch, Rz.bestEpoch], test_acc: [U.test?.accuracy, Rz.test?.accuracy] };
  };
  const spe = Math.ceil(data.train.length / 16);
  put("head_resume", await resumeCheck(be, 8e-4, spe + 1));
  put("lora_resume", await resumeCheck(lb, 5e-4, spe + 1));
  R.ok = Object.values(R).every((v) => v.ok);
  return R;
}
