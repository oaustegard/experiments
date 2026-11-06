// GPU Bridge job body (async function (ctx)): refit the shipped spikes of one set to the ±31 base.
// Inputs: base (u8, out/base31/weights.bin), man (JSON manifest), data (u8, run/refit31_inputs.py output),
// cfg = { sha, models }. Code from jsDelivr at cfg.sha; the shipped spikes and tokenizer from cfg.models.
//
// Per spike, the recipe of the spikes' own quantized-base stage (run/lora.py QW-, run/everyday_train.py
// stage 2): start from the shipped spike, 5 epochs of LoRA + head (lr 5e-4, head 8e-4, AdamW wd 0.01,
// warmup 6% then linear decay, clip 1.0, batch 16, seed 0) with ettin-wgsl-train's WGSL trainer, best epoch
// by validation KL, temperatures fitted on validation. One difference: epoch 0 (the shipped spike as it is,
// on the ±31 base) is a candidate, so a refit that does not help on validation keeps the shipped weights.
// Before training, the trainer's logits on the first spike are checked against SpikeEngine's on the ±31 base.
const { cfg, base, man, data } = ctx.inputs;
const root = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const code = root + "decider-lora-creator/web/";
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const device = await ctx.gpu();
const J = async (u) => JSON.parse(new TextDecoder().decode(await ctx.fetchBytes(u)));
const set = JSON.parse(new TextDecoder().decode(data));
if (cfg.only) set.spikes = set.spikes.filter((s) => cfg.only.includes(s.id));   // refit a subset
const tok = await C.loadTokenizer(cfg.models + "base/tokenizer.json");
const W = () => base.buffer.slice(base.byteOffset, base.byteOffset + base.byteLength);
const shipped = await Promise.all(set.spikes.map(async (s) => ({ ...s, sm: await J(cfg.models + `${set.set}/${s.file}.json`), bytes: await ctx.fetchBytes(cfg.models + `${set.set}/${s.file}.bin`) })));
const exs = shipped.map((s) => ({ train: C.examples(s.sm, s.train, tok.build), val: C.examples(s.sm, s.val, tok.build) }));
const maxLen = Math.max(...exs.flatMap((e) => [...e.train, ...e.val]).map((e) => e.ids.length));
const out = { describe: ctx.describe ? await ctx.describe() : null, sha: cfg.sha, set: set.set, maxLen, spikes: {} };

// forward check: the WGSL trainer and SpikeEngine (head-only path) on the same ±31 base and spike
const lb = await Bk.LoraBackend.create(device, man, W(), { module: root + "ettin-wgsl-train/web/train.js", pad: tok.pad, maxSeqs: 16, maxRows: 16 * maxLen });
{
  const s0 = shipped[0], hb = await Bk.HeadBackend.create(device, man, W(), { init: { sm: s0.sm, bytes: s0.bytes } });
  await lb.loadSpike(s0.sm, s0.bytes);
  const v = exs[0].val.slice(0, 40), a = await lb.score(v), b = await hb.score(v);
  let d = 0, same = 0;
  a.forEach((z, i) => { z.forEach((x, j) => { d = Math.max(d, Math.abs(x - b[i][j])); }); same += Array.from(z).indexOf(Math.max(...z)) === Array.from(b[i]).indexOf(Math.max(...b[i])); });
  out.forward_check = { spike: s0.id, sequences: v.length, max_abs_logit_diff: d, argmax_same: same };
  ctx.log(`forward check ${s0.id}: max |dlogit| ${d.toExponential(2)}, argmax ${same}/${v.length}`);
  hb.destroy();
  if (d > 1e-2) throw new Error(`trainer and engine disagree on the ±31 base (max |dlogit| ${d})`);
}

for (let k = 0; k < shipped.length; k++) {
  const s = shipped[k], e = exs[k], t0 = performance.now();
  await lb.loadSpike(s.sm, s.bytes);
  const epochs = [];
  const res = await C.train(lb, { train: e.train, val: e.val, test: [] }, { epochs: cfg.epochs ?? 5, lr: 5e-4, bs: 16, seed: 0, startCandidate: cfg.startCandidate ?? true }, (ev) => {
    if (ev.kind === "epoch") epochs.push({ epoch: ev.epoch, trainLoss: ev.trainLoss, valKL: ev.val?.kl, valAcc: ev.val?.accuracy });
    if (ev.kind === "step" && ev.step % 10 === 0) ctx.progress(0.05 + 0.95 * (k + ev.step / ev.of) / shipped.length, `${s.id} step ${ev.step}/${ev.of}`);
  });
  const sm = { ...res.spike.sm, id: s.sm.id, description: s.sm.description, questions: s.sm.questions, temps: res.temps,
               refit: { base: "t4-w8L31", from: "shipped spike", epochs: cfg.epochs ?? 5, best_epoch: res.bestEpoch, start_candidate: cfg.startCandidate ?? true, sha: cfg.sha } };
  out.spikes[s.id] = { sm, bestEpoch: res.bestEpoch, epochs, ms: performance.now() - t0 };
  out[`bin_${s.id}`] = res.spike.bytes;
  ctx.log(`${s.id}: best epoch ${res.bestEpoch}, val acc ${epochs[0].valAcc?.toFixed(3)} -> ${res.val?.accuracy?.toFixed(3)} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}
return out;
