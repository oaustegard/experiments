// GPU Bridge job 2: round-2 engine (fused conv+pool, y-LSTM v2, x-LSTM v2), greedy vs beam
// decoding cost, per-stage GPU time, text parity with the reference (greedy and beam).
const SHA = "2bec748379e66b1a9fdaa24d95aaa475686e12f8";
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${SHA}/tesseract-gpu/web/`;
const E = await ctx.importModule(base + "engine.js");
const dec = new TextDecoder();
const inp = ctx.inputs;
const widths = Array.from(inp.widths);
const H = 36;
const all = [];
let off = 0;
for (const w of widths) { all.push({ data: inp.lines.subarray(off, off + H * w), width: w, height: H }); off += H * w; }
const expect = JSON.parse(dec.decode(inp.expect));
const gpu = await E.Gpu.create();
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const order = widths.map((w, i) => [w, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
const pick = (n) => (n === 1 ? [all[order[Math.floor(order.length / 2)]]] : all.slice(0, n));
const out = { device: gpu.info, models: {} };
const plan = [["fast", inp.man_fast, inp.w_fast], ["best", inp.man_best, inp.w_best]];
let done = 0;
const total = plan.length * (2 * 4 + 6);
for (const [name, manB, wB] of plan) {
  const man = JSON.parse(dec.decode(manB));
  const model = new E.TessModel(gpu, man, wB.buffer.slice(wB.byteOffset, wB.byteOffset + wB.byteLength));
  const r = { batches: {}, stages: {}, parity: {} };
  for (const decoder of ["greedy", "beam"]) {
    for (const n of [1, 16, 64, all.length]) {
      const lines = pick(n);
      await E.recognize(gpu, model, lines, { decoder });
      const runs = [];
      let res;
      for (let k = 0; k < 3; k++) { res = await E.recognize(gpu, model, lines, { decoder }); runs.push(res.ms); }
      const m = {};
      for (const key of ["prep", "gpu", "readback", "decode", "total"]) m[key] = med(runs.map((x) => x[key]));
      r.batches[`${decoder}_${n}`] = { lines: n, steps: res.ms.steps, px: lines.reduce((a, l) => a + l.width, 0), ...m };
      if (n === all.length) {
        const want = decoder === "beam" ? expect[name + "_beam"] : expect[name];
        r.parity[decoder] = res.filter((o, i) => o.text === want[i]).length + "/" + n;
        r["mismatch_" + decoder] = res.map((o, i) => [o.text, want[i]]).filter(([a, b]) => a !== b).slice(0, 4);
      }
      ctx.progress(++done / total, `${name} ${decoder} ${n}`);
    }
  }
  for (let st = 1; st <= 6; st++) {
    const g = [];
    for (let k = 0; k < 3; k++) g.push((await E.recognize(gpu, model, all, { stages: st })).ms.gpu);
    r.stages[st] = med(g);
    ctx.progress(++done / total, `${name} stage ${st}`);
  }
  out.models[name] = r;
  ctx.log(name, JSON.stringify(r.batches[`beam_${all.length}`]), "parity", JSON.stringify(r.parity));
}
return out;
