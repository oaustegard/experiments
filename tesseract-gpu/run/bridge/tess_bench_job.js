// GPU Bridge job: tesseract-gpu WebGPU engine on a real GPU.
// Inputs: lines (u8, 343 English benchmark lines at height 36, concatenated), widths (u32),
// man_best / man_fast (manifest.json bytes), w_best / w_fast (weights.bin), expect (JSON bytes:
// the NumPy reference's greedy text per line for best and fast).
const SHA = "fce8860e2a799a5b527b03c79c1b84c261e111a5";
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
ctx.log("adapter", JSON.stringify(gpu.info));
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const order = widths.map((w, i) => [w, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
const pick = (n) => (n === 1 ? [all[order[Math.floor(order.length / 2)]]] : all.slice(0, n));
const out = { device: gpu.info, models: {} };
const plan = [["fast", inp.man_fast, inp.w_fast], ["best", inp.man_best, inp.w_best]];
let done = 0;
const total = plan.length * 2 * 4 + plan.length * 6;
for (const [name, manB, wB] of plan) {
  const man = JSON.parse(dec.decode(manB));
  const model = new E.TessModel(gpu, man, wB.buffer.slice(wB.byteOffset, wB.byteOffset + wB.byteLength));
  const r = { batches: {}, stages: {}, parity: {} };
  for (const ver of ["v2", "v1"]) {
    for (const n of [1, 16, 64, all.length]) {
      const lines = pick(n);
      await E.recognize(gpu, model, lines, { xlstm: ver }); // warm (pipelines)
      const ms = [], gms = [];
      let res;
      for (let k = 0; k < 3; k++) { res = await E.recognize(gpu, model, lines, { xlstm: ver }); ms.push(res.ms.total); gms.push(res.ms.gpu); }
      r.batches[`${ver}_${n}`] = { lines: n, total_ms: med(ms), gpu_ms: med(gms), prep_ms: res.ms.prep, steps: res.ms.steps, px: lines.reduce((a, l) => a + l.width, 0) };
      if (n === all.length) {
        const want = expect[name];
        r.parity[ver] = res.filter((o, i) => o.text === want[i]).length + "/" + n;
        if (ver === "v2") r.mismatch = res.map((o, i) => [o.text, want[i]]).filter(([a, b]) => a !== b).slice(0, 5);
      }
      ctx.progress(++done / total, `${name} ${ver} ${n}`);
    }
  }
  for (let st = 1; st <= 6; st++) {
    const g = [];
    for (let k = 0; k < 3; k++) g.push((await E.recognize(gpu, model, all, { stages: st })).ms.gpu);
    r.stages[st] = med(g);
    ctx.progress(++done / total, `${name} stage ${st}`);
  }
  out.models[name] = r;
  ctx.log(name, JSON.stringify(r.batches[`v2_${all.length}`]), "parity", JSON.stringify(r.parity));
}
return out;
