// Per-kernel timing of a WGSL engine at two commits: which kernel class moved between them.
// For each engine, embeds one query (so the batch buffers hold it), then for each kernel class
// (matmul*, norm*, attn, ...) records a pass of only that class's dispatches, `reps` times over,
// and times submit-to-done. Classes are matched by name prefix, so a renamed variant
// (matmul -> matmul_q4) still pairs up. Same loading as jobs/wgsl-ab.
// inputs.cfg = { a: sha, b: sha, reps: 50, rounds: 4, full: false, tokens: 16 }
const cfg = ctx.inputs.cfg;
const GH = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@";
const SITE = "https://austegard.com/bsky/";
const HF = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/onnx/model_q4.onnx_data";
const dev = await ctx.gpu();
const bytes = async (url) => {
  if (ctx.cache.has(url)) return ctx.cache.get(url);
  const b = await ctx.fetchBytes(url);
  ctx.cache.set(url, b);
  return b;
};
const json = async (url) => JSON.parse(new TextDecoder().decode(await bytes(url)));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const cls = (kind) => kind.split("_")[0];

const q = cfg.b;
const models = [
  { name: "eg2-cut", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    manifest: `${GH}${q}/eg2-wgsl-engine/web/manifest.json`, data: `${SITE}image-search/model/ort/model_t3w.onnx_data` },
  { name: "ettin-17m-t4", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    manifest: `${SITE}embedders/models/ettin/manifest.json`, data: `${SITE}embedders/models/ettin/model.onnx` },
];
if (cfg.full) models.push({ name: "eg2-full", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
  manifest: `${GH}${q}/eg2-wgsl-engine/web/manifest_full.json`, data: HF });

const ids = Array.from({ length: cfg.tokens ?? 16 }, (_, i) => 2 + i * 7);
const out = { cfg, models: {} };
for (const m of models) {
  const [man, data] = await Promise.all([json(m.manifest), bytes(m.data)]);
  const eng = {};
  for (const v of ["a", "b"]) {
    const mod = await import(`${GH}${cfg[v]}/${m.file}`);
    eng[v] = await mod[m.cls].create(dev, man, data, { maxTokens: 512, maxSeqs: 16 });
    await eng[v].embed([ids]);
  }
  const n = [ids.length, 1];
  const classes = [...new Set(eng.a.steps.map((s) => cls(s.kind)))];
  const time = async (e, c) => {
    const steps = e.steps.filter((s) => cls(s.kind) === c);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let r = 0; r < cfg.reps; r++) for (const s of steps) { pass.setPipeline(e.pipes[s.kind]); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid(n)); }
    pass.end();
    const t = performance.now();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    return { ms: (performance.now() - t) / cfg.reps, count: steps.length };
  };
  const res = {};
  for (let r = 0; r < cfg.rounds; r++) {
    ctx.progress(r / cfg.rounds, `${m.name}: round ${r + 1} of ${cfg.rounds}`);
    for (const c of classes) {
      for (const v of r % 2 ? ["b", "a"] : ["a", "b"]) {
        const x = await time(eng[v], c);
        const k = (res[c] = res[c] ?? { a: [], b: [], count_a: 0, count_b: 0 });
        k[v].push(x.ms); k[`count_${v}`] = x.count;
      }
    }
  }
  out.models[m.name] = Object.fromEntries(Object.entries(res).map(([c, k]) =>
    [c, { count_a: k.count_a, count_b: k.count_b, msPerForward_a: +median(k.a).toFixed(4), msPerForward_b: +median(k.b).toFixed(4) }]));
  ctx.log(m.name, JSON.stringify(out.models[m.name]));
  eng.a.destroy(); eng.b.destroy();
}
return out;
