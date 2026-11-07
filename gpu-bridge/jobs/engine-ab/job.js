// Timing and output A/B of the decision engines at two commits, on the same weights and inputs: the
// gate for a kernel-library migration step (outputs identical or within tolerance, time within 5%).
// Loads each commit's engine through jsDelivr and the weights from the live site. Rounds alternate
// a/b order; reports median ms per score() call per batch shape, and max |diff| between a and b.
// inputs.cfg = { a: sha, b: sha, reps: 30, rounds: 6, engines: ["decider", "spike"] }
const cfg = ctx.inputs.cfg;
const GH = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@";
const SITE = "https://austegard.com/experiments/";
const dev = await ctx.gpu();
const bytes = async (url) => { if (ctx.cache.has(url)) return ctx.cache.get(url); const b = await ctx.fetchBytes(url); ctx.cache.set(url, b); return b; };
const json = async (url) => JSON.parse(new TextDecoder().decode(await bytes(url)));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
let s = 7; const rnd = (n) => { s = (s * 1103515245 + 12345) % 2147483648; return s % n; };
const seq = (len, npos) => ({ ids: Array.from({ length: len }, (_, i) => i === 0 ? 50281 : 1000 + rnd(40000)), pos: Array.from({ length: npos }, (_, i) => 1 + i * 3) });
const batches = { short: [seq(60, 3)], case5: Array.from({ length: 5 }, () => seq(300, 4)), long: [seq(700, 5), seq(129, 2)] };

const ENGINES = {
  decider: { file: "browser-decider/web/engine.js", cls: "DeciderEngine", opts: { maxTokens: 4096, maxSeqs: 16, maxPos: 256 },
    load: async () => { const b = SITE + "browser-decider/web/models/decider/"; return { m: await json(b + "manifest.json"), w: await bytes(b + "weights.bin") }; } },
  spike: { file: "adapter-library/web/engine.js", cls: "SpikeEngine", opts: { maxTokens: 4096, maxSeqs: 16, maxPos: 2048 },
    load: async () => {
      const b = SITE + "adapter-library/web/models/workflows/", sys = await json(b + "system.json");
      const bb = new URL(sys.base ?? "base/", b).href;
      const f = sys.spikes[0].file;
      return { m: await json(bb + "manifest.json"), w: await bytes(bb + "weights.bin"), spike: [await json(b + f + ".json"), new Uint8Array(await bytes(b + f + ".bin"))] };
    } },
};

const out = { cfg, engines: {} };
const names = cfg.engines ?? Object.keys(ENGINES);
for (const [ni, name] of names.entries()) {
  const E = ENGINES[name], { m, w, spike } = await E.load();
  const eng = {};
  for (const v of ["a", "b"]) {
    const mod = await ctx.importModule(`${GH}${cfg[v]}/${E.file}`);
    eng[v] = await mod[E.cls].create(dev, structuredClone(m), w, E.opts);
    if (spike) eng[v].setSpike(spike[0], spike[1]);
  }
  const res = {};
  for (const [bn, batch] of Object.entries(batches)) {
    const za = await eng.a.score(batch), zb = await eng.b.score(batch);
    let maxDiff = 0;
    za.forEach((x, i) => { for (let j = 0; j < x.length; j++) maxDiff = Math.max(maxDiff, Math.abs(x[j] - zb[i][j])); });
    const t = { a: [], b: [] };
    for (let r = 0; r < (cfg.rounds ?? 6); r++) {
      ctx.progress((ni + r / (cfg.rounds ?? 6)) / names.length, `${name} ${bn} round ${r + 1}`);
      for (const v of r % 2 ? ["b", "a"] : ["a", "b"]) {
        const t0 = performance.now();
        for (let k = 0; k < (cfg.reps ?? 30); k++) await eng[v].score(batch);
        t[v].push((performance.now() - t0) / (cfg.reps ?? 30));
      }
    }
    res[bn] = { tokens: batch.reduce((x, q) => x + q.ids.length, 0), maxDiff, aMs: median(t.a), bMs: median(t.b), ratio: median(t.b) / median(t.a) };
    ctx.log(`${name} ${bn}: a ${res[bn].aMs.toFixed(2)} ms, b ${res[bn].bMs.toFixed(2)} ms, max |diff| ${maxDiff}`);
  }
  out.engines[name] = res;
  eng.a.destroy(); eng.b.destroy();
}
return out;
