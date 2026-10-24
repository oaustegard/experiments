// Validate refractometer on Dawn (npm `webgpu`) in Node: compile every WGSL variant, run every
// sweep's verification against the JS references, run negative controls (deliberately broken
// kernels must be rejected), and exercise both timing paths. Writes ../results/<name>.json.
//
//   (cd run && npm install)   # Dawn for Node; no adapter? apt-get install -y mesa-vulkan-drivers
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json node run/check.mjs [--name lavapipe_check]
import { create, globals } from 'webgpu';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as R from '../web/refract.js';
Object.assign(globalThis, globals);

const here = fileURLToPath(new URL('.', import.meta.url));
const name = process.argv.includes('--name') ? process.argv[process.argv.indexOf('--name') + 1] : 'lavapipe_check';
const out = { started: new Date().toISOString() };
const t0 = performance.now();
const log = (...a) => console.log(`[${((performance.now() - t0) / 1000).toFixed(1)}s]`, ...a);

const ctx = await R.openDevice(create([]));
out.adapter = { info: ctx.info, key: ctx.key, features: ctx.features, timing: ctx.timing };
log('adapter', ctx.key, ctx.features.join(','));

// 1. Compile everything.
const ks = R.allKernels(ctx);
const bad = [];
for (const k of ks) {
  try { await R.pipeline(ctx, k.code, k.label); } catch (e) { bad.push({ label: k.label, error: e.message }); }
}
out.compile = { kernels: ks.length, failed: bad.length, failures: bad };
log(`compiled ${ks.length} kernels, ${bad.length} failed`);
for (const b of bad.slice(0, 5)) log('  ', b.error.slice(0, 300));

// 2. Verify every config of every full sweep (one dispatch + compare, no timing).
const summarize = (r) => {
  const by = {};
  for (const p of r.points ?? []) by[p.status] = (by[p.status] ?? 0) + 1;
  return { skipped: r.skipped, statuses: by, errors: (r.points ?? []).filter((p) => p.status !== 'verified' && p.status !== 'ok').slice(0, 5).map((p) => ({ config: p.config, status: p.status, error: p.error })), maxErr: Math.max(0, ...(r.points ?? []).filter((p) => p.verify).map((p) => p.verify.maxErr / (typeof p.verify.tol === 'number' && p.verify.tol > 0 ? p.verify.tol : 1))) };
};
out.verify = {};
for (const [id, s] of Object.entries(R.SWEEPS)) {
  const t = performance.now();
  const r = await s.run(ctx, { verifyOnly: true, maxBytes: 128 << 20 });
  out.verify[id] = { ...summarize(r), seconds: +((performance.now() - t) / 1000).toFixed(1) };
  log('verify', id, JSON.stringify(out.verify[id].statuses), out.verify[id].skipped ?? '');
}

// 3. Negative controls: broken variants must come back 'wrong', never 'ok'.
const orig = { ...R.WGSL };
const mutants = {
  reduce: (c) => c.replace('var acc = 0.0;', 'var acc = select(0.0, -1.0, l == 0u);'),
  gemvQ4: (c) => c.replace('- z;\n      let hi', '- z + 0.0625;\n      let hi'),
  scan2: (c) => c.replace('6u - 2u * m', '2u * m'),
  matmul: (c) => c.replace('col0 + lid.x + j * WX] = f32', 'col0 + (lid.x + 1u) % WX + j * WX] = f32'),
  copy: (c) => c.replace('dst[i] = src[i]', 'dst[i] = src[min(i + 1u, N4 - 1u)]'),
};
out.negative = {};
for (const [k, mut] of Object.entries(mutants)) R.WGSL[k] = (a) => mut(orig[k](a));
for (const [id, opts] of [['bandwidth', { maxBytes: 64 << 10 }], ['reduce', { quick: true }], ['gemv', { quick: true, shapes: [R.EG2_SHAPES[3]] }], ['scan', { quick: true }], ['matmul', { quick: true }]]) {
  const r = await R.SWEEPS[id].run(ctx, { verifyOnly: true, ...opts });
  const st = {};
  for (const p of r.points) st[p.status] = (st[p.status] ?? 0) + 1;
  out.negative[id] = st;
  log('negative', id, JSON.stringify(st));
}
Object.assign(R.WGSL, orig);

// 4. Timing paths: a quick timed run per sweep, timestamp then wall.
out.timed = {};
for (const mode of ctx.features.includes('timestamp-query') ? ['timestamp', 'wall'] : ['wall']) {
  ctx.timing = mode;
  const res = {};
  for (const id of ['bandwidth', 'reduce', 'subgroup']) {
    res[id] = await R.SWEEPS[id].run(ctx, { quick: true, reps: 5, warmup: 1, maxBytes: 4 << 20, targetMs: 20 });
  }
  res.gemv = await R.SWEEPS.gemv.run(ctx, { quick: true, reps: 5, warmup: 1, targetMs: 20, shapes: [R.EG2_SHAPES[3]] });
  const prof = R.makeProfile(ctx, res, { note: 'lavapipe (CPU) timings: exercise the code path only' });
  out.timed[mode] = { best: prof.best, modes: [...new Set(Object.values(res).flatMap((r) => (r.points ?? []).map((p) => p.time?.mode)).filter(Boolean))], quantized: Object.values(res).some((r) => (r.points ?? []).some((p) => p.time?.quantized)) };
  if (mode === 'timestamp') writeFileSync(`${here}../results/${name}_profile.json`, JSON.stringify(prof) + '\n');
  log('timed', mode, JSON.stringify(out.timed[mode].modes));
}
out.uncaptured = ctx.uncaptured;
out.lost = ctx.lost.reason;
out.seconds = +((performance.now() - t0) / 1000).toFixed(1);
writeFileSync(`${here}../results/${name}.json`, JSON.stringify(out, null, 1));
log('done');
process.exit(bad.length || ctx.uncaptured.length ? 1 : 0);
