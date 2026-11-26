// Browser side of the parity check: runs every fixture case (make_ref.py) through OcriaGpu.infer and
// compares with onnxruntime's outputs for the same input. run_parity.mjs serves web/gpu/ at /gpu/ and
// test/data/ at /fix/ and calls the default export.
import { OcriaGpu } from "/gpu/engine.js";

const MODELS = ["det", "router", "rec_latin", "rec_world", "rec_rtl", "rec_cjk"];
const DET_THRESH = 0.55;

async function arr(meta) {
  const r = await fetch("/fix/" + meta.file);
  if (!r.ok) throw new Error(`${meta.file}: ${r.status}`);
  const b = await r.arrayBuffer();
  return meta.dtype === "int32" ? new Int32Array(b) : new Float32Array(b);
}

function maxAbs(a, b) {
  let d = 0;
  for (let i = 0; i < b.length; i++) { const e = Math.abs(a[i] - b[i]); if (!(e <= d)) d = e; }
  return d;
}

function argmaxRows(a, cols) {
  const out = [];
  for (let r = 0; r < a.length / cols; r++) {
    let best = 0;
    for (let c = 1; c < cols; c++) if (a[r * cols + c] > a[r * cols + best]) best = c;
    out.push(best);
  }
  return out;
}

// CTC collapse of ids[j, 0 .. t)
function collapse(ids, j, T, t) {
  const s = [];
  let prev = 0;
  for (let k = 0; k < t; k++) { const c = ids[j * T + k]; if (c !== 0 && c !== prev) s.push(c); prev = c; }
  return s.join(",");
}

// ids / conf agreement of one batch against one reference
function recCompare(out, ref, widths, stride, acc) {
  const [N, T] = out.ids.shape;
  for (let j = 0; j < N; j++) {
    const tv = Math.min(Math.floor(widths[j] / stride), T);
    for (let k = 0; k < T; k++) {
      const i = j * T + k;
      const same = out.ids.data[i] === ref.ids[i];
      acc.all += 1; acc.allSame += same ? 1 : 0;
      if (k < tv) {
        acc.valid += 1; acc.validSame += same ? 1 : 0;
        const d = Math.abs(out.conf.data[i] - ref.conf[i]);
        acc.confMax = Math.max(acc.confMax, d);
        acc.confSum += d;
      }
    }
    acc.lines += 1;
    acc.textSame += collapse(out.ids.data, j, T, tv) === collapse(ref.ids, j, T, tv) ? 1 : 0;
  }
}

const newAcc = () => ({ all: 0, allSame: 0, valid: 0, validSame: 0, confMax: 0, confSum: 0, lines: 0, textSame: 0 });
const finish = (a) => ({
  ids_agree_all: a.allSame / a.all, ids_agree_valid: a.validSame / a.valid, timesteps: a.all, valid_timesteps: a.valid,
  conf_max_abs: a.confMax, conf_mean_abs: a.confSum / a.valid, text_identical: a.textSame / a.lines, lines: a.lines,
});

export default async function run({ only } = {}) {
  const man = await (await fetch("/fix/manifest.json")).json();
  const models = only && only.length ? only : MODELS;
  const t0 = performance.now();
  const eng = await OcriaGpu.load("/gpu/models/", { models });
  const loadMs = performance.now() - t0;
  const adapter = eng.device.adapterInfo || {};
  // warm-up: compile, allocate
  const warm = {};
  for (const m of models) {
    const t = performance.now();
    if (m === "det") await eng.infer(m, [1, 3, 64, 64], new Float32Array(3 * 64 * 64));
    else if (m === "router") await eng.infer(m, [1, 1, 32, 512], new Float32Array(32 * 512));
    else await eng.infer(m, [1, 1, 32, 128], new Float32Array(32 * 128));
    warm[m] = Math.round(performance.now() - t);
  }
  const res = {};
  for (const m of models) res[m] = { cases: [], ms: 0, calls: 0, items: 0 };
  for (const c of man.cases) {
    if (!models.includes(c.model)) continue;
    const r = res[c.model];
    let x;
    if (c.model === "det") {
      const g = await arr(c.input_gray);
      x = new Float32Array(3 * g.length);
      x.set(g, 0); x.set(g, g.length); x.set(g, 2 * g.length);
    } else {
      x = await arr(c.input);
    }
    const t = performance.now();
    const out = await eng.infer(c.model, c.shape, x);
    const ms = performance.now() - t;
    r.ms += ms; r.calls += 1; r.items += c.shape[0];
    if (c.model === "det") {
      const ref = await arr(c.ref.prob), orig = await arr(c.orig.prob);
      const p = out.prob.data;
      if (p.length !== ref.length) throw new Error(`det ${c.id}: ${p.length} values, ref ${ref.length}`);
      let mis = 0;
      for (let i = 0; i < ref.length; i++) if ((p[i] > DET_THRESH) !== (ref[i] > DET_THRESH)) mis++;
      r.cases.push({ id: c.id, page: c.page, shape: c.shape, out_shape: out.prob.shape, max_abs: maxAbs(p, ref),
        max_abs_vs_fp32: maxAbs(p, orig), mask_mismatch: mis / ref.length, ms: Math.round(ms) });
    } else if (c.model === "router") {
      const cs = { id: c.id, n: c.shape[0], ms: Math.round(ms) };
      for (const k of ["group", "flip"]) {
        const ref = await arr(c.ref[k]), orig = await arr(c.orig[k]);
        const cols = k === "group" ? 4 : 2;
        const a = argmaxRows(out[k].data, cols), b = argmaxRows(ref, cols), bo = argmaxRows(orig, cols);
        cs[k] = { max_abs: maxAbs(out[k].data, ref), argmax_agree: a.filter((v, i) => v === b[i]).length / a.length,
          max_abs_vs_fp32: maxAbs(out[k].data, orig), argmax_agree_vs_fp32: a.filter((v, i) => v === bo[i]).length / a.length };
      }
      r.cases.push(cs);
    } else {
      const ref = { ids: await arr(c.ref.ids), conf: await arr(c.ref.conf) };
      const altKind = c.model === "rec_latin" ? "noopt" : "orig";
      const alt = { ids: await arr(c[altKind].ids), conf: await arr(c[altKind].conf) };
      if (out.ids.data.length !== ref.ids.length) throw new Error(`${c.id}: ${out.ids.data.length} ids, ref ${ref.ids.length}`);
      r.acc = r.acc || newAcc(); r.alt = r.alt || newAcc();
      recCompare(out, ref, c.widths, c.stride, r.acc);
      recCompare(out, alt, c.widths, c.stride, r.alt);
      r.altKind = altKind;
      r.cases.push({ id: c.id, shape: c.shape, T: out.ids.shape[1], ms: Math.round(ms) });
    }
  }
  const summary = {};
  let ok = true;
  for (const m of models) {
    const r = res[m];
    const s = { calls: r.calls, items: r.items, total_ms: Math.round(r.ms), ms_per_call: Math.round(r.ms / r.calls), warmup_ms: warm[m] };
    if (m === "det") {
      s.max_abs = Math.max(...r.cases.map((c) => c.max_abs));
      s.max_abs_vs_fp32 = Math.max(...r.cases.map((c) => c.max_abs_vs_fp32));
      s.mask_mismatch = Math.max(...r.cases.map((c) => c.mask_mismatch));
      s.pages = r.cases;
      s.pass = s.max_abs < 1e-3;
    } else if (m === "router") {
      for (const k of ["group", "flip"]) {
        s[k] = {
          max_abs: Math.max(...r.cases.map((c) => c[k].max_abs)),
          argmax_agree: r.cases.reduce((a, c) => a + c[k].argmax_agree * c.n, 0) / r.items,
          max_abs_vs_fp32: Math.max(...r.cases.map((c) => c[k].max_abs_vs_fp32)),
          argmax_agree_vs_fp32: r.cases.reduce((a, c) => a + c[k].argmax_agree_vs_fp32 * c.n, 0) / r.items,
        };
      }
      s.pass = s.group.max_abs < 1e-3 && s.flip.max_abs < 1e-3 && s.group.argmax_agree === 1 && s.flip.argmax_agree === 1;
    } else {
      s.vs_ref = finish(r.acc);
      s["vs_" + r.altKind] = finish(r.alt);
      s.lines_per_s = Math.round(1000 * r.items / r.ms * 10) / 10;
      s.batches = r.cases;
      s.pass = s.vs_ref.ids_agree_all >= (m === "rec_latin" ? 0.99 : 0.999);
    }
    ok = ok && s.pass;
    summary[m] = s;
  }
  return { ok, adapter: { vendor: adapter.vendor, architecture: adapter.architecture, description: adapter.description },
    load_ms: Math.round(loadMs), onnxruntime: man.onnxruntime, levels: man.levels, models: summary };
}
