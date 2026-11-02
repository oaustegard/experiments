// GPU Bridge job: compare TF.js WebGPU against TF.js CPU, values and gradients, for the ops the
// image-to-wgsl trainer uses, at the trainer's shapes (P pixels x G splats per chunk).
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
              "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js");
const P = ctx.inputs.cfg.P, GS = ctx.inputs.cfg.G;

function rand(n, lo, hi, seed) { let a = seed >>> 0; const out = new Float32Array(n);
  for (let i = 0; i < n; i++) { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); out[i] = lo + (hi - lo) * (((t ^ (t >>> 14)) >>> 0) / 4294967296); } return out; }

function strictUpper(G) { const a = new Float32Array(G * G); for (let l = 0; l < G; l++) for (let j = 0; j < l; j++) a[l * G + j] = 1; return tf.tensor2d(a, [G, G]); }

// Each case: f(inputs) -> scalar; inputs are [G,P] weights w, [G,3] colours col, [P,3] colour c.
const CASES = {
  transpose: (w) => tf.sum(w.transpose().mul(tf.range(0, w.shape[0]).reshape([1, -1]))),
  cumsum_last: (w) => { const L = tf.cumsum(w.transpose(), 1); return tf.sum(L.mul(L)); },
  matmul_tri: (w) => { const L = tf.matMul(w.transpose(), strictUpper(w.shape[0])); return tf.sum(L.mul(L)); },
  matmul_col: (w, col) => { const y = tf.matMul(w.transpose(), col); return tf.sum(y.mul(y)); },
  composite_tri: (w, col, c) => {
    const wt = w.transpose(), lom = tf.log(tf.sub(1, tf.minimum(wt, 0.9999)));
    const after = tf.exp(tf.matMul(lom, strictUpper(wt.shape[1]))), total = tf.exp(tf.sum(lom, 1, true));
    const out = c.mul(total).add(tf.matMul(wt.mul(after), col)); return tf.sum(out.mul(out));
  },
  composite_cumsum: (w, col, c) => {
    const wt = w.transpose(), G = wt.shape[1];
    const L = tf.cumsum(tf.log(tf.sub(1, tf.minimum(wt, 0.9999))), 1), Lend = L.slice([0, G - 1], [-1, 1]);
    const out = c.mul(tf.exp(Lend)).add(tf.matMul(wt.mul(tf.exp(Lend.sub(L))), col)); return tf.sum(out.mul(out));
  },
};

async function evalOn(backend, name, G) {
  await tf.setBackend(backend); await tf.ready();
  const w = tf.tensor2d(rand(G * P, 0, 0.6, 1 + G), [G, P]), col = tf.tensor2d(rand(G * 3, 0, 1, 2 + G), [G, 3]),
        c = tf.tensor2d(rand(P * 3, 0, 1, 3), [P, 3]);
  const f = CASES[name];
  const { value, grads } = tf.valueAndGrads((a, b, d) => f(a, b, d).add(tf.sum(b).mul(1e-30)).add(tf.sum(d).mul(1e-30)))([w, col, c]);
  const out = { value: (await value.data())[0], gw: await grads[0].data(), gcol: await grads[1].data() };
  [w, col, c, value, ...grads].forEach((t) => t.dispose());
  return out;
}

function relErr(a, b) { let num = 0, den = 0; for (let i = 0; i < a.length; i++) { num = Math.max(num, Math.abs(a[i] - b[i])); den = Math.max(den, Math.abs(b[i])); } return num / Math.max(den, 1e-30); }

const rows = [];
let k = 0;
for (const name of Object.keys(CASES)) for (const G of GS) {
  try {
    const c = await evalOn("cpu", name, G);
    for (const be of ctx.inputs.cfg.backends) {
      const g = await evalOn(be, name, G);
      rows.push({ backend: be, op: name, G, valueRel: Math.abs(g.value - c.value) / Math.max(Math.abs(c.value), 1e-30),
                  gradWRel: relErr(g.gw, c.gw), gradColRel: relErr(g.gcol, c.gcol) });
    }
  } catch (e) { rows.push({ op: name, G, error: String(e.message || e).slice(0, 300) }); }
  ctx.progress(++k / (Object.keys(CASES).length * GS.length), `${name} G=${G}`);
}
for (const r of rows) ctx.log(JSON.stringify(r));
return { P, rows };
