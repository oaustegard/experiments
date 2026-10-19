import { Fluid2D, vortexBlobs, hsl } from './fluid2d.js';
import { Filaments } from './filament3d.js';
import { plot, energySpectrum, css } from './plots.js';
import { Cascade, ScalingWindow } from './cascade.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = (window.__ns = { ready: false, gpu: false, errors: [], frames: {}, sims: {} });
window.addEventListener('error', (e) => NS.errors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => NS.errors.push(String(e.reason)));

// ---------- visibility: only step what is on screen ----------
const visible = new Set();
const io = new IntersectionObserver((es) => {
  for (const e of es) (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id));
}, { rootMargin: '120px' });
$$('section[id]').forEach((s) => io.observe(s));
const loops = []; // { section, fn }
function onFrame(section, fn) { loops.push({ section, fn }); }
let lastT = performance.now();
function tick(t) {
  const dt = Math.min(0.05, (t - lastT) / 1000); lastT = t;
  for (const l of loops) {
    if (!visible.has(l.section) && !NS.forceAll) continue;
    try { l.fn(dt); NS.frames[l.section] = (NS.frames[l.section] ?? 0) + 1; } catch (e) { NS.errors.push(`${l.section}: ${e.message}`); }
  }
  requestAnimationFrame(tick);
}

function seg(el, onPick) {
  const bs = $$('button', el);
  bs.forEach((b) => b.addEventListener('click', () => { bs.forEach((x) => x.setAttribute('aria-pressed', x === b)); onPick(b.dataset.v); }));
}
function slider(id, fmt, on) {
  const el = $(id), out = $(`${id}-v`);
  const up = () => { const v = +el.value; if (out) out.textContent = fmt(v); on(v); };
  el.addEventListener('input', up); up();
  return el;
}
function sizeCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(2, Math.round(r.width * dpr)), h = Math.max(2, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
}

// ---------- chapter 1: the terms ----------
function chapter1(device) {
  const cv = $('#c1'); sizeCanvas(cv);
  const f = new Fluid2D(device, cv, { W: 320, H: 180, bc: 0, dt: 1, nu: 0, dyeDecay: 0.997, maccormack: true, mode: 0, gain: 1.6 });
  NS.sims.c1 = f;
  f.clear(); f.attachPointer({ radius: 5, strength: 1.2 });
  let demo = true, phase = 0;
  cv.addEventListener('pointerdown', () => { demo = false; $('#c1-hint').style.opacity = 0; });
  const gains = { 0: 1.6, 1: 4, 2: 0.9, 3: 25 };
  const setGain = () => { f.gain = f.mode === 3 && !f.terms.pressure ? 1.2 : gains[f.mode]; };
  seg($('#c1-view'), (v) => { f.mode = +v; setGain(); });
  const setTerm = (term, on) => {
    f.terms[term] = on;
    $$(`[data-term=${term}]`).forEach((el) => { el.classList.toggle('off', !on); if (el.matches('.switch')) el.setAttribute('aria-pressed', on); });
    $$('.term-note').forEach((n) => { n.hidden = n.dataset.for !== term || on; });
    setGain();
  };
  $$('[data-term]').forEach((el) => el.addEventListener('click', () => setTerm(el.dataset.term, !f.terms[el.dataset.term])));
  slider('#c1-nu', (v) => (10 ** v < 0.0011 ? '0' : (10 ** v).toPrecision(2)), (v) => { f.nu = 10 ** v < 0.0011 ? 0 : 10 ** v; });
  $('#c1-reset').addEventListener('click', () => f.clear());
  NS.setTerm = setTerm;
  onFrame('ch1', () => {
    sizeCanvas(cv);
    if (demo) {
      // A slow Lissajous stirrer until the reader takes over.
      phase += 0.022;
      const x = f.W * (0.5 + 0.33 * Math.sin(phase)), y = f.H * (0.5 + 0.3 * Math.sin(phase * 1.7 + 0.5));
      const vx = f.W * 0.33 * Math.cos(phase) * 0.022 * 6, vy = f.H * 0.3 * 1.7 * Math.cos(phase * 1.7 + 0.5) * 0.022 * 6;
      f.splat(x, y, vx, vy, hsl((phase * 0.05) % 1, 0.85, 0.5), 6);
    }
    f.step(); f.render();
  });
}

// ---------- chapter 2: Reynolds number ----------
function chapter2(device) {
  const cv = $('#c2'); sizeCanvas(cv);
  const W = 320, H = 108, U = 0.5, R = 11;
  const f = new Fluid2D(device, cv, { W, H, bc: 2, dt: 1, inflow: U, obst: [W * 0.2, H / 2 + 0.6, R, 1], dyeDecay: 0.9995, maccormack: true, diffuseIters: 40, pressureIters: 40, mode: 0, gain: 1.4 });
  NS.sims.c2 = f;
  f.clear(); f.attachPointer({ radius: 4, strength: 0.6 });
  const D = 2 * R;
  const regime = (re) => re < 5 ? 'Flow stays attached and wraps round the cylinder.'
    : re < 47 ? 'Two standing eddies sit behind the cylinder.'
      : re < 190 ? 'The wake oscillates and sheds a von Kármán street.'
        : 'In a real wake, the street itself turns three-dimensional and turbulent here. This grid is 2D and too coarse to follow.';
  slider('#c2-re', (v) => `${Math.round(10 ** v)}`, (v) => {
    const re = 10 ** v; f.nu = (U * D) / re; $('#c2-regime').textContent = regime(re); NS.re = re;
  });
  seg($('#c2-view'), (v) => { f.mode = +v; f.gain = +v === 1 ? 8 : 1.4; });
  $('#c2-reset').addEventListener('click', () => f.clear());
  onFrame('ch2', () => { sizeCanvas(cv); f.step(); f.step(); f.render(); });
}

// ---------- chapter 3: 2D vs 3D ----------
function chapter3(device) {
  const ca = $('#c3a'); sizeCanvas(ca);
  const N = 256;
  const f = new Fluid2D(device, ca, { W: N, H: N, bc: 1, dt: 1, nu: 0, dyeDecay: 1, maccormack: true, mode: 1, gain: 5 });
  NS.sims.c3a = f;
  let hist2 = [], w0 = 0;
  const seed2 = (s) => { const b = vortexBlobs(N, N, 18, { radius: 11, strength: 1.3, seed: s, periodic: true }); f.setFields(b.vel, b.dye); hist2 = []; w0 = 0; };
  seed2(7);
  f.attachPointer({ radius: 6, strength: 0.8 });
  f.onStats = (s) => { if (!w0) w0 = s.maxVort; hist2.push([s.time, s.maxVort / w0]); };
  $('#c3a-reset').addEventListener('click', () => seed2(1 + Math.floor(Math.random() * 1e6)));

  const cb = $('#c3b'); sizeCanvas(cb);
  const fil = new Filaments(device, cb, { M: 384 });
  NS.sims.c3b = fil;
  let hist3 = [];
  const load = (p) => { fil.load(p); hist3 = [[0, 1]]; };
  load('headon');
  fil.onStats = (s) => hist3.push([s.time, s.maxStretch]);
  seg($('#c3b-preset'), (v) => load(v));
  $('#c3b-reset').addEventListener('click', () => load(fil.preset));
  $('#c3b-smoke').addEventListener('change', (e) => { fil.showSmoke = e.target.checked; });
  let n = 0, hold = 0;
  onFrame('ch3', () => {
    sizeCanvas(ca); sizeCanvas(cb);
    f.step(); f.render();
    if (!fil.done) { fil.step(2); hold = 0; } else if (++hold > 180) load(fil.preset);
    fil.render();
    if (++n % 6 === 0) f.measure();
    if (n % 6 === 3) {
      plot($('#p3a'), [{ pts: hist2, color: css('--t-vis'), label: '2D: max|ω| ÷ start' }], { ylim: [0, 1.6], xlabel: 'time steps', hlines: [{ y: 1, color: css('--ink-2') }] });
      plot($('#p3b'), [{ pts: hist3, color: css('--t-adv'), label: '3D: max stretch = ω ÷ ω₀' }], { ylim: [0, Math.max(4, ...hist3.map((p) => p[1] * 1.15))], xlim: [0, fil.tmax], xlabel: 'time', hlines: [{ y: 1, color: css('--ink-2') }] });
    }
  });
  NS.hist3 = () => hist3; NS.hist2 = () => hist2;
}

// ---------- chapter 4: BKM ----------
function bkmToy() {
  const draw = (alpha) => {
    const T = 1, pts = [], ipts = [];
    let I = 0, last = 0;
    for (let i = 0; i <= 2000; i++) {
      const t = (T * i) / 2000 * 0.9995;
      const w = 1 / (T - t) ** alpha;
      I += w * (t - last); last = t;
      pts.push([t, w]); ipts.push([t, I]);
    }
    plot($('#p4a'), [
      { pts, color: css('--t-adv'), label: 'max|ω|(t) = (T* − t)^−α' },
      { pts: ipts, color: css('--accent'), label: '∫₀ᵗ max|ω| ds', dash: [5, 3] },
    ], { xlim: [0, 1], ylim: [0.5, 2000], logy: true, xlabel: 't  (T* = 1)', vlines: [{ x: 1, color: css('--ink-2'), label: 'T*' }] });
    const fin = alpha < 1;
    $('#p4a-verdict').innerHTML = fin
      ? `With α = ${alpha.toFixed(2)} the integral converges to ${(1 / (1 - alpha)).toFixed(2)} at T*. The vorticity is infinite at T*, and still the criterion says the solution does not break down there.`
      : `With α = ${alpha.toFixed(2)} the integral diverges at T*. This is the growth rate the criterion requires before a smooth solution can break down.`;
  };
  slider('#p4a-alpha', (v) => v.toFixed(2), draw);
}

function chapter4(device) {
  const cv = $('#c4'); sizeCanvas(cv);
  let f = null, hist = [], integral = 0, lastT = 0, spec = null;
  const make = (N) => {
    f = new Fluid2D(device, cv, { W: N, H: N, bc: 1, dt: 1, nu: f?.nu ?? 0.002, dyeDecay: 1, maccormack: true, mode: 1, gain: 5, forceMode: 1, forceAmp: 0.0006 * (N / 256), forceK: 4, diffuseIters: 12 });
    NS.sims.c4 = f;
    const b = vortexBlobs(N, N, 30, { radius: N / 26, strength: 0.6, seed: 3, periodic: true });
    f.setFields(b.vel, b.dye);
    f.attachPointer({ radius: 5, strength: 0.6 });
    hist = []; integral = 0; lastT = 0; spec = null;
    f.onStats = (s) => { integral += s.maxVort * (s.time - lastT); lastT = s.time; hist.push([s.time, s.maxVort, integral]); };
  };
  make(256);
  seg($('#c4-n'), (v) => make(+v));
  slider('#c4-nu', (v) => (v <= -3.99 ? '0' : (10 ** v).toPrecision(2)), (v) => { f.nu = v <= -3.99 ? 0 : 10 ** v; });
  const forceEl = $('#c4-force');
  forceEl.addEventListener('change', () => { f.terms.forcing = forceEl.checked; });
  $('#c4-reset').addEventListener('click', () => make(f.W));
  let n = 0, specBusy = false;
  onFrame('ch4', () => {
    sizeCanvas(cv);
    f.terms.forcing = forceEl.checked;
    f.step(); f.render();
    if (++n % 8 === 0) f.measure();
    if (n % 45 === 0 && !specBusy) {
      specBusy = true;
      const N = f.W;
      f.readVelocity().then((v) => { if (v && f.W === N) spec = energySpectrum(v, N); specBusy = false; }).catch(() => { specBusy = false; });
    }
    if (n % 8 === 4) {
      plot($('#p4b'), [
        { pts: hist.map((h) => [h[0], h[1]]), color: css('--t-adv'), label: 'max|ω|' },
        { pts: hist.map((h) => [h[0], h[2] / Math.max(1, h[0]) ]), color: css('--accent'), dash: [5, 3], label: '∫max|ω| dt ÷ t' },
      ], { xlabel: 'time steps', ylim: [0, Math.max(0.05, ...hist.map((h) => h[1] * 1.2))] });
      if (spec) {
        const N = f.W, kmax = N / 2;
        const ref = spec.find((p) => p[0] === 8)?.[1] ?? 1e-6;
        const ref3 = [[4, ref * 8], [kmax, ref * (8 / kmax) ** 3]];
        plot($('#p4c'), [
          { pts: spec, color: css('--t-vis'), label: 'E(k)' },
          { pts: ref3, color: css('--ink-2'), dash: [4, 4], label: 'k⁻³ reference' },
        ], { logx: true, logy: true, xlim: [1, 256], ylim: [Math.max(1e-14, Math.min(...spec.map((p) => p[1]).filter((v) => v > 0)) * 0.5), Math.max(...spec.map((p) => p[1])) * 3], xlabel: 'wavenumber k', vlines: [{ x: kmax, color: css('--t-adv'), label: `grid limit k = N/2 = ${kmax}` }] });
      }
    }
  });
  NS.spec = () => spec; NS.hist4 = () => hist;
}

// ---------- chapter 5: cascade ----------
function chapter5() {
  const c = new Cascade($('#c5'), $('#c5-readout'));
  NS.sims.c5 = c;
  const win = new ScalingWindow($('#c5w'), (l, m) => { $('#c5-lambda').value = l; $('#c5-mu').value = m; sync(); });
  const verdict = () => {
    const l = c.lambda, m = c.mu, d = c.derived(0);
    const parts = [];
    parts.push(`T* = ${d.Tstar.toFixed(3)}.`);
    if (m > l * l) parts.push(`μ > λ², so each layer's Reynolds number is ${(1 / d.ratioRe).toFixed(2)}× smaller than the last; starting from 2000, viscosity wins by about layer ${d.dies}.`);
    else parts.push(`μ ≤ λ², so the Reynolds number does not fall from layer to layer and viscosity never catches up.`);
    if (d.ratioE3 > 1.0001) parts.push(`In 3D each layer would need ${d.ratioE3.toFixed(2)}× the energy of the last, which is impossible for a fluid that starts with finite energy.`);
    else parts.push('In 3D the layer energies shrink, so finite energy allows it.');
    if (d.ratioE2 > 1.0001 && m <= l * l) parts.push('In 2D the same cascade would need growing energy: there the two conditions only meet on the line μ = λ².');
    $('#c5-verdict').textContent = parts.join(' ');
  };
  const sync = () => {
    const l = +$('#c5-lambda').value, m = +$('#c5-mu').value;
    $('#c5-lambda-v').textContent = l.toFixed(2); $('#c5-mu-v').textContent = m.toFixed(2);
    c.setParams(l, m); win.set(l, m); verdict();
  };
  $('#c5-lambda').addEventListener('input', sync); $('#c5-mu').addEventListener('input', sync);
  sync();
  $('#c5-play').addEventListener('click', (e) => { c.playing = !c.playing; e.target.textContent = c.playing ? 'Pause' : 'Play'; });
  onFrame('ch5', (dt) => { c.frame(dt); });
  window.addEventListener('resize', () => win.draw());
}

// ---------- boot ----------
async function boot() {
  bkmToy();
  chapter5();
  let device = null;
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    NS.adapter = adapter; // keep the adapter reachable: Chromium drops the device when it is collected
    device = await adapter?.requestDevice();
  } catch (e) { NS.errors.push(`gpu: ${e.message}`); }
  if (!device) {
    document.body.classList.add('nogpu');
  } else {
    NS.gpu = true;
    device.addEventListener('uncapturederror', (e) => NS.errors.push(`webgpu: ${e.error.message}`));
    device.lost.then((i) => NS.errors.push(`device lost: ${i.message}`));
    const only = new URLSearchParams(location.search).get('only');
    for (const [i, ch] of [chapter1, chapter2, chapter3, chapter4].entries()) {
      if (only && !only.split(',').includes(String(i + 1))) continue;
      try { ch(device); } catch (e) { NS.errors.push(`${ch.name}: ${e.message}`); console.error(e); }
    }
  }
  NS.ready = true;
  requestAnimationFrame(tick);
}
boot();
