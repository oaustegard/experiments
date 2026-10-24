// CPU rasterizer that ports the two WGSL passes in web/index.html line for line
// (bone density/glow pass + radiograph composite), used to make test-frame.png
// headlessly. The geometry comes from the page's own XR block (extracted, not copied).
// Inputs: real MediaPipe Pose Landmarker (lite) output for pose.jpg, and AP-10K
// ground-truth keypoints for two mmpose test images driving the dog and cat templates.
const fs = require('fs');
const path = require('path');
const jpeg = require('jpeg-js');
const { PNG } = require('pngjs');
const src = require('./extract.js');
const XR = (() => { const module = { exports: {} }; new Function('module', src.skeleton)(module); return module.exports; })();

const D = __dirname;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;

function tube(x, shell) {
  const outer = Math.sqrt(Math.max(1 - x * x, 0));
  const ri = Math.max(1 - shell, 0.001);
  const xi = x / ri;
  const inner = Math.sqrt(Math.max(1 - xi * xi, 0)) * ri;
  return Math.pow(Math.max(outer - 0.75 * inner, 0), 0.8);
}

// Bone pass: additive density, max glow (same blend states as the page's pipeline).
function rasterBones(pr, W, H, fadeMul = 1) {
  const dens = new Float32Array(W * H), glow = new Float32Array(W * H);
  const f = pr.f;
  for (let n = 0; n < pr.n; n++) {
    const o = n * 12;
    const a = [f[o], f[o + 1], f[o + 2], f[o + 3]], b = [f[o + 4], f[o + 5], f[o + 6], f[o + 7]], c = [f[o + 8], f[o + 9], f[o + 10], f[o + 11]];
    const t = Math.round(c[0]);
    const g = c[3] * 3 + 1.5;
    let lo, hi;
    if (t === 0) { const r = Math.max(b[0], b[1]) * (1 + b[2]) + g; lo = [Math.min(a[0], a[2]) - r, Math.min(a[1], a[3]) - r]; hi = [Math.max(a[0], a[2]) + r, Math.max(a[1], a[3]) + r]; }
    else { const r = Math.max(a[2], a[3]) * 1.42 + g; lo = [a[0] - r, a[1] - r]; hi = [a[0] + r, a[1] + r]; }
    const x0 = Math.max(0, Math.floor(lo[0])), x1 = Math.min(W - 1, Math.ceil(hi[0])), y0 = Math.max(0, Math.floor(lo[1])), y1 = Math.min(H - 1, Math.ceil(hi[1]));
    const I = b[3] * fadeMul;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = x + 0.5, py = y + 0.5;
      let d = 0, sd = 1e9, keep = true;
      if (t === 0) {
        const bax = a[2] - a[0], bay = a[3] - a[1], pax = px - a[0], pay = py - a[1];
        const h = (pax * bax + pay * bay) / Math.max(bax * bax + bay * bay, 1e-6);
        const flags = Math.round(c[2]);
        if ((h < 0 && !(flags & 1)) || (h > 1 && !(flags & 2))) keep = false;
        const hc = clamp(h, 0, 1);
        const bump = Math.exp(-Math.pow(hc / 0.08, 2)) + Math.exp(-Math.pow((1 - hc) / 0.08, 2));
        const R = Math.max(mix(b[0], b[1], hc) * (1 + b[2] * bump), 0.5);
        const q = Math.hypot(pax - bax * hc, pay - bay * hc);
        sd = q - R; d = tube(q / R, c[1]);
      } else if (t === 1) {
        const cs = Math.cos(-b[0]), sn = Math.sin(-b[0]);
        const vx = px - a[0], vy = py - a[1];
        const lx = (cs * vx - sn * vy) / Math.max(a[2], 0.5), ly = (sn * vx + cs * vy) / Math.max(a[3], 0.5);
        const k = Math.hypot(lx, ly);
        sd = (k - 1) * Math.min(a[2], a[3]); d = tube(k, c[1]);
      } else {
        const cs = Math.cos(-b[0]), sn = Math.sin(-b[0]);
        const vx = px - a[0], vy = py - a[1];
        const lx = cs * vx - sn * vy, ly = sn * vx + cs * vy;
        const cr = b[1];
        const qx = Math.abs(lx) - a[2] + cr, qy = Math.abs(ly) - a[3] + cr;
        sd = Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - cr;
        const m = Math.min(a[2], a[3]);
        d = (1 - smoothstep(-1, 0.5, sd)) * (0.55 + 0.45 * smoothstep(-0.3 * m, 0, sd));
      }
      if (!keep) continue;
      const aa = 1 - smoothstep(-0.75, 0.75, sd);
      const gg = Math.max(I, 0) * Math.exp(-Math.max(sd, 0) / Math.max(c[3], 1));
      const i = y * W + x;
      dens[i] += d * I * aa;
      glow[i] = Math.max(glow[i], gg * 0.55);
    }
  }
  return { dens, glow };
}

function hash(x, y) { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); }

// Composite pass. Here screen == video (no cover crop, no mirror), so toVid is identity in pixels.
function composite(img, W, H, mask, maskW, maskH, bones, tissue, timeSec = 1.234) {
  const L = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) L[i] = (0.299 * img[i * 4] + 0.587 * img[i * 4 + 1] + 0.114 * img[i * 4 + 2]) / 255;
  const lum = (x, y) => L[clamp(Math.round(y), 0, H - 1) * W + clamp(Math.round(x), 0, W - 1)];
  const out = Buffer.alloc(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const px = x + 0.5, py = y + 0.5;
    const l = lum(x, y);
    const gx = lum(x + 1.5, y) - lum(x - 1.5, y), gy = lum(x, y + 1.5) - lum(x, y - 1.5);
    const edge = clamp(Math.hypot(gx, gy) * 2.5, 0, 1);
    let film = Math.pow(1 - l, 1.6) * 0.38 + edge * 0.55;
    let m = 0;
    if (mask) m = mask[clamp(Math.floor(py / H * maskH), 0, maskH - 1) * maskW + clamp(Math.floor(px / W * maskW), 0, maskW - 1)] / 255;
    film = mix(film, film * (0.4 + 0.6 * m) + m * 0.2, tissue ? 1 : 0);
    const i = y * W + x;
    const Dd = Math.max(bones.dens[i], 0), G = bones.glow[i];
    const bb = 1 - Math.exp(-1.4 * Dd);
    const fc = clamp(film, 0, 1);
    let col = [mix(0.006, 0.42, fc), mix(0.02, 0.72, fc), mix(0.04, 0.95, fc)];
    col = [mix(col[0], 0.9, bb) + 0.25 * G * 0.5, mix(col[1], 0.97, bb) + 0.6 * G * 0.5, mix(col[2], 1.0, bb) + 1.0 * G * 0.5];
    const fr = (v) => v - Math.floor(v);
    const grain = (hash(px + fr(timeSec * 7) * 113, py + fr(timeSec * 3) * 71) - 0.5) * 0.05;
    const scan = 0.97 + 0.03 * Math.sin(py * 1.7);
    const cx = (px / W - 0.5) * 1.4, cy = (py / H - 0.5) * 1.4;
    const vig = 1 - 0.45 * (cx * cx + cy * cy);
    for (let k = 0; k < 3; k++) {
      // canvas is sRGB-encoded 8-bit, same as writing to a bgra8unorm swapchain
      out[i * 4 + k] = clamp(Math.round(Math.max((col[k] * scan + grain) * vig, 0) * 255), 0, 255);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

// --- geometry sanity checks on the primitive list ---
function checkPrims(name, pr, W, H) {
  const f = pr.f; let bad = 0, inside = 0;
  for (let n = 0; n < pr.n; n++) {
    const o = n * 12;
    for (let k = 0; k < 12; k++) if (!Number.isFinite(f[o + k])) bad++;
    const x = f[o], y = f[o + 1];
    if (x > -0.1 * W && x < 1.1 * W && y > -0.1 * H && y < 1.1 * H) inside++;
  }
  return { name, prims: pr.n, nonFinite: bad, anchorsInFrame: `${inside}/${pr.n}` };
}

const results = [];
function panel(file, build, mask, maskW, maskH, label) {
  const raw = jpeg.decode(fs.readFileSync(path.join(D, file)), { useTArray: true });
  const W = raw.width, H = raw.height;
  const pr = new XR.Prims(4096);
  build(pr, W, H);
  results.push(checkPrims(label, pr, W, H));
  const bones = rasterBones(pr, W, H);
  const px = composite(raw.data, W, H, mask, maskW, maskH, bones, !!mask);
  return { W, H, px, label };
}

// Human: MediaPipe Pose Landmarker lite output on pose.jpg (normalized coords -> px)
const poseJ = JSON.parse(fs.readFileSync(path.join(D, 'pose_landmarks.json'), 'utf8')).pose;
// segmentation mask stored as text: run-length pairs "<level hex><count hex>," at 16 levels
const maskJ = JSON.parse(fs.readFileSync(path.join(D, 'pose_mask.json'), 'utf8'));
const mask = new Uint8Array(maskJ.w * maskJ.h);
{ let o = 0; for (const r of maskJ.rle.split(',')) { if (!r) continue; const v = parseInt(r[0], 16) * 17, c = parseInt(r.slice(1), 16); mask.fill(v, o, o + c); o += c; } }
const human = panel('pose.jpg', (pr, W, H) => {
  const P = poseJ.lm.map(l => [l[0] * W, l[1] * H]), V = poseJ.lm.map(l => l[3]);
  XR.buildHuman(pr, P, V);
  // proportion check: humerus primitive length vs shoulder-elbow landmark distance
  const d = (i, j) => Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]);
  results.push({ name: 'human landmark distances (px)', shoulderW: d(11, 12).toFixed(1), upperArmL: d(11, 13).toFixed(1), thighL: d(23, 25).toFixed(1), shinL: d(25, 27).toFixed(1) });
}, mask, maskJ.w, maskJ.h, 'human (MediaPipe lite landmarks)');

// Quadrupeds: AP-10K ground-truth keypoints (antelope -> dog template, jaguar -> cat template)
const gt = JSON.parse(fs.readFileSync(path.join(D, 'ap10k_gt.json'), 'utf8'));
function quad(file, species) {
  const a = gt.annotations.find(x => x.file === file);
  const K = [], C = [];
  for (let i = 0; i < 17; i++) { K.push([a.keypoints[i * 3], a.keypoints[i * 3 + 1]]); C.push(a.keypoints[i * 3 + 2] > 0 ? 1 : 0); }
  return panel(file, (pr) => XR.buildQuad(pr, K, C, species, 0.8), null, 0, 0, `${species} template (AP-10K GT keypoints, ${file})`);
}
const dog = quad('ap10k.jpg', 'dog');
const cat = quad('ap10k_37516.jpg', 'cat');

// Simulated SimCC decode round-trip on synthetic heatmaps (checks argmax/2 scaling)
{
  const bins = 512, K = 17, sx = new Float32Array(K * bins), sy = new Float32Array(K * bins);
  const want = []; for (let k = 0; k < K; k++) { const x = 10 + k * 13.5, y = 240 - k * 11; want.push([x, y]); sx[k * bins + Math.round(x * 2)] = 0.9; sy[k * bins + Math.round(y * 2)] = 0.8; }
  const { pts, conf } = XR.decodeSimcc(sx, sy, K, bins);
  const err = Math.max(...pts.map((p, i) => Math.hypot(p[0] - want[i][0], p[1] - want[i][1])));
  results.push({ name: 'SimCC decode round-trip', maxErrPx: err, conf0: conf[0] });
}
// One-Euro: jitter reduction on a noisy static point
{
  const f = new XR.OneEuro(1.2, 0.02); let s = 0, sr = 0, prev = null, prevR = null; let seed = 1;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (let i = 0; i < 300; i++) { const raw = 100 + rnd() * 6; const v = f.filter(raw, i / 30); if (prev !== null) { s += Math.abs(v - prev); sr += Math.abs(raw - prevR); } prev = v; prevR = raw; }
  results.push({ name: 'One-Euro on static point with ±3 px noise @30 fps', rawFrameToFrame: (sr / 299).toFixed(2), filtered: (s / 299).toFixed(2) });
}

// Assemble: human on top, dog and cat below (scaled to half width each)
function scaleBox(p, W, H, w, h) { // box filter for downscale
  const o = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const x0 = Math.floor(x * W / w), x1 = Math.max(x0 + 1, Math.floor((x + 1) * W / w)), y0 = Math.floor(y * H / h), y1 = Math.max(y0 + 1, Math.floor((y + 1) * H / h));
    const acc = [0, 0, 0]; let n = 0;
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) { const i = (yy * W + xx) * 4; acc[0] += p[i]; acc[1] += p[i + 1]; acc[2] += p[i + 2]; n++; }
    const j = (y * w + x) * 4; o[j] = acc[0] / n; o[j + 1] = acc[1] / n; o[j + 2] = acc[2] / n; o[j + 3] = 255;
  }
  return o;
}
const OW = 1000, gap = 4;
const halfW = (OW - gap) / 2;
const dH = Math.round(dog.H * halfW / dog.W), cH = Math.round(cat.H * halfW / cat.W), rowH = Math.max(dH, cH);
const OH = human.H + gap + rowH;
const png = new PNG({ width: OW, height: OH });
png.data.fill(0); for (let i = 3; i < png.data.length; i += 4) png.data[i] = 255;
const blit = (p, w, h, ox, oy) => { for (let y = 0; y < h; y++) p.copy(png.data, ((oy + y) * OW + ox) * 4, y * w * 4, (y + 1) * w * 4); };
blit(human.px, human.W, human.H, 0, 0);
blit(scaleBox(dog.px, dog.W, dog.H, halfW, dH), halfW, dH, 0, human.H + gap);
blit(scaleBox(cat.px, cat.W, cat.H, halfW, cH), halfW, cH, halfW + gap, human.H + gap);
const outPath = path.join(D, '..', 'test-frame.png');
fs.writeFileSync(outPath, PNG.sync.write(png));
console.log(JSON.stringify(results, null, 1));
console.log('wrote', outPath, OW + 'x' + OH);
