'use strict';
// ---------------------------------------------------------------- basics
const W = 1920, H = 1080, FPS = 30, TAU = Math.PI * 2;
const D = window.DATA;
const NF = Math.floor(D.dur * FPS);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, k) => a + (b - a) * k;
const smooth = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const smoother = x => { x = clamp(x); return x * x * x * (x * (x * 6 - 15) + 10); };
const rnd = (i, s = 0) => { const v = Math.sin(i * 12.9898 + s * 78.233) * 43758.5453; return v - Math.floor(v); };
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const rgba = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
const mix = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
const scale3 = (c, k) => [c[0] * k, c[1] * k, c[2] * k];
function popOut(x) { // 0..1 -> overshoot pop
  x = clamp(x); const c1 = 1.9, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}
function keyed(keys, t) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) if (t <= keys[i][0]) {
    const [ta, va] = keys[i - 1], [tb, vb] = keys[i];
    return tb > ta ? va + (vb - va) * smooth((t - ta) / (tb - ta)) : vb;
  }
  return keys[keys.length - 1][1];
}
function linkeys(keys, t) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) if (t <= keys[i][0]) {
    const [ta, va] = keys[i - 1], [tb, vb] = keys[i];
    return tb > ta ? va + (vb - va) * (t - ta) / (tb - ta) : vb;
  }
  return keys[keys.length - 1][1];
}
function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function glow(ctx, x, y, r, col, a) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, rgba(col, a)); g.addColorStop(1, rgba(col, 0));
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
}
function ik(ax, ay, bx, by, L1, L2, sign) {
  const dx = bx - ax, dy = by - ay; const d = Math.max(1e-6, Math.min(Math.hypot(dx, dy), (L1 + L2) * 0.999));
  const a = Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
  const base = Math.atan2(dy, dx) + sign * a; return [ax + L1 * Math.cos(base), ay + L1 * Math.sin(base)];
}

// ---------------------------------------------------------------- audio features
const BEATS = D.beats;
function beatPulse(t, tau = 0.13) {
  let lo = 0, hi = BEATS.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (BEATS[m] <= t) lo = m; else hi = m - 1; }
  return BEATS[lo] <= t ? Math.exp(-(t - BEATS[lo]) / tau) : 0;
}
function nearestBeat(t) {
  let best = BEATS[0]; for (const b of BEATS) if (Math.abs(b - t) < Math.abs(best - t)) best = b; return best;
}
const energy = t => D.rms[Math.min(D.rms.length - 1, Math.max(0, Math.round(t * FPS)))];
const onset = t => D.ons[Math.min(D.ons.length - 1, Math.max(0, Math.round(t * FPS)))];

// ---------------------------------------------------------------- route + odometer
const RT = D.route, STEP = RT.step;
function rAt(arr, s) {
  const f = clamp(s / STEP, 0, arr.length - 1.001), i = Math.floor(f); return lerp(arr[i], arr[i + 1], f - i);
}
const altAt = s => rAt(RT.alt, s);
const gradeAt = s => rAt(RT.grade, s);
function spdAt(s) { let a = 0; for (let k = -3; k <= 3; k++) a += rAt(RT.spd, s + k * 120); return a / 7; }
// song time -> metres along the real ride (anchors chosen so each lyric lands on the matching real terrain)
const ANCH0 = [[0, 0], [19, 0], [26.5, 400], [46.3, 4500], [55.8, 6500], [81.9, 11800], [91, 14500], [96.5, 15300],
  [101.6, 17000], [109.4, 25500], [111.1, 25900], [124.1, 26750], [130.8, 27400], [153.9, 30300], [166.4, 33300], [169.6, 34000], [184.2, 35100], [198.5, 37800],
  [205, 38400], [214, 39500], [232, 40495], [400, 40495]];
const RM = s => Math.max(0, s - RT.shift);               // original-activity metres -> metres from the ride start
const ANCH = ANCH0.map(([t, s]) => [t, RM(s)]); ANCH[2][1] = 300; // slow roll-out before the first push
const ODO = new Float64Array(NF + 90), RATE = new Float64Array(NF + 90);
(function () {
  for (let f = 0; f < ODO.length; f++) {
    const t = f / FPS; let i = 1; while (i < ANCH.length - 1 && ANCH[i][0] < t) i++;
    const [ta, sa] = ANCH[i - 1], [tb, sb] = ANCH[i]; RATE[f] = (sb - sa) / (tb - ta);
  }
  const R2 = new Float64Array(RATE.length), k = 36; // 1.2 s box blur so scroll speed has no kinks
  for (let f = 0; f < RATE.length; f++) { let a = 0, n = 0; for (let j = -k; j <= k; j++) { const g = f + j; if (g >= 0 && g < RATE.length) { a += RATE[g]; n++; } } R2[f] = a / n; }
  let s = 0; for (let f = 0; f < ODO.length; f++) { ODO[f] = s; s += R2[f] / FPS; }
  const sc = RT.len / ODO[Math.round(232 * FPS)]; for (let f = 0; f < ODO.length; f++) ODO[f] *= sc; for (let f = 0; f < RATE.length; f++) RATE[f] = R2[f] * sc;
})();
function odo(t) { const f = clamp(t * FPS, 0, ODO.length - 1.001), i = Math.floor(f); return lerp(ODO[i], ODO[i + 1], f - i); }
function odoRate(t) { return RATE[Math.min(RATE.length - 1, Math.max(0, Math.round(t * FPS)))]; }
// cumulative ride time along the route (for the clock): integrate ds / v
const CUMT = new Float64Array(RT.n); (function () {
  let a = 0; for (let i = 1; i < RT.n; i++) { a += STEP / Math.max(3, RT.spd[i]); CUMT[i] = a; }
  const sc = (77 * 60) / CUMT[RT.n - 1]; for (let i = 0; i < RT.n; i++) CUMT[i] *= sc;
})();
function rideSeconds(t) { const s = odo(t); const f = clamp(s / STEP, 0, RT.n - 1.001), i = Math.floor(f); return lerp(CUMT[i], CUMT[i + 1], f - i); }
function clockStr(t) {
  const sec = t < 9.5 ? -45 + t * 4.7 : rideSeconds(t);
  const tot = 6 * 3600 + 15 * 60 + Math.floor(sec); const h = Math.floor(tot / 3600) % 12 || 12, m = Math.floor(tot / 60) % 60;
  return `${h}:${String(m).padStart(2, '0')}`;
}
const milesAt = t => odo(t) / 1609.344;
const climbFt = t => rAt(RT.gain, odo(t)) * 3.28084;
const mphAt = t => spdAt(odo(t)) * 2.23694;

// ---------------------------------------------------------------- time of day
const SKYK = [ // t, top, horizon, light(0..1), sun elevation
  [0, '#5d86c4', '#ffcf94', 1.0], [45, '#5578b8', '#ffb36f', 0.95], [90, '#4a5aa6', '#f59a6d', 0.82], [125, '#47408f', '#ee7f78', 0.68],
  [155, '#2f2a73', '#c4608a', 0.6], [185, '#1a1d55', '#6c4a86', 0.54], [205, '#0b1130', '#2f2a58', 0.5], [250, '#04061a', '#161430', 0.46]];
SKYK.forEach(k => { k[1] = hex(k[1]); k[2] = hex(k[2]); });
function todAt(t) {
  let i = 1; while (i < SKYK.length - 1 && SKYK[i][0] < t) i++;
  const a = SKYK[i - 1], b = SKYK[i], k = smooth((t - a[0]) / (b[0] - a[0]));
  return { top: mix(a[1], b[1], k), hor: mix(a[2], b[2], k), light: lerp(a[3], b[3], k) };
}
class Pal {
  constructor(t) {
    const d = todAt(t); this.top = d.top; this.hor = d.hor; this.light = d.light;
    this.night = smooth((t - 150) / 55); this.t = t;
    this.sunE = clamp(1 - t / 148, 0, 1);               // 1 = low golden sun, 0 = set
    this.shadowK = lerp(0.55, 3.2, 1 - this.sunE) * (1 - smooth((t - 150) / 25)); // shadow lean (cot of sun elevation)
    this.shadowA = 0.32 * (1 - smooth((t - 140) / 35));
    this.sunDir = 1;                                    // sun is ahead (right)
  }
  c(col, depth = 0) {
    if (typeof col === 'string') col = hex(col);
    let b = scale3(col, this.light);
    b = mix(b, scale3(this.top, 0.35), this.night * 0.22);
    b = [b[0] * (1 - 0.1 * this.night), b[1] * (1 - 0.02 * this.night), Math.min(255, b[2] * (1 + 0.2 * this.night) + 12 * this.night)];
    return mix(b, mix(this.hor, this.top, 0.35), depth * (0.55 - 0.25 * this.night));
  }
  s(col, depth = 0, a = 1) { return rgba(this.c(col, depth), a); }
}

// ---------------------------------------------------------------- shared text helpers
function setFont(ctx, name, px) { ctx.font = `${px}px "${name}"`; }
function outlined(ctx, text, x, y, fill, stroke = 'rgba(8,8,24,0.9)', lw = 10) {
  ctx.lineJoin = 'round'; ctx.lineWidth = lw; ctx.strokeStyle = stroke; ctx.strokeText(text, x, y); ctx.fillStyle = fill; ctx.fillText(text, x, y);
}
function textW(ctx, text) { return ctx.measureText(text).width; }
