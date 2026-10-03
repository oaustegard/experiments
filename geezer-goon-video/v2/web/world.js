'use strict';
// Side-view world: terrain follows the real elevation profile of the ride.
const BETA = 6;            // screen px per route metre
const VS = 13;             // vertical px per metre of elevation (exaggerated 2.2x)
const CX = W * 0.5, Y0 = H * 0.84;
const LANE0 = 72, LANE1 = -52, ROAD_FAR = -128, ROAD_NEAR = 150;
const SC0 = 186, SC1 = 138;

// ---------------------------------------------------------------- pack formation (screen x as a fraction of W)
const XK = {
  me:     [[0, -0.9], [5, -0.9], [9.5, 0.02], [13, 0.10], [164, 0.10], [168, 0.07], [180, 0.05], [184, 0.1], [188, 0.22], [190, 0.30], [195, 0.35], [198, 0.10], [236, 0.10], [247, 1.35]],
  angelo: [[0, -0.9], [4, -0.9], [8, 0.0], [12, 0.25], [121.5, 0.25], [126.5, 0.97], [135, 0.97], [140, 0.25], [184, 0.25], [188, 0.36], [195, 0.33], [198, 0.25], [236, 0.25], [247, 1.3]],
  gary:   [[0, -0.9], [3, -0.9], [7, -0.02], [11, 0.40], [39.6, 0.40], [42.0, 0.475], [58, 0.475], [63, 0.40], [184, 0.40], [188, 0.52], [195, 0.50], [198, 0.40], [236, 0.40], [247, 1.3]],
  phil:   [[0, -0.9], [2, -0.9], [6.5, 0.1], [10.5, 0.55], [39.6, 0.55], [41.6, 0.95], [74, 0.95], [78, 0.55], [184, 0.55], [187.5, 0.70], [195, 0.64], [198, 0.55], [205, 0.55], [207, 0.92], [214, 0.9], [218, 0.55], [236, 0.55], [246, 1.3]],
  hulk:   [[0, -0.9], [1, -0.9], [8, 0.2], [13, 0.70], [39.6, 0.70], [41.6, 0.64], [74, 0.64], [78, 0.70], [184, 0.70], [188, 0.83], [195, 0.79], [198, 0.70], [236, 0.70], [246, 1.35]],
  karim:  [[0, -0.9], [8, -0.9], [14.2, 0.85], [39.6, 0.85], [41.6, 0.79], [74, 0.79], [78, 0.85], [186.3, 0.85], [188, 1.00], [195, 0.96], [198, 0.85], [205, 0.85], [207, 1.04], [214, 1.0], [218, 0.85], [236, 0.85], [245, 1.35]],
};
FARS.forEach((r, i) => { const b = 0.03 + 0.098 * i; XK[r.id] = [[0, -0.9 - 0.04 * i], [2 + i * 0.7, -0.9 - 0.04 * i], [7 + i * 0.6, b], [237, b], [247.5, b + 1.15]]; });
const ALL = NAMED.map(n => CAST[n]).concat(FARS);
const slotX = (id, t) => keyed(XK[id], t) * W;
// intervals where a rider is coasting (Gary sits on the wheel and does not move)
const COAST = { gary: [[42.7, 46.3]] };

// ---------------------------------------------------------------- terrain
function altCam(t) { const s = odo(t); let a = 0; for (let k = -4; k <= 4; k++) a += altAt(s + k * 150); return a / 9; }
function terrain(t) {
  const s0 = odo(t), aC = altCam(t);
  return {
    s0, aC, lift: (aC - 60) * VS,
    yc: x => Y0 - (altAt(s0 + (x - CX) / BETA) - aC) * VS,
    slope: x => -gradeAt(s0 + (x - CX) / BETA) * VS / BETA,
  };
}

// ---------------------------------------------------------------- precomputed rider physics (crank / wheel angles)
const PHYS = {};
(function () {
  for (const R of ALL) {
    const lane1 = R.id.startsWith('far'), sc = lane1 ? SC1 : SC0, off = rnd(R.id.length * 7 + R.id.charCodeAt(R.id.length - 1), 9) * TAU;
    const crank = new Float32Array(NF + 2), wang = new Float32Array(NF + 2), wspin = new Float32Array(NF + 2);
    let c = off, w = 0, prev = slotX(R.id, 0);
    for (let f = 0; f <= NF + 1; f++) {
      const t = f / FPS, x = slotX(R.id, t), vx = (x - prev) * FPS; prev = x;
      const v = Math.max(0, odoRate(t) * BETA + vx);                  // ground speed in px/s
      const spin = v / (0.34 * sc);
      let coast = false; (COAST[R.id] || []).forEach(([a, b]) => { if (t >= a && t < b) coast = true; });
      const cad = (coast || v < 70) ? 0 : clamp(0.85 + 0.55 * Math.min(1.4, v / 1500), 0.85, 1.6) * (0.95 + 0.1 * rnd(f > 0 ? R.id.length : 1, 2));
      c += cad * TAU / FPS; w += Math.min(spin, 12) / FPS; crank[f] = c; wang[f] = w; wspin[f] = spin;
    }
    PHYS[R.id] = { crank, wang, wspin };
  }
})();
const physAt = (id, t, key) => { const a = PHYS[id][key], f = clamp(t * FPS, 0, NF), i = Math.floor(f); return lerp(a[i], a[i + 1], f - i); };

// ---------------------------------------------------------------- layout: where every rider is this frame
function layout(t, opt = {}) {
  const T = terrain(t), out = [];
  const gather = opt.gather;
  for (const R of ALL) {
    const lane1 = R.id.startsWith('far');
    let x = slotX(R.id, t);
    const sy = opt.squeeze || 0;                                                     // funnel to single file
    const y0 = T.yc(x) + (lane1 ? LANE1 : LANE0);
    const slope = Math.atan(T.slope(x));
    const grade = gradeAt(T.s0 + (x - CX) / BETA);
    const tend = R.id === 'angelo' ? 1 : (R.big ? 0.55 : (R.id === 'gary' ? 0.25 : 0.7));
    const stand = clamp((grade - 0.03) / 0.022) * tend;
    out.push({ R, x, y: y0, s: lane1 ? SC1 : SC0, slope, lane1, stand, depth: lane1 ? 0.2 : 0 });
  }
  out.sort((a, b) => a.y - b.y);
  return { T, riders: out, byId: Object.fromEntries(out.map(r => [r.R.id, r])) };
}

// ---------------------------------------------------------------- backdrop layers
function drawSky(ctx, P, t, lift) {
  const hy = H * 0.64 + lift * 0.9;
  const g = ctx.createLinearGradient(0, -H, 0, hy); g.addColorStop(0, rgba(P.top)); g.addColorStop(1, rgba(P.hor));
  ctx.fillStyle = g; ctx.fillRect(-W, -H * 2, W * 3, hy + H * 2);
  ctx.fillStyle = rgba(mix(P.hor, P.top, 0.1)); ctx.fillRect(-W, hy - 1, W * 3, H * 3);
  const p = t / D.dur, sa = smooth((p - 0.55) / 0.2);
  if (sa > 0) for (let i = 0; i < 170; i++) {
    const x = rnd(i, 1) * W, y = rnd(i, 2) * hy * 0.9, tw = 0.55 + 0.45 * Math.sin(t * (1 + rnd(i, 3) * 3) + i);
    ctx.fillStyle = `rgba(255,255,240,${sa * tw * (0.25 + 0.75 * rnd(i, 4))})`; ctx.beginPath(); ctx.arc(x, y, 0.8 + 1.5 * rnd(i, 5), 0, TAU); ctx.fill();
  }
  // sun
  const sunK = smooth(t / 152), sy = lerp(H * 0.30, hy + 70, sunK), sx = W * 0.8;
  if (t < 170) {
    const a = 1 - smooth((t - 130) / 30);
    ctx.save(); ctx.globalCompositeOperation = 'lighter'; glow(ctx, sx, sy, 520, [255, 176, 100], 0.26 * a); glow(ctx, sx, sy, 190, [255, 214, 150], 0.26 * a); ctx.restore();
    ctx.fillStyle = rgba([255, lerp(240, 150, sunK), lerp(185, 90, sunK)], a); ctx.beginPath(); ctx.arc(sx, sy, 84, 0, TAU); ctx.fill();
  }
  // moon
  if (p > 0.5) {
    const k = smooth((p - 0.5) / 0.5), mx = W * 0.15 + W * 0.05 * k, my = lerp(hy + 100, H * 0.17, k);
    ctx.save(); ctx.globalCompositeOperation = 'lighter'; glow(ctx, mx, my, 250, [190, 205, 255], 0.3 * k); ctx.restore();
    ctx.fillStyle = '#f4f1e2'; ctx.beginPath(); ctx.arc(mx, my, 62, 0, TAU); ctx.fill();
    ctx.fillStyle = 'rgba(160,158,150,0.45)'; [[-18, -12, 12], [15, 10, 9], [5, -25, 6], [-10, 20, 7], [24, -14, 5]].forEach(([dx, dy, r]) => { ctx.beginPath(); ctx.arc(mx + dx, my + dy, r, 0, TAU); ctx.fill(); });
  }
  // clouds
  const scroll = odo(t) * BETA;
  for (let i = 0; i < 7; i++) {
    const cx = ((rnd(i, 7) * (W + 900) - 0.02 * scroll - t * 7) % (W + 900) + W + 900) % (W + 900) - 450, cy = 0.07 * H + rnd(i, 8) * 0.3 * H;
    const col = mix(P.hor, [255, 255, 255], 0.3 * (1 - P.night)); ctx.fillStyle = rgba(col, 0.5 * (1 - 0.55 * P.night));
    for (let j = 0; j < 5; j++) { ctx.save(); ctx.translate(cx + j * 58 - 116, cy + Math.sin(j * 1.7) * 12); ctx.scale(1.8, 0.7); ctx.beginPath(); ctx.arc(0, 0, 40 + 18 * rnd(i * 5 + j, 9), 0, TAU); ctx.fill(); ctx.restore(); }
  }
  return hy;
}
function drawHills(ctx, P, t, lift) {
  const sc = odo(t) * BETA;
  [[0.04, 0.60, 48, '#4d6f93', 0.85], [0.10, 0.645, 42, '#40704f', 0.62]].forEach(([f, base, amp, col, depth], L) => {
    const off = f * sc, ly = lift * (1 - f);
    ctx.beginPath(); ctx.moveTo(-700, H * 2);
    for (let sx = -700; sx <= W + 700; sx += 24) {
      const wx = sx + off, y = base * H + ly - amp * (Math.sin(wx / 330 + L) * 0.6 + Math.sin(wx / 131 + 2 * L) * 0.3 + Math.sin(wx / 57) * 0.1);
      ctx.lineTo(sx, y);
    }
    ctx.lineTo(W + 700, H * 2); ctx.closePath(); ctx.fillStyle = P.s(col, depth); ctx.fill();
  });
}
function drawTemple(ctx, P, t, lift) {
  if (t < 109 || t > 133) return;
  const f = 0.1, x = W * 0.62 - (odo(t) - odo(112)) * BETA * f, base = 0.665 * H + lift * (1 - f), a = smooth((t - 109) / 2) * (1 - smooth((t - 131) / 2));
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = P.s('#f3efe6', 0.5); ctx.fillRect(x - 120, base - 120, 240, 130);
  [[-95, 300], [0, 360], [95, 300]].forEach(([dx, h]) => { ctx.fillRect(x + dx - 20, base - h + 50, 40, h - 50); ctx.beginPath(); ctx.moveTo(x + dx - 20, base - h + 50); ctx.lineTo(x + dx, base - h - 40); ctx.lineTo(x + dx + 20, base - h + 50); ctx.fill(); });
  ctx.fillStyle = 'rgba(255,230,160,0.9)'; [[-95, 300], [0, 360], [95, 300]].forEach(([dx, h]) => { ctx.beginPath(); ctx.arc(x + dx, base - h - 46, 6, 0, TAU); ctx.fill(); });
  ctx.restore();
}
function houseRow(ctx, P, t, lift) {
  const f = 0.3, sp = 260, off = f * odo(t) * BETA, k0 = Math.floor((off - 800) / sp), base = 0.705 * H + lift * (1 - f);
  for (let k = k0; k < k0 + Math.ceil((W + 1600) / sp); k++) {
    if (rnd(k, 61) < 0.45) continue;
    const x = k * sp - off + rnd(k, 62) * 60, w = 120 + 90 * rnd(k, 63), h = 70 + 50 * rnd(k, 64);
    ctx.fillStyle = P.s(['#9c8f86', '#8a96a3', '#a58f7a', '#7f8b7a'][Math.floor(rnd(k, 65) * 4)], 0.45); ctx.fillRect(x, base - h, w, h);
    ctx.fillStyle = P.s('#4a3b36', 0.45); ctx.beginPath(); ctx.moveTo(x - 10, base - h); ctx.lineTo(x + w / 2, base - h - 46); ctx.lineTo(x + w + 10, base - h); ctx.closePath(); ctx.fill();
    for (let wi = 0; wi < 3; wi++) { const lit = P.night > 0.2 && rnd(k * 3 + wi, 66) < 0.7; ctx.fillStyle = lit ? `rgba(255,214,130,${0.55 + 0.4 * P.night})` : P.s('#506070', 0.4); ctx.fillRect(x + 14 + wi * (w - 30) / 3, base - h + 20, 18, 22); }
  }
}
function drawTrees(ctx, P, t, lift) {
  const sc = odo(t) * BETA;
  // treeline
  { const f = 0.25, sp = 58, off = f * sc, k0 = Math.floor((off - 800) / sp), base = 0.725 * H + lift * (1 - f);
    ctx.fillStyle = P.s('#2f5a3a', 0.4); ctx.fillRect(-700, base - 6, W + 1400, 0.1 * H);
    for (let k = k0; k < k0 + Math.ceil((W + 1600) / sp); k++) { const x = k * sp - off + rnd(k, 11) * 30, r = 38 + 34 * rnd(k, 12); ctx.beginPath(); ctx.arc(x, base - r * 0.5 + 10 * rnd(k, 13), r, 0, TAU); ctx.fill(); } }
  houseRow(ctx, P, t, lift);
  { const f = 0.55, sp = 150, off = f * sc, k0 = Math.floor((off - 800) / sp), base = 0.775 * H + lift * (1 - f * 0.6);
    for (let k = k0; k < k0 + Math.ceil((W + 1600) / sp); k++) {
      if (rnd(k, 21) < 0.22) continue;
      const x = k * sp - off + rnd(k, 22) * 60, h = 170 + 150 * rnd(k, 23), col = ['#2d6a4f', '#40916c', '#1b4332', '#52796f', '#6a994e'][Math.floor(rnd(k, 24) * 5)];
      tube(ctx, [[x, base], [x, base - h * 0.55]], 14, P.s('#4a3728', 0.25)); ctx.fillStyle = P.s(col, 0.25);
      if (rnd(k, 25) < 0.3) { for (let j = 0; j < 3; j++) { const w = (0.34 - j * 0.08) * h, y0 = base - h * (0.25 + j * 0.22); ctx.beginPath(); ctx.moveTo(x - w, y0); ctx.lineTo(x, y0 - h * 0.38); ctx.lineTo(x + w, y0); ctx.closePath(); ctx.fill(); } }
      else for (let j = 0; j < 4; j++) { const a = j * 1.6 + rnd(k, 26) * 3; ctx.beginPath(); ctx.arc(x + Math.cos(a) * h * 0.13, base - h * 0.72 + Math.sin(a) * h * 0.1, h * (0.2 + 0.05 * rnd(k, 27 + j)), 0, TAU); ctx.fill(); }
    } }
}

// ---------------------------------------------------------------- road
function drawRoad(ctx, P, t, T) {
  const xs = []; for (let x = -720; x <= W + 720; x += 36) xs.push(x);
  const poly = (dy0, dy1, col) => {
    ctx.beginPath(); xs.forEach((x, i) => { const y = T.yc(x) + dy0; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xs[i], T.yc(xs[i]) + dy1); ctx.closePath(); ctx.fillStyle = col; ctx.fill();
  };
  poly(-175, ROAD_FAR + 2, P.s('#5c8a4a', 0.2));              // far verge
  poly(ROAD_FAR, ROAD_NEAR, P.s('#3b3e45'));                   // asphalt
  poly(ROAD_NEAR, 1400, P.s('#2e3d27'));                       // near verge
  poly(ROAD_NEAR - 2, ROAD_NEAR + 10, P.s('#9a9a92'));         // gutter
  poly(ROAD_FAR + 6, ROAD_FAR + 10, P.s('#e8e8e0'));           // edge line
  const sc = odo(t) * BETA, sp = 170, k0 = Math.floor((sc - 800) / sp);
  ctx.fillStyle = P.s('#f2c230');
  for (let k = k0; k < k0 + Math.ceil((W + 1600) / sp); k++) { const x = k * sp - sc + CX - CX; const xx = x, y = T.yc(xx) + 8; ctx.save(); ctx.translate(xx, y); ctx.rotate(Math.atan(T.slope(xx))); ctx.fillRect(0, -3, 84, 7); ctx.restore(); }
  ctx.fillStyle = P.s('#2a2c31', 0, 0.6);
  const sp2 = 97, k2 = Math.floor((sc - 800) / sp2);
  for (let k = k2; k < k2 + Math.ceil((W + 1600) / sp2); k++) { const x = k * sp2 - sc, y = T.yc(x) + ROAD_FAR + 24 + rnd(k, 32) * (ROAD_NEAR - ROAD_FAR - 48); ctx.fillRect(x + rnd(k, 31) * 80, y, 18 + 30 * rnd(k, 33), 3); }
  // potholes for Ridge -> Ross
  if (t > 90 && t < 104) {
    const sp3 = 230, k3 = Math.floor((sc - 800) / sp3);
    for (let k = k3; k < k3 + Math.ceil((W + 1600) / sp3); k++) { if (rnd(k, 41) < 0.35) continue; const x = k * sp3 - sc + rnd(k, 42) * 100, y = T.yc(x) + ROAD_FAR + 30 + rnd(k, 43) * (ROAD_NEAR - ROAD_FAR - 60);
      ctx.fillStyle = P.s('#1c1d20'); ctx.beginPath(); ctx.ellipse(x, y, 26 + 22 * rnd(k, 44), 8 + 6 * rnd(k, 44), 0, 0, TAU); ctx.fill(); }
  }
}

// ---------------------------------------------------------------- props on the far verge
const SIGNS = [[27.0, 'JONES BRIDGE RD', 0.62], [58.5, 'KENSINGTON PKWY', 0.62], [80.2, 'EAST WEST HWY', 0.7], [92.0, 'RIDGE RD', 0.62], [99.0, 'ROSS DR', 0.62],
  [111.3, 'MORMON HILL', 0.45], [134.5, 'KENSINGTON PKWY', 0.62], [154.5, 'BEACH DR', 0.45], [157.5, 'FRANKLIN ST', 0.62], [163.5, 'GROSVENOR LN', 0.62], [167.0, 'KNOWLES AVE', 0.62],
  [184.4, 'WEXFORD DR', 0.5], [186.0, 'GARRETT PARK RD', 0.9], [234.0, 'CEDAR LN & BEACH DR', 0.76], [218, 'KENSINGTON PKWY', 0.62]];
const LIGHTS = [[28.0, 0.72, [[0, 'r'], [29.6, 'g']]], [83.0, 0.62, [[0, 'g'], [82.4, 'y'], [84.6, 'r']]], [196.4, 0.66, [[0, 'y'], [195.3, 'r']]]];
const GRADE_SIGN = [112.0, 0.66];
const lightState = (keys, t) => { let s = keys[0][1]; keys.forEach(([a, b]) => { if (t >= a) s = b; }); return s; };
const propX = (xf, ts, t) => CX + ((odo(ts) - odo(t)) * BETA + (xf - 0.5) * W);

function drawSign(ctx, P, x, base, text) {
  ctx.font = '36px "Bebas Neue"'; const w = textW(ctx, text) + 34;
  tube(ctx, [[x, base], [x, base - 250]], 7, P.s('#8a8f96')); rrect(ctx, x - w / 2, base - 264, w, 50, 6); ctx.fillStyle = P.s('#1f6f43'); ctx.fill();
  ctx.strokeStyle = P.s('#f2f2f2'); ctx.lineWidth = 2.5; ctx.stroke(); ctx.fillStyle = P.s('#f6f6f6'); ctx.fillText(text, x - w / 2 + 17, base - 227);
}
function drawLight(ctx, P, x, base, state, t) {
  tube(ctx, [[x, base], [x, base - 300], [x - 120, base - 300]], 9, P.s('#3a3d42'));
  const hx = x - 140, hy = base - 300; rrect(ctx, hx - 26, hy - 10, 52, 138, 8); ctx.fillStyle = P.s('#222428'); ctx.fill();
  [['r', [255, 38, 30]], ['y', [255, 184, 14]], ['g', [26, 255, 115]]].forEach(([nm, col], i) => {
    const cy = hy + 20 + i * 40, on = state === nm;
    if (on) { ctx.save(); ctx.globalCompositeOperation = 'lighter'; glow(ctx, hx, cy, 95, col, 0.6); ctx.restore(); }
    ctx.fillStyle = rgba(on ? col : scale3(col, 0.22)); ctx.beginPath(); ctx.arc(hx, cy, 14, 0, TAU); ctx.fill();
  });
}
function drawHalfGate(ctx, P, x, base) {
  tube(ctx, [[x, base], [x, base - 112]], 14, P.s('#dcdcdc'));
  for (let i = 0; i < 6; i++) { ctx.fillStyle = P.s(i % 2 ? '#f7f7f7' : '#e63946'); const a0 = i / 6, a1 = (i + 1) / 6; ctx.beginPath(); ctx.moveTo(x + a0 * 78, base - 100 + a0 * 80 - 9); ctx.lineTo(x + a1 * 78, base - 100 + a1 * 80 - 9); ctx.lineTo(x + a1 * 78, base - 100 + a1 * 80 + 9); ctx.lineTo(x + a0 * 78, base - 100 + a0 * 80 + 9); ctx.fill(); }
  rrect(ctx, x - 80, base - 210, 160, 66, 6); ctx.fillStyle = P.s('#f7f7f7'); ctx.fill(); tube(ctx, [[x, base - 144], [x, base - 112]], 6, P.s('#dcdcdc'));
  ctx.fillStyle = P.s('#b3202d'); ctx.font = '25px "Bebas Neue"'; ctx.fillText('ROAD CLOSED', x - 66, base - 183); ctx.fillText('TO MOTOR VEHICLES', x - 66, base - 156);
}
function drawWalker(ctx, P, x, base, t, col, h = 1, stroller = false) {
  const s = 95 * h, ph = t * 5.5; ctx.save(); ctx.translate(x, base); ctx.scale(s, s);
  for (const sg of [1, -1]) { const a = 0.35 * Math.sin(ph) * sg; tube(ctx, [[0, -0.9], [0.45 * Math.sin(a), -0.9 + 0.9 * Math.cos(a)]], 0.12, P.s('#2b2d42')); }
  tube(ctx, [[0, -0.9], [0.02, -1.5]], 0.3, P.s(col)); tube(ctx, [[0, -1.45], [0.18 * Math.sin(ph + 1), -1.0]], 0.09, P.s(col));
  ctx.fillStyle = P.s('#d9a07a'); ctx.beginPath(); ctx.arc(0.03, -1.72, 0.14, 0, TAU); ctx.fill();
  if (stroller) { tube(ctx, [[0.1, -1.1], [0.45, -1.05], [0.55, -0.35]], 0.05, P.s('#333')); ctx.fillStyle = P.s('#5aa9e6'); ctx.beginPath(); ctx.moveTo(0.5, -0.9); ctx.bezierCurveTo(0.55, -1.35, 1.2, -1.3, 1.15, -0.6); ctx.lineTo(0.5, -0.6); ctx.fill(); [0.6, 1.05].forEach(wx => { ctx.fillStyle = P.s('#222'); ctx.beginPath(); ctx.arc(wx, -0.12, 0.12, 0, TAU); ctx.fill(); }); }
  ctx.restore();
}
function drawDog(ctx, P, x, base, t, hand) {
  const ph = t * 9; ctx.save(); ctx.translate(x, base); const s = 55; ctx.scale(s, s); ctx.fillStyle = P.s('#8b5a2b');
  ctx.save(); ctx.scale(1, 0.55); ctx.beginPath(); ctx.arc(0, -1.3, 0.55, 0, TAU); ctx.fill(); ctx.restore(); ctx.beginPath(); ctx.arc(0.62, -0.95, 0.25, 0, TAU); ctx.fill();
  [[-0.35, 0], [-0.2, 1.6], [0.3, 0.8], [0.42, 2.4]].forEach(([lx, o]) => { const a = 0.45 * Math.sin(ph + o); tube(ctx, [[lx, -0.55], [lx + 0.35 * Math.sin(a), 0]], 0.1, P.s('#8b5a2b')); });
  tube(ctx, [[-0.5, -0.8], [-0.8, -1.1 + 0.1 * Math.sin(ph * 2)]], 0.08, P.s('#8b5a2b')); ctx.restore();
  ctx.strokeStyle = P.s('#e63946'); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x + 0.62 * s, base - 0.95 * s); ctx.bezierCurveTo(x + 1.5 * s, base - 0.3 * s, hand[0] - 60, hand[1] + 40, hand[0], hand[1]); ctx.stroke();
}
function drawLamps(ctx, P, t, T) {
  if (P.night < 0.15) return;
  const f = 1.0, sp = 1150, sc = odo(t) * BETA, k0 = Math.floor((sc - 900) / sp);
  for (let k = k0; k < k0 + Math.ceil((W + 1800) / sp); k++) {
    const x = k * sp - sc + 300, y = T.yc(x) + ROAD_FAR + 2;
    tube(ctx, [[x, y], [x, y - 330], [x + 70, y - 345]], 9, P.s('#2c2f35'));
    ctx.save(); ctx.globalCompositeOperation = 'lighter'; const a = P.night;
    glow(ctx, x + 78, y - 340, 130, [255, 190, 100], 0.75 * a);
    const g = ctx.createRadialGradient(x + 78, y + 60, 10, x + 78, y + 60, 360); g.addColorStop(0, `rgba(255,185,100,${0.28 * a})`); g.addColorStop(1, 'rgba(255,185,100,0)');
    ctx.save(); ctx.translate(x + 78, y + 90); ctx.scale(1, 0.32); ctx.translate(-(x + 78), -(y + 60)); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x + 78, y + 60, 360, 0, TAU); ctx.fill(); ctx.restore();
    ctx.restore();
  }
}
function drawProps(ctx, P, t, T) {
  const base = r => T.yc(r) + ROAD_FAR + 6;
  for (const [ts, text, xf] of SIGNS) { const x = propX(xf, ts, t); if (x > -300 && x < W + 300) drawSign(ctx, P, x, base(x), text); }
  for (const [ts, xf, keys] of LIGHTS) { const x = propX(xf, ts, t); if (x > -300 && x < W + 400) drawLight(ctx, P, x, base(x), lightState(keys, t), t); }
  { const x = propX(GRADE_SIGN[1], GRADE_SIGN[0], t); if (x > -200 && x < W + 200) { const b = base(x);
      tube(ctx, [[x, b], [x, b - 200]], 7, P.s('#8a8f96')); ctx.save(); ctx.translate(x, b - 250); ctx.rotate(Math.PI / 4); ctx.fillStyle = P.s('#ffcc00'); ctx.fillRect(-55, -55, 110, 110); ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 5; ctx.strokeRect(-55, -55, 110, 110); ctx.restore();
      ctx.fillStyle = '#1a1a1a'; ctx.font = '54px "Anton"'; ctx.fillText('7%', x - 38, b - 230); } }
  for (const tg of [45.2, 100.8, 227.5]) { const x = propX(0.62, tg, t); if (x > -400 && x < W + 300) drawHalfGate(ctx, P, x, base(x)); }
  if (t > 48 && t < 61) { const x = propX(0.78, 52.3, t) - (t - 52.3) * 40; drawWalker(ctx, P, x, base(x) - 4, t, '#c77dff', 0.95, true); }
  if (t > 84 && t < 97) { const x = propX(0.72, 89.8, t) - (t - 89.8) * 40; drawWalker(ctx, P, x, base(x) - 2, t, '#ff8fab', 0.95); drawWalker(ctx, P, x + 70, base(x) - 1, t + 0.3, '#4cc9f0', 1.02); drawDog(ctx, P, x - 330, base(x), t, [x + 12, base(x) - 100]); }
}

// ---------------------------------------------------------------- the world, drawn in screen coordinates (camera transform is applied by the caller)
function drawBackdrop(ctx, P, t, T) {
  const hy = drawSky(ctx, P, t, T.lift); drawHills(ctx, P, t, T.lift); drawTemple(ctx, P, t, T.lift); drawTrees(ctx, P, t, T.lift); drawRoad(ctx, P, t, T); drawLamps(ctx, P, t, T); drawProps(ctx, P, t, T);
  return hy;
}
function drawRiders(ctx, P, t, L, opt = {}) {
  const lights = smooth((t - 168) / 28) * (1 - smooth((t - 242) / 6));
  const only = opt.lane1 ? 1 : (opt.lane0 ? 0 : -1);
  for (const r of L.riders) { if (only === 1 && !r.lane1) continue; if (only === 0 && r.lane1) continue;
    if (P.shadowA > 0.02) {
      ctx.save(); ctx.translate(r.x, r.y); ctx.scale(r.s, r.s); ctx.rotate(r.slope); ctx.transform(1, 0, P.shadowK, -0.16, 0, 0);
      drawRider(ctx, r.R, { P, flat: `rgba(8,6,20,${P.shadowA * (r.lane1 ? 0.8 : 1)})`, crank: physAt(r.R.id, t, 'crank'), wang: 0, stand: r.stand, t }); ctx.restore();
    }
  }
  for (const r of L.riders) { if (only === 1 && !r.lane1) continue; if (only === 0 && r.lane1) continue;
    ctx.save(); ctx.translate(r.x, r.y); ctx.scale(r.s, r.s); ctx.rotate(r.slope);
    const o = (opt.over && opt.over[r.R.id]) || {};
    drawRider(ctx, r.R, Object.assign({ P, crank: physAt(r.R.id, t, 'crank'), wang: physAt(r.R.id, t, 'wang'), wspin: physAt(r.R.id, t, 'wspin'), stand: r.stand,
      depth: r.depth, lights, t, seed: r.R.id.length + (r.R.id.charCodeAt(r.R.id.length - 1) % 5), rim: Math.max(P.sunE * 0.9, P.night * 0.6) * (r.lane1 ? 0.5 : 1), rimCol: P.night > 0.4 ? [150, 185, 255] : [255, 214, 150], stress: 0 }, o));
    ctx.restore();
  }
}
function drawForeground(ctx, P, t) {
  const f = 1.45, sp = 280, off = f * odo(t) * BETA, k0 = Math.floor((off - 800) / sp);
  for (let k = k0; k < k0 + Math.ceil((W + 1600) / sp); k++) {
    const x = k * sp - off + rnd(k, 51) * 120; ctx.fillStyle = P.s('#1f3a1c');
    for (let j = 0; j < 5; j++) { const a = -Math.PI / 2 + (j - 2) * 0.3; ctx.beginPath(); ctx.moveTo(x + j * 8 - 16, H + 60); ctx.lineTo(x + j * 8 - 16 + 70 * Math.cos(a), H + 20 - 90 * (0.6 + 0.4 * rnd(k * 5 + j, 52))); ctx.lineTo(x + j * 8 - 8, H + 60); ctx.fill(); }
  }
}
