'use strict';
// Top-down scenes (gate, drone, pothole chart) and the close-up face.
const ORDER = ['karim', 'hulk', 'phil', 'gary', 'angelo'].concat(FARS.map(f => f.id)).concat(['far0']); // 5 + 12 + extras
const FIELD = (() => { // 38 riders: the six named plus generated field
  const arr = []; const named = ['karim', 'hulk', 'phil', 'gary', 'angelo']; named.forEach(n => arr.push(CAST[n]));
  for (let i = 0; i < 32; i++) { const base = FARS[i % FARS.length]; arr.push(Object.assign({}, base, { id: 'f' + i, build: 0.92 + 0.14 * rnd(i, 5), helm: ['#ffffff', '#222222', '#e63946', '#ffd23f', '#2ec4b6'][i % 5], jersey: FAR_J[(i * 5 + 3) % FAR_J.length], frame: FAR_J[(i * 5 + 3) % FAR_J.length] })); }
  arr.splice(30, 0, CAST.me); arr.length = 38; return arr;
})();

function tdRider(c, R, x, y, s, ph, P, o = {}) {
  const flat = o.flat, C = col => flat || P.s(col, o.depth || 0);
  c.save(); c.translate(x, y); c.scale(s, s); if (o.rot) c.rotate(o.rot);
  const wl = (cx) => { c.fillStyle = C('#111114'); rrect(c, cx - 0.34, -0.026, 0.68, 0.052, 0.025); c.fill(); };
  wl(-0.62); wl(0.62);
  c.strokeStyle = C(R.frame || R.jersey); c.lineWidth = R.bike === 'gravel' ? 0.05 : 0.036; c.lineCap = 'round'; c.beginPath(); c.moveTo(-0.62, 0); c.lineTo(0.6, 0); c.stroke();
  c.strokeStyle = C('#1d1d22'); c.lineWidth = 0.032; c.beginPath(); c.moveTo(0.42, -0.22); c.lineTo(0.42, 0.22); c.stroke();
  for (const sg of [-1, 1]) { // legs
    const a = ph + (sg > 0 ? 0 : Math.PI), fx = -0.02 + 0.17 * Math.cos(a), fy = sg * 0.115;
    c.strokeStyle = C('#17171c'); c.lineWidth = 0.13; c.beginPath(); c.moveTo(-0.14, sg * 0.12); c.lineTo(fx + 0.1 * Math.max(0, Math.sin(a)), fy); c.stroke();
    c.fillStyle = C('#f4f4f4'); c.beginPath(); c.ellipse(fx + 0.04, fy, 0.09, 0.04, 0, 0, TAU); c.fill();
  }
  c.fillStyle = C(R.jersey); c.beginPath(); c.ellipse(0.1, 0, 0.36, 0.2, 0, 0, TAU); c.fill();
  if (!flat) { c.strokeStyle = P.s(R.acc, 0); c.lineWidth = 0.035; c.beginPath(); c.moveTo(-0.15, 0); c.lineTo(0.3, 0); c.stroke(); }
  for (const sg of [-1, 1]) { c.strokeStyle = C(R.jersey); c.lineWidth = 0.08; c.beginPath(); c.moveTo(0.25, sg * 0.19); c.lineTo(0.41, sg * 0.21); c.stroke(); c.fillStyle = C(R.skin); c.beginPath(); c.arc(0.43, sg * 0.22, 0.04, 0, TAU); c.fill(); }
  if (R.pack) { c.fillStyle = C('#3b3b44'); rrect(c, -0.14, -0.17, 0.3, 0.34, 0.05); c.fill(); if (!flat) { c.fillStyle = P.s('#e9c46a', 0); c.fillRect(-0.1, -0.1, 0.1, 0.08); } }
  if (R.letter && !flat) { c.fillStyle = P.s(R.acc, 0); c.font = '0.2px "Anton"'; c.save(); c.translate(0.02, 0.07); c.rotate(Math.PI / 2); c.fillText(R.letter, -0.05, 0); c.restore(); }
  c.fillStyle = C(R.helm); c.beginPath(); c.arc(0.46, 0, 0.108, 0, TAU); c.fill();
  if (!flat) { c.fillStyle = 'rgba(8,10,16,0.9)'; c.fillRect(0.53, -0.07, 0.035, 0.14); c.strokeStyle = 'rgba(0,0,0,0.25)'; c.lineWidth = 0.012; c.beginPath(); c.moveTo(0.38, -0.08); c.lineTo(0.52, -0.02); c.moveTo(0.38, 0.08); c.lineTo(0.52, 0.02); c.stroke(); }
  c.restore();
  if (o.lights > 0 && !flat) {
    c.save(); c.globalCompositeOperation = 'lighter'; const L = o.lights;
    const gx = x + 0.65 * s, g = c.createLinearGradient(gx, y, gx + 2.6 * s, y); g.addColorStop(0, `rgba(255,248,225,${0.4 * L})`); g.addColorStop(1, 'rgba(255,248,225,0)');
    c.fillStyle = g; c.beginPath(); c.moveTo(gx, y); c.lineTo(gx + 2.6 * s, y - 0.5 * s); c.lineTo(gx + 2.6 * s, y + 0.5 * s); c.closePath(); c.fill();
    glow(c, gx, y, 0.2 * s, [255, 255, 235], 0.9 * L); glow(c, x - 0.7 * s, y, 0.3 * s, [255, 40, 40], 0.8 * L * (0.5 + 0.5 * Math.sin(t3(x))));
    c.restore();
  }
}
const t3 = x => x * 0.05 + (typeof window.__t === 'number' ? window.__t * 6 : 0);

function tdTree(c, P, x, y, r, seed, sunK) {
  c.fillStyle = `rgba(6,10,6,${0.28 * (P.shadowA > 0.02 ? 1 : 0.4)})`; c.beginPath(); c.ellipse(x - r * 0.25 * sunK, y + r * 0.25 * sunK, r * 1.05, r * 0.95, 0, 0, TAU); c.fill();
  const base = ['#2d6a4f', '#40916c', '#1b4332', '#52796f', '#3f7f3a'][Math.floor(rnd(seed, 3) * 5)];
  c.fillStyle = P.s(base, 0.05); c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill();
  for (let k = 0; k < 6; k++) { const a = rnd(seed * 7 + k, 4) * TAU, d = r * 0.55 * rnd(seed * 3 + k, 5); c.fillStyle = P.s(base, 0.0 - 0.0); c.globalAlpha = 0.5; c.beginPath(); c.arc(x + Math.cos(a) * d, y + Math.sin(a) * d, r * 0.42, 0, TAU); c.fill(); c.globalAlpha = 1; }
  c.fillStyle = `rgba(255,236,170,${0.17 * P.light})`; c.beginPath(); c.arc(x + r * 0.28, y - r * 0.28, r * 0.62, 0, TAU); c.fill();
}

function sceneTopDown(c, t, shot) {
  const P = new Pal(t), tl = t - shot.t0, kind = shot.kind || 'gate', night = P.night, lights = smooth((t - 168) / 28);
  const yC = H * 0.66, hw = 205, xg = W * 0.52, s = 70, v = 600, sunK = Math.max(0.35, P.shadowK * 0.45);
  window.__t = t;
  const ycAt = x => kind === 'drone' ? yC + 70 * Math.sin((x + tl * 260) / 900) + 30 * Math.sin((x + tl * 260) / 330) : yC;
  // ground
  const gg = c.createLinearGradient(0, 0, 0, H); gg.addColorStop(0, P.s('#3f7a3d', 0.1)); gg.addColorStop(1, P.s('#356b34', 0.1)); c.fillStyle = gg; c.fillRect(0, 0, W, H);
  for (let i = 0; i < 90; i++) { c.fillStyle = P.s(rnd(i, 2) > 0.5 ? '#4b8a45' : '#2f6330', 0.1); c.globalAlpha = 0.5; c.beginPath(); c.ellipse(rnd(i, 3) * W, rnd(i, 4) * H, 60 + 70 * rnd(i, 5), 18 + 22 * rnd(i, 6), rnd(i, 7), 0, TAU); c.fill(); c.globalAlpha = 1; }
  const off = kind === 'drone' ? tl * 260 : 0;
  // road
  c.beginPath(); for (let x = -40; x <= W + 40; x += 30) { const y = ycAt(x) - hw; x === -40 ? c.moveTo(x, y) : c.lineTo(x, y); } for (let x = W + 40; x >= -40; x -= 30) c.lineTo(x, ycAt(x) + hw); c.closePath(); c.fillStyle = P.s('#3d4047'); c.fill();
  c.strokeStyle = P.s('#e8e8e0'); c.lineWidth = 4; [-hw + 14, hw - 14].forEach(dy => { c.beginPath(); for (let x = -40; x <= W + 40; x += 30) { const y = ycAt(x) + dy; x === -40 ? c.moveTo(x, y) : c.lineTo(x, y); } c.stroke(); });
  c.strokeStyle = P.s('#f2c230'); c.lineWidth = 6; c.setLineDash([70, 70]); c.lineDashOffset = off; c.beginPath(); for (let x = -40; x <= W + 40; x += 30) { const y = ycAt(x); x === -40 ? c.moveTo(x, y) : c.lineTo(x, y); } c.stroke(); c.setLineDash([]);
  // shoulder path (bottom)
  c.fillStyle = P.s('#8e8a80', 0.1); c.beginPath(); for (let x = -40; x <= W + 40; x += 30) { const y = ycAt(x) + hw + 42; x === -40 ? c.moveTo(x, y) : c.lineTo(x, y); } for (let x = W + 40; x >= -40; x -= 30) c.lineTo(x, ycAt(x) + hw + 120); c.closePath(); c.fill();
  // trees (top + bottom rows)
  const tk0 = Math.floor((off - 300) / 210);
  for (let k = tk0; k < tk0 + 13; k++) { const x = k * 210 - off + rnd(k, 8) * 90; tdTree(c, P, x, ycAt(x) - hw - 150 - 60 * rnd(k, 9), 110 + 60 * rnd(k, 10), k, sunK); }
  for (let k = tk0; k < tk0 + 13; k++) { const x = k * 230 - off * 1.05 + rnd(k, 18) * 90; tdTree(c, P, x, Math.min(H + 20, ycAt(x) + hw + 250 + 40 * rnd(k, 19)), 105 + 55 * rnd(k, 20), k + 50, sunK); }
  // lamp pools at night
  if (night > 0.15) { c.save(); c.globalCompositeOperation = 'lighter'; for (let k = tk0; k < tk0 + 8; k++) { const x = k * 420 - off + 120, y = ycAt(x) - hw - 20; glow(c, x, y + 120, 300, [255, 190, 100], 0.32 * night); glow(c, x, y, 40, [255, 220, 150], 0.8 * night); } c.restore(); }

  if (kind === 'gate') {
    const gap = 150, y0 = yC, T0 = 0.5, dt = 0.22, n = 38;
    // gate arms
    const arm = (yFrom, yTo) => { const n6 = 8; for (let i = 0; i < n6; i++) { c.fillStyle = P.s(i % 2 ? '#f7f7f7' : '#e63946'); const a = lerp(yFrom, yTo, i / n6), b = lerp(yFrom, yTo, (i + 1) / n6); c.fillRect(xg - 9, Math.min(a, b), 18, Math.abs(b - a) + 1); } c.fillStyle = P.s('#dcdcdc'); c.beginPath(); c.arc(xg, yFrom, 20, 0, TAU); c.fill(); };
    c.fillStyle = 'rgba(6,4,18,0.25)'; c.fillRect(xg - 9 - 26 * sunK, yC - hw - 20 + 14, 18, hw - gap / 2 + 20); c.fillRect(xg - 9 - 26 * sunK, yC + gap / 2 + 14, 18, hw - gap / 2 + 20);
    arm(yC - hw - 20, yC - gap / 2); arm(yC + hw + 20, yC + gap / 2);
    [[xg - 70, yC - hw - 70], [xg - 70, yC + hw + 70]].forEach(([x, y]) => { c.fillStyle = P.s('#f7f7f7'); rrect(c, x - 50, y - 18, 100, 36, 4); c.fill(); c.fillStyle = P.s('#b3202d'); c.fillRect(x - 50, y - 3, 100, 6); });
    // walkers on the shoulder path
    const walkers = [];
    if (shot.walk === 'stroller') { const wx = lerp(xg + 520, xg - 120, tl / 9.5); walkers.push([wx, yC + hw + 82, '#c77dff', true], [wx + 62, yC + hw + 90, '#ffd166', false]); }
    if (shot.walk === 'scatter') { for (let i = 0; i < 6; i++) { const hop = smooth((tl - 3.4 - i * 0.18) / 0.5); walkers.push([xg - 340 + i * 150, yC + hw + 86 + 170 * hop - 90 * Math.sin(hop * Math.PI) * 0, ['#ff8fab', '#4cc9f0', '#ffd166', '#c77dff', '#06d6a0', '#ef476f'][i], false]); } }
    walkers.forEach(([x, y, col, st]) => {
      c.fillStyle = 'rgba(6,4,18,0.28)'; c.beginPath(); c.ellipse(x - 14 * sunK, y + 10 * sunK, 30, 22, 0, 0, TAU); c.fill();
      if (st) { c.fillStyle = P.s('#5aa9e6'); rrect(c, x + 26, y - 20, 62, 40, 12); c.fill(); c.fillStyle = '#222'; [x + 34, x + 80].forEach(wx => { c.fillRect(wx - 6, y - 26, 12, 6); c.fillRect(wx - 6, y + 20, 12, 6); }); c.strokeStyle = P.s('#333'); c.lineWidth = 4; c.beginPath(); c.moveTo(x + 24, y - 10); c.lineTo(x + 10, y); c.lineTo(x + 24, y + 10); c.stroke(); }
      c.fillStyle = P.s(col); c.beginPath(); c.ellipse(x, y, 16, 26, 0, 0, TAU); c.fill(); c.fillStyle = P.s('#d9a07a'); c.beginPath(); c.arc(x + 4, y, 12, 0, TAU); c.fill(); c.fillStyle = P.s('#3a2a1a'); c.beginPath(); c.arc(x + 1, y, 11, Math.PI * 0.4, Math.PI * 1.6); c.fill();
    });
    // riders: rider i reaches the gap at T0 + i*dt
    const funnel = x => x < xg - 60 ? smooth((x - (xg - 700)) / 640) : 1 - smooth((x - (xg + 60)) / 620);
    const list = [];
    for (let i = 0; i < n; i++) {
      const x = xg + v * (t > 0 ? tl - (T0 + i * dt) : 0), lane = ((i % 4) - 1.5) * 100, f = funnel(x);
      const y = y0 + lane * (1 - f) + 6 * Math.sin(i * 1.7 + tl * 2) * (1 - f);
      list.push([x, y, i]);
    }
    list.forEach(([x, y, i]) => { if (x > -160 && x < W + 160 && P.shadowA > 0.02) tdRider(c, FIELD[i], x - 62 * sunK, y + 38 * sunK, s * 1.04, tl * 5 + i, P, { flat: `rgba(6,4,18,${P.shadowA})` }); });
    list.forEach(([x, y, i]) => { if (x > -160 && x < W + 160) tdRider(c, FIELD[i], x, y, s, tl * 5.2 + i * 1.3, P, { lights }); });
    // shout rings
    const rs = tl; if (rs < 1.6) { const lead = list[0]; for (let k = 0; k < 3; k++) { const a = clamp(rs / 1.4 - k * 0.15); if (a <= 0 || a >= 1) continue; c.strokeStyle = `rgba(255,255,255,${0.65 * (1 - a)})`; c.lineWidth = 7 * (1 - a) + 1; c.beginPath(); c.arc(lead[0], lead[1], 40 + 380 * a, 0, TAU); c.stroke(); } }
    // sorry bubbles
    if (shot.walk === 'stroller') for (let k = 0; k < 4; k++) { const ts = 3.6 + k * 0.38; if (tl > ts && tl < ts + 1.4) { const r = list[Math.min(37, 10 + k * 5)]; bubble(c, r[0] + 20, r[1] - 120 - 20 * k % 3, 'SORRY!', tl - ts, '#ffffff', '#1b1b1b', 34); } }
    if (shot.walk === 'scatter') for (let k = 0; k < 3; k++) { const ts = 3.9 + k * 0.5; if (tl > ts && tl < ts + 1.3) bubble(c, xg - 280 + k * 260, yC + hw + 20, ['GET OUT THE WAY!', 'COMING THROUGH', 'HEADS UP'][k], tl - ts, '#ffd23f', '#1b1b1b', 30); }
  } else { // drone: the whole pack, two abreast
    const rows = 19, sp = 104, head = lerp(W * 0.99, W * 0.97, smooth(tl / 6.5));
    const list = [];
    for (let i = 0; i < 38; i++) { const r = Math.floor(i / 2), x = head - r * sp + (i % 2) * 28 + 4 * Math.sin(tl * 1.6 + i), y = ycAt(x) + (i % 2 ? 62 : -62) + 7 * Math.sin(tl * 2.1 + i * 0.7); list.push([x, y, i]); }
    list.forEach(([x, y, i]) => { if (P.shadowA > 0.02) tdRider(c, FIELD[i], x - 70 * sunK, y + 44 * sunK, s * 1.04, tl * 5.4 + i, P, { flat: `rgba(6,4,18,${P.shadowA})` }); });
    list.forEach(([x, y, i]) => tdRider(c, FIELD[i], x, y, s, tl * 5.6 + i * 1.3, P, { lights }));
    if (shot.tags) shot.tags.forEach(([id, label, t0, t1]) => { const r = list[ORDER_IDX(id)]; if (r && t >= t0 && t < t1) tag(c, r[0], r[1] - 60, label, '', (t - t0) / 0.3, (t1 - t) / 0.2); });
  }
  // light wash
  const g = c.createLinearGradient(0, 0, 0, H); g.addColorStop(0, `rgba(255,200,130,${0.13 * P.sunE})`); g.addColorStop(1, 'rgba(255,120,60,0)'); c.fillStyle = g; c.fillRect(0, 0, W, H);
  const dark = c.createLinearGradient(0, 0, 0, H * 0.55); dark.addColorStop(0, 'rgba(4,6,16,0.5)'); dark.addColorStop(1, 'rgba(4,6,16,0)'); c.fillStyle = dark; c.fillRect(0, 0, W, H * 0.55);
}
const ORDER_IDX = id => ({ karim: 0, hulk: 1, phil: 2, gary: 3, angelo: 4, me: 30 })[id] || 0;

// ---------------------------------------------------------------- the pothole chart (top-down, annotated)
function scenePothole(c, t, shot) {
  const P = new Pal(t), tl = t - shot.t0, d = shot.t1 - shot.t0, off = tl * 300, yC = H * 0.64, hw = 215;
  c.fillStyle = '#34503a'; c.fillRect(0, 0, W, H);
  c.fillStyle = '#5a5e68'; c.fillRect(0, yC - hw, W, hw * 2);
  for (let i = 0; i < 420; i++) { const x = ((rnd(i, 1) * (W + 400) - off) % (W + 400) + W + 400) % (W + 400) - 200; c.fillStyle = `rgba(${rnd(i, 4) > 0.5 ? '30,32,38' : '90,94,104'},0.45)`; c.fillRect(x, yC - hw + rnd(i, 2) * hw * 2, 3 + 14 * rnd(i, 3), 2 + 3 * rnd(i, 5)); }
  const holes = []; for (let k = 0; k < 14; k++) holes.push({ x: 380 + k * 330 + rnd(k, 11) * 120, y: yC - 160 + rnd(k, 12) * 320, r: 24 + 28 * rnd(k, 13), cm: 3 + Math.floor(rnd(k, 14) * 9) });
  // cracks
  c.strokeStyle = 'rgba(15,15,20,0.7)'; c.lineWidth = 3; for (let k = 0; k < 8; k++) { let x = 200 + k * 520 + rnd(k, 31) * 150 - off, y = yC - hw + 20; c.beginPath(); c.moveTo(x, y); for (let j = 0; j < 9; j++) { x += (rnd(k * 9 + j, 32) - 0.4) * 60; y += 50; c.lineTo(x, y); } c.stroke(); }
  holes.forEach(h => { const x = h.x - off; if (x < -100 || x > W + 100) return; c.fillStyle = 'rgba(8,8,12,0.85)'; c.beginPath(); c.ellipse(x, h.y, h.r * 1.2, h.r, 0.2, 0, TAU); c.fill(); c.strokeStyle = 'rgba(160,165,175,0.55)'; c.lineWidth = 4; c.stroke(); });
  // the straight "line" vs the pothole chart
  const pathY = x => { let y = yC; holes.forEach(h => { const hx = h.x - off, dy = h.y - yC, need = h.r + 62 - Math.abs(dy); if (need > 0) y += -Math.sign(dy || 1) * need * Math.exp(-Math.pow((x - hx) / 130, 2)); }); return y; };
  const reveal = clamp(tl / (d * 0.45));
  c.strokeStyle = 'rgba(255,255,255,0.55)'; c.lineWidth = 5; c.setLineDash([22, 18]); c.beginPath(); c.moveTo(0, yC); c.lineTo(W, yC); c.stroke(); c.setLineDash([]);
  c.font = '34px "Permanent Marker"'; c.fillStyle = 'rgba(255,255,255,0.85)'; c.fillText('the line', 70, yC - 26);
  c.strokeStyle = '#ffd23f'; c.lineWidth = 9; c.lineCap = 'round'; c.lineJoin = 'round'; c.beginPath(); for (let x = 0; x <= W * reveal; x += 14) { const y = pathY(x); x === 0 ? c.moveTo(x, y) : c.lineTo(x, y); } c.stroke();
  c.fillStyle = '#ffd23f'; c.fillText('the pothole chart', W * 0.5, yC + hw + 66 - 2);
  // pack follows the weaving line
  const trail = ['karim', 'phil', 'hulk', 'gary', 'angelo', 'me'];
  trail.forEach((id, k) => { const x = W * 0.8 - k * 190, y = pathY(x), y2 = pathY(x + 20); tdRider(c, CAST[id], x, y, 104, tl * 6 + k, P, { rot: Math.atan2(y2 - y, 20) }); });
  // chart furniture
  c.fillStyle = 'rgba(8,10,16,0.8)'; rrect(c, 40, H - 128, 520, 90, 14); c.fill(); c.font = '30px "Bebas Neue"'; c.letterSpacing = '1px'; c.fillStyle = 'rgba(255,255,255,0.7)';
  c.fillText('RIDGE RD', 66, H - 76); c.fillText('ROSS DR', 438, H - 76); c.strokeStyle = 'rgba(255,255,255,0.35)'; c.lineWidth = 3; c.beginPath(); c.moveTo(66, H - 62); c.lineTo(530, H - 62); c.stroke();
  c.fillStyle = '#ffd23f'; c.beginPath(); c.arc(66 + 464 * reveal, H - 62, 8, 0, TAU); c.fill();
  holes.forEach(h => { const x = h.x - off; if (x < 80 || x > W - 80) return; c.font = '28px "Permanent Marker"'; c.fillStyle = 'rgba(255,255,255,0.8)'; c.fillText(`-${h.cm} cm`, x - 40, h.y - h.r - 16); });
  const dk = c.createLinearGradient(0, 0, 0, H * 0.5); dk.addColorStop(0, 'rgba(4,6,16,0.55)'); dk.addColorStop(1, 'rgba(4,6,16,0)'); c.fillStyle = dk; c.fillRect(0, 0, W, H * 0.5);
}

// ---------------------------------------------------------------- close-up face (eyes going wide)
function sceneFace(c, t, shot) {
  const P = new Pal(t), tl = t - shot.t0, d = shot.t1 - shot.t0, k = smooth(tl / d);
  const g = c.createLinearGradient(0, 0, 0, H); g.addColorStop(0, P.s('#2a2350')); g.addColorStop(1, P.s('#6c3d6f')); c.fillStyle = g; c.fillRect(0, 0, W, H);
  // streaking road + speed lines
  for (let i = 0; i < 46; i++) { const y = H * (0.05 + 0.9 * rnd(i, 1)), L = 400 + 900 * rnd(i, 2), x = W - ((t * (2600 + 1600 * rnd(i, 3)) + rnd(i, 4) * W * 2) % (W * 2.2)) + 200; const gr = c.createLinearGradient(x, y, x + L, y); gr.addColorStop(0, 'rgba(255,255,255,0)'); gr.addColorStop(1, `rgba(255,255,255,${0.22 + 0.25 * rnd(i, 5)})`); c.fillStyle = gr; c.fillRect(x, y, L, 3 + 4 * rnd(i, 6)); }
  // the pack pulling away, blurred, in the distance
  c.save(); c.filter = 'blur(10px)'; for (let i = 0; i < 6; i++) { const r = CAST[NAMED[i]]; c.save(); c.translate(W * 0.8 + 140 * k * (1 + i * 0.25) + i * 80, H * 0.9 - (i % 2) * 40); c.scale(130, 130); drawRider(c, r, { P, crank: t * 9 + i, wang: t * 9, wspin: 14, depth: 0.2, t }); c.restore(); } c.restore();
  // big face
  const z = lerp(1500, 2350, k), shakeX = 6 * Math.sin(t * 61), shakeY = 5 * Math.cos(t * 53);
  c.save(); c.translate(W * 0.38 + shakeX, H * 0.64 + shakeY); c.scale(z, z); c.rotate(-0.05);
  const R = Object.assign({}, CAST.me, { stache: true });
  // neck + collar + shoulder
  c.fillStyle = P.s(R.skin); c.fillRect(-0.12, 0.05, 0.1, 0.3); c.fillStyle = P.s(R.jersey); c.beginPath(); c.moveTo(-0.45, 0.5); c.lineTo(-0.22, 0.2); c.lineTo(0.06, 0.24); c.lineTo(0.12, 0.5); c.closePath(); c.fill();
  drawHead(c, R, 0, 0, { eyes: 'wide', mood: 'o', stress: 1, t, look: 0.002 * Math.sin(t * 5), P }, col => P.s(col));
  c.restore();
  // eye-bulge ring + sweat flicks
  c.strokeStyle = `rgba(255,255,255,${0.5 * k})`; c.lineWidth = 6; c.beginPath(); c.arc(W * 0.38 + z * 0.058, H * 0.64 - z * 0.026, 60 + 40 * k + 6 * Math.sin(t * 40), 0, TAU); c.stroke();
  c.font = '90px "Anton"'; c.fillStyle = '#ffd23f'; outlined(c, '!!', W * 0.38 + z * 0.22, H * 0.64 - z * 0.2 + 6 * Math.sin(t * 30), '#ffd23f', 'rgba(8,8,24,0.9)', 14);
}
