'use strict';
// Route map, bike computer, calendar, plus the overlay helpers (bubbles, tags, stickers, dimension lines).

// ---------------------------------------------------------------- overlay helpers
function bubble(c, x, y, text, age, bg, fg, size = 34) {
  const k = popOut(age / 0.25); if (k <= 0.02) return; c.save(); c.translate(x, y); c.scale(k, k);
  c.font = `${size}px "Permanent Marker"`; const w = textW(c, text) + 34, h = size + 24;
  rrect(c, -w / 2 + 4, -h / 2 + 6, w, h, 16); c.fillStyle = 'rgba(0,0,0,0.3)'; c.fill();
  rrect(c, -w / 2, -h / 2, w, h, 16); c.fillStyle = bg; c.fill(); c.beginPath(); c.moveTo(-14, h / 2 - 2); c.lineTo(-26, h / 2 + 24); c.lineTo(8, h / 2 - 2); c.closePath(); c.fill();
  c.fillStyle = fg; c.fillText(text, -w / 2 + 17, size * 0.36); c.restore();
}
function tag(c, x, y, name, sub, inK, outK = 1) {
  const k = popOut(inK) * clamp(outK); if (k <= 0.02) return; c.save(); c.translate(clamp(x, 130, W - 130), y); c.scale(k, k);
  c.font = '42px "Anton"'; const w1 = textW(c, name); c.font = '26px "Bebas Neue"'; const w2 = sub ? textW(c, sub) : 0, w = Math.max(w1, w2) + 40, h = sub ? 90 : 62;
  rrect(c, -w / 2, -h - 16, w, h, 10); c.fillStyle = 'rgba(8,8,14,0.92)'; c.fill(); c.beginPath(); c.moveTo(-12, -16); c.lineTo(0, 0); c.lineTo(12, -16); c.closePath(); c.fill();
  c.font = '42px "Anton"'; c.fillStyle = '#ffd23f'; c.fillText(name, -w1 / 2, -h - 16 + 46); if (sub) { c.font = '26px "Bebas Neue"'; c.fillStyle = '#fff'; c.fillText(sub, -w2 / 2, -h - 16 + 78); } c.restore();
}
function sticker(c, t, s) {
  const [t0, t1, text, xf, yf, rot, bg, fg, size] = s; if (!text || t < t0 || t > t1 + 0.25) return;
  const k = popOut((t - t0) / 0.3) * (1 - smooth((t - t1) / 0.25)); if (k <= 0.02) return;
  c.save(); c.translate(xf * W, yf * H); c.rotate(rot + 0.02 * Math.sin(t * 3)); c.scale(k, k);
  c.font = `${size}px "${size >= 50 ? 'Anton' : 'Permanent Marker'}"`; const w = textW(c, text) + 46, h = size * 1.5;
  rrect(c, -w / 2 + 6, -h / 2 + 8, w, h, 12); c.fillStyle = 'rgba(0,0,0,0.35)'; c.fill(); rrect(c, -w / 2, -h / 2, w, h, 12); c.fillStyle = bg; c.fill();
  c.fillStyle = fg; c.fillText(text, -w / 2 + 23, size * 0.38); c.restore();
}
function dimension(c, x0, y0, x1, y1, label, col = '#ff3b4f', a = 1) {
  c.save(); c.globalAlpha = a; c.strokeStyle = col; c.fillStyle = col; c.lineWidth = 5; c.lineCap = 'round';
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
  const ang = Math.atan2(y1 - y0, x1 - x0), nx = -Math.sin(ang) * 16, ny = Math.cos(ang) * 16;
  c.beginPath(); c.moveTo(x0 - nx, y0 - ny); c.lineTo(x0 + nx, y0 + ny); c.moveTo(x1 - nx, y1 - ny); c.lineTo(x1 + nx, y1 + ny); c.stroke();
  c.font = '36px "Permanent Marker"'; const w = textW(c, label); outlined(c, label, (x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - 22, col, 'rgba(8,8,24,0.9)', 8); c.restore();
}
// giant type punched in behind the pack
function bigWord(c, t, t0, text, x, y, size, rot, col = '#ffd23f') {
  const age = t - t0; if (age < 0 || age > 1.9) return; const k = popOut(age / 0.18), a = 1 - smooth((age - 1.2) / 0.7);
  c.save(); c.translate(x, y); c.rotate(rot); c.scale(k, k); c.globalAlpha = a; c.font = `${size}px "Anton"`; c.letterSpacing = '4px'; const w = textW(c, text);
  outlined(c, text, -w / 2 + 10, 14, 'rgba(0,0,0,0.35)', 'rgba(0,0,0,0)', 0); outlined(c, text, -w / 2, 0, col, 'rgba(10,8,30,0.9)', 16); c.restore();
}

// ---------------------------------------------------------------- route map (north points right so the long route fits a landscape frame)
const MAPB = (() => { let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (let i = 0; i < RT.n; i++) { x0 = Math.min(x0, RT.mx[i]); x1 = Math.max(x1, RT.mx[i]); y0 = Math.min(y0, RT.my[i]); y1 = Math.max(y1, RT.my[i]); } return { x0, x1, y0, y1 }; })();
function mapPt(i, z, cx, cy, fi) { // route sample i -> screen; z = px per metre; focus sample fi sits at (cx, cy)
  const fx = (RT.my[fi] - MAPB.y0), fy = (RT.mx[fi] - MAPB.x0), px = (RT.my[i] - MAPB.y0), py = (RT.mx[i] - MAPB.x0);
  return [cx + (px - fx) * z, cy + (py - fy) * z];
}
function sceneMap(c, t, shot) {
  const P = new Pal(t), tl = t - shot.t0, d = shot.t1 - shot.t0, kind = shot.kind || 'draw', n = RT.n;
  const night = 0.85; c.fillStyle = '#070b17'; c.fillRect(0, 0, W, H);
  // blueprint grid
  c.strokeStyle = 'rgba(80,120,190,0.10)'; c.lineWidth = 1; for (let x = 0; x < W; x += 60) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); } for (let y = 0; y < H; y += 60) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
  // camera: z (px/m) and focus index
  const fitZ = 1500 / (MAPB.y1 - MAPB.y0), headI = Math.round(clamp(shot.head ? shot.head(t) : odo(t), 0, RT.len - 1) / STEP);
  let z = fitZ, fi = Math.round(n / 2), cx = W * 0.5, cy = H * 0.58;
  if (shot.zoom) { const q = shot.zoom(t); z = fitZ * q[0]; fi = q[1] === 'head' ? headI : q[1]; }
  else { const mid = (MAPB.y0 + MAPB.y1) / 2; fi = 0; cx = W * 0.5 - ((mid - RT.my[0]) * z) - 0; cy = H * 0.6 + 0; // centre whole route
    // find bounding centre
    const ccx = ((MAPB.y0 + MAPB.y1) / 2 - RT.my[0]) * z, ccy = ((MAPB.x0 + MAPB.x1) / 2 - RT.mx[0]) * z; cx = W * 0.5 - ccx; cy = H * 0.6 - ccy; }
  const pts = new Array(n); for (let i = 0; i < n; i++) pts[i] = mapPt(i, z, cx, cy, fi);
  const strokeRange = (a, b, w, col, alpha = 1) => { c.strokeStyle = col; c.globalAlpha = alpha; c.lineWidth = w; c.beginPath(); for (let i = a; i <= b; i++) i === a ? c.moveTo(pts[i][0], pts[i][1]) : c.lineTo(pts[i][0], pts[i][1]); c.stroke(); c.globalAlpha = 1; };
  c.lineCap = 'round'; c.lineJoin = 'round';
  const drawn = kind === 'draw' ? Math.round((n - 1) * smooth(tl / (d * 0.82))) : (kind === 'loop' ? n - 1 : n - 1);
  // faint full route
  strokeRange(0, n - 1, 6, 'rgba(120,160,230,0.22)');
  // speed-coloured route up to `drawn`
  const col = i => { const v = clamp((RT.spd[i] * 2.237 - 14) / 14); return `rgb(${Math.round(lerp(60, 255, v))},${Math.round(lerp(150, 235, v))},${Math.round(lerp(255, 120, v))})`; };
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < drawn; i += 4) { const j = Math.min(drawn, i + 4); c.strokeStyle = col(i); c.globalAlpha = 0.28; c.lineWidth = 16; c.beginPath(); c.moveTo(pts[i][0], pts[i][1]); for (let q = i + 1; q <= j; q++) c.lineTo(pts[q][0], pts[q][1]); c.stroke(); }
  c.restore();
  for (let i = 0; i < drawn; i += 4) { const j = Math.min(drawn, i + 4); c.strokeStyle = col(i); c.lineWidth = 6; c.beginPath(); c.moveTo(pts[i][0], pts[i][1]); for (let q = i + 1; q <= j; q++) c.lineTo(pts[q][0], pts[q][1]); c.stroke(); }
  // highlight window (bridge: the fast stretch)
  if (shot.hl) { const [a, b] = shot.hl.map(m => Math.round(m / STEP)); const pulse = 0.6 + 0.4 * Math.sin(t * 6); c.save(); c.globalCompositeOperation = 'lighter'; strokeRange(a, b, 26, '#ffe08a', 0.18 * pulse); strokeRange(a, b, 9, '#ffffff', 0.9); c.restore(); }
  // pack worm: 38 dots trailing the head
  const wormN = 38, head = headI;
  if (kind !== 'draw') for (let k = 0; k < wormN; k++) { const j = Math.max(0, head - k * (shot.wormGap || 2)), p = pts[j], rr = 8 - k * 0.07;
    c.save(); c.globalCompositeOperation = 'lighter'; glow(c, p[0], p[1], rr * 3.2, [255, 210, 90], 0.55); c.restore(); c.fillStyle = k < 6 ? CAST[NAMED[(k * 5 + 5) % 6]].jersey : FAR_J[k % FAR_J.length]; c.beginPath(); c.arc(p[0], p[1], rr, 0, TAU); c.fill(); }
  if (kind === 'draw' && drawn > 0) { const p = pts[drawn]; c.save(); c.globalCompositeOperation = 'lighter'; glow(c, p[0], p[1], 40, [255, 255, 255], 0.9); c.restore(); }
  // start / finish and turnaround
  const mark = (i, label, up) => { const p = pts[i]; c.fillStyle = '#fff'; c.beginPath(); c.arc(p[0], p[1], 11, 0, TAU); c.fill(); c.fillStyle = '#0b1020'; c.beginPath(); c.arc(p[0], p[1], 5, 0, TAU); c.fill();
    c.font = '34px "Bebas Neue"'; c.letterSpacing = '2px'; const w = textW(c, label); outlined(c, label, p[0] - w / 2, p[1] + (up ? -26 : 48), '#ffffff', 'rgba(5,8,18,0.95)', 7); };
  if (drawn > 5) mark(0, 'START · FINISH', true);
  const ti = Math.round(RM(35100) / STEP); if (drawn > ti) mark(ti, 'GARRETT PARK RD', false);
  (shot.labels || []).forEach(([m, text, t0]) => { if (t >= t0) { const i = Math.round(m / STEP), p = pts[i], k = popOut((t - t0) / 0.3); c.save(); c.translate(p[0], p[1]); c.scale(k, k); c.font = '34px "Bebas Neue"'; c.letterSpacing = '2px'; const w = textW(c, text) + 30; rrect(c, -w / 2, -92, w, 50, 8); c.fillStyle = 'rgba(255,214,90,0.96)'; c.fill(); c.beginPath(); c.moveTo(-10, -43); c.lineTo(0, -26); c.lineTo(10, -43); c.fill(); c.fillStyle = '#0b1020'; c.fillText(text, -w / 2 + 15, -56); c.restore(); } });
  // compass + scale
  c.save(); c.translate(120, H - 130); c.strokeStyle = 'rgba(255,255,255,0.6)'; c.lineWidth = 3; c.beginPath(); c.arc(0, 0, 38, 0, TAU); c.stroke(); c.fillStyle = '#ff3b4f'; c.beginPath(); c.moveTo(34, 0); c.lineTo(-10, -12); c.lineTo(-10, 12); c.closePath(); c.fill(); c.fillStyle = '#fff'; c.font = '28px "Bebas Neue"'; c.fillText('N', 48, 9); c.restore();
  c.strokeStyle = 'rgba(255,255,255,0.6)'; c.lineWidth = 4; c.beginPath(); c.moveTo(60, H - 60); c.lineTo(60 + 1000 * z, H - 60); c.stroke(); c.font = '26px "Bebas Neue"'; c.fillStyle = 'rgba(255,255,255,0.7)'; c.fillText('1 KM', 60, H - 70);
  if (shot.stats) { const k = smooth((tl - 0.9) / 0.6); c.save(); c.globalAlpha = k; c.font = '120px "Anton"'; c.fillStyle = '#ffd23f'; outlined(c, '25 MI', W * 0.05, H * 0.44, '#ffd23f', 'rgba(5,8,18,0.95)', 14); c.font = '120px "Anton"'; outlined(c, '600 FT', W * 0.05, H * 0.44 + 130, '#2ec4b6', 'rgba(5,8,18,0.95)', 14); c.restore(); }
  if (shot.loop) { // ghost pack looping the route
    const laps = 3, ph = (tl / d) * laps; for (let l = 0; l < 3; l++) { const q = ((ph - l * 0.18) % 1 + 1) % 1, i = Math.floor(q * (n - 1)), p = pts[i]; c.save(); c.globalCompositeOperation = 'lighter'; glow(c, p[0], p[1], 44, l === 0 ? [255, 230, 120] : [120, 180, 255], 0.7 - l * 0.18); c.restore(); }
    c.save(); c.globalCompositeOperation = 'lighter'; strokeRange(0, n - 1, 16, '#ffd23f', 0.12 + 0.1 * Math.sin(tl * 3)); c.restore();
  }
  const dk = c.createRadialGradient(W / 2, H / 2, H * 0.4, W / 2, H / 2, H); dk.addColorStop(0, 'rgba(0,0,0,0)'); dk.addColorStop(1, 'rgba(0,0,0,0.5)'); c.fillStyle = dk; c.fillRect(0, 0, W, H);
}

// ---------------------------------------------------------------- bike computer (head unit) macro
function sceneComputer(c, t, shot) {
  const P = new Pal(Math.min(t, 20)), tl = t - shot.t0, mode = shot.mode || 'boot', k = shot.k ? shot.k(t) : 1;
  // golden-hour bokeh backdrop
  const g = c.createLinearGradient(0, 0, 0, H); g.addColorStop(0, mode === 'boot' ? '#34527f' : '#241a45'); g.addColorStop(1, mode === 'boot' ? '#d9946a' : '#5a2f6a'); c.fillStyle = g; c.fillRect(0, 0, W, H);
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < 38; i++) { const x = (rnd(i, 1) * W * 1.2 + t * (4 + 6 * rnd(i, 2))) % (W * 1.2) - W * 0.1, y = H * (0.1 + 0.8 * rnd(i, 3)), r = 30 + 90 * rnd(i, 4); glow(c, x, y, r, rnd(i, 5) > 0.5 ? [255, 190, 110] : [140, 180, 255], 0.16 + 0.1 * rnd(i, 6)); }
  c.restore();
  // handlebar + stem
  c.save(); c.translate(W * 0.64, H * (shot.cy || 0.5) + 8 * Math.sin(t * 1.3)); c.rotate(-0.07 + 0.01 * Math.sin(t * 0.9)); const sc = shot.sc || 1; c.scale(sc, sc);
  c.fillStyle = '#16171c'; rrect(c, -90, 330, 180, 300, 40); c.fill(); c.fillStyle = '#23252c'; rrect(c, -1100, 430, 2200, 70, 35); c.fill();
  // device
  const dw = 640, dh = 860; c.shadowColor = 'rgba(0,0,0,0.5)'; c.shadowBlur = 50; c.shadowOffsetY = 28; rrect(c, -dw / 2, -dh / 2 + 60, dw, dh, 56); c.fillStyle = '#0d0e12'; c.fill(); c.shadowColor = 'transparent';
  rrect(c, -dw / 2 + 22, -dh / 2 + 82, dw - 44, dh - 44, 38); c.fillStyle = '#05070b'; c.fill();
  const sx = -dw / 2 + 40, sy = -dh / 2 + 100, sw = dw - 80, sh = dh - 80; c.save(); rrect(c, sx, sy, sw, sh, 28); c.clip();
  const on = smooth((tl - 0.5) / 0.4); c.fillStyle = `rgba(14,18,28,${on})`; c.fillRect(sx, sy, sw, sh); c.globalAlpha = on;
  c.font = '30px "Bebas Neue"'; c.letterSpacing = '2px'; c.fillStyle = '#7ee0ff'; c.fillText('GPS', sx + 26, sy + 52); c.fillStyle = '#ff6b8b'; c.fillText('HR', sx + 110, sy + 52); c.fillStyle = '#9aa3b5'; c.fillText('100%', sx + sw - 90, sy + 52);
  const lab = (s, x, y) => { c.font = '28px "Bebas Neue"'; c.letterSpacing = '2px'; c.fillStyle = 'rgba(255,255,255,0.55)'; c.fillText(s, x, y); };
  if (mode === 'boot') {
    const logo = smooth((tl - 1.0) / 0.5), clk = tl > 9.5 - shot.t0 ? 1 : 0, tt = shot.t0 + tl;
    c.globalAlpha = on * logo; c.font = '98px "Anton"'; c.fillStyle = '#ffd23f'; c.fillText('GEEZER', sx + 40, sy + 300); c.fillText('GOON', sx + 40, sy + 410); c.globalAlpha = on;
    lab('WEDNESDAY', sx + 40, sy + 480); const acq = clamp((tl - 3.0) / 2.0); lab(acq >= 1 ? 'SATELLITES  12' : 'ACQUIRING' + '.'.repeat(1 + Math.floor(tl * 3) % 3), sx + 40, sy + 530);
    c.fillStyle = '#2ec4b6'; c.fillRect(sx + 40, sy + 560, (sw - 80) * acq, 8);
    lab('TIME', sx + 40, sy + 660); const flip = tt >= 9.5; c.font = '120px "Anton"'; c.fillStyle = flip ? '#fff' : '#9aa3b5'; c.fillText(flip ? '6:15' : '6:14', sx + 40, sy + 770);
    if (flip) { c.globalAlpha = 1 - smooth((tt - 9.5) / 0.35); c.fillStyle = '#fff'; c.fillRect(sx, sy, sw, sh); }
  } else if (mode === 'speed') {
    const mph = mphAt(t) + 0.35 * Math.sin(t * 7); lab('SPEED', sx + 40, sy + 120); c.font = '330px "Anton"'; c.fillStyle = '#ffd23f'; c.fillText(mph.toFixed(1), sx + 20 + 3 * Math.sin(t * 50), sy + 440 + 3 * Math.cos(t * 43));
    c.font = '60px "Bebas Neue"'; c.fillStyle = '#fff'; c.fillText('MPH', sx + sw - 120, sy + 500);
    lab('HEART RATE', sx + 40, sy + 540); c.font = '120px "Anton"'; c.fillStyle = '#ff6b8b'; c.fillText(String(Math.round(138 + 26 * smooth((t - 154) / 18))), sx + 40, sy + 650);
    lab('MI', sx + 360, sy + 540); c.font = '120px "Anton"'; c.fillStyle = '#fff'; c.fillText(milesAt(t).toFixed(1), sx + 360, sy + 650);
    c.strokeStyle = '#2ec4b6'; c.lineWidth = 6; c.beginPath(); const fi = Math.round(odo(t) / STEP); for (let q = 0; q < 60; q++) { const i = clamp(fi - 60 + q, 0, RT.n - 1), x = sx + 40 + q * (sw - 80) / 59, y = sy + 790 - (RT.alt[i] - 60) * 2.2; q ? c.lineTo(x, y) : c.moveTo(x, y); } c.stroke();
  } else { // elapsed
    const e = clamp((tl - 0.2) / 1.3), mins = 77 * smooth(e), mm = Math.floor(mins), ss = Math.floor((mins - mm) * 60);
    lab('ELAPSED TIME', sx + 40, sy + 120); c.font = '215px "Anton"'; c.fillStyle = '#ffd23f'; c.fillText(`${mm}:${String(ss).padStart(2, '0')}`, sx + 20, sy + 400);
    lab('DISTANCE  MI', sx + 40, sy + 520); c.font = '120px "Anton"'; c.fillStyle = '#fff'; c.fillText((RT.len / 1609.344 * smooth(e)).toFixed(1), sx + 40, sy + 640);
    lab('CLIMB  FT', sx + 40, sy + 730); c.font = '120px "Anton"'; c.fillStyle = '#2ec4b6'; c.fillText(String(Math.round(600 * smooth(e))), sx + 40, sy + 850);
    if (e >= 1) { c.globalAlpha = 1; const a = 0.5 + 0.5 * Math.sin(tl * 9); c.fillStyle = `rgba(255,59,79,${a})`; c.font = '60px "Bebas Neue"'; c.fillText('● SAVED', sx + 360, sy + 80); }
  }
  c.restore(); c.restore();
  // glass glare
  c.save(); c.globalCompositeOperation = 'lighter'; const gr = c.createLinearGradient(W * 0.42, 0, W * 0.78, H); gr.addColorStop(0.35, 'rgba(255,255,255,0)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.05)'); gr.addColorStop(0.65, 'rgba(255,255,255,0)'); c.fillStyle = gr; c.fillRect(0, 0, W, H); c.restore();
}

// ---------------------------------------------------------------- calendar flip
function sceneCalendar(c, t, shot) {
  const tl = t - shot.t0, d = shot.t1 - shot.t0, P = new Pal(t); const mode = shot.mode || 'tuethu';
  c.fillStyle = P.s('#27234f'); c.fillRect(0, 0, W, H);
  c.save(); c.globalCompositeOperation = 'lighter'; for (let i = 0; i < 24; i++) glow(c, rnd(i, 1) * W, rnd(i, 2) * H, 100 + 160 * rnd(i, 3), [255, 150, 200], 0.06); c.restore();
  let faces;
  if (mode === 'tuethu') faces = [[0, 'TUE?'], [1.6, 'THU?'], [3.3, '???'], [4.8, 'WED']]; else faces = [[0, 'WED'], [1.0, 'WED'], [2.0, 'WED'], [3.0, 'WED']];
  let cur = 0, ch = 0; faces.forEach(([a], i) => { if (tl >= a) { cur = i; ch = a; } });
  const flip = smooth((tl - ch) / 0.2), day = faces[cur][1];
  const week = mode === 'weeks' ? Math.floor(tl / 0.8) : 0;
  c.save(); c.translate(W / 2, H * 0.58); c.rotate(0.03 * Math.sin(tl * 3)); c.scale(1.5, 1.5);
  rrect(c, -230, -230, 460, 440, 28); c.fillStyle = 'rgba(0,0,0,0.35)'; c.save(); c.translate(10, 14); c.fill(); c.restore();
  rrect(c, -230, -230, 460, 440, 28); c.fillStyle = '#fff'; c.fill(); c.save(); rrect(c, -230, -230, 460, 440, 28); c.clip(); c.fillStyle = '#e63946'; c.fillRect(-230, -230, 460, 110); c.restore();
  c.fillStyle = '#fff'; c.font = '66px "Bebas Neue"'; c.letterSpacing = '4px'; const l = mode === 'weeks' ? `WEEK ${week + 1}` : 'RIDE DAY'; c.fillText(l, -textW(c, l) / 2, -148);
  c.save(); c.scale(1, Math.max(0.06, flip)); c.fillStyle = '#16161c'; c.font = '200px "Anton"'; const wd = textW(c, day); c.fillText(day, -wd / 2, 60 / Math.max(0.06, flip) * flip); c.restore();
  if (day === 'WED' || mode === 'weeks') { c.strokeStyle = '#e63946'; c.lineWidth = 16; c.beginPath(); c.moveTo(-170, 100); c.lineTo(165, 118); c.stroke(); }
  if (mode === 'weeks') for (let w = 0; w <= week && w < 6; w++) { c.fillStyle = '#2ec4b6'; c.beginPath(); c.arc(-170 + w * 68, 168, 20, 0, TAU); c.fill(); }
  c.restore();
}

// ---------------------------------------------------------------- gears (Angelo): a card drawn over the world
function drawGearsCard(c, t, t0, t1) {
  if (t < t0 || t > t1 + 0.25) return; const k = popOut((t - t0) / 0.3) * (1 - smooth((t - t1) / 0.25)); if (k <= 0.02) return;
  c.save(); c.translate(60, 560); c.scale(k, k); rrect(c, 0, 0, 470, 270, 20); c.fillStyle = 'rgba(8,10,16,0.88)'; c.fill();
  c.font = '30px "Bebas Neue"'; c.letterSpacing = '2px'; c.fillStyle = 'rgba(255,255,255,0.7)'; c.fillText('REAR COGS', 26, 46);
  c.fillStyle = '#fff'; c.fillText('EVERYONE ELSE', 26, 112); c.fillStyle = '#ffd23f'; c.fillText('ANGELO', 26, 214);
  for (let i = 0; i < 13; i++) { const on = (t - t0) * 9 > i; const h = 22 + (12 - i) * 3.4; c.fillStyle = on ? '#cfd4dc' : 'rgba(255,255,255,0.15)'; rrect(c, 200 + i * 18, 126 - h, 12, h, 3); c.fill(); }
  c.font = '44px "Anton"'; c.fillStyle = '#fff'; c.fillText('13', 436, 122);
  c.fillStyle = '#ffd23f'; rrect(c, 200, 160, 14, 48, 3); c.fill(); c.font = '44px "Anton"'; c.fillText('1', 436, 212); c.restore();
}
function drawBackpackIcons(c, t, t0, x, y) {
  [['LAPTOP', 0.0, -150, -100, '#8ecae6'], ['BADGE', 0.45, 20, -190, '#ffd166'], ['SHOES', 0.9, 160, -80, '#ef476f']].forEach(([name, dt, dx, dy, col]) => {
    const age = t - t0 - dt; if (age < 0 || age > 3.2) return; const k = popOut(age / 0.3), fl = 6 * Math.sin(t * 3 + dx);
    c.save(); c.translate(x + dx * 1.1, y + dy + fl); c.scale(k, k); c.rotate(0.06 * Math.sin(t * 2 + dy));
    rrect(c, -95, -50, 190, 100, 16); c.fillStyle = col; c.fill(); c.strokeStyle = 'rgba(8,8,24,0.9)'; c.lineWidth = 6; c.stroke(); c.fillStyle = '#10121c';
    if (name === 'LAPTOP') { rrect(c, -48, -30, 96, 52, 6); c.fill(); c.fillStyle = col; c.fillRect(-70, 24, 140, 8); }
    else if (name === 'BADGE') { rrect(c, -32, -34, 64, 68, 8); c.fill(); c.fillStyle = col; c.beginPath(); c.arc(0, -10, 12, 0, TAU); c.fill(); c.fillRect(-22, 8, 44, 6); }
    else { c.beginPath(); c.ellipse(-20, 8, 40, 18, 0, 0, TAU); c.fill(); c.beginPath(); c.ellipse(34, 14, 36, 16, 0, 0, TAU); c.fill(); }
    c.font = '34px "Anton"'; c.fillStyle = '#fff'; const w = textW(c, name); outlined(c, name, -w / 2, 88, '#fff', 'rgba(8,8,24,0.95)', 8); c.restore();
  });
}
