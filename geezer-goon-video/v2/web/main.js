'use strict';
const canvas = document.getElementById('c'); const ctx = canvas.getContext('2d', { alpha: false });
const bgc = document.createElement('canvas'); bgc.width = W; bgc.height = H; const bgx = bgc.getContext('2d', { alpha: false });
const bloomc = document.createElement('canvas'); bloomc.width = 480; bloomc.height = 270; const bloomx = bloomc.getContext('2d');
const sceneA = document.createElement('canvas'); sceneA.width = W; sceneA.height = H; const sax = sceneA.getContext('2d', { alpha: false });
const grain = document.createElement('canvas'); grain.width = grain.height = 256; (function () {
  const g = grain.getContext('2d'), im = g.createImageData(256, 256);
  for (let i = 0; i < 256 * 256; i++) { const v = 128 + (rnd(i, 3) - 0.5) * 120; im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v; im.data[i * 4 + 3] = 255; } g.putImageData(im, 0, 0);
})();

// ---------------------------------------------------------------- camera
function camAt(shot, t, L) {
  const c = shot.cam || {}, tl = t - shot.t0, d = shot.t1 - shot.t0, k = smooth(tl / d);
  const two = (a, dflt) => { a = a === undefined ? dflt : a; return Array.isArray(a) ? lerp(a[0], a[1], k) : a; };
  const z = two(c.z, 1) * (1 + (shot.punch ? 0.016 * beatPulse(t) * clamp(energy(t) * 1.3) : 0));
  const pan = [two(c.px, 0), two(c.py, 0)], rot = two(c.rot, 0) + (shot.shake ? 0.0035 * Math.sin(t * 2.1) : 0);
  let fx = W / 2, fy = H / 2; const w = two(c.w, c.who ? 1 : 0);
  if (c.who && L && L.byId[c.who]) { const r = L.byId[c.who]; fx = lerp(W / 2, r.x + two(c.fdx, 0) * r.s, w); fy = lerp(H / 2, r.y - two(c.fdy, 0.9) * r.s, w); }
  const sh = shot.shake ? [Math.sin(t * 37) * 2.4, Math.cos(t * 31) * 2.4] : [0, 0];
  return { z, px: pan[0] + sh[0], py: pan[1] + sh[1], rot, fx, fy, dof: two(c.dof, 0) };
}
function applyCam(c, cam) { c.setTransform(1, 0, 0, 1, 0, 0); c.translate(W / 2 + cam.px, H / 2 + cam.py); c.scale(cam.z, cam.z); c.rotate(cam.rot); c.translate(-cam.fx, -cam.fy); }
function projCam(cam, x, y) { // world(screen@z1) -> final screen
  let dx = x - cam.fx, dy = y - cam.fy; dx *= cam.z; dy *= cam.z; const cs = Math.cos(cam.rot), sn = Math.sin(cam.rot);
  return [W / 2 + cam.px + dx * cs - dy * sn, H / 2 + cam.py + dx * sn + dy * cs];
}

// ---------------------------------------------------------------- lyrics
const LYR = D.lines.map(([a, b, s]) => ({ t0: a, t1: b, text: s.toUpperCase() }));
function lineWords(l) {
  const words = l.text.split(' '), win = Math.min((l.t1 - l.t0) * 0.93, 0.072 * l.text.length + 0.3);
  const w = words.map(x => x.length + 2), sum = w.reduce((a, b) => a + b, 0); let acc = 0; const st = [], en = [];
  words.forEach((x, i) => { st.push(l.t0 + acc / sum * win); acc += w[i]; en.push(l.t0 + acc / sum * win); }); return { words, st, en };
}
function lyricLayout(c, text, size, maxw) {
  c.font = `${size}px "Anton"`; c.letterSpacing = `${size * 0.012}px`; const sp = textW(c, ' ') + size * 0.1, words = text.split(' ');
  const rows = []; let cur = [], cw = 0;
  words.forEach((wd, i) => { const ww = textW(c, wd); if (cur.length && cw + sp + ww > maxw) { rows.push({ cur, cw }); cur = []; cw = 0; } cw += (cur.length ? sp : 0) + ww; cur.push({ i, wd, ww }); });
  rows.push({ cur, cw }); return { rows, sp };
}
function drawLyrics(c, t, o = {}) {
  let idx = -1; for (let i = 0; i < LYR.length; i++) if (LYR[i].t0 <= t) idx = i; if (idx < 0 || t > 244.6) return;
  const l = LYR[idx], age = t - l.t0, shout = l.text === 'GATE UP!';
  const fadeIn = smooth(age / 0.18), outA = idx + 1 < LYR.length ? 1 - smooth((t - (LYR[idx + 1].t0 - 0.12)) / 0.12) : 1 - smooth((t - l.t1) / 0.5);
  const a = fadeIn * clamp(outA);
  const { words, st, en } = lineWords(l), size = shout ? 190 : (o.size || (l.text.length > 46 ? 74 : 86)), CXL = (o.cx || 0.44) * W;
  const yMid = (o.y || 0.32) * H;
  c.save(); c.textBaseline = 'alphabetic'; c.globalAlpha = a;
  if (shout) {
    const k = popOut(age / 0.2), shake = 16 * Math.exp(-age / 0.25);
    c.translate(CXL + shake * Math.sin(t * 90), yMid); c.rotate(-0.045); c.scale(k, k); c.font = `${size}px "Anton"`; const wd = textW(c, 'GATE UP!');
    outlined(c, 'GATE UP!', -wd / 2, size * 0.34, '#ff3b4f', 'rgba(255,255,255,0.95)', 20); c.restore(); return;
  }
  const { rows, sp } = lyricLayout(c, l.text, size, (o.maxw || 0.66) * W);
  const top = yMid - (rows.length - 1) * size * 0.59 + (1 - fadeIn) * 22;
  rows.forEach((row, ri) => {
    let x = CXL - row.cw / 2; const y = top + ri * size * 1.17;
    row.cur.forEach(({ i, wd, ww }) => {
      c.font = `${size}px "Anton"`; c.letterSpacing = `${size * 0.012}px`;
      const prog = clamp((t - st[i]) / Math.max(0.08, en[i] - st[i])), on = t >= st[i], done = t >= en[i];
      const pop = on ? 1 + 0.07 * Math.exp(-(t - st[i]) / 0.13) : 1;
      c.save(); c.translate(x + ww / 2, y); c.scale(pop, pop);
      const base = done ? '#ffffff' : (on ? '#ffffff' : 'rgba(255,255,255,0.4)');
      c.lineJoin = 'round'; c.lineWidth = size * 0.14; c.strokeStyle = 'rgba(10,8,30,0.92)'; c.strokeText(wd, -ww / 2, size * 0.02 + 6);
      c.lineWidth = size * 0.11; c.strokeText(wd, -ww / 2, 0);
      c.fillStyle = base; c.fillText(wd, -ww / 2, 0);
      if (on && !done) { c.save(); c.beginPath(); c.rect(-ww / 2 - 4, -size, (ww + 8) * prog, size * 1.3); c.clip(); c.fillStyle = '#ffd23f'; c.fillText(wd, -ww / 2, 0); c.restore(); }
      else if (done) { c.globalAlpha = a * (0.35 * Math.exp(-(t - en[i]) / 0.5)); c.fillStyle = '#ffd23f'; c.fillText(wd, -ww / 2, 0); c.globalAlpha = a; }
      c.restore(); x += ww + sp;
    });
  });
  c.restore();
}

// ---------------------------------------------------------------- HUD + section label
function sectionAt(t) { let s = D.sections[0]; D.sections.forEach(x => { if (x[0] <= t + 1e-6) s = x; }); return s; }
function drawHUD(c, t, o = {}) {
  const k = smooth((t - 9.6) / 0.6) * (1 - smooth((t - 242.5) / 1.0)) * (o.hide ? 0 : 1); if (k <= 0.01) return;
  const x0 = W - 472, y0 = 34 - (1 - k) * 320, w = 450, h = 262;
  c.save(); rrect(c, x0, y0, w, h, 22); c.fillStyle = 'rgba(8,10,16,0.82)'; c.fill(); c.strokeStyle = 'rgba(255,255,255,0.16)'; c.lineWidth = 2; c.stroke();
  const lab = (s, x, y) => { c.font = '22px "Bebas Neue"'; c.letterSpacing = '1px'; c.fillStyle = 'rgba(255,255,255,0.55)'; c.fillText(s, x, y); };
  const val = (s, x, y, px, col = '#fff') => { c.font = `${px}px "Anton"`; c.fillStyle = col; c.fillText(s, x, y); };
  const ff = odoRate(t) > 380;
  lab('SPEED  MPH', x0 + 26, y0 + 38); val((mphAt(t) * (t < 19 ? 0 : Math.min(1, (t - 19) / 5))).toFixed(1).padStart(4, ' '), x0 + 24, y0 + 130, 90, '#ffd23f');
  lab('TIME', x0 + 280, y0 + 38); val(clockStr(t) + ' PM', x0 + 280, y0 + 82, 38);
  lab('DIST  MI', x0 + 280, y0 + 112); val(milesAt(t).toFixed(1), x0 + 280, y0 + 154, 38);
  lab(`CLIMB  ${Math.round(climbFt(t))} FT`, x0 + 26, y0 + 166);
  if (ff) { c.fillStyle = '#ff3b4f'; c.font = '20px "Bebas Neue"'; c.fillText('▶▶ FAST FORWARD', x0 + 232, y0 + 166); }
  const px0 = x0 + 26, py0 = y0 + 180, pw = w - 52, ph = 58, aMin = 28, aMax = 118, n = RT.n;
  const prof = () => { c.beginPath(); c.moveTo(px0, py0 + ph); for (let i = 0; i < n; i += 8) c.lineTo(px0 + pw * i / (n - 1), py0 + ph - ph * (RT.alt[i] - aMin) / (aMax - aMin)); c.lineTo(px0 + pw, py0 + ph); c.closePath(); };
  prof(); c.fillStyle = 'rgba(255,255,255,0.14)'; c.fill();
  const frac = odo(t) / RT.len; c.save(); c.beginPath(); c.rect(px0, py0 - 6, pw * frac, ph + 12); c.clip(); prof(); c.fillStyle = 'rgba(46,196,182,0.85)'; c.fill(); c.restore();
  const mi = Math.min(n - 1, Math.round(odo(t) / STEP)); c.fillStyle = '#ffd23f'; c.beginPath(); c.arc(px0 + pw * frac, py0 + ph - ph * (RT.alt[mi] - aMin) / (aMax - aMin), 7, 0, TAU); c.fill();
  c.restore();
  const [st, name] = sectionAt(t), kk = smooth((t - st) / 0.4);
  c.save(); c.translate(40 - (1 - kk) * 300, 58); c.font = '26px "Bebas Neue"'; c.letterSpacing = '2px'; c.fillStyle = 'rgba(255,255,255,0.75)'; c.fillText('GEEZER GOON  ·  G2  ·  WEDNESDAY', 0, 0);
  c.font = '54px "Anton"'; c.letterSpacing = '1px'; outlined(c, name, 0, 64, '#ffd23f', 'rgba(8,8,24,0.9)', 7); const e = textW(c, name); c.fillStyle = '#ffd23f'; c.fillRect(0, 78, e * kk, 5); c.restore();
}

// ---------------------------------------------------------------- post
function post(c, t, o = {}) {
  // bloom
  bloomx.globalCompositeOperation = 'copy'; bloomx.filter = 'blur(5px) brightness(1.15)'; bloomx.drawImage(canvas, 0, 0, 480, 270); bloomx.filter = 'none';
  c.save(); c.globalCompositeOperation = 'lighter'; c.globalAlpha = (o.bloom === undefined ? 0.15 : o.bloom) * (0.75 + 0.25 * beatPulse(t)); c.drawImage(bloomc, 0, 0, W, H); c.restore();
  // vignette
  const g = c.createRadialGradient(W / 2, H / 2, H * 0.42, W / 2, H / 2, H * 1.05); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.55)'); c.fillStyle = g; c.fillRect(0, 0, W, H);
  // grain
  c.save(); c.globalCompositeOperation = 'overlay'; c.globalAlpha = 0.07; const gx = Math.floor(rnd(Math.floor(t * 30), 1) * 256), gy = Math.floor(rnd(Math.floor(t * 30), 2) * 256);
  for (let x = -gx; x < W; x += 256) for (let y = -gy; y < H; y += 256) c.drawImage(grain, x, y); c.restore();
}

