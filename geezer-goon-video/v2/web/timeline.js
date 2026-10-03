'use strict';
// ---------------------------------------------------------------- shot list (times follow the lyric timing from the song file)
const SH = [];
const S = (t0, t1, type, p = {}) => SH.push(Object.assign({ t0, t1, type }, p));
const FOLLOW = (who, z, o = {}) => Object.assign({ z, who, w: 1, fdy: 0.9, px: -160, py: 215 }, o);

// intro: head-unit boot, then the pack gathers at the corner and rolls out
S(0, 9.5, 'computer', { mode: 'boot', hud: false, lyr: { cx: 0.27, y: 0.42, maxw: 0.44, size: 74 } });
S(9.5, 14.7, 'side', { cam: { z: [1, 1.08], px: [0, 0] }, flash: 1 });
S(14.7, 19.2, 'side', { cam: { z: [1.1, 2.0], who: 'karim', w: [0.2, 1], fdy: 0.9, px: [0, -150], py: [0, 150], dof: [0, 4] } });
S(19.2, 26.5, 'side', { cam: { z: [2.0, 1.0], who: 'karim', w: [1, 0], fdy: 0.9, px: [-150, 0], py: [150, 0], dof: [4, 0] }, shake: 0 });
// verse 1
S(26.5, 30.0, 'side', { cam: { z: [1.0, 1.12] }, punch: 1, flash: 1 });
S(30.0, 33.0, 'side', { cam: FOLLOW('karim', [1.5, 1.7], { dof: 3 }), punch: 1 });
S(33.0, 36.4, 'side', { cam: FOLLOW('hulk', [1.7, 2.0], { dof: 4, px: -220 }) });
S(36.4, 39.6, 'side', { cam: FOLLOW('hulk', [2.0, 1.45], { dof: [4, 2], px: -200 }) });
S(39.6, 42.7, 'side', { cam: FOLLOW('phil', [1.5, 1.9], { dof: 3 }) });
S(42.7, 46.3, 'side', { cam: FOLLOW('gary', [1.9, 2.1], { dof: 4, fdx: 0.2, px: -60 }) });
S(46.3, 55.8, 'topdown', { kind: 'gate', walk: 'stroller', flash: 1 });
// chorus 1
S(55.8, 59.1, 'side', { cam: { z: [1.0, 1.06] }, punch: 1, flash: 1 });
S(59.1, 62.3, 'side', { cam: FOLLOW('karim', [1.45, 1.25], { w: [1, 0.6], dof: 2 }), punch: 1 });
S(62.3, 66.1, 'side', { cam: FOLLOW('me', [3.0, 3.5], { fdy: 0.38, px: -80, py: 60, dof: 8 }), punch: 1 });
S(66.1, 72.0, 'side', { cam: { z: [1.0, 1.28], who: 'phil', w: [0, 0.45], py: [0, -40] }, punch: 1 });
S(72.0, 75.0, 'map', { kind: 'draw', stats: 1, lyr: { y: 0.15 }, hud: false });
S(75.0, 78.4, 'side', { cam: { z: [1.18, 1.0], rot: [0.02, 0] } });
S(78.4, 81.94, 'calendar', { mode: 'weeks', hud: false, lyr: { y: 0.14 } });
// verse 2
S(81.94, 85.5, 'side', { cam: { z: [1.0, 1.18], who: 'phil', w: [0, 0.5], py: [0, 30] } });
S(85.5, 88.4, 'side', { cam: FOLLOW('karim', [1.5, 1.75], { dof: 3 }), punch: 1, shake: 1 });
S(88.4, 91.7, 'side', { cam: { z: [1.18, 1.06] } });
S(91.7, 95.1, 'side', { cam: FOLLOW('hulk', [1.3, 1.3], { w: 0.6, dof: 2 }), shake: 1 });
S(95.1, 97.7, 'side', { cam: { z: [1.2, 1.2], who: 'karim', w: 0.3 } });
S(97.7, 101.65, 'pothole', { hud: false, lyr: { y: 0.2 } });
S(101.65, 111.1, 'topdown', { kind: 'gate', walk: 'scatter', flash: 1 });
// verse 3: Mormon Hill
S(111.1, 114.4, 'side', { cam: { z: [1.0, 1.1], rot: [0, -0.02] }, flash: 1 });
S(114.4, 117.7, 'side', { cam: FOLLOW('angelo', [1.8, 2.05], { fdy: 0.8, px: -140, py: 230, dof: 4 }) });
S(117.7, 120.9, 'side', { cam: FOLLOW('angelo', [2.5, 2.9], { fdx: -0.15, fdy: 1.0, px: -60, py: 190, dof: 6 }) });
S(120.9, 124.1, 'side', { cam: FOLLOW('angelo', [1.7, 1.35], { dof: 3 }) });
S(124.1, 127.1, 'side', { cam: { z: [1.0, 1.0] } });
S(127.1, 130.8, 'side', { cam: { z: [1.05, 1.22], who: 'karim', w: [0, 0.5] }, mute: 1 });
// chorus 2
S(130.8, 134.0, 'map', { kind: 'worm', hud: false, lyr: { y: 0.15 }, head: t => lerp(RM(27000), RM(29600), (t - 130.8) / 3.2), wormGap: 3, zoom: t => [lerp(1.6, 4.2, smooth((t - 130.8) / 3.2)), 'head'], flash: 1 });
S(134.0, 137.2, 'side', { cam: FOLLOW('karim', [1.25, 1.1], { w: [1, 0.5], dof: 1 }), punch: 1 });
S(137.2, 143.7, 'side', { cam: FOLLOW('phil', [1.25, 1.65], { w: [0.4, 1], fdy: 0.7, py: 130 }), punch: 1 });
S(143.7, 149.8, 'topdown', { kind: 'drone' });
S(149.8, 153.9, 'side', { cam: { z: [1.0, 1.0] } });
// bridge: the speed comes up
S(153.9, 163.2, 'map', { kind: 'worm', hud: false, lyr: { y: 0.15 }, head: t => odo(t), hl: [RM(30300), RM(33800)], wormGap: 2, flash: 1,
  zoom: t => [lerp(1.0, 3.0, smooth((t - 154.5) / 6)), 'head'], labels: [[RM(30400), 'BEACH DR', 154.4], [RM(31500), 'FRANKLIN ST', 157.4]] });
S(163.2, 166.4, 'side', { cam: FOLLOW('karim', [1.3, 1.5], { w: 0.6 }), shake: 1, punch: 1 });
S(166.4, 169.6, 'computer', { mode: 'speed', hud: false, k: () => 1, lyr: { cx: 0.27, y: 0.42, maxw: 0.44, size: 74 }, flash: 1 });
S(169.6, 171.0, 'side', { cam: FOLLOW('me', [1.6, 2.0], { dof: 3 }), shake: 1 });
S(171.0, 173.2, 'side', { cam: { z: [4.2, 5.3], who: 'me', w: 1, fdx: 0.41, fdy: 1.42, px: [-140, -60], py: [90, 60], dof: 12 }, shake: 1 });
S(173.2, 176.1, 'side', { cam: { z: [1.05, 1.12] }, shake: 1 });
S(176.1, 184.23, 'calendar', { mode: 'tuethu', hud: false, lyr: { y: 0.14 } });
// verse 4
S(184.23, 186.46, 'side', { cam: { z: [1.0, 1.15] }, shake: 1, flash: 1 });
S(186.46, 192.1, 'side', { cam: FOLLOW('karim', [1.6, 1.3], { w: [1, 0.7], dof: 2 }), shake: 1, punch: 1 });
S(192.1, 195.3, 'side', { cam: FOLLOW('me', [2.1, 2.3], { fdy: 0.6, px: -100, py: 235, dof: 6 }), shake: 1 });
S(195.3, 198.5, 'side', { cam: { z: [1.1, 1.0], who: 'karim', w: [0.4, 0] } });
// final chorus (night)
S(198.5, 201.8, 'split', { flash: 1, who: ['karim', 'phil', 'hulk'], lyr: { y: 0.15 } });
S(201.8, 204.9, 'side', { cam: { z: [1.0, 1.1] }, punch: 1 });
S(204.9, 211.3, 'side', { cam: FOLLOW('karim', [1.8, 1.5], { w: [1, 0.8], fdy: 0.7, dof: 3 }), shake: 1, punch: 1, flash: 1 });
S(211.3, 214.07, 'topdown', { kind: 'drone' });
S(214.07, 216.54, 'computer', { mode: 'elapsed', sc: 0.84, cy: 0.43, hud: false, lyr: { cx: 0.27, y: 0.42, maxw: 0.44, size: 74 } });
S(216.54, 224.28, 'map', { kind: 'loop', loop: 1, hud: false, lyr: { y: 0.15 }, title: 1 });
// outro
S(224.28, 230.5, 'topdown', { kind: 'gate', walk: 'none', flash: 1 });
S(230.5, 233.9, 'side', { cam: FOLLOW('karim', [1.4, 1.15], { w: [1, 0.3] }) });
S(233.9, 237.1, 'side', { cam: { z: [1.0, 1.0] } });
S(237.1, 260, 'side', { cam: { z: [1.0, 1.0] } });
const shotAt = t => { for (let i = SH.length - 1; i >= 0; i--) if (t >= SH[i].t0) return SH[i]; return SH[0]; };
const FLASHES = SH.filter(s => s.flash).map(s => s.t0);

// ---------------------------------------------------------------- annotations
const TAGS = [['karim', 'KARIM', 'ON THE FRONT', 14.7, 22.5], ['hulk', 'THE HULK', 'BACK AND STRONG', 36.4, 39.5], ['phil', 'PHIL', 'TAKING A PULL', 39.6, 42.7],
  ['gary', 'GARY', 'NOT MOVING', 42.7, 46.2], ['angelo', 'ANGELO', 'ONE GEAR', 114.4, 120.9], ['me', 'ME', 'HANGING ON', 169.6, 171.0], ['me', 'ME', 'FAT TIRES', 192.1, 195.3]];
const STICKERS = [
  [36.8, 39.5, 'SEGMENTS TO OWN', 0.56, 0.13, 0.05, '#ffd23f', '#1b1b1b', 46],
  [62.3, 66.1, 'NOT A RACE*', 0.6, 0.13, 0.06, '#ffffff', '#e63946', 50], [63.9, 66.1, '*legs disagree', 0.6, 0.2, -0.04, '#1b1b1b', '#ffd23f', 34],
  [137.2, 143.7, 'NOT A RACE*', 0.6, 0.13, 0.06, '#ffffff', '#e63946', 50], [139.4, 143.7, '*legs disagree', 0.6, 0.2, -0.04, '#1b1b1b', '#ffd23f', 34],
  [205.2, 211.3, 'NOT A RACE*', 0.58, 0.13, 0.06, '#ffffff', '#e63946', 50], [207.2, 211.3, '*legs disagree', 0.58, 0.2, -0.04, '#1b1b1b', '#ffd23f', 34],
  [111.2, 114.3, 'OH YOU SUCK', 0.56, 0.13, 0.07, '#e63946', '#ffffff', 52], [120.9, 124.1, 'A WEEK OF MONDAYS', 0.5, 0.13, -0.04, '#264653', '#e9c46a', 40],
  [186.6, 190.2, 'JUMP!', 0.6, 0.13, -0.06, '#e63946', '#ffffff', 56], [195.4, 198.5, "CAUGHT 'EM", 0.56, 0.13, 0.05, '#2ec4b6', '#0b1d1f', 50],
  [75.2, 78.0, '25 MI · 600 FT', 0.3, 0.13, -0.04, '#2ec4b6', '#0b1d1f', 46],
];
const KEYS = Object.keys(CAST);
function riderOver(t) {
  const o = {};
  if (t >= 42.7 && t < 46.3) o.gary = { eyes: 'squint' };
  if (t >= 32.9 && t < 39.6) o.hulk = { mood: t < 36.4 ? 'grit' : 'smirk' };
  if (t >= 160 && t < 184.3) o.me = { eyes: 'wide', mood: 'o', stress: smooth((t - 160) / 9), look: 0.004 * Math.sin(t * 4) };
  else if (t >= 186 && t < 198.5) o.me = { mood: 'grit', stress: 0.7 };
  if (t >= 205 && t < 211.3) { o.karim = { mood: 'grit' }; o.phil = { mood: 'grit' }; }
  if (t >= 114 && t < 125) o.angelo = { mood: 'grit', stress: 0.8 };
  if (t >= 124.1 && t < 131) o.angelo = { mood: 'smirk' };
  if (t >= 127.1 && t < 130.8) { o.karim = { mood: 'neutral' }; }
  return o;
}
// "Geezer Goon" punched in behind the pack on each sung word
const BIGW = []; LYR.forEach(l => { if (l.text.startsWith('GEEZER GOON')) { const { words, st } = lineWords(l); words.forEach((w, i) => BIGW.push([st[i], w.replace(',', ''), i % 2 === 0 ? 0.27 : 0.6, i % 2 === 0 ? -0.05 : 0.045])); } });
function drawBigWords(c, t) {
  c.save(); c.setTransform(1, 0, 0, 1, 0, 0); BIGW.forEach(([t0, w, xf, rot]) => bigWord(c, t, t0, w, xf * W, H * 0.2, 230, rot, w === 'GEEZER' ? '#ffd23f' : '#ff7a59')); c.restore();
}

// ---------------------------------------------------------------- scene dispatch
function drawSide(c, t, shot, P) {
  const L = layout(t), cam = camAt(shot, t, L); CAST.hulk.cast = t >= 32.9 && t < 36.6; const over = riderOver(t);
  c.setTransform(1, 0, 0, 1, 0, 0); c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
  if (cam.dof > 0.8) {
    bgx.setTransform(1, 0, 0, 1, 0, 0); bgx.fillStyle = '#000'; bgx.fillRect(0, 0, W, H); applyCam(bgx, cam); drawBackdrop(bgx, P, t, L.T); drawBigWords(bgx, t); applyCam(bgx, cam); drawRiders(bgx, P, t, L, { lane1: true });
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.filter = `blur(${(cam.dof * cam.z * 0.5).toFixed(1)}px)`; c.drawImage(bgc, 0, 0); c.filter = 'none'; c.restore();
    applyCam(c, cam); drawRiders(c, P, t, L, { lane0: true, over }); drawForeground(c, P, t);
  } else {
    applyCam(c, cam); drawBackdrop(c, P, t, L.T); drawBigWords(c, t); applyCam(c, cam); drawRiders(c, P, t, L, { over }); drawForeground(c, P, t);
  }
  c.setTransform(1, 0, 0, 1, 0, 0);
  sideOverlays(c, t, shot, cam, L);
}
function drawSplit(c, t, shot, P) {
  const L = layout(t), over = riderOver(t), n = shot.who.length, pw = W / n;
  c.setTransform(1, 0, 0, 1, 0, 0); c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
  shot.who.forEach((id, i) => {
    const k = smooth((t - shot.t0) / (shot.t1 - shot.t0)), cam = { z: 1.5 + 0.25 * k, fx: 0, fy: 0, rot: (i - 1) * 0.015, px: -W / 2 + pw * (i + 0.5) + (i - 1) * 40, py: 130, dof: 0 };
    const r = L.byId[id]; cam.fx = r.x + 0.2 * r.s; cam.fy = r.y - 0.9 * r.s;
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.beginPath(); c.rect(pw * i + 3, 0, pw - 6, H); c.clip(); applyCam(c, cam); drawBackdrop(c, P, t, L.T); applyCam(c, cam); drawRiders(c, P, t, L, { over }); c.restore();
  });
}
function speedLines(c, t, k) {
  if (k <= 0.01) return;
  for (let i = 0; i < 30; i++) { const y = H * (0.06 + 0.7 * rnd(i, 61)), len = 220 + 460 * rnd(i, 62), x = W - ((t * (2100 + 1500 * rnd(i, 63)) + rnd(i, 64) * W * 2) % (W * 1.7)) + 100;
    const g = c.createLinearGradient(x, y, x + len, y); g.addColorStop(0, 'rgba(255,255,255,0)'); g.addColorStop(1, `rgba(255,255,255,${0.32 * k})`); c.fillStyle = g; c.fillRect(x, y, len, 2 + 2.5 * rnd(i, 65)); }
}
function sideOverlays(c, t, shot, cam, L) {
  const pr = (id, dx = 0, h = 1.62) => { const r = L.byId[id]; return projCam(cam, r.x + dx * r.s, r.y - h * r.s); };
  TAGS.forEach(([id, name, sub, a, b]) => { if (t >= a && t < b + 0.2) { const p = pr(id, 0.25); tag(c, p[0], p[1] - 4, name, sub, (t - a) / 0.3, (b + 0.2 - t) / 0.2); } });
  if (t >= 83.0 && t < 85.4) { const p = pr('phil', 0.5, 1.9); bubble(c, p[0] + 80, p[1] - 30, 'MAKE IT!!', t - 83.0, '#ffd23f', '#1b1b1b', 44); }
  if (t >= 170 && t < 176.5) { const p = pr('me', 0.5, 1.9); bubble(c, p[0] + 70, p[1] - 40, '?!', t - 170, '#ffffff', '#1b1b1b', 52); }
  if (t >= 186.5 && t < 189.6) { const p = pr('karim', 0.2, 1.9); bubble(c, p[0] - 30, p[1] - 30, 'GO!', t - 186.5, '#e63946', '#ffffff', 46); }
  if (t >= 33.1 && t < 35.0) { const p = pr('hulk', 0.7, 1.8); bubble(c, p[0] + 60, p[1] - 30, 'OW.', t - 33.1, '#ffffff', '#1b1b1b', 42); }
  if (t >= 36.6 && t < 39.5) { const p = pr('hulk', 0.7, 1.8); bubble(c, p[0] + 60, p[1] - 30, 'BACK.', t - 36.6, '#ffd23f', '#1b1b1b', 42); }
  if (t >= 43.2 && t < 46.2) { const g = L.byId.gary, h = L.byId.hulk, a = projCam(cam, g.x + 0.92 * g.s, g.y - 0.34 * g.s), b = projCam(cam, h.x - 0.76 * h.s, h.y - 0.34 * h.s); dimension(c, a[0], a[1], Math.max(b[0], a[0] + 24), b[1], '0 cm', '#ff3b4f', smooth((t - 43.2) / 0.3)); }
  if (t >= 120.9 && t < 124.1) { const a = pr('angelo', 0, 2.1), p = projCam(cam, L.byId.angelo.x, L.byId.angelo.y - 1.9 * L.byId.angelo.s); }
  if (t >= 124.4 && t < 127.1) { const an = L.byId.angelo, nx = L.byId.karim, a = projCam(cam, nx.x + 0.92 * nx.s, nx.y - 0.4 * nx.s), b = projCam(cam, an.x - 0.76 * an.s, an.y - 0.4 * an.s); dimension(c, a[0], a[1] - 90, b[0], b[1] - 90, '1 BIKE LENGTH', '#ff3b4f', smooth((t - 124.4) / 0.3)); }
  if (t >= 127.1 && t < 130.8) NAMED.forEach((id, i) => { const p = pr(id, 0.3, 1.75); bubble(c, p[0] + 40, p[1] - 28, '. . .', t - 127.1 - i * 0.12, '#ffffff', '#1b1b1b', 36); });
  if (t >= 173.3 && t < 176.1) { const seq = ['karim', 'hulk', 'phil', 'gary', 'angelo', 'me']; seq.forEach((id, i) => { const ts = 173.3 + i * 0.46; if (t >= ts && t < ts + 0.7) { const p = pr(id, 0.3, 1.75); bubble(c, p[0] + 30, p[1] - 30, id === 'me' ? 'ME?!' : 'GEEZER?', t - ts, id === 'me' ? '#ff3b4f' : '#ffd23f', id === 'me' ? '#fff' : '#1b1b1b', 38); } }); }
  if (t >= 160 && t < 176.1 && !(t >= 171 && t < 173.2)) { const m = L.byId.me, g = L.byId.gary, a = projCam(cam, m.x + 0.92 * m.s, m.y - 0.34 * m.s), b = projCam(cam, g.x - 0.76 * g.s, g.y - 0.34 * g.s); if (b[0] - a[0] > 10) dimension(c, a[0], a[1], b[0], b[1], 'THE GAP', '#ff3b4f', smooth((t - 160) / 0.4)); }
  if (t >= 117.7 && t < 120.9) { const p = pr('angelo', -0.05, 1.0); drawBackpackIcons(c, t, 117.7, p[0] - 40, p[1] - 150); }
  drawGearsCard(c, t, 114.8, 117.6);
  if (t >= 233.6 && t < 244) { const x = propX(0.55, 233.0, t) + 0, a = smooth((t - 233.6) / 0.6) * (1 - smooth((t - 241) / 2)); }
  // atmosphere
  const sl = Math.max(smooth((t - 160) / 12) * (1 - smooth((t - 198.5) / 2)) * 0.5, shot.shake ? 0.55 : 0) * (t > 150 ? 1 : 0.6);
  speedLines(c, t, (t >= 154 && t < 184.2) ? sl : (t >= 184.2 && t < 198.5 ? 0.9 : (t >= 204.9 && t < 211.3 ? 1 : (t > 85.5 && t < 88.4 ? 0.7 : 0))));
  if (shot.mute) { const k = smooth((t - 127.1) / 0.4) * (1 - smooth((t - 130.4) / 0.4)); c.save(); c.globalCompositeOperation = 'saturation'; c.fillStyle = '#808080'; c.globalAlpha = 0.85 * k; c.fillRect(0, 0, W, H); c.restore(); c.fillStyle = `rgba(10,12,28,${0.28 * k})`; c.fillRect(0, 0, W, H); }
}

function titleSlam(c, t) {
  const age = t - 26.5; if (age < 0 || age > 2.6) return; const k = popOut(age / 0.22), out = smooth((age - 1.6) / 0.8), a = 1 - out;
  c.save(); c.translate(W * 0.44, H * 0.17); c.scale(k * (1 + 0.12 * out), k * (1 + 0.12 * out)); c.rotate(-0.03); c.globalAlpha = a; c.font = '190px "Anton"'; c.letterSpacing = '6px'; const w = textW(c, 'GEEZER GOON');
  outlined(c, 'GEEZER GOON', -w / 2 + 12, 22, 'rgba(0,0,0,0.4)', 'rgba(0,0,0,0)', 0); outlined(c, 'GEEZER GOON', -w / 2, 0, '#ffd23f', 'rgba(10,8,30,0.95)', 22);
  c.font = '44px "Bebas Neue"'; c.letterSpacing = '5px'; const s = 'WEDNESDAY  ·  6:15 PM  ·  KENSINGTON MD'; const w2 = textW(c, s); outlined(c, s, -w2 / 2, 62, '#ffffff', 'rgba(10,8,30,0.95)', 10); c.restore();
}
function endCard(c, t) {
  if (t < 243.2) return; const a = smooth((t - 243.2) / 1.4); c.fillStyle = `rgba(3,3,12,${0.6 * a})`; c.fillRect(0, 0, W, H);
  c.save(); c.globalAlpha = a; c.translate(W / 2, H * 0.4); c.font = '210px "Anton"'; c.letterSpacing = '5px'; let w = textW(c, 'GEEZER GOON'); outlined(c, 'GEEZER GOON', -w / 2, 0, '#ffd23f', 'rgba(10,8,30,0.95)', 18);
  c.font = '64px "Bebas Neue"'; c.letterSpacing = '5px'; const s = 'SAME TIME  ·  SAME CORNER  ·  SAME OLD TOWN'; w = textW(c, s); outlined(c, s, -w / 2, 110, '#ffffff', 'rgba(10,8,30,0.95)', 9);
  c.globalAlpha = smooth((t - 244.6) / 1.0) * a; c.font = '52px "Bebas Neue"'; const s2 = 'NEXT WEDNESDAY  ·  6:15 PM'; w = textW(c, s2); outlined(c, s2, -w / 2, 190, '#2ec4b6', 'rgba(10,8,30,0.95)', 8); c.restore();
  const f = smooth((t - 248.0) / 1.3); if (f > 0) { c.fillStyle = `rgba(0,0,0,${f})`; c.fillRect(0, 0, W, H); }
}
function loopTitle(c, t, shot) {
  const tl = t - shot.t0, k = smooth((tl - 0.6) / 0.6); c.save(); c.globalAlpha = k; c.font = '150px "Anton"'; c.letterSpacing = '4px'; let w = textW(c, 'NEXT WEDNESDAY');
  outlined(c, 'NEXT WEDNESDAY', W * 0.04, H * 0.33, '#ffd23f', 'rgba(5,8,18,0.95)', 16); c.font = '70px "Bebas Neue"'; w = textW(c, '6:15 PM  ·  SAME ROUTE');
  outlined(c, '6:15 PM  ·  SAME ROUTE', W * 0.04, H * 0.33 + 84, '#ffffff', 'rgba(5,8,18,0.95)', 9); c.restore();
}

function frame(t) {
  const shot = shotAt(t), P = new Pal(t), c = ctx; let lyr = { y: 0.30 };
  c.setTransform(1, 0, 0, 1, 0, 0); c.globalAlpha = 1; c.globalCompositeOperation = 'source-over';
  switch (shot.type) {
    case 'side': drawSide(c, t, shot, P); break;
    case 'split': drawSplit(c, t, shot, P); break;
    case 'topdown': sceneTopDown(c, t, shot); lyr = { y: 0.32 }; break;
    case 'map': sceneMap(c, t, shot); break;
    case 'computer': sceneComputer(c, t, shot); break;
    case 'calendar': sceneCalendar(c, t, shot); break;
    case 'pothole': scenePothole(c, t, shot); break;
    case 'face': sceneFace(c, t, shot); break;
  }
  if (shot.lyr) lyr = shot.lyr;
  if (t > 26.4 && t < 29.2) lyr = { y: 0.38 };
  c.setTransform(1, 0, 0, 1, 0, 0);
  if (shot.type === 'side' || shot.type === 'topdown' || shot.type === 'split') { STICKERS.forEach(s => sticker(c, t, s)); }
  if (shot.type === 'map' && shot.title) loopTitle(c, t, shot);
  if (t >= 9.6 && t < 19) { const n = Math.round(38 * smooth((t - 9.6) / 5.2)); const k = popOut((t - 9.6) / 0.3) * (1 - smooth((t - 18.2) / 0.6)); if (k > 0.02) { c.save(); c.translate(W * 0.2, H * 0.17); c.scale(k, k); c.rotate(-0.04); c.font = '90px "Anton"'; const s = `${n} / 38 RIDERS`; const w = textW(c, s) + 60; rrect(c, -w / 2, -80, w, 120, 14); c.fillStyle = '#e63946'; c.fill(); c.fillStyle = '#fff'; c.fillText(s, -w / 2 + 30, 18); c.restore(); } }
  titleSlam(c, t);
  drawLyrics(c, t, lyr);
  if (shot.type !== 'computer' || t > 9.5) drawHUD(c, t, { hide: shot.hud === false || (t > 26.2 && t < 29.0) });
  let fl = 0; FLASHES.forEach(ft => { if (t >= ft && t - ft < 0.3) fl = Math.max(fl, Math.exp(-(t - ft) / 0.1)); }); if (fl > 0.02) { c.fillStyle = `rgba(255,255,255,${0.5 * fl})`; c.fillRect(0, 0, W, H); }
  post(c, t, { bloom: shot.type === 'topdown' ? 0.1 : undefined });
  endCard(c, t);
  const f0 = 1 - smooth(t / 0.8); if (f0 > 0) { c.fillStyle = `rgba(0,0,0,${f0})`; c.fillRect(0, 0, W, H); }
}
window.renderFrame = t => { frame(t); return true; };
window.ready = document.fonts.ready.then(() => Promise.all(['Anton', 'Bebas Neue', 'Permanent Marker'].map(f => document.fonts.load(`40px "${f}"`)))).then(() => true);
