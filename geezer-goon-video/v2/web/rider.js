'use strict';
// Side-view cyclist rig. Units are metres, origin = ground below the bottom bracket, +x forward, y down.
const CAST = {
  me:     { id: 'me', name: 'ME', jersey: '#1d6fb8', acc: '#ffd23f', skin: '#e8b894', helm: '#f4f4f4', bike: 'gravel', frame: '#6b7f3a', build: 1.04, hair: '#5a3b24', stache: true },
  angelo: { id: 'angelo', name: 'ANGELO', jersey: '#2a9d8f', acc: '#e9c46a', skin: '#c98f63', helm: '#262626', bike: 'single', frame: '#cfd4dc', build: 0.96, hair: '#1b1b1b', pack: true },
  gary:   { id: 'gary', name: 'GARY', jersey: '#f4a261', acc: '#264653', skin: '#f0c4a0', helm: '#e63946', bike: 'road', frame: '#f4a261', build: 1.0, hair: '#8a8a8a', stache: true, upright: 1 },
  phil:   { id: 'phil', name: 'PHIL', jersey: '#e63946', acc: '#ffffff', skin: '#e0ac86', helm: '#ffffff', bike: 'road', frame: '#e63946', build: 1.05, hair: '#3a2a1a' },
  hulk:   { id: 'hulk', name: 'THE HULK', jersey: '#3d3d4a', acc: '#ffb000', skin: '#d9a07a', helm: '#ffb000', bike: 'road', frame: '#2b2b33', build: 1.22, hair: '#1b1b1b', letter: 'H', big: 1 },
  karim:  { id: 'karim', name: 'KARIM', jersey: '#ffd23f', acc: '#1b1b1b', skin: '#b98056', helm: '#1b1b1b', bike: 'road', frame: '#ffd23f', build: 1.0, hair: '#1b1b1b' },
};
const FAR_J = ['#8e44ad', '#16a085', '#d35400', '#2c3e50', '#c0392b', '#f1c40f', '#27ae60', '#e84393', '#0984e3', '#7f8c8d', '#e67e22', '#2980b9'];
const FARS = FAR_J.map((j, i) => ({ id: 'far' + i, jersey: j, acc: '#ffffff', skin: ['#e8b894', '#c98f63', '#8d5a3b', '#f0c4a0'][i % 4],
  helm: ['#ffffff', '#222222', '#e63946', '#ffd23f'][i % 4], bike: 'road', frame: j, build: 0.94 + 0.12 * rnd(i, 3), hair: '#3a2a1a' }));
const NAMED = ['me', 'angelo', 'gary', 'phil', 'hulk', 'karim'];

function limb(ctx, ax, ay, bx, by, r1, r2, fill) {
  const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1e-6, nx = -dy / L, ny = dx / L;
  ctx.fillStyle = fill; ctx.beginPath();
  ctx.moveTo(ax + nx * r1, ay + ny * r1); ctx.lineTo(bx + nx * r2, by + ny * r2); ctx.lineTo(bx - nx * r2, by - ny * r2); ctx.lineTo(ax - nx * r1, ay - ny * r1); ctx.closePath();
  ctx.moveTo(ax + r1, ay); ctx.arc(ax, ay, r1, 0, TAU); ctx.moveTo(bx + r2, by); ctx.arc(bx, by, r2, 0, TAU); ctx.fill();
}
function tube(ctx, pts, w, col) {
  ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke();
}
function wheel(ctx, cx, cy, r, tyre, ang, spin, C, detail) {
  ctx.strokeStyle = C('#101013'); ctx.lineWidth = tyre; ctx.beginPath(); ctx.arc(cx, cy, r - tyre / 2, 0, TAU); ctx.stroke();
  if (!detail) return;
  ctx.strokeStyle = C('#b8bcc6'); ctx.lineWidth = 0.012; ctx.beginPath(); ctx.arc(cx, cy, r - tyre - 0.012, 0, TAU); ctx.stroke();
  const fast = clamp((spin - 7) / 7);
  if (fast < 1) {
    ctx.strokeStyle = C('#d0d4dc'); ctx.globalAlpha = 0.6 * (1 - fast); ctx.lineWidth = 0.006; ctx.beginPath();
    for (let n = 0; n < 12; n++) { const a = ang + n * TAU / 12; ctx.moveTo(cx, cy); ctx.lineTo(cx + (r - tyre) * Math.cos(a), cy + (r - tyre) * Math.sin(a)); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (fast > 0) { ctx.globalAlpha = 0.16 * fast; ctx.fillStyle = C('#cfd4de'); ctx.beginPath(); ctx.arc(cx, cy, r - tyre, 0, TAU); ctx.fill(); ctx.globalAlpha = 1; }
  ctx.fillStyle = C('#8a8f98'); ctx.beginPath(); ctx.arc(cx, cy, 0.026, 0, TAU); ctx.fill();
}

function drawHead(ctx, R, hx, hy, o, C) {
  const eyes = o.eyes || 'shades', mood = o.mood || 'neutral', st = o.stress || 0;
  const skin = C(R.skin);
  ctx.fillStyle = skin; ctx.beginPath(); ctx.arc(hx, hy, 0.095, 0, TAU); ctx.fill();
  ctx.beginPath(); ctx.arc(hx + 0.05, hy + 0.045, 0.052, 0, TAU); ctx.fill();                  // jaw
  ctx.beginPath(); ctx.moveTo(hx + 0.078, hy - 0.03); ctx.lineTo(hx + 0.128, hy + 0.012); ctx.lineTo(hx + 0.08, hy + 0.024); ctx.closePath(); ctx.fill(); // nose
  ctx.fillStyle = C(R.skin); ctx.globalAlpha = o.flat ? 1 : 0.7; ctx.beginPath(); ctx.arc(hx - 0.03, hy + 0.012, 0.022, 0, TAU); ctx.fill(); ctx.globalAlpha = 1;
  if (R.beard) { ctx.fillStyle = C(R.hair); ctx.beginPath(); ctx.arc(hx + 0.045, hy + 0.05, 0.056, 0.15, Math.PI - 0.1); ctx.lineTo(hx - 0.02, hy + 0.02); ctx.closePath(); ctx.fill(); }
  if (R.stache) { ctx.fillStyle = C(R.hair === '#8a8a8a' ? '#9a9a9a' : R.hair); ctx.beginPath(); ctx.moveTo(hx + 0.06, hy + 0.034); ctx.lineTo(hx + 0.115, hy + 0.036); ctx.lineTo(hx + 0.108, hy + 0.05); ctx.lineTo(hx + 0.06, hy + 0.05); ctx.fill(); }
  if (!o.flat) {
    if (eyes === 'shades') {
      ctx.fillStyle = 'rgba(8,10,16,0.95)'; rrect(ctx, hx + 0.012, hy - 0.048, 0.1, 0.042, 0.016); ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 0.006; ctx.beginPath(); ctx.moveTo(hx + 0.03, hy - 0.04); ctx.lineTo(hx + 0.055, hy - 0.04); ctx.stroke();
    } else if (eyes === 'wide') {
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(hx + 0.058, hy - 0.026, 0.034, 0, TAU); ctx.fill();
      ctx.strokeStyle = '#222'; ctx.lineWidth = 0.006; ctx.stroke();
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(hx + 0.07, hy - 0.026 + (o.look || 0), 0.013, 0, TAU); ctx.fill();
      ctx.strokeStyle = '#2a1b10'; ctx.lineWidth = 0.011; ctx.beginPath(); ctx.moveTo(hx + 0.02, hy - 0.08); ctx.lineTo(hx + 0.09, hy - 0.07); ctx.stroke();
    } else if (eyes === 'squint') {
      ctx.strokeStyle = '#2a1b10'; ctx.lineWidth = 0.01; ctx.beginPath(); ctx.moveTo(hx + 0.035, hy - 0.022); ctx.lineTo(hx + 0.085, hy - 0.03); ctx.stroke();
    } else {
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(hx + 0.06, hy - 0.024, 0.017, 0, TAU); ctx.fill();
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(hx + 0.066, hy - 0.024, 0.009, 0, TAU); ctx.fill();
    }
    ctx.strokeStyle = 'rgba(60,20,20,0.85)'; ctx.lineWidth = 0.008; ctx.lineCap = 'round'; ctx.beginPath();
    if (mood === 'grit') { ctx.moveTo(hx + 0.05, hy + 0.06); ctx.lineTo(hx + 0.1, hy + 0.055); ctx.stroke(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.014; ctx.beginPath(); ctx.moveTo(hx + 0.056, hy + 0.06); ctx.lineTo(hx + 0.096, hy + 0.056); ctx.stroke(); }
    else if (mood === 'o') { ctx.fillStyle = 'rgba(70,15,15,0.9)'; ctx.beginPath(); ctx.ellipse(hx + 0.082, hy + 0.062, 0.016, 0.022, 0, 0, TAU); ctx.fill(); }
    else { ctx.moveTo(hx + 0.055, hy + 0.062); ctx.lineTo(hx + 0.095, hy + 0.058 + (mood === 'smirk' ? -0.012 : 0)); ctx.stroke(); }
  }
  // helmet
  ctx.fillStyle = C(R.helm); ctx.beginPath(); ctx.moveTo(hx - 0.118, hy + 0.012);
  ctx.bezierCurveTo(hx - 0.122, hy - 0.17, hx + 0.075, hy - 0.185, hx + 0.125, hy - 0.04); ctx.lineTo(hx + 0.105, hy - 0.052);
  ctx.bezierCurveTo(hx + 0.04, hy - 0.07, hx - 0.06, hy - 0.05, hx - 0.095, hy + 0.012); ctx.closePath(); ctx.fill();
  if (!o.flat) {
    ctx.strokeStyle = 'rgba(0,0,0,0.28)'; ctx.lineWidth = 0.009; ctx.beginPath();
    for (let k = 0; k < 3; k++) { const a = -2.5 + k * 0.52; ctx.moveTo(hx + 0.02 * Math.cos(a) , hy - 0.03 + 0.02 * Math.sin(a)); ctx.lineTo(hx + 0.13 * Math.cos(a), hy - 0.03 + 0.15 * Math.sin(a) * 0.95); }
    ctx.stroke();
    ctx.strokeStyle = 'rgba(20,20,20,0.7)'; ctx.lineWidth = 0.007; ctx.beginPath(); ctx.moveTo(hx - 0.01, hy - 0.02); ctx.lineTo(hx + 0.03, hy + 0.062); ctx.stroke(); // strap
    if (o.rim) { const rc = o.rimCol || [255, 214, 150]; ctx.strokeStyle = `rgba(${rc[0]},${rc[1]},${rc[2]},${0.55 * o.rim})`; ctx.lineWidth = 0.012; ctx.beginPath(); ctx.arc(hx + 0.004, hy - 0.03, 0.117, -1.1, -0.1); ctx.stroke(); }
  }
  if (st > 0 && !o.flat) { // sweat drops
    ctx.fillStyle = 'rgba(190,225,255,0.9)';
    for (let k = 0; k < 3; k++) { const p = (o.t * 1.7 + k * 0.33) % 1; ctx.beginPath(); ctx.ellipse(hx - 0.06 - 0.12 * p + k * 0.02, hy - 0.1 + 0.2 * p * p, 0.011, 0.018, 0.5, 0, TAU); ctx.globalAlpha = st * (1 - p); ctx.fill(); }
    ctx.globalAlpha = 1;
  }
}

function drawRider(ctx, R, o) {
  const P = o.P, crank = o.crank || 0, wang = o.wang || 0, wspin = o.wspin || 0, stand = o.stand || 0, depth = o.depth || 0;
  const flat = o.flat || null, detail = !flat, g = R.build || 1;
  const C = c => flat || P.s(c, depth);
  const wr = 0.34, ay = -wr, tyre = R.bike === 'gravel' ? 0.075 : 0.042;
  const RA = [-0.42, ay], FA = [0.58, ay], BB = [0, ay + 0.07];
  const ST = [-0.17, ay - 0.55], HT = [0.47, ay - 0.52], HB = [0.51, ay - 0.36];
  const bob = 0.011 * Math.sin(crank * 2) * (1 - 0.4 * stand), up = R.upright || 0;
  const hip = [-0.19 + 0.12 * stand, ay - 0.62 - 0.18 * stand + bob];
  const sh = [0.27 - 0.07 * up + 0.02 * stand, ay - 0.98 - 0.07 * up - 0.03 * stand + bob * 0.5];
  const hand = [0.53, ay - 0.50];
  const jer = C(R.jersey), acc = C(R.acc), skin = C(R.skin), shorts = flat || P.s('#17171c', depth);
  const frame = C(R.frame || R.jersey), CR = 0.17;
  ctx.save();
  if (stand) { ctx.translate(0, 0); ctx.rotate(0.035 * stand * Math.sin(crank)); }
  // far leg
  const fa = crank + Math.PI, fpx = BB[0] + CR * Math.cos(fa), fpy = BB[1] + CR * Math.sin(fa);
  const fk = ik(hip[0], hip[1], fpx, fpy, 0.45, 0.46, -1), farShade = flat || P.s('#0e0e12', depth);
  limb(ctx, hip[0], hip[1], fk[0], fk[1], 0.075 * g, 0.052, farShade); limb(ctx, fk[0], fk[1], fpx, fpy, 0.052, 0.034, farShade);
  limb(ctx, fpx - 0.06, fpy + 0.005, fpx + 0.1, fpy + 0.005, 0.03, 0.026, farShade);
  // wheels
  wheel(ctx, RA[0], RA[1], wr, tyre, wang, wspin, C, detail); wheel(ctx, FA[0], FA[1], wr, tyre, wang, wspin, C, detail);
  // frame
  const fw = R.bike === 'gravel' ? 0.04 : 0.032;
  tube(ctx, [RA, BB, ST, RA], fw, frame); tube(ctx, [ST, HT, HB, BB], fw, frame);
  tube(ctx, [HB, FA], 0.028, C('#2a2a2a'));
  tube(ctx, [[HT[0] + 0.00, HT[1]], [HT[0] + 0.02, HT[1] - 0.05], [0.57, ay - 0.57]], 0.025, C('#2a2a2a'));
  tube(ctx, [[0.57, ay - 0.57], [0.64 + (R.bike === 'gravel' ? 0.02 : 0), ay - 0.52], [0.61, ay - 0.44], [0.55, ay - 0.45]], 0.025, C('#1c1c1c'));
  tube(ctx, [[ST[0] + 0.02, ST[1]], [ST[0] - 0.02 + 0.0, ST[1] - 0.09 + 0.03 * stand]], 0.028, C('#2a2a2a'));
  tube(ctx, [[ST[0] - 0.13, ST[1] - 0.095 + 0.03 * stand], [ST[0] + 0.07, ST[1] - 0.09 + 0.03 * stand]], 0.042, C('#101010'));
  if (detail) {
    ctx.strokeStyle = C('#9aa0aa'); ctx.lineWidth = 0.012; ctx.beginPath(); ctx.arc(BB[0], BB[1], 0.095, 0, TAU); ctx.stroke();
    ctx.strokeStyle = C('#4a4e56'); ctx.lineWidth = 0.007; ctx.beginPath(); ctx.moveTo(BB[0], BB[1] - 0.095); ctx.lineTo(RA[0], RA[1] - 0.03); ctx.moveTo(BB[0], BB[1] + 0.095); ctx.lineTo(RA[0], RA[1] + 0.03); ctx.stroke();
    ctx.fillStyle = C('#d8dde6'); ctx.beginPath(); ctx.arc(RA[0], RA[1], R.bike === 'single' ? 0.04 : 0.055, 0, TAU); ctx.fill();
    if (R.bike !== 'single') { ctx.strokeStyle = C('#3a3d44'); ctx.lineWidth = 0.014; ctx.beginPath(); ctx.moveTo(RA[0] + 0.01, RA[1] + 0.03); ctx.lineTo(RA[0] - 0.04, RA[1] + 0.12); ctx.stroke(); }
    ctx.save(); ctx.translate(0.2, ay - 0.2); ctx.rotate(-0.55); ctx.fillStyle = C('#cfd4dc'); rrect(ctx, -0.03, -0.065, 0.06, 0.13, 0.025); ctx.fill(); ctx.restore();  // bottle
  }
  // near leg
  const px = BB[0] + CR * Math.cos(crank), py = BB[1] + CR * Math.sin(crank);
  tube(ctx, [BB, [px, py]], 0.03, C('#555a60'));
  const nk = ik(hip[0], hip[1], px, py, 0.45, 0.46, -1);
  limb(ctx, hip[0], hip[1], nk[0], nk[1], 0.085 * g, 0.058, shorts); limb(ctx, nk[0], nk[1], px, py, 0.056, 0.036, skin);
  limb(ctx, nk[0], nk[1], lerp(nk[0], px, 0.28), lerp(nk[1], py, 0.28), 0.058, 0.052, skin);
  limb(ctx, lerp(nk[0], px, 0.72), lerp(nk[1], py, 0.72), px, py, 0.042, 0.036, C('#f4f4f4'));            // sock
  limb(ctx, px - 0.07, py + 0.004, px + 0.105, py + 0.002, 0.034, 0.027, C('#f6f6f6'));                      // shoe
  // torso
  const tg = flat ? flat : (() => { const gr = ctx.createLinearGradient(0, sh[1] - 0.08, 0, hip[1] + 0.08); gr.addColorStop(0, P.s(R.jersey, depth)); gr.addColorStop(1, rgba(scale3(P.c(R.jersey, depth), 0.72), 1)); return gr; })();
  ctx.fillStyle = tg; ctx.beginPath(); ctx.moveTo(hip[0] - 0.08, hip[1] + 0.05);
  ctx.bezierCurveTo(hip[0] - 0.05, hip[1] - 0.25 * g, sh[0] - 0.25, sh[1] - 0.05 - 0.02 * (g - 1), sh[0] + 0.02, sh[1] - 0.06 - 0.03 * (g - 1));
  ctx.lineTo(sh[0] + 0.07, sh[1] + 0.08); ctx.bezierCurveTo(sh[0] - 0.15, sh[1] + 0.12, hip[0] + 0.12 + 0.03 * (g - 1), hip[1] - 0.05, hip[0] + 0.10 + 0.04 * (g - 1), hip[1] + 0.06); ctx.closePath(); ctx.fill();
  if (detail) {
    tube(ctx, [[hip[0] - 0.02, hip[1] - 0.12], [sh[0] - 0.02, sh[1]]], 0.034, acc);
    if (R.letter) { ctx.save(); ctx.translate(0.02 + 0.06 * stand, ay - 0.80 - 0.1 * stand); ctx.rotate(-0.55); ctx.fillStyle = acc; ctx.font = '0.17px "Anton"'; ctx.fillText(R.letter, -0.04, 0.06); ctx.restore(); }
    if (o.rim) { const rc = o.rimCol || [255, 214, 150]; ctx.strokeStyle = `rgba(${rc[0]},${rc[1]},${rc[2]},${0.5 * o.rim})`; ctx.lineWidth = 0.014; ctx.beginPath(); ctx.moveTo(hip[0] - 0.08, hip[1] + 0.02); ctx.bezierCurveTo(hip[0] - 0.05, hip[1] - 0.25 * g, sh[0] - 0.25, sh[1] - 0.05, sh[0] + 0.02, sh[1] - 0.06); ctx.stroke(); }
  }
  if (R.pack && detail) {
    ctx.save(); ctx.translate(-0.02 + 0.03 * stand, ay - 0.86 - 0.12 * stand); ctx.rotate(-0.62);
    ctx.fillStyle = P.s('#3b3b44', depth); rrect(ctx, -0.18, -0.15, 0.36, 0.22, 0.06); ctx.fill();
    ctx.fillStyle = P.s('#5a5a66', depth); rrect(ctx, -0.11, -0.13, 0.22, 0.05, 0.02); ctx.fill();
    ctx.fillStyle = P.s('#e9c46a', depth); rrect(ctx, -0.14, -0.02, 0.12, 0.07, 0.015); ctx.fill();     // laptop sticker
    ctx.strokeStyle = P.s('#e63946', depth); ctx.lineWidth = 0.012; ctx.beginPath(); ctx.moveTo(0.1, -0.15); ctx.lineTo(0.16, -0.26); ctx.stroke(); // badge lanyard
    ctx.restore();
  }
  // arm
  const el = ik(sh[0], sh[1], hand[0], hand[1], 0.30, 0.30, 1);
  limb(ctx, sh[0], sh[1], el[0], el[1], 0.052 * Math.sqrt(g), 0.042, jer); limb(ctx, el[0], el[1], hand[0], hand[1], 0.04, 0.03, skin);
  if (R.cast && detail) { limb(ctx, lerp(el[0], hand[0], 0.1), lerp(el[1], hand[1], 0.1), lerp(el[0], hand[0], 0.8), lerp(el[1], hand[1], 0.8), 0.052, 0.05, C('#f2f2ee')); }
  ctx.fillStyle = skin; ctx.beginPath(); ctx.arc(hand[0] + 0.01, hand[1] + 0.005, 0.03, 0, TAU); ctx.fill();
  // head
  drawHead(ctx, R, sh[0] + 0.14 - 0.04 * up, sh[1] - 0.105 - 0.01 * up, Object.assign({}, o, { flat }), C);
  ctx.restore();
  // lights (night)
  if (o.lights > 0 && !flat) {
    const L = o.lights, blink = (Math.floor(o.t * 3 + (o.seed || 0) * 7) % 2 === 0) ? 1 : 0.3;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    glow(ctx, ST[0] - 0.03, ST[1] + 0.07, 0.2, [255, 50, 40], 0.5 * L * blink);
    ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = `rgba(255,70,70,${L * blink})`; ctx.beginPath(); ctx.arc(ST[0] - 0.03, ST[1] + 0.07, 0.03, 0, TAU); ctx.fill();
    ctx.globalCompositeOperation = 'lighter';
    const fx = 0.6, fy = ay - 0.55, gr = ctx.createLinearGradient(fx, fy, fx + 2.2, fy);
    gr.addColorStop(0, `rgba(255,248,225,${0.2 * L})`); gr.addColorStop(1, 'rgba(255,248,225,0)');
    ctx.fillStyle = gr; ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(fx + 2.2, fy - 0.05); ctx.lineTo(fx + 2.2, fy + 0.55); ctx.closePath(); ctx.fill();
    glow(ctx, fx, fy, 0.16, [255, 255, 235], 0.8 * L);
    ctx.restore();
  }
}
