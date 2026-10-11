// Film layer over the Second Sky app. The page's draw() renders stars, Milky Way and every figure dimly;
// this file adds the camera choreography, the cord, the inked figures, and the text, all as a pure
// function of time t so frames can be rendered in any order.
/* global SKY, TL, state, proj, draw, ctx, W, H, base, d3, consGeo, byId, DPR */
(function () {
  const ease = d3.easeCubicInOut, clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const geo = new Map(consGeo.map(g => [g.c.id, g]));
  state.drift = false; state.layers.names = false;   // the app's idle drift would redraw between frames
  const stops = TL.stops;                           // [{cons, t0 fly start, t1 arrive, t2 leave, lines:[{t,dur,text}]}]
  const FIT = 0.66;                                 // figure spans this share of the frame height
  const kFit = id => clamp((FIT * H / 2) / (base * 2 * Math.tan(geo.get(id).radius / 2)), 1.1, 8);
  const START = TL.start;                           // [lon, lat] the film opens on
  const center = id => geo.get(id).c.center;

  function camera(t) {
    if (t < stops[0].t0) {                           // prologue: wide, slow turn
      const u = t / stops[0].t0;
      return { c: [START[0] + 14 * u, START[1]], k: 0.72 + 0.06 * u };
    }
    for (let j = 0; j < stops.length; j++) {
      const s = stops[j], prev = j ? camera(s.t0 - 1e-3) : camera(stops[0].t0 - 1e-3);
      if (t < s.t1) {                                // flying in
        const u = ease(clamp((t - s.t0) / (s.t1 - s.t0)));
        const to = center(s.cons), k1 = kFit(s.cons);
        const d = d3.geoDistance(prev.c, to), dip = clamp(d * 0.55, 0, 0.6);
        return { c: d3.geoInterpolate(prev.c, to)(u), k: (prev.k + (k1 - prev.k) * u) * (1 - dip * Math.sin(Math.PI * u)), fly: u };
      }
      if (j === stops.length - 1 || t < stops[j + 1].t0) {   // holding: slow push-in and drift
        const end = j === stops.length - 1 ? TL.outro : stops[j + 1].t0;
        const u = clamp((t - s.t1) / Math.max(1, end - s.t1));
        const c = center(s.cons);
        if (j === stops.length - 1 && t > TL.outro) {         // the pull-back at the end
          const v = ease(clamp((t - TL.outro) / 7));
          const hold = { c: [c[0] + 0.4, c[1]], k: kFit(s.cons) * 1.08 };
          return { c: d3.geoInterpolate(hold.c, TL.endCenter)(v), k: hold.k + (0.62 - hold.k) * v };
        }
        return { c: [c[0] + 0.4 * u, c[1]], k: kFit(s.cons) * (1 + 0.08 * u) };
      }
    }
  }

  // Draw the first fraction t of a set of screen polylines, measured by length.
  function partial(lines, t, close) {
    const segs = []; let total = 0;
    for (const pl of lines) for (let i = 1; i < pl.length; i++) {
      const a = pl[i - 1], b = pl[i]; if (!a || !b) continue;
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]); segs.push([a, b, len]); total += len;
    }
    let left = total * t;
    ctx.beginPath();
    for (const [a, b, len] of segs) {
      if (left <= 0) break;
      const f = Math.min(1, left / len);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
      left -= len;
    }
    ctx.stroke();
  }
  const screen = lines => lines.map(pl => pl.map(p => proj(p)));

  function inked(id, prog, alpha) {
    const c = byId.get(id);
    ctx.save(); ctx.lineJoin = ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(143,180,220,0.6)'; ctx.shadowBlur = 6 * alpha;
    ctx.strokeStyle = `rgba(160,195,232,${0.95 * alpha})`; ctx.lineWidth = 1.7;
    partial(screen(c.figure), clamp(prog * 1.25 - 0.25), false);
    ctx.shadowColor = 'rgba(216,179,106,0.7)'; ctx.shadowBlur = 5 * alpha;
    ctx.strokeStyle = `rgba(232,196,118,${alpha})`; ctx.lineWidth = 2;
    partial(screen(c.lines.map(l => [[l[0], l[1]], [l[2], l[3]]])), clamp(prog * 1.6), false);
    ctx.restore();
    if (prog > 0.6) {                                  // the named stars
      ctx.save(); ctx.globalAlpha = alpha * clamp((prog - 0.6) / 0.3);
      ctx.font = '15px "IM Fell English SC", Georgia, serif'; ctx.fillStyle = '#e8c476'; ctx.strokeStyle = '#e8c476';
      for (const s of c.stars) { const p = proj(s.p); if (!p) continue;
        ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 7); ctx.lineWidth = 1; ctx.stroke(); ctx.fillText(s.name, p[0] + 10, p[1] + 15); }
      ctx.restore();
    }
  }

  function cord(t, cam) {
    const pts = [];
    for (let j = 0; j < stops.length; j++) {
      const s = stops[j];
      if (t < s.t0) break;
      if (t < s.t1) { pts.push(cam.c); break; }        // paying out behind the camera
      pts.push(center(s.cons));
    }
    if (!pts.length) return;
    const from = [TL.anchor].concat(pts);               // the string starts at the pole, where Hesk sat
    let a = 1;
    if (t > TL.cut) a = clamp(1 - (t - TL.cut) / 2.5);  // the cut: the cord flares, then lets go
    const flare = t > TL.cut - 0.4 && t < TL.cut + 0.6 ? 1 : 0;
    if (a <= 0) return;
    ctx.save(); ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(232,196,118,0.8)'; ctx.shadowBlur = 8 + 14 * flare;
    ctx.strokeStyle = `rgba(232,196,118,${0.55 * a + 0.4 * flare})`; ctx.lineWidth = 1.3 + flare;
    ctx.setLineDash([1, 5]);
    ctx.beginPath(); d3.geoPath(proj, ctx)({ type: 'LineString', coordinates: from }); ctx.stroke();
    ctx.restore();
  }

  function text(str, x, y, font, color, align = 'center', alpha = 1) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha = alpha; ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
    ctx.shadowColor = 'rgba(5,8,18,0.95)'; ctx.shadowBlur = 12; ctx.fillText(str, x, y); ctx.restore();
  }
  function wrap(str, font, maxW) {
    ctx.save(); ctx.font = font; const words = str.split(' '), out = []; let line = '';
    for (const w of words) { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
    out.push(line); ctx.restore(); return out;
  }
  const fade = (t, a, b, f = 0.6) => clamp(Math.min((t - a) / f, (b - t) / f));

  function caption(t) {
    const all = [TL.intro.lines, ...stops.map(s => s.lines)].flat();
    const l = all.find(l => t >= l.t - 0.15 && t < l.t + l.dur + 0.35);
    if (!l) return;
    const a = fade(t, l.t - 0.15, l.t + l.dur + 0.35, 0.25);
    const font = 'italic 27px "IM Fell English", Georgia, serif';
    const rows = wrap(l.text, font, Math.min(980, W - 80));
    const y0 = H - 46 - (rows.length - 1) * 36;
    ctx.save(); ctx.globalAlpha = a * 0.55;
    const g = ctx.createLinearGradient(0, y0 - 60, 0, H); g.addColorStop(0, 'rgba(5,8,18,0)'); g.addColorStop(1, 'rgba(5,8,18,0.9)');
    ctx.fillStyle = g; ctx.fillRect(0, y0 - 60, W, H - y0 + 60); ctx.restore();
    rows.forEach((r, i) => text(r, W / 2, y0 + i * 36, font, '#e6e9f1', 'center', a));
  }

  function titleCard(t) {
    stops.forEach((s, j) => {
      const end = j === stops.length - 1 ? TL.outro : stops[j + 1].t0;
      const a = fade(t, s.t1 - 0.4, end, 0.8);
      if (a <= 0) return;
      const c = byId.get(s.cons);
      text(`${j + 1} of ${stops.length}`, 44, 62, '13px "IBM Plex Mono", monospace', '#8d97b3', 'left', a);
      text(c.name, 42, 112, '50px "IM Fell English SC", Georgia, serif', '#e8c476', 'left', a);
      text(c.gloss, 44, 146, 'italic 25px "IM Fell English", Georgia, serif', '#a9c6e6', 'left', a);
    });
  }

  window.filmFrame = function (t) {
    const cam = camera(t);
    state.rot = [-cam.c[0], -cam.c[1], 0]; state.k = cam.k; state.sel = null;
    draw();
    // figures already told stay lit; the current one inks itself in
    stops.forEach((s, j) => {
      if (t < s.t1 - 0.2) return;
      const next = j === stops.length - 1 ? Infinity : stops[j + 1].t0;
      if (t < next) inked(s.cons, clamp((t - s.t1 + 0.2) / 3.2), 1);
      else inked(s.cons, 1, 0.55);
    });
    cord(t, cam);
    // opening title
    const it = fade(t, 1.0, TL.intro.lines[0].t + 2.5, 1.2);
    text(TL.title, W / 2, H * 0.42, '64px "IM Fell English SC", Georgia, serif', '#e8e4da', 'center', it);
    text('as Aun told it on the north cliff, in his last winter', W / 2, H * 0.42 + 46, 'italic 24px "IM Fell English", Georgia, serif', '#a9c6e6', 'center', it);
    titleCard(t);
    caption(t);
    // closing card
    const ec = fade(t, TL.cut + 2.2, TL.duration + 5, 1.5);
    text(TL.title, W / 2, H * 0.44, '58px "IM Fell English SC", Georgia, serif', '#e8e4da', 'center', ec);
    text('The constellations of the Ilvari, a people who never lived, on the real stars', W / 2, H * 0.44 + 44, 'italic 22px "IM Fell English", Georgia, serif', '#a9c6e6', 'center', ec);
    text('THE SECOND SKY  ·  muninn.austegard.com/scratch/second-sky', W / 2, H * 0.44 + 90, '14px "IBM Plex Mono", monospace', '#8d97b3', 'center', ec);
    // fade from and to black
    const black = Math.max(clamp(1 - t / 1.2), clamp((t - (TL.duration - 1.5)) / 1.5));
    if (black > 0) { ctx.save(); ctx.fillStyle = `rgba(0,0,0,${black})`; ctx.fillRect(0, 0, W, H); ctx.restore(); }
  };
  window.filmInfo = () => Object.fromEntries(consGeo.map(g => [g.c.id, { center: g.c.center, radius: g.radius }]));
})();
