// Self-painting SVG: any image traced into flat-color regions, then given a
// timeline (sketch → paint → night → dawn → day → fall → erase) compiled to CSS
// keyframes. Browser port of ../build.py and ../timeline.py; the tracer
// internals (blur, kmeans*, majority, absorbSmall, components, traceLoops,
// isJunction, simplifyLoop, curvedPathData) come from svg-in-browser/web/trace.js.

// ---------- timeline compiler (timeline.py) ----------

const EASE = {
  linear: "linear", in: "cubic-bezier(.55,0,1,.45)", out: "cubic-bezier(.23,1,.32,1)",
  inOut: "cubic-bezier(.77,0,.175,1)", sine: "cubic-bezier(.37,0,.63,1)",
};

const n3 = (v) => {
  if (typeof v === "string") return v;
  let t = (Math.round(v * 1000) / 1000).toString();
  if (t.startsWith("0.")) t = t.slice(1);
  else if (t.startsWith("-0.")) t = "-" + t.slice(2);
  return t === "-0" ? "0" : t;
};

/** States on an absolute clock; each segment's easing sits on its first state. */
class Motion {
  constructor(start) { this.points = [[0, { ...start }, null]]; }
  get now() { return this.points[this.points.length - 1][0]; }
  at(t) {
    if (t < this.now - 1e-9) throw new Error(`at(${t}) goes back before ${this.now}`);
    if (t > this.now + 1e-9) this.points.push([t, { ...this.points[this.points.length - 1][1] }, null]);
    return this;
  }
  to(dur, ease, values) {
    const last = this.points[this.points.length - 1];
    last[2] = ease;
    this.points.push([this.now + dur, { ...last[1], ...values }, null]);
    return this;
  }
  keyframes(duration) {
    const pts = this.points.map((p) => [...p]);
    if (pts[pts.length - 1][0] < duration - 1e-9) pts.push([duration, { ...pts[pts.length - 1][1] }, null]);
    const groups = new Map();
    for (const [t, vals, ease] of pts) {
      let decl = Object.entries(vals).map(([k, v]) => `${k}:${n3(v)}`).join(";");
      if (ease && ease !== "linear") decl += `;animation-timing-function:${EASE[ease] || ease}`;
      if (!groups.has(decl)) groups.set(decl, []);
      groups.get(decl).push(n3((100 * Math.min(t, duration)) / duration) + "%");
    }
    return [...groups].map(([d, p]) => `${p.join(",")}{${d}}`).join("");
  }
}

/** Interns keyframe bodies and animation rules by their text. */
class Sheet {
  constructor(duration, prefix) { this.duration = duration; this.prefix = prefix; this.frames = new Map(); this.rules = new Map(); }
  cls(motion, duration = this.duration, delay = 0) {
    const body = motion.keyframes(duration);
    if (!this.frames.has(body)) this.frames.set(body, `k${this.prefix}${this.frames.size}`);
    let rule = `animation-name:${this.frames.get(body)}`;
    if (duration !== this.duration) rule += `;animation-duration:${n3(duration)}s`;
    if (delay) rule += `;animation-delay:${n3(delay)}s`;
    if (!this.rules.has(rule)) this.rules.set(rule, `${this.prefix}${this.rules.size}`);
    return this.rules.get(rule);
  }
  css() {
    return [...[...this.frames].map(([b, k]) => `@keyframes ${k}{${b}}`), ...[...this.rules].map(([r, c]) => `.${c}{${r}}`)].join("\n");
  }
}

// ---------- color grades (build.py) ----------

const toHex = (c) => "#" + c.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, "0")).join("");
const fromHex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
const lumOf = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

function rgbToHls([r, g, b]) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, l, 0];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  let h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, l, s];
}
function hlsToRgb([h, l, s]) {
  if (!s) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
  const f = (t) => {
    t = (t + 1) % 1;
    return t < 1 / 6 ? p + (q - p) * 6 * t : t < 1 / 2 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p;
  };
  return [f(h + 1 / 3), f(h), f(h - 1 / 3)];
}

/** Lamps, stars, lit windows: bright and warm (or near white). */
function isLight(c) {
  const [h, , s] = rgbToHls(c);
  return lumOf(c) > 0.62 && c[0] >= c[2] && ((h > 0.08 && h < 0.2) || s < 0.25);
}
function night(c, lit) {
  if (lit) return mix(c, fromHex("#fff1a8"), 0.35);
  const [h, l, s] = rgbToHls(c);
  const dh = ((0.63 - h + 1.5) % 1) - 0.5;
  return hlsToRgb([(h + 0.55 * dh + 1) % 1, l * 0.42, s * 0.85]);
}
const DAWN = [[0, "#2b1838"], [0.35, "#7a3f6b"], [0.6, "#e0866e"], [0.85, "#ffd39a"], [1, "#fff4dc"]];
function dawn(c) {
  const L = lumOf(c);
  for (let i = 1; i < DAWN.length; i++) {
    if (L <= DAWN[i][0]) {
      const [a, ca] = DAWN[i - 1], [b, cb] = DAWN[i];
      return mix(c, mix(fromHex(ca), fromHex(cb), (L - a) / (b - a)), 0.55);
    }
  }
  return c;
}

// ---------- tracing into regions ----------

/** Labels → regions: each 4-connected component's loops, color, area and centroid. */
async function traceRegions(image, { K, minArea, device, eps, seed }) {
  const { width: w, height: h, data } = image;
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const px = blur(data, w, h);
  const centers = seedCenters(px, K, rand);
  let labels = device ? await kmeansGPU(device, px, centers, K, 20) : kmeansCPU(px, centers, K, 20);
  labels = absorbSmall(majority(labels, w, h, K), w, h, minArea);
  const { comp, count } = components(labels, w, h);
  const ids = new Uint32Array(comp);
  const loops = traceLoops(ids, w, h, count);
  const W = w + 1, cache = new Map();
  const junction = (v) => isJunction(ids, w, h, v % W, (v - (v % W)) / W);
  const area = new Float64Array(count), sx = new Float64Array(count), sy = new Float64Array(count), lab = new Int32Array(count);
  for (let p = 0; p < w * h; p++) {
    const c = comp[p], x = p % w;
    area[c]++; sx[c] += x; sy[c] += (p - x) / w; lab[c] = labels[p];
  }
  const regions = [];
  for (let c = 0; c < count; c++) {
    const simple = loops[c].map((l) => simplifyLoop(l, junction, W, eps, cache)).filter((l) => l.length >= 3);
    if (!simple.length) continue;
    const L = lab[c];
    regions.push({
      loops: simple, area: area[c], cx: sx[c] / area[c] / w, cy: sy[c] / area[c] / h,
      color: shortHex(hex(centers.subarray(L * 4, L * 4 + 3))),
    });
  }
  return { regions, junction, W };
}

/** Underdrawing: borders of a heavily blurred 5-color posterization, smoothed, canvas edges cut. */
async function underdrawing(image, width, height, device) {
  const { width: w0, height: h0, data } = image;
  const f = Math.min(1, 340 / Math.max(w0, h0));
  const w = Math.max(8, Math.round(w0 * f)), h = Math.max(8, Math.round(h0 * f));
  let small = downsample(data, w0, h0, w, h);
  let px = blur(small, w, h);
  for (let i = 0; i < 2; i++) {
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let p = 0; p < w * h; p++) { rgba[p * 4] = px[p] & 255; rgba[p * 4 + 1] = (px[p] >> 8) & 255; rgba[p * 4 + 2] = (px[p] >> 16) & 255; }
    px = blur(rgba, w, h);
  }
  let s = 3;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const K = 5, centers = seedCenters(px, K, rand);
  let labels = kmeansCPU(px, centers, K, 20);
  for (let i = 0; i < 3; i++) labels = majority(labels, w, h, K);
  labels = absorbSmall(labels, w, h, Math.round(w * h * 0.0025));
  const loops = traceLoops(labels, w, h, K);
  const W = w + 1, cache = new Map();
  const junction = (v) => isJunction(labels, w, h, v % W, (v - (v % W)) / W);
  const kx = width / w, ky = height / h, lines = [];
  for (const ls of loops) for (const loop of ls) {
    const ids = simplifyLoop(loop, junction, W, 1.2, cache);
    if (ids.length < 3) continue;
    let pts = ids.map((v) => [(v % W) * kx, ((v - (v % W)) / W) * ky]);
    for (let r = 0; r < 2; r++) {
      const next = [];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length];
        next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]], [0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
      }
      pts = next;
    }
    const m = 3, inside = pts.map(([x, y]) => x > m && x < width - m && y > m && y < height - m);
    let runs;
    if (inside.every(Boolean)) runs = [[pts, "Z"]];
    else {
      const k = inside.indexOf(false), cur = [];
      runs = [];
      for (let i = 0; i < pts.length; i++) {
        const j = (k + i) % pts.length;
        if (inside[j]) cur.push(pts[j]);
        else if (cur.length) { runs.push([cur.splice(0), ""]); }
      }
      if (cur.length) runs.push([cur, ""]);
    }
    for (const [run, close] of runs) {
      let len = 0;
      for (let i = 1; i < run.length; i++) len += Math.hypot(run[i][0] - run[i - 1][0], run[i][1] - run[i - 1][1]);
      if (run.length < 2 || len < 40 * width / 1000) continue;
      const d = "M" + run.map(([x, y]) => `${Math.round(x)} ${Math.round(y)}`).join("L") + close;
      lines.push({ cx: run.reduce((a, p) => a + p[0], 0) / run.length / width, d });
    }
  }
  return lines.sort((a, b) => a.cx - b.cx);
}

// ---------- the scene ----------

const CANVAS = "#ebe3d0", CHARCOAL = "#4a3f35";

/**
 * @param {ImageData} image
 * @param {{K?: number, loop?: number, width?: number, device?: GPUDevice|null, title?: string}} opts
 * @returns {Promise<{svg: string, stats: object}>}
 */
export async function selfPainting(image, opts = {}) {
  const { K = 40, loop = 48, width = 1000, device = null, title = "A picture painting itself" } = opts;
  const t0 = performance.now();
  const { width: w, height: h } = image;
  const height = Math.round((h * width) / w);
  const sx = width / w, sy = height / h;
  const D = loop, z = D / 48;  // every beat scales with the loop length
  const B = {
    sketch: 0.3, sketched: 3.2, paint: 2, painted: 17, dusk: 22, night: 24.5, nightEnd: 29,
    dawn: 31.5, dawnEnd: 34.5, day: 37, fall: 38.5, fallen: 45, erase: 45.2, erased: 47.2,
  };
  for (const k in B) B[k] *= z;
  const fifth = (t) => Math.round(t * 5) / 5;
  const tenth = (t) => Math.round(t * 10) / 10;

  const minArea = Math.max(6, Math.round((w * h) / 25000));
  const { regions, junction, W } = await traceRegions(image, { K, minArea, device, eps: 1, seed: 1 });
  const tTrace = performance.now();

  const colors = [...new Set(regions.map((r) => r.color))];
  const cidx = new Map(colors.map((c, i) => [c, i]));
  const base = colors.map((c) => fromHex(c.length === 4 ? "#" + [...c.slice(1)].map((x) => x + x).join("") : c));
  // A color counts as lamplight only when it is bright and warm AND mostly
  // lives in small regions; otherwise a white dog or a cloud would glow.
  const small = new Float64Array(colors.length), all = new Float64Array(colors.length);
  for (const r of regions) {
    const c = cidx.get(r.color);
    all[c] += r.area;
    if (r.area < w * h * 0.002) small[c] += r.area;
  }
  const lit = base.map((c, i) => isLight(c) && small[i] / all[i] > 0.5);

  const sheet = new Sheet(D, "m"), sketchSheet = new Sheet(D, "s"), glowSheet = new Sheet(D, "g");
  let s = 7;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  const palette = (fn) => Object.fromEntries(base.map((c, i) => [`--c${i}`, toHex(fn(c, i))]));
  const orig = palette((c) => c);
  const grade = new Motion(orig).at(B.dusk).to(B.night - B.dusk, "inOut", palette((c, i) => night(c, lit[i])))
    .at(B.nightEnd).to(B.dawn - B.nightEnd, "inOut", palette(dawn))
    .at(B.dawnEnd).to(B.day - B.dawnEnd, "inOut", orig);
  const canvasGrade = new Motion({ fill: CANVAS }).at(B.dusk).to(2.5 * z, "inOut", { fill: "#1d2233" })
    .at(B.nightEnd).to(2.5 * z, "inOut", { fill: "#e9c7b0" }).at(B.dawnEnd).to(2.5 * z, "inOut", { fill: CANVAS });

  // Order: big forms first, then a left-to-right, top-to-bottom sweep.
  const logA = regions.map((r) => Math.log(r.area + 1));
  const amin = Math.min(...logA), amax = Math.max(...logA);
  const paintSpan = B.painted - 0.7 * z - B.paint, fallSpan = B.fallen - 0.8 * z - B.fall;
  const slots = new Map(), lights = [];
  const total = w * h;
  regions.forEach((r, i) => {
    const size = amax > amin ? 1 - (logA[i] - amin) / (amax - amin) : 0;
    const sweep = 0.75 * r.cx + 0.25 * r.cy;
    const order = Math.min(1, Math.max(0, 0.55 * size + 0.45 * sweep + (rand() - 0.5) * 0.06));
    const start = fifth(B.paint + order * paintSpan);
    if (!slots.has(start)) slots.set(start, []);
    const c = cidx.get(r.color);
    slots.get(start).push([c, r]);
    if (lit[c] && r.area < total * 0.002) lights.push([c, r]);
  });

  const pathsByColor = (items) => {
    const by = new Map();
    for (const [c, r] of items) { if (!by.has(c)) by.set(c, []); by.get(c).push(...r.loops); }
    return [...by].map(([c, loops]) => `<path class="c${c}" d="${curvedPathData(loops, junction, W, sx, sy, 0)}"/>`);
  };

  const fills = [];
  for (const start of [...slots.keys()].sort((a, b) => a - b)) {
    const drop = fifth(B.fall + ((start - B.paint) / paintSpan) * fallSpan);
    // Inherited properties only (fill/stroke opacity, stroke width): an
    // animated opacity or transform makes the browser composite each group as
    // its own layer on every frame, which cost 5x the frame time in testing.
    const dab = new Motion({ "fill-opacity": 0, "stroke-opacity": 0, "stroke-width": 9 }).at(start)
      .to(0.9 * z, "out", { "fill-opacity": 1, "stroke-opacity": 1, "stroke-width": 1 })
      .at(drop).to(0.8 * z, "in", { "fill-opacity": 0, "stroke-opacity": 0, "stroke-width": 0 });
    fills.push(`<g class="${sheet.cls(dab)}">`, ...pathsByColor(slots.get(start)), "</g>");
  }

  const sketches = (await underdrawing(image, width, height, device)).map(({ cx, d }) => {
    const start = tenth(B.sketch + cx * (B.sketched - 1.2 * z - B.sketch) + rand() * 0.3 * z);
    const gone = tenth(B.erase + cx * (B.erased - 0.6 * z - B.erase));
    const line = new Motion({ "stroke-dashoffset": 1 }).at(start).to(1.0 * z, "inOut", { "stroke-dashoffset": 0 })
      .at(gone).to(0.6 * z, "in", { "stroke-dashoffset": -1 });
    return `<path class="${sketchSheet.cls(line)}" d="${d}" pathLength="1"/>`;
  });

  // Night glow: a soft radial halo over each small bright region, in six
  // twinkle groups. Halos instead of blurred copies: a blur filter re-runs on
  // every repaint and cost 4-5x the frame time in testing.
  lights.sort((a, b) => b[1].area - a[1].area).splice(300);
  // Plain opacity is right here, unlike on the paint groups: six small groups
  // measured 56 fps against 12 for fill-opacity driven by custom properties.
  const envelope = new Motion({ opacity: 0 }).at(B.dusk + 1 * z).to(2 * z, "inOut", { opacity: 1 })
    .at(B.nightEnd - 0.5 * z).to(2.5 * z, "inOut", { opacity: 0 });
  const twinkle = [];
  for (let k = 0; k < 6; k++) {
    const mine = lights.filter((_, i) => i % 6 === k);
    if (!mine.length) continue;
    const period = [2.4, 3.2, 4.0][k % 3];
    const tw = new Motion({ opacity: 0.3 }).to(period / 2, "sine", { opacity: 1 }).to(period / 2, "sine", { opacity: 0.3 });
    twinkle.push(`<g class="${glowSheet.cls(tw, period, -Math.round(period * k / 6 * 100) / 100)}">`,
      ...mine.map(([, r]) => {
        const rad = Math.sqrt(r.area * sx * sy) * 1.6 + 6 * width / 1000;
        return `<circle cx="${Math.round(r.cx * width)}" cy="${Math.round(r.cy * height)}" r="${Math.round(rad)}"/>`;
      }), "</g>");
  }

  const rootCls = sheet.cls(grade), canvasCls = sheet.cls(canvasGrade), envCls = glowSheet.cls(envelope);

  const esc = (t) => t.replace(/[<&"]/g, (ch) => ({ "<": "&lt;", "&": "&amp;", '"': "&quot;" })[ch]);
  const css = [
    ...colors.map((c, i) => `@property --c${i}{syntax:'<color>';inherits:true;initial-value:${c}}`),
    colors.map((_, i) => `.c${i}{fill:var(--c${i});stroke:var(--c${i})}`).join(""),
    `#v,:where(#v *){animation-duration:${n3(D)}s;animation-iteration-count:infinite;animation-fill-mode:both;animation-timing-function:linear}`,
    `#k path{fill:none;stroke:${CHARCOAL};stroke-width:${n3(1.4 * width / 1000)};stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1 1;opacity:.7}`,
    sheet.css(), sketchSheet.css(), glowSheet.css(),
    `@media (prefers-reduced-motion:reduce){#v,#v *{animation:none!important}#k,#glow{display:none}}`,
  ].join("\n");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" id="v" class="${rootCls}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="t">
<title id="t">${esc(title)}</title>
<defs><radialGradient id="halo"><stop offset="0" stop-color="#fff3b0" stop-opacity=".85"/><stop offset=".35" stop-color="#ffe680" stop-opacity=".35"/><stop offset="1" stop-color="#ffd860" stop-opacity="0"/></radialGradient></defs>
<style><![CDATA[
${css}
]]></style>
<rect class="${canvasCls}" width="${width}" height="${height}" fill="${CANVAS}"/>
<g id="k">
${sketches.join("\n")}
</g>
<g id="f" fill-rule="evenodd" stroke-width="1" stroke-linejoin="round">
${fills.join("\n")}
</g>
<g id="glow" class="${envCls}" fill="url(#halo)">
${twinkle.join("\n")}
</g>
</svg>
`;
  return {
    svg,
    stats: {
      backend: device ? "webgpu" : "js", w, h, K, regions: regions.length, colors: colors.length,
      slots: slots.size, sketchLines: sketches.length, lights: lights.length,
      keyframes: sheet.frames.size + sketchSheet.frames.size + glowSheet.frames.size, bytes: svg.length,
      ms: { trace: Math.round(tTrace - t0), total: Math.round(performance.now() - t0) },
    },
  };
}
