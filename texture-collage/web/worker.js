// Collage worker: keeps the image, its segmentation and the prepared tiles, so the page's thread
// only draws. Messages: image, segment, choose, render (each answered with the same id).
import * as C from "./collage.js";

let img = null, seg = null, rdesc = null, table = null;
const tiles = new Map();     // key -> prepared tile

const H = {
  table(m) { table = m.table; return {}; },
  image(m) {
    img = { W: m.W, H: m.H, lab: C.rgbToLab(new Uint8Array(m.rgba), m.W * m.H, 4) };
    seg = null; rdesc = null;
    return {};
  },
  segment(m) {
    const t0 = performance.now();
    seg = C.segment(img.lab, img.W, img.H, { K: m.K });
    const t1 = performance.now();
    rdesc = C.regionDescriptors(img.lab, seg);
    const t2 = performance.now();
    const preview = C.labToRgba(C.renderSolid(seg), img.W * img.H);
    return { K: seg.K, labels: seg.labels.slice(), stats: seg.stats.map((s) => ({ mean: s.mean, frac: s.frac })),
      solid: preview, ms: { segment: t1 - t0, describe: t2 - t1 }, transfer: [preview.buffer] };
  },
  choose(m) {
    const regions = rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac }));
    const pick = C.chooseClasses(regions, table, m.opts);
    return { pick };
  },
  render(m) {
    const t0 = performance.now();
    const tex = m.tiles.map((t) => {
      const key = `${t.key}@${t.scale}`;
      if (!tiles.has(key)) tiles.set(key, C.prepTile(new Uint8Array(t.rgb), t.S, t.scale));
      return tiles.get(key);
    });
    if (tiles.size > 400) tiles.clear();
    const out = C.render(img.lab, seg, tex, { ...m.opts, scales: m.tiles.map((t) => t.scale), rdesc });
    const rgba = C.labToRgba(out, img.W * img.H);
    return { rgba, ms: performance.now() - t0, transfer: [rgba.buffer] };
  },
};

onmessage = (e) => {
  const m = e.data;
  try {
    const r = H[m.type](m);
    const transfer = r.transfer || [];
    delete r.transfer;
    postMessage({ id: m.id, ...r }, transfer);
  } catch (err) {
    postMessage({ id: m.id, error: err.message || String(err) });
  }
};
