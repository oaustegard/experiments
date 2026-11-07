// Region worker for the splat collage: cuts the picture and picks classes off the page's thread.
import * as C from "../web/collage.js";

let img = null, seg = null, rdesc = null, table = null;

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
    const solid = C.labToRgba(C.renderSolid(seg), img.W * img.H);
    return { K: seg.K, labels: seg.labels.slice(), stats: seg.stats, rdesc: rdesc.map((d) => ({ theta: d.theta, f: d.f })),
      solid, ms: { segment: t1 - t0, describe: t2 - t1 }, transfer: [solid.buffer] };
  },
  choose(m) {
    const regions = rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac }));
    return { pick: C.chooseClasses(regions, table, m.opts) };
  },
  // Lab stats, orientation and coherence of a freshly generated tile painted on the GPU (RGBA bytes)
  describeTile(m) {
    const S = m.S, rgba = new Uint8Array(m.rgba), lab = C.rgbToLab(rgba, S * S, 4), L = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) L[i] = lab[3 * i];
    const d = C.describe(L, null, S, S, true);
    return { stats: C.labStats(lab, S * S), theta: d.theta, gcoh: d.f[3] };
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
