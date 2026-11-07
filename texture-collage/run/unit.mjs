// Unit checks for collage.js: colour round trip, segmentation of a synthetic image, descriptor
// orientation and the alignment rotation in render().
import * as C from "../web/collage.js";
let fails = 0;
const ok = (cond, msg) => { console.log((cond ? "ok   " : "FAIL ") + msg); if (!cond) fails++; };

// 1. Lab round trip
{
  const rgb = new Uint8Array([10, 200, 30, 255, 250, 250, 250, 255, 0, 0, 0, 255, 128, 64, 220, 255]);
  const back = C.labToRgba(C.rgbToLab(rgb, 4, 4), 4);
  let e = 0; for (let i = 0; i < 16; i++) e = Math.max(e, Math.abs(back[i] - rgb[i]));
  ok(e <= 1, `Lab round trip max error ${e}`);
}
// 2. two-colour image with a small speck: 2 regions, speck merged
{
  const W = 120, H = 80, rgb = new Uint8Array(4 * W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, left = x < 50, speck = x > 90 && x < 93 && y > 30 && y < 33;
    const c = speck ? [0, 0, 255] : left ? [200, 40, 40] : [40, 180, 60];
    rgb.set([...c, 255], 4 * i);
  }
  const lab = C.rgbToLab(rgb, W * H, 4);
  const seg = C.segment(lab, W, H, { K: 3, minFrac: 0.002 });
  const { sizes } = C.components(seg.labels, W, H);
  ok(sizes.length === 2, `synthetic: ${sizes.length} components (want 2), K=${seg.K}`);
}
// 3. stripes: descriptor theta and render alignment
{
  const S = 64, tile = new Uint8Array(3 * S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {    // vertical stripes: gradient along x, theta 0
    const v = Math.round(127 + 120 * Math.sin((2 * Math.PI * 4 * x) / S));
    tile.set([v, v, v], 3 * (y * S + x));
  }
  const t = C.prepTile(tile, S, 1);
  ok(Math.abs(t.theta) < 0.05 && t.gcoh > 0.9, `stripe tile theta ${t.theta.toFixed(3)} gcoh ${t.gcoh.toFixed(2)}`);
  // region: diagonal stripes at 45 degrees
  const W = 96, H = 96, rgb = new Uint8Array(4 * W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = Math.round(127 + 100 * Math.sin((2 * Math.PI * (x + y)) / 20));
    rgb.set([v, v, v, 255], 4 * (y * W + x));
  }
  const lab = C.rgbToLab(rgb, W * H, 4);
  const seg = { W, H, K: 1, labels: new Uint8Array(W * H), blurR: 1, stats: C.regionStats(lab, new Uint8Array(W * H), 1, W * H) };
  const rd = C.regionDescriptors(lab, seg);
  ok(Math.abs(rd[0].theta - Math.PI / 4) < 0.05, `diagonal region theta ${rd[0].theta.toFixed(3)} (want ${(Math.PI / 4).toFixed(3)})`);
  for (const align of [false, true]) {
    const out = C.render(lab, seg, [t], { align, feather: 0, rdesc: rd });
    const L = new Float32Array(W * H); for (let i = 0; i < W * H; i++) L[i] = out[3 * i];
    const d = C.describe(L, null, W, H);
    const err = Math.abs(((d.theta - rd[0].theta + Math.PI / 2) % Math.PI + Math.PI) % Math.PI - Math.PI / 2);
    ok(align ? err < 0.05 : err > 0.5, `render align=${align}: orientation error ${err.toFixed(3)} rad`);
  }
}
console.log(fails ? `${fails} failed` : "all passed");
process.exit(fails ? 1 : 0);
