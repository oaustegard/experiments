// Renders the page's 3D skull (XR.buildSkull) alone at a grid of head poses through the CPU port of the
// shaders, for checking the anatomy by eye. Writes skull-sheet.png: columns are yaw, rows pitch and jaw.
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const { rasterBones, composite, XR } = require('./render-cpu.js');
const S = 300, k = 9;  // tile px, px per cm
const rot = (yaw, pitch) => {  // canonical -> camera rows: yaw about y, then pitch about x
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
  return [0, 1, 2].map(i => [0, 1, 2].map(j => Rx[i][0] * Ry[0][j] + Rx[i][1] * Ry[1][j] + Rx[i][2] * Ry[2][j]));
};
const yaws = [0, 30, 60, 90, 135, 180], rows = [[0, 0], [-20, 0], [0, 0.45]];
const out = new PNG({ width: S * yaws.length, height: S * rows.length });
const black = new Uint8Array(S * S * 4);
rows.forEach(([pitch, jaw], r) => yaws.forEach((yaw, c) => {
  const R = rot(yaw * Math.PI / 180, pitch * Math.PI / 180);
  const H = { M: [R[0].map(v => v * k), R[1].map(v => -v * k)], b: [S / 2, S / 2 - 1.5 * k], k, jaw };
  const pr = new XR.Prims(1024); pr.glow = 3;
  XR.buildSkull(pr, H);
  const px = composite(black, S, S, null, 0, 0, rasterBones(pr, S, S), false);
  for (let y = 0; y < S; y++) px.copy(out.data, ((r * S + y) * out.width + c * S) * 4, y * S * 4, (y + 1) * S * 4);
}));
fs.writeFileSync(path.join(__dirname, '..', 'skull-sheet.png'), PNG.sync.write(out));
console.log('wrote skull-sheet.png');
