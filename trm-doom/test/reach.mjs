// Guard for grid changes, both ways. MUST: places the bot reached in real play (bridge runs doom-lift-* and
// doom-cut-*, 2026-10-30) that are open from the start in the WAD's own state (doors shut, lifts at rest, no keys)
// must stay reachable on the whole-level grid. APART: two cells either side of a wall must not be joined within a
// few steps; pooling used to join them (E1M2's locked blue door, E1M3's switch room and the corridor behind it).
// The earlier list (E1M1 blue key, E1M2 red key, E1M4 keys and exit) was dropped: those passed only through such
// joins, and the bot never reached them in play.  node test/reach.mjs
import { readFileSync } from "node:fs"; import { gunzipSync } from "node:zlib";
import { makeGridder, wadState, mapVerts, passable } from "../web/grid.js";
const D = new URL("../../../oaustegard.github.io/fun-and-games/doom/", import.meta.url).pathname;
const { readWad, buildGfx, buildMap } = await import(D + "wad.js");
const u8 = gunzipSync(readFileSync(D + "e1.wad.gz")); const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)); const gfx = buildGfx(wad);
const MUST = {
  E1M2: [[1662, -1037], [1664, -960], [464, -16], [1664, -1376], [672, -624]],
  E1M3: [[-1040, 1360], [-952, 1896]],
  E1M4: [[-736, -48]],
  E1M5: [[-4400, 1192], [-4510, 1415], [-2016, 1456]],
  E1M6: [[-33, 116], [-560, 1256], [96, -168]],
  E1M7: [[-408, 352], [-416, 1888]],
};
const APART = { E1M2: [["blue door", [320, -970], [320, -1045], 4]], E1M3: [["switch room", [-1030, 1900], [-952, 1870], 6]] };
let fail = 0;
for (const name of new Set([...Object.keys(MUST), ...Object.keys(APART)])) {
  const map = buildMap(wad, gfx, name, 2), gr = makeGridder(map, mapVerts(wad, name)), b = map.bmap;
  const cs = 48, n = Math.ceil(Math.max(b.w, b.h) * 128 / cs / 8) * 8 + 8;
  const grid = gr.coarse(b.x + b.w * 64, b.y + b.h * 64, wadState(map), cs, n), g = grid.g;
  const cell = (x, y) => Math.floor((y - grid.oy) / cs) * n + Math.floor((x - grid.ox) / cs);
  const dist = (x, y) => { const d = new Int32Array(n * n).fill(-1), s = cell(x, y), q = [s]; d[s] = 0;
    for (let h = 0; h < q.length; h++) { const o = q[h], i = o % n, j = (o / n) | 0;
      for (const p of [i > 0 ? o - 1 : -1, i < n - 1 ? o + 1 : -1, j > 0 ? o - n : -1, j < n - 1 ? o + n : -1]) if (p >= 0 && d[p] < 0 && passable(g[p])) { d[p] = d[o] + 1; q.push(p); } }
    return d; };
  const d0 = dist(map.player.x, map.player.y);
  for (const [x, y] of MUST[name] || []) { // the cell or a neighbour: an item can sit by a wall
    let ok = false; for (const dx of [-48, 0, 48]) for (const dy of [-48, 0, 48]) ok ||= d0[cell(x + dx, y + dy)] >= 0;
    if (!ok) fail++; console.log(ok ? "ok  " : "FAIL", name, "reach", x, y); }
  for (const [what, a, z, min] of APART[name] || []) { const k = dist(...a)[cell(...z)], ok = k < 0 || k > min;
    if (!ok) fail++; console.log(ok ? "ok  " : "FAIL", name, what, "apart", k); }
}
process.exit(fail ? 1 : 0);
