// Guard for grid changes: places a player can walk to from the start in the WAD's own state (doors shut,
// lifts up, no keys) must stay reachable on the whole-level grid. Each was checked by hand; a grid rule that
// cuts one (E1M1's blue key went missing with a 40-unit drop margin) fails here.  node test/reach.mjs
import { readFileSync } from "node:fs"; import { gunzipSync } from "node:zlib";
import { makeGridder, wadState, mapVerts, passable } from "../web/grid.js";
const D = new URL("../../../oaustegard.github.io/fun-and-games/doom/", import.meta.url).pathname;
const { readWad, buildGfx, buildMap } = await import(D + "wad.js");
const u8 = gunzipSync(readFileSync(D + "e1.wad.gz")); const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)); const gfx = buildGfx(wad);
const MUST = { E1M1: [["blue key", 2192, 576]], E1M2: [["red key", 672, -2960]], E1M4: [["yellow key", -960, 640], ["blue key", 3456, 1472], ["exit", 608, 2136]] };
let fail = 0;
for (const [name, pts] of Object.entries(MUST)) {
  const map = buildMap(wad, gfx, name, 2), gr = makeGridder(map, mapVerts(wad, name)), b = map.bmap;
  const cs = 48, n = Math.ceil(Math.max(b.w, b.h) * 128 / cs / 8) * 8 + 8;
  const grid = gr.coarse(b.x + b.w * 64, b.y + b.h * 64, wadState(map), cs, n), g = grid.g;
  const cell = (x, y) => Math.floor((y - grid.oy) / cs) * n + Math.floor((x - grid.ox) / cs);
  const s0 = cell(map.player.x, map.player.y), seen = new Uint8Array(n * n), q = [s0]; seen[s0] = 1;
  for (let h = 0; h < q.length; h++) { const o = q[h], i = o % n, j = (o / n) | 0;
    for (const p of [i > 0 ? o - 1 : -1, i < n - 1 ? o + 1 : -1, j > 0 ? o - n : -1, j < n - 1 ? o + n : -1]) if (p >= 0 && !seen[p] && passable(g[p])) { seen[p] = 1; q.push(p); } }
  for (const [what, x, y] of pts) { const ok = !!seen[cell(x, y)]; if (!ok) fail++; console.log(ok ? "ok  " : "FAIL", name, what); }
}
process.exit(fail ? 1 : 0);
