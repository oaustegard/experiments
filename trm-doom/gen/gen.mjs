// Training data: random positions, sector states and goals on the E1 levels, labelled by BFS.
// node gen/gen.mjs OUTDIR COUNT SEED MAPS(comma) [coarseFrac]
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { makeGridder, shortestUnion, wadState, mapVerts, C, N, passable, MANUAL_DOORS } from "../web/grid.js";

const DOOM = process.env.DOOM_DIR || new URL("../../../oaustegard.github.io/games/doom/", import.meta.url).pathname;
const { readWad, buildGfx, buildMap } = await import(DOOM + "wad.js");
const { MOBJINFO, MF } = await import(DOOM + "info.js");
const [out, count, seed, mapsArg, coarseFrac = "0.3"] = process.argv.slice(2);
let s = +seed >>> 0 || 1;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
const u8 = gunzipSync(readFileSync(DOOM + "e1.wad.gz"));
const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
const gfx = buildGfx(wad);
const LIFTS = new Set([10, 21, 62, 88, 120, 121, 122, 123]);

const levels = mapsArg.split(",").map((name) => {
  const map = buildMap(wad, gfx, name, 3);
  const gr = makeGridder(map, mapVerts(wad, name));
  const doors = [...gr.doorKey.keys()];
  const lifts = new Set();
  map.lines.forEach((l) => { if (LIFTS.has(l.special) && l.tag) map.sectors.forEach((sec, i) => { if (sec.tag === l.tag) lifts.add(i); }); });
  const nb = map.sectors.map(() => new Set());
  for (const l of map.lines) if (l.fsec >= 0 && l.bsec >= 0) { nb[l.fsec].add(l.bsec); nb[l.bsec].add(l.fsec); }
  const solids = map.things.filter((t) => { const f = MOBJINFO[t.mt][21]; return (f & MF.SOLID) && !(f & MF.COUNTKILL); })
    .map((t) => [t.x, t.y, MOBJINFO[t.mt][16]]);
  const b = map.bmap;
  return { name, map, gr, doors, lifts: [...lifts], nb, solids, box: [b.x, b.y, b.x + b.w * 128, b.y + b.h * 128] };
});

function state(L) {
  const st = wadState(L.map);
  for (const d of L.doors) if (rnd() < 0.35) st.ceil[d] = Math.min(...[...L.nb[d]].map((k) => st.ceil[k])) - 4;
  for (const d of L.lifts) if (rnd() < 0.3) st.floor[d] = Math.min(st.floor[d], ...[...L.nb[d]].map((k) => st.floor[k]));
  st.cards = 0; for (let k = 0; k < 6; k++) if (rnd() < 0.25) st.cards |= 1 << k;
  st.solids = L.solids.filter(() => rnd() < 0.9);
  return st;
}

function standable(L, st, x, y) {
  const sec = L.map.spawnSector(x, y);
  return L.gr.inSector(sec, x, y) && st.ceil[sec] - st.floor[sec] >= 56;
}

function sample(L) {
  const st = state(L);
  let x, y;
  for (let t = 0; ; t++) {
    x = L.box[0] + rnd() * (L.box[2] - L.box[0]); y = L.box[1] + rnd() * (L.box[3] - L.box[1]);
    if (standable(L, st, x, y)) break;
  }
  const isCoarse = rnd() < +coarseFrac;
  const { g } = isCoarse ? L.gr.coarse(x, y, st, 96) : L.gr.fine(x, y, st);
  const c0 = (N / 2) * N + N / 2;
  g[c0] = C.START;
  // goal: mostly a reachable cell, sometimes one that isn't
  const reach = new Uint8Array(N * N), q = [c0]; reach[c0] = 1;
  while (q.length) { const o = q.pop(), i = o % N, j = (o / N) | 0;
    for (const p of [i > 0 ? o - 1 : -1, i < N - 1 ? o + 1 : -1, j > 0 ? o - N : -1, j < N - 1 ? o + N : -1])
      if (p >= 0 && !reach[p] && passable(g[p])) { reach[p] = 1; q.push(p); } }
  const cand = [], other = [];
  for (let k = 0; k < N * N; k++) if (k !== c0 && passable(g[k])) (reach[k] ? cand : other).push(k);
  const pool = rnd() < 0.12 && other.length ? other : cand.length ? cand : other;
  if (!pool.length) return null;
  g[pool[Math.floor(rnd() * pool.length)]] = C.GOAL;
  return { g, y: shortestUnion(g), coarse: isCoarse };
}

function npy(path, arr, shape) {
  let h = `{'descr': '|u1', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  h += " ".repeat(64 - ((10 + h.length + 1) % 64)) + "\n";
  const head = Buffer.alloc(10); head.write("\x93NUMPY", 0, "latin1"); head[6] = 1; head[7] = 0; head.writeUInt16LE(h.length, 8);
  writeFileSync(path, Buffer.concat([head, Buffer.from(h, "latin1"), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)]));
}

mkdirSync(out, { recursive: true });
const n = +count, X = new Uint8Array(n * N * N), Y = new Uint8Array(n * N * N), M = new Uint8Array(n * 2);
const t0 = Date.now();
for (let k = 0; k < n; ) {
  const li = Math.floor(rnd() * levels.length), r = sample(levels[li]);
  if (!r) continue;
  X.set(r.g, k * N * N); Y.set(r.y, k * N * N); M[2 * k] = li; M[2 * k + 1] = r.coarse ? 1 : 0;
  k++;
  if (k % 5000 === 0) console.error(k, ((Date.now() - t0) / k).toFixed(2), "ms/sample");
}
npy(`${out}/x.npy`, X, [n, N * N]); npy(`${out}/y.npy`, Y, [n, N * N]); npy(`${out}/meta.npy`, M, [n, 2]);
console.error("done", n, (Date.now() - t0) / 1000, "s");
