// Bigger grids to test whether the 32×32-trained conv TRM extrapolates: node gen/gen_big.mjs OUT COUNT N CS MAPS
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { makeGridder, shortestUnion, wadState, mapVerts, C, passable } from "../web/grid.js";
const DOOM = new URL("../../../oaustegard.github.io/fun-and-games/doom/", import.meta.url).pathname;
const { readWad, buildGfx, buildMap } = await import(DOOM + "wad.js");
const [out, count, nS, csS, mapsArg, seedS = "12345"] = process.argv.slice(2); const n = +nS, cs = +csS;
let s = +seedS >>> 0 || 1; const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
const u8 = gunzipSync(readFileSync(DOOM + "e1.wad.gz"));
const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)); const gfx = buildGfx(wad);
const levels = mapsArg.split(",").map((name) => { const map = buildMap(wad, gfx, name, 3); return { map, gr: makeGridder(map, mapVerts(wad, name)), b: map.bmap }; });
const X = new Uint8Array(+count * n * n), Y = new Uint8Array(+count * n * n), Dd = new Int32Array(+count);
for (let k = 0; k < +count;) {
  const L = levels[Math.floor(rnd() * levels.length)], st = wadState(L.map);
  const x = L.b.x + rnd() * L.b.w * 128, y = L.b.y + rnd() * L.b.h * 128, sec = L.map.spawnSector(x, y);
  if (!L.gr.inSector(sec, x, y) || st.ceil[sec] - st.floor[sec] < 56) continue;
  const { g } = L.gr.coarse(x, y, st, cs, n); const c0 = (n / 2) * n + n / 2; g[c0] = C.START;
  const reach = [], seen = new Uint8Array(n * n), q = [c0]; seen[c0] = 1;
  for (let h = 0; h < q.length; h++) { const o = q[h], i = o % n, j = (o / n) | 0;
    for (const p of [i > 0 ? o - 1 : -1, i < n - 1 ? o + 1 : -1, j > 0 ? o - n : -1, j < n - 1 ? o + n : -1]) if (p >= 0 && !seen[p] && passable(g[p])) { seen[p] = 1; q.push(p); reach.push(p); } }
  if (reach.length < 20) continue;
  let gc = reach[Math.floor(reach.length * rnd())];
  if (rnd() < 0.1) { const other = []; for (let o = 0; o < n * n; o++) if (!seen[o] && passable(g[o])) other.push(o); if (other.length) gc = other[Math.floor(rnd() * other.length)]; }
  g[gc] = C.GOAL; X.set(g, k * n * n); Y.set(shortestUnion(g, n), k * n * n); k++;
}
mkdirSync(out, { recursive: true });
const npy = (p, a, shape, d = "|u1") => { let h = `{'descr': '${d}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`; h += " ".repeat(64 - ((10 + h.length + 1) % 64)) + "\n";
  const hd = Buffer.alloc(10); hd.write("\x93NUMPY", 0, "latin1"); hd[6] = 1; hd.writeUInt16LE(h.length, 8); writeFileSync(p, Buffer.concat([hd, Buffer.from(h, "latin1"), Buffer.from(a.buffer, a.byteOffset, a.byteLength)])); };
npy(`${out}/x.npy`, X, [+count, n * n]); npy(`${out}/y.npy`, Y, [+count, n * n]);
console.error("done");
