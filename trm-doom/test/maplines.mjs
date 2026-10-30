// node test/maplines.mjs MAP > lines.json : [[x1, y1, x2, y2, oneSided, special]]
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { mapVerts } from "../web/grid.js";
const D = new URL("../../../oaustegard.github.io/fun-and-games/doom/", import.meta.url).pathname;
const { readWad, buildGfx, buildMap } = await import(D + "wad.js");
const u8 = gunzipSync(readFileSync(D + "e1.wad.gz"));
const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
const name = process.argv[2], map = buildMap(wad, buildGfx(wad), name, 3), V = mapVerts(wad, name);
process.stdout.write(JSON.stringify(map.lines.map((l) => [...V[l.v1], ...V[l.v2], l.bsec < 0 ? 1 : 0, l.special])));
