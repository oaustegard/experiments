import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
const D = "/home/user/oaustegard/oaustegard.github.io/fun-and-games/doom/";
const { readWad, buildGfx, buildMap } = await import(D + "wad.js");
const u8 = gunzipSync(readFileSync(D + "e1.wad.gz"));
const wad = readWad(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
const gfx = buildGfx(wad);
for (const m of wad.maps()) {
  const map = buildMap(wad, gfx, m, 3);
  const b = map.bmap;
  console.log(m, "lines", map.lines.length, "sectors", map.sectors.length, "bmap", b.x, b.y, b.w * 128, b.h * 128, "things", map.things.length);
}
