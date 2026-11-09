// node web/export_bin.mjs kodim23 [kodim05 ...]   (no ids: every gallery image)
// Writes web/blobs/<id>.bin, the packed splat table of each gallery image (see emitBin in trainer.js).
import { mkdirSync, writeFileSync } from "node:fs";
import { IMAGES } from "./shaders.js";
import { emitBin } from "./trainer.js";

const out = new URL("./blobs/", import.meta.url);
mkdirSync(out, { recursive: true });
const want = process.argv.slice(2);
for (const e of IMAGES) {
  if (want.length && !want.includes(e.id)) continue;
  const bin = Buffer.from(emitBin(e));
  writeFileSync(new URL(`${e.id}.bin`, out), bin);
  console.log(e.id, e.n, "splats", bin.length, "bytes");
}
