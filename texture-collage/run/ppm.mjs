// Minimal PPM (P6) read/write for the Node harness.
import fs from "node:fs";
export function readPPM(path) {
  const b = fs.readFileSync(path);
  let pos = 0; const tok = [];
  while (tok.length < 4) {
    while (/\s/.test(String.fromCharCode(b[pos]))) pos++;
    if (b[pos] === 35) { while (b[pos] !== 10) pos++; continue; }
    let s = ""; while (!/\s/.test(String.fromCharCode(b[pos]))) s += String.fromCharCode(b[pos++]);
    tok.push(s);
  }
  pos++;
  const W = +tok[1], H = +tok[2], rgb = new Uint8Array(b.buffer, b.byteOffset + pos, 3 * W * H);
  const rgba = new Uint8Array(4 * W * H);
  for (let i = 0; i < W * H; i++) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2]; rgba[4 * i + 3] = 255; }
  return { W, H, rgba };
}
export function writePPM(path, rgba, W, H) {
  const rgb = Buffer.alloc(3 * W * H);
  for (let i = 0; i < W * H; i++) { rgb[3 * i] = rgba[4 * i]; rgb[3 * i + 1] = rgba[4 * i + 1]; rgb[3 * i + 2] = rgba[4 * i + 2]; }
  fs.writeFileSync(path, Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), rgb]));
}
