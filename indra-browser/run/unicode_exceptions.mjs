// The characters where web/uprops.js's regex approximation differs from Python's
// (run/unicode_props.py output): writes them as {char: flags}.
//   node run/unicode_exceptions.mjs PROPS.json OUT.json
import fs from "node:fs";
import { approxFlags } from "../web/uprops.js";
const [src, dst] = process.argv.slice(2);
const { default: def, props } = JSON.parse(fs.readFileSync(src, "utf8"));
const meta = JSON.parse(fs.readFileSync(process.env.META ?? "/tmp/indra-models/meta.json", "utf8"));
const chars = new Set();
for (const [k, v] of Object.entries(meta)) if (k.startsWith("vocab")) for (const t of JSON.parse(v).chars) for (const c of t) chars.add(c);
for (let c = 0xac00; c < 0xd7a4; c++) chars.add(String.fromCodePoint(c));
const exc = {};
const bits = {};
for (const c of chars) {
  const want = c in props ? props[c] : def;
  const got = approxFlags(c);
  if (want !== got) {
    exc[c] = want;
    const d = want ^ got;
    for (let b = 1; b <= 2048; b <<= 1) if (d & b) bits[b] = (bits[b] ?? 0) + 1;
  }
}
fs.writeFileSync(dst, JSON.stringify(exc));
console.log(chars.size, "chars,", Object.keys(exc).length, "exceptions; by bit", bits);
