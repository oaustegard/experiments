// ocria.js end to end in Node, against the run/lp_pages.mjs reading of the same pixels with the same weights.
//
// Fixtures are raw gray pages (u32 width, u32 height, then the bytes) from run/make_fixtures.py. Each goes
// through ocria.js as the browser page sends it (RGBA -> gray -> one-page PDF, packs unpacked in JS) and
// its markdown is compared with REF/pages/<id>.json (lp_pages.mjs on the Python-wrapped PDF, the unpacked
// arm of the same packs). Runs inside the hub's scripts/jail.sh: it executes LiteParse's wasm build.
//
//   node web/test/check_ocria.mjs --pkg PKG --packs web/models-L127 --fixtures FIXDIR --ref out/ARM --out res.json
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const here = path.dirname(new URL(import.meta.url).pathname);
const { loadOcria } = await import(pathToFileURL(path.join(here, "..", "ocria.js")).href);
const lp = await import(pathToFileURL(path.resolve(args.pkg, "liteparse_wasm.js")).href);
await lp.default({ module_or_path: fs.readFileSync(path.resolve(args.pkg, "liteparse_wasm_bg.wasm")) });
const localFetch = async (url) => {
  const p = url.replace(/^file:\/\//, "");
  return { ok: fs.existsSync(p), status: 404, arrayBuffer: async () => fs.readFileSync(p).buffer.slice(0) };
};
const ocr = await loadOcria({
  base: "file://" + path.resolve(args.packs) + "/", layout: args.layout !== "off",
  scripts: (args.scripts || "world,rtl,cjk").split(","), lp, fetch: localFetch, log: (m) => console.log(m),
});

const rows = [];
for (const f of fs.readdirSync(args.fixtures).filter((x) => x.endsWith(".gray")).sort()) {
  const id = f.slice(0, -5);
  const b = fs.readFileSync(path.join(args.fixtures, f));
  const w = b.readUInt32LE(0), h = b.readUInt32LE(4);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { const v = b[8 + i]; rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = v; rgba[4 * i + 3] = 255; }
  const page = await ocr.read({ data: rgba, width: w, height: h });
  const ref = JSON.parse(fs.readFileSync(path.join(args.ref, "pages", id + ".json"), "utf8"));
  const same = page.markdown === ref.markdown;
  rows.push({ id, same, ms: Math.round(page.ms), items: page.items.length, chars: page.markdown.length, ref_chars: (ref.markdown || "").length });
  console.log(same ? "same" : "DIFF", id, Math.round(page.ms) + " ms");
}
const out = { pages: rows.length, same_markdown: rows.filter((r) => r.same).length, load_ms: Math.round(ocr.loadMs), pack_bytes: ocr.packBytes, rows };
fs.writeFileSync(args.out, JSON.stringify(out, null, 1));
console.log(JSON.stringify({ pages: out.pages, same_markdown: out.same_markdown, pack_bytes: out.pack_bytes }));
