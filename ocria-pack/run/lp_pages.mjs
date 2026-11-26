// Parse OmniParseBench pages with LiteParse's wasm build and OCRia-nano-1, one JSON per page.
//
// Runs inside the hub's scripts/jail.sh (it executes @llamaindex/liteparse-wasm, code nobody here
// wrote). The models come from a local folder laid out like the Hugging Face repo, so a packed arm is
// a folder of the same files rebuilt from the packed bytes. Pages: PDFs as the dataset ships them,
// image pages as the one-page PDFs run/png2pdf.py makes.
//
//   node run/lp_pages.mjs --pkg npm/package/pkg --models models --pdf PDFDIR --list pages.txt \
//        --out out/ARM [--shard 0/4] [--no-layout] [--scripts latin,cjk,rtl,world] [--dpi 250] [--dump-items]
import fs from "node:fs";
import path from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : true]);
    return acc;
  }, []),
);
const pkg = path.resolve(args.pkg);
const { default: init, LiteParse, OcriaNano } = await import(path.join(pkg, "liteparse_wasm.js"));
await init({ module_or_path: fs.readFileSync(path.join(pkg, "liteparse_wasm_bg.wasm")) });

const M = (f) => {
  const p = path.join(args.models, f);
  return fs.existsSync(p) ? new Uint8Array(fs.readFileSync(p)) : undefined;
};
const scripts = (args.scripts || "latin,cjk,rtl,world").split(",");
const rec = (s) => (scripts.includes(s) && M(`rec_${s}.onnx`)
  ? { onnx: M(`rec_${s}.onnx`), charset: M(`rec_${s}.charset.json`), meta: M(`rec_${s}.meta.json`) }
  : undefined);
const models = {
  det: M("det.onnx"), detMeta: M("det.meta.json"), router: M("router.onnx"),
  recLatin: rec("latin"), recCjk: rec("cjk"), recRtl: rec("rtl"), recWorld: rec("world"),
};
if (!args["no-layout"]) models.tablenet = M("tablenet.bin");
for (const k of Object.keys(models)) if (models[k] === undefined) delete models[k];
const t0 = performance.now();
const ocria = OcriaNano.load(models);
const loadMs = performance.now() - t0;

const parser = LiteParse.withOcriaNano({
  ocrEnabled: true, outputFormat: "markdown", dpi: Number(args.dpi || 250), keepHeadersFooters: true,
  imageMode: "off", extractBlocks: true, quiet: true, continueOnPageError: true,
}, ocria);

let ids = fs.readFileSync(args.list, "utf8").split(/\s+/).filter(Boolean);
if (args.shard) {
  const [k, n] = args.shard.split("/").map(Number);
  ids = ids.filter((_, i) => i % n === k);
}
fs.mkdirSync(path.join(args.out, "pages"), { recursive: true });
let done = 0;
for (const id of ids) {
  const dst = path.join(args.out, "pages", `${id}.json`);
  if (fs.existsSync(dst)) continue;
  const rec = { page_id: id };
  const t = performance.now();
  try {
    const bytes = new Uint8Array(fs.readFileSync(path.join(args.pdf, `${id}.pdf`)));
    const res = await parser.parse(bytes);
    const pg = res.pages[0];
    rec.markdown = pg.markdown;
    rec.text = pg.text;
    rec.width = pg.width;
    rec.height = pg.height;
    rec.blocks = (pg.blocks || []).map((b) => ({
      kind: b.kind, bbox: b.bbox, text: b.text ?? null,
      lines: b.lines, header: b.header?.map((c) => c.text), rows: b.rows?.map((r) => r.map((c) => c.text)),
    }));
    rec.items = pg.textItems.length;
    if (args["dump-items"]) rec.text_items = pg.textItems.map((x) => ({ text: x.text, x: x.x, y: x.y, w: x.width, h: x.height, conf: x.confidence ?? null }));
    rec.ocr_items = pg.textItems.filter((x) => x.confidence !== undefined && x.confidence !== null).length;
    if (res.pageErrors?.length) rec.page_errors = res.pageErrors;
  } catch (e) {
    rec.error = String(e?.stack || e).slice(0, 1500);
  }
  rec.ms = Math.round(performance.now() - t);
  fs.writeFileSync(dst + ".tmp", JSON.stringify(rec));
  fs.renameSync(dst + ".tmp", dst);
  done++;
  if (done % 25 === 0) console.log(`${args.shard || ""} ${done}/${ids.length} ${((performance.now() - t0) / 1000).toFixed(0)}s`);
}
console.log(JSON.stringify({ shard: args.shard || null, done, total: ids.length, load_ms: Math.round(loadMs) }));
