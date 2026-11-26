// Capture the page raster LiteParse hands to OCR (a JS ocrEngine sees it as PNG), at 250 DPI.
import fs from "node:fs";
import path from "node:path";
const pkg = path.resolve("pkg");
const { default: init, LiteParse } = await import(path.join(pkg, "liteparse_wasm.js"));
await init({ module_or_path: fs.readFileSync(path.join(pkg, "liteparse_wasm_bg.wasm")) });
const ids = fs.readFileSync("ref40.txt", "utf8").split(/\s+/).filter(Boolean);
for (const id of ids) {
  let got = null;
  const eng = { async recognize(png, w, h) { got = { png: new Uint8Array(png), w, h }; return []; } };
  const p = new LiteParse({ ocrEnabled: true, outputFormat: "text", dpi: 250, quiet: true, imageMode: "off", ocrEngine: eng });
  await p.parse(new Uint8Array(fs.readFileSync(`pdf/${id}.pdf`)));
  if (got) fs.writeFileSync(`out/${id}.png`, got.png);
  console.log(id, got ? `${got.w}x${got.h} ${got.png.length}` : "no OCR call");
}
