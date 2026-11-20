// End-to-end check of the PDF text-layer front end in headless Chromium (SwiftShader WebGPU), on
// web/samples/born-digital.pdf (run/make_textlayer_pdf.py). Needs network for jsDelivr (pdf.js, layout engine, models).
//   node run/check_textlayer.mjs        writes results/textlayer_check.json
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync, writeFileSync } from "node:fs";
import { looseCer } from "../web/pdftext.js";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const truth = JSON.parse(readFileSync(join(root, "results/textlayer_truth.json"), "utf8"));
const srv = spawn("python3", ["-m", "http.server", "8766", "--bind", "127.0.0.1"], { cwd: join(root, ".."), stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const checks = [];
const summary = {};
const check = (name, pass, detail) => { checks.push({ name, pass, detail }); console.log(`${pass ? "ok  " : "FAIL"} ${name}${detail !== undefined ? ": " + JSON.stringify(detail) : ""}`); };
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("http://127.0.0.1:8766/tesseract-gpu/web/", { waitUntil: "load" });
  await page.waitForFunction(() => /ready|error/.test(document.getElementById("status").className), null, { timeout: 180000 });
  let run = 0;
  async function step(action) {
    await action();
    run += 1;
    await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n || document.getElementById("status").classList.contains("error"), run, { timeout: 300000 });
    const lines = await page.$$eval("#lines li", (els) => els.map((e) => ({ src: e.dataset.src, text: e.querySelector(".t").textContent })));
    const timings = await page.$$eval("#timings tr", (rs) => rs.map((r) => r.textContent));
    return { lines, timings, status: await page.textContent("#status-text") };
  }
  const best = (want, lines) => Math.min(...lines.map((l) => looseCer(want, l.text)));

  // page 1: columns from the layer, the picture's lines OCR'd
  const p1 = await step(() => page.setInputFiles("#file", join(root, "web/samples/born-digital.pdf")));
  console.log(p1.status, "\n ", p1.timings.join(" | "));
  const l1 = p1.lines.filter((l) => l.src === "layer"), o1 = p1.lines.filter((l) => l.src === "ocr");
  Object.assign(summary, { p1_layer_lines: l1.length, p1_ocr_lines: o1.length, p1_columns_exact: `${truth.page1.columns.filter((w) => best(w, l1) === 0).length}/${truth.page1.columns.length}` });
  check("p1 every column line comes from the layer, exactly", truth.page1.columns.every((w) => best(w, l1) === 0), truth.page1.columns.map((w) => best(w, l1)));
  check("p1 the two columns are separate lines", l1.length >= 9, l1.length);
  check("p1 text inside the picture is OCR'd", o1.filter((l) => l.text !== "no text").length >= 3, o1.slice(0, 4).map((l) => l.text));
  check("p1 no OCR line duplicates a layer line", o1.every((o) => best(o.text, l1) > 0.3), o1.map((o) => Number(best(o.text, l1).toFixed(2))));

  // page 2: the wrong invisible layer is rejected, the page OCR'd
  const p2 = await step(() => page.selectOption("#pdfpage", "2"));
  console.log(p2.status);
  const m2 = /median CER (\d+)% on (\d+) lines/.exec(p2.status);
  if (m2) Object.assign(summary, { p2_spot_median_cer: Number(m2[1]) / 100, p2_spot_lines: Number(m2[2]) });
  check("p2 wrong layer rejected", /disagrees with the page image/.test(p2.status), p2.status);
  check("p2 every line OCR'd", p2.lines.length > 5 && p2.lines.every((l) => l.src === "ocr"), p2.lines.length);
  check("p2 no wrong-layer text in output", p2.lines.every((l) => best(l.text, truth.page2.wrong_layer.map((t) => ({ text: t }))) > 0.5));

  // page 3: text only, no OCR, the superscript kept on its line
  const p3 = await step(() => page.selectOption("#pdfpage", "3"));
  console.log(p3.status, "\n ", p3.timings.join(" | "));
  summary.p3_ocr_lines = p3.lines.filter((l) => l.src === "ocr").length;
  check("p3 all lines from the layer", p3.lines.length === truth.page3.lines.length && p3.lines.every((l) => l.src === "layer"), p3.lines.map((l) => l.text));
  check("p3 superscript joins its line", p3.lines.some((l) => l.text === truth.page3.superscript_after + "1" || l.text === truth.page3.superscript_after + " 1"), p3.lines[1] && p3.lines[1].text);
  check("p3 no recognition rows in timings", !p3.timings.some((t) => /Recognition/.test(t)), p3.timings);

  // modes: OCR every page, then text only on page 2
  const p3o = await step(() => page.selectOption("#pdfmode", "ocr"));
  check("ocr mode reads page 3 by OCR", p3o.lines.length > 0 && p3o.lines.every((l) => l.src === "ocr"), p3o.lines.map((l) => l.text));
  check("ocr mode reads page 3 well", truth.page3.lines.every((w) => best(w, p3o.lines) < 0.1), truth.page3.lines.map((w) => Number(best(w, p3o.lines).toFixed(3))));
  await page.selectOption("#pdfmode", "text"); run += 1;
  await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n, run, { timeout: 300000 });
  const p2t = await step(() => page.selectOption("#pdfpage", "2"));
  check("text mode trusts even the wrong layer", p2t.lines.length === truth.page2.wrong_layer.length && p2t.lines.every((l) => l.src === "layer"), p2t.lines.length);

  // an image is unaffected
  await page.selectOption("#pdfmode", "auto"); run += 1;
  await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n, run, { timeout: 300000 });
  const im = await step(() => page.setInputFiles("#file", join(root, "web/samples/english.png")));
  check("image input: all OCR, PDF control hidden", im.lines.every((l) => l.src === "ocr") && (await page.isHidden("#pdfmodes")), im.lines.length);
  check("no page errors", !errors.length, errors);
} finally {
  await browser.close();
  srv.kill();
}
const pass = checks.every((c) => c.pass);
summary.checks = `${checks.filter((c) => c.pass).length}/${checks.length}`;
writeFileSync(join(root, "results/textlayer_check.json"), JSON.stringify({ pass, summary, checks }, null, 1) + "\n");
console.log(pass ? "\nALL PASS" : "\nSOME FAILED");
process.exit(pass ? 0 : 1);
