// OCRia-nano-1's OCR on WebGPU with no LiteParse runtime: the WGSL engine (engine.js) runs the nets, and
// page/ocr.js (a JS port of LiteParse's ocria page logic) does everything around them.
//
//   import { loadOcriaGpu } from "https://austegard.com/experiments/ocria-pack/web/gpu/ocria-gpu.js";
//   const ocr = await loadOcriaGpu({ scripts: ["world", "rtl", "cjk"] });
//   const lines = await ocr.read(canvas);   // [{ text, bbox: [x0,y0,x1,y1], confidence, polygon }], page pixels
//
// Lines only: tables and Markdown need LayoutNet and LiteParse's layout code, which this path leaves out.
import { OcriaGpu } from "./engine.js";
import { ocrPage } from "../page/ocr.js";

export async function loadOcriaGpu(opts = {}) {
  const base = (opts.base || new URL("./models/", import.meta.url).href).replace(/\/?$/, "/");
  const scripts = (opts.scripts || []).filter((s) => ["world", "rtl", "cjk"].includes(s));
  const f = opts.fetch || fetch;
  const t0 = performance.now();
  const eng = await OcriaGpu.load(base, { models: ["det", "router", "rec_latin", ...scripts.map((s) => "rec_" + s)],
                                          device: opts.device, fetch: f });
  const json = async (p) => (await f(base + p)).json();
  const charsets = { latin: await json("rec_latin/charset.json") };
  for (const s of scripts) charsets[s] = await json(`rec_${s}/charset.json`);
  if (charsets.rtl) charsets.rtl = { chars: charsets.rtl, visual_rtl: !!(await json("rec_rtl/meta.json")).visual_rtl };
  const detMeta = await json("det/meta.json");
  const infer = (model, shape, data, outputs) => eng.infer(model, shape, data, outputs);
  return {
    name: "OCRia-nano-1 (WebGPU)", engine: eng, scripts, loadMs: performance.now() - t0,
    /** Read one page: a canvas, an ImageData, or {data, w, h, ch} with ch 1, 3 or 4. */
    async read(src) {
      let img = src;
      if (src.getContext) {
        const d = src.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, src.width, src.height);
        img = { data: d.data, w: d.width, h: d.height, ch: 4 };
      } else if (src.width && src.data && !src.w) {
        img = { data: src.data, w: src.width, h: src.height, ch: 4 };
      }
      return ocrPage(img, infer, charsets, { detMeta });
    },
  };
}
