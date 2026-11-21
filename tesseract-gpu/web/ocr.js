// Tesseract on WebGPU for other pages: an image or canvas in, its text lines out.
//
//   const { createReader } = await import("https://austegard.com/experiments/tesseract-gpu/web/ocr.js");
//   const reader = await createReader();                  // English fine-tuned for scans (5.2 MB, 8-bit)
//   const { items, meanConf } = await reader.read(canvas);
//   // items: [{ text, confidence, box: { x, y, width, height } }] in page pixels, Tesseract's reading order
//
// The same pipeline as this experiment's demo page: Tesseract's layout (tesseract.js-core, tesslayout.js), box clean-up
// (pagepipe.js), line flat-field and tight crop (lineprep.js), Tesseract's LSTM on WebGPU with its beam search
// (engine.js), and lines under MIN_CONF dropped as junk. Needs WebGPU; createReader throws without it.
import { Gpu, recognize } from "./engine.js";
import { loadTraineddata } from "./traineddata.js";
import { tessLayout } from "./tesslayout.js";
import { cleanLine } from "./lineprep.js";
import { MIN_CONF, refineBoxes, tidy, cropOf } from "./pagepipe.js";

const HERE = new URL("./", import.meta.url);
// dict: the model's word lists in the beam search; they cost the fine-tuned model 0.6 points (results/ft_eval.json).
export const MODELS = {
  "eng-ft": { name: "English, fine-tuned for scans", url: new URL("models/eng-ft/fast/eng.traineddata", HERE).href, dict: false },
  "eng-ft-best": { name: "English, fine-tuned for scans (float)", url: new URL("models/eng-ft/best/eng.traineddata", HERE).href, dict: false },
};

function grayOf(source) {
  const w = source.width, h = source.height;
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(source, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const g = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = (rgba[4 * i] * 4899 + rgba[4 * i + 1] * 9617 + rgba[4 * i + 2] * 1868 + 8192) >> 14;
  return { w, h, data: g };
}

export async function createReader({ model = "eng-ft" } = {}) {
  if (!navigator.gpu) throw new Error("Tesseract on WebGPU needs a browser with WebGPU");
  const spec = MODELS[model];
  if (!spec) throw new Error(`unknown model ${model}`);
  const gpu = await Gpu.create();
  const tm = await loadTraineddata(gpu, spec.url);
  return {
    name: spec.name,
    async read(source) {
      const gray = grayOf(source);
      const layout = await tessLayout(gray.data, gray.w, gray.h);
      const boxes = refineBoxes(gray, layout.lines.map((l) => l.box));
      const idx = [], crops = [];
      boxes.forEach((b, i) => { const c = cropOf(gray, b); if (c) { idx.push(i); crops.push(cleanLine(c)); } });
      const out = crops.length ? await recognize(gpu, tm, crops, { dict: spec.dict }) : [];
      const items = [];
      idx.forEach((i, k) => {
        const r = out[k];
        const text = r ? tidy((r.text || "").trim()) : "";
        if (!text || r.conf == null || r.conf < MIN_CONF) return;
        const [x0, y0, x1, y1] = boxes[i];
        items.push({ text, confidence: r.conf, box: { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } });
      });
      const meanConf = items.length ? items.reduce((a, it) => a + it.confidence, 0) / items.length : 0;
      return { items, meanConf, ms: out.ms || null };
    },
  };
}
