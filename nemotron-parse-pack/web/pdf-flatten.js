/* PDF pages to flattened canvases, as anything-to-text renders them
   (oaustegard.github.io web-utilities/anything-to-text.html: PDF_DOC_OPTS, renderPdfOnce, renderPdfCanvas).

   - pdf.js 6 decodes JBIG2 and JPEG 2000 images with WebAssembly fetched from wasmUrl. Without it those images
     are skipped with only a console warning, so a compressed scan (MRC: a JPEG background with the text in a
     JBIG2 mask on top) renders as the background alone.
   - AnnotationMode.ENABLE paints every annotation's appearance stream into the canvas, so filled-in form fields
     and FreeText notes that were never flattened appear as a viewer shows them.

   openPdf(fileOrBytes) -> {numPages, page(n, {longEdge, dpi}) -> canvas, destroy()} */
import * as pdfjsLib from "https://esm.sh/pdfjs-dist@6.0.227";

pdfjsLib.GlobalWorkerOptions.workerSrc = "https://esm.sh/pdfjs-dist@6.0.227/build/pdf.worker.min.mjs";
const ASSETS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.0.227/";   /* must match the version imported above */
const DOC_OPTS = {
  wasmUrl: ASSETS + "wasm/",
  cMapUrl: ASSETS + "cmaps/", cMapPacked: true,
  standardFontDataUrl: ASSETS + "standard_fonts/",
  iccUrl: ASSETS + "iccs/",
};

/* Render at a scale that gives the page's long edge `longEdge` pixels (the way the benchmark and the model's
   processor size a page), or at `dpi` when given. White background: pdf.js leaves transparent pixels otherwise. */
async function renderPage(doc, n, { longEdge = 2048, dpi = null } = {}) {
  const page = await doc.getPage(n);
  const base = page.getViewport({ scale: 1 });
  const scale = dpi ? dpi / 72 : longEdge / Math.max(base.width, base.height);
  const viewport = page.getViewport({ scale });
  const cv = document.createElement("canvas");
  cv.width = Math.round(viewport.width); cv.height = Math.round(viewport.height);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  await page.render({ canvasContext: ctx, viewport, annotationMode: pdfjsLib.AnnotationMode.ENABLE }).promise;
  page.cleanup();
  return cv;
}

export async function openPdf(src) {
  const data = src instanceof Uint8Array ? src : new Uint8Array(await src.arrayBuffer());
  const doc = await pdfjsLib.getDocument({ data, ...DOC_OPTS }).promise;
  return { numPages: doc.numPages, page: (n, opts) => renderPage(doc, n, opts), destroy: () => doc.destroy() };
}
