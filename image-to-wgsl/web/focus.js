// Focus maps for foveated encoding: where in the image the encoder should spend its splats.
// Each returns a Float32Array of W*H weights in [0, 1], to pass to trainer.js encode() as opts.focus.
// Everything runs in the browser; the image never leaves the page (the models are downloaded, the pixels are not uploaded).
//
//   const f = await focusMap("point", img, {point: {x: 0.5, y: 0.3}, radius: 0.2});
//   const f = await focusMap("face", img);                       // null when no face is found
//   const d = await depthMap(img);                               // normalised depth, 1 = nearest, 0 = farthest
//   const f = await focusMap("depth", img, {depth: 0.8, width: 0.15});
//
// img is {rgba, width, height} (rgba: Uint8ClampedArray or Uint8Array, W*H*4).

// Pinned CDN versions (both published 2026-10-06, about four weeks before this file).
export const MEDIAPIPE_VERSION = "1.1.0";
export const TRANSFORMERS_VERSION = "4.3.1";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}`;
export const DEPTH_MODEL = "onnx-community/depth-anything-v2-small";

export async function focusMap(mode, img, opts = {}) {
  if (mode === "point") return pointMap(img, opts);
  if (mode === "face") return faceMap(img, opts);
  if (mode === "depth") return depthFocus(img, opts);
  throw new Error(`unknown focus mode ${mode}`);
}

// ------------------------------------------------------------------ point

// Gaussian around opts.point ({x, y} in 0..1); opts.radius is its sigma as a fraction of the shorter side.
function pointMap({ width: W, height: H }, { point = { x: 0.5, y: 0.5 }, radius = 0.2 } = {}) {
  const s = Math.max(1e-6, radius) * Math.min(W, H), cx = point.x * W, cy = point.y * H, k = -0.5 / (s * s);
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const dy = y + 0.5 - cy;
    for (let x = 0; x < W; x++) { const dx = x + 0.5 - cx; out[y * W + x] = Math.exp(k * (dx * dx + dy * dy)); }
  }
  return out;
}

// ------------------------------------------------------------------ face

let faceDetector = null;           // Promise<FaceDetector>, loaded once per page
function loadFaceDetector(minConfidence) {
  faceDetector ??= (async () => {
    const { FaceDetector, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    return FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
      runningMode: "IMAGE", minDetectionConfidence: minConfidence,
    });
  })().catch((e) => { faceDetector = null; throw e; });
  return faceDetector;
}

// Face boxes in pixels, [{x, y, w, h, score}].
export async function detectFaces(img, { minConfidence = 0.5 } = {}) {
  const det = await loadFaceDetector(minConfidence);
  await det.setOptions({ minDetectionConfidence: minConfidence });
  const data = new ImageData(new Uint8ClampedArray(img.rgba.buffer, img.rgba.byteOffset, img.width * img.height * 4).slice(),
                             img.width, img.height);
  const r = det.detect(data);
  return (r.detections ?? []).map((d) => ({ x: d.boundingBox.originX, y: d.boundingBox.originY,
    w: d.boundingBox.width, h: d.boundingBox.height, score: d.categories?.[0]?.score ?? null }));
}

// Union (max) of soft ellipses around each face box. opts.scale: ellipse semi-axes over half the box (1.4 covers
// the hairline and chin the detector's box leaves out); opts.soft: edge width as a fraction of the semi-axis.
async function faceMap(img, { scale = 1.4, soft = 0.25, minConfidence = 0.5, faces = null } = {}) {
  const boxes = faces ?? await detectFaces(img, { minConfidence });
  if (!boxes.length) return null;
  const W = img.width, H = img.height, out = new Float32Array(W * H);
  for (const b of boxes) {
    const cx = b.x + b.w / 2, cy = b.y + b.h * 0.45, ax = b.w / 2 * scale, ay = b.h / 2 * scale * 1.15;
    const x0 = Math.max(0, Math.floor(cx - ax * (1 + 4 * soft))), x1 = Math.min(W, Math.ceil(cx + ax * (1 + 4 * soft)));
    const y0 = Math.max(0, Math.floor(cy - ay * (1 + 4 * soft))), y1 = Math.min(H, Math.ceil(cy + ay * (1 + 4 * soft)));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const u = (x + 0.5 - cx) / ax, v = (y + 0.5 - cy) / ay, r = Math.sqrt(u * u + v * v);
      const t = Math.max(0, r - 1) / soft, w = Math.exp(-0.5 * t * t);    // 1 inside, Gaussian falloff outside
      const i = y * W + x; if (w > out[i]) out[i] = w;
    }
  }
  return out;
}

// ------------------------------------------------------------------ depth

let depthPipe = null;              // Promise<pipeline>, loaded once per page
function loadDepth({ device = "wasm", dtype = "q8" } = {}) {
  depthPipe ??= (async () => {
    const { pipeline } = await import(TRANSFORMERS_URL);
    return pipeline("depth-estimation", DEPTH_MODEL, { device, dtype });
  })().catch((e) => { depthPipe = null; throw e; });
  return depthPipe;
}

const depthCache = new WeakMap();  // img -> Promise<Float32Array>, so a re-pick of the depth does not rerun the model

// Monocular depth, bilinearly resampled to W x H and normalised to 0..1 (1 = nearest, 0 = farthest).
// Depth Anything predicts relative inverse depth, so 0.5 is halfway in disparity, not in metres.
export function depthMap(img, opts = {}) {
  if (!depthCache.has(img)) depthCache.set(img, computeDepth(img, opts).catch((e) => { depthCache.delete(img); throw e; }));
  return depthCache.get(img);
}

async function computeDepth(img, opts) {
  const pipe = await loadDepth(opts);
  const { RawImage } = await import(TRANSFORMERS_URL);
  const W = img.width, H = img.height;
  const raw = new RawImage(new Uint8ClampedArray(img.rgba.buffer, img.rgba.byteOffset, W * H * 4).slice(), W, H, 4);
  const { predicted_depth: t } = await pipe(raw);
  const dims = t.dims, h = dims[dims.length - 2], w = dims[dims.length - 1], src = Float32Array.from(t.data);
  const out = new Float32Array(W * H);
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < H; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * h / H - 0.5)), y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * w / W - 0.5)), x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      const v = (src[y0 * w + x0] * (1 - tx) + src[y0 * w + x1] * tx) * (1 - ty) + (src[y1 * w + x0] * (1 - tx) + src[y1 * w + x1] * tx) * ty;
      out[y * W + x] = v; if (v < lo) lo = v; if (v > hi) hi = v;
    }
  }
  const k = hi > lo ? 1 / (hi - lo) : 0;
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) * k;
  return out;
}

// Weight exp(-(d - depth)^2 / (2 width^2)) on the normalised depth map: in focus at one distance.
async function depthFocus(img, { depth = 1, width = 0.15, ...rest } = {}) {
  const d = await depthMap(img, rest), out = new Float32Array(d.length), k = -0.5 / (Math.max(1e-6, width) ** 2);
  for (let i = 0; i < d.length; i++) { const e = d[i] - depth; out[i] = Math.exp(k * e * e); }
  return out;
}
