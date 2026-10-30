// MediaPipe Hand Landmarker off the main thread, so a 10-30 ms detection never holds up the audio blocks.
// The page posts { bitmap, t } (an upright camera frame and its capture time in ms); the worker answers with
// { t, hands: [{ tip: [x, y], box: [x0, y0, x1, y1], score }], ms } in 0..1 frame coordinates, or { error }.
// Fingertip = landmark 8 (index finger tip); box = all 21 landmarks.
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
let lm = null, ready = null, lastT = -1;

async function init(delegate) {
  const mp = await import(MP + "/vision_bundle.mjs");
  const files = await mp.FilesetResolver.forVisionTasks(MP + "/wasm");
  // In a module worker MediaPipe's loader falls back to import(), which keeps the classic script's ModuleFactory in
  // module scope and then fails "ModuleFactory not set". Evaluating the loader globally first sets it where it looks.
  if (!self.ModuleFactory) (0, eval)(await (await fetch(files.wasmLoaderPath)).text());
  const opts = (d) => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
  try { lm = await mp.HandLandmarker.createFromOptions(files, opts(delegate)); return delegate; }
  catch (e) { if (delegate === "CPU") throw e; lm = await mp.HandLandmarker.createFromOptions(files, opts("CPU")); return "CPU"; }
}

onmessage = async ({ data }) => {
  if (data.type === "init") {
    ready = init(data.delegate || "GPU").then((d) => postMessage({ type: "ready", delegate: d }), (e) => postMessage({ type: "error", error: String(e?.message || e) }));
    return;
  }
  const { bitmap, t } = data;
  try {
    await ready;
    if (!lm) { bitmap.close(); return postMessage({ type: "hands", t, hands: [], ms: 0 }); }
    const t0 = performance.now(), ts = Math.max(lastT + 1, Math.round(t)); lastT = ts;
    const r = lm.detectForVideo(bitmap, ts);
    bitmap.close();
    const hands = (r.landmarks || []).map((L, k) => {
      let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
      for (const p of L) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
      return { tip: [L[8].x, L[8].y], box: [x0, y0, x1, y1], score: r.handedness?.[k]?.[0]?.score ?? 0 };
    });
    postMessage({ type: "hands", t, hands, ms: performance.now() - t0 });
  } catch (e) { bitmap.close?.(); postMessage({ type: "error", error: String(e?.message || e) }); }
};
