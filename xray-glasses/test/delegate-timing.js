// Where does MediaPipe's start-up time go? In headless Chromium (SwiftShader for WebGL/WebGPU, so GPU numbers are
// upper bounds, not phone numbers): download time of the detector and pose models, then createFromOptions and
// the first two detections on the CPU and GPU delegates. Writes ../results/delegate_timing.json.
// usage: node delegate-timing.js   (models and tasks-vision@1.1.0 come from their CDNs through $HTTPS_PROXY)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const page = `<!doctype html><script type="module">
const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0";
const DET = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite";
const POSE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const out = {}; const T = () => performance.now(); const r = (v) => Math.round(v);
try {
  let t = T(); const mp = await import(MP + "/vision_bundle.mjs"); const files = await mp.FilesetResolver.forVisionTasks(MP + "/wasm"); out.runtime_ms = r(T() - t);
  t = T(); const det = new Uint8Array(await (await fetch(DET)).arrayBuffer()); out.detector_download = { bytes: det.length, ms: r(T() - t) };
  t = T(); const pose = new Uint8Array(await (await fetch(POSE)).arrayBuffer()); out.pose_download = { bytes: pose.length, ms: r(T() - t) };
  const img = new ImageData(640, 480);
  for (const delegate of ["CPU", "GPU"]) {
    t = T(); const d = await mp.ObjectDetector.createFromOptions(files, { baseOptions: { modelAssetBuffer: det, delegate }, runningMode: "VIDEO", scoreThreshold: 0.35, maxResults: 5, categoryAllowlist: ["person", "cat", "dog"] });
    const init = T() - t; t = T(); d.detectForVideo(img, 1); const f1 = T() - t; t = T(); d.detectForVideo(img, 2); const f2 = T() - t; d.close();
    out["detector_" + delegate] = { init_ms: r(init), first_ms: r(f1), second_ms: r(f2) };
    t = T(); const p = await mp.PoseLandmarker.createFromOptions(files, { baseOptions: { modelAssetBuffer: pose, delegate }, runningMode: "VIDEO", numPoses: 1 });
    const pi = T() - t; t = T(); p.detectForVideo(img, 1); out["pose_" + delegate] = { init_ms: r(pi), first_ms: r(T() - t) }; p.close();
  }
} catch (e) { out.error = String(e && e.stack || e); }
window.R = out;
</script>`;
(async () => {
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: 'localhost' } : undefined;
  const b = await chromium.launch({ proxy, args: ['--ignore-certificate-errors', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await (await b.newContext({ ignoreHTTPSErrors: true })).newPage();
  await p.route('http://localhost/', r => r.fulfill({ contentType: 'text/html', body: page }));
  await p.goto('http://localhost/');
  const res = await (await p.waitForFunction(() => window.R, null, { timeout: 280000 })).jsonValue();
  await b.close();
  res.adapter = 'SwiftShader (headless Chromium); 640x480 blank frame';
  fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
  fs.writeFileSync(path.join(__dirname, '..', 'results', 'delegate_timing.json'), JSON.stringify(res, null, 1));
  console.log(JSON.stringify(res, null, 1));
})().catch(e => { console.error(e.message); process.exit(1); });
