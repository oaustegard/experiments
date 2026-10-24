# X-ray glasses

A single web page that opens the camera, finds a person, cat or dog, and draws a
live X-ray view: the camera feed rendered as a radiograph, with a skeleton fitted
to the subject's pose and following it as they move. Page: `web/index.html`.

The skeleton is a template drawn to fit the detected joints. It is not a scan and
shows nothing about the real anatomy in front of the camera.

## Pipeline

```
camera (getUserMedia, 960x540 ideal)
  ├─ Auto mode: EfficientDet-Lite0 (COCO, allowlist person/cat/dog, CPU delegate), every 8-20 frames
  ├─ person ─> MediaPipe Pose Landmarker lite ─> 33 landmarks + segmentation mask
  └─ cat/dog ─> square crop from the box (x1.25) ─> RTMPose-m AP-10K ─> 17 keypoints
                (each crop is centred on the previous frame's keypoints; the detector re-runs every 6-8 frames)
  ─> One-Euro filter per coordinate (min cutoff 1.2 Hz, beta 0.02)
  ─> JS template builder: keypoints -> ~370-590 bone primitives (capsule / ellipse / rounded box)
  ─> WebGPU pass 1: one instanced quad per primitive, SDF fragment shader,
     additive "density" + max "glow" into two r16float targets
  ─> WebGPU pass 2: full-screen composite: inverted luminance + Sobel edges in a cyan
     tint, soft tissue from the pose mask, bone density as 1-exp(-1.4 D), glow, grain,
     scanlines, vignette. The video frame comes in as a texture_external.
```

Bone sizes are fractions of a body scale taken from the keypoints: for humans the
shoulder-to-hip distance (or 1.35x shoulder width when the hips are out of frame);
for animals the neck-to-tail-root distance. Long bones are tapered capsules with
epiphysis bulges at both ends. Their density is the chord length through a hollow
tube, so the cortex shows brighter than the marrow. The human template has a
skull, orbits, nasal aperture, teeth, mandible, 7 cervical vertebrae, 17 thoracic
and lumbar vertebrae, 12 rib pairs with costal cartilage, sternum, clavicles,
scapulae, pelvis with obturator foramina, paired forearm and shank bones, carpals
and metacarpals with phalanges, and tarsals and metatarsals. The cat and dog
templates share a skeleton (spine with spinous processes, 13 rib pairs, sternebrae,
scapulae, pelvis, limbs, paws) and differ in skull shape, orbit size, bone
thickness, chest depth and tail length and curl. AP-10K has no tail-tip keypoint,
so the tail is a guess that sways.

Controls: Auto / Human / Cat / Dog, Flip camera (front/back), Soft tissue on/off.
Readouts: render fps and smoothed model time per frame. The R/L lead marker
marks the subject's right side.

## Models

| Stage | Model | Size | Licence | Served from |
|---|---|---|---|---|
| Human pose | MediaPipe Pose Landmarker lite, float16 v1 | 5.8 MB | Apache-2.0 | storage.googleapis.com/mediapipe-models (CORS `*`) |
| Detector (Auto, cat/dog box) | EfficientDet-Lite0 int8 v1, COCO | 4.6 MB | Apache-2.0 | storage.googleapis.com/mediapipe-models |
| Animal pose | RTMPose-m trained on AP-10K, LiteRT fp16 (`litert-community/RTMPose-Animal-AP10K-LiteRT` at commit `1daee634`) | 27.5 MB | Apache-2.0 (mmpose); AP-10K dataset | `./rtm_animal_fp16.bin` beside the page (self-hosted copy, sha256 `7fbfc03239cebc1f657efab722849bc9b30d36d3d4371118aed2aaf4041e742f`, equal to the LFS oid at that commit); Hugging Face as fallback |
| Runtimes | `@mediapipe/tasks-vision@1.1.0`, `@litertjs/core@2.5.3` (+ `@litertjs/wasm-utils@2.5.3`) | ~13 MB wasm each | Apache-2.0 | cdn.jsdelivr.net |

The animal model is fetched only when a cat or dog is first needed. The page
starts with the human models and the detector only.

Why these choices:
- MediaPipe 1.1.1 is the current `latest`, but it was published 2026-10-14, less
  than two weeks ago. The page pins 1.1.0 (2026-10-06).
  LiteRT.js 2.6.0 is from 2026-10-22, so the page pins 2.5.3 (2026-07-17), the
  version used for the edge-compat browser measurement below.
- No ONNX export of an AP-10K or AnimalPose model could be fetched from this
  sandbox: mmpose's download.openmmlab.com and the Xet CDN that Hugging Face
  redirects LFS files to are blocked here. The LiteRT RTMPose-m AP-10K conversion
  is the smallest browser-proven quadruped model I found. Its card reports a full
  WebGPU delegation in LiteRT.js 2.5.3 (Chromium 151, M4 Max, 6.2 ms p50, output
  matches CPU). The wasm/XNNPACK path ran at 336 ms. Other options were larger or
  not browser-ready: ViTPose++-s APT-36K ONNX is 97 MB; DeepLabCut
  SuperAnimal-Quadruped ships PyTorch weights only (RTMPose-s 24 MB `.pt`, plus a
  separate detector); the YOLOv8-pose AP-10K repos found have no weights.
- The model is committed as `web/rtm_animal_fp16.bin`, a byte-for-byte copy of
  `rtm_animal_fp16.tflite`, because pages_site.py ships `.bin` files under 50 MB and
  not `.tflite`. The page tries that path first.

Prior art: searches turned up many MediaPipe pose demos that draw stick figures.
None found renders an anatomical X-ray skeleton, and none handles animals.

## Hosting notes

The page is meant for https://austegard.com/experiments/xray-glasses/web/, where
camera access works. It would not work as a claude.ai artifact: that sandbox
refuses the camera without a declared capability, and its CSP blocks the model
fetches from storage.googleapis.com and huggingface.co.

## Verification

First pass (claude.ai sandbox, no GPU, no camera, Hugging Face LFS blocked):

- **JS syntax**: the inline classic script and the module script were extracted
  (`test/extract.js`) and checked with `node --check`. Both pass.
- **Landmark-to-bone mapping on real model output**: MediaPipe Pose Landmarker
  lite (the same `.task` file the page loads, run through Python mediapipe 0.10.32)
  ran on `test/pose.jpg`. Its 33 landmarks and segmentation mask drove the page's
  own skeleton builder, extracted from `index.html` rather than copied. The result
  was 583 primitives, all values finite, all anchors in frame.
- **SimCC decoding**: synthetic heads decode back to the exact pixel (argmax/2).
- **One-Euro filter**: on a static point with ±3 px noise at 30 fps, mean
  frame-to-frame movement fell from 1.97 px to 0.36 px.
- **Detector**: EfficientDet-Lite0 found the person in `pose.jpg` (0.96) and the cat
  in MediaPipe's `cat.jpg` (0.66). On a cats-and-dogs photo it labelled both dogs
  "cat" (0.57-0.61), so Auto mode can pick the wrong species template. The Cat and
  Dog buttons override the label.

Second pass ([#105](https://github.com/oaustegard/experiments/issues/105), Claude Code
cloud container, Firecracker VM, no GPU, no camera):

- **RTMPose on the AP-10K test images**: `test/run-rtmpose.py` runs the committed
  `.bin` through `ai-edge-litert` with the page's crop (square, 1.25x the box, black
  outside the frame, 256 px) and normalisation (ImageNet mean/std, RGB, NCHW). The
  crop is resampled the way canvas `drawImage` does it, plain bilinear with no
  low-pass filter. The AP-10K ground-truth box stands in for the EfficientDet box. `test/verify-rtmpose.js`
  decodes the heads with the page's own `decodeSimcc` and maps them back as
  `runAnimal()` does. Mean keypoint error against ground truth, over the 16
  annotated keypoints of each image:

  | image | crop side | mean error | max error | swapped order |
  |---|---|---|---|---|
  | `ap10k.jpg` (antelope) | 536 px | **4.1 px** | 10.5 px | 159.7 px |
  | `ap10k_37516.jpg` (jaguar) | 1365 px | **15.3 px** | 54.8 px (neck) | 425.5 px |

  One crop pixel is 2.1 and 5.3 image pixels respectively, so both are about 2-3 crop
  pixels on average. Peak scores run 0.86-1.49, so the page's 0.3 threshold keeps
  all 17. The resampling barely matters: the crop Chromium's own `drawImage` makes
  from the same call (mean difference 0.35 and 4.0 levels from the emulation) gives
  4.2 and 15.2 px, and PIL's antialiased resize (`python3 run-rtmpose.py pil`) gives
  4.3 and 14.3 px.
- **Output order and split ratio**: the model has two `[1, 17, 512]` outputs,
  `output_0` and `output_1` in the signature and graph order. `output_0` is
  simcc_x, as the page assumes: swapping them puts keypoints 160-430 px off.
  Decoding at split ratios 1, 1.5, 2, 2.5 and 3 gives mean errors of 456, 151,
  4.1, 93 and 154 px on the antelope, so the ratio is 2. LiteRT.js 2.5.3 itself
  (wasm accelerator, headless Chromium) returns the outputs in the same order:
  its `outs[0]` matches Python's `output_0` to 7e-6 on the same input. The page
  needed no change.
- **test-frame.png** now drives the cat and dog templates with these RTMPose
  keypoints and scores (`test/rtm_keypoints.json`) instead of ground truth. Top:
  human (MediaPipe). Bottom left: dog template on the antelope. Bottom right: cat
  template on the jaguar. It is still the CPU port of the shaders.
- **WGSL**: `test/wgsl-check.js` passes on Dawn (npm `webgpu` 0.6.2) over Mesa 25.2.8
  lavapipe: both modules compile with no messages and both pipelines are created.
  Copy work completes here (a buffer write maps back), unlike the claude.ai sandbox.
  Any draw crashes the process: a bare one-triangle pass segfaults, and the full
  render aborts with the same "futex facility returned an unexpected error code"
  seen there. `LP_NUM_THREADS=0` or `1` turns the abort into a segfault.
- **The shaders, executed**: `test/render-gpu.js` runs both passes, unmodified, in
  headless Chromium with SwiftShader as the WebGPU adapter, with the page's
  pipelines, blend states, formats, uniforms and bind groups. The frame goes
  through `importExternalTexture` as a `VideoFrame`, so the `texture_external`
  path is exercised too. No WebGPU errors. Output: `test-frame-gpu.png`. Difference
  from the CPU port per 8-bit channel:

  | panel | mean | p50 | p99 |
  |---|---|---|---|
  | human | 3.8 | 3 | 15 |
  | dog | 8.4 | 6 | 41 |
  | cat | 5.7 | 4 | 33 |

  The p50 is the grain term (±6 levels, computed with f32 `sin` on the GPU and f64
  in JS). Of the pixels off by more than 30 in any channel, 78-93% sit on an image
  edge: the CPU port reads luminance nearest-neighbour at the ±1.5 px Sobel taps,
  while the shader samples bilinearly. The dog panel has the most because the grass
  is edges throughout. SwiftShader timings (0.2-0.5 s per frame) say nothing about a
  real GPU.

- **Detector start-up** (after Oskar reported the detector loading slowly): the model
  downloads in 0.5 s, so its size was never the cost. The page created it on
  MediaPipe's GPU delegate, after the pose model had finished. On SwiftShader that
  delegate took 3.6-3.8 s for the first detection and ~0.95 s for later ones. The
  CPU delegate (XNNPACK, the int8 model's native path) took 1.1 s to init and 107 ms
  per detection. The page now creates the detector on the CPU and loads it alongside
  the pose model. Booting the whole page headless with a fake camera, the models
  were ready at 1.7 s instead of 6.2 s. SwiftShader overstates any GPU cost, so
  the size of the gain on a phone is unmeasured; the CPU path costs nothing at the
  rate the detector runs.

Not tested, needs real hardware:

- Anything on a real GPU: frame rate, `importExternalTexture` on Safari, MediaPipe's
  GPU delegate, the LiteRT.js WebGPU compile of the animal model and whether its
  output order matches the wasm path, and its latency on phones.
- Camera start and flip, portrait layout on a real phone, permission prompts.
- Frame rate. Nothing here was measured. Expected order of magnitude, not a
  measurement: render cost is small (hundreds of instanced quads plus one
  full-screen pass), so on a recent phone the render should hold 60 fps and
  tracking should run at camera rate (30 fps) in human mode. Animal mode adds
  RTMPose-m, which edge-compat measured at 6.2 ms on an M4 Max and which LiteRT
  reports at ~5 ms on Android GPU via native LiteRT. The browser figure on a phone
  is unknown. Pose latency appears live in the top-right readout.

## Files

- `web/index.html`: the page. Inline JS and WGSL, no build step.
- `web/rtm_animal_fp16.bin`: the RTMPose-m AP-10K LiteRT model (27.5 MB).
- `test/extract.js`: pulls the inline JS and WGSL blocks out of the page.
- `test/wgsl-check.js`: Dawn compile plus pipeline validation (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json node wgsl-check.js`; needs `mesa-vulkan-drivers`).
- `test/render-cpu.js`: CPU port of the shaders; writes `test-frame.png`.
- `test/render-gpu.js`: runs the page's shaders in headless Chromium on SwiftShader; writes `test-frame-gpu.png` and the CPU/GPU difference.
- `test/run-rtmpose.py`, `test/verify-rtmpose.js`: run RTMPose with the page's preprocessing, decode with the page's decoder, score against ground truth; write `test/rtm_keypoints.json`.
- `test/make-fixtures.py`: downloads the test images and regenerates the landmark, mask and ground-truth fixtures.
- `test/pose_landmarks.json`, `test/pose_mask.json` (16-level run-length text), `test/ap10k_gt.json`, `test/rtm_keypoints.json`: fixtures.
- `test-frame.png`: CPU-rendered test frame. `test-frame-gpu.png`: the same frame from the WGSL on SwiftShader.

To reproduce: `cd test && npm i && pip install mediapipe==0.10.32 ai-edge-litert pillow && python3 make-fixtures.py && python3 run-rtmpose.py && node verify-rtmpose.js && node render-cpu.js && node render-gpu.js`.

Test images: `pose.jpg` from MediaPipe's test assets
(storage.googleapis.com/mediapipe-assets); `ap10k.jpg` and `ap10k_37516.jpg` from
mmpose `tests/data/ap10k` (Apache-2.0 repository, AP-10K images).
