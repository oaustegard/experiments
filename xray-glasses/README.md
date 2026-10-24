# X-ray glasses

A single web page that opens the camera, finds a person, cat or dog, and draws a
live X-ray view: the camera feed rendered as a radiograph, with a skeleton fitted
to the subject's pose and following it as they move. Page: `web/index.html`.

The skeleton is a template drawn to fit the detected joints. It is not a scan and
shows nothing about the real anatomy in front of the camera.

## Pipeline

```
camera (getUserMedia, 960x540 ideal)
  ├─ Auto mode: EfficientDet-Lite0 (COCO, allowlist person/cat/dog), every 8-20 frames
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
| Animal pose | RTMPose-m trained on AP-10K, LiteRT fp16 (`litert-community/RTMPose-Animal-AP10K-LiteRT` at commit `1daee634`) | 27.5 MB | Apache-2.0 (mmpose); AP-10K dataset | `./rtm_animal_fp16.bin` beside the page if present, else Hugging Face (allows the austegard.com origin) |
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
- At 27.5 MB the model is over the 15 MB limit for a claude.ai artifact supporting
  file, so it was not put in the repo. pages_site.py ships `.bin` files under 50 MB.
  To serve it from austegard.com, copy `rtm_animal_fp16.tflite` to
  `web/rtm_animal_fp16.bin`; the page tries that path first.

Prior art: searches turned up many MediaPipe pose demos that draw stick figures.
None found renders an anatomical X-ray skeleton, and none handles animals.

## Hosting notes

The page is meant for https://austegard.com/experiments/xray-glasses/web/, where
camera access works. It would not work as a claude.ai artifact: that sandbox
refuses the camera without a declared capability, and its CSP blocks the model
fetches from storage.googleapis.com and huggingface.co.

## Verification

Done in a container with no GPU and no camera:

- **JS syntax**: the inline classic script and the module script were extracted
  (`test/extract.js`) and checked with `node --check`. Both pass.
- **WGSL**: both shader modules compile with Dawn (npm `webgpu` 0.6.2 on Mesa
  lavapipe) with no errors or warnings. Both render pipelines were created with
  the page's exact descriptors: r16float targets, additive and max blend,
  bgra8unorm output (`test/wgsl-check.js`). A deliberately broken shader fails the
  same check, so the validator is live. GPU execution was not possible: queue
  work never completes under lavapipe in this sandbox.
- **Landmark-to-bone mapping on real model output**: MediaPipe Pose Landmarker
  lite (the same `.task` file the page loads, run through Python mediapipe 0.10.32)
  ran on `test/pose.jpg`. Its 33 landmarks and segmentation mask drove the page's
  own skeleton builder, extracted from `index.html` rather than copied. The result
  was 583 primitives, all values finite, all anchors in frame.
- **Quadruped templates**: driven by AP-10K ground-truth keypoints from two mmpose
  test images (an antelope driving the dog template, a jaguar driving the cat
  template). RTMPose itself was not run here: the model file could not be
  downloaded.
- **test-frame.png**: a CPU port of both WGSL passes (`test/render-cpu.js`,
  same SDFs, blend rules and composite maths) rendered the three cases above. Top:
  human. Bottom left: dog template. Bottom right: cat template. This is a CPU
  rendering of the shader maths, not a WebGPU frame.
- **SimCC decoding**: synthetic heads decode back to the exact pixel (argmax/2).
- **One-Euro filter**: on a static point with ±3 px noise at 30 fps, mean
  frame-to-frame movement fell from 1.97 px to 0.36 px.
- **Detector**: EfficientDet-Lite0 found the person in `pose.jpg` (0.96) and the cat
  in MediaPipe's `cat.jpg` (0.66). On a cats-and-dogs photo it labelled both dogs
  "cat" (0.57-0.61), so Auto mode can pick the wrong species template. The Cat and
  Dog buttons override the label.

Not tested, needs real hardware:

- Anything on a GPU: the actual WebGPU frame, frame rate, `importExternalTexture`
  on Safari, MediaPipe's GPU delegate, LiteRT.js WebGPU compile of the animal
  model, its output order (assumed simcc_x then simcc_y, per the model card) and
  its latency on phones.
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
- `test/extract.js`: pulls the inline JS and WGSL blocks out of the page.
- `test/wgsl-check.js`: Dawn compile plus pipeline validation (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json node wgsl-check.js`).
- `test/render-cpu.js`: CPU port of the shaders; writes `test-frame.png`.
- `test/make-fixtures.py`: regenerates the landmark, mask and ground-truth fixtures.
- `test/pose_landmarks.json`, `test/pose_mask.json` (16-level run-length text), `test/ap10k_gt.json`: fixtures written by `make-fixtures.py`.
- `test-frame.png`: the CPU-rendered test frame.

The repo copy holds the text files only. The three test JPEGs, `test-frame.png`,
`pose_mask.json` (28 KB of generated run-length data) and `package-lock.json` were
left out because the commit path used here accepts only text sent inline.
`make-fixtures.py` downloads the JPEGs and writes the mask again. To reproduce:
`cd test && npm i && python3 make-fixtures.py && node render-cpu.js`.
Regenerating the fixtures this way gave a byte-identical `test-frame.png`.

Test images: `pose.jpg` from MediaPipe's test assets
(storage.googleapis.com/mediapipe-assets); `ap10k.jpg` and `ap10k_37516.jpg` from
mmpose `tests/data/ap10k` (Apache-2.0 repository, AP-10K images).
