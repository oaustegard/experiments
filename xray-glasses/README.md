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
  ├─ person ─> MediaPipe Pose Landmarker lite ─> 33 landmarks + world landmarks + segmentation mask
  │          ─> Face Landmarker (when the head is > 30 px) ─> head rotation, 478 landmarks, jawOpen
  │          ─> head pose: face-model rotation fitted to 24 face landmarks, else nose + ears of the pose
  └─ cat/dog ─> square crop from the box (x1.25) ─> RTMPose-m AP-10K ─> 17 keypoints
                (each crop is centred on the previous frame's keypoints; the detector re-runs every 6-8 frames)
  ─> One-Euro filter per coordinate (min cutoff 1.2 Hz, beta 0.02)
  ─> JS template builder: keypoints -> ~370-720 bone primitives (capsule / ellipse / rounded box);
     the human skull is a 3D model projected through the head pose
  ─> WebGPU pass 1: one instanced quad per primitive, SDF fragment shader,
     additive "density" + max "glow" into two r16float targets
  ─> WebGPU pass 2: full-screen composite: inverted luminance + Sobel edges in a cyan
     tint, soft tissue from the pose mask, bone density as 1-exp(-1.4 D), glow, grain,
     scanlines, vignette. The video frame is drawn into a 2D canvas (at most 640 px)
     and copied into a texture each frame; RESULTS.md (log 5) says why not texture_external.
```

Bone sizes are fractions of a body scale taken from the keypoints: for humans the
shoulder-to-hip distance (or 1.35x shoulder width when the hips are out of frame);
for animals the neck-to-tail-root distance. Long bones are tapered capsules with
epiphysis bulges at both ends. Their density is the chord length through a hollow
tube, so the cortex shows brighter than the marrow. The human template has a
3D skull (below), 6 cervical vertebrae below it, 17 thoracic
and lumbar vertebrae, 12 rib pairs with costal cartilage, sternum, clavicles,
scapulae, pelvis with obturator foramina, paired forearm and shank bones, carpals
and metacarpals with phalanges, and tarsals and metatarsals. The cat and dog
templates share a skeleton (spine with spinous processes, 13 rib pairs, sternebrae,
scapulae, pelvis, limbs, paws) and differ in skull shape, orbit size, bone
thickness, chest depth and tail length and curl. AP-10K has no tail-tip keypoint,
so the tail is a guess that sways.

The human skull is modelled in 3D, in centimetres, in the frame of MediaPipe's
canonical face model (x toward the subject's left, y up, z out of the face). It
is about 150 primitives. The vault is a solid ellipsoid minus a slightly smaller
one, so it is brightest where the beam runs tangent to the bone, as on a film.
Then the brow ridges, the frontal and maxillary sinuses, and orbits with
rounded-square rims. Then the nasal bones, the pear-shaped aperture and the
septum. Then the cheekbones and zygomatic arches, the maxilla and hard palate,
and two arches of 16 teeth with crowns and roots. Then the mandible with its rami,
condyles and coronoid processes. Last, the skull base: the petrous ridges, clivus,
sella turcica, mastoids, ear canals and the occipital protuberance. Each frame a
head pose maps it to the image orthographically: an ellipsoid projects to an
exact ellipse and a capsule to a capsule, so the existing 2D primitives and
shaders draw it unchanged. The mandible and lower teeth hinge about the condyles
by up to 26 degrees, driven by the face model's `jawOpen` blendshape.
`skull-sheet.png` shows it alone at yaw 0-180 degrees, pitched, and with the jaw
open.

The head pose comes from the Face Landmarker when it finds this person's face: its
rotation matrix, plus a least-squares scale and offset over 24 landmarks. Otherwise
it comes from the pose: the image positions of the nose and ears, with depth
from the world landmarks. Head pose smooths with the same One-Euro filter as the
joints.

Controls: Auto / Human / Cat / Dog, Flip camera (front/back), Soft tissue on/off.
Readouts: render fps and smoothed model time per frame. For a person, the view as
a radiograph names it: AP (facing the camera), PA (facing away) or lateral (head
turned more than ~70 degrees), and whether the skull follows the face mesh or the
pose. The R lead marker sits on the subject's right side as the landmarks place
it, so it changes sides when they turn around.

Facing away is read from the pose's world landmarks: the nose's depth minus the
shoulders'. With hysteresis, the page switches to PA above +0.05 m and back to AP
below -0.02 m. A face that the face model finds always means AP. MediaPipe Pose
labels left and right correctly from behind (the shoulders swap sides in the
image), so the skeleton needs no change for PA. On a radiograph, AP and PA images
of the same skeleton are nearly mirror images.

## Models

| Stage | Model | Size | Licence | Served from |
|---|---|---|---|---|
| Human pose | MediaPipe Pose Landmarker lite, float16 v1 | 5.8 MB | Apache-2.0 | storage.googleapis.com/mediapipe-models (CORS `*`) |
| Detector (Auto, cat/dog box) | EfficientDet-Lite0 int8 v1, COCO | 4.6 MB | Apache-2.0 | storage.googleapis.com/mediapipe-models |
| Head pose and jaw | MediaPipe Face Landmarker, float16 v1 (BlazeFace short-range + 478-point mesh + blendshapes) | 3.8 MB | Apache-2.0 | storage.googleapis.com/mediapipe-models; optional, the page runs without it |
| Animal pose | RTMPose-m trained on AP-10K, LiteRT fp16 (`litert-community/RTMPose-Animal-AP10K-LiteRT` at commit `1daee634`) | 27.5 MB | Apache-2.0 (mmpose); AP-10K dataset | `./rtm_animal_fp16.bin` beside the page (self-hosted copy, sha256 `7fbfc03239cebc1f657efab722849bc9b30d36d3d4371118aed2aaf4041e742f`, equal to the LFS oid at that commit); Hugging Face as fallback |
| Runtimes | `@mediapipe/tasks-vision@1.1.0`, `@litertjs/core@2.5.3` (+ `@litertjs/wasm-utils@2.5.3`) | ~13 MB wasm each | Apache-2.0 | cdn.jsdelivr.net |

The animal model is fetched only when a cat or dog is first needed. The page
starts with the human models and the detector only.

Why these choices:
- MediaPipe 1.1.1 is the current `latest`, but it was published 2026-10-14, less
  than two weeks ago. The page pins 1.1.0 (2026-10-06).
  LiteRT.js 2.6.0 is from 2026-10-22, so the page pins 2.5.3 (2026-07-17), the
  version used for the edge-compat browser measurement cited in RESULTS.md (Open).
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

The page lives at https://austegard.com/fun-and-games/xray-glasses/, published
from the site repo (`oaustegard.github.io`, `fun-and-games/xray-glasses/`) with
the model file beside it. This directory is the source and holds the tests: edit
`web/index.html` here, then copy `web/index.html` and `web/rtm_animal_fp16.bin`
there. It was first published at `/experiments/xray-glasses/web/`, which no
longer exists; the experiments index now links to the site page. The link
preview, `images/xray-glasses-og.jpg` in the site repo, is `test/pose.jpg`
blended into its own X-ray. It was rendered by the page's shaders on SwiftShader
(`test/render-gpu.js`).

Camera access needs a normal https page. It would not work as a claude.ai
artifact: that sandbox refuses the camera without a declared capability, and its
CSP blocks the model fetches from storage.googleapis.com and huggingface.co.

## Results

Measurements, what was verified and how, and what is still open: [`RESULTS.md`](RESULTS.md). Every
number there comes from a script in `test/` writing to `results/`.

## Files

- `web/index.html`: the page. Inline JS and WGSL, no build step.
- `web/rtm_animal_fp16.bin`: the RTMPose-m AP-10K LiteRT model (27.5 MB).
- `test/extract.js`: pulls the inline JS and WGSL blocks out of the page.
- `test/wgsl-check.js`: Dawn compile plus pipeline validation (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json node wgsl-check.js`; needs `mesa-vulkan-drivers`).
- `test/render-cpu.js`: CPU port of the shaders; writes `test-frame.png`.
- `test/render-gpu.js`: runs the page's shaders in headless Chromium on SwiftShader; writes `test-frame-gpu.png` and the CPU/GPU difference.
- `test/run-rtmpose.py`, `test/verify-rtmpose.js`: run RTMPose with the page's preprocessing, decode with the page's decoder, score against ground truth; write `test/rtm_keypoints.json`.
- `test/render-skull.js`: the 3D skull alone at a grid of head poses; writes `skull-sheet.png`.
- `test/facing-signals.py`: front/back signals on a labelled image set (downloads into `test/facing-cache/`).
- `test/run-page.js`: drives the page in headless Chromium with a fake camera; prints the readout, saves a screenshot.
- `test/make-fixtures.py`: downloads the test images and regenerates the landmark, mask and ground-truth fixtures.
- `test/pose_landmarks.json`, `test/pose_mask.json` (16-level run-length text), `test/ap10k_gt.json`, `test/rtm_keypoints.json`: fixtures.
- `test-frame.png`: CPU-rendered test frame. `test-frame-gpu.png`: the same frame from the WGSL on SwiftShader. `skull-sheet.png`: the skull at yaw 0, 30, 60, 90, 135 and 180 degrees; rows level, pitched 20 degrees, jaw open.

To reproduce: `cd test && npm i && pip install mediapipe==0.10.32 ai-edge-litert pillow && python3 make-fixtures.py && python3 run-rtmpose.py && node verify-rtmpose.js && node render-cpu.js && node render-gpu.js`.

Test images: `pose.jpg` from MediaPipe's test assets
(storage.googleapis.com/mediapipe-assets); `ap10k.jpg` and `ap10k_37516.jpg` from
mmpose `tests/data/ap10k` (Apache-2.0 repository, AP-10K images).
