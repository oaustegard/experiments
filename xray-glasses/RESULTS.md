# X-ray glasses: results

[X-ray glasses](https://austegard.com/fun-and-games/xray-glasses/) is a web page that films you, or your
cat or dog, and draws a moving X-ray-style skeleton over the picture, entirely in the browser. This
writeup asks how it knows where the bones go, which way you are facing and how your head is turned, and
how accurate each part is. [`README.md`](README.md) walks through how the page is built.

## Answer

Two pose models find joints, MediaPipe Pose Landmarker 33 points on a person and RTMPose 17 on a cat or
dog, and the page stretches a stock skeleton between them. It tells which way a person faces from the
pose model's depth estimates, since the nose is nearer the camera than the shoulders when you face it. A
third model, MediaPipe Face Landmarker, reports the head's rotation and how far the mouth is open; the
page turns the 3D skull and opens its jaw to match.

On two animal photos with hand-marked joints, in a best case, the model's joints land a mean 1% of the animal's length away on an antelope <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k.jpg/mean_err_pct_of_length]] -->
and 1.4% on a jaguar. <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k_37516.jpg/mean_err_pct_of_length]] -->
Facing direction was called right on 20/20 photos of people facing the camera. <!-- @evidence[[results/facing.json#signals.nose_depth.front_called_front]] -->
From behind it was right on 6/9: in two the pose model found no one, and one showed two people side by side, read as one person facing the camera. <!-- @evidence[[results/facing.json#signals.nose_depth.back_called_back_of_all_labelled]] -->
The skull sits a median 0.41 cm from the face Face Landmarker detects; how closely that face follows the real head was not tested. <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_median]] -->

Not measured: frame rate and battery use on a phone, and how closely the drawn jaw follows the real one.

## Findings

1. **On two test photos, the animal joints land a mean 1-1.4% of the animal's length from the marked
   ones.** In each photo 16 of the 17 joints are marked; one eye is not. The antelope, 429 px long, gets a mean miss of 4.1 px; <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k.jpg/mean_err_px]] -->
   the jaguar, 1092 px long, 15.3 px, <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k_37516.jpg/mean_err_px]] -->
   with its worst joint, the neck, 54.8 px off. <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k_37516.jpg/max_err_px]] -->
   Length here is the longer side of the hand-drawn box around the animal. These are best cases: the test
   cut the picture the model sees from that hand-drawn box, while the live page cuts it from its detector's
   box or the previous frame's joints, which fit less tightly.
   Two photos is a small sample (Round 2).
2. **The page tells facing the camera from facing away by where the nose is in depth.** Facing the camera, the nose was always at least 3.4 cm nearer than the shoulders; <!-- @evidence[[results/facing.json#nose_depth_m.front_nose_lead_min_cm]] -->
   from behind it was 12 to 19 cm farther. The page switches to PA (the radiograph label for a back view)
   once the nose is 5 cm behind the shoulders and back to AP (front view) once it is 2 cm in front, so its
   own thresholds give the same calls: 20 of 20 photos facing the camera, and 6 of 9 from behind. Of the
   three misses from behind, the pose model found no one in two, and read a photo of two people side by
   side as one person facing the camera. The pose model's visibility scores cannot do this job; they rate
   the eyes and nose 0.99-1.0 visible from behind. The page's third label, lateral, for a head turned
   side-on, was not tested (Round 4, `results/facing.json`).
3. **The skull sits on Face Landmarker's face to within 0.4 cm at the median and 0.75 cm at worst.**
   Face Landmarker fits a standard face to every face it finds and reports its rotation. Turning the page's skull by that rotation and scaling it to the face puts it a median 0.41 cm from the detected landmarks <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_median]] -->
   and 0.75 cm on the worst of 20 faces, turned between 47 degrees left and 56 degrees right. <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_max]] -->
   This measures how well the skull follows Face Landmarker, not how well Face Landmarker follows a real
   head. From across a room the face is too small for it, and the page estimates the head from the nose and ears instead: a median 8.4 degrees off Face Landmarker in how far the head is turned, <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.hybrid.median_abs_err_deg.yaw]] -->
   4.9 degrees in tilt <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.hybrid.median_abs_err_deg.pitch]] -->
   and 3.2 degrees in roll (Round 6). <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.hybrid.median_abs_err_deg.roll]] -->
4. **The detector can mistake a dog for a cat.** On one photo of cats and dogs it labelled both dogs
   "cat". In Auto mode that draws the cat skeleton; the Cat and Dog buttons override the label (Round 1).
5. **The page starts tracking about 1.4 s after Start is pressed.** That is with the pose model, the detector and Face Landmarker loading at once, measured in a headless browser with a software GPU (Round 6). <!-- @evidence[[results/startup.json#/working copy/tracking_starts_s]] -->
   Without Face Landmarker it is 1.2 s; <!-- @evidence[[results/startup.json#/be1f2f8b/tracking_starts_s]] -->
   before the detector moved off the GPU and started loading alongside the pose model, that was 2.4 s. <!-- @evidence[[results/startup.json#/93559fcb/tracking_starts_s]] -->
   A phone may differ in either direction.

## Method

Every test drives code extracted from the page itself (`test/extract.js`).

- **Animal joints**: the two AP-10K photos that ship with mmpose, the library RTMPose comes from, with
  16 marked joints each. The crop, resizing and colour normalisation follow the page
  (`test/run-rtmpose.py`); decoding uses the page's own decoder (`test/verify-rtmpose.js`). Metric: mean
  distance from the marked joints in image pixels. The crop starts from the marked box, not the
  detector's. Controls: swapping the model's two outputs puts the antelope's joints 159.7 px off, and
  decoding at the wrong resolution (1.5 instead of 2 bins per pixel) 151.2 px off, so the page reads the
  model correctly; LiteRT.js in a browser returns the outputs in the same order as Python. A PIL-resized
  crop and Chromium's own canvas crop give the same accuracy within 1 px.
- **Facing**: 29 photos labelled by hand, 20 facing the camera (three-quarter views included, side views
  left out) from mmpose test data and MediaPipe's examples, 9 from behind from Wikimedia Commons and
  Human3.6M. Pose Landmarker lite and Face Landmarker as the page runs them; each candidate signal scored
  as front views called front and back views called back (`test/facing-signals.py`).
- **Head pose**: heads cropped from every person found in the mmpose test photos, plus two MediaPipe
  portraits; Face Landmarker found a face in 20 of the 30. Skull fit: mean distance between the page's
  24 reference points on the turned skull and the detected face landmarks. Fallback: the difference in
  turn, tilt and roll from Face Landmarker's rotation (`test/head-pose-check.py`).
- **Start-up**: the page booted in headless Chromium with a software GPU (SwiftShader) and a fake camera
  showing a still photo, timing each start-up step, for the page before and after the detector change
  and as it is now (`test/startup-time.js`). Per-model download and first-detection times:
  `test/delegate-timing.js`.
- **Rendering**: a CPU copy of the page's two shader passes (`test/render-cpu.js`) against the shaders
  themselves on SwiftShader (`test/render-gpu.js`); `test/run-page.js` drives the whole page with a fake
  camera.
- **Not covered**: any real phone or GPU, frame rate, battery, the lateral label, the jaw's accuracy.

## Log

### Round 1: First build (claude.ai sandbox, 2026-10-24)

No GPU, no camera, and Hugging Face's file downloads were blocked.

- The page's scripts were extracted and pass `node --check`.
- Pose Landmarker lite on `test/pose.jpg` drove the page's skeleton builder: 583 bone shapes, all finite,
  all inside the frame.
- The animal-model decoder turns synthetic outputs back into the exact pixel.
- The One-Euro smoothing filter cut frame-to-frame jitter on a still point with ±3 px noise from 1.97 px
  to 0.36 px.
- EfficientDet-Lite0 found the person in `pose.jpg` (score 0.96) and the cat in MediaPipe's `cat.jpg`
  (0.66). On a photo of cats and dogs it labelled both dogs "cat" (0.57-0.61).
- Animal model choice: RTMPose-m AP-10K as LiteRT is the smallest quadruped pose model found that runs in
  a browser. ViTPose++-s APT-36K as ONNX is 97 MB; DeepLabCut's SuperAnimal-Quadruped ships PyTorch
  weights only; the YOLOv8-pose AP-10K repositories found have no weights.

### Round 2: Self-hosted model and real checks (issue #105, Claude Code cloud container, 2026-10-24)

[#105](https://github.com/oaustegard/experiments/issues/105) asked for the animal model to be
self-hosted, its outputs checked against the page's decoder, the shaders run and the page published.
No GPU or camera here either.

- `web/rtm_animal_fp16.bin` is the model file at Hugging Face commit `1daee634`; its sha256 matches the
  one Hugging Face lists.
- RTMPose through the page's crop and decoder, against the marked joints:

  | image | crop side | mean miss | worst joint | outputs swapped |
  |---|---|---|---|---|
  | antelope (`ap10k.jpg`) | 536 px | 4.1 px | 10.5 px | 159.7 px |
  | jaguar (`ap10k_37516.jpg`) | 1365 px | 15.3 px | 54.8 px (neck) | 425.5 px |

  Decoding at 1, 1.5, 2, 2.5 and 3 bins per pixel gave 456, 151, 4.1, 93 and 154 px on the antelope.
  LiteRT.js 2.5.3 in headless Chromium matched Python's first output to 7e-6. The page needed no change.
- `test-frame.png` now draws the cat and dog skeletons from RTMPose's joints instead of the marked ones.
- WGSL: Dawn (npm `webgpu` 0.6.2) on Mesa's lavapipe compiles both shaders and builds both pipelines, but
  crashes on any draw. Headless Chromium on SwiftShader runs them; its `importExternalTexture` needs
  `--use-angle=swiftshader` and `--use-vulkan=swiftshader` together.
- The shaders on SwiftShader against their CPU copy, mean difference per colour channel out of 255: 3.8
  (person), 8.4 (antelope), 5.7 (jaguar). Most of it is the film grain, which is random by design (±6),
  and edges, where the CPU copy samples the picture more coarsely than the GPU.

### Round 3: Slow detector (2026-10-24)

Oskar: the detector loads slowly for a 4.6 MB model. Its download took 0.5 s. The page created it on
MediaPipe's GPU path and only after the pose model had loaded; on SwiftShader the GPU path's first
detection took 3.6-3.8 s, against about 0.1 s per detection on the CPU. The detector moved to the CPU and
now loads alongside the pose model. A one-off boot of the whole page took 6.2 s before and 1.7 s after;
Round 6 re-measured it with a committed script.

### Round 4: 3D skull, facing detection, face model (2026-10-24)

Oskar asked for a better skull, whether facing toward or away can be detected, and whether there is a
separate face model.

- The skull became a 3D model of about 150 parts in centimetres, turned with the head each frame. Seen
  along the beam, a 3D ellipsoid is exactly an ellipse and a 3D tube is a 2D tube, so the existing
  shaders draw it unchanged. `skull-sheet.png` shows it from front to back, tilted and with the jaw open.
- MediaPipe Face Landmarker (3.8 MB) was added. It is optional and loads alongside the other models. Its
  rotation matrix is column-major in the browser API.
- Facing: the nose-depth signal and two others scored on the labelled photos; the table is finding 2's
  source. The page shows AP, PA or lateral and moves the R marker to the subject's right.
- Head pose, measured at the time as a per-axis average: within about 0.4 cm on 18 faces; the
  nose-and-ears fallback 9.5 degrees median off in turn.
- The page ran end to end headless on a frontal portrait, a business portrait, a head turned 42 degrees
  (all "AP, face mesh"), a person walking away ("PA, pose", R marker on the left) and the yoga photo,
  whose face is too small for Face Landmarker ("AP, pose").

### Round 5: Background rotated on iOS (2026-10-24)

Oskar saw the camera background turned 90 degrees clockwise on his phone, with the skeleton in the
right place. The background was the only part read through `importExternalTexture`. The page now draws
each frame into a canvas of at most 640 px and copies that into a texture. Headless runs with a landscape
and a portrait camera were both upright.

### Round 6: Measurements moved into committed scripts (2026-10-24)

Rounds 3 and 4 had measured with throwaway scripts. Now `test/head-pose-check.py`,
`test/delegate-timing.js` and `test/startup-time.js` write to `results/`, as do `facing-signals.py`,
`verify-rtmpose.js` and `render-gpu.js`. The animal, facing and rendering numbers came out the same.
Three changed:

- Skull fit, measured as a distance instead of a per-axis average and on 20 faces: median 0.41 cm, worst
  0.75 cm. The nose-and-ears fallback on 19 heads: 8.4 degrees median off in turn.
- The GPU path's first detection took 2770 ms (Round 3: 3.6-3.8 s), the CPU path's 283 ms. The pose
  model's GPU path stalls the same way on its first call (2832 ms) and still runs on the GPU.
- Start-up from Start to tracking: 2.4 s before the detector change, 1.2 s after, 1.4 s with Face
  Landmarker. Round 3's 6.2 s and 1.7 s were a single run.

### Round 7: iOS fix confirmed (2026-10-24)

Oskar confirmed the camera background is upright on his iPhone.

### Open

- Frame rate, battery use and Face Landmarker's cost per frame on a phone.
- Whether the pose model should also leave the GPU path, given its first-call stall.
- The lateral label, which no test photo covered.
- LiteRT.js on a phone's GPU for the animal model, and whether its outputs keep the same order there.
- Camera start, switching cameras and permission prompts on real devices.
