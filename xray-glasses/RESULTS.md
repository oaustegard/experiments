# X-ray glasses: results

What the page does and how it works, for someone who has tried it: [`README.md`](README.md).
Live: https://austegard.com/fun-and-games/xray-glasses/

## Answer

How does the page know where to draw the bones, which way you are facing and how your head is turned,
and how close does it get? MediaPipe Pose Landmarker finds 33 points on a person and RTMPose finds 17 on
an animal; the page stretches a stock skeleton between them, so it shows how the subject stands and nothing
about their real bones.

On the two AP-10K test photos the animal joints land a mean 4.1 px from the hand-marked ones on an antelope <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k.jpg/mean_err_px]] -->
and 15.3 px on a jaguar that fills a crop 1365 px across. <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k_37516.jpg/mean_err_px]] -->
Facing toward or away comes from the pose model's depth estimate of the nose against the shoulders: right on 20/20 photos facing the camera <!-- @evidence[[results/facing.json#signals.nose_depth.front_called_front]] -->
and 6/7 from behind. <!-- @evidence[[results/facing.json#signals.nose_depth.back_called_back]] -->
The skull is posed by MediaPipe Face Landmarker and sits a median 0.41 cm from the face's landmarks. <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_median]] -->
The iOS camera background, at first rotated a quarter turn, is fixed and confirmed on an iPhone.

Frame rate and battery cost on phones are unmeasured; measuring them could change what runs on the GPU.

## Findings

1. **Facing direction comes from the pose model's depth estimate.** When a person faces the camera their
   nose is nearer than their shoulders; facing away, it is farther. That called 20/20 photos facing the camera <!-- @evidence[[results/facing.json#signals.nose_depth.front_called_front]] -->
   and 6/7 photos from behind. <!-- @evidence[[results/facing.json#signals.nose_depth.back_called_back]] -->
   Nose depth minus shoulder depth was never above -0.034 m facing the camera; the page switches to PA only above +0.05 m. <!-- @evidence[[results/facing.json#nose_depth_m.front_max]] -->
   The pose model's visibility scores are no help: they report the eyes and nose as visible from behind
   (0.99-1.0). The one miss is two people side by side with their backs to the camera, read as one person
   facing it. Hand-labelled set of 29 photos (`test/facing-signals.py`).
2. **The skull turns with the head because Face Landmarker reports the head's rotation.** Projecting the
   face model's standard face through that rotation lands a median 0.41 cm from the detected landmarks <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_median]] -->
   and 0.75 cm on the worst face, over heads turned -47 to +56 degrees. <!-- @evidence[[results/head_pose.json#face_fit.resid_cm_max]] -->
   When the face is too small for Face Landmarker, the page estimates the head from the nose and ears instead: a median 8.4 degrees off in yaw <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.hybrid.median_abs_err_deg.yaw]] -->
   and 3.2 degrees in roll. <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.hybrid.median_abs_err_deg.roll]] -->
   Using the pose model's 3D points alone, roll is 7.0 degrees off, so the page takes positions from the 2D points and only depth from the 3D ones. <!-- @evidence[[results/head_pose.json#pose_fallback_vs_face.world.median_abs_err_deg.roll]] -->
3. **On iOS 26 the camera background showed a quarter turn off; copying the frame through a canvas fixed
   it.** WebKit's `importExternalTexture`, the fast path from a video to WebGPU, ignores the camera's
   rotation, while the pose models read the frame upright. Every iOS browser uses WebKit. Drawing each
   frame into a 2D canvas and copying that into a texture is upright everywhere, confirmed on an iPhone.
   Also reported in [tfjs#8733](https://github.com/tensorflow/tfjs/issues/8733).
4. **The animal model's outputs needed no reinterpretation.** RTMPose returns x before y, and decodes at two bins per pixel. Swapping the outputs would put the antelope's joints 159.7 px off <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k.jpg/swapped_mean_err_px]] -->
   and decoding at 1.5 bins per pixel 151.2 px off. <!-- @evidence[[results/rtmpose_canvas.json#/images/ap10k.jpg/mean_err_by_split_ratio/1.5]] -->
   LiteRT.js in the browser returns the outputs in the same order as Python.
5. **The detector loaded slowly because MediaPipe's GPU delegate stalls on its first detection.** In a headless browser that first detection took 2770 ms on the GPU <!-- @evidence[[results/delegate_timing.json#detector_GPU.first_ms]] -->
   against 283 ms on the CPU, <!-- @evidence[[results/delegate_timing.json#detector_CPU.first_ms]] -->
   while downloading the 4.6 MB model took 550 ms. <!-- @evidence[[results/delegate_timing.json#detector_download.ms]] -->
   The detector now runs on the CPU. The pose model's GPU delegate shows the same first-call stall (2832 ms) and still runs on the GPU; a phone measurement will decide whether it moves too. <!-- @evidence[[results/delegate_timing.json#pose_GPU.first_ms]] -->
6. **A 3D skull can be drawn with the page's 2D bone shapes.** Seen along the beam, a 3D ellipsoid is
   exactly an ellipse and a 3D tube is a 2D tube, so the shaders needed no change. Run on a software GPU, the page's shaders match their CPU version to a mean 3.82 levels out of 255 per colour channel. <!-- @evidence[[results/gpu_vs_cpu.json#per_8bit_channel.human.mean]] -->
7. **For anyone testing WebGPU pages headless:** Dawn on Mesa lavapipe checks WGSL but crashes on any
   draw in these containers; headless Chromium on SwiftShader runs them, and its `importExternalTexture`
   needs `--use-angle=swiftshader` and `--use-vulkan=swiftshader` together (log 2).

## Method

- **Page under test**: `web/index.html`, inline JS and WGSL. Every test extracts code from it
  (`test/extract.js`) rather than copying it.
- **Animal pose**: RTMPose-m AP-10K LiteRT fp16 on the two mmpose AP-10K test images, with the
  ground-truth box standing in for the detector box. Crop, resampling and normalisation follow the page
  (`test/run-rtmpose.py`). Decoding uses the page's `decodeSimcc` (`test/verify-rtmpose.js`). Metric: mean
  Euclidean keypoint error in image pixels over annotated keypoints. Arms: page output order and swapped;
  split ratios 1-3; canvas-style and PIL resampling.
- **Facing**: 29 hand-labelled images: 20 facing the camera, including three-quarter views, from mmpose
  test data and MediaPipe assets; 9 facing away, from Wikimedia Commons and Human3.6M. Pose Landmarker lite
  and Face Landmarker as on the page. Three signals, each scored as front-called-front and
  back-called-back (`test/facing-signals.py`).
- **Head pose**: heads cropped from every person Pose Landmarker full finds in the mmpose test images, plus
  two MediaPipe portraits. Face fit: mean Euclidean distance between 24 projected canonical landmarks and the
  detected ones, in px and cm. Pose fallback: absolute yaw/pitch/roll difference from the face model's
  rotation (`test/head-pose-check.py`).
- **Rendering**: CPU port of both WGSL passes (`test/render-cpu.js`), against the page's WGSL executed in
  headless Chromium on SwiftShader (`test/render-gpu.js`). Metric: per-channel absolute difference.
  Skull alone at a pose grid: `test/render-skull.js`.
- **Start-up**: tasks-vision 1.1.0 in headless Chromium, SwiftShader GL. Model download time,
  `createFromOptions` and the first two detections per delegate (`test/delegate-timing.js`).
- **End to end**: the page in headless Chromium with SwiftShader WebGPU and a fake camera playing a still
  (`test/run-page.js`).
- **Controls**: the CPU port and GPU run share inputs. Swapped-output and alternative split-ratio decodes
  bound the RTMPose check. A deliberately broken shader fails `wgsl-check.js`, so that check is live.
- **Not covered**: any real GPU, any phone, frame rate.

## Log

### 1. First build (claude.ai sandbox, 2026-10-24)

No GPU, no camera, Hugging Face LFS blocked.

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

### 2. Handoff, issue #105 (Claude Code cloud container, 2026-10-24)

[#105](https://github.com/oaustegard/experiments/issues/105): self-host the animal model, verify its
output order, run the shaders, publish. Firecracker VM, no GPU, no camera.

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
  path is exercised too (since replaced by a copied texture, round 5). No WebGPU errors. Output: `test-frame-gpu.png`. Difference
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

### 3. Detector loading slowly (2026-10-24)

Oskar: the detector loads slowly for its 4.6 MB.

**Detector start-up** (after Oskar reported the detector loading slowly): the model
  downloads in 0.5 s, so its size was never the cost. The page created it on
  MediaPipe's GPU delegate, after the pose model had finished. On SwiftShader that
  delegate took 3.6-3.8 s for the first detection and ~0.95 s for later ones. The
  CPU delegate (XNNPACK, the int8 model's native path) took 1.1 s to init and 107 ms
  per detection. The page now creates the detector on the CPU and loads it alongside
  the pose model. Booting the whole page headless with a fake camera, the models
  were ready at 1.7 s instead of 6.2 s. SwiftShader overstates any GPU cost, so
  the size of the gain on a phone is unmeasured; the CPU path costs nothing at the
  rate the detector runs.

### 4. 3D skull, facing, face model (2026-10-24)

Oskar asked for a much better skull, whether facing forward or backward can be detected, and whether
there is a separate face model.

front/back detection and whether there is a separate face model):

- **Facing signals** (`test/facing-signals.py`): 20 front and 9 back views,
  hand-labelled. The front views are mmpose test images and MediaPipe assets,
  with three-quarter views counted as front and side views left out. The back
  views are Wikimedia Commons photos and one Human3.6M frame. Pose found a person
  in 27 of the 29; the two misses were back views.

  | signal | front called front | back called back |
  |---|---|---|
  | nose world depth minus shoulders' < 0 | 20/20 | 6/7 |
  | subject's left shoulder right of their right in the image | 19/20 | 6/7 |
  | Face Landmarker finds a face | 4/20 | 7/7 |

  Pose visibility scores were 0.99-1.0 for the nose, eyes and ears on every back
  view, so they say nothing. Front views had nose depths of -0.034 to -0.196 m;
  the six detected back views had +0.12 to +0.19 m. The miss is a photo of two
  women side by side from behind, which pose read as one person facing the
  camera. Small set; treat the thresholds as provisional.
- **Face-model head pose**: the facial transformation matrix's rotation, plus a
  least-squares scale and offset, put 24 canonical landmarks a mean 0.4-6.5 px
  from the detected ones (at most 0.4 cm; remeasured as distance in round 6). That held on 18 faces at 3-16 px/cm,
  with yaw from -47 to +56 degrees and pitch up to 20. The page reads
  `facialTransformationMatrixes[0].data` as column-major. With that, the skull on
  a head turned 42 degrees turns with the face. A transposed matrix would invert
  the turn; that was inferred, not tested.
- **Pose-only head pose**: against the face model on 17 heads, image positions
  of the nose and ears with world-landmark depth were off by a median 9.5 degrees
  in yaw, 4.9 in pitch and 3.2 in roll. World landmarks alone were 9.7 / 6.6 /
  7.0, worse in roll.
- **The page end to end** (`test/run-page.js`): headless Chromium, SwiftShader
  WebGPU and a fake camera playing a still. All four models load. The readout and
  marker were "AP view · face mesh", R right on the frontal portrait, the
  business photo and a 42-degree turned head. They were "PA view · pose", R left
  on a person walking away, and "AP view · pose" on the yoga photo, whose face is
  too small for the face detector. On the portrait the orbits sit on the eyes and
  the mandible on the jaw.
- `test-frame.png` and `test-frame-gpu.png` now draw the 3D skull, posed from the
  pose landmarks (the fixtures hold no face). The GPU frame is still within 3.8
  levels of the CPU port on the human panel.

### 5. Background rotated on iOS (2026-10-24)

Oskar saw the faint camera background rotated 90 degrees clockwise
on his phone while the skeleton was placed right. The background was the only part
that read the camera through `importExternalTexture`. On iOS 26, WebKit hands
that path the sensor frame without its rotation (same symptom in
[tfjs#8733](https://github.com/tensorflow/tfjs/issues/8733): WebGL and desktop
WebGPU fine, iOS WebGPU 90 degrees off). MediaPipe reads the frame its own way
and gets it upright. The page now draws each frame into a 2D canvas, which
applies the rotation in every browser, and copies that into an rgba8unorm
`texture_2d` with `copyExternalImageToTexture`. The shader's Sobel step is now in
texels of that texture. The GPU/CPU comparison is unchanged (3.8 / 8.4 / 5.7). The
page runs end to end headless with both a 960x540 and a 540x960 camera, upright
both times. The fix on an iPhone is not verified here.

### 6. Writeup: measurements moved into committed scripts (2026-10-24)

Rounds 3 and 4 measured with throwaway scripts. They are now `test/head-pose-check.py`,
`test/delegate-timing.js` and the results files that `facing-signals.py`, `verify-rtmpose.js` and
`render-gpu.js` write to `results/`. Reruns changed two readings:

- **Face-model fit.** Round 4 averaged per-axis error and reported "at most 0.4 cm" on 18 faces. Measured
  as Euclidean distance, on 20 faces (the two portraits included), the median is 0.41 cm and the worst
  0.75 cm (a 5.9 px residual on a small head at 8 px/cm). The pose fallback against it, now on 19 heads,
  is 8.4 / 4.9 / 3.2 degrees median (yaw / pitch / roll), against round 4's 9.5 / 4.9 / 3.2 on 17.
- **Delegate timing.** In a fresh run the GPU first detection took 2770 ms (round 3: 3.6-3.8 s), the CPU
  first detection 283 ms and the second 83 ms. The pose model's GPU delegate shows the same first-call
  stall (2832 ms against 121 ms on the CPU). It stays on the GPU because SwiftShader overstates GPU cost.
  A phone measurement should decide it.

The RTMPose, facing and GPU-versus-CPU numbers reproduced exactly.

### 7. iOS fix confirmed; writeups rewritten for readers (2026-10-24)

Oskar confirmed the camera background is upright on his iPhone with the canvas copy from round 5. He also
asked for writeups aimed at someone who has tried the page rather than at its build history.
`README.md` now explains the page in that order: how it finds a person or an animal, how the skeleton
and the 3D skull are drawn, what AP and PA mean, its limits, and the measured accuracy. The Answer and
Findings above were rewritten the same way; the numbers did not change.

### Open

- The Face Landmarker's cost per frame on a phone. It runs every frame while a
  head is larger than 30 px, alongside the pose.
- Anything on a real GPU: frame rate, the per-frame canvas copy on Safari, MediaPipe's
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
- The pose model's GPU delegate, which stalls on its first call on SwiftShader (round 6).
