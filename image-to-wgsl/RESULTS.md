# image-to-wgsl: a photograph as one WGSL shader

The question, from Oskar on 2026-11-02, was whether an image could be compressed into a WGSL program, aiming for novelty over practicality. The model was the image-to-svg skill, which traces an image into flat-colour paint regions and writes them out as SVG paths.

The codec built here stores an image as an ordered list of soft, rotated ellipses, called splats, packed into an integer array inside one fragment shader. The shader decodes the list and paints the splats per pixel. The compressed size is the gzip size of the shader source after minification. A prefix is the first *n* splats of the list. Training makes every power-of-two prefix a usable image, so the shader can paint itself in, coarse to fine.

A later question, from Oskar on 2026-11-03 ([#108](https://github.com/oaustegard/experiments/issues/108)), was whether zooming into the shader could add real detail from a larger source image, as deep-zoom image viewers do with tile pyramids. That part uses six 2040-pixel DIV2K photographs instead of the Kodak set, which has nothing larger than 768×512.

The test images are six photographs from the Kodak set, 768×512 each. Quality is PSNR in dB against the original. Live gallery: [austegard.com/experiments/image-to-wgsl/web/](https://austegard.com/experiments/image-to-wgsl/web/).

## Answer

On the six Kodak photographs, the 1,024-splat shader reaches a median 26.11 dB at a median 11,450 bytes. <!-- @evidence[[results/headline.json#median_psnr.1024]] --> <!-- @evidence[[results/headline.json#median_bytes.1024]] -->
At that byte count, AVIF is 5.6 dB better (median over the six images). <!-- @evidence[[results/headline.json#gap_vs_best_db.1024]] -->
JPEG, encoded at half, quarter or eighth size and upscaled, is 1.72 dB better. <!-- @evidence[[results/headline.json#gap_vs_jpeg_down_db.1024]] -->
It trails every codec tested at every length from 16 to 1,024 splats.

Three properties hold. First, prefix training costs 0.4 dB. <!-- @evidence[[results/headline.json#prefix_cost_db.64]] -->
Its prefixes are 4 to 8 dB better than those of a list trained without prefixes (one image). Second, on Chromium's software renderer and on an iPhone, every pixel decodes within 1 of 255 levels of PyTorch. <!-- @evidence[[results/headline.json#iphone.max_abs_vs_torch]] -->
Third, it draws at any resolution with no extra bytes.

Zooming adds detail. On six 2040-pixel photos fitted at 768, detail tiles trained on demand beat the zoomed splat list by a median 5.69 dB at 2,040 pixels, <!-- @evidence[[results/deepzoom.json#median.gain_L2]] -->
but trail bicubic upscaling of the 768-pixel photo below about 1,024 splats a tile.

What would change the answer is a cheaper splat. Each one costs 10.5 gzipped bytes for 61 bits of fields. <!-- @evidence[[results/headline.json#bytes.per_splat_gz]] --> <!-- @evidence[[results/headline.json#bytes.raw_bits_per_splat]] -->

## Findings

1. **Gap to JPEG, WebP and AVIF at equal bytes.** The median gap to the best conventional encode is 4.02, 3.63 and 5.6 dB at 16, 128 and 1,024 splats. <!-- @evidence[[results/headline.json#gap_vs_best_db.16]] --> <!-- @evidence[[results/headline.json#gap_vs_best_db.128]] --> <!-- @evidence[[results/headline.json#gap_vs_best_db.1024]] -->
   Below about 3 KB the best encode is AVIF of a downscaled image, because full-size AVIF cannot go that small. Above that, the best encode is full-size AVIF. Per image at 1,024 splats, AVIF leads by 3.9 to 6.8 dB. <!-- @evidence[[results/headline.json#avif_gap_1024_db.min]] --> <!-- @evidence[[results/headline.json#avif_gap_1024_db.max]] -->
   The weakest baseline is downscaled JPEG, which still leads by 0.93 dB at 256 splats. <!-- @evidence[[results/headline.json#gap_vs_jpeg_down_db.256]] -->
   Busy texture fares worst: on kodim05 (foliage) the shader reaches 20.36 dB, and AVIF at the same size reaches 24.15 dB (`results/rd.png`, `web/shaders.js`).
2. **Cost of training every prefix.** Each prefix-trained list was compared with lists trained for one length only, at 64 and 256 splats. Its 64-splat prefix is 0.4 dB worse, and its 256-splat prefix is 0.36 dB worse (medians over six images). <!-- @evidence[[results/headline.json#prefix_cost_db.64]] --> <!-- @evidence[[results/headline.json#prefix_cost_db.256]] -->
   A 1,024-splat list trained without prefixes has much worse prefixes: 8.36 dB worse at 64 splats and 4.13 dB worse at 512. This was measured on kodim23 only, because the matching kodim05 run did not finish. <!-- @evidence[[results/headline.json#untrained_prefix_gap_db.64]] --> <!-- @evidence[[results/headline.json#untrained_prefix_gap_db.512]] -->
3. **Decode on SwiftShader and on an iPhone.** The reference is the PyTorch renderer used in training, applied to the same rounded values the shader holds. On the iPhone (Chrome on iOS, which runs WebKit and Metal), every kodim23 prefix shader was at most 1 of 255 levels off the reference, on at most 0.02% of pixels. <!-- @evidence[[results/headline.json#iphone.frac_px_differ_max_pct]] -->
   Repeat renders were byte-identical (`results/bridge_i2w-render-5.json`). On SwiftShader, Chromium's CPU implementation of WebGPU, the PSNR between each shader's render and the reference was at least 71.7 dB. <!-- @evidence[[results/headline.json#swiftshader_vs_torch_db_min]] -->
4. **SIREN residual.** A SIREN is a small neural network with sine activations. A 16-unit one was added inside the shader, after the splats, to fit what the splats miss. It gains 0.05 dB for 787 bytes. <!-- @evidence[[results/headline.json#residual.median_gain_db]] --> <!-- @evidence[[results/headline.json#residual.median_extra_bytes]] -->
   Spending the same bytes on more splats, read off the splat curve between 512 and 1,024, gains about 0.2 dB. <!-- @evidence[[results/headline.json#residual.median_splat_gain_same_bytes_db]] -->
5. **The image-to-svg route.** Tracing the image into paint regions with image-to-svg, then compiling them into a shader with wgsl-forge (a sibling experiment), gives a much larger shader for lower quality. On kodim23 the 16-colour trace makes a 17,160-byte shader at 22.48 dB, while 128 splats reach 23.9 dB in 2,131 bytes. <!-- @evidence[[results/headline.json#svg_arm.kodim23_K16.gz]] --> <!-- @evidence[[results/headline.json#kodim23.n128.psnr]] -->
   On kodim05, Chromium would not compile the 8- and 16-colour traces, because one array constructor held more than 32,767 elements (`results/svg_arm.json`).
6. **Array size limits on iOS.** WebKit, which every browser on iOS uses, rejects a `const` array with more than 2,047 elements. Chromium accepts larger ones. With 32-bit words and 61 bits per splat, one array holds about 1,072 splats, so the 1,024-splat shaders fit. <!-- @evidence[[results/headline.json#splat_cap_one_array]] -->
   Two layouts get past the limit, and both compile: four words packed per `vec4u` element, or the words split across several arrays. On the iPhone, the 1,024-splat shader rendered 1.36 times slower as `vec4u` and 2.32 times slower split into four arrays. <!-- @evidence[[results/headline.json#layout_slowdown.vec4u]] --> <!-- @evidence[[results/headline.json#layout_slowdown.split512]] -->
   At 4,096 splats (7,682 words) neither worked on the iPhone. The `vec4u` render did not finish within minutes, and the split render lost the GPU device.
7. **Drawing time.** In the per-pixel shader every pixel evaluates every splat. On the iPhone the 1,024-splat shader takes 77 ms at 768×512 and 1949 ms at 3840×2560, which is 25 times the pixels for about 25 times the time. <!-- @evidence[[results/headline.json#iphone.ms_1024.768x512]] --> <!-- @evidence[[results/headline.json#iphone.ms_1024.3840x2560]] -->
   An instanced renderer that draws each splat as a quad over its footprint, and lets the GPU blend them, draws the same list on the iPhone in 3 ms at 768×512 and 15 ms at 3840×2560 (`results/bridge_i2w-speed-iphone.json`). The gallery and the downloads use it. The per-pixel shader remains the file whose bytes are counted.
8. **Payload text format.** The shipped shader holds one bitstream of fields packed into 32-bit words. An alternative held one array per field, as small decimal integers. After gzip the alternative was 0.4% smaller, which is too little to matter. <!-- @evidence[[results/headline.json#columnar_smaller_pct]] -->
9. **Training in the browser.** The gallery's encoder trains in WebGPU compute shaders. Each pixel meets only the splats whose footprint reaches its 16×16 tile (`web/gputrain.js`). On an M5 Max this fitted 2,048 splats to a 484×512 image in 2.3 s at 28.07 dB. <!-- @evidence[[results/bridge_i2w-wgsl-train-1.json#runs.2.seconds]] --> <!-- @evidence[[results/bridge_i2w-wgsl-train-1.json#runs.2.points.7.psnr_torch]] -->
   TensorFlow.js on WebGL took 63.3 s for 26.81 dB. <!-- @evidence[[results/bridge_i2w-m5-big2048.json#outputs.runs.0.seconds]] --> <!-- @evidence[[results/bridge_i2w-m5-big2048.json#outputs.runs.0.points.7.psnr]] -->
   An iPhone fitted 8,192 splats in 60.7 s at 30.19 dB, where TF.js had crashed at 1,024. <!-- @evidence[[results/bridge_i2w-wgsl-phone-d.json#runs.0.seconds]] --> <!-- @evidence[[results/bridge_i2w-wgsl-phone-d.json#runs.0.points.9.psnr_torch]] -->
10. **Foveation.** Training with a focus map, here soft ellipses around a detected face, moves quality to where the map points (`web/focus.js`). On the Mona Lisa at 1,024 splats the face-weighted PSNR rose from 26.94 to 28.19 dB. <!-- @evidence[[results/bridge_i2w-focus-face-1.json#runs.2.points.6.psnr_focus_torch]] --> <!-- @evidence[[results/bridge_i2w-focus-face-1.json#runs.3.points.6.psnr_focus_torch]] -->
   The whole-image PSNR fell from 26.89 to 26.37 dB. <!-- @evidence[[results/bridge_i2w-focus-face-1.json#runs.2.points.6.psnr_torch]] --> <!-- @evidence[[results/bridge_i2w-focus-face-1.json#runs.3.points.6.psnr_torch]] -->

11. **Deep zoom adds detail from a larger source.** On six DIV2K photos (2040 pixels wide) with a 1,024-splat base list fitted at 768 pixels, detail tiles of 256 splats raised PSNR over the zoomed base list by a median 4.01 dB at 1,536 pixels wide and 5.69 dB at 2,040. <!-- @evidence[[results/deepzoom.json#median.gain_L1]] --> <!-- @evidence[[results/deepzoom.json#median.gain_L2]] -->
   On Oskar's 1512×2016 night street photo, fitted at 576×768, the gains were 3.75 dB at 1,152 pixels wide and 5.23 dB at the full 1,512. <!-- @evidence[[results/deepzoom.json#private_photo.gain_L1]] --> <!-- @evidence[[results/deepzoom.json#private_photo.gain_L2]] -->
   At 256 splats a tile, bicubic upscaling of the 768-pixel downscaled photo stays ahead, by a median 1.79 dB at 2,040 pixels, though not on 0803 (a flower on a blurred background). <!-- @evidence[[results/deepzoom.json#median.bicubic_lead_L2]] -->
   With 1,024 splats a tile, 0801 (a penguin on rocks) beat bicubic by 0.82 dB and 0803 (a flower) by 6.19 dB, at a median 0.49 and 0.65 s a tile. <!-- @evidence[[results/deepzoom.json#sweep.0801.1024.L2.vs_bicubic]] --> <!-- @evidence[[results/deepzoom.json#sweep.0803.1024.L2.vs_bicubic]] --> <!-- @evidence[[results/deepzoom.json#sweep.0801.1024.tile_seconds_median]] --> <!-- @evidence[[results/deepzoom.json#sweep.0803.1024.tile_seconds_median]] -->
   A median 256-splat tile trained in 0.32 s on the M5 Max. <!-- @evidence[[results/deepzoom.json#tile_seconds.median]] -->
   Drawing the tiles over the base list with the page's renderer scored within 0.026 dB of the trainer's own measurement on every image and level. <!-- @evidence[[results/deepzoom.json#median.drawn_minus_trained_max_abs]] -->
   No phone was timed (`results/deepzoom.json`).
12. **Painting over the coarser levels beats fitting each tile alone.** With the same splats in each tile, tiles fitted alone, over a plain background instead of the coarser picture, scored a median 0.29 dB lower at 1,536 pixels and 0.94 dB lower at 2,040. <!-- @evidence[[results/deepzoom.json#median.vs_indep_L1]] --> <!-- @evidence[[results/deepzoom.json#median.vs_indep_L2]] -->
   At 512 splats a tile the gap at 2,040 pixels was 0.97 and 1.79 dB on the two photos tried. <!-- @evidence[[results/deepzoom.json#sweep.0801.512.L2.vs_indep_same_n]] --> <!-- @evidence[[results/deepzoom.json#sweep.0803.512.L2.vs_indep_same_n]] -->
   Training every power-of-two prefix of a tile, which would let a view draw a cheaper partial tile, cost 0.35 dB at 1,536 pixels and 0.31 dB at 2,040, so the tiles are trained for their full length only. <!-- @evidence[[results/deepzoom.json#median.prefix_cost_L1]] --> <!-- @evidence[[results/deepzoom.json#median.prefix_cost_L2]] -->
   Tile seams cost little: pixels touching a tile boundary were a median 0.54 dB worse than pixels 16 or more pixels from any boundary. The base list alone, which has no tiles, differed by 0.2 dB between the same two groups of pixels. <!-- @evidence[[results/deepzoom.json#median.seam_drop_detail]] --> <!-- @evidence[[results/deepzoom.json#median.seam_drop_base]] -->

## Method

**Codec** (`codec.py`). The background is a bilinear blend of four corner colours. Over it, splats are painted in list order with `c = mix(c, colour, w)`, where `w = alpha · exp(-½ · min(r², 40)^(2^k))`. Here `r` is the distance from the splat centre, measured in the splat's own rotated and scaled frame. `k` ranges from 0 to 3: at 0 the splat is a Gaussian, and higher values give it a flatter top and a sharper edge.

Each splat has 61 bits of fields:
- x 10 bits, y 9 (the image is 1.5 times as wide as it is tall);
- log₂ σ, 6 bits per axis;
- angle, 6 bits;
- colour, 6 bits per channel;
- alpha, 4 bits;
- `k`, 2 bits.

The fields are bit-packed into one `const` array of `u32` in the shader. wgsl-forge minifies the shader, and the rate is the size of the minified source under `gzip -9`. A shader with no splats, which is the fixed overhead of the decoder, takes 694 bytes. <!-- @evidence[[results/headline.json#bytes.overhead_gz]] -->

**Training.** PyTorch on CPU, with Adam on 16,384 random pixels per step. The list grows from 16 to 1,024 splats, doubling each stage. New splats are placed where the current error is highest, sized for the new count. A prefix-trained run minimises the mean log-MSE over every power-of-two prefix; a plain run minimises the log-MSE of the full list only. A final stage trains through rounding to the bit layout (a straight-through estimator), so the shader holds exactly the values that were trained. A 1,024-splat fit took a median of 11.1 minutes on 4 vCPUs. <!-- @evidence[[results/headline.json#fit_minutes_median]] -->

**Runs**, on Kodak images kodim01, 05, 07, 15, 20 and 23:
- one prefix-trained 1,024-splat run per image, measured at every power-of-two prefix;
- plain runs of 64 and 256 splats per image, plus 1,024 on kodim23 (the kodim05 run did not finish);
- the SIREN residual on each prefix-trained run;
- image-to-svg in `painting` mode with 4, 8 and 16 colours, compiled by wgsl-forge, on kodim23 and kodim05.

**Baselines** (`baselines.py`, Pillow 12.3): JPEG, WebP and AVIF over a quality sweep. At full size none of them reaches below about 2.5 KB. So each codec also encodes the image at 1/2, 1/4 or 1/8 size, and the result is upscaled bicubically. For each codec, the comparison takes the best quality it reaches at each byte count, interpolating in log bytes, and reads it off at the shader's byte count.

**Metric.** PSNR on 8-bit RGB against the Kodak PNG. No perceptual metric was run.

**Decode checks.** Every shader was rendered in headless Chromium on SwiftShader (`verify_gpu.py`). The kodim23 shaders and the payload layouts were also rendered on Oskar's iPhone through the GPU bridge, a browser page that runs WebGPU jobs pushed from this session (`bridge/job.js`).

**Encoding in the browser** (`web/gputrain.js`, `web/page.js`). The live gallery's "Your image…" button fits splats to an uploaded image in the browser and builds the same shader that `codec.py` writes. Training runs in WebGPU compute shaders over 16×16 tiles, with every pixel used in every step. It was checked against PyTorch autograd before any real run. The first version trained with TensorFlow.js 4.22 on its WebGL backend (`web/trainer.js`), which is now the fallback. TF.js's WebGPU backend gave wrong gradients. Splat counts are offered up to 8,192 when a time estimate from timed steps on the uploaded image is at most 5 minutes. An optional focus map (a tapped spot, detected faces, or a band of depth from a monocular depth model) weights the loss and the placement of new splats, and adds a focus-weighted PSNR to the scores. `bridge/gputrain_job.js` runs the trainer on a device through the GPU bridge, and `bridge/train_job.js` runs the TF.js trainer.

**Deep zoom** (`web/deepzoom.js`, `bridge/deepzoom_job.js`). The purpose is the gallery's: an uploaded image is fitted as a splat list at 512 or 768 pixels, and zooming in should show more than that list holds. The page therefore keeps the uploaded source, at most 4,096 pixels on its longer side, and builds a pyramid above the fitted size: level L has 2^L times the fitted resolution, capped at the source, which is the last level. For a 2040-pixel photo fitted at 768, level 1 is 1,536 pixels wide and level 2 is the source. Each level is cut into 256-pixel tiles. A tile is trained by the WebGPU trainer on its crop widened by 16 pixels on each side, over a per-pixel background: the render of the base list (the splat list fitted at 768) and of every coarser level there. It is drawn clipped to its 256-pixel core, so inside the core the screen shows exactly what it was trained on. The coarser-level tiles under a tile's crop train first. The page trains the tiles a view needs, nearest its centre first, at the level whose resolution covers the screen's device pixels. The test set is the first six DIV2K validation images (0801 to 0806, 2040 pixels wide; `data/fetch_div2k.py`). Each was fitted at 768 pixels with 1,024 splats; the levels were then 1,536 pixels wide and the source. PSNR per level pools the squared 8-bit error over every tile's core against the level image, the source downscaled by the browser's canvas (`imageSmoothingQuality` high). The zoomed base list is the splat list redrawn at the level's resolution. Arms: tiles painted over the coarser levels at 256 splats (512 and 1,024 on 0801 and 0803); the same with every prefix trained; tiles fitted alone, over the codec's usual background (a blend of four corner colours), at 256 and 512 splats; the base list alone; and bicubic upscaling (Keys, a = −0.5) of the 768-pixel downscaled photo. All runs used Oskar's M5 Max laptop through the GPU bridge. A tile's packed fields take 61 bits a splat, so the 256-splat tiles of a 2040-pixel photo hold about 125 KB before gzip, against about 160 KB for a quality-90 JPEG of the 768-pixel image (0801); the detail is trained in the browser from the source, so these bytes are not sent anywhere.

**Rebuild.** Run `data/fetch.sh`, then `python3 run_sweep.py` (about two hours on 4 vCPUs), `python3 baselines.py`, `python3 svg_arm.py kodim23 kodim05`, `python3 analyze.py --gpu`, `python3 make_headline.py` and `python3 web/build.py`. Deep zoom: `python3 data/fetch_div2k.py`, then per image `python3 bridge/make_deepzoom_job.py data/div2k/<id>.png <dir> --cfg ...` and a GPU-bridge submit of `bridge/deepzoom_job.js` (configurations in the result files), then `python3 deepzoom_analyze.py`.

## Log

### 2026-11-02 — build, sweep, iPhone checks

Asked: "Could we do an image to wgsl compression approach similar to the image to SVG paint idea? The goal is more novelty than practicality", then "Yes". Prior art, searched before writing code:
- COIN and COIN++ store SIREN weights as the compressed image;
- GaussianImage fits 2D Gaussians to images;
- LIVE adds vector paths one at a time.

Three searches did not find the combination built here, which counts shader bytes as the rate, trains every prefix to be valid and puts a residual network inside the shader.

A first 128-splat kodim23 fit, with shorter training than the sweep, reached 23.45 dB at 2,141 bytes. Its SwiftShader decode was within 1 level of torch, and a SIREN residual added 0.27 dB for 780 bytes. Prediction before the sweep: the shader would lose to AVIF on photographs at matched PSNR, and might win on graphics, large displays or perceptual metrics. Measured: it loses to every baseline, including downscaled JPEG. No graphics image or perceptual metric was run.

The sweep ran as one background job. The harness stopped it at its two-hour limit with 25 of 26 runs done; the missing run is kodim05's plain 1,024-splat run.

On the iPhone, through the GPU bridge, the jobs went as follows:
- **`i2w-render-1`** found the 2,047-element limit: a 4,096-splat shader failed to compile.
- **`i2w-render-2`** stalled on a 4,096-splat `vec4u` shader at 1920×1280 and was withdrawn. A job that hits its timeout returns nothing, so later jobs stopped timing a shader after one frame over 2 s and used timeouts of a few minutes.
- **`i2w-render-5`** gave the decode and timing numbers above.
- **`i2w-layout-6`** timed the layouts on one 1,024-splat image: 78 ms per frame for one plain array, 106 ms for `vec4u` and 181 ms for four arrays. It then lost the device on the 4,096-splat split shader.

Three tooling problems came up and were worked around:
- wgsl-forge's SVG reader ignores the id-scoped CSS classes that image-to-svg uses for colour, so every path rendered black. `svg_arm.inline_classes` writes the colours onto each path before compiling.
- On the gallery page, an `img { display: block }` rule overrode the `hidden` attribute and covered the canvas.
- In headless Chromium, reading a WebGPU canvas back through a 2D `drawImage` returned zeros, while screenshots showed the frame.

### 2026-11-02 — encoding an uploaded image in the browser

Asked: "Can't we make this interactive, allowing a user to upload an image?" The trainer was ported to JavaScript on TensorFlow.js 4.22, with the WGSL emitter alongside it, and the gallery gained a "Your image…" button. On a 32-splat test in headless Chromium, the JavaScript trainer's own PSNR, PyTorch rendering the same splats and Chromium rendering its shader all gave 20.089 dB against the 192×128 image, so the two trainers decode identically. Its hand-compacted shader was 1,083 gzipped bytes, against 1,078 from `codec.py` plus the wgsl-forge minifier.

Training on Oskar's iPhone through the GPU bridge found three problems with TF.js's WebGPU backend. The check compared values and gradients against TF.js's CPU backend, for P = 2,048 pixels and chunks of G splats painted together (`results/bridge_i2w-gradcheck-2.json`):
- `matMul` of a `[P, G]` by a `[G, 3]` matrix gave wrong values and gradients from G = 64;
- the gradient of `cumsum` was wrong from a row length of 32, and for axis 0 its gradient came back transposed;
- every op was correct with chunks of at most 32 splats, but training with that cap crashed the tab twice.

On the WebGL backend, every op matched the CPU backend to within about 1e-6, relative to the largest value. The trainer now sums with triangular matrix products instead of `cumsum`, and trains on WebGL. On the iPhone, WebGL trained 64 splats of a 384×256 kodim23 in 26 s (22.85 dB) and 256 splats in 52 s. In the 256-splat run every prefix beat the one before: 18.79, 20.84, 23.04, 24.97 and 25.86 dB at 16, 32, 64, 128 and 256 splats (`results/bridge_i2w-train-6.json`). A 1,024-splat run crashed the tab on WebGL as well, even when each training step sampled only 2,048 pixels instead of 8,192, so the page offers at most 256 splats. Only the iPhone was tested.

### 2026-11-02 — limits by device, and training changes, on an M5 Max

Asked, after a 256-splat Mona Lisa from the iPhone came out soft: raise the limit on a stronger GPU, set it from the device, and improve training. All runs below are on Oskar's M5 Max through the GPU bridge, TF.js WebGL, Mona Lisa at 484×512 (`results/bridge_i2w-m5-*.json`).

| run | splats | time | PSNR |
|---|---|---|---|
| trainer as shipped | 256 | 21 s | 24.12 dB |
| trainer as shipped | 1,024 | 57 s | 25.49 dB |
| trainer as shipped | 2,048 | 50 s | 25.75 dB |
| `WEBGL_USE_SHAPES_UNIFORMS` on | 1,024 | 29 s | 25.48 dB |
| twice the pixels per step and twice the steps, plus edge-aligned init and importance sampling | 1,024 | 47 s | 26.34 dB |
| the same | 2,048 | 63 s | 26.81 dB |

The shipped trainer barely improved past 1,024 splats, because each step samples fewer pixels as the list grows: at 2,048 splats a step saw 1,024 pixels. With the TF.js flag on, compiled shaders take tensor shapes as uniforms, which removed about 28 s of recompiling between stages at the same quality.

An ablation at 256 splats compared new splats stretched along the local edge direction (from a structure tensor), half the sampled pixels drawn in proportion to the error (weighted so the loss stays unbiased), and moving splats that cover under two pixels of ink. On the Mona Lisa, edge-aligned initialisation added 0.40 dB and importance sampling 0.13 dB more; on kodim23 at 384×256 they added 0.02 dB together. No splat ever met the respawn criterion. Twice the steps added 0.78 and 0.34 dB with the other two changes on. The page now sets edge-aligned init, importance sampling and the TF.js flag everywhere. On computers it also doubles pixels per step and steps, and goes up to 2,048 splats and 768-pixel uploads. Phones keep 256 splats and 512 pixels.

The page times two short training steps before offering splat counts and labels each with an estimate. On the M5 the estimates were 65 s for 512 splats (actual 60 s) and 74 s for 1,024 (actual 45 s). Above 2,047 words the shader packs its data into `vec4u`, which decodes within 1 level of the PyTorch decode on SwiftShader.

### 2026-11-02 — drawing each splat as its own quad

Asked: "Why is this so much more efficient rendering 3d splats?" (about splat.radiancefields.net), then "build the instanced renderer". The per-pixel shader evaluates every splat at every pixel. 3D Gaussian splatting viewers instead draw each splat as a quad covering only its footprint, and the GPU's blending composites them. `web/instanced.js` does that for these lists:
- one instanced quad per splat, sized to where alpha times falloff drops below 1/1024;
- painted in list order with "over" blending, which computes `mix(c, colour, w)` exactly;
- blended into an `rgba32float` target when the device has `float32-blendable`, else `rgba16float`, then copied to the screen.

The data is the same packed words as the per-pixel shader. Drawing the first n instances paints the n-splat prefix.

Frame times through the GPU bridge, median of 5 (`results/bridge_i2w-speed-m5.json`, `results/bridge_i2w-speed-iphone.json`):

| list | device | per-pixel at 768×512 / 1920×1280 / 3840×2560 | instanced, same sizes |
|---|---|---|---|
| kodim23, 1,024 splats | M5 Max | 10.4 / 62.3 / 243.8 ms | 0.3 / 0.7 / 1.9 ms |
| Mona Lisa, 2,048 splats | M5 Max | 28.8 / 128.6 / 499.5 ms | 0.3 / 0.6 / 1.4 ms |
| kodim23, 1,024 splats | iPhone | 80 / 475 / 1,894 ms | 3 / 7 / 15 ms |
| Mona Lisa, 2,048 splats | iPhone | 144 / 887 / 3,160 ms | timed out |

That is 35 to 357 times faster on the M5 and 27 to 126 times on the iPhone. The iPhone job ran out of time before its instanced Mona Lisa rows. A diagnostic job then drew the full 2,048-splat list on the iPhone in 3 ms (`results/bridge_i2w-diag-iphone.json`). On SwiftShader the instanced draw was within 3 of 255 levels of the PyTorch decode, 59 to 68 dB between them. The difference comes from the 1/1024 cutoff and the blending precision. The gallery now draws with the instanced renderer. The per-pixel shader is still the file whose bytes are counted.

### 2026-11-02 — training in WGSL compute shaders

Asked: "build the footprint-limited WGSL trainer". In the TF.js trainer every sampled pixel evaluates every splat, so it can afford a few thousand pixels per step, fewer as the list grows. `web/gputrain.js` trains the way 3D Gaussian splatting's tile rasterisers do. Every step uses every pixel, and each pixel meets only the splats whose footprint (weight at least 1/1024) reaches its 16×16 tile. A step is a fixed sequence of compute dispatches:
1. Decode each splat and find the tiles it reaches.
2. List the splats per tile, in list order, so nothing is sorted.
3. Paint each tile, keeping the colour and log-transmittance at every power-of-two prefix.
4. Reduce the prefix losses.
5. Walk the lists again for the gradients. A splat's transmittance comes from the stored logs as exp(L_k − L_≤i), with no division by (1 − w).
6. Sum each splat's per-tile partials through a slot map, which avoids float atomics.
7. Take an Adam step.

The model matches `trainer.js` and `codec.py`, with four differences:
- every pixel is used in every step;
- the weight is clamped at 0.9999;
- tails below 1/1024 are ignored;
- the background is fixed at the quadrant means.

Against PyTorch autograd on SwiftShader (`bridge/gradcheck_gpu.py`, no cutoff, 48×32 pixels, three prefixes), the loss agreed to 1e-5. The worst relative gradient error was 5.8e-4 with 2 splats, 8.8e-4 with 12 and 5.7e-4 with 40. Two bugs came first, both now in METHODS:
- A loop bound read from a storage buffer counts as non-uniform, so the barriers inside the loop did not compile.
- On SwiftShader, an array declared inside a loop without an initialiser kept the previous iteration's values, and the gradients correlated only 0.7 with PyTorch's.

Mona Lisa at 484×512 through the GPU bridge, with the default schedule: 150 steps per growth stage, 500 at full size and 200 rounding to bits (`results/bridge_i2w-wgsl-*.json`).

| device | trainer | splats | time | PSNR |
|---|---|---|---|---|
| M5 Max | TF.js on WebGL, shipped settings | 256 | 21 s | 24.12 dB |
| M5 Max | TF.js on WebGL, best settings (entry above) | 1,024 / 2,048 | 47 / 63 s | 26.34 / 26.81 dB |
| M5 Max | WGSL | 256 / 1,024 / 2,048 | 1.4 / 1.8 / 2.3 s | 24.69 / 26.89 / 28.07 dB |
| M5 Max | WGSL | 8,192 | 4.1 s | 30.14 dB |
| iPhone | WGSL | 256 / 1,024 / 2,048 | 14.6 / 26.2 / 34.3 s | 24.61 / 27.00 / 27.95 dB |
| iPhone | WGSL | 4,096 / 8,192 | 45.7 / 60.7 s | 29.07 / 30.19 dB |

Each PSNR is the GPU's own measurement of the rounded list. PyTorch re-rendered every returned list without the cutoff, and its PSNR agreed within 0.01 dB at every prefix of every run.

At 2,048 splats on the M5, three and eight times the default steps gave 27.95 and 28.10 dB, against 28.07 dB, so the default schedule has converged. More splats do help. At three times the steps, 4,096 splats reached 29.07 dB in 8.3 s and 8,192 reached 30.24 dB in 11.9 s. Under TF.js, 1,024 splats crashed the iPhone tab. The WGSL trainer ran 8,192 there. The gzipped per-pixel shader of the M5's 8,192-splat list is 85,905 bytes, and its 2,048-splat prefix is 22,735 bytes.

The page now trains with `gputrain.js` and offers up to 8,192 splats on every device. TF.js is the fallback, used when the URL has `?tf` or the WebGPU trainer fails. The splat menu's estimates come from 20 timed steps at 256 and at 2,048 splats on the uploaded image. On the iPhone that predicted 33.5 s for 2,048 splats, and the run took 34.3 s.

Past about 4,295 splats a list needs more than 2,047 `vec4u` elements, more than WebKit allows in one array. Both emitters now split the data over several arrays. The iPhone drew the M5's 8,192-splat list at 30.24 dB, as PyTorch does, in 3 ms per frame. On SwiftShader the split per-pixel shader was within 1 level of PyTorch.

Asked at the same time: the HTML download now opens on the finished picture, without playing. A "Paint it in" button replays the painting. A slider steps through the measured prefixes and shows each one's splats, bytes and PSNR. The download carries no source image and no JPEG or WebP comparison.

Asked next: "We should allow zooming into the generated wgsl to show off how it's NOT pixellated/rasterized". The instanced renderer now takes a camera: a zoom factor, up to 64×, about a point of the image. It redraws the splats at that scale, so their edges stay smooth at any magnification. In the gallery and the download, the wheel, a pinch or the + and − buttons zoom, and dragging pans. In the gallery, the original and the JPEG, WebP and AVIF files zoom along with the shader. They are scaled nearest-neighbour, so their pixels show. Wheel zoom and drag were checked in headless Chromium. Pinch was not tested on a phone.

### 2026-11-02 — foveation: training toward a face, a spot or a distance

Asked earlier: "include a foveation option - like - focus on the face, say, or do a depth field analysis and focus on a particular distance", then "yes, wire up foveation". A focus map is one weight per pixel. `web/focus.js` builds it three ways:
- a Gaussian spot around a point the user taps;
- soft ellipses around the faces that MediaPipe's face detector finds;
- a band of depth around the distance of a tapped point, from the Depth Anything V2 small model run with transformers.js.

Everything runs in the browser, and the image is never uploaded. Both trainers floor the map at 1/20 of its maximum and normalise it to mean 1. The floor keeps every pixel counting a little. Both trainers then use the map as per-pixel loss weights and to choose where new splats go, so the WebGPU trainer stores the weights beside the target colours. A second evaluation pass weights the 8-bit error by the same map, giving a focus-weighted PSNR that the page reports beside the whole-image PSNR, for the JPEG and WebP comparisons too. Against PyTorch autograd with Gaussian pixel weights (`bridge/gradcheck_gpu.py 48 32 40 focus`), the weighted loss agreed to 2e-5, and the worst relative gradient error was 3.7e-4.

In headless Chromium on the 484×512 Mona Lisa, the face detector found her face (score 0.93) in 2 s, and the depth model took 12 s on the CPU. Depth Anything's map put her hands nearest, so a fixed "nearest" depth would have focused on the hands. The page therefore picks the depth from a tapped point, starting at the centre.

On the M5 Max through the GPU bridge, Mona Lisa at 484×512 was trained evenly and trained with the face map. Both runs were scored on the same face map (`results/bridge_i2w-focus-face-1.json`):

| splats | even: whole / face-weighted | face focus: whole / face-weighted |
|---|---|---|
| 256 | 24.69 / 24.61 dB | 23.99 / 25.63 dB |
| 1,024 | 26.89 / 26.94 dB | 26.37 / 28.19 dB |
| 2,048 | 28.07 / 28.29 dB | 27.40 / 29.29 dB |

Focusing on the face gained 1.0 to 1.25 dB there and cost 0.5 to 0.7 dB over the whole image. The gain was larger in short prefixes: 1.35 dB at 64 splats and 1.4 dB at 32. On the face, 1,024 focused splats came within 0.1 dB of 2,048 even ones. The focused runs took no longer. The face-weighted PSNR still counts the rest of the image at a twentieth of the weight, so it is not a PSNR of the face region alone.

The iPhone gave the same trade at 1,024 splats (`results/bridge_i2w-focus-phone-2.json`): trained evenly, 27.00 dB whole and 27.21 dB face-weighted in 26.2 s; trained on the face, 26.42 and 28.32 dB in 27.8 s. PyTorch re-renders of both lists matched to the second decimal. On the phone the depth model ran in 5.4 s. The face detector could not be tested through the bridge, because the bridge runs jobs in a Web Worker and MediaPipe's bundle needs `document`. In the page the detector runs on the main thread, and it worked there in headless Chromium. In the headless page tests, the spot and distance modes both moved the focus map to a tapped point and trained with it (`bridge/focus_page_test.mjs`).

### 2026-11-03 — deep zoom: detail tiles from a larger source

Asked (#108): "could we enhance an image when zoomed in on, sort of like the deep zoom approach, but using shaders? Obviously only up to a max resolution, but a 1x display on a phone is not super dense." Until now the zoom redrew the same list at any scale, smooth but with nothing new, and uploads were downscaled to 512 or 768 pixels before fitting.

Prior art, searched first. LIG ("Large Images are Gaussians", AAAI 2025) fits a large image with 2D Gaussians in two levels, a coarse fit to a downsampled image and then most Gaussians on the high frequencies. Image-GS (SIGGRAPH 2025) adds Gaussians where the error stays high, which gives it a level-of-detail ordering. ACORN fits gigapixel images with a quadtree of blocks. Babylon Lite and MapTiler GeoSplats stream 3D splats by level of detail. Five searches did not find splat tiles trained on demand in the browser for a 2D deep zoom.

Alternatives weighed before building:
- a quadtree of independent tiles, each a full picture of its region (measured as "alone" below);
- residual tiles painted over the coarser levels (built);
- detail lists trained so every prefix works (measured, as a per-tile option);
- one bigger list trained at the high resolution. Its per-prefix snapshots alone would need 1.2 GB at 3072×2048, too much for a phone.

The trainer gained a per-pixel background. Against PyTorch autograd on SwiftShader, painting over another image, the worst relative gradient error was 3.1e-4 (`bridge/gradcheck_gpu.py 48 32 40 bgimg`). The renderer gained a second instanced pass for detail splats, read from a storage buffer with one clip rectangle per tile, and a float readback used as the training background. On SwiftShader, `bridge/page_deepzoom_test.mjs` uploaded a 1600×1060 crop, zoomed 16×, and drew the two tiles the view needed, with no errors.

Full-pyramid runs on the M5 Max through the GPU bridge, 256 splats a tile. PSNR at 1,536 and at 2,040 pixels wide: base list alone / with detail / tiles fitted alone / bicubic of the 768-pixel photo. "Tiles" counts both levels; "training" is the time spent training them on the M5 (`results/bridge_dz-*-arms-1.json`):

| image | source | base list at 768 px | 1,536 px wide | 2,040 px wide | tiles | training |
|---|---|---|---|---|---|---|
| 0801 | 2040×1356 | 22.11 | 21.26 / 25.08 / 24.78 / 31.77 | 20.95 / 26.53 / 25.60 / 29.97 | 72 | 21.1 s |
| 0802 | 2040×1356 | 25.28 | 24.76 / 31.64 / 31.23 / 37.20 | 24.56 / 33.07 / 32.00 / 35.33 | 72 | 21.3 s |
| 0803 | 2040×1536 | 28.17 | 27.37 / 35.30 / 34.54 / 37.74 | 27.09 / 38.21 / 36.06 / 36.13 | 78 | 32.3 s |
| 0804 | 2040×1200 | 21.64 | 20.59 / 24.79 / 24.54 / 29.23 | 20.14 / 25.93 / 24.98 / 27.25 | 64 | 18.7 s |
| 0805 | 2040×1536 | 23.93 | 22.83 / 26.00 / 25.73 / 31.43 | 22.34 / 26.83 / 25.94 / 29.32 | 78 | 25.6 s |
| 0806 | 2040×1356 | 25.23 | 23.59 / 27.10 / 26.86 / 29.97 | 22.90 / 27.75 / 26.86 / 27.96 | 72 | 25.3 s |

Splats a tile, painted over the coarser levels (`results/bridge_dz-*-n*-1.json`):

| image | splats a tile | 1,536 px | 2,040 px | median tile |
|---|---|---|---|---|
| 0801 | 256 / 512 / 1,024 | 25.08 / 26.85 / 28.90 | 26.53 / 28.50 / 30.80 | 0.32 / 0.38 / 0.49 s |
| 0803 | 256 / 512 / 1,024 | 35.30 / 37.73 / 39.78 | 38.21 / 40.33 / 42.33 | 0.39 / 0.51 / 0.65 s |

Prediction before the runs: detail tiles would beat the zoomed base list easily and lose to bicubic upscaling of the 768-pixel photo, because the base list is itself 3 to 9 dB below that photo. Measured at 256 splats a tile: the first held on every image; the second held on four images, tied on 0806 (27.75 against 27.96 dB) and failed on 0803 (38.21 against 36.13 dB). At 1,024 splats a tile, both images tried beat bicubic at 2,040 pixels.

On Oskar's 1512×2016 night street photo, fitted at 576×768, 78 tiles trained in 25.7 s. Base list alone against with detail: 20.09 and 23.84 dB at 1,152 pixels wide, 19.63 and 24.86 dB at the full size (`results/deepzoom_private.json`; the photo and its renders stay out of the repo). Frame times on the M5 Max stayed near 1 ms at 1920×1280 with 16,446 detail splats (`results/bridge_dz-0801-res-1.json`). No phone was timed: Oskar asked to stay on the Mac where possible.

The gallery now keeps an upload's source, up to 4,096 pixels, and trains the tiles a view needs in the background: 1,024 splats a tile on computers and 512 on phones. A phone screen at 1× already shows about twice the pixels of a 512-pixel fit, so on a phone the detail starts before any zoom. The Kodak gallery images are 768×512 and have no larger source, so they gain nothing. Trained detail is not in the downloads.
