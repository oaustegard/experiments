# image-to-wgsl: a photograph as one WGSL shader

The question, from Oskar on 2026-11-02, was whether an image could be compressed into a WGSL program, aiming for novelty over practicality. The model was the image-to-svg skill, which traces an image into flat-colour paint regions and writes them out as SVG paths.

The codec built here stores an image as an ordered list of soft, rotated ellipses, called splats, packed into an integer array inside one fragment shader. The shader decodes the list and paints the splats per pixel. The compressed size is the gzip size of the shader source after minification. A prefix is the first *n* splats of the list. Training makes every power-of-two prefix a usable image, so the shader can paint itself in, coarse to fine.

The test images are six photographs from the Kodak set, 768×512 each. Quality is PSNR in dB against the original. Live gallery: [austegard.com/experiments/image-to-wgsl/web/](https://austegard.com/experiments/image-to-wgsl/web/).

## Answer

On the six Kodak photographs, the 1,024-splat shader reaches a median 26.11 dB at a median 11,450 bytes. <!-- @evidence[[results/headline.json#median_psnr.1024]] --> <!-- @evidence[[results/headline.json#median_bytes.1024]] -->
At that byte count, AVIF is 5.6 dB better (median over the six images). <!-- @evidence[[results/headline.json#gap_vs_best_db.1024]] -->
JPEG, encoded at half, quarter or eighth size and upscaled, is 1.72 dB better. <!-- @evidence[[results/headline.json#gap_vs_jpeg_down_db.1024]] -->
The shader trails every codec tested at every list length from 16 to 1,024 splats, so as compression it loses.

Three properties hold. First, a prefix-trained list costs 0.4 dB against a list trained for one length only. <!-- @evidence[[results/headline.json#prefix_cost_db.64]] -->
Its prefixes are 4 to 8 dB better than those of a list trained without prefixes (one image). Second, the decode matches on two very different GPUs: on Chromium's software renderer and on an iPhone, every pixel is within 1 of 255 levels of the PyTorch decode. <!-- @evidence[[results/headline.json#iphone.max_abs_vs_torch]] -->
Third, the shader can be drawn at any resolution with no extra bytes, though drawing time grows with the pixel count.

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
7. **Drawing time.** Every pixel evaluates every splat, because no splat is skipped by position. On the iPhone the 1,024-splat shader takes 77 ms at 768×512 and 1949 ms at 3840×2560, which is 25 times the pixels for about 25 times the time. <!-- @evidence[[results/headline.json#iphone.ms_1024.768x512]] --> <!-- @evidence[[results/headline.json#iphone.ms_1024.3840x2560]] -->
   A 16-splat prefix takes 4 ms at 768×512. At that size the shader's fixed cost dominates. <!-- @evidence[[results/headline.json#iphone.ms_16.768x512]] -->
8. **Payload text format.** The shipped shader holds one bitstream of fields packed into 32-bit words. An alternative held one array per field, as small decimal integers. After gzip the alternative was 0.4% smaller, which is too little to matter. <!-- @evidence[[results/headline.json#columnar_smaller_pct]] -->

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

**Encoding in the browser** (`web/trainer.js`, `web/page.js`). The live gallery's "Your image…" button fits splats to an uploaded image in the browser with TensorFlow.js 4.22, and builds the same shader that `codec.py` writes. It trains on TF.js's WebGL backend, because the WebGPU backend gave wrong gradients, and stops at 256 splats, because 1,024 crashed an iPhone tab (log entry below). `bridge/train_job.js` runs the same trainer on a device through the GPU bridge.

**Rebuild.** Run `data/fetch.sh`, then `python3 run_sweep.py` (about two hours on 4 vCPUs), `python3 baselines.py`, `python3 svg_arm.py kodim23 kodim05`, `python3 analyze.py --gpu`, `python3 make_headline.py` and `python3 web/build.py`.

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
