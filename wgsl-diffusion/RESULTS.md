# wgsl-diffusion — a texture diffusion model in hand-written WGSL

An 8.38M-parameter class-conditional U-Net, trained with rectified flow on the 47 classes of the
Describable Textures Dataset (DTD), generates seamless textures in the browser. Every layer runs in
WGSL compute shaders Muninn (Claude Code) wrote for it; there is no ML library on the page. On a Mac
(Apple GPU, Chrome 156) 20 sampling steps with classifier-free guidance take 0.44 s at 128×128,
1.12 s at 256×256 and 3.69 s at 512×512. A 128×128 sample from fixed noise matches PyTorch at
85.8 dB PSNR on that GPU.

Live page: `web/index.html` ([austegard.com/experiments/wgsl-diffusion/web/](https://austegard.com/experiments/wgsl-diffusion/web/)),
WebGPU required (Chrome or Edge 113+, Safari 26+, Firefox 141+). Pick a texture class and press
Generate: the page shows the model's current estimate of the clean image at every step, then repeats
the finished tile across the screen. Size (64 to 512 px), steps, guidance strength and zoom are
sliders; Seamless toggles circular padding, Auto-dream cycles random classes, Save PNG downloads the tile.

![12 classes; each 128×128 tile is drawn 2×2, so a seam would show as a cross in the middle](results/gallery_tiled.png)

Oskar, 2026-11-07: "Can we implement diffusion in wgsl? We sorta kinda achieved the _effect_ with
the image to shaders experiment - but that used a real image as the target" (that is
[`image-to-wgsl/`](../image-to-wgsl/RESULTS.md), which fits one photo). He had been reading SKILD
([arXiv 2605.26032](https://arxiv.org/abs/2605.26032)), a diffusion model that noises frequencies
in order and trains on CIFAR-10, ImageNet and critical Ising spin fields. Offered Ising fields or
textures as the dataset, he picked textures ("Go for purtty"), then asked for a faster conv kernel
than the first one, 512×512 tiles and quantization.

## Model and training

- **Architecture** (`train/model.py`): conv-only U-Net, channels 64/128/192/256 at four resolutions,
  one GroupNorm + FiLM ResBlock per level on the way down, one per level on the way up after a
  nearest upsample and skip concat, one middle block. Time (sinusoidal, MLP) and class (embedding, plus
  a null class for guidance) are summed into one 256-d vector that sets each block's FiLM scale and
  shift. There is no attention, so the network is translation-equivariant and runs at any size that is
  a multiple of 16.
- **Objective**: rectified flow, x_t = (1 − t)·x₀ + t·ε, the model predicts v = ε − x₀, with t drawn
  logit-normal(0, 1). Classes are dropped to the null class 10% of the time for classifier-free
  guidance. Sampling is Euler from t = 1 to 0.
- **Data** (`train/train.py`): all 5,640 DTD images (`tanganke/dtd` train + test parquet), each resized
  so its short side is 96, 128 or 176 and centre-cropped square. Each training example is a random
  64×64 crop from one of those scales, with random flips and transposes.
- **Run**: 60,000 steps at batch 128, AdamW 2e-4 with 1k warmup and cosine decay to 2e-5, EMA 0.9995,
  bf16 autocast, on a Colab A100 at 8.5 s per 100 steps (87 minutes). Loss went from 1.06 at step 200
  to 0.11 (`results/train_log.jsonl`). The shipped weights are the EMA, exported as f16 (16.8 MB).

The SKILD paper's own model is a 121M-parameter NCSN++ U-Net sampled in 1,000 steps, about 103 s per
image on a Titan RTX by its Table E.12. Its frequency-ordered (DCT) forward process is not used here.
Plain rectified flow already resolves coarse structure first, which shows on the page as the
per-step estimate of the clean image (x₀) sharpens.

## Seamless tiles from circular padding

The network trains with zero padding and samples with circular padding, so the left edge sees the right
edge as its neighbour and copies of the tile meet without a seam. `run/seam_check.py` measures the mean
jump across the wrap boundary divided by the mean jump between neighbouring interior pixels, over 12
classes at 128×128 (`results/seam.json`):

| padding at sampling | median seam ratio |
|---|---|
| zero | 3.93 |
| circular | 1.00 |

A tile can be generated larger than the 64×64 training crops. Its features stay at the trained scale:
a 512 tile holds more cracks or cells before it repeats, each the size it is in a 128 tile.

## The WGSL runtime

`web/diffusion.js` holds every kernel and the planner, which turns the layer list into dispatches and
GPU buffers for one image size. Guidance runs the network on a batch of two each step, the chosen class
and the null class. One sampling step is about 80 dispatches in a single compute pass:

- **conv** (v2; v1, the first version kept for comparison, has each thread compute one pixel × 16
  channels): an 8×8 pixel tile × 32 output channels per workgroup of 64 threads; each thread keeps
  4 adjacent pixels × 8 channels in vec4 accumulators. Input tiles and weights go through workgroup
  memory 4 input channels at a time. The GroupNorm affine, FiLM and SiLU of the conv's input are
  applied as each value is loaded, so normalized activations are never written out. Weights are
  re-laid out to [Cin][K·K][Cout] at load time so that load is contiguous, and K and stride are
  pipeline constants. Layers with few output tiles split their input channels across up to 16
  workgroups (split-K) and a second pass adds the partial sums, bias and residual.
- **gnorm**: one workgroup per (item, group), two-pass mean and variance, writing the per-channel
  (a, b) that the next conv applies as silu(a·x + b).
- **upcat**: nearest 2× upsample of one tensor concatenated with another along channels.
- **step**: classifier-free guidance, the Euler update of the f32 state, the x₀ estimate for display,
  and the next network input for both batch items.

The embedding MLP and every block's FiLM projection run in JS for all steps before sampling starts, and
are copied into place per step. A liveness pass over the op list reuses activation buffers.

Activations are stored as f16 pairs packed into u32 with `pack2x16float`, so no `shader-f16` feature is
needed. Arithmetic and the sampler state stay f32.

## Correctness

- SwiftShader (Chromium's software GPU) in headless Chromium (`run/test_gpu.mjs`, fixtures from `train/export.py fixture` with
  random weights): forward passes at 64×64 zero-padded, 48×80 and 64×64 circular, plus a 3-step guided
  sample, all within 9e-6 relative of PyTorch for v1, v2 and v2 without split-K. The f16-storage
  variants come within 1e-3 relative (f16's own precision) on a forward pass and 3.6e-3 on the sample.
- Apple GPU through the GPU bridge (a page on Oskar's Mac that runs WebGPU jobs pushed to a repo and
  commits the results back), final weights (`results/bench_mac.json`): a 128×128 sample, 20 steps,
  guidance 3, from fixed noise, against PyTorch f32 on the same f16 weights: max difference 5.0e-4,
  85.8 dB PSNR. On earlier 10k-step weights, f32 storage gave 141.8 dB and f16 storage 86.7 dB in the
  same test.

## Speed and memory on the Mac

20 steps, guidance 3, seamless, f16 storage; split-K is used when a conv has fewer than 256 workgroups.
The adapter reports `apple metal-3`, Chrome 156:

| size | 20 steps | per step | GPU buffers |
|---|---|---|---|
| 64² | 0.28 s | 14.0 ms | 13 MB |
| 128² | 0.44 s | 21.8 ms | 40 MB |
| 192² | 0.74 s | 37.0 ms | 82 MB |
| 256² | 1.12 s | 56.2 ms | 143 MB |
| 384² | 2.20 s | 110 ms | 323 MB |
| 512² | 3.69 s | 185 ms | 574 MB |

Without guidance (one batch item) 128² takes 0.26 s. PyTorch on 4 CPU cores of the cloud container
this was built in takes 2.7 s for the guided 128² run.

How each change moved the numbers (20 guided steps, same Mac; blank cells were not measured):

| change | 64² | 256² | 512² memory |
|---|---|---|---|
| v1 conv: 8×8 pixels × 16 channels, one pixel per thread | 488 ms | 2260 ms | |
| v2 conv, no split-K | 392 ms | 1121 ms | |
| v2 conv, split-K target 256 | 267 ms | 1044 ms | 1147 MB |
| f16 activation storage | 271 ms | 1121 ms | 574 MB |

v1 reached 1.5 TFLOP/s at 256² and v2 about 3.2. Computing every step's FiLM vectors before sampling
and submitting each step before waiting on the previous one changed nothing measurable, so the
remaining 64² time is spent on the GPU.
f16 storage costs 3–7% speed on this GPU and halves memory. The page disables a size when its largest
buffer exceeds the adapter's `maxStorageBufferBindingSize`.

## Precision

`run/quant_sim.py` simulates storage formats in PyTorch and compares 20-step guided samples from the
same noise with the f32 run, over 8 classes at 128×128 (final weights, `results/quant_sim.json`):

| format | median PSNR | worst |
|---|---|---|
| f16 activations, f16 weights | 84.1 dB | 79.5 dB |
| bf16 activations | 65.4 dB | 55.5 dB |
| int8 weights, symmetric per output row | 34.4 dB | 29.9 dB |

Above about 40 dB the difference is not visible at 8 bits per channel; 30 to 35 dB can show in smooth
gradients. An intermediate 15k-step checkpoint gave 42.3 dB with int8 weights, so the final weights
quantize worse. The page loads the f16 weights. `export.py` also writes `weights_q8.bin` (8.4 MB), and
`TextureDiffusion.create(device, "./model", {weights: "q8"})` in `diffusion.js` loads it.

## Prior art

Two searches found [favicon-diffusion](https://awesome.ecosyste.ms/projects/github.com%2Fneelr%2Ffavicon-diffusion),
a diffusion transformer in hand-written WGSL, and [web-stable-diffusion](https://github.com/cnye36/web-stable-diffusion),
Stable Diffusion in the browser from TVM-generated shaders. Circular padding for tileable textures is
established ([arXiv 2410.07128](https://arxiv.org/pdf/2410.07128), [DreamPBR](https://arxiv.org/html/2404.14676v1)).
I did not find a hand-written conv U-Net for class-conditional seamless textures at arbitrary size in
those searches.

## Not done

- No phone was tested. 512² needs 574 MB of buffers.
- f16 arithmetic (`shader-f16`) and int8 activations were not tried.
- Better int8 weights (per input-channel groups, or quantization-aware fine-tuning) were not tried.
- SKILD's frequency-space forward process was not tried.

## Reproducing

```bash
python3 train/train.py --out run1 --steps 60000          # on a CUDA GPU (87 min on an A100)
python3 train/export.py weights run1/ema.pt web/model     # f16 and int8 weight files
python3 train/export.py fixture web/test && node run/test_gpu.mjs   # WGSL against PyTorch
cd web && python3 -m http.server 8000                     # open http://127.0.0.1:8000/
```

## Files

- `train/model.py`, `train/train.py`, `train/export.py`: model, Colab training, f16/int8 export and test fixtures.
- `run/colab_launch.py`, `run/keepalive.sh`: start training on a Colab VM, keep it alive, and copy the
  log, samples and checkpoint back to the local machine.
- `run/test_gpu.mjs` with `web/_test.html`: SwiftShader check against the fixtures. `run/shot.mjs`: page screenshot.
- `run/seam_check.py`, `run/quant_sim.py`, `run/gallery.py`, `run/ref_from_bin.py`: measurements.
- `bridge/*.js`: GPU-bridge jobs for the Mac timings and correctness.
- `web/`: the page (`index.html`, `app.js`), the runtime (`diffusion.js`) and the weights (`model/`).
- `ERRORS.md`: one wrong turn, a GPU "mismatch" that was two different checkpoints.
