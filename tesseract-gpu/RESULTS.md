# tesseract-gpu: Tesseract's LSTM recognizer on the GPU

Tesseract 5 reads a text line with a small LSTM network and a beam search, one line at a time on the CPU. The network
is a 3x3 convolution, max-pooling, one LSTM that runs down each pixel column and keeps its last output (the "y-LSTM"),
three LSTMs that run along the line (the "x-LSTMs": forward, reversed, forward), and a softmax over the model's
character codes at each step. Google publishes two sets of these models: tessdata_best with float weights and
tessdata_fast, smaller, with 8-bit weights and 8-bit activations. This experiment runs that same network on the GPU
for many lines at once, in two forms: a PyTorch engine for servers (`server/torch_engine.py`) and a WebGPU engine for
the browser (`web/engine.js`, with a demo page at https://austegard.com/experiments/tesseract-gpu/web/). Both load any
of the 320 LSTM models in tessdata_best and tessdata_fast unchanged. Oskar Austegard, who owns this repo, asked for it,
and asked that the work be split into small parallel tasks for Claude Haiku. Error rates are CER: character edits
divided by the length of the true text. Mac timings come from the GPU Bridge, this repo's job runner in a browser tab
on Oskar's Mac; "SwiftShader" is Chrome's CPU implementation of WebGPU, used for headless correctness checks here. See `NOTICE.md` for the licences of the
Tesseract code, the tessdata models and the Indra OCR models the demo page uses.

## Answer

Can Tesseract's recognizer run on a GPU and still be Tesseract? Yes. On 686 English, French and Ancient Greek
benchmark lines, the reference that both engines are checked against reproduces Tesseract 5.3.4's own
softmax output to within 3.3e-5 for the float models <!-- @evidence[[results/summary.json#parity.best_max_dp]] -->
and 1.3e-6 for the 8-bit models, <!-- @evidence[[results/summary.json#parity.fast_max_dp]] --> with the same best
character at every timestep, and a port of Tesseract's beam search returns its exact characters on all 1,372
readings. <!-- @evidence[[results/summary.json#beam.identical]] --> On Oskar's Mac
(Chrome 157), 343 lines take 176 ms of GPU time with eng best, <!-- @evidence[[results/summary.json#mac.best_gpu_ms]] -->
and the text equals the reference on every line (beam decoding not included). On a Kaggle Tesla T4 the server engine reads 1,143 lines per second, <!-- @evidence[[results/summary.json#t4.best_cudnn_b4096]] -->
still limited by preparing lines on the CPU rather than by the GPU. Stock
Tesseract took 0.19 s a line in each of four parallel processes on four CPUs, about 21 lines per second in all. <!-- @evidence[[results/summary.json#baseline_binary.best_s_per_line_4proc]] --> Packing
tessdata_best's weights to 8 bits keeps its error rate within 0.1 point on all three languages. Making the LSTM itself cheaper would need retraining (Finding 7, Log).

## Findings

1. **The network can be reproduced exactly, given two version details.** Tesseract 5.3.4 fills the convolution's
   out-of-image border with values from `std::minstd_rand` seeded per model; current master switched to a 64-bit LCG,
   so border columns, and sometimes a line's first or last character, depend on the Tesseract version. tessdata_fast's
   8-bit models also need float32 arithmetic op for op, down to `IntCastRounded(float)` adding `0.5F` in float: in
   float64 the reference diverged from Tesseract on 20 to 24% of 8-bit lines; in float32 it gives the same best code
   on 100% <!-- @evidence[[results/summary.json#parity.argmax_all_steps_share_min]] --> of the lines at every timestep (`results/parity_tess.json`).
2. **Greedy decoding is not Tesseract's decoding.** Taking the best code per timestep gives Tesseract's text on only
   76% <!-- @evidence[[results/summary.json#parity.greedy_eq_beam_min]] --> to 85% <!-- @evidence[[results/summary.json#parity.greedy_eq_beam_max]] --> of lines, at the same average error
   rate. The port of `RecodeBeamSearch` (no dictionary) in `tessgpu/beam.py` and `web/beam.js` returns Tesseract's ids
   on 1,372/1,372 readings (686 lines, each with tessdata_best and tessdata_fast), <!-- @evidence[[results/summary.json#beam.lines]] --> and the WebGPU engine uses it by
   default. In JavaScript it costs about 0.45 ms a line after a 4x speed-up (`run/bench_beam.mjs`, node).
3. **The WebGPU engine equals the reference on a real GPU for float models.** Mac job `tess-gpu-2`: eng best 343/343
   lines identical with beam decoding; <!-- @evidence[[results/summary.json#mac.best_parity_beam]] --> eng fast 339/343,
   <!-- @evidence[[results/summary.json#mac.fast_parity_beam]] --> where Metal's float rounding (fused multiply-adds)
   flips an 8-bit rounding step and the line diverges from there. Headless SwiftShader matches the reference on every
   line, 8-bit included (`web/test/check_engine.js`).
4. **Speed on a Mac comes from batching every line through each LSTM step.** 343 lines take 176 ms <!-- @evidence[[results/summary.json#mac.best_gpu_ms]] -->
   of GPU time for eng best and 153 ms <!-- @evidence[[results/summary.json#mac.fast_gpu_ms]] --> for eng fast,
   about 2,000 lines a second, against 223 ms in the first build. <!-- @evidence[[results/summary.json#mac.best_gpu_ms_r1]] -->
   The three sequential x-LSTMs take about 105 ms of the 176; one dispatch per timestep covers every line still
   running. Before the beam decoder was sped up it cost 1,339 ms for the same batch on the Mac
   (<!-- @evidence[[results/summary.json#mac.best_beam_decode_ms_r2]] -->job `tess-gpu-2`; not re-measured since).
5. **The server engine is bound by its host-side preparation.** On a Kaggle T4 at 4,096 lines, cuDNN mode reads 1,143 <!-- @evidence[[results/summary.json#t4.best_cudnn_b4096]] -->
   lines/s with eng best and 1,165 <!-- @evidence[[results/summary.json#t4.fast_cudnn_b4096]] --> with eng fast, exact
   mode 431 and 497. Building the convolution windows with tensor ops instead of per-line NumPy took cuDNN mode from
   289 lines/s; <!-- @evidence[[results/summary.json#t4_run1.best_cudnn_b4096]] --> best and fast still run at about the same rate, so the remaining
   limit is the per-line preprocessing (resizing and Tesseract's black and white levels) on the CPU. Exact mode on CUDA matches the reference
   (24/24 lines, max |Δp| 4.6e-6). Exact mode reproduces the reference; cuDNN mode uses PyTorch's `nn.LSTM` (plain tanh and sigmoid, no 8-bit
   activations) and gives the same text as exact mode on 100% of 1,024 best lines and 87% of fast lines.
6. **tessdata_best packs to 8 bits.** CER on the benchmark sample (600 lines, beam decoding): eng 1.26% at f32, 1.26%
   at f16 (identical text), 1.28% with per-row 8-bit weights; <!-- @evidence[[results/summary.json#pack.best_eng_q8]] -->
   fra 2.05/2.05/2.03%, grc 4.90/4.90/5.00%. <!-- @evidence[[results/summary.json#pack.best_grc_q8]] --> 6-bit holds
   for English and French but takes Greek to 5.45%; <!-- @evidence[[results/summary.json#pack.best_grc_q6]] -->
   4-bit breaks (fra 12.2%, grc 19.9%). <!-- @evidence[[results/summary.json#pack.best_fra_q4]] --> eng best at 8 bits is
   1.5 MB against 15.4 MB for the .traineddata file. Google's own 8-bit tessdata_fast scores 1.35/2.21/5.01% with 0.32x of best's compute. These
   rates are on the recognizer alone over the 600-line sample, so they do not compare with the stock-Tesseract figures
   in Method.
7. **The last LSTM is most of the work.** For eng best, `Lfx512` is 73% of the multiply-adds per timestep and the three
   x-direction LSTMs together 81%; <!-- @evidence[[results/summary.json#flops.best_eng_seq_share]] --> eng fast's
   network costs 0.32x as much. <!-- @evidence[[results/summary.json#flops.fast_eng_vs_best]] --> All 320 models share
   one topology (12 size variants; `notes/inventory.md`), so one set of kernels serves every language.
8. **Inverted lines can be caught before the network.** Tesseract reruns a line inverted when its mean best-path
   probability is under 0.7. Checking whether most pixels are darker than halfway between Tesseract's own black and
   white levels flagged 86/86 inverted and 0/600 normal synthetic lines; on 800 real scanned lines it flagged one, which
   has not been inspected. The engines invert such lines up front and run the network once.

## Method

**Reference.** `tessgpu/traineddata.py` reads a .traineddata file (container, network tree, float and 8-bit weight
matrices, unicharset, recoder). `tessgpu/ref.py` is a NumPy forward pass written from Tesseract's source: input
normalisation (`NetworkIO::FromPixes`), the seeded convolution padding, lookup-table tanh and sigmoid, 8-bit
activations, state clipping, the y-summarising LSTM. Parity is against Tesseract 5.3.4 itself: `harness/tess_dump.cpp`
links the library (built from source inside `scripts/jail.sh`: no network, no secrets) and dumps
`LSTMRecognizer::RecognizeLine`'s softmax and `RecodeBeamSearch`'s ids for each line; `harness/tess_layers.cpp` dumps
every layer, which located the padding difference. Lines are resized to the model height (36 or 48 px) before both
sides read them, and Tesseract's `--psm 13` path feeds such a line to the network unscaled.

**Benchmark.** 4,800 synthetic lines: 600 sentences (300 English and 150 French from Wikipedia, 150 Ancient Greek
from Homer via the Perseus Digital Library), each rendered under eight conditions: clean, blur, noise, JPEG, low resolution, rotation, shading, inversion; `notes/data.md`) and 800 real
scanned lines from FUNSD (forms) and SROIE (receipts), two public document datasets (`notes/data_real.md`). Stock Tesseract 5.3.4 (`--oem 1 --psm 7`) scores 1.79% <!-- @evidence[[results/summary.json#baseline_binary.best_cer]] --> CER
with best and 1.92% <!-- @evidence[[results/summary.json#baseline_binary.fast_cer]] --> with fast on the synthetic set.
That score includes Tesseract's page-level post-processing on full-size images. Parity, beam and packing use every
seventh line (686: 343 English, 172 French, 171 Greek, all conditions; `run/parity_prep.py`) resized to the model height
and read by the recognizer alone; packing skips the 86 inverted lines, leaving 600.

**Engines.** WebGPU (`web/`): lines are packed longest first; a fused kernel normalises bytes through a per-line
256-entry table, convolves, applies the 9-to-16 tanh layer and max-pools; the y-LSTM runs every column in parallel; each
x-LSTM step is one dispatch over the lines still running, with gate-interleaved weights; a tiled kernel computes the
softmax; all of it is one submit. `web/traineddata.js` parses any .traineddata in the browser, identical to
`run/pack.py`. PyTorch (`server/`): exact mode mirrors the reference; cuDNN mode maps Tesseract's gates onto
`nn.LSTM` with packed sequences. Checks: `web/test/check_*.js` (per kernel and end to end, headless SwiftShader,
fixtures from `run/make_fixtures.py`), `run/check_torch.py`, `run/check_beam.py`/`.mjs`, `run/check_traineddata.mjs`.
Real-GPU timing: GPU Bridge jobs on Oskar's Mac (`run/bridge/`, results in `results/bridge_tess-gpu-*_mac.json`) and a
Kaggle T4 batch job (`run/kaggle/server_bench.py`, `results/kaggle_t4_server_bench_1.json`).

**Demo page.** Indra OCR (Datalab's 12M-parameter OCR pipeline, ported to WebGPU in this repo's `indra-browser/`) finds and orders the lines and reads math lines
as LaTeX; every other line goes through this engine in one batch. Models load from jsDelivr on first use.

**Delegation.** The work ran as one coordinating session and 17 Claude Haiku subagents, each with a task, a
deliverable and a check it had to pass: benchmark data (2), model inventory, math and image-path surveys, the four
WGSL kernels, the PyTorch engine, the beam-search port and its speed-up, the JS model reader, the output kernel rewrite,
the demo page, note revisions, host-side vectorisation. The 17 agents used about 2.9M tokens, 120k to 390k each. Every kernel passed
its parity check on delivery; the reference, the parity harness, the engine integration and the performance work stayed
with the coordinator.

## Log

### Round 1 (2026-11-19): parity, engines, first numbers

Asked: a GPU version of Tesseract for servers and WebGPU, using the approach of `gliner-browser` and `indra-browser`
(pack an existing model's weights as small as the benchmark allows and run it on hand-written WGSL kernels), with parallel Haiku tasks, plus ideas for image handling, math and the LSTM itself. Context: earlier
the same day `indra-browser` Round 4 found Tesseract passing 17.6% of Omni Parse Bench text tests against Indra's
40.3%, (Omni Parse Bench: Datalab's 2,937-page document benchmark), so this port is about speed and Tesseract's 100+
languages, not accuracy.

Parity path: the first reference matched Tesseract's interior characters but not the line ends; a per-layer dump showed
the convolution's border padding differed (5.3.4's `std::minstd_rand` versus master's LCG, which the reference had
copied). 8-bit lines then diverged on about a fifth of lines until the reference ran float32 op for op.

Engine path: round-1 kernels on the Mac (job `tess-gpu-1`): 223 ms GPU for 343 lines with eng best, of which 60 ms was
the convolution stage scanning every line per thread and filling a 462 MB intermediate buffer. Round 2 fused
convolution and pooling with a binary-searched line lookup, rewrote the y-LSTM with coalesced weights, tiled the output
layer and moved normalisation to the GPU: 176 ms (job `tess-gpu-2`). The x-LSTM kernel with transposed weights (v2)
was not faster than v1 on Apple's GPU; it stays as the default because it matches v1 exactly.

Kaggle T4 (`tess-gpu-server-bench`, 2026-11-19): 214 to 289 lines/s, flat across batch and model; profile points at
per-line NumPy in the host code. After the conv windows were built with batched tensor ops (2026-11-20, version 2 of
the same job): 1,143 lines/s (best, cuDNN) and 431 (best, exact); `results/kaggle_t4_server_bench_2.json`.

Tesseract's own recognizer, in-process on one thread with the model loaded once, took 68.5 ms a line with eng best and
14.1 ms with eng fast on the (busy) container (`results/tess_recognizer_cpu_timing.json`).

Open: vectorising the remaining per-line preprocessing in the server engine; Mac timing after the beam speed-up; 8-bit
dot products (`dot4I8Packed`) for tessdata_fast; f16 weights in the kernels; a CER comparison of the demo pipeline
(Indra layout plus Tesseract recognizer) on Omni Parse Bench pages; distilling the LSTM into a model whose x-direction
layers run in parallel over time, which needs training.
