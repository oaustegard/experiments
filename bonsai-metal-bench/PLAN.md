# bonsai-metal-bench: round 2 tasks for the M5 Max

Round 1 ran on 2026-11-27 (`results/20261127-1800`, `1830`, `1858`); `ane-workloads/RESULTS.md`
covers the Neural Engine half. These tasks follow from what round 1 found. All run on the M5 Max,
plugged in, Energy Mode not Low Power, nothing else heavy running. Do them in order. Each ends with
a results directory committed to `main`; a script bug gets fixed in the script with a one-line note
of what broke, but what is measured does not change.

Reference numbers from round 1 (PQ2_0 unless named): bandwidth probe 549.6 GB/s; decode 47.3 tok/s,
65% of a 72.7 tok/s ceiling (PTQ1_0 44.3, 51% of 87.3); decode at depth 8192 / 32768 44.7 / 35.7;
prefill 809 / 746 / 668 tok/s at 512 / 2048 / 8192 tokens; ubatch 256 to 2048 flat (622 to 643);
MTP 45.8 to 54.3 tok/s (1.19x) at 0.914 acceptance with 1 draft token; CPU memory streaming costs
GPU decode 33 to 37%, a sustained Neural Engine model 7 to 10%.

## Task 1. Write RESULTS.md for round 1

Write `bonsai-metal-bench/RESULTS.md` from the three round-1 runs, in the repo's shape (Answer,
Findings, Method, Log; see the root `CLAUDE.md`), with `<!-- @evidence[[...]] -->` after each number
in Answer and Findings. Run `python3 _lib/results_lint.py bonsai-metal-bench/RESULTS.md` and
`python3 _lib/evidence.py bonsai-metal-bench/RESULTS.md`; both must pass. Update the root
`README.md` index row (status: done, round 1) and add `METHODS.md` entries for what holds beyond
this experiment: CPU memory streaming costs GPU decode about a third on Apple Silicon, a
Neural Engine model running continuously costs it 7 to 10%, and Bonsai's MTP head gains 1.19x at
91% acceptance on Metal. Write it for a reader who has never seen this session.

## Task 2. Is prefill held back by the matrix-unit (tensor API) path?

Round 1's logs say `has tensor = true` with the embedded library, so the M5's matrix units were
available. 700 tok/s is about 34 TFLOPS (2 x 24.35B parameters per token), which is low for them.
Find out whether the tensor path helps PQ2_0 at all.

- With the round-1 fork build, run `llama-bench -m <PQ2_0> -ngl 99 -fa 1 -p 512,2048,8192 -n 0 -r 3 -o json`
  twice: as is, and with `GGML_METAL_TENSOR_DISABLE=1`. Repeat both for PTQ1_0. Then repeat the
  ubatch sweep (`-p 8192 -n 0 -b 2048 -ub 256,512,1024,2048 -r 2`) both ways, PQ2_0 only.
- Keep each run's stderr. Note which mul_mm / mul_mm_id kernels it reports, and any line about
  `ggml-tensor.metallib` or the tensor API being disabled.
- Comparison point: upstream llama.cpp (release b11556 or newer, which runs the `Q2_0` file) built
  with Metal, same prefill sizes, tensor API on and off.
- Report prefill tok/s and the implied TFLOPS for every combination, in one table.
- If the numbers barely move with the tensor path off, the PQ2_0 matmul isn't using the matrix
  units effectively, and a fused dequant-plus-`matmul2d` kernel is the work. If they drop a lot,
  the units are in use, and the bottleneck is elsewhere: dequantisation, the Hadamard transform,
  or attention.

## Task 3. Why MTP gains only 1.19x at 91% acceptance

With one drafted token accepted 91% of the time, each verify step should yield about 1.9 tokens,
so a nearly free verify would approach 1.9x. Measure where the gap goes.

- `llama-server` on `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf`, the round-1 prompts (`bench.py --only mtp`
  does this), with `--spec-draft-n-max` 1, 2 and 3, and `--spec-draft-p-min` 0.6 and 0.3. Record
  decode tok/s, `draft_n`, `draft_n_accepted`.
- `presets/llama.cpp/mtp-metal.json` (in the GGUF repo) lists env vars for kernels "in testing":
  `GGML_METAL_PQ2_COEF=1`, `LLAMA_MTP_PREFILL_LEAN=1`, `GGML_METAL_SSM_CONV_ROWS=1`. Grep the fork
  for each one; for those present, rerun the n-max=1 case with them set.
- For each setting compute the verify cost: (tokens per step) / (speedup). A value near 1 means
  verifying is nearly free; the round-1 setting works out to about 1.6.
- Report one table, plus a plain statement of the best setting found.

## Task 4. Does 8-bit KV pay at long context?

Bonsai's weights are small, so the attention KV cache (64 KiB per token in fp16 over 16 full-attention
layers) is a large share of what decode reads at long context. Round 1 saw decode fall from 47.3 to
35.7 tok/s between depth 0 and 32768.

- `llama-bench -m <PQ2_0> -ngl 99 -fa 1 -p 0 -n 128 -d 0,8192,32768,65536 -r 2`, with
  `-ctk f16 -ctv f16` and with `-ctk q8_0 -ctv q8_0`.
- Prediction to test: at depth 32768 the KV is about 2.1 GB of about 9.7 GB read per token in fp16,
  so q8_0 should give roughly +10 to +12% there, about +3% at 8192, and nothing at depth 0.
- Report tok/s for both, the measured gain against that prediction, and any quality warning the
  fork prints for q8_0 KV.

## Task 5 (optional, larger). OCR on the Neural Engine with Indra instead of Vision

`ane-workloads` found Apple Vision scores 39.8 on Omni Parse Bench against Indra's 78.8, and the
loss is in Vision's recogniser. Indra (about 12M parameters: line detector, per-script CTC
recognisers, reading-order transformer; see `indra-browser/`) is small enough for the Neural Engine.

- New experiment `indra-coreml/`: convert Indra's detector, Latin recogniser and reading-order model
  to Core ML at fixed shapes (`crossenc_coreml.py` in `ane-workloads` shows the conversion and the
  placement check). The word model and the other scripts can stay on the CPU.
- Done when: Core ML output matches the existing reference (`indra-browser/run/ref.py`) character for
  character on its 15 reference pages, and it is scored on the same 2,937 pages and grader as
  `ane-workloads/ocr_ane.py`, with ms per page with the GPU idle and busy, op placement, and the
  score beside Vision's 39.8 and Indra's 78.8.
- Write it up as its own `RESULTS.md`.

## What runs elsewhere

The context filter's fact-recall test (`ane-context-filter`) needs the private transcript data and
runs in the cloud session, not here.
