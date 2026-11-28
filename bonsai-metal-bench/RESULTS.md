# RESULTS: Ternary Bonsai 2.5 27B on an M5 Max, against the memory-bandwidth ceiling

Ternary Bonsai 2.5 is PrismML's 1.76-bit compression of Qwen3.8-27B. This experiment ran it on one
Apple M5 Max (40 GPU cores, 128 GB, macOS 27.2.1, plugged in, Energy Mode Automatic) through
PrismML's llama.cpp fork and through MLX. The question: how far from the machine's memory-bandwidth
ceiling does decoding run, and where is the time going, so we know which kernel work would pay
before anyone writes any. It is the first run of [`bench.py`](bench.py); one machine, one run per
phase, so differences under about 5% are noise. Raw output is in `results/`.

## Answer

Decoding PQ2_0 with the fork runs at 47.3 tok/s, 65% of the ceiling that the measured 549.6 GB/s allows. <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#llama.PQ2_0.decode.rows.0.avg_ts]] -->
The file-size ceiling slightly overstates the bytes, so the real gap is somewhat smaller than 35%,
and this run did not profile where the rest goes. Prefill runs on the M5's matrix units: switching the tensor API off cuts it from 802 to 268 tok/s at 512 tokens. <!-- @evidence[[followups/tensor_ab.json#tensor_on.pp512.mean]] -->
Prefill tile size does nothing (622 to 643 tok/s across ubatch 256 to 2048). Multi-token prediction speeds decode 1.19x. <!-- @evidence[[results/20261127-1830-Apple-M5-Max/summary.json#mtp.mtp_decode_ratio]] -->
Of the background loads tried, only CPU memory streaming slows GPU decode (33 to 37%); the Neural
Engine, CPU arithmetic and CPU matrix math are within noise. The synthetic Core ML encoder ran
entirely on the Neural Engine.

What would change this: the same run on another chip, or a profiler capture showing what decode
waits on.

## Findings

1. **Decode is 65% of the bandwidth ceiling, and the smaller file is not faster.** PQ2_0 (7.25 GB): 47.3 tok/s against a 72.7 ceiling. <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#derived.PQ2_0.ceiling_tg0]] -->
   PTQ1_0 (5.99 GB, 1.76 bits per weight): 44.3 against 87.3, 51%. <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#llama.PTQ1_0.decode.rows.0.avg_ts]] -->
   (`results/20261127-1800-Apple-M5-Max/summary.json`, `llama.*.decode`, `derived`)
2. **Prefill uses the matrix units, for about 3x.** `llama-bench` with the Metal tensor API on
   against `GGML_METAL_TENSOR_DISABLE=1`: 802 vs 268 tok/s at 512 tokens, 741 vs 257 at 2048, two
   rounds each. The device log says `has tensor = true` and the source agrees: `kernel_mul_mm`
   takes a `matmul2d` (MetalPerformancePrimitives) path when `GGML_METAL_HAS_TENSOR` is set, and the
   fork turns it on by default only for M5, M6, A19 and A20. The off figure at 512 tokens: 268 tok/s. <!-- @evidence[[followups/tensor_ab.json#tensor_off.pp512.mean]] -->
   (`followups/tensor_ab.json`; fork file `ggml/src/ggml-metal/ggml-metal-device.m`)
3. **Prefill tile size is flat; prompt length costs 17%.** 8192-token prompt at ubatch 256, 512,
   1024: 622, 643, 636 tok/s. <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#llama.PQ2_0.ubatch_sweep.rows.1.avg_ts]] -->
   Prefill falls from 809 tok/s at 512 tokens to 668 at 8192. <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#llama.PQ2_0.prefill.rows.2.avg_ts]] -->
   (`results/20261127-1800-Apple-M5-Max/summary.json`, `llama.PQ2_0.ubatch_sweep`, `prefill`)
4. **Decode slows with context.** 47.3 tok/s at depth 0, 44.7 at 8192, 35.7 at 32768 (PQ2_0). <!-- @evidence[[results/20261127-1800-Apple-M5-Max/summary.json#llama.PQ2_0.decode.rows.2.avg_ts]] -->
   (`results/20261127-1800-Apple-M5-Max/summary.json`, `llama.PQ2_0.decode`)
5. **MTP helps decode by 19% and leaves prefill alone.** `llama-server` on PQ2_0-mtp, six prompts:
   45.8 tok/s off, 54.3 on, draft acceptance 0.914. <!-- @evidence[[results/20261127-1830-Apple-M5-Max/summary.json#mtp.mtp_on.draft_acceptance]] -->
   The 8K prefill is 632 and 673 tok/s. (`results/20261127-1830-Apple-M5-Max/summary.json`, `mtp`)
6. **Only CPU memory streaming slows GPU decode.** Decode against the mean of the run's first and
   last baseline, two runs: Neural Engine encoder -4.6% and +1.0%; CPU arithmetic -4.3% and +0.3%;
   CPU matrix math (Accelerate) -4.2% and +1.3%; CPU memory streaming -37.4% and -33.2%; Neural
   Engine plus memory streaming -37.6% and -36.1%. The baselines of one run differ by 1 to 5%, so
   the first three rows are within noise. Prefill (2048): memory streaming -21% to -24%; the others
   -8% to +4%. The loads slow too: the Neural Engine encoder by 3 to 4%, memory streaming 25%, CPU
   arithmetic and matrix math 18 to 28%. Memory-streaming decode in the second run: 30.1 tok/s. <!-- @evidence[[results/20261127-1858-Apple-M5-Max/summary.json#contention.cpu_mem.tg128]] -->
   (`results/20261127-1800-Apple-M5-Max/summary.json` and `results/20261127-1858-Apple-M5-Max/summary.json`, `contention`)
7. **The synthetic Core ML encoder is fully on the Neural Engine, and it is 3x faster there than on
   the CPU.** The 140 operations the first read listed as having no device are constants. 0.52 ms per
   call on the Neural Engine, 1.54 on the CPU. All 74 compute operations are on the Neural Engine. <!-- @evidence[[results/20261127-1858-Apple-M5-Max/summary.json#contention.ane_setup.placement.MLNeuralEngineComputeDevice]] -->
   (`followups/synth_placement.txt`, `results/20261127-1858-Apple-M5-Max/summary.json`)
8. **MLX is in the same range as the fork, with a different shape.** Median of three runs after a
   512-token prompt: 48.5 tok/s decode, 835 prefill; after 32768 tokens, 36.8 decode, 617 prefill.
   The MTP head gives 1.11x (43.9 to 48.9 tok/s). The conditions differ from `llama-bench` (prompt
   then decode, rather than decode at a set depth), so this is a comparison of range, not of
   kernels. Decode after the 512-token prompt: 48.5 tok/s. <!-- @evidence[[results/20261127-1830-Apple-M5-Max/summary.json#mlx.runs.512.1.generation_tps]] -->
   (`results/20261127-1830-Apple-M5-Max/summary.json`, `mlx`)

## Method

`bench.py` runs five phases, each a subset of `--only bw|llama|mtp|contention|mlx`.
`bw`: MLX fp16 matvec and sum over 4 GiB; the best of the two is the bandwidth. `llama`: the
PrismML llama.cpp fork (commit 43f31f4, built from source) `llama-bench` with `-ngl 99 -fa 1` on
PQ2_0 and PTQ1_0: prefill at 512, 2048 and 8192 tokens (3 reps), 128-token decode at depth 0, 8192
and 32768 (2 reps), and an 8192-token prefill at ubatch 256 to 2048. The decode ceiling is
bandwidth divided by the GGUF size plus the Gated DeltaNet state read and written plus 64 KiB of
fp16 KV per token of depth. `mtp`: `llama-server` on PQ2_0-mtp with the released Metal preset,
speculation off and on, six prompts plus one 8K prefill. `contention`: `llama-bench` pp2048 and
tg128 on PQ2_0 alone, then with one background load at a time (a 6-layer width-384 Core ML encoder
on the Neural Engine; 6 processes each of CPU memory streaming, NEON arithmetic and Accelerate
sgemm; Neural Engine plus memory streaming), with a baseline before and after. `mlx`: the prism-ml
MLX pack through its loader, three runs per prompt size, MTP head off and on.
Model files come from `prism-ml/Ternary-Bonsai-2.5-27B-GGUF`. Plugged in, Energy Mode Automatic,
no other load running. The follow-ups (`followups/`) ran afterwards on an idle machine.

## Log

### Round 1: the first Mac run, 2026-11-27

Three runs: `bw`, `llama` and `contention` together (`20261127-1800-Apple-M5-Max`, 22 min); `mtp`
and `mlx` (`20261127-1830-Apple-M5-Max`, 28 min); `contention` again (`20261127-1858-Apple-M5-Max`,
7 min). Tables and numbers are in each run's `summary.md`.

What broke. The synthetic encoder's placement check failed in the first run: it loaded the model
inside a call chain that let coremltools free the compiled `.mlmodelc` before `MLComputePlan` read
it. One line in `bench.py` now keeps the model object alive; nothing measured changed. The first
run's `contention` record still carries the error (`ane_setup.placement_error`); `contention` was
run again for the placement, and the second run's baselines (44.0 and 46.1 tok/s decode, against
47.6 and 47.3 in the first) show the run-to-run spread. The `bw` phase was not selected in the
later runs, so their `summary.md` shows no bandwidth; the ceilings come from the first run.
The MLX figures in `summary.md` are the best of three; this file uses the median.

### Round 2: follow-ups on an idle machine, 2026-11-27

The first read of the placement showed 140 operations with no device. `followups/synth_placement.py`
rebuilds the model and counts by operation type: the 140 are `const`, the 74 compute operations
(12 per layer for 6 layers, plus 2 casts) are all on the Neural Engine. The first loop did not skip
constants, which `crossenc_coreml.py` does.

Whether prefill uses the matrix units was an open question after the first run. The fork's source
sets the tensor path by device name, and the device log of the run says it was on. `followups/tensor_ab.json`
is the direct check: tensor API disabled, prefill falls to a third. That the Gated DeltaNet kernel
(`kernels/gated_delta_net.metal`) has no matrix-unit code is from reading the source, not from a
profile. It is one reason prefill might be flat in ubatch size; nothing here shows it is the reason.
