# inkling-8gb-m1 — running Inkling 2 Mini on an 8 GB M1 MacBook Air

Inkling 2 Mini (Thinking Machines, 20 Nov 2026) is a 26.4B-parameter mixture-of-experts model: every
layer has 128 experts and each token uses 8 of them. Its makers ask for 32 GB of memory. The question
was whether it can be squeezed onto an 8 GB M1 MacBook Air, first by running the smallest published
quantization, then by deleting the experts the model uses least so the file fits in memory.

Sibling experiment: [`inkling-rlm-lora/`](../inkling-rlm-lora/README.md) (LoRA post-training of the
same model, a different question).

## Answer

It runs but doesn't fit, and pruning doesn't make it fit at acceptable quality. Reference: unsloth's UD-Q4_K_XL ("Q4"); top-token agreement is how often a model's likeliest next token equals Q4's.

- The smallest quant (UD-IQ1_S, 7.76 GB) generates up to 4.6 tokens/s with Metal off and a small batch; speed depends on how much of the file macOS still caches. <!-- @evidence[[results/summary.json#speed_m1.iq1s_device_none_b128_tps_best]] -->
- llama.cpp defaults gave 0.8 tokens/s: `-ngl 0` (no layers on the GPU) still reserves a 1.9 GB Metal buffer. <!-- @evidence[[results/summary.json#speed_m1.iq1s_ngl0_default_tps]] -->
- Keeping 64 of 128 experts doubles speed to 9.5 tokens/s. <!-- @evidence[[results/summary.json#speed_m1.pruned64_device_none_tps]] -->
- But experts chosen from general text keep only 76.6% agreement on held-out code (unpruned 92.6%). <!-- @evidence[[results/summary.json#quality_colab.general_selected.64.code_same_top_vs_q4]] -->
- Experts chosen from code keep code far better but wreck prose: WikiText perplexity 50.6. An equal-weight blend kept neither well. <!-- @evidence[[results/summary.json#quality_colab.code_selected.64.wiki_ppl]] -->
- The faster option that fits is Nemotron 3.5 Bolt E4B: 15.8 tokens/s, under 3 GB in RAM; quality not compared head to head. <!-- @evidence[[results/summary.json#speed_m1.nemotron_bolt_e4b_metal_tps]] -->

What would change this: training the routers to work within a 48-expert subset (untested; last Log round).

## Findings

1. **`-ngl 0` does not keep llama.cpp off the GPU.** It still reserved a 1856.79 MiB Metal compute buffer at a 2K context (2492.79 MiB at 8K); on a unified-memory Mac that is RAM taken from the model. <!-- @evidence[[results/summary.json#speed_m1.metal_compute_buffer_mib_ngl0_ctx2048]] -->
   `--device none -b 128 -ub 128` left a 145.43 MiB CPU buffer and raised generation from 0.8 to 1.8–4.6 tokens/s; the two flags were changed together, so which one matters is not isolated. (Round 3) <!-- @evidence[[results/summary.json#speed_m1.cpu_compute_buffer_mib_device_none_b128]] -->
2. **Expert usage is not skewed enough to prune.** The 48 most-used experts per layer account for a mean 0.611 of all expert selections on calibration text (64 account for 0.726), so pruning to a size that fits redirects 27–39% of the model's choices. (Round 4) <!-- @evidence[[results/summary.json#quality_colab.general_selected.48.coverage_mean]] -->
3. **General-purpose pruning degrades fast.** Held-out code perplexity goes from 2.48 unpruned to 4.18 at 64 experts and 5.83 at 48. (Round 4) <!-- @evidence[[results/summary.json#quality_colab.general_selected.48.code_ppl]] -->
4. **Code and prose use different experts.** Choosing from code usage holds code perplexity at 2.98 at 64 experts while WikiText perplexity becomes 50.6. (Round 5) <!-- @evidence[[results/summary.json#quality_colab.code_selected.64.code_ppl]] -->
   A blend at 64 gives WikiText 22.0, still three times unpruned IQ1_S's 7.47. (Round 6) <!-- @evidence[[results/summary.json#quality_colab.blended_selected.64.wiki_ppl]] -->
5. **The quantization hurts code less than prose.** On held-out code UD-IQ1_S matches the 4-bit build's top token 92.6% of the time, against 85.6% on WikiText. (Round 4) <!-- @evidence[[results/summary.json#quality_colab.code_iq1s_same_top_vs_q4]] -->
6. **A model that fits is far faster.** Nemotron 3.5 Bolt E4B (6.18 GB file, 2.87 GB in RAM, presumably because Gemma 4's per-layer embedding table is read a row at a time) ran at 15.8 tokens/s with Metal and 12.1 without: Metal-off pays only when the model does not fit. (Round 7) <!-- @evidence[[results/summary.json#speed_m1.nemotron_bolt_e4b_cpu_tps]] -->

## Method

- **Hardware.** M1 MacBook Air, 8 GB, macOS 27.2. Chrome, System Settings, Stocks and LogiTune were quit
  first; Apple Intelligence's on-device model (`TGOnDeviceInferenceProviderService`, ~4 GB footprint,
  mostly swapped) could not be stopped. Colab L4 (24 GB) for quality measurements.
- **Runtime.** Mainline llama.cpp at e50c254 (includes the inkling2_loop fix, PR #32704, merged
  2026-11-21). Model files are unsloth's `Mainline_Support/` layout, renamed from `.gguf_file`.
- **Speed.** `llama-cli -st` on one prompt (`prompt_code.txt` or a Fibonacci prompt), reporting llama.cpp's
  generation tokens/s. Single runs; page-cache state moves the 7.76 GB file between 2 and 4.6 tokens/s.
- **Expert usage.** `expert-usage.patch` adds an eval callback to `llama-perplexity` that counts the
  `ffn_moe_topk-N` selections per layer slot (`LLAMA_EXPERT_USAGE=out.tsv`). Loop passes (slots 24–47) are
  folded onto their physical layers 6–17: layers 6–17 run three times per token with shared weights, and llama.cpp gives
  each extra pass its own slot number. General usage: 160 × 512 tokens of bartowski's calibration-v6
  text. Code usage: 300 × 512 tokens from 57 older files in local repos (not committed; some are private).
- **Pruning.** `prune_tools.py keep` picks the K most-used experts per layer (blends several usage files
  with equal weight). `mask` sets the pruned experts' selection bias (`exp_probs_b`) to −1e9 in place, so
  quality can be measured without rewriting the file. `slice` writes a smaller GGUF keeping only those
  experts' rows and setting `expert_count` to K; llama.cpp loads it unchanged.
- **Reproducibility.** The code calibration set is not committed (it includes private repos), so the code-selected and
  blended rows cannot be reproduced exactly from this folder.
- **Quality.** `llama-perplexity --kl-divergence` against logits saved from UD-Q4_K_XL, 512-token chunks:
  WikiText-2 test (40 chunks) and held-out code (42 files added to oaustegard's public repos since
  1 Nov 2026, listed in `results/heldout_code_files.txt`, vendored Doom source excluded). Metrics:
  perplexity and top-token agreement with Q4. "Routing covered" is the share of all expert selections in the usage
  counts that land on the kept experts, averaged over layers.

## Log

### Round 1 — does it run at all (2026-11-21)

7.76 GB UD-IQ1_S, CPU only, `-ngl 0`, 4K context: 0.8 tokens/s, correct iterative Fibonacci. Attention
on Metal with experts on CPU (`--cpu-moe`): 0.4 tokens/s. Everything on Metal: GPU out of memory.

### Round 2 — expert pruning plan and tools

Inspected the GGUF: expert tensors are 6.53 GB of the 7.76 GB. Each token reads up to about 800 MB of
experts, and only about 4.5 GB of RAM was free, so a large share comes from the SSD every token. Wrote the usage recorder (the
top-k tensor is a strided view of the argsort; reading it as contiguous gave garbage) and the slicer (the
`gguf` writer needs the raw bytes passed as int8 so it takes the element shape as given). A dummy file
keeping every other expert loaded and generated coherent text at 4.49 GB.

### Round 3 — the Metal buffer

The dummy 64-expert file still page-faulted at 1.5 tokens/s. `-v` showed `MTL0 compute buffer size =
1856.79 MiB` with `-ngl 0`. With `--device none -b 128 -ub 128`: dummy 64-expert file 8.8 tokens/s, full
file 4.1. Thread sweep on the full file: 4 threads 2.3, 6 threads 2.1, 8 threads 1.8 tokens/s. Also:
`llama-perplexity` returns NaN for this model at a 64-token context; 512 works.

### Round 4 — general-text expert selection, quality sweep (Colab L4)

Sanity: Q4 WikiText perplexity 6.77, IQ1_S 7.47, IQ1_S same top token as Q4 85.6%. Held-out code: Q4 2.37,
IQ1_S 2.48, agreement 92.6%. The first sweep attempt failed because pip's `gguf` package does not know
type 42 (NVFP4); the repo's `gguf-py` does. File sizes for K other than 64 are computed from the measured
64-expert file.

| Experts | File (GB) | Routing covered | Code ppl | Code top-1 vs Q4 | Wiki ppl | Wiki top-1 vs Q4 |
|---|---|---|---|---|---|---|
| 128 | 7.76 | 100% | 2.48 | 92.6% | 7.47 | 85.6% |
| 96 | 6.13 | 90.3% | 2.81 | 87.5% | 8.09 | 82.6% |
| 80 | 5.30 | 82.3% | 3.33 | 82.3% | 9.46 | 77.1% |
| 64 | 4.49 | 72.6% | 4.18 | 76.6% | 11.40 | 72.0% |
| 56 | 4.08 | 67.1% | 4.79 | 73.2% | 13.03 | 67.5% |
| 48 | 3.68 | 61.1% | 5.83 | 68.9% | 14.97 | 64.3% |
| 40 | 3.26 | 54.6% | 7.13 | 65.2% | 19.51 | 59.0% |
| 32 | 2.85 | 47.4% | 9.65 | 60.1% | 26.53 | 53.1% |

Prediction before the run: code perplexity above 3 at 64 experts and below. It held (4.18).

### Round 5 — code-selected experts

| Experts | Routing covered (code) | Code ppl | Code top-1 vs Q4 | Wiki ppl | Wiki top-1 vs Q4 |
|---|---|---|---|---|---|
| 96 | 95.2% | 2.55 | 91.0% | 13.85 | 65.4% |
| 64 | 83.2% | 2.98 | 86.2% | 50.59 | 43.2% |
| 48 | 73.6% | 3.59 | 80.9% | 91.55 | 35.8% |

### Round 6 — blended selection and a side-by-side on the M1

Equal-weight blend of code and general usage:

| Experts | Code ppl | Code top-1 vs Q4 | Wiki ppl | Wiki top-1 vs Q4 |
|---|---|---|---|---|
| 72 | 2.80 | 87.4% | 17.38 | 60.9% |
| 64 | 2.98 | 85.5% | 22.04 | 56.4% |
| 56 | 3.18 | 83.7% | 29.08 | 51.5% |
| 48 | 3.55 | 80.7% | 37.97 | 47.8% |

Sliced the blended 64-expert file (4.49 GB) and ran `prompt_code.txt` through it and the full file, same
seed. Pruned: 9.5 tokens/s, reasoning visibly muddled (a draft loop that never appended anything).
Full: 4.6 tokens/s, clean reasoning and a correct draft. Both hit the 450-token cap before finishing.

### Round 7 — the strongest model that does fit (2026-11-22)

Not Inkling. Among Nov 2026 releases, nothing between 4B and 12B fits in 8 GB (Qwen 3.8 has no small
sizes; Gemma 4 12B is 7 GB at 4-bit). Nemotron 3.5 Bolt E4B (NVIDIA's Q4_K_M, 6.18 GB) on the same prompt:
15.8 tokens/s with `-ngl 99 -b 256 -ub 256`, 2.87 GB resident, complete and correct answer; 12.1 tokens/s
CPU-only. NVIDIA's own table puts it ahead of Gemma 4 E4B and Qwen 3.5 4B on most instruction-following
and multi-turn tests; Qwen 3.5 4B leads on long context and Tau2-Bench.

Next step proposed, not run: freeze everything and train only the routers to imitate the full model
within the blended 48-expert set, on held-out code. Needs the 20 GB checkpoint in PyTorch on a GPU with more
than 24 GB. Go on if code perplexity falls from 3.55 (blended 48, round 6) toward 2.5–2.8 without WikiText getting worse.

Cost: about 4.5 Colab L4 hours (balance afterwards 288.3 units).
