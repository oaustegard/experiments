# Quantization and drafter costs for Baguettotron

Follow-up to [`RESULTS.md`](RESULTS.md), which found Monad too slow to draft for
Baguettotron. Two questions: what would 4-bit quantization take, and what would a
purpose-built drafter take.

Short version. The 4-bit quants already exist, PleIAs ships them officially, and
on this CPU they buy 1.19× at best — the runtime is a bigger lever than the bit
width. A drafter is the opposite case: an EAGLE-style head measures at cost ratio
c = 0.059 against Monad's 0.476, which is the difference between 0.90× and a
projected 1.9–2.8×. The cost is a few GPU-hours, and the training data harvest is
what dominates it.

## a) 4-bit quantization

Already published, so the build cost is zero. What is on the Hub:

| Artifact | What |
|---|---|
| [`PleIAs/Baguettotron-GGUF`](https://huggingface.co/PleIAs/Baguettotron-GGUF) | Official. Q4_0, Q4_1, Q4_K_M, Q4_K_S, IQ4_NL, IQ4_XS, plus Q5/Q6/Q8/F16/BF16 |
| `PrunaAI/PleIAs-Baguettotron-HQQ-4bit-smashed` | HQQ 4-bit |
| `keisuke-miyako/Baguettotron-onnx-int4` | ONNX int4 |
| `usermma/Baguettotron-mlx-{2,3,4,5,6,8}Bit` | MLX ladder |
| `jncraton/Baguettotron-ct2-int8` | CTranslate2 int8 |

Building one anyway would cost minutes: `llama-quantize` over 321M parameters
needs no GPU, no training and no calibration data for the round-to-nearest
formats. An imatrix quant wants a few hundred MB of calibration text and still
finishes inside an hour on CPU.

### Measured throughput

llama.cpp, 4 threads, batch 1, 48 tokens, same machine as everything else here.

| Quant | File | ms/token | tok/s | vs Q8_0 |
|---|---|---|---|---|
| Q8_0 | 344 MB | 28.3 | 35.3 | 1.00× |
| Q4_K_M | 240 MB | 29.3 | 34.1 | 0.97× |
| Q4_0 | 202 MB | 23.9 | 41.9 | 1.19× |

Q4_K_M is slower than Q8_0. Its per-block scales and mins cost more to unpack
than the halved weight traffic saves at this size.

The reason the whole ladder is flat: decode here is not weight-bandwidth-bound.
Halving the weight bytes in torch by moving fp32 → bf16, which leaves FLOPs and
per-layer overhead untouched, gives 1.13× on Baguettotron and 1.11× on Monad, so
weight traffic is only 19–23% of decode time. Taking the weights to zero bytes
would cap out near 1.3×.

Precision is the small lever. The runtime is the large one — llama.cpp Q8_0 runs
at 28.3 ms/token against torch fp32's 111 ms on the same cores, 3.9× for no
change in precision at all.

## b) A drafter

Monad loses because c = 0.476. An EAGLE-style head is a different object: one FC
layer (2h → h) plus one decoder layer, reusing the target's embedding and LM head
([Li et al. 2401.15077](https://arxiv.org/abs/2401.15077)). It shares the
target's tokenizer by construction, so the 1.27 draft-steps-per-target-token
penalty from `RESULTS.md` disappears.

Built against Baguettotron's real config and timed on this CPU:

| Component | ms | Note |
|---|---|---|
| Embedding lookup | 0.010 | reused from target |
| FC 1152 → 576 | 0.070 | trained |
| One decoder layer | 1.43 | trained |
| LM head, 65,536-wide | 5.51 | reused from target |
| **Draft step** | **7.01** | |
| **Target step (80 layers)** | **118.7** | |
| **c** | **0.059** | vs Monad's 0.476 |

Trainable parameters: **4.2M**, 1.3% of the target. Baguettotron's 80×576 shape
makes a single decoder layer unusually cheap in proportion — the published EAGLE
drafts run 3–5% of their targets.

The LM head is 78% of that draft step. A 576-wide hidden state makes everything
else nearly free, so the 65,536-token output projection is the whole cost.
NVIDIA's NeMo recipe cuts the draft head to a smaller vocabulary for this reason,
and it is the largest single lever available here:

| Draft vocab | Draft step | c | Projected speedup at α=0.6 / 0.7 |
|---|---|---|---|
| 65,536 | 9.55 ms | 0.081 | 1.75× / 2.10× |
| 32,000 | 3.87 ms | 0.033 | 2.05× / 2.56× |
| 16,384 | 2.21 ms | 0.019 | 2.19× / 2.78× |
| 8,192 | 1.35 ms | 0.011 | 2.28× / 2.93× |
| 4,096 | 1.17 ms | 0.010 | 2.30× / 2.96× |

Returns flatten below 16k because the decoder layer becomes the floor. On English
technical prose the 8,192 most frequent types cover 98.75% of token occurrences
and 16,384 cover all of them, so a 16k draft head gives up nothing measurable on
that domain. That corpus is narrow and coverage moves with domain.

α is assumed here, not measured. Everything above is a cost measurement.

### Acceptance at this scale

Nobody has published EAGLE or Medusa on a dense target under 1.7B
([EAGLE model table](https://github.com/SafeAILab/EAGLE)), so 321M is untested.
The trend across scale gives no reason to expect a fall-off: Tencent AngelSlim's
EAGLE-3 sweep on one H20 at batch 1 measured Qwen3-1.7B at **1.69× with
acceptance length 2.17**, the highest in a series running to 32B
([model card](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)).

Overhead is the failure mode to watch. Speculative decoding on already-fast
small models has lost before: five draft/target configs on an
M3-class laptop had **three decelerate**, with every 1.5B-target config below
break-even ([2607.17283](https://arxiv.org/html/2607.17283)); EAGLE-3 on
M3 Ultra with a 4-bit 8B target measured 1.05×, and 0.94× with an fp16 draft
([mlx-lm #890](https://github.com/ml-explore/mlx-lm/discussions/890)). The
measured c = 0.059 is what says this case is not that one.

### Training cost

Published reference points, all on targets 20×+ larger:

| Recipe | Data | Compute |
|---|---|---|
| Medusa-1, Vicuna-7B | 60k ShareGPT samples | **5 h on one A100** ([2401.10774](https://arxiv.org/abs/2401.10774)) |
| EAGLE, LLaMA2-Chat-70B | 68k ShareGPT dialogues | 1–2 days on 4× A100 ([2401.15077](https://arxiv.org/abs/2401.15077)) |
| NeMo EAGLE-3, Llama-3.1-8B | 200k samples, 1 epoch | ~2 h on 8× A100 |
| Kimi K2.5 EAGLE-3 | 600k samples, 6B tokens | 1500 H200-hours |

The frozen-target forward passes dominate, and Baguettotron is 22× smaller than
Vicuna-7B, so the Medusa-1 reference scales to well under one A100-hour for the
same data volume. Call a first drafter **a few single-GPU hours end to end**,
dominated by harvesting hidden states rather than by the optimizer — 4.2M
trainable parameters is nothing.

Two caveats on the data. EAGLE's original paper reports that target-regenerated
answers give "only a slight improvement" over fixed ShareGPT text, but SpecForge
measured that claim failing at scale: moving from ShareGPT+UltraChat to 1.4M
regenerated conversations lifted acceptance length 2.82 → 3.48, worth another
1.17× ([2603.18567](https://arxiv.org/pdf/2603.18567)). Regenerating is the
expensive path, and it is the one that works.

### Doing it on this container

Harvesting hidden states is a batched forward pass, so the 9 tok/s decode rate is
not the constraint. Measured on 4 cores:

| Batch × seqlen | tokens/s |
|---|---|
| 1 × 512 | 192.0 |
| 4 × 512 | 169.8 |
| 8 × 512 | 141.9 |
| 8 × 1024 | 132.2 |

Throughput falls as batch grows — 4 cores are already saturated at batch 1,
seqlen 512, and larger batches only add cache pressure.

At 192 tok/s a 50M-token harvest takes **72 hours**, which outruns a CCotw
session. Generating a regenerated corpus is worse: 35 tok/s through llama.cpp
puts 50M generated tokens at 16 days. The training itself is trivial anywhere;
the data is what needs a rented GPU.

## Files

| File | What |
|---|---|
| `bench_gguf.py` | llama.cpp quant ladder → `gguf_latency.json` |
| `dtype_scaling.py` | fp32 vs bf16, which isolates the weight-bandwidth share → `dtype_scaling.json` |
| `eagle_cost.py` | EAGLE draft-step components vs target step → `eagle_cost.json` |
| `eagle_vocab.py` | Draft cost vs draft-vocabulary size → `eagle_vocab.json` |
| `vocab_coverage.py` | Token coverage of a reduced draft vocabulary → `vocab_coverage.json` |
| `train_feasibility.py` | Hidden-state harvest throughput on 4 cores → `train_feasibility.json` |

Measurements are noisy at the ±20-30% level on shared cores; the 65,536 LM head
timed 5.51 ms in `eagle_cost.py` and 8.54 ms in `eagle_vocab.py`. Every conclusion
here rests on a ratio wide enough to survive that spread.
