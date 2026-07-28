# Verifying LFM2.5-230M claims by actually running the model

**Date:** 2026-06-25
**Trigger:** Bluesky claim to test — [@sungkim.bsky.social](https://bsky.app/profile/sungkim.bsky.social/post/3mp4zqketts25)

## The claim

> Liquid AI's LFM2.5-230M (open-weight and run fast anywhere (CPUs, NPUs, and GPUs)
> - 230M parameters, built on the LFM2 architecture
> - Pre-trained on 19T tokens, with a 32K context extension
> - Post-trained with distillation from LFM2.5-350M

## Method

Spec-sheet checking (reading the HF model card) is not verification. So we
loaded the actual weights and ran inference on the container CPU.

- Environment: 4 vCPU, 15 GB RAM, `torch` CPU build, `transformers` 5.12.1
- Script: [`run.py`](run.py) — downloads `LiquidAI/LFM2.5-230M`, counts params
  from loaded tensors, reports config, runs a real generation, times tok/s.

## Results (ran it)

```
architecture class : Lfm2ForCausalLM
config model_type  : lfm2
param count        : 229,693,184  (229.7M)
max_position_emb   : 128000
weights in RAM     : 919 MB (fp32)
load time          : 2.4s
CPU inference      : 29.2 tok/s on 4 threads (fp32, stock HF transformers)
```

Sample output (prompt: "Explain what a hash map is in two sentences."):

> A hash map is a data structure that stores key-value pairs using a hash
> function to map keys to indices of an array, enabling fast retrieval,
> insertion, and deletion operations. In this context, a hash map allows for
> efficient storage and access of data by leveraging the underlying hash
> table's ability to quickly retrieve values based on their keys.

Coherent, on-topic, stopped cleanly at EOS.

## Verdict by claim

| Claim | How verified | Result |
|---|---|---|
| Open-weight | `from_pretrained` pulled weights, ungated | ✅ ran it |
| 230M params | counted tensors in loaded model | ✅ 229.7M |
| LFM2 architecture | loaded class `Lfm2ForCausalLM`, `model_type=lfm2` | ✅ ran it |
| Runs on CPU | generated at 29.2 tok/s on 4 vCPUs (fp32) | ✅ ran it |
| Runs on NPU/GPU | no NPU/GPU in container | ⚠️ card-only |
| 19T pre-train tokens | training-time fact, not runtime-observable | ⚠️ card-only |
| Distilled from LFM2.5-350M | training-time fact, not runtime-observable | ⚠️ card-only |
| 32K context | card says 32,768; loaded config `max_position_embeddings=128000` | ⚠️ minor mismatch |

## Caveats

1. **Throughput is the slow path.** 29 tok/s = stock HF fp32 on cloud vCPUs.
   The card's 42 tok/s (RPi5) / 213 tok/s (Galaxy S25) come from quantized
   GGUF/optimized runtimes. "Runs fast on CPU" holds; our number is conservative.
2. **Context nuance.** Post/card say 32K (the trained/extension context); the
   deployed config carries a 128K positional ceiling. Not a falsehood, but the
   exact number to rely on depends on which you mean.

## Bottom line

The post is **accurate**. Every runtime-checkable claim (open weights, 230M,
LFM2, CPU inference) holds when you load and run the model. The training-process
claims (19T tokens, distillation) aren't runtime-observable and rest on Liquid
AI's card — nothing observed contradicts them.
