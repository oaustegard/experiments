# ettin-ane: the Ettin-17m decider on the Apple Neural Engine

Run 2026-11-16 on Oskar's M1 MacBook Air (8 GB, macOS 27). Question: can the packed Ettin-17m decision model
(`browser-decider/web/models/decider`, the fully fine-tuned build with its own head) run on the Mac's Neural
Engine through Core ML, with the same answers, and how fast is it there compared with the CPU and GPU?

The Neural Engine is the neural-network accelerator in Apple silicon (16 cores and about 11 TOPS on the M1). It
is reachable only through Core ML, which decides per operation whether to use it.

## Answer

Yes. One 512-token decision takes 2.9 ms with the Neural Engine, against 6.3 ms on the CPU and 13.6 ms on the
GPU, all through Core ML. The answers match the fp32 PyTorch model: accuracy 0.747 on the 490 test questions that
fit in 512 tokens, the same as the reference on those questions, with the top answer unchanged on 488 of 490.
Core ML assigns 715 of the model's 716 operations to the Neural Engine; only the token-embedding lookup stays on
the CPU.

| Core ML compute units | 512 tokens: ms median (p90) | accuracy | top answer same as fp32 | 256 tokens: ms median (p90) |
|---|---|---|---|---|
| CPU only | 6.26 (6.86) | 0.743 | 488 / 490 | 2.64 (2.92) |
| CPU + GPU | 13.56 (15.13) | 0.749 | 489 / 490 | 7.72 (10.40) |
| CPU + Neural Engine | 3.07 (3.70) | 0.747 | 488 / 490 | 1.30 (1.61) |
| All | 2.93 (3.12) | 0.747 | 488 / 490 | 1.32 (1.52) |

One question per call, 200 timed calls after 5 warm-up calls, timed around `MLModel.predict` in Python. The
Hugging Face `ModernBertModel` in PyTorch on the same CPU, unpadded, takes 10.8 ms per question and scores 0.750
on all 500. Source: `results/summary.json`.

Other measurements:

- **Fit.** The 500 questions run 126 to 596 tokens (median 307, 95th percentile 390). 490 fit in 512 tokens and
  154 in 256. The other 10 are not scored here.
- **Size.** The Core ML package is 34.6 MB in fp16. The 4-bit token table and int8 layers of the 11.5 MB pack are
  decoded to floats before conversion and not re-compressed.
- **Load.** The first load with the Neural Engine takes 3.3 to 4.2 s, against 0.5 to 1.2 s for the CPU alone.
- **Conversion.** 9 s per length.

## What had to change

`run/ane_model.py` is the model rewritten in the layout Apple's `ml-ane-transformers` describes, loaded from
the same weights:

- activations are `(1, C, 1, S)` and every linear layer is a 1×1 convolution;
- attention is computed one head at a time with two einsums;
- the sequence length is fixed, with the rotary position tables and the local-attention mask (a token attends
  within 64 positions on the non-global layers) stored as constants for that length;
- padding is an additive −1e4 on the key axis, so ModernBERT's unpadding path is not needed;
- the decision head runs at every position and the caller reads the logits at the `[MASK]` positions.

In PyTorch at fp32 the rewrite matches the Hugging Face model to 1.2e-06 in probability on every question.

## Not measured

- LoRA spikes. Only the fine-tuned base is ported. Core ML's multifunction models are the likely route.
- Power draw.
- Timing from Swift. The Python wrapper adds overhead to every number in the table.
- Ettin-32m and Ettin-68m.
- One run on one machine; the intervals between compute units are not bootstrapped.

## Reproduce

```bash
python3 run/port.py     # parity, conversion, timing, compute plan -> results/summary.json; models land in out/
```

Needs macOS on Apple silicon with torch 2.5.1, coremltools 8.3.0, transformers 4.51.3, tokenizers and
numpy < 2.2. transformers 5.x refuses torch below 2.6, and coremltools 8.3 is tested up to torch 2.5.

## Follow-on

The same layout was applied to `mdbr-leaf-ir` (a 6-layer BERT embedder) for a local semantic search over the
write-ups in `~/Projects`: 2.3 ms per 256-token passage on the Neural Engine against 6.4 ms on the CPU, 19,469
passages indexed in 90 s. That build is not in this repo.
