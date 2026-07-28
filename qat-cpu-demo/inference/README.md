# inference/ — running the real Gemma 4 E2B QAT checkpoint, GPU-less

Follow-up to the parent experiment's part 2: don't just *say* the int4 QAT
checkpoint runs on this box — pull it and clock it.

**Verdict: it runs, ~20 tok/s on 4 CPU cores.** Faster than reading speed, no
GPU, no quota, ~$0.

## What ran

| | |
|---|---|
| Model | [`google/gemma-4-E2B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) — the post's headline int4 QAT checkpoint |
| File | `gemma-4-E2B_q4_0-it.gguf`, **3.35 GB** on disk / 3.10 GiB in GGUF terms |
| Params | **4.63 B total**, 2 B "effective" (E2B = elastic/MatFormer) |
| Runner | `llama.cpp` @ `308f61c`, built from source CPU-only (cmake + gcc 13) |
| Hardware | 4 vCPU, 15 GB RAM, **no GPU** (`torch.cuda.is_available() == False`) |
| Quant | Q4_0, weight-only int4 — Google's QAT checkpoint, not a post-hoc requant |

The "~1 GB" headline from the blog is the *mobile / PLE-stripped* variant; the
standard Q4_0 GGUF is 3.35 GB. Both fit this box with room to spare.

## Throughput (`llama-bench`, 4 threads)

| test | tokens/s |
|------|---------:|
| prompt processing (pp128) | **182.1 ± 0.9** |
| token generation (tg64)   | **20.5 ± 0.3** |

A one-shot `llama-completion` run measured 135 t/s prompt eval and 22.7 t/s
generation — same ballpark. Generation at ~20 tok/s is comfortably above human
reading speed; first-token latency on a short prompt is sub-second.

The bottleneck is exactly what was predicted in the parent writeup: **tok/s on
CPU**, not RAM (peak ~3.5 GB resident) and not disk. Gating was a non-issue —
the repo is `gated: false` and `HF_TOKEN` is present anyway.

## Sample output

Prompt: *"In 3 sentences, explain why quantization-aware training beats
post-training quantization at very low bit-widths."* (temp 0.7, seed 42)

> Quantization-aware training (QAT) is superior to post-training quantization
> (PTQ) at very low bit-widths because QAT allows the model to learn how to
> represent its weights in a way that is robust to the quantization noise,
> whereas PTQ applies quantization based on a single forward pass, which often
> results in significant accuracy degradation when the quantization noise is too
> large to be tolerated. QAT, by contrast, uses the quantization process as a
> differentiable part of the training loop, allowing the model to adapt its
> representations to the quantization constraints during the learning process,
> leading to much better quantization accuracy.

Full text in [`sample_out.txt`](sample_out.txt). Note the closure: the model's
own explanation is precisely the mechanism the parent experiment *measured* —
at int2, PTQ degraded perplexity 4× while QAT held the baseline.

## Reproduce

```bash
cd experiments/qat-cpu-demo/inference

# 1. model (3.35 GB, ungated)
curl -sSL -o gemma-4-E2B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf

# 2. runner — build from master (Gemma 4 is new; prebuilt wheels may lag the arch)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-bench llama-completion -j4

# 3. benchmark + generate
llama.cpp/build/bin/llama-bench -m gemma-4-E2B_q4_0-it.gguf -t 4 -p 128 -n 64 -r 2
llama.cpp/build/bin/llama-completion -m gemma-4-E2B_q4_0-it.gguf -t 4 -n 200 -no-cnv -e \
  -p "<start_of_turn>user\nYour question here<end_of_turn>\n<start_of_turn>model\n"
```

The `.gguf` and `llama.cpp/` clone are gitignored (heavy, regenerable).

## Part 3 — the `local-llm` container layer

Having shown the *technique* (parent) and the *checkpoint* (above) run GPU-less,
the question became: when is that actually worth doing, and can it be made
cheap? Answer: only when you need **white-box access** the harness Agent tool
can't give — token logprobs, embeddings, synchronous in-script inference — and
it's made cheap by baking the runtime into a container layer.

Shipped: [`layers/Containerfile.local-llm`](../../../layers/Containerfile.local-llm)
(opt-in, ~650 MB cached) + helpers [`scripts/llm_local.py`](../../../scripts/llm_local.py)
(`embed`/`score`/`generate`) and [`scripts/fetch-model.sh`](../../../scripts/fetch-model.sh).

### What was validated (`validate_runtime.py`, this dir)

`llama-cpp-python==0.3.26` (prebuilt CPU wheel) on 4 cores:

| capability | model | result |
|---|---|---|
| embeddings | EmbeddingGemma 300M Q8 | 768-dim, cos(paraphrase)=0.73 vs cos(unrelated)=0.20, ~25 ms/doc |
| logprob zero-shot classify | Gemma 3 270M Q8 | correct polarity, ~150 ms/choice (see fix below) |
| generation | Gemma 3 270M Q8 | coherent |
| **Gemma 4 E2B load** | gemma-4-E2B Q4_0 | **loads & runs** ("capital of France"→" Paris.") — the pip wheel suffices, **no from-source CLI needed in the layer** |

### Finding: don't score logprobs via the high-level `echo` path

The obvious way to get sequence logprobs — `create_completion(..., echo=True,
logprobs=1)` — is **pathologically slow** on Gemma: llama-cpp-python
post-processes the full ~262k-token vocab in pure Python at every position
(>2 min/call, killed). `llm_local.score()` instead reads raw logits via the
low-level `eval()` + `llm.scores` and does the softmax over only the target
token in numpy: **>2 min → ~150 ms**. That's why `score()` exists as a helper
rather than a one-liner.

### Usage

```python
from scripts.llm_local import embed, score, generate
embed(["a", "b"])                                  # local vector index, no API
score("Sentiment:", [" positive", " negative"])    # zero-shot classify
generate("Once upon a time", n=40)
```

The layer is **not** in the default composition — add `"local-llm"` to
`.claude/container-layers.json` only for sessions that need it. First opt-in
boot builds + caches it (~650 MB); thereafter it's a single fetch.

