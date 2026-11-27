# bonsai-metal-bench

A benchmark for Ternary Bonsai 2.5 27B on an Apple Silicon Mac. It measures how far
llama.cpp and MLX run from the Mac's memory-bandwidth ceiling, so we can tell which kernel
work would pay before writing any. Bonsai 2.5 is PrismML's ternary compression of
Qwen3.8-27B: 1.76 bits per weight, 5.93 GB. On Macs, PrismML published only an M5 Pro run
(30.2 tok/s decode, 416 tok/s prefill on PQ2_0) and an M4 Pro MTP run (1.10–1.27x).

**Status: written, not yet run.** It needs a Mac GPU, and the container it was written in
has none.

## Run

On the Mac, plugged in, Energy Mode set to Automatic or High Power:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments
cd experiments && git sparse-checkout set bonsai-metal-bench && cd bonsai-metal-bench
brew install cmake          # once; Xcode command line tools are also needed
python3 bench.py            # all phases, about 40–60 min, about 30 GB of downloads into work/
```

`--only llama|mtp|mlx|bw` runs a subset. The script refuses to run on battery or in Low Power
Mode (Low Power Mode halves GPU throughput) unless you pass `--force`.

## What it measures

| Phase | What | Question it answers |
|---|---|---|
| `bw` | MLX fp16 matvec and sum over 4 GiB | the bandwidth ceiling, the roofline anchor |
| `llama` | PrismML llama.cpp fork, built from source; `llama-bench` on PQ2_0 and PTQ1_0: prefill at 512/2048/8192 tokens; 128-token decode at depth 0/8192/32768; prefill ubatch 256–2048 | how far decode is from bandwidth ÷ bytes per token; how fast decode falls as KV and GDN state grow; whether larger prefill tiles help on the M5 matrix units |
| `mtp` | `llama-server` on PQ2_0-mtp with the released Metal preset, MTP off vs on, six real prompts plus one 8K prefill | whether the MTP verification path is the cheap win |
| `mlx` | the prism-ml MLX pack through its bundled loader: prefill and decode at 512–32768 tokens, MTP head off vs on | how MLX's generic affine 2-bit kernels compare with the fork's ternary kernels |

Results land in `results/<timestamp>-<chip>/`: `summary.md` (tables, including decode as
a percentage of its ceiling), `summary.json`, `env.json` (chip, power state, wired limit)
and `raw/`. Commit that directory, or paste `summary.md` back.

The decode ceiling is bandwidth ÷ (GGUF file size + GDN state read and written + 64 KiB ×
depth of fp16 KV). The file size overcounts bytes, because only one row of the embedding
table is read per token.
