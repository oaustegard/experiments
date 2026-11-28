# tooling: lane-batched Inkling 2 Mini for Claude Code

An MCP server (`inkling_batch_mcp.py`) and the engine patch it needs (`erisinky-batch-json.patch`). Written during
Step 0 of `inkling-fact-forwarding`, where the measurements that shaped it are in [`ENGINE.md`](../ENGINE.md). The
experiment's harness will call the same `inkling batch-json` command.

## What it does

Runs many Inkling 2 Mini lanes on one Apple GPU through upstream erisinky's batch engine.

| Tool | What it does |
|---|---|
| `batch_map` | N independent prompts, one lane each, results in order. Thinking off by default. |
| `batch_samples` | N samples of one prompt from one shared prefill, with an optional vote tally. |
| `batch_split_ask` | **Experimental.** Split a long text into shards, extract facts per shard in lanes, answer from the pooled facts with a vote. Not shown to beat one lane reading the whole text. |
| `batch_status` | Engine, model, patch support, power state, other GPU-heavy processes. |

Each call starts the engine, runs one batch and exits, so no GPU memory is held between calls. Text in, text out: the
model gets no file, shell or network access. Output is unverified.

Measured on an M5 Max through the tool: 38, 100, 182 and 279 tok/s aggregate decode at 1, 4, 8 and 16 lanes (short
outputs). The single-stream engine decodes at 90-105 tok/s, so lanes win from about 4 lanes up. Prefill does not
parallelise (about 3.5-4k tok/s however the text is split), and a lane's output is not bit-identical to a solo run.

## Build the engine

Pinned to upstream `Eriskii/erisinky` commit `4be5bed685e937e4cb6f0143bb17afa5cb51a763`. Requires Apple silicon with
Metal 4 and a Rust toolchain.

```bash
mkdir erisinky-batch
curl -L https://codeload.github.com/Eriskii/erisinky/tar.gz/4be5bed685e937e4cb6f0143bb17afa5cb51a763 \
  | tar -xz --strip-components=1 -C erisinky-batch
cd erisinky-batch
patch -p1 < ../erisinky-batch-json.patch      # path to the patch in this directory
cargo build --release                          # about 30 s
```

The model is LM Studio's `Inkling-2-Mini-Q4_K_M.gguf` (erisinky cannot load the Unsloth quant). The patch adds
`inkling batch-json --prompts-esc FILE --out FILE`: one prompt per line with `\n \r \t \\` escaped, `--think-off`,
`--shared` for N samples of one prompt, and per-prompt JSONL with prompt and generated token counts, a finish reason
(`stop | length | context`), reasoning and content, then a `{"summary":...}` line. A prompt that does not fit `-c`
becomes an error row, not a crash. User text is tokenised without parsing control tokens.

The patch is a diff against erisinky, which is GPL-3.0-or-later; treat the patch as covered by that license.

## Register and test

```bash
claude mcp add inkling-batch \
  -e INKLING_WORKSPACE=/path/to/workspace \
  -e INKLING_BATCH_BIN=/path/to/erisinky-batch/target/release/inkling \
  -- python3 /path/to/tooling/inkling_batch_mcp.py

INKLING_BATCH_BIN=/path/to/erisinky-batch/target/release/inkling python3 test_batch.py
```

Environment (all optional): `INKLING_BATCH_MODEL`, `INKLING_BATCH_CTX_CAP` (default 65536), `INKLING_BATCH_MAX_LANES`
(64), `INKLING_BATCH_MAX_MEM_GB` (48). Metal does not work inside the Claude Code Bash sandbox, so run the test outside
it. The test waits up to 4 minutes for a quiet GPU.

## Limits worth knowing

- Greedy decoding is refused (it loops on this model); top-k is capped at 64 by the engine's GPU sampler.
- Low Power Mode refuses a run. Another GPU-heavy process (a `llama-bench` cut decode from 100 to 5 tok/s in testing)
  is reported as a warning in the result, not a refusal.
- Only samples of one prompt share a prefill. Different prompts over the same context each prefill it again.
- Per-prompt wall-clock is not reported, only batch totals.
