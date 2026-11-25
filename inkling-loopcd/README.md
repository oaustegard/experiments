# inkling-loopcd — LoopCD guidance for Inkling 2 Mini on a Mac

LoopCD ([arXiv 2610.02185](https://arxiv.org/abs/2610.02185), Apple, October 2026) improves looped
transformers at decoding time, with no training. A looped model runs the same block of layers several
times per token. LoopCD compares the state after the last pass with the state after the first and
pushes further in that direction:

    h' = h_R + ω (h_R − h_1)

The paper tests Ouro, Huginn, Parcae and a looped Qwen3. Inkling 2 Mini (`model_type: inkling2_loop`)
is also looped: layers 0–5 run once, 6–17 run three times, 18–23 run once. This directory applies
LoopCD to it in mlx-lm and measures whether it helps.

**Status: tests pass, sweep not yet run.** The 7 tests in `test_loopcd.py` pass on MLX 0.32.4 (CPU) with mlx-lm PR #2086. The sweep needs the real model and a 32 GB+ Mac.

## What's here

| file | what it does |
|---|---|
| `loopcd.py` | Patches mlx-lm's Inkling 2 forward. `enable(model, omega, passes=None, guide_blocks=True, guide_prompt=True)`, `disable(model)`. Also a one-prompt CLI. |
| `test_loopcd.py` | Checks on a tiny random Inkling 2 loop model: ω=0 reproduces the stock forward; ω>0 changes it; prefill plus one decode step matches one full forward at 3 and 2 passes, so the caches stay consistent; `passes` resizes the cache and `disable` restores it. |
| `sweep.py` | Runs every setting on the same problems with greedy decoding (GSM8K test by default, or your own JSONL). Reports accuracy, paired wins and losses against ω=0 with an exact McNemar p, generated tokens and tokens/s. Resumable. |

## How it maps onto Inkling

LoopCD-Hidden is the variant implemented here. It needs one run of the coda, which costs a few
vector operations per token. LoopCD-Logits would run the coda twice, and Inkling's coda has attention
layers, so it would need a second KV cache for six layers. It is not implemented.

Inkling differs from the paper's models in two ways:

- **h₁ is the model with no loop.** Each pass restarts from the prelude output and reads the previous
  pass only through a gated message bank. Pass 1 has no messages, so h₁ is what the model computes with
  the loop removed.
- **The coda reads more than the hidden state.** It also reads a stack of block states, which are
  attention residuals over earlier block boundaries. After every pass the stack has the same shape,
  and passes differ only in the core's own entry. By default both the hidden state and the stack are
  extrapolated. `--hidden-only` guides the hidden state alone, as an ablation.

The coda's KV caches are written from the guided states, so the prompt and the generated tokens see
the same coda inputs.

## Run it

You need a Mac with at least 32 GB of memory. The 4-bit MLX weights are 17.3 GB. The model support is
still in mlx-lm pull request #2086:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/2086/head" pytest
cd inkling-loopcd

python -m pytest test_loopcd.py -q          # seconds, no download

python loopcd.py --omega 0.25 --prompt "How many primes are below 100?"   # downloads 17 GB once

python sweep.py --n 100                     # ω ∈ {0, 0.1, 0.25, 0.5}, 3 passes
python sweep.py --n 100 --settings 0,p2:0,p2:0.25 --out results/passes.jsonl
```

The sweep writes `results/sweep.jsonl` (one row per problem and setting) and `results/sweep.summary.json`.
It uses Inkling's `reasoning_effort=low` to keep answers short. `--effort medium --max-tokens 16384`
gives the model more room. GSM8K may already be near the ceiling for this model. If ω=0 scores above
about 90%, a harder set with numeric answers will show more, such as AIME (`--data aime.jsonl` with
`question` and `answer` fields).

## What to expect

The paper's largest gains come from models with many loop passes. Huginn has 32, and its HumanEval
pass@1 went from 22.6% to 31.7%. Inkling has three passes, so the first and last states are closer
together and the guidance has less to work with. The paper's own screening put ω near 0.2–0.3 for
generation and near 0.5 for multiple choice.

The second test is `p2:ω`: two passes with guidance against three without. If two guided passes match
three plain ones, decoding gets faster, because each pass reads its own set of experts from memory.

Unverified: a benchmark table circulating with this paper credits LoopCD with gains on Inkling 2 Mini
(for example ARC-AGI v2 public eval 47.8% → 55.0%). The text of the paper we read does not mention
Inkling. This sweep is the check.
