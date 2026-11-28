# inkling-fact-forwarding: split-context lanes on Inkling 2 Mini

**Status: planned, unrun.** This plan was written in a cloud session with no Mac GPU. A Claude Code session on the M5 Max (128 GB) carries it out, with the lane-batched ErisInky build of Inkling 2 Mini that Oskar runs there (up to 64 concurrent lanes, more than 500 tok/s aggregate).

## Question

Does Inkling 2 Mini answer better when the context is split across lanes that forward the facts they find, than when a single lane reads all of it or several lanes vote, at the same total token budget? If it does, what does the split save in wall-clock time on the M5 Max?

## Source

Recursive Collective Intelligence (arXiv [2611.32606](https://arxiv.org/abs/2611.32606), paper-watch 2026-11-27, rated MEDIUM). It grades a group of agents against three baselines at equal token budget: one agent, a compute-matched majority vote, and compute-matched pass@n. With four Qwen3.8-27B agents on hidden-profile tasks, where each agent holds part of the facts:

- Debate protocols beat one agent but not pass@n, because messages carry conclusions drawn from partial facts and readers adopt them.
- Forwarding raw facts reaches pass@n and beyond: 0.959 accuracy against 0.942 for an equal-budget agent holding every fact (Table 2, checked against the full text), and +0.31 over pass@n.
- Shared context plus forwarding scored 0.972, the best row.
- Fact forwarding reached the top level at 9B, 27B and 35B-A3B.

The paper releases no code, and its benchmarks (SplitBench, its HiddenBench split) may not be public. Inkling 2 Mini, at 26.4B total and 3.4B active, is closest to the 35B-A3B in that list. Nobody has run this on Inkling.

## Why the M5 Max makes this worth testing

- **Prefill dominates agent work.** In inky's measured agent trace (memory 13ac8a1c), prefill outweighs decode 116:1, with a median prompt of 21K tokens and a median completion of 108. Four lanes on ~5K shards prefill a quarter of the tokens each, and full-attention cost drops more than linearly.
- **Lanes are cheap but not free.** Layr measured 1.92x aggregate at batch 4 against solo. inky's expert-grouped MoE gives 1.32x at 8 lanes and 2.83x at 64. So the throughput side favours 8 shards, or several tasks at 4 shards each, over a single task at 4.
- **Inkling is sensitive to KV precision** (erisinky: 8-bit KV flips predictions). Keep KV at the engine's default precision in every arm.

## Step 0: engine interface (do this first, write it down)

The public `Eriskii/erisinky` repo (HEAD 0e8c39d) shows no batched-lane server, so the build on the Mac is a different one: a fork, a later commit, or inky. Before writing the harness, record in `ENGINE.md`:

1. Repo, commit and model file (quant, sha256).
2. How to submit N concurrent requests (HTTP endpoint, CLI, or library), and whether those requests share a batch.
3. Whether it has a prefix cache, and whether a shared instruction prefix is reused across lanes.
4. Per-request prompt tokens, completion tokens and wall-clock time, plus aggregate tok/s.
5. Greedy decoding with a fixed seed, and whether lane outputs are bit-identical to solo runs (inky's batched path is bit-exact; check this one).
6. Power state (`pmset -g`). Low Power Mode halves GPU throughput, so refuse to run in it, as `bonsai-metal-bench` does.

If the engine cannot report per-request token counts, count them with the model's tokenizer in the harness.

## Task sets

Each task set needs a checkable answer. No judge model is used unless it is first calibrated (see Scoring).

**A. Synthetic hidden-profile tasks (primary, exact-match).** Generate the tasks with a seeded script so that the answer requires pooling facts from every shard:

- Pick one of K candidates (K = 4–6) satisfying 4 constraints. Each shard holds one constraint fact, buried in ~4–5K tokens of filler from a fixed public-domain corpus.
- Shards also hold true but irrelevant facts about the candidates, and some carry a misleading comment ("so it's probably B"), mirroring the paper's shared-context failure.
- Design the decoys so that every subset of 3 shards supports a wrong candidate. Verify this by brute force for every task.
- 200 tasks: 50 for prompt development and 150 held out. Freeze the generator and its seed before the first held-out run.

**B. Natural long-document QA (secondary).** Use LooGLE long-dependency QA or the loogle/longreason subsets in `a1zhang/synthetic-rlm-sft-data` (already used by `inkling-rlm-lora/`). Keep documents of 16–32K tokens with short-form answers, and split each into 4 contiguous shards at paragraph boundaries. Score by normalized exact match and token F1. Here an answer may sit in one shard, which tests whether splitting hurts when the facts aren't spread out.

Check whether the paper's HiddenBench (65 official tasks) is public. If it is, run it as set C, so the results can be compared with the paper's numbers.

## Arms

All arms use greedy decoding with the same system prompt, and the same answer format: a final line `ANSWER: <x>`.

| Arm | Lanes | Each lane reads | Rounds |
|---|---|---|---|
| `solo` | 1 | full context | answer |
| `vote` | n_B | full context, sampled at T = 0.7 with distinct seeds | answer; majority, ties broken by the first lane |
| `passn` | n_B | as `vote` | oracle: correct if any lane is correct (upper bound on selection, not deployable) |
| `fwd` | 4 | one shard | R1: extract quoted facts with shard id, no answer. R2: every lane gets all forwarded facts. R3: every lane answers; majority |
| `fwd-hub` | 4 | one shard | R1 as `fwd`; R2: one hub lane gets all facts and answers |
| `ctx-fwd` | 4 | one shard | R1 as `fwd`, but lanes may add a comment; R2 forwards facts and comments (the paper's shared-context condition) |
| `fwd8` | 8 | one of 8 shards | as `fwd` (throughput point where expert grouping starts to pay) |

**Equal budget.** Measure each `fwd*` arm's mean total tokens per task (prompt plus completion, all rounds, all lanes). Set n_B for `vote` and `passn` to round(that total / `solo` tokens), with a minimum of 1. Report budgets in the results table so the comparison is visible. Repeat it with prompt tokens discounted by the engine's prefix-cache hit rate, if it has one.

## Scoring

- Accuracy per arm per task set, with a paired bootstrap 95% CI on every difference against `solo` and against `vote` (10,000 resamples over tasks).
- The paper's ladder per arm: level 1 beats `solo`, level 2 beats `vote`, level 3 beats `passn`. "Beats" means the CI of the difference excludes 0.
- Wall-clock per task, with tasks run concurrently up to 64 lanes total. Report tasks per hour for each arm, the metric that matters on this machine.
- Fact-extraction recall in R1 for set A: the fraction of the 4 constraint facts that appear in the forwarded facts (string match against the generator's ground truth). This separates extraction failures from reasoning failures.
- No judge model. If one is added later for set B, follow `judge-eval-discipline`: binary verdicts, TPR and TNR above 90% against Oskar's labels before trusting it, then Rogan-Gladen correction.

## Predictions (registered before any run)

- **P1** (0.65): on set A, `fwd` beats `vote` at equal budget (level 2).
- **P2** (0.35): on set A, `fwd` beats `passn` (level 3). It is lower than P1 because the paper's level-3 results used 9B–35B models, and Inkling's active path is 3.4B.
- **P3** (0.7): `ctx-fwd` scores below `fwd` on set A, because the misleading comments get adopted.
- **P4** (0.6): on set B, `fwd` is within 5 points of `solo` (with a CI that includes 0) and does not beat `vote`, because natural QA answers are rarely spread across shards.
- **P5** (0.75): at equal tokens, `fwd` finishes a 20K-context task in less wall-clock time than `solo`, because of the shorter prefills. `fwd8` delivers more tasks per hour than `fwd`.
- **P6** (0.5): R1 fact-extraction recall on set A is below 90%. If so, extraction rather than reasoning sets the ceiling.

Report each prediction as held or failed, with the number.

## Deliverables

- `ENGINE.md` (step 0), `gen_tasks.py` (set A, seeded, with the decoy brute-force check), `harness.py` (arms, lane submission, token accounting, resumable via one JSONL row per task × arm), `score.py`, and `results/summary.json`.
- `RESULTS.md` with the ladder table, budgets, tasks per hour, the prediction scorecard, and any `ERRORS.md` entries.
- Update this experiment's row in the root README index.
- Push to `main` as work lands. Work done on the Mac goes to GitHub, never local-only.

## Run (once the harness exists)

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments
cd experiments && git sparse-checkout set inkling-fact-forwarding && cd inkling-fact-forwarding
# start the ErisInky lane server per ENGINE.md, plugged in, not in Low Power Mode
python3 gen_tasks.py --seed 20261127
python3 harness.py --set A --arms solo,vote,passn,fwd,fwd-hub,ctx-fwd,fwd8 --lanes 64
python3 score.py
```

Origin: Oskar, 2026-11-27. After the paper-watch review of 2611.32606 he asked whether a task split across ~4 ErisInky lanes could use the paper's findings, then asked to "write up the experiment on a format reachable by my M5 mac".
