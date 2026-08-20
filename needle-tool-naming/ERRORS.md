# Errors — needle-tool-naming

What was wrong, how it surfaced, and which way it pushed the conclusion.

## 1. The mechanism this probe was built on was the wrong mechanism

**What.** The probe was proposed off a reading of the Cactus attention-only
paper: since a SAN's content-write path is the attention output projection, and
`needle-bsky`'s LoRA left `profile` and `identity` at exactly their base
accuracy, the fine-tune must have missed the write path.

**Cause.** Reasoning from the paper's architecture to the shipped trainer
without reading the trainer. `needle/model/finetune.py` sets
`LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")`, so
`out_proj` — the write path — *is* adapted. Resolved against the real
checkpoint, LoRA reaches **28.31M of 45.21M** parameters. What it cannot reach
is the two Engram tables (8.39M), their key/value projections, the embedding,
the contrastive head and the confidence head.

**How caught.** Loading `checkpoints/needle2.pkl` and calling
`lora_target_paths()` on it, before writing any code that depended on the claim.

**Direction.** Would have produced a confident wrong explanation for the
fine-tune negative and, worse, a second fine-tune run — roughly two hours of
CPU — chasing a target that was never frozen.

## 2. The pre-registered hypothesis was wrong about the size of the effect

**What.** P5 predicted `names-only` − `desc-only` ≥ 0.15. Measured **0.037**,
p=0.82. Four other numeric predictions missed, three of them narrowly.

**Cause.** The hypothesis was formed from seven queries in two categories, all
of which pointed the same way, and generalised to a main effect over 54 queries
and 15 categories. The category-level signal was real — `profile` moves 0.250 →
0.750 when the rotation puts "follower" on the profile tool — it is simply
smaller than the aggregate can resolve.

**How caught.** By running the pre-registered design rather than a
demonstration. `PREREG.md` was committed before the first variant ran
(`08e1fa3`), so the miss is on the record rather than reframed afterwards.

**Direction.** Reported as the result. Had the predictions been written after
the runs, the `adversarial` `profile` flip would have made a tidy and misleading
headline.

## 3. `refusal_acc` is `None` when an evalset has no off-topic items

**What.** The first smoke run crashed formatting its own summary line:
`TypeError: unsupported format string passed to NoneType.__format__`.

**Cause.** `needle-bsky`'s `summarize()` returns `None` for a rate over an empty
subset, and the six-query smoke set had no off-topic queries. Not a scoring bug
— the 62-query runs were never affected.

**How caught.** The smoke run, which existed for this.

**Direction.** None. Cost one minute.

## 4. Two rounding disagreements between the prose and the artifacts

**What.** `recheck.py` failed on first run: the writeup said the two channels
sum to 0.703 (0.370 + 0.333) where the unrounded artifacts give 0.704, and
quoted `needle-bsky`'s headline +26 points where the artifacts give 0.259.

**How caught.** `recheck.py`, on its first execution, which is the entire reason
that fixture exists.

**Direction.** Cosmetic. Recorded because the base rate of this class of drift
is the useful number, not any individual instance.
