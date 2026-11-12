# meta2-paperwatch

[paper-watch](../paper-watch/README.md) reviews about 30 trending arXiv papers a day for one reader, Oskar. He runs
an LLM agent with external memory, vector-quantization and retrieval tools, WebGPU kernels for small in-browser
models, and Claude Code tooling. Haiku subagents read each paper in full and rate its practical use for him as
NONE, LOW, MEDIUM or HIGH. Opus then re-checks every MEDIUM or HIGH rating (step 4 of
[`PROCEDURE.md`](../paper-watch/PROCEDURE.md)), because Haiku inflates relevance and trusts model cards. The
final ratings are stored in `seen.json`, called *the ledger* below.

This experiment tests the method of [Meta 2](https://arxiv.org/abs/2611.05139) (arXiv 2611.05139) on the Haiku
reviewer. Offline, Meta 2 distils a short frozen *Meta-Skill* by contrasting runs with and without access to the checked rating. Online, a separate *evolver* call looks at the agent's output at a checkpoint. It either keeps it
(KEEP), or sends the agent back with a notice and possibly a *standing instruction*, a rule added to the agent's
prompt (EVOLVE). Those edits are discarded when the task ends.

Four conditions, each run three times over 33 test papers:
- **A**: the current `PROCEDURE.md` reviewer prompt.
- **B**: A plus the Meta-Skill.
- **C**: B plus an evolver whose edits reset per paper.
- **D**: C with edits that accumulate across papers.

*Gold* is a new Opus rating of each paper, made without seeing the ledger. *MEDIUM+* means MEDIUM or HIGH. Issue:
[#112](https://github.com/oaustegard/experiments/issues/112).

## Answer

Does a Meta-2-style Meta-Skill plus evolver make Haiku's ratings better? No, and nothing should be adopted.

The Meta-Skill (B) raised exact agreement with gold from 0.44 to 0.69. Most of that gain comes from rating more
papers LOW: answering LOW for every paper scores 0.67. Weighted kappa moved only from 0.23 to 0.26, inside the
run spread. MEDIUM+ false positives fell from 43% to 6%. MEDIUM+ recall fell from 100% to 33%, measured on just
4 gold-MEDIUM test papers. Opus already re-checks every MEDIUM+ rating, so a false positive gets corrected, while
a real MEDIUM that Haiku calls LOW is never seen again.

The evolver (C) changed nothing on identical drafts (C − B = +0.00) and used 2.8x the tokens. Accumulating edits
(D) beat per-paper reset by one paper, which is inside the noise.

The production ledger beats every Haiku condition on kappa (0.54) and recall, though not on exact agreement.
The Meta-Skill was distilled by Haiku, not Opus, and gold is stricter than the ledger: it rated 6 of the ledger's 8
MEDIUMs LOW. Spot-checking those six, listed under the recommendation in the Log, would settle which rating
standard is right.

## Findings

1. **The Meta-Skill turns Haiku into a LOW-predictor.** Exact agreement: A 0.44, B 0.69, always-LOW 0.67. <!-- @evidence[[results/metrics.json#conditions.A.exact.mean]] --> <!-- @evidence[[results/metrics.json#conditions.B.exact.mean]] --> <!-- @evidence[[results/metrics.json#always_LOW_on_test.exact]] -->
   Weighted kappa: A 0.23, B 0.26. <!-- @evidence[[results/metrics.json#conditions.A.kappa_w.mean]] --> <!-- @evidence[[results/metrics.json#conditions.B.kappa_w.mean]] -->
   Signed error fell from +0.58 to +0.15 rating steps, and the five entries in [`meta_skill.md`](meta_skill.md) all push ratings down. (`results/metrics.json`)
2. **The false positives it removes are the ones the pipeline already catches.** MEDIUM+ false-positive rate: A 0.43, B 0.06. <!-- @evidence[[results/metrics.json#conditions.A.fp_medium_plus.mean]] --> <!-- @evidence[[results/metrics.json#conditions.B.fp_medium_plus.mean]] -->
   MEDIUM+ recall: A 1.00, B 0.33. <!-- @evidence[[results/metrics.json#conditions.A.recall_medium_plus.mean]] --> <!-- @evidence[[results/metrics.json#conditions.B.recall_medium_plus.mean]] -->
   The test set has only 4 gold-MEDIUM+ papers, all MEDIUM, so one paper moves recall by 25 points. The one gold-HIGH paper is in the offline set. (`results/runs/B_r*.json`)
3. **The evolver does not move ratings.** On the same drafts, C − B exact agreement is 0.00. <!-- @evidence[[results/metrics.json#paired_C_vs_B_exact.diff]] -->
   It chose EVOLVE on 0.68 of drafts. <!-- @evidence[[results/metrics.json#conditions.C.evolve_rate.mean]] -->
   The rating changed on 1.3 papers per run, 0.7 toward gold and 0.7 away; C's weighted kappa (0.19) sits below B's (0.26), within B's run range. Its notices and standing instructions police how faithfully the review quotes numbers (which baseline a range belongs to, which table a size comes from), not the rating. (`results/runs/C_r*.json`, `evolver` field)
4. **Per-paper reset vs accumulation: no detectable difference.** D − C exact agreement is +0.03, one paper, with interval −0.02 to +0.08. <!-- @evidence[[results/metrics.json#paired_D_vs_C_exact.diff]] -->
   D ended each run carrying 22–27 standing instructions, and its MEDIUM+ recall fell to 0.08. <!-- @evidence[[results/metrics.json#conditions.D.recall_medium_plus.mean]] --> (`results/runs/D_r*.json`, `standing_after`)
5. **Cost.** Tokens per paper: A 40.0K, B 41.1K, C 114.7K, D 122.9K. At list price that is about $0.002, $0.002, $0.013 and $0.017 per paper. C and D read the paper three times when they send a draft back: once for the draft, once for the evolver and once for the revision. (`results/metrics.json`, `tokens_per_paper`)
6. **The production ledger beats every Haiku-only condition.** On the 33 test papers it scores exact 0.70. <!-- @evidence[[results/metrics.json#ledger_on_test.exact]] -->
   Its weighted kappa is 0.54 and its MEDIUM+ recall 0.75. <!-- @evidence[[results/metrics.json#ledger_on_test.kappa_w]] --> (`results/metrics.json`, `ledger_on_test`)
7. **Gold is consistent but stricter than the ledger.** Two blind Opus passes agreed on 51 of 55 papers (exact 0.93, kappa 0.88). <!-- @evidence[[results/metrics.json#gold_self_agreement.exact]] -->
   Gold matches the ledger exactly on 0.64 of papers. <!-- @evidence[[results/metrics.json#gold_vs_ledger.all.exact]] -->
   Of the ledger's 10 MEDIUM+ papers, gold agrees exactly on 1. It rated 6 of the 8 ledger MEDIUMs LOW in both passes, raised 1 MEDIUM to HIGH, and lowered both HIGHs to MEDIUM. (`gold.json`)
8. **Verifier access helps Haiku less than expected.** On the offline set, plain Haiku hit gold 0.36 of the time. <!-- @evidence[[results/offline.json#plain_exact]] -->
   Told the gold rating and its key numbers, it hit 0.59, so it still argued its way to its own rating 41% of the time. <!-- @evidence[[results/offline.json#informed_exact]] --> (`results/offline.json`)

## Method

**Data.** The fixture is the 55 papers in `paper-watch/seen.json` with `read == "full"` (the two read only as
abstracts are excluded). The full text was fetched from `arxiv.org/html/<id>v1`, with a PDF fallback for 3
papers whose HTML was missing or cut short. Reference lists are dropped, link targets are kept so code and
weights links can be checked, and the text is capped at 160K characters. That cap truncated 5 papers.
`fetch.py` does all of this. The PDF-sourced papers carry no link URLs.

**Gold.** The issue asked for at least 24 Opus labels: every MEDIUM/HIGH plus a random sample of LOW/NONE. With
the date split, that would have left about 14 test papers, so all 55 papers got gold labels instead. Each was
rated blind to the ledger by `claude -p --model opus`, with the paper-watch profile and the step-4 checks:
verify the numbers the rating rests on, check that code is actually linked, correct for topic-overlap
inflation. There were two independent passes, and a third broke the 4 disagreements (`gold.py`, `gold.json`).
Gold distribution: 1 HIGH, 4 MEDIUM, 35 LOW, 15 NONE.

**Split.** Papers are sorted by `first_seen`, then by feed rank within a day. The earliest 40% (22 papers, 2026-11-01 to
-02) form the offline set, and the remaining 33 (2026-11-02 to -12) form the test set. No test paper or test
label reaches the Meta-Skill.

**Offline distillation** (`offline.py`). Haiku reviewed each offline paper in two settings. A *plain* setting used the reviewer prompt, twice. An *informed* setting used the same prompt plus the gold rating and key numbers
(Meta 2's verifier access), once. One call then distilled the contrasts into the Meta-Skill. Its prompt condenses Meta 2's three offline stages (extract lessons per paper, merge them across papers, update the skill): act/avoid/hold entries, each needing support from at least 2 papers, conditions
that can be observed without the answer, and at most 400 words. The result is
[`meta_skill.md`](meta_skill.md), 368 words, frozen before any test call, and used unedited. The distill call ran on Haiku, matching Meta 2's one-model design (Opus was out of weekly quota).

**Test conditions** (`run_test.py`). The evolver runs once, after the reviewer has finished its draft. All four conditions use Haiku 5.5 through `claude -p` with tools off. Each
ran 3 times over the 33 test papers.
- **A**: the baseline prompt. Its profile and output format are read live from `paper-watch/PROCEDURE.md`, with
  the full text supplied in the prompt.
- **B**: A plus the Meta-Skill in the system prompt.
- **C**: B plus an evolver at the terminal checkpoint. The evolver is Haiku with the paper text, the draft and
  the Meta-Skill. It returns KEEP, or EVOLVE with one send-back notice and optionally one standing instruction,
  and the reviewer then revises once. Edits are discarded after each paper. C reuses B's draft for the same paper
  and run, so C − B isolates the evolver.
- **D**: like C, but standing instructions accumulate through each run in `first_seen` order, and later drafts
  are written under them.

As in Meta 2, the evolver never writes the review itself; it only sends notices and instructions.

**Metrics** (`analyze.py`). The metrics are:
- exact agreement with gold
- linear-weighted kappa
- MEDIUM+ false-positive rate (gold NONE/LOW rated MEDIUM+)
- MEDIUM+ recall
- mean signed error in rating steps
- tokens per paper (input, cache and output, summed over every call for the paper)

Each is reported as the mean and range over the 3 runs. Condition differences use a paired bootstrap over papers
of per-paper exact agreement averaged across runs (10,000 resamples). One paper is worth 3.0 points of exact
agreement. A difference of one or two papers is inside the noise.

**Differences from production.** Production reviewers fetch the paper themselves and batch 5 papers per agent.
Here the text is supplied in the prompt and each paper gets its own call. The Opus step-4 check is not part of
any condition. It appears only in the ledger row.

**Prior art.** Meta 2's own results come from two agentic coding benchmarks (Terminal-Bench 2.1, DeepSWE 1.1); its limitations section says it is untested on tasks with "less explicit completion criteria", which a rating task is. No earlier paper-watch calibration experiment exists in this repo.

## Log

### 2026-11-12: the only round

- Fetched 55 full texts. The HTML for 2610.19159 stopped after Table 1, so a PDF fallback was added for any HTML
  conversion under 30K characters.
- Gold: 114 Opus calls, $16 at list price; 4 papers needed a third-pass tie-break.
- The first distillation attempt on Opus failed on the account's weekly limit, after the 66 offline Haiku
  reviews had been checkpointed. The distill was switched to Haiku. Its first answer was over the word limit,
  and the second was 368 words.
- Test stage: 33 papers × 3 runs × 4 conditions (396 records), with no failed calls and no unparsed ratings. It
  took 24 minutes at 8 parallel calls plus 3 sequential D chains, about $3.40 at list price.

Results table (mean over 3 runs, range in brackets):

| | exact | weighted kappa | MEDIUM+ FP rate | MEDIUM+ recall | signed error | tokens / paper |
|---|---|---|---|---|---|---|
| A baseline | 0.44 [0.39, 0.52] | 0.23 [0.20, 0.29] | 0.43 [0.31, 0.48] | 1.00 | +0.58 | 40.0K |
| B + Meta-Skill | 0.69 [0.61, 0.73] | 0.26 [0.15, 0.35] | 0.06 [0.00, 0.14] | 0.33 [0.25, 0.50] | +0.15 | 41.1K |
| C + evolver, reset | 0.69 [0.67, 0.70] | 0.19 [0.14, 0.24] | 0.03 [0.00, 0.07] | 0.17 [0.00, 0.25] | +0.11 | 114.7K |
| D + evolver, accumulate | 0.72 [0.70, 0.73] | 0.25 [0.21, 0.27] | 0.00 | 0.08 [0.00, 0.25] | +0.04 | 122.9K |
| always LOW | 0.67 | 0.00 | 0.00 | 0.00 | +0.09 | — |
| ledger (Haiku + Opus check) | 0.70 | 0.54 | 0.14 | 0.75 | +0.18 | — |

Paired differences in exact agreement (95% bootstrap interval):
- B − A: +0.24 [+0.06, +0.41]
- C − A: +0.24 [+0.05, +0.42]
- D − A: +0.27 [+0.07, +0.47]
- C − B: +0.00 [−0.04, +0.04]
- D − C: +0.03 [−0.02, +0.08]

Gold-MEDIUM test papers and how often each condition hit MEDIUM+ (out of 3 runs):

| paper | A | B | C | D |
|---|---|---|---|---|
| 2610.14224 | 3 | 1 | 0 | 0 |
| 2610.36848 | 3 | 0 | 0 | 0 |
| 2611.05306 | 3 | 3 | 2 | 1 |
| 2611.05139 (Meta 2 itself) | 3 | 0 | 0 | 0 |

**What to adopt in `PROCEDURE.md`.** Adopt neither the Meta-Skill nor the evolver. The Meta-Skill turns visible
false positives into invisible misses, and the evolver costs 2.8x the tokens with no change in ratings. The open
question is gold versus the ledger on the six ledger MEDIUMs that gold rated LOW. If Oskar sides with gold on
those six, the change belongs in step 4, the Opus check: add the gold prompt's explicit checks there (is the
gain inside the paper's own run spread; is the overlap with his stack only shared vocabulary). That change
would leave the reviewer prompt alone. The six papers are 2610.32285, 2610.39302, 2610.20424, 2610.12176,
2610.11765 and 2610.37741.

**Files.**
- `gold.json`: the gold labels, with both passes and the ledger rating.
- `meta_skill.md`: the frozen Meta-Skill.
- `results/offline.json`: offline ratings and the distiller's analysis.
- `results/runs/<cond>_r<k>.json`: raw per-run output, covering every paper's draft and final review, the
  evolver's JSON, standing instructions and tokens.
- `results/metrics.json`: everything in the tables above.
- Raw call records go to `cache/` and are gitignored. Running `python3 fetch.py && python3 gold.py && python3
  offline.py && python3 run_test.py && python3 analyze.py` rebuilds them.
