# Margin-based lossy verification on the Baguettotron drafter

At γ=1 the margin rule lifts acceptance from 0.350 to 0.527 and wall-clock from
1.09× to 1.25×, and it pays for that by leaving the greedy path after 15 of 48
generated tokens. The ceiling is set by one number: the median margin at a
mismatch is 0.0001. When this drafter and Baguettotron disagree, the target is
usually certain the draft is wrong, so there is very little near-tie mass for
any threshold to harvest.

Source: AdaptiveSpec, arXiv [2609.02897](https://arxiv.org/abs/2609.02897)
(Urbán, Kwon, Venieris, Mascolo — Samsung AI Center Cambridge / University of
Cambridge). Run 2026-09-05 by CCotw. Code `margin_verify.py`, data
`margin_verify.json`.

## The rule

At each candidate position the target's own distribution decides whether a
mismatched draft token survives:

```
margin(k) = p_target(draft_k) / p_target(top1_k) = exp(logit[k, draft_k] - logit[k, top1_k])
```

Accept when `margin(k) >= κ`. The softmax denominators cancel, so the ratio is a
subtraction of two logits the verify pass already computed. No lookahead
window, no trained judge, no dependence on draft length.

An exact match scores margin exactly 1.0, so one condition covers both regimes
and κ=1.0 is the lossless control. `lossless_control` in the DAG asserts that
κ=1.0 reproduces plain greedy token-for-token at every γ before the sweep runs;
it passed, so the rows below measure the rule rather than a decoder bug.

The paper's second axis, per-step draft-tree shaping from a Draft Confidence
Score, is not implemented here. It presupposes tree attention, a verify-token
budget, and one pre-captured CUDA graph per `(n_steps, top-k, ndt)` triplet.
This harness drafts a chain on CPU and has none of that.

## Setup

Baguettotron 321M fp32 target, the 4.2M-param EAGLE head from
`eagle_head_s8.pt`, 4 CPU threads, 4 chat-templated prompts, 48 new tokens each,
greedy. κ ∈ {1.0, 0.5, 0.3, 0.2, 0.1, 0.05} × γ ∈ {1,2,3} × promotion modes
`first` (one promotion per verify round) and `all` (the test runs at every
mismatch in the block). `projected_cached_draft` applies the same
`(1-α^(γ+1))/((1-α)(1+γc))` model at c=0.049 used elsewhere in this directory,
because the head still re-runs its whole prefix each round and wall-clock
understates it.

## Results

γ=1, averaged over the four prompts:

| κ | acceptance | wall-clock | projected | promotions | greedy prefix | mean logprob |
|---|---|---|---|---|---|---|
| 1.0 (lossless) | 0.350 | 1.095× | 1.287× | 0 | 48/48 | −0.282 |
| 0.5 | 0.370 | 1.099× | 1.306× | 1 | 36.8/48 | −0.299 |
| 0.3 | 0.422 | 1.171× | 1.355× | 7 | 26.0/48 | −0.334 |
| 0.2 | 0.430 | 1.150× | 1.363× | 8 | 19.0/48 | −0.409 |
| 0.1 | 0.527 | 1.246× | 1.456× | 16 | 14.8/48 | −0.453 |
| 0.05 | 0.503 | 1.199× | 1.433× | 15 | 14.8/48 | −0.432 |

γ=2 moves the same direction from a lower base (0.244 → 0.354 at κ=0.05, 1.108×
→ 1.232×). γ=3 starts below parity at 0.86× and stays below it at 0.99×, so the
rule does not rescue a γ the drafter cannot support.

The acceptance lift holds on every prompt individually at γ=1, κ=0.1: 0.412 →
0.412, 0.333 → 0.548, 0.455 → 0.600, 0.200 → 0.548. All four speed up.
Magnitudes vary a lot. Sixteen promotion events across four sequences is a
pilot; the direction is what these numbers support, and not the size.

## Mismatch-margin distribution

The κ=1.0 control never promotes, so it observes the mismatch-margin
distribution undisturbed:

| γ | mismatches | p50 | p90 | max | ≥0.3 | ≥0.1 | ≥0.05 |
|---|---|---|---|---|---|---|---|
| 1 | 90 | 0.0001 | 0.274 | 0.426 | 5.5% | 15.8% | 21.1% |
| 2 | 108 | 0.0001 | 0.173 | 0.556 | 4.9% | 14.4% | 18.8% |
| 3 | 113 | 0.0000 | 0.151 | 0.462 | 3.5% | 15.7% | 20.0% |

Half of all disagreements are ones where Baguettotron gives the drafted token
about 1/10,000 of its top choice's probability, and no mismatch anywhere in the
sweep exceeded 0.556. A 4.2M head trained on next-token prediction alone
proposes tokens the target rejects outright. Near-synonyms are rare among its
misses. The paper reports its gains on 8B targets with published EAGLE-3 heads at
acceptance 0.7–0.8; a stronger drafter should put far more mass near 1.0, which
is where a threshold has something to work with. This result is about our
drafter.

## Non-monotonic acceptance in κ

κ=0.05 scores *worse* than κ=0.1 at γ=1 on acceptance (0.503 vs 0.527) and on
speed (1.199× vs 1.246×). Admitting more tokens made the run slower.

Promoting a token changes the context every later draft step conditions on, so
the trajectory a lower κ produces is a different sequence with its own
acceptance profile. Acceptance is therefore not a monotone function of κ.

The paper carries a much larger version of the same shape and does not name it:
in its Table 4(a), DeepSeek-R1-8B on MATH-500 goes κ=0.10 → 2.93×/97%, κ=0.20 →
1.80×/111%, κ=0.30 → 1.66×/109% — throughput collapsing 1.6× across one κ step
while recovery exceeds 100%. Their per-cell κ picks are searched against the
same benchmark that reports the result, with no held-out split, so a
trajectory-dependent metric is being tuned on its own test set.

## greedy_prefix against mean logprob

`greedy_prefix` is brittle: one early promotion truncates it for the whole
sequence, which is why it ranges 3–31 of 48 across prompts at a single setting.
Mean logprob under the target is steadier and degrades everywhere: −0.282
lossless against −0.453 at κ=0.1.

Neither is task accuracy. Running GSM8K on a 321M reasoning model would measure
Baguettotron, not the verifier, so this sweep does not report a recovery
percentage and the paper's 93%-to-lossless figure has no counterpart here.

## Operating points

κ=0.3 costs 0.05 nats of mean logprob for +7% wall-clock and +5% projected.
κ=0.1 costs 0.17 nats for +14% and +13%. Both are below the 1.32× that the
KV-cache fix in `HANDOFF.md` projects for the lossless decoder, so on this
drafter the cache work is worth more than the verifier relaxation and the two
are independent.

## Running it

```
python3 margin_verify.py eagle_head_s8.pt
```

The DAG is `env_check → load → baselines → lossless_control → sweep → report`,
journaled to `.margin_verify_journal.jsonl`; a re-run replays completed tasks and
only executes what changed. Full grid is ~15 min on 4 CPU cores.
