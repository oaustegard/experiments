# Pre-registration — how much of Needle 2's depth can be deleted without retraining?

Written while the first sweep was running and before any of its results were
read. Predictions are directional where no number was available to predict
against.

## Question

`needle-depth-growth/` showed the stack can be *grown* losslessly but that the
new layers need pretraining to be worth anything. The opposite operation needs
no training at all: delete layers, keep the rest, measure. If a 45M model on a
phone can lose a quarter of its depth and still route tools, that is a real
deployment result — depth is the linear term in both latency and RAM.

## What is held fixed

`needle-bsky/evalset.jsonl` (62 queries, 54 routable, 8 off-topic), the
`tuned-min` schema arm, and `needle-bsky`'s scoring code imported unchanged —
the same configuration `needle-tool-naming` measured at 0.611 routable.

## The two structural constraints, and why cuts are multiples of four

* **MHC lanes.** `Stack.__call__` assigns lane `i % 4` to layer `i`. Deleting
  layers renumbers everything after the cut, so a survivor keeps its lane only
  if the number of layers removed before it is a multiple of 4. Every cut here
  is a contiguous block of size 4, 8 or 12, which satisfies that for every
  survivor. A cut of any other size would perturb lane assignment on top of
  depth, and the measurement could not separate the two.
* **Engram sites.** `engram_layers=(2, 15)` is absolute. A site inside the cut
  is *destroyed* — losing 4.19M parameters of n-gram memory alongside the four
  layers — and a site after the cut is renumbered. Cuts starting at 0–2 swallow
  site 2; cuts starting at 12–15 swallow site 15. **Those eight arms are
  confounded** and are reported separately rather than averaged in.

## The control

Every pruned arm is compared against an **unpruned export**, not against the
stock engine. `Needle(weights=...)` reports `confidence: None` for any supplied
`.cact`, so the confidence head is out of scope for this whole experiment and
the metric is routing accuracy. The control also isolates whatever the export
path itself costs.

## Predictions

| # | prediction |
|---|---|
| P1 | the unpruned export reproduces `canon`'s 0.611 routable, since it re-exports the same weights at the checkpoint's own CQ2 spec |
| P2 | **at least one 4-layer cut lands within 0.05 of the control** — some depth is redundant |
| P3 | the tolerant region is **mid-to-late**, roughly starts 14–22, following the deeper-layers-are-redundant result (Gromov et al. 2024; ShortGPT) |
| P4 | cutting at the very start (0) and at the very end (23) are the two worst unconfounded positions |
| P5 | the eight Engram-destroying cuts are worse than their neighbours at comparable depth |
| P6 | a 12-layer cut (27 → 15, −44%) is **not** viable without retraining: below 0.35 routable everywhere |

**The interesting outcome is P2 failing** — that would say this model has no
redundant depth at all, which for a 45M model already trained to a 27-layer
budget is plausible and would be the more useful finding for anyone tempted to
compress it.

## Reported regardless

Latency is the reason to do this at all, so the control and the best surviving
cut are re-timed on an idle box afterwards — `METHODS.md` already records that a
busy 4-core container inflates Needle latency by an order of magnitude, and this
sweep runs 25 arms back to back.
