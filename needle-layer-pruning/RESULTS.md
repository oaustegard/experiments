# needle-layer-pruning — how much of Needle 2's depth can be deleted without retraining?

`needle-depth-growth/` showed the stack can be grown losslessly but that the new
layers need pretraining to be worth anything. Pruning is the operation that needs
no training at all: delete layers, re-export, measure.

**The budget is four layers, at exactly one position out of twenty-four, for 3.7
accuracy points — and it buys a 9% smaller file and no faster turns.** Two
further findings are worth more than that headline: the redundant depth is in the
*middle* of the stack, not the deep end, which inverts the large-model result;
and the standard cheap heuristic for finding it does not work here at all
(Spearman **+0.061**), so the 24 evaluations were not avoidable.

Predictions were committed in [`PREREG.md`](PREREG.md) at `0e36eac`, before any
result was read. Two held, three failed, one is inverted.

## Setup

`needle-bsky/evalset.jsonl` (62 queries, 54 routable, 8 off-topic), the
`tuned-min` schema arm, `needle-bsky`'s scoring code imported unchanged — the
same configuration measured at 0.611 in both sibling experiments.

Cuts are contiguous blocks whose size is a multiple of 4, because that is what
preserves the MHC lane assignment (`Stack.__call__` gives layer `i` lane `i % 4`,
so a survivor keeps its lane only if the number of layers removed before it is a
multiple of 4). `engram_layers=(2, 15)` is absolute: a cut swallowing a site
destroys a 4.19M-parameter n-gram table, which happens for starts 0–2 and 12–15
— **seven** arms, not the eight `PREREG.md` says (`ERRORS.md` #1).

Every arm is compared against an **unpruned export**, never the stock engine,
because `Needle(weights=...)` reports `confidence: None` for any supplied
`.cact`. The confidence head is therefore out of scope for this whole experiment.

**The control validates the export path.** It scores 0.611 routable / 0.613 tool
/ 0.625 refusal / 0.537 args — identical to `needle-bsky`'s `tuned-min` and
`needle-tool-naming`'s `canon`, and reproduced on five separate runs across the
session. **P1 held.** Re-exporting the fp16 checkpoint at its own CQ2 spec
reproduces the shipped engine's behaviour exactly.

## Every position of a four-layer cut (27 → 23 layers, −14.8% depth)

Paired exact McNemar against the control over the same 54 routable queries.

| cut | routable | Δ | refusal | Engram lost | ctl-only | arm-only | p |
|---|---|---|---|---|---|---|---|
| `[ 9,13)` | **0.574** | −0.037 | 0.375 | — | 10 | 8 | 0.81 |
| `[12,16)` | 0.500 | −0.111 | 0.625 | site 15 | 7 | 1 | 0.070 |
| `[ 7,11)` | 0.481 | −0.130 | 0.000 | — | 10 | 3 | 0.092 |
| `[13,17)` | 0.463 | −0.148 | 0.250 | site 15 | 11 | 3 | 0.057 |
| `[ 8,12)` | 0.444 | −0.167 | 0.750 | — | 13 | 4 | **0.049** |
| `[10,14)` | 0.407 | −0.204 | 0.625 | — | 14 | 3 | **0.013** |
| `[ 4, 8)` | 0.389 | −0.222 | 0.625 | — | 15 | 3 | **0.008** |
| `[11,15)` | 0.389 | −0.222 | 0.500 | — | 16 | 4 | **0.012** |
| `[14,18)` | 0.370 | −0.241 | 0.125 | site 15 | 15 | 2 | **0.002** |
| `[ 6,10)` | 0.352 | −0.259 | 0.625 | — | 17 | 3 | **0.003** |
| `[ 5, 9)` | 0.241 | −0.370 | 0.875 | — | 22 | 2 | **<0.0001** |
| `[18,22)` | 0.185 | −0.426 | 0.625 | — | 27 | 4 | **<0.0001** |
| `[ 1, 5)` | 0.167 | −0.444 | 1.000 | site 2 | 25 | 1 | **<0.0001** |
| `[23,27)` | 0.167 | −0.444 | 0.750 | — | 24 | 0 | **<0.0001** |
| `[16,20)` | 0.148 | −0.463 | 0.875 | — | 26 | 1 | **<0.0001** |
| `[17,21)` | 0.148 | −0.463 | 0.375 | — | 28 | 3 | **<0.0001** |
| `[ 3, 7)` | 0.130 | −0.481 | 0.625 | — | 28 | 2 | **<0.0001** |
| `[ 2, 6)` | 0.111 | −0.500 | 0.375 | site 2 | 30 | 3 | **<0.0001** |
| `[19,23)` | 0.056 | −0.555 | 0.875 | — | 30 | 0 | **<0.0001** |
| `[15,19)` | 0.037 | −0.574 | 1.000 | site 15 | 31 | 0 | **<0.0001** |
| `[20,24)` | 0.018 | −0.593 | 0.750 | — | 32 | 0 | **<0.0001** |
| `[ 0, 4)` | 0.000 | −0.611 | 1.000 | site 2 | 33 | 0 | **<0.0001** |
| `[21,25)` | 0.000 | −0.611 | 1.000 | — | 33 | 0 | **<0.0001** |
| `[22,26)` | 0.000 | −0.611 | 1.000 | — | 33 | 0 | **<0.0001** |

**One position of twenty-four survives.** `[9,13)` costs 0.037 at p=0.81 — 10
queries the control got and the pruned model lost, 8 the other way. That is a
genuine null at n=54, not a small loss. Twenty of the twenty-four are significant
at p<0.05, and five leave a model that refuses almost everything (`refusal`
1.000 means it emits Needle's empty call for every query, routable and off-topic
alike — those arms are not misrouting, they are dead).

**P2 held, barely.** Some depth is redundant, but "some" is one block in
twenty-four.

## The redundant depth is in the middle, and P3 was backwards

The tolerant zone is starts **7–14**. Starts 16–23 measure 0.000–0.185; starts
0–3 measure 0.000–0.167. `PREREG.md` predicted the tolerant region would be
mid-to-late (starts 14–22), following Gromov et al. 2024 and ShortGPT, where the
*deeper* layers of a large decoder are the redundant ones. **That is falsified
here, and inverted at the deep end**: `[20,24)`, `[21,25)` and `[22,26)` are
three of the five worst positions in the sweep.

P4 half-held: cutting at 0 is jointly the worst (0.000), but the tail `[23,27)`
is not — at 0.167 it is mid-table, better than eight other positions.

## The cheap heuristic does not find it

The standard depth-pruning score is the angular distance between the
representation entering a block and the one leaving it: low distance means the
block is near-identity and should be the one to delete. Computed on the same 62
queries, one forward pass, no engine:

| | |
|---|---|
| Spearman(heuristic, measured accuracy) over 24 blocks | **+0.061** |
| heuristic's top pick | `[20,24)` → measured **0.018** (3rd worst) |
| sweep's actual best | `[9,13)` → ranked **10th** by the heuristic |

The failure is not a sign error, it is an inability to separate. Ranked by the
heuristic, the six most redundant-looking blocks are:

| block | angular distance | measured routable |
|---|---|---|
| `[20,24)` | 0.0385 | 0.018 |
| `[19,23)` | 0.0403 | 0.056 |
| `[18,22)` | 0.0462 | 0.185 |
| `[ 7,11)` | 0.0503 | **0.481** |
| `[21,25)` | 0.0503 | 0.000 |
| `[22,26)` | 0.0541 | 0.000 |

The 3rd-best position in the whole sweep and the two worst sit inside a band of
0.016 in the score. Four of those six are late blocks that **barely move the
residual stream and are still load-bearing** — whatever they do, the output head
is calibrated to it — while `[7,11)`, with an indistinguishable distance, is
genuinely near-redundant.

The practical consequence is that the 24 evaluations were not avoidable. A method
validated on 7B–70B decoders transferred to a 45M attention-only model with a
rank correlation indistinguishable from zero.

## Destroying an Engram table costs less than removing the wrong four layers

**P5 is inverted.** `[12,16)` and `[13,17)` each destroy the Engram site at layer
15 — a 4.19M-parameter hash-indexed n-gram table, **9.3% of the model** — *and*
remove four layers, and they rank 2nd and 4th best at 0.500 and 0.463.
`[21,25)` and `[22,26)` destroy no table, remove four layers, and score 0.000.

So on this task the entire n-gram memory subsystem is worth less than four
correctly-chosen layers. That is consistent with what the two sibling experiments
found about where this model's routing competence lives — in selection over
provided context, not in recalled facts — and it is the clearest single piece of
evidence for it.

## Deeper cuts collapse

| cut | layers | best routable | worst |
|---|---|---|---|
| 4 layers, 24 positions | 23 | 0.574 | 0.000 |
| 8 layers, starts 6–11 | 19 | 0.148 (`[10,18)`) | 0.018 |
| 12 layers, start 8 | 15 | **0.000** (refuses everything) | — |

**P6 held**, and with more margin than predicted: the 12-layer cut does not reach
0.35, it reaches zero. Note the 8-layer sweep covered the *tolerant* zone found
by the 4-layer sweep and still topped out at 0.148, so this is not a matter of
having looked in the wrong place.

## What pruning actually buys

Per-token throughput improves about as depth says it should. End-to-end turn
latency does not move at all.

| | control (27L) | `[9,13)` (23L) | change |
|---|---|---|---|
| prefill | 423.9 tok/s | 518.6 tok/s | **+22.3%** |
| decode | 177.8 tok/s | 228.8 tok/s | **+28.7%** |
| turn latency, 18 tools, 3 interleaved reps | 1227 ms | 1227 ms | **0.0%** |
| exported blob | 13.74 MB | 12.49 MB | −9.1% |
| routable | 0.611 | 0.574 | −0.037 |

23/27 layers predicts a 17.4% per-token speedup; the measured 22–29% is at or
above that. But the turn does not get faster, for two compounding reasons.
`needle-bsky` measured that declaring a sixth tool costs 3.6× the per-turn
latency and then nothing more out to 18, because the contrastive retrieval head
is a **fixed per-turn cost** above five — and a degraded model emits more tokens,
spending the per-token gain. Both push the same way, and 1227 → 1227 ms with
per-run medians of 1224/1227/1255 against 1185/1227/1265 leaves no room for a
real effect.

Wall-clock at a five-tool catalogue said the pruned model was *slower* (247 → 354
ms median). That is the output-length confound, not a depth effect, which is why
the table above reports per-token rates (`ERRORS.md` #3).

## The verdict

Removing 4 of 27 layers at the single position that tolerates it costs 3.7 points
of routing accuracy, saves 1.25 MB of a 13.74 MB blob, and makes turns no faster.
Every other cut in the sweep costs 11 to 61 points. For a model whose deployment
case is a phone or a microcontroller, 9% of a 14 MB file is not worth a
measurable accuracy loss and a bespoke, unvalidatable-by-heuristic surgery — and
the two things that *would* justify it, a latency win or a much smaller model,
are not available.

The useful sentence for anyone else holding a small tool-calling model: **a 45M
model trained to a 27-layer budget has essentially no slack in its depth.** The
redundancy that large decoders carry in their deeper layers is not there.

## Reproduction

```bash
python3 -m pip install --break-system-packages cactus-needle
python3 run_sweep.py --count 4 --control      # 25 arms, ~35 min on 4 CPU cores
python3 run_sweep.py --count 8 --starts 6 7 8 9 10 11
python3 run_sweep.py --count 12 --starts 8
python3 analyze.py                            # ranking + paired tests -> analysis.json
python3 importance.py --count 4               # the heuristic and its correlation
python3 timing.py --reps 3 --ctl X.cact --cut Y.cact
python3 throughput.py --weights X.cact --label control
python3 recheck.py                            # every number above against the artifacts
```

## Caveats

- **n=54 routable.** One query is 1.85 points. The surviving arm's null (p=0.81)
  is a null at this n, not a demonstration of equivalence; a 3.7-point loss is
  what it looks like, and the sweep cannot resolve smaller.
- **One task, one catalogue.** Routing over 18 Bluesky read tools. A pruned model
  might hold up better on something less discriminative, and the throughput
  conclusion is specific to a catalogue large enough that retrieval dominates the
  turn.
- **Cuts are contiguous and lane-aligned.** Non-contiguous pruning, or cuts of
  size 1–3, would perturb the MHC lane assignment on top of depth. Whether a
  lane-aware renumbering would open up finer-grained pruning is untested.
- **No healing.** The literature's depth-pruning results usually recover several
  points with a short post-prune fine-tune. That is unavailable here: the shipped
  trainer is LoRA-only, it would take the confidence head with it
  (`needle-bsky`), and 800 templated rows are the whole corpus on hand.
- **Confidence is out of scope**, since every `weights=` path reports `None`.
  A pruned model's gate calibration is unmeasured and should be assumed broken.
