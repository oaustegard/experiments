# needle-esp32-budget — what it would take to run Needle 2 on an ESP32 with 16 MB

The three preceding experiments asked "can this model be made smaller" without a
number to hit. This prices it against one: **16 MB**.

**Short version: the model file fits and nothing else does.** The weights are
11.16–12.36 MB against 16 MB of PSRAM, comfortably — but the measured working
set is **43.8 MB**, and a routing turn over this catalogue costs roughly
**50 GFLOP** because the tool schemas are re-prefilled every time. The file size
everyone optimises is the one thing that was never the problem.

Everything below separates what was **measured here**, what is **read off
Cactus's own code**, and what is **inferred about hardware I do not have**.

## Measured: the weight budget

| configuration | blob | source |
|---|---|---|
| shipped (`embedding=4,mhc=4,default=2`) | **13.74 MB** | `needle-quantization/` |
| `default=2` — the 4-bit protection dropped | **12.36 MB** | `needle-quantization/`, p=1.00, no accuracy cost |
| best surviving depth cut on top of that | ~11.2 MB | `needle-layer-pruning/`, costs 3.7 accuracy points |

12.36 MB is the honest floor at no accuracy cost. Quantizing further is not
available: ternary packs into the same bytes as 2-bit by construction and costs
39 points; 3- and 4-bit are larger for nothing.

## Measured: the actual runtime footprint, which is the answer

The engine reports `peak_ram_mb` on every completion and
`needle_bsky.router` already threads it onto each Decision — `eval.py` just
drops it. For a memory budget that field is the whole question. `rss_probe.py`
runs twelve real routing turns and reports it beside process RSS:

| configuration | weights | engine peak RAM | process RSS |
|---|---|---|---|
| stock engine weights | 13.74 MB | **43.8 MB** | 43.9 MB |
| shipped spec via `weights=` | 13.74 MB | 57.1 MB | 57.1 MB |
| `default=2` via `weights=` | 12.36 MB | 53.7 MB | 54.3 MB |
| `[9,13)` pruned + `default=2` | 11.16 MB | 50.4 MB | 50.4 MB |

The Python interpreter and imports account for 14.3 MB of that (`rss_before`),
so the stock path costs roughly **29.5 MB of engine against 13.74 MB of weights
— about 16 MB of runtime on top of the model itself.**

The `weights=` rows carry ~13 MB more than the stock row, which is the blob
being resident more than once (Python `bytes` plus the engine's own copy); the
RSS delta tracks about 2.75x each weight delta, consistent with that. The stock
row is therefore the honest one to plan against.

**Against 16 MB this is not close.** The model file fits with room to spare; the
working set is roughly three times the budget. Reducing the weights does not fix
it, because the weights are not what is over.

## From Cactus's code: the KV budget, which cannot be tuned from here

`architecture.py` carries `KV_BUDGET_BYTES = 11 MiB + 512 KiB` and a `per_pos`
formula that, for this config (27 layers, 4 KV heads, head_dim 64, 2 Engram
sites, 8-bit KV with fp32 group scales), works out to **16,704 bytes per cached
position** — 4.28 MB at the pinned 256 window, 3.69 MB at 23 layers. That is
part of the ~16 MB of runtime above, not additional to it.

**And the window is not a knob.** `write_export(..., kv_window=...)` writes the
value into the `.cact` header, so it looked like a way to buy back 1.6 MB.
Exporting at 8, 160, 192, 224, 256, 384 and 512 and scoring all seven on the
62-query set gives **one distinct outcome** — 0.611 / 0.613 / 0.625 / 0.537,
identical at every window. An 8-token attention window cannot serve a
~481-token prompt. The field is recorded and ignored on this inference path;
`export.py` describes it as "the sliding-window width the model was **trained**
with".

## Measured: the two size reductions do not compose

`needle-layer-pruning` found one survivable cut and `needle-quantization` found
one free bit spec. Applied together they are **not** free:

| build | weights | routable |
|---|---|---|
| `[9,13)` pruned, shipped bits | 12.49 MB | 0.574 |
| unpruned, `default=2` | 12.36 MB | 0.630 |
| **both** | **11.16 MB** | **0.389** |

A −0.037 change and a +0.018 change compose to **−0.222**. Neither experiment
could have predicted that from its own arm, and the smallest build in this
series is also the least accurate by a wide margin — 44 points below the
regex baseline that needs no model at all.

## Measured: the compute per routing decision, which is the real wall

| quantity | value |
|---|---|
| 18 `tuned-min` schemas as JSON | **1,642 tokens** |
| the 5 the retrieval head renders | **467 tokens** |
| a query | ~14 tokens |
| prefill throughput, 4-core x86 | 423.9 tok/s |
| decode throughput, 4-core x86 | 177.8 tok/s |
| median turn, 18 declared | ~1,206 ms |

481 prefill tokens ÷ 423.9 tok/s = **1.13 s**, against a measured 1.21 s turn.
**The turn is prefill.** The tools are re-rendered and re-prefilled every time —
there is no prefix cache — which `needle-bsky` also saw from the other side when
a sixth declared tool cost 3.6x the per-turn latency.

At 2 FLOP per parameter per token, one routing decision is

```
45.2M params x 2 x 481 tokens  =  43.5 GFLOP
+ attention 2 x 27 x 481^2 x 512 =  6.4 GFLOP
                                   ----------
                                  ~50 GFLOP per query
```

## Inferred, and the part of this that most needs checking

I do not have an ESP32 and these figures are from training data, not
measurement. **Check them against the specific part before believing them.**

- No ESP32 has 16 MB of internal SRAM. "16 MB" is almost certainly **flash**, or
  **PSRAM** on a larger part. Which one changes the answer, so it is the first
  thing to pin down.
- ESP32-S3: ~512 KB SRAM, up to ~8 MB PSRAM, 16 MB flash typical. Xtensa LX7 at
  240 MHz. Plausible sustained int8 throughput is a fraction of a GFLOP/s.
- ESP32-P4: more SRAM, up to ~32 MB PSRAM, RISC-V at ~400 MHz with vector/AI
  extensions — perhaps a few GFLOP/s.

Against ~50 GFLOP per query that is **tens of seconds to minutes per routing
decision**, not the 1.2 s measured here. PSRAM bandwidth compounds it: a dense
model reads all 12.36 MB of weights per decoded token, and octal PSRAM is tens
of MB/s, not the tens of GB/s a phone has.

Also worth knowing before planning a port: `cactus-needle`'s own
`agent/fetch.py` builds platform tags for **manylinux, macOS and Windows only**.
There is no Xtensa or RISC-V MCU wheel in the package. The `.cact` file is
portable; the engine that reads it, as distributed here, is not.

## What actually fits, if the 16 MB is flash

Weights are read-only, so the sane architecture puts the `.cact` in flash and
keeps only the working set in RAM:

| | |
|---|---|
| model file | 11.16–12.36 MB — fits 16 MB with room |
| measured working set | **43.8 MB** on the only build available here, ~29.5 MB of it engine |
| per query | ~50 GFLOP |

With 16 MB of PSRAM the model *file* is a non-problem. The working set is
roughly 2.7x the budget and the compute is one to two orders of magnitude out.
Nothing in the three preceding experiments moves either: depth is not prunable,
precision is already at the knee, the KV window is inert, and the two size
reductions do not compose.

The measurement caveat is real and load-bearing: this is an x86 build inside a
Python process with a general-purpose allocator, and `peak_ram_mb` tracks
process RSS. An embedded build would be leaner. What transfers is the *shape* —
non-weight runtime is comparable to the weights themselves — not the absolute
number. Anyone seriously targeting this part should measure the embedded engine,
because that is the number that decides it and it is not knowable from here.

## The lever that is left, and it is not in the model

Prefill dominates because 467 tokens of tool schema are re-read every turn. That
is the only large, untouched term, and this repo has already measured what
shrinking it is worth:

| approach | routable | cost per query |
|---|---|---|
| 18 declared, `tuned` wording | 0.704 | ~50 GFLOP |
| deterministic regex → ≤5-tool agent | 0.722 | same per-turn model cost, fewer turns above threshold |
| **regex only, no model at all** (`monad-bsky/regex_only.py`) | **0.833** | **0.022 ms** |

For *this* catalogue the honest recommendation is uncomfortable: **20 lines of
regex route these 18 Bluesky tools more accurately than the model does, at
essentially zero compute, and would run on an ESP32 with 15.9 MB to spare.** The
45M model earns its place where the input is open-ended enough that rules cannot
cover it — and on that catalogue, the prompt would be shorter and the arithmetic
above would improve accordingly.

## Answered since first writing

The 16 MB is **PSRAM**, which settles the capacity half in the model's favour
and does not help: at 43.8 MB measured working set the binding constraint was
never the file. Which ESP32 part still matters for the compute half — S3 and P4
differ by roughly an order of magnitude — but not enough to close a 50 GFLOP
gap.

## Reproduction

```bash
python3 run_kv.py       # the seven-window ladder, ~12 min
```

`results_kv_kv*.json` carry the per-query rows; `size_kv*.json` the blob sizes.
