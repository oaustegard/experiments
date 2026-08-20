# needle-esp32-budget — what it would take to run Needle 2 on an ESP32 with 16 MB

The three preceding experiments asked "can this model be made smaller" without a
number to hit. This prices it against one: **16 MB**.

**Short version: the bytes almost fit and the arithmetic does not.** A routing
turn over this catalogue costs roughly **50 GFLOP**, every turn, because the tool
schemas are re-prefilled each time. That is 1.2 s on a 4-core x86 container and
one to two orders of magnitude worse on any ESP32. Memory is the constraint that
looks binding and compute is the one that actually is.

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

## From Cactus's code: the KV budget, which cannot be tuned from here

`architecture.py` carries `KV_BUDGET_BYTES = 11 MiB + 512 KiB` and a `per_pos`
formula that, for this config (27 layers, 4 KV heads, head_dim 64, 2 Engram
sites, 8-bit KV with fp32 group scales), works out to **16,704 bytes per cached
position**:

| window | KV cache |
|---|---|
| 512 | 8.55 MB |
| 256 (what the checkpoint pins) | **4.28 MB** |
| 160 (`KV_WINDOW_MIN`) | 2.67 MB |

So the resident total at the shipped settings is **13.74 + 4.28 = 18.02 MB**, or
**12.36 + 4.28 = 16.64 MB** at the best free weight spec — over 16 MB before a
single byte of activations, tokenizer, grammar table, engine code or
application.

**And the window is not a knob.** `write_export(..., kv_window=...)` writes the
value into the `.cact` header, so it looked like a way to buy back 1.6 MB.
Exporting at 8, 160, 192, 224, 256, 384 and 512 and scoring all seven on the
62-query set gives **one distinct outcome** — 0.611 / 0.613 / 0.625 / 0.537,
identical to four decimal places at every window:

| kv_window | routable | tool | refusal | args |
|---|---|---|---|---|
| 8 | 0.611 | 0.613 | 0.625 | 0.537 |
| 160 → 512 | 0.611 | 0.613 | 0.625 | 0.537 |

An 8-token attention window cannot serve a ~481-token prompt. The field is
recorded and ignored on this inference path — `export.py` even describes it as
"the sliding-window width the model was **trained** with". So the 4.28 MB is a
number to plan around, not one to reduce from the export side.

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
| flash | 12.36 MB of 16 MB — leaves ~3.6 MB for app and partition table, and **no room for a second OTA slot** |
| RAM | 4.28 MB KV + activations — fits 8 MB PSRAM, does not fit 512 KB SRAM |
| per query | ~50 GFLOP |

The capacity story is survivable. The compute story is not, and nothing in the
three preceding experiments moves it: depth is not prunable, precision is
already at the knee, and the KV window is inert.

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

## Open question

Whether the 16 MB is flash or PSRAM, and which ESP32 part, decides the RAM half
of this. The compute half is unaffected either way.

## Reproduction

```bash
python3 run_kv.py       # the seven-window ladder, ~12 min
```

`results_kv_kv*.json` carry the per-query rows; `size_kv*.json` the blob sizes.
