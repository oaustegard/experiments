# Luna on-prem vs API — what actually decides it

**Status:** done. Negative result for the framing the question arrived in.
**Date:** 2026-08-15. **Cost:** ~15 web searches, no GPU time, no API spend, ~2 h wall clock.

## The question, and the question underneath it

Asked: *research commercial electricity pricing in Montgomery County MD, find
hardware sufficient to match GPT-5.6 Luna, and compare the raw electricity cost
of running it locally against Luna's $0.22/M input.*

Answered: electricity is **`ANCHOR-input-usd-per-mtok: 0.00349`** per million
input tokens at Montgomery County commercial rates — Luna's direct price is
**57×** that, and its batch price still **29×**.

Then the requester supplied the constraints that make it a real question — 800
office seats, staggered M–F, a 2–3 h nightly batch — and the answer inverted.
Not the arithmetic; the arithmetic held. The **framing**. Electricity is
**`ANCHOR-power-share-total: 3.1`%** of the cost of owning the box. The 57×
is true, load-bearing for nothing, and the most misleading number in this file.

What decides it is a **memory floor**. Matching Luna means running an open
model of equal measured capability; the only one is DeepSeek V4 Pro at 1.6 T
params / ~800 GB in NVFP4; the smallest thing that holds 800 GB is an 8×B200
node at ~$450 k. You buy that floor whether 800 office users fill 5% of it or
85%. At 800 seats they fill **`ANCHOR-peak-util-b: 16.9`%** at the busiest
hour of the busiest day.

## Capability match

Sizing hardware from Luna's parameter count is impossible — OpenAI has not
published one, and the third-party "27 B dense" and "8.8 B dense" claims found
in search summaries are unreliable and appear to conflate other model cards.
So the match is on *measured capability*, via the Artificial Analysis
Intelligence Index:

| model | AA index | weights |
|---|---|---|
| GPT-5.6 Luna (max) | 52 | closed |
| **DeepSeek V4 Pro 0813** (reasoning, max effort) | **53** | **MIT, ungated** |
| Luna (xhigh / high / medium) | 50 / 47 / 39 | closed |

V4 Flash is *not* a match and was rejected as one: Luna beats it 92.3 vs 71.2
GPQA, 62.7 vs 49.1 SWE-bench Pro, 84.7 vs 49.1 Terminal-Bench 2.0. It stays in
`params.json` because it is the honest fallback if the requirement is
sovereignty rather than parity — 284 B/13 B active, ~150 GB, two GPUs instead
of eight.

## Electricity — the part that was asked for and does not matter

Montgomery County is Pepco territory (~567 k customers across Montgomery +
Prince George's). No county tariff exists; you take Pepco's Maryland schedules.

| basis | ¢/kWh |
|---|---|
| EIA MD commercial, Aug 2026 — **model default** | 16.4 |
| EIA MD commercial, Dec 2025 | 15.86 |
| EIA MD industrial | 13.59 |
| Pepco commercial, quoted through 2026-05-31 | 13.25 |
| Pepco SOS price-to-compare, summer 2026 — **supply only, not all-in** | 11.29 |

A 24/7 node is a demand-metered ~100%-load-factor customer, the best case for
amortising demand charges, so the low end is reachable with competitive supply.
Pepco filed for **+23% on distribution** in Nov 2025 and MD commercial rates
already rose ~26% after an early-2026 case. Both are noise against a $450 k box:
at scenario B the whole electricity line is
**$`ANCHOR-power-usd-per-year: 8,710`/yr**, or $11 per seat per year.

## The 800-seat model

800 seats, 55% daily-active, 250 workdays, 07:00–18:30 coverage, 15% of a
day's requests in the busiest hour (this is where the power-curve distribution
enters — it drives peak sizing, not annual totals), 2.5 h nightly batch.

| | req/day/active | in/req | out/req | in/day | out/day | peak util | Luna $/yr |
|---|---|---|---|---|---|---|---|
| **A** light chat/office | 30 | 5 k | 700 | 0.07 B | 0.01 B | 5% | 6,072 |
| **B** docs + some agentic | 60 | 15 k | 1.2 k | 0.40 B | 0.03 B | 17% | **`ANCHOR-api-b-usd: 29,304`** |
| **C** heavy agentic, whole co. | 120 | 40 k | 3 k | 2.11 B | 0.16 B | 85% | 153,120 |

Against self-hosting at **$`ANCHOR-selfhost-total-usd: 278,710`/yr**
(capex/3 + power + $120 k colo/network/spares/~0.4 FTE), one node:

| scenario | Luna interactive | + saturated batch | self-host | verdict |
|---|---|---|---|---|
| A | 6,072 | 78,476 | 278,480 | **API**, 3.5× |
| B | 29,304 | 101,708 | 278,710 | **API**, 2.7× |
| C | 153,120 | 225,524 | 280,006 | **API**, 1.2× |

Duty cycle across the year: 11–26%.

## The finding: the batch window cannot close the gap, by construction

The nightly batch is the one window where the box could saturate. It can't
close it, and not for a reason that depends on any usage estimate.

- Box capacity, 2.5 h fully saturated: **`ANCHOR-batch-tok-night: 1.984`B**
  input tokens/night.
- Break-even against all-in self-host cost at Luna's batch price:
  **`ANCHOR-batch-breakeven-tok-night: 7.64`B** input tokens/night.

The box is short by **3.9×** at 100% saturation. There is no batch workload you
can invent that fixes this, because the ceiling is the hardware's, not the
workload's. Even at the 35% MFU end of the sensitivity band it reaches 2.31 B.
And the batch tier is priced at $0.10/M — half the interactive rate — so the
one window where you could run the box flat out is also the window where the
API is cheapest. Those two facts compound.

## Where it does flip

Not on seat count. On **token intensity per node**:

| seats | scenario | peak | nodes | Luna + batch | self-host | verdict |
|---|---|---|---|---|---|---|
| 800 | B | 17% | 1 | 101,708 | 278,710 | API |
| 2,000 | B | 42% | 1 | 145,664 | 279,196 | API |
| 4,000 | B | 85% | 1 | 218,924 | 280,006 | API |
| 6,000 | B | 127% | 2 | 364,588 | 439,202 | API |
| 2,000 | **C** | 211% | 3 | 600,012 | 599,208 | **self-host** |
| 4,000 | **C** | 423% | 5 | 1,127,620 | 920,030 | **self-host** |

Scenario B never flips, at any seat count. Adding seats adds nodes as fast as
it adds savings once you pass one node, so seat count alone does not converge —
what converges is tokens *per node*. Only scenario C's intensity fills a node
enough to matter, and it does so at ~2,000 seats.

Two caveats on those self-host wins. Every one of them is a multi-node fleet
with no N+1 (add a node to each row for redundancy), and ops is held flat at
$120 k while the fleet grows, which flatters the self-host column at 5–7 nodes.

## The 8×B200 decode result

An 8×B200 node costs **$0.083 per million output tokens** in electricity;
a GB200 NVL72 rack costs **$0.014** — 5.8× better, on the same GPUs at the
same price per kWh. Decode on a 1.6 T MoE is all-to-all bound, so throughput
per GPU rises with NVLink-domain size (976 tok/s/GPU on an 8-GPU node vs 6,644
on the 72-GPU rack). Prefill, being compute-bound, is nearly flat between them
($0.0030 vs $0.0035/M — the node is *better*, having a lower PUE).

The practical consequence: **the small box is a fine prefill engine and a poor
decode engine.** Anyone sizing on-prem inference from rack-scale benchmarks
will overestimate a single node's decode economics by ~6×.

## What would change the answer

In descending order of leverage:

1. **A workload 4× scenario C's intensity**, or ~2,000 seats of genuine
   agentic use. That is the only thing here that flips it.
2. **A capability requirement that isn't Luna-parity.** V4 Flash fits in
   ~150 GB — 2× RTX PRO 6000 Blackwell, ~$27 k in GPUs, 1.2 kW. Two orders of
   magnitude off the $450 k floor. This is the largest cost lever in the whole
   analysis and it is a *scope* decision, not a procurement one.
3. **Prompt caching.** Input is >90% of token volume in every scenario. As a
   scale reference DeepSeek prices cache hits at $0.003625/M — within 4% of the
   raw electricity cost computed above. Cache discipline moves the bill more
   than any hardware decision in this document.
4. **Electricity.** Free power does not flip any scenario at 800 seats.
   `recheck.py` asserts this as a negative control.

## Reproduce

```bash
python3 model.py --all-scenarios --first-pass --sweep-mfu
python3 model.py --seats 2000 --scenario C          # the crossover
python3 model.py --ops 0 --nodes 2                  # floor cost, with N+1
python3 recheck.py                                  # 97 checks, ~4 s
```

Drop real telemetry into `params.json:workload.scenarios` — that is the input
worth replacing first, and the one with the least support behind it.

## Confidence

**Provenance is the weak point and it is uniform.** The session's egress proxy
allowlisted nothing: `WebFetch` and `curl` returned `EGRESS_BLOCKED` or a 403
CONNECT tunnel for openai.com, openrouter.ai, artificialanalysis.ai, eia.gov,
pepco.com, venturebeat.com, together.ai, deepseek.ai and every other primary
source. **Every figure in `params.json` was read out of a search engine's
summary of a page, not the page.** Rows are tagged `confidence` accordingly.
Nothing here should inform a purchase order until the tariff schedule and the
InferenceX throughput numbers are re-read at source.

Ranked by how much a wrong value would move the conclusion:

| input | confidence | if wrong |
|---|---|---|
| workload scenarios A–C | **authored, unsourced** | dominates everything; replace with telemetry |
| $450 k capex, $120 k ops | secondary / authored | sets the floor the whole result turns on |
| prefill MFU 30% | estimated | swept 20–35%; batch route fails across the whole band |
| 8×B200 idle 2.7 kW | estimated | only affects the electricity line, which is 3% |
| electricity rates | secondary | **immaterial — proven by negative control** |

The published per-seat figures I could find (FinOps LLM: "light analyst" =
3 M input tok/day) sit ~7× above even scenario C per active user. I rejected
them as not credible for office work rather than use them; had I used them the
verdict would have flipped, so this is the single judgement call most worth
challenging.

Prior-art checks both clean: `repo-index/ask.py` on *"cost model comparing
self-hosted GPU inference electricity against API token pricing"* returned
nothing related, and account-wide `xr` topped out at 0.478 on an unrelated
file — inside the in-corpus-miss band.

See [`ERRORS.md`](ERRORS.md) — four errors, two of them mine in the model code,
and every one pointed the same way.
