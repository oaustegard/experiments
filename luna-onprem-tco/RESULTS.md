# Luna on-prem vs API — what actually decides it

**Status:** done, two passes. Negative result for the framing the question arrived in.
**Date:** 2026-08-15 (fleet), 2026-08-16 (single-GPU). **Cost:** ~21 web searches, no GPU time, no API spend, ~3 h wall clock.

Two scales, one price book. `model.py` asks *"do we buy a cluster for 800
seats"* and answers **no, by 2.7×**. `hourly.py` asks *"is one RTX 5090 cheaper
than the API while it runs"* and answers **yes, above ~4 h/day of flat-out
generation**. The same principle decides both, in opposite directions: what
matters is not the per-token cost but how much of a fixed capacity you fill.

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

## Second pass: one RTX 5090, per hour — `hourly.py`

Same question, 1/1000th the scale, and the answer is different in kind. Local
model: **Qwen3.8-27B** (released 14 Aug 2026, Apache 2.0, 27.78 B **dense**,
262 K context, Gated DeltaNet attention + multi-token prediction). Unlike V4
Flash in the fleet analysis, this is a near-match on coding — **SWE-Bench Pro
61.7% against Luna's 62.7%** — though Luna still leads decisively on knowledge
breadth (GPQA 92.3%) and on context (1.05 M vs 262 K).

**Electricity is $`ANCHOR-5090-power-res: 0.180`/hr** — 600 W card + 90 W CPU +
40 W platform, ÷ 0.90 PSU = **`ANCHOR-5090-wall-w: 811` W** at the wall, at
Maryland's EIA-corrected **residential** 22.2 ¢/kWh ($0.133 at the commercial
rate; +28% if summer AC has to remove the heat).

Two premise corrections came first, and both moved the answer:

**The quoted 180–200 TPS is above the single-stream decode roofline.** 27.78 B
dense at ~4.25 effective bits is 14.8 GB, read once per token from 1,792 GB/s,
so the hard ceiling is **`ANCHOR-5090-ceiling: 121` tok/s** and ~97 at a
realistic 80% MBU. 190 is **1.56×** the hard ceiling. Reachable — this model
ships multi-token prediction, and batching or speculative decoding get there —
but not by plain decode. Every result is reported at both 190 and 95.

**Prefill and decode share the one card.** There is no separate prefill pool on
a desktop. At ~2,073 tok/s prefill (the measured 7,200 tok/s for Llama-3.1-8B
scaled by parameters), the advertised decode rate is never the sustained rate:

| profile | billed in:out | cache hit | fresh in:out | sustained out | Luna $/hr | × electricity |
|---|---|---|---|---|---|---|
| chat / assistant | 8:1 | 60% | 3.2 | 147 tok/s | $1.11 | 6.2× res, 8.3× comm |
| agentic coding | 40:1 | 85% | 6.0 | **`ANCHOR-agentic-sustained: 122.6`** tok/s | **$`ANCHOR-agentic-luna-hr: 1.49`** | **8.3×** res, 11.2× comm |
| bulk generation | 1.5:1 | 10% | 1.35 | 167 tok/s | $0.90 | 5.0× res, 6.8× comm |

**So 5–11×, against the fleet analysis's 57×.** The gap collapsed because a
consumer card serving one stream is a poor token engine: **$0.19–0.26 per
million output tokens** against the 8×B200 node's $0.083 and the NVL72 rack's
$0.014. Single-stream inference is the least energy-efficient way to make a
token, and batching is most of what datacenter economics *is*.

### A price-book result worth carrying separately

Luna bills cache **writes** at 1.25× uncached input ($0.25/M) and reads at 0.1×
($0.02/M). So **caching only pays above a
`ANCHOR-cache-breakeven: 21.7`% hit rate** — below it, turning caching on costs
more than re-sending. The `bulk` profile above sits at 10% and therefore runs
uncached, deliberately.

### And capex decides it again — but this time it is winnable

The RTX 5090 is in a shortage: US street median **$4,700** in August 2026
(in-stock range $3,900–5,000); the $1,999 Founders Edition does not exist at
retail and AIB baselines start at $2,900. Call the box $6,000.

| flat-out h/day | capex $/hr | + power | all-in |
|---|---|---|---|
| 1 | 5.48 | 0.18 | $5.66 |
| 2 | 2.74 | 0.18 | $2.92 |
| 4 | 1.37 | 0.18 | $1.55 |
| 8 | 0.68 | 0.18 | $0.86 |
| 24 | 0.23 | 0.18 | $0.41 |

Break-even against Luna, at 190 tok/s:

| profile | $6,000 street box | $3,299 MSRP-era box |
|---|---|---|
| chat / assistant | **`ANCHOR-chat-be-hday: 5.9` h/day** | 3.2 h/day |
| agentic coding | **`ANCHOR-agentic-be-hday: 4.2` h/day** | 2.3 h/day |
| bulk generation | 7.6 h/day | 4.2 h/day |

At the roofline-respecting 95 tok/s these become 12.3 / 7.5 / 18.3 h/day — chat
and bulk stop being reachable at all.

Two things fall out. **The GPU shortage roughly doubles the break-even duty
cycle**, which makes the used-card market a bigger swing factor than
electricity, model choice or rate schedule. And **"h/day" means flat-out
generation, not hours with the tool open** — interactive chat emits tokens
perhaps 5–10% of wall-clock time, so 4 h/day of real generation is ~1.8 M
output tokens/day. That is an always-on agent or a batch pipeline. A person
typing will never reach it, and conflating the two is the easiest way to talk
yourself into the purchase.

### The lever

Every number above is **single-stream**, the worst case for both throughput and
energy per token. Serving 4–8 concurrent requests should multiply aggregate
output for the same 811 W, cutting $/M output proportionally and collapsing the
break-even duty cycle — and a box running an agent fleet reaches "flat out"
honestly where a chat session never will. No measured batched figure for
Qwen3.8-27B on a 5090 exists yet, so this is flagged as the thing to benchmark,
not claimed.

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
python3 model.py --seats 2000 --scenario C          # the fleet crossover
python3 model.py --ops 0 --nodes 2                  # floor cost, with N+1

python3 hourly.py                                   # single-GPU, both decode branches
python3 hourly.py --decode 120 --ac                 # roofline-respecting, with summer AC
python3 hourly.py --capex 3299                      # the box at MSRP-era card prices

python3 recheck.py                                  # 154 checks, ~8 s
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
| workload scenarios A–C, hourly profiles | **authored, unsourced** | dominates everything; replace with telemetry |
| $450 k capex, $120 k ops; $6,000 desktop | secondary / authored | sets the floor both results turn on |
| 190 tok/s decode premise | **user-supplied, above roofline** | halving it roughly doubles every break-even duty cycle; both branches reported |
| prefill MFU 30% (fleet) | estimated | swept 20–35%; batch route fails across the whole band |
| 2,073 tok/s prefill (5090) | **scaled, not measured** | sets the prefill-contention term; an 8 B measurement extrapolated 3.5× |
| 8×B200 idle 2.7 kW | estimated | only affects the electricity line, which is 3% |
| electricity rates | secondary | **immaterial at fleet scale — proven by negative control**; matters more per-hour, still not decisive |

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
