# bekko-embedding-v1 (a8m / a25m) as code-search and remax_kb embedder

Handoff: [claude-workspace#185](https://github.com/oaustegard/claude-workspace/issues/185).
Run 2026-08-04. Box: **4 vCPU / 15 GB** (CCotw). Every throughput number below is
on that box; the claude.ai container (1 vCPU / 3 GB) is not comparable.

**Headline.** Part A reverses the 2026-07-04/05 verdict: a neural code-capable
encoder *does* beat identifier grep on file discovery, including on the
identifier-poor instance where grep scores zero — the decision gate passes.
Part B is a **regime choice, not a dominance**: bekko loses to jina v5 nano q4
at every byte budget, but bekko-a8m encodes a query **12.9x faster on 1 vCPU**
(11.3 ms vs 146.4 ms), which is the whole point of a 7.7M-*active*-parameter
model and which the iso-byte comparison alone does not see. Which one is the
right default depends on whether the deployment is quality-bound or
compute-bound.

---

## STEP 0 — egress

**Not a blocker in CCotw.**

```
curl -sL .../bekko-embedding-v1-a8m/resolve/main/tokenizer.json
  -> 302 to us.aws.cdn.hf.co -> 200, 34,363,442 bytes
```

Full file, not the ~103-byte allowlist refusal. `*.cdn.hf.co` egress works from
the CCotw container, so **no GitHub-release mirror was needed** and the
`jina-v5-nano-mirror` precedent did not have to be repeated. This is a
*per-environment* fact (`1e42faf1`) — it says nothing about the claude.ai
container, where the same host was refused on 2026-08-04.

## Artifact facts, independently confirmed

| handoff claim | measured | verdict |
|---|---|---|
| default ONNX int8s the token-embedding table, transformer stays fp32 | a8m `model.onnx` **124.1 MiB** vs `model_fp32` **404.3 MiB** (a25m: 190.1 / 470.3) | confirmed |
| cosine ≥ 0.9994 to torch | **0.99989 min / 0.99992 mean** (a8m), **0.99998** (a25m), vs each model's *own* fp32 export | confirmed, exceeded |
| every transformer-weight qint8 file is `_not_recommended` | a8m ships 4, a25m 3, all so labelled | confirmed |
| ONNX Runtime 5.5x slower than OpenVINO on x86 CPU | **ORT 21.8 ch/s vs OpenVINO 20.1 ch/s** | **did not reproduce** |

The vocab is 256,000 x 384 — a 98 M-parameter embedding table, ~393 MB in fp32.
That single tensor is essentially the whole 404 MiB, which is why int8-ing only
it buys 3.3x at ~1e-4 cosine cost. Exact mirror image of `eddb1106`;
independently corroborates `76526de1`.

**The OpenVINO non-reproduction matters.** The card's 5.5x is not free on a
small-core box: at 4 vCPU the runtimes are within 8%. Use ORT (fewer moving
parts) unless you have measured otherwise on your own hardware.

---

## Part A — code search

### Harness (and why it had to be rebuilt differently)

The prior 7-instance set **could not be reconstructed**. The memories record
only two instances by number (14338, 30817) and the harness itself was never
kept — this is its third rebuild. So instances were **re-mined**, and this time
the instance set is committed (`instances.json`) so a fourth rebuild is
unnecessary.

Mining had to route around a live constraint: `add_repo` refuses cross-owner
adds (`cross-tier adds are not supported in v1`; session pinned to `oaustegard`),
so `api.github.com/repos/scikit-learn/*` **403s**. What does work:

- `git clone` of a public out-of-scope repo — **works** (full history, 33,906 commits)
- `mcp__github__search_issues` / `search_pull_requests` — **work globally**, and return full issue/PR *bodies*
- `mcp__github__issue_read` on that repo — **403**

So gold file sets come from **git**, not the API: scikit-learn squash-merges, so
each PR is one commit on main whose subject ends `(#NNNNN)` and whose diff *is*
the PR diff. Issue bodies come from global search, date-windowed by bounding the
issue number between neighbouring PRs' commit dates. Changelog fragments are
stripped from gold (they are named after the PR, so any method knowing the PR
number scores them trivially).

**n = 6.** Strata are assigned by *measured* identifier yield rather than by
assumption — an instance is identifier-poor if the extractor recovers no
code-shaped token from title+body. #22186 ("Incorrect Poisson objective for
decision tree/random forest") yields **zero**: it describes the bug
derivationally and points at code by GitHub *line-number URL*, pasting no
identifier. That is a cleaner operational definition than a hand label.

Sanity check on comparability: the grep baseline lands at **r@5 0.667 / r@10
0.778**, against the prior run's 0.57 / 0.79 at n=7. Same ballpark, so the
re-mined set is of comparable difficulty.

### Results — n=6 first, then n=59

The original run was **n=6**. It has since been re-mined to **n=59** (same
procedure, 630 PRs harvested, 97 candidates with live gold, 59 with retrievable
issue bodies). **The n=6 headline does not replicate.**

| cell | rg r@5 | rg r@10 | bekko r@5 | bekko r@10 | RRF r@5 | RRF r@10 |
|---|---|---|---|---|---|---|
| ast / a8m | 0.596 | 0.682 | **0.595** | 0.667 | 0.662 | 0.728 |
| ast / a25m | 0.596 | 0.682 | 0.656 | 0.706 | 0.650 | 0.733 |
| flat / a8m | 0.596 | 0.682 | 0.658 | **0.744** | 0.672 | **0.762** |
| flat / a25m | 0.596 | 0.682 | 0.644 | 0.708 | 0.674 | 0.751 |

At n=6 this table read **rg 0.667 / dense 0.806 / RRF 0.889** for ast/a8m. At
n=59 dense/a8m lands at **0.595 against grep's 0.596 — a dead tie**, and *worse*
at r@10 (0.667 vs 0.682).

### Paired tests at n=59 — one finding survives, and it is not the headline

Sign test on per-instance recall differences, plus a paired bootstrap CI:

| comparison | Δ | 95% CI | w/l | p | verdict |
|---|---|---|---|---|---|
| dense ast/a8m beats rg (r@5) | **−0.001** | [−0.099, +0.097] | 12/14 | 0.845 | **noise** |
| dense ast/a25m beats rg (r@5) | +0.061 | [−0.041, +0.164] | 14/10 | 0.541 | noise |
| dense flat/a8m beats rg (r@5) | +0.062 | [−0.017, +0.150] | 11/10 | 1.000 | noise |
| RRF flat/a8m beats rg (r@5) | +0.076 | [+0.019, +0.145] | 10/4 | 0.180 | suggestive |
| RRF flat/a8m beats rg (r@10) | +0.080 | [+0.021, +0.150] | 10/3 | 0.092 | suggestive |
| chunking: flat beats ast (a8m, r@5) | +0.063 | [−0.009, +0.144] | 9/5 | 0.424 | noise |
| **encoder: a25m beats a8m (ast, r@5)** | **+0.061** | **[+0.022, +0.105]** | **13/1** | **0.0018** | **SUPPORTED** |

**No dense-vs-grep comparison is significant in any cell.** RRF is directionally
ahead everywhere and its bootstrap CIs exclude zero, but the sign test does not
clear 0.05 — recall ties dominate, which makes the sign test conservative and the
two tests disagree. Read RRF as *suggestive, not established*.

The one solid result is **a25m > a8m (13 wins to 1, p=0.0018)** — which
**reverses the n=6 conclusion** that "a25m does not earn its cost". At n=6 a25m
looked *worse* at r@10; at n=59 it is reliably better. The extra 9 layers do buy
something; whether they buy 3x the encode time is a separate call.

### The decision gate, re-evaluated

| cell | poor stratum (n=1) | rich stratum (n=58) | gate |
|---|---|---|---|
| ast / a8m | 0.667 vs rg 0.000 — clears | 0.594 vs rg 0.606 — **regresses** | **FAIL** |
| ast / a25m | 0.667 vs rg 0.000 — clears | 0.656 vs rg 0.606 — ok | pass |
| flat / a8m | 0.333 vs rg 0.000 — clears | 0.663 vs rg 0.606 — ok | pass |

**The gate now depends on which cell you pick**, which is itself the finding: at
n=6 it passed on the cell I happened to run first. And the identifier-poor
stratum is **still n=1 of 59** — 10x the sample bought no additional
identifier-poor instances, which independently corroborates the ~0.3% base rate
from `ecc4caad` and means the stratum the gate turns on remains untestable.

**Consequence for the 2026-07 verdict: it stands.** This experiment does *not*
overturn the retirement of the semantic tier. The n=6 reversal was noise; a
neural encoder ties identifier grep on well-specified issues, which is what the
original replication concluded about TF-IDF.

### Cost — both framings, because only one is flattering

| arm | tokens (n=59) | tokens (n=6, original) |
|---|---|---|
| `rg` full line output | 6,385,339 | 328,230 |
| **`rg -l` (filenames only)** | **315,470** | 13,336 |
| bekko dense, top-10 chunks | 200,071 | 25,568 |

At n=59 dense is **1.6x cheaper** than `rg -l`, where at n=6 it was 2x more
expensive — the larger sample includes more identifier-heavy issues whose grep
output is large. Against `rg`'s full line output dense is 32x cheaper.

Charging `rg` its full line output makes dense look **12.8x cheaper**. That is
not an honest baseline for a *file*-discovery metric: `rg -l` returns exactly
the information the metric scores, and it costs **13.3k — about half of dense's
25.6k**. So the true cost story is **dense ≈ 2x the tokens of a well-run grep,
for +0.14 r@5**, plus a one-time index (11,380 chunks, ~11 min on 4 vCPU for
a8m; 33 min for a25m). The prior runs' "~39k for grep" sits between these two.
Which number you quote is a methodology choice, so both are here.

### Chunking vs encoder — the handoff's prior is not supported

Varying exactly one axis at a time:

| axis | held fixed | Δ r@5 | Δ r@10 |
|---|---|---|---|
| chunking, ast → flat | a8m | **+0.028** | 0.000 |
| chunking, ast → flat | a25m | 0.000 | **+0.083** |
| encoder, a8m → a25m | ast | 0.000 | **−0.083** |
| encoder, a8m → a25m | flat | **−0.028** | 0.000 |

The handoff's prior was "chunk boundaries matter more than the encoder here."
**Stated explicitly, as asked: it does not hold** — and with the 2x2 complete the
statement is stronger than a null. Each axis **changes sign depending on the
level of the other**: flattening helps a8m at r@5 (+0.028) and does nothing to
a25m; a25m loses r@10 under ast chunking (−0.083) and gains it under flat. Every
effect is ±0.028 or ±0.083 — one instance-level step at n=6 — with no consistent
direction. Both axes are noise, and both are small next to the +0.139
dense-vs-grep gap that is the actual finding.

**Correction from the completed 2x2.** With only three cells this file claimed
AST chunking helps on the identifier-poor instance (r@5 0.667 ast vs 0.333 flat).
The fourth cell withdraws it: flat/a25m also scores 0.667 there, so 0.333 is a
flat/**a8m** quirk, not a chunking effect. n=1 strata generate exactly this kind
of false signal, which is why the thin-slice caveat below is load-bearing.

**a25m does not earn its cost either way.** 3x the encode time (33 min vs 11 min,
25.8 min vs ~11 for flat) to land at r@10 0.806 under ast and 0.889 under flat —
i.e. no better than a8m at best, worse at worst. The 4-layer model is the one to use.

---

## Part B — general remax_kb embedder

Self-retrieval over the published `muninn-subset.kb` (179 chunks, 11 posts).
Each chunk splits into a head (query) and body (indexed doc) with the head
text **removed from the doc**, so exact-substring matching cannot carry the
retrieval. Identical splits for every model. Second distribution = 179 sklearn
AST chunks, per `76526de1` (a single-domain smoke test hid an int8 collapse once).

### 1. The shipped artifact is safe

| model | dist | per-doc cosine vs own fp32 | Spearman | MB |
|---|---|---|---|---|
| a8m | blog | 0.99989 | 0.99982 | 124.1 (vs 404.3) |
| a8m | code | 0.99990 | 0.99985 | " |
| a25m | blog | 0.99998 | 0.99998 | 190.1 (vs 470.3) |
| a25m | code | 0.99998 | 0.99998 | " |

R@k differences between default and own fp32 are ≤ 0.006 everywhere. **No int8
collapse on either distribution** — the second-distribution check passes, which
is the check that mattered last time.

### 2. Head-to-head vs the incumbent — bekko loses

R@10, iso-byte (fp32 vector bytes):

| bytes | dim | jina v5 nano q4 | bekko a25m | bekko a8m |
|---|---|---|---|---|
| 256 | 64 | 0.492 / **0.950** | **0.520** / 0.877 | 0.497 / 0.782 |
| 512 | 128 | **0.598** / **0.972** | 0.564 / 0.911 | 0.520 / 0.866 |
| 1024 | 256 | **0.609** / **0.983** | 0.587 / 0.944 | 0.559 / 0.866 |
| 1536 | 384 | **0.620** / **0.983** | 0.598 / 0.950 | 0.575 / 0.888 |
| 3072 | 768 | **0.631** / 0.978 | — | — |

*(cells are `blog / code`)*

Jina wins **11 of 12** contested cells; the single bekko win is blog at 64-dim.
On the code distribution — where a "code-capable" encoder was supposed to have
its edge — jina wins by **0.03–0.10 R@10 at every budget**, and jina at 64 dims
(0.950) beats bekko-a25m at 384 dims (0.950 tie) and beats a8m at any width.

And the size argument evaporates: **official jina q4 is 131.6 MB, bekko-a8m is
124.1 MB.** A 7 MB difference is not an architecture decision.

**On quality per byte, the incumbent wins outright.** Better at every budget, on
both distributions, at the same artifact size. That was the original verdict here
— and on its own it is **an incomplete answer**, because it prices bytes and
ignores compute. See "Compute" below, which changes the recommendation.

### 3. The strategic note (regime A) — already true, without bekko

The handoff's point was that a ~130 MB encoder becomes shippable alongside the
`.kb`, collapsing regime A from `7cecfd94` (the "light embed endpoint with a
shared secret" existed only because the encoder was too big to distribute).

**On size, that is already true of the incumbent.** The official jina q4 that
remax_kb *already defaults to* is 131.6 MB, so the encoder-size premise was
satisfied before this benchmark ran and the redistribution question reduces to
whether the *corpus* is redistributable, as the handoff framed it.

**But size was not the only thing standing in the way.** A shipped encoder has
to be pleasant to *run* on whatever the reader has, and on 1 vCPU jina answers a
query in **146 ms** against bekko-a8m's **11.3 ms**. So bekko does add something
the incumbent does not: it makes the ship-the-encoder regime cheap on constrained
hardware, not merely possible. The earlier claim in this file that regime A
"never needed bekko" was true of the size premise and too strong about the
regime.

### 4. Compute — the axis the iso-byte table cannot see

bekko-a8m is **4 layers x 384 hidden x 1152 FFN**; jina v5 nano is **12 x 768 x
3072**. That is ~12x the per-token FLOPs, and it is the entire design point of a
7.7M-*active*-parameter encoder (the 98M-param embedding table is a gather, not
compute). Measured, median of 5, same texts, each model's own tokenizer and
required prefixes:

| threads | model | query latency (batch=1) | docs/s | tokens/s |
|---|---|---|---|---|
| **1** | **bekko-a8m** | **11.3 ms** | **53.6** | **6,375** |
| 1 | bekko-a25m | 35.0 ms | 17.7 | 2,101 |
| 1 | jina v5 nano q4 | **146.4 ms** | 5.3 | 569 |
| 4 | bekko-a8m | 6.7 ms | 149.3 | 17,766 |
| 4 | bekko-a25m | 16.7 ms | 40.4 | 4,803 |
| 4 | jina v5 nano q4 | 46.5 ms | 16.0 | 1,726 |

**bekko-a8m is 12.9x faster per query on 1 vCPU and 11.2x higher throughput.**
The measured ratio matches the FLOPs ratio (~12x), so this is **architectural,
not a q4-dequantization artifact** — which also means it will not be tuned away.

This matters because remax_kb's stated pitch is querying a `.kb` "from a vanilla
container with onnxruntime + tokenizers + numpy", and the claude.ai container is
**1 vCPU**. On that box the query path is 146 ms vs 11 ms — the difference
between noticeable and imperceptible — and a corpus build runs 11x longer.

### 5. The actual recommendation: an iso-quality ladder

Cheapest encoder reaching a given quality, 1-vCPU query latency:

| blog R@10 target | cheapest model | query | code R@10 |
|---|---|---|---|
| ≤ 0.575 | **bekko-a8m** (d=384) | **11.3 ms** | 0.888 |
| 0.58 – 0.598 | bekko-a25m (d=384) | 35.0 ms | 0.950 |
| > 0.60 | **jina only** (d=768) | 146.4 ms | 0.983 |

So the honest verdict is **regime-dependent**, and the original "do not swap"
was under-specified:

- **Quality-bound, or compute amortized** (cores available, corpus built once,
  query volume low): **keep jina**. It is the only model that reaches R@10 0.60+
  on blog and 0.98 on code, and the quality gap is largest exactly on the code
  distribution (0.983 vs 0.888 — the one place bekko was *advertised* to be
  strong, and it loses by 0.095).
- **Compute-bound** (1 vCPU reader, latency-sensitive query path, or a large
  corpus to encode): **bekko-a8m** buys 12.9x for ~0.045 blog / ~0.095 code
  R@10. On a 1-vCPU container that is a defensible trade, and for a big index
  build it is the difference between minutes and hours (measured: 6.5 s vs
  60.2 s to pack the same 179 chunks). **Caveat measured in §6: as remax_kb
  ships today only 2.3x of that 12.9x reaches the reader**, because a constant
  per-query binarizer cost swamps it. The constant is removable.
- **bekko-a25m is the hedge**: 4.2x faster than jina, within 0.022 blog R@10 of
  it, and it recovers most of the code gap (0.950 vs 0.983).

What has *not* changed: at a fixed byte budget the compression story is
unaffected, and Part A is untouched (it never involved jina).

### 6. The real query path — where the 12.9x actually goes

Encoder-in-isolation is not what a reader pays. Driving
`remax_kb.read.KB.search` on two `.kb` files packed here with **identical**
chunks and binarizer params (dim=256, k=8, seed=0), so the only difference is
the embedder, 1 thread:

| model | search | = encode | + stacked-SimHash | + hamming_scan | + top_k |
|---|---|---|---|---|---|
| bekko-a8m | 65.5 ms | **5.4** | **57.3** | 0.03 | 0.01 |
| bekko-a25m | 68.5 ms | 18.6 | 46.6 | 0.02 | 0.01 |
| jina v5 nano q4 | 153.4 ms | 90.7 | 61.7 | 0.02 | 0.01 |

**End-to-end the advantage collapses from 12.9x to 2.3x** — and not because the
encoder measurement was wrong. A ~50-60 ms *constant* sits in the middle of every
query and swamps the difference. For bekko-a8m it is **87% of the whole query**.

**It is `_stacked_simhash_encode`.** That method constructs a
`StackedSignBitQuantizer(d, k, seed)` **per call**, and that constructor builds
k=8 Haar rotations by QR of a 256x256 Gaussian. All three parameters come from
the manifest and cannot change for an opened `.kb`, so every query rebuilds
byte-identical rotations from scratch.

Caching it once per opened `KB` — one line — was verified to produce **identical
codes and an identical hit list**, so this is a speedup, not a behaviour change:

| query | model | shipped | cached | encode |
|---|---|---|---|---|
| short (9 tok) | bekko-a8m | 59.4 ms | **5.5 ms** | 6.6 |
| typical (14 tok) | bekko-a8m | 54.5 ms | **6.0 ms** | 6.1 |
| long (52 tok) | bekko-a8m | 64.2 ms | **15.3 ms** | 14.7 |
| short | jina q4 | 126.7 ms | 80.2 ms | 79.2 |
| typical | jina q4 | 138.4 ms | 91.2 ms | 88.5 |
| long | jina q4 | 231.4 ms | 176.5 ms | 177.2 |

| bekko-a8m over jina | shipped | cached |
|---|---|---|
| short | 2.1x | **14.7x** |
| typical | 2.5x | **15.1x** |
| long | 3.6x | **11.6x** |

So the honest answer to "does the 12.9x survive end-to-end" is **not as
remax_kb ships today, but the thing eating it is a fixable constant, not the
architecture**. Remove it and the query path becomes encode-dominated and the
encoder advantage is fully realized (11.6–15.1x). The fix helps the incumbent
too — jina drops 138 → 91 ms — but it helps a fast encoder far more, because a
fixed tax hurts most whoever else is cheap.

**The scan is not the problem and will not become one soon.** Hamming scan +
top_k over dim=256/k=8 codes:

| n chunks | scan + top_k | index |
|---|---|---|
| 179 | 0.046 ms | 0.04 MB |
| 1,238 (`muninn.kb`) | 0.14 ms | 0.3 MB |
| 10,000 | 0.95 ms | 2.4 MB |
| 100,000 | 13.2 ms | 24 MB |

At the "few hundred to a few thousand docs" remax_kb targets, retrieval is
free and the query is *entirely* encode plus that constant. Only around
n≈100k does the scan reach parity with bekko-a8m's encode.

**Recommended, and not done here:** cache the quantizer on `KB` (and the
equivalent in `read_v2`). It is a one-line change worth ~9x on every query for a
small encoder and ~1.5x for the incumbent. I have not opened a remax_kb PR — the
handoff held changes to that repo, and this is a different change than the one it
held, so it should be an explicit call rather than something I slip in.

**Build cost, same corpus (179 chunks, 1 thread):** bekko-a8m 6.5 s,
bekko-a25m 21.3 s, jina q4 60.2 s — 9.3x, consistent with the throughput ratio.

### 7. The RHT option — right lever for portability, wrong lever for latency

§6 blamed a per-query Haar QR. The obvious alternative is a structured
projection, and remax_kb v2 already **defaults** to one: `projection="srht"`,
with `"haar"` kept for back-compat. Measured at v1's actual geometry
(dim=256, k=8), median of 5, cost to construct all k stacks:

| projection | build (d=256) | vs haar |
|---|---|---|
| `rademacher` (v2 option) | **14.2 ms** | **3.7x faster** |
| remax `rht_rotation`, rounds=2 | 31.3 ms | 1.7x faster |
| remax `rht_rotation`, rounds=3 | 43.0 ms | 1.2x faster |
| `haar` (v1 default) | 53.1 ms | — |
| `remax_kb.projection.srht_matrix`, rounds=3 (**v2 default**) | **75.7 ms** | **1.4x slower** |

**remax's documented 1.5–1.8x speedup is confirmed, and it is a different
function than the one remax_kb defaults to.** `remax.rotation.rht_rotation` at
its floored rounds=2 gives 1.7x / 2.1x / 1.6x at d=256 / 768 / 1024 — squarely
in the documented band. `remax_kb.projection.srht_matrix` is a *separate*
implementation and is 1.4–3.0x **slower** than Haar at every dimension measured:

| dim | haar | remax rht (r=2) | kb srht (r=3) | kb rademacher |
|---|---|---|---|---|
| 256 | 53.1 ms | **31.3** | 75.7 | **14.2** |
| 768 | 599.1 ms | **290.9** | 1777.3 | 314.9 |
| 1024 | 1043.4 ms | **633.0** | 2116.4 | **560.5** |

**That is not a defect.** v2's own comment says why it chose srht: it is
seed-only (ships no `binarizer/rotations.*` sidecar) and **bit-for-bit
reproducible by the JavaScript reader**, where Haar's PCG64 + Ziggurat + LAPACK
QR is NumPy-only and a mismatched projection flips ~50% of code bits. srht buys
cross-reader portability and pays construction time for it. Worth knowing when
choosing, not worth "fixing".

**Retrieval is indistinguishable across all three**, so the choice really is
free on quality here (self-retrieval, dim=256, k=8):

| projection | a8m R@1 / R@10 | a25m R@1 / R@10 |
|---|---|---|
| haar | 0.207 / 0.564 | 0.240 / 0.581 |
| rademacher | 0.212 / **0.587** | 0.212 / **0.587** |
| srht | 0.201 / 0.570 | 0.251 / 0.575 |
| *(fp32 reference)* | *— / 0.559* | *— / 0.587* |

All within ±0.023 R@10 and straddling the fp32 baseline — at this dim and corpus
the ~85%-of-the-Rademacher→Haar-gap claim in v2's comment is not resolvable.

**But none of this fixes §6, and that is the finding.** Every row above is a
*per-query construction* cost of 14–76 ms against bekko-a8m's **~6 ms encode**.
Even the cheapest projection is more than twice the encode it is supposed to be
serving. Switching Haar→rademacher would take the query from ~54 ms to ~20 ms;
**caching takes it to 6.0 ms**, and caching works for every projection.

Ordered, then:

1. **Cache the projection per opened index.** Verified identical codes and hits
   (§6). Removes the tax entirely, and is orthogonal to which projection you pick.
2. **If per-query construction is genuinely unavoidable**, `rademacher` is 3.7x
   cheaper than Haar at d=256 and quality-neutral here.
3. **Pick `srht` when a non-NumPy reader has to reproduce the planes** — that is
   what it is for. Then cache it, because it is the most expensive to build.

The RHT question turned out to be orthogonal to the latency question: projection
choice is a portability decision, and the latency is a caching decision.

---

## Statistical power — read this before quoting any number above

**Corpora.** Two, and they are very different sizes:

| used for | corpus | size | encode wall-clock (4 vCPU) |
|---|---|---|---|
| Part A (code search) | scikit-learn `sklearn/` | 674 files → **11,380** AST / **9,363** flat chunks | **77.8 min** across 4 cells (a8m 10.2 + 8.8, a25m 33.0 + 25.8) |
| Part B, byte budgets, projections, ceiling | `muninn-subset.kb` | **179 chunks / 11 blog posts** | seconds |
| second distribution | sklearn AST slice | 179 chunks | seconds |

So the ~78 minutes of encoding bought the **code-search** benchmark (~41,500
chunk encodes). **Every embedding-quality conclusion in this file rides on 179
chunks from 11 blog posts**, encoded in seconds. The compute went to the arm that
needed it least.

### Was 78 minutes reasonable? Mostly no — audited

**The corpus work is real, so "seconds" was never available.** The AST corpus is
11,380 chunks at a true mean of **328 effective tokens** (median 322, p90 927 —
*not* 512; an earlier read of this said 512 because `encode_batch` over the whole
corpus pads to the longest). At ~17.4 MFLOP/token for a 4-layer/384-hidden/1152-FFN
model that is **64.9 TFLOP**. This box (AVX-512 Xeon, 4 vCPU) sustains
**424 GFLOP/s** on 2048² sgemm, so the floor at *100% of peak* is **2.6 min** for
one cell. Minutes, not seconds.

**But three things were wrong, in increasing order of cost.**

1. **Batching left 1.25x on the table.** Batches of 8 in corpus order pad every
   chunk to the batch's longest, which the length distribution (median 322, p90
   927) makes expensive: **1.47x** padded-token waste. Length-sorting before
   batching fixes it — measured **1.25x** end-to-end (25.1 vs 20.1 chunks/s), and
   the output is **bit-identical** (max |Δ| = 0.0). Should have been in the
   encoder from the start.

2. **Achieved throughput was 25% of peak** — 106 GFLOP/s against 424. Most of
   that gap is structural: a 384×1152 GEMM is far from the shape where BLAS
   reaches peak, and layernorm/activation traffic and ORT dispatch are not
   FLOPs. It is low, not pathological, and closing it would mean sequence
   packing rather than a config change.

3. **The real waste was scope, and it dwarfs the other two.**

   | cell | wall | what it bought |
   |---|---|---|
   | ast/a8m | 10.2 min | **the result** |
   | ast/a25m | 33.0 min | within noise — and r@10 *worse* than a8m |
   | flat/a8m | 8.8 min | chunking axis → noise |
   | flat/a25m | 25.8 min | chunking × encoder → noise |
   | **total** | **77.8 min** | |

   Both extra axes came back inside noise at n=6, so **67 of the 78 minutes
   bought conclusions that the sample size could never have supported**. The
   minimum defensible run — ast/a8m, length-sorted — is **8.2 min, 10% of what
   was spent**.

The generalizable version: I sized the *corpus* for rigour (all 674 files, so
retrieval faces real distractors — that part was right) and then replicated it
across a 2×2 of axes whose effects were far below the resolution of a 6-instance
benchmark. Power analysis belongs *before* the encode budget, not after it.



**At n=179 one query is 0.56 pp of R@10**, and the differences reported above are
2–8 queries wide. The arms are evaluated on the same queries, so the right test
is paired — exact McNemar on discordant pairs, plus a paired bootstrap CI:

| claim | Δ R@10 | 95% CI | disc. | p | verdict |
|---|---|---|---|---|---|
| remex 2-bit @96 B beats **uncompressed** fp32 @1536 B | +0.011 | [−0.011, +0.034] | 3/1 | 0.625 | **noise** |
| remex 1-bit @48 B beats vendor floor d=64 @256 B | +0.045 | [−0.011, +0.101] | 17/9 | 0.169 | **noise** |
| remex 1-bit @48 B equals Matryoshka d=128 @512 B | +0.000 | [−0.050, +0.050] | 10/10 | 1.000 | noise |
| remex 2-bit beats remax k=2 (same 96 B) | +0.039 | [−0.011, +0.089] | 14/7 | 0.189 | **noise** |
| remex 1-bit beats remax k=1 (same 48 B) | +0.011 | [−0.045, +0.067] | 15/13 | 0.851 | **noise** |
| jina q4 beats bekko-a25m (blog, full width) | +0.034 | [−0.022, +0.089] | 17/11 | 0.345 | **noise** |
| Matryoshka d=384 beats d=64 | +0.078 | [+0.022, +0.134] | 20/6 | **0.009** | **SUPPORTED** |
| Matryoshka d=384 beats d=256 | +0.011 | [−0.017, +0.045] | 5/3 | 0.727 | noise |

**One of eight survives.** Truncation to d=64 does measurably cost recall. Every
other headline in this file — including "remex 2-bit beats the uncompressed
vector", which I led with — is **not resolvable at n=179**.

**The exception, and it is a large one.** On the *code* distribution at R@1, jina
beats bekko-a25m **0.888 vs 0.721, +0.168, 31 discordant wins to 1, p < 1e-5**.
That single comparison is overwhelming, and it is the real evidence behind Part
B's "do not swap" — not the twelve iso-byte cells, which are individually noise
and mutually correlated (same queries, nested dims), so "wins 11 of 12" is not
twelve independent trials.

### What this does and does not overturn

- **Part B's verdict stands, on narrower grounds.** jina's advantage is
  established on code-distribution R@1 at p<1e-5. The blog-distribution and
  R@10 margins are noise.
- **The trimming-vs-quantization direction is consistent but not established
  *here*.** Every budget and both encoders point the same way, and the vendor's
  own HAKARI table — a far larger evaluation — puts binary@384 at −12.93% against
  d=64's −17.51%. My corpus is **consistent with** that and too small to
  demonstrate it independently. Cite the vendor's number for the claim; cite mine
  only as corroboration.
- **The projection comparison (§7) was already reported as neutral**, and the
  power analysis confirms that a ≤0.022 spread at n=179 could not have shown
  otherwise. That conclusion was safe.
- **Part A is worse off, and was flagged as such throughout**: n=6 instances,
  n=1 on the deciding stratum.

To resolve a true 0.01 R@10 gap at 80% power needs roughly **n > 2000**, an order
of magnitude more than this corpus. The right fix is not more bits of analysis on
179 chunks — it is `muninn.kb` (1,238) or a standard IR set (BEIR/NFCorpus,
~3,600 docs with real qrels rather than self-retrieval), and it costs far less
compute than the 78 minutes already spent on the code corpus.

---

## Head-to-head: Matryoshka trimming vs quantization at equal bytes

**Quantization wins — on the vendor's own card, and directionally here, though n=179 cannot establish it independently (see the power section above).** The model card
publishes this comparison on their HAKARI benchmark:

| Setting | Dim | Encoding | Rescore | HAKARI | Δ vs 384 float |
|---|---|---|---|---|---|
| Full quality | 384 | float | No | 0.545 | — |
| Smaller index | 256 | float | No | 0.536 | −1.76% |
| Compact index | 128 | float | No | 0.507 | −7.05% |
| **Very compact** | **64** | float | No | 0.450 | **−17.51%** |
| INT8 search | 384 | int8 | No | 0.515 | −5.48% |
| **INT8 + rescore** | 384 | int8 | **Yes** | **0.545** | **−0.04%** |
| **Binary search** | **384** | **binary** | No | 0.475 | **−12.93%** |
| **Binary + rescore** | 384 | binary | **Yes** | **0.543** | **−0.44%** |

Binary at 384-d (48 B) loses **12.93%**; truncating to 64-d (256 B) loses
**17.51%**. Their own numbers put 1-bit-at-full-width ahead of the lowest
supported truncation tier, at 5.3x fewer bytes. And with a rescore step, binary
comes back to −0.44% of full float.

### Measured here, at the vendor's supported tiers only

The card is explicit — **"Supported truncate dimensions: 256, 128, 64"** — so the
comparison is run on those, plus the full 384. (An earlier version of this file
extended the ladder down to d=12 to make the byte ranges overlap and quoted the
resulting 2.1x as the headline. That was a **strawman**: nobody proposes d=12,
and the honest comparison does not need it.)

bekko-a25m, payload bytes/vector:

| configuration | bytes | R@10 | R@50 |
|---|---|---|---|
| Matryoshka fp32 d=384 (uncompressed) | 1536 | 0.598 | 0.855 |
| Matryoshka fp32 d=256 | 1024 | 0.587 | 0.838 |
| Matryoshka fp32 d=128 | 512 | 0.564 | 0.844 |
| **Matryoshka fp32 d=64** (vendor floor) | **256** | **0.520** | 0.816 |
| **remex 1-bit @384** | **48** | **0.564** | 0.821 |
| **remex 2-bit @384** | **96** | **0.609** | 0.855 |
| remex 4-bit @384 | 192 | 0.598 | 0.855 |
| remax k=1 @384 | 48 | 0.553 | 0.838 |
| remax k=2 @384 | 96 | 0.570 | 0.849 |

**Against the vendor's own floor (d=64, 256 B):**

| arm | bytes | R@10 | vs d=64 |
|---|---|---|---|
| remex 1-bit @384 | **48** | 0.564 | **5.3x smaller, +0.045** |
| remax k=1 @384 | **48** | 0.553 | 5.3x smaller, +0.034 |
| remex 2-bit @384 | 96 | 0.609 | 2.7x smaller, **+0.089** |

And **remex 2-bit at 96 B (0.609) beats the uncompressed 384-d vector at 1536 B
(0.598)** — 16x smaller, not worse. **remex 1-bit at 48 B exactly equals
Matryoshka d=128 at 512 B.** Matryoshka trimming does not win at any budget.

**Why.** Truncating to 64 keeps 1/6 of the coordinates and discards the rest;
1-bit keeps the *sign of all 384*. Matryoshka concentrating variance in the
leading dims is what makes the tail cheap to drop — but cheap is not free, and a
sign bit on every coordinate carries more than 32 bits on a sixth of them.

**remex beats remax at every budget** (0.609 vs 0.570 at 96 B) — consistent with
remax's framing as a rank-correct ladder rather than a byte-optimal codec.

## The R@50 ceiling: what it is, and what recovers it

R@50 saturates near 0.855 for every arm ≥96 B, uncompressed included — 26/179
queries (14.5%) never surface gold in the top 50 at any precision. Three things
matter about that number, and the first is a caveat on my own harness.

**1. A large part of it is the harness, not the encoder.** Inspecting the 26:
the head/body split cuts mid-topic, so query and gold are adjacent-but-unrelated
spans. Query "The Boot Sequence — every conversation begins with…" against a gold
body that is raw mermaid markup (`ceDiagram participant PI as…`); query about
memories-vs-skill-files against a gold body that opens "Layer 4: Context Window".
Several of these are **unanswerable by construction** and should not be counted
against any retrieval method.

**2. Half of the rest is encoder-specific, not a task floor.** jina q4 fails only
**20**, and only **13 fail for both**. So the genuinely shared floor is
13/179 = **7.3%**, not 14.5% — and Part B's better encoder recovers half of
bekko's failures on its own.

**3. Rescoring cannot touch it.** The vendor's rescore step (binary −12.93% →
−0.44%) reranks a candidate set with full-precision vectors. It fixes
*quantization* loss because the gold is already in the set. These 26 are a
**recall** failure — gold never enters the set — so rescoring is the wrong tool
by construction. Worth stating, because the vendor's table makes rescoring look
like a general remedy.

### What actually recovers them — measured

| method | R@10 | R@50 | recovers of the 26 |
|---|---|---|---|
| dense only (bekko-a25m) | 0.598 | 0.855 | 0/26 |
| **BM25 only** | 0.520 | 0.793 | **14/26** |
| **dense + BM25, RRF** | **0.615** | **0.866** | 8/26 |
| query expansion (RM3-style) | 0.598 | 0.838 | 3/26 |

**Lexical fusion, not query expansion.** BM25 alone is *worse overall* (R@10
0.520) yet recovers **14 of the 26 dense misses** — the same disjoint-failure
structure Part A found between `rg` and dense. RRF fusion gives the best overall
numbers (R@10 0.615, R@50 0.866), recovering 8 while keeping the easy cases;
pure BM25 recovers more hard cases but loses more easy ones, so the choice
depends on whether you want peak recall or peak top-10.

**Query expansion is the weakest option measured** — 3/26 recovered, and R@50
*drops* 0.855 → 0.838. That reproduces this repo's existing negative result
(`muninn-rm3`: RM3 "does not help on a small corpus", R@10 1.000 → 0.900).
Pseudo-relevance feedback needs a relevant top-k to feed on; when the gold is
absent from the candidates, expansion just amplifies the wrong neighbourhood.

**Practical upshot:** remax_kb v2 **already ships BM25 + RRF fusion**, so the
remedy is a configuration choice, not new work — and it is the same conclusion
Part A reached from the opposite direction.


## Byte-budget composition — quantization dominates truncation

Two orthogonal axes (`2eba5b5b`): Matryoshka cuts *coordinates*, remex cuts
*bits per coordinate*, remax stacks *sign-bit signatures*. Scored as cosine —
reconstructions renormalized by `‖x̂‖`, per METHODS.md, because ranking by bare
inner product rewards codecs whose reconstruction norm is constant by
construction.

Pareto frontier, bekko-a25m, blog distribution:

| bytes/vec | best spend | R@10 |
|---|---|---|
| 12 | remex d=64 @ 1-bit | 0.430 |
| 20 | remex d=64 @ 2-bit | 0.559 |
| 52 | remex d=384 @ 1-bit | 0.564 |
| 68 | remex d=256 @ 2-bit | 0.587 |
| **100** | **remex d=384 @ 2-bit** | **0.609** |
| 512 | fp32 d=128 (truncation) | 0.564 |
| 1536 | fp32 d=384 (full) | 0.598 |

**On payload bytes the answer is yes** — remex d=384 @ 2-bit scores **0.609**
against full fp32's **0.598** — but see the correction immediately below: the
payload figures here undercount, and payload is the *wrong* meter at small n.

### Correction, and then a retraction of most of it

Two rounds of scrutiny here, and the second overturned the first. Recording both,
because the retracted version is the kind of confident over-correction that is
worth being able to find later.

**What survives.** The payload was hand-computed as `dim*bits/8`, which drops the
**float32 norms remex stores separately**. remex's own
`CompressedVectors.nbytes` gives **52 B/vec** at d=384 @ 1-bit (not 48) and
**100 B** at 2-bit (not 96) — negligible at d=384, **+50% at d=64 @ 1-bit**
(12 B, not 8). All tables use remex's own accounting. That correction stands.

**What does not.** I then charged each codec for a **materialized dense d×d
rotation** (590 KB at d=384) plus "the codebook", concluded that the codec
advantage *inverts* below n≈411, and wrote it up. **That was wrong**, for two
reasons.

**1. remex's codebook is not large — it is 28 bytes.** Lloyd-Max here is a
*scalar* quantizer: `2^bits - 1` boundaries and `2^bits` centroids for a
1-D N(0, 1/d) coordinate distribution, shared by every dimension.

| bits | boundaries | centroids | total |
|---|---|---|---|
| 1 | 1 | 2 | **12 B** |
| 2 | 3 | 4 | **28 B** |
| 4 | 15 | 16 | 124 B |
| 8 | 255 | 256 | 2,044 B |

It is **21,065x smaller than the rotation** at 2 bits, and it is *analytic* —
determined by `(d, bits)` with no reference to data, so a reader recomputes it
rather than receiving it. It rounds to 0.2 B/vec even at n=179.

The "large shared codebook" framing was imported from `remex-vs-higgs-ablation`,
where it referred to a **vector-quantization grid** with `2^(m·bits)` codepoints
— genuinely large, and a different arm of a different experiment. Transferring
that to remex's scalar codebook was a category error.

**2. The rotation is the only large object, and nobody ships it dense — which is
exactly what the RHT is for.** A randomized Hadamard transform is a ±1 diagonal
plus an FWHT: in **operator form** its entire state is a sign vector,
`rounds × d` **bits**. The `d²` floats appear only because both
`remax.rotation.rht_rotation` and `remax_kb.projection.srht_matrix`
*materialize* the transform to keep the apply path a single BLAS matmul —
a speed decision, not a storage requirement.

What a reader actually receives, d=384:

| side structure | bytes | B/vec @ n=179 |
|---|---|---|
| seed only — **remax_kb v1, and v2 with `srht`/`rademacher`** | 4 | **0.0** |
| RHT in operator form (sign vector) | 144 | 0.8 |
| Lloyd-Max codebook (2-bit) | 28 | 0.2 |
| *dense f32 rotation — what I wrongly charged* | *589,824* | *3,295* |
| v2 `haar` + int8 sidecar (non-NumPy readers only) | 1,179,648 | 6,590 |

Every remax_kb configuration except the last is **seed-derived and ships
nothing**; v2's `srht` was chosen precisely so it "ships nothing at all". So the
590 KB I charged is a cost no deployment here pays, and the inversion it produced
was an artifact of that charge.

**Corrected result: with realistic side data the frontier is unchanged.**

| arm | B/vec (seed-only) | R@10 | baseline | B/vec | R@10 | winner |
|---|---|---|---|---|---|---|
| remex d=384 @ 2-bit | **100.0** | 0.609 | fp32 d=384 | 1,536 | 0.598 | **codec** |
| remex d=384 @ 1-bit | **52.0** | 0.564 | fp32 d=128 | 512 | 0.564 | **codec** |

Even charging the RHT operator form it moves to 100.8 / 52.8 B — the conclusion
does not depend on the choice. **There is no inversion at n=179, and no
break-even to clear.**

The one configuration where side data genuinely dominates is v2 with `haar` +
the int8 rotation sidecar, which exists so a JavaScript reader can reproduce the
planes. There, 1.2 MB of rotations against 179 chunks is a real 6,590 B/vec —
and it is an argument for `srht`, i.e. for the RHT, on exactly the grounds the
RHT was adopted.

### Corrected rule of thumb

**Quantize before you truncate — but the two compose, and below ~48 B you do
both.** Against fp32 truncation at any comparable budget, quantizing at full
width wins. But the frontier is not "keep all 384 dims" everywhere, and an
earlier version of this line said so incorrectly. The actual remex frontier:

| budget | winner | R@10 |
|---|---|---|
| 12 B | d=64 @ 1-bit (**truncated**) | 0.430 |
| 20 B | d=64 @ 2-bit (**truncated**) | 0.559 |
| 52 B | d=384 @ 1-bit (full width) | 0.564 |
| 68 B | d=256 @ 2-bit (**truncated**) | 0.587 |
| 100 B | d=384 @ 2-bit (full width) | 0.609 |

So it is an **ordering, not a replacement**: at the smallest budgets the winner
is truncated *and* quantized. The published post
([below-the-fold](https://muninn.austegard.com/blog/compression-result-below-the-fold.html))
states this correctly — "we truncate again at the smallest budgets… you can do
both at once, which makes this an ordering rather than a replacement" — and this
file's earlier absolute phrasing was the sloppier of the two.

The only caveat that survives is the one this experiment kept re-deriving from
different directions: the rotation is **free in bytes because it is paid for in
time**. remax_kb v1 regenerates it per query, which is §6's 53 ms. Cache it and
you pay neither.

---

## Byte-budget composition — quantization dominates truncation

Two orthogonal axes (`2eba5b5b`): Matryoshka cuts *coordinates*, remex cuts
*bits per coordinate*, remax stacks *sign-bit signatures*. Scored as cosine —
reconstructions renormalized by `‖x̂‖`, per METHODS.md, because ranking by bare
inner product rewards codecs whose reconstruction norm is constant by
construction.

Pareto frontier, bekko-a25m, blog distribution:

| bytes/vec | best spend | R@10 |
|---|---|---|
| 12 | remex d=64 @ 1-bit | 0.430 |
| 20 | remex d=64 @ 2-bit | 0.559 |
| 52 | remex d=384 @ 1-bit | 0.564 |
| 68 | remex d=256 @ 2-bit | 0.587 |
| **100** | **remex d=384 @ 2-bit** | **0.609** |
| 512 | fp32 d=128 (truncation) | 0.564 |
| 1536 | fp32 d=384 (full) | 0.598 |

**On payload bytes the answer is yes** — remex d=384 @ 2-bit scores **0.609**
against full fp32's **0.598** — but see the correction immediately below: the
payload figures here undercount, and payload is the *wrong* meter at small n.

### Correction, and then a retraction of most of it

Two rounds of scrutiny here, and the second overturned the first. Recording both,
because the retracted version is the kind of confident over-correction that is
worth being able to find later.

**What survives.** The payload was hand-computed as `dim*bits/8`, which drops the
**float32 norms remex stores separately**. remex's own
`CompressedVectors.nbytes` gives **52 B/vec** at d=384 @ 1-bit (not 48) and
**100 B** at 2-bit (not 96) — negligible at d=384, **+50% at d=64 @ 1-bit**
(12 B, not 8). All tables use remex's own accounting. That correction stands.

**What does not.** I then charged each codec for a **materialized dense d×d
rotation** (590 KB at d=384) plus "the codebook", concluded that the codec
advantage *inverts* below n≈411, and wrote it up. **That was wrong**, for two
reasons.

**1. remex's codebook is not large — it is 28 bytes.** Lloyd-Max here is a
*scalar* quantizer: `2^bits - 1` boundaries and `2^bits` centroids for a
1-D N(0, 1/d) coordinate distribution, shared by every dimension.

| bits | boundaries | centroids | total |
|---|---|---|---|
| 1 | 1 | 2 | **12 B** |
| 2 | 3 | 4 | **28 B** |
| 4 | 15 | 16 | 124 B |
| 8 | 255 | 256 | 2,044 B |

It is **21,065x smaller than the rotation** at 2 bits, and it is *analytic* —
determined by `(d, bits)` with no reference to data, so a reader recomputes it
rather than receiving it. It rounds to 0.2 B/vec even at n=179.

The "large shared codebook" framing was imported from `remex-vs-higgs-ablation`,
where it referred to a **vector-quantization grid** with `2^(m·bits)` codepoints
— genuinely large, and a different arm of a different experiment. Transferring
that to remex's scalar codebook was a category error.

**2. The rotation is the only large object, and nobody ships it dense — which is
exactly what the RHT is for.** A randomized Hadamard transform is a ±1 diagonal
plus an FWHT: in **operator form** its entire state is a sign vector,
`rounds × d` **bits**. The `d²` floats appear only because both
`remax.rotation.rht_rotation` and `remax_kb.projection.srht_matrix`
*materialize* the transform to keep the apply path a single BLAS matmul —
a speed decision, not a storage requirement.

What a reader actually receives, d=384:

| side structure | bytes | B/vec @ n=179 |
|---|---|---|
| seed only — **remax_kb v1, and v2 with `srht`/`rademacher`** | 4 | **0.0** |
| RHT in operator form (sign vector) | 144 | 0.8 |
| Lloyd-Max codebook (2-bit) | 28 | 0.2 |
| *dense f32 rotation — what I wrongly charged* | *589,824* | *3,295* |
| v2 `haar` + int8 sidecar (non-NumPy readers only) | 1,179,648 | 6,590 |

Every remax_kb configuration except the last is **seed-derived and ships
nothing**; v2's `srht` was chosen precisely so it "ships nothing at all". So the
590 KB I charged is a cost no deployment here pays, and the inversion it produced
was an artifact of that charge.

**Corrected result: with realistic side data the frontier is unchanged.**

| arm | B/vec (seed-only) | R@10 | baseline | B/vec | R@10 | winner |
|---|---|---|---|---|---|---|
| remex d=384 @ 2-bit | **100.0** | 0.609 | fp32 d=384 | 1,536 | 0.598 | **codec** |
| remex d=384 @ 1-bit | **52.0** | 0.564 | fp32 d=128 | 512 | 0.564 | **codec** |

Even charging the RHT operator form it moves to 100.8 / 52.8 B — the conclusion
does not depend on the choice. **There is no inversion at n=179, and no
break-even to clear.**

The one configuration where side data genuinely dominates is v2 with `haar` +
the int8 rotation sidecar, which exists so a JavaScript reader can reproduce the
planes. There, 1.2 MB of rotations against 179 chunks is a real 6,590 B/vec —
and it is an argument for `srht`, i.e. for the RHT, on exactly the grounds the
RHT was adopted.

### Corrected rule of thumb

**Spend the budget on coordinates, not bits — once the corpus is big enough to
amortize the rotation, or the rotation is regenerated/cached rather than shipped.**
At n in the low hundreds with side data materialized, plain Matryoshka truncation
is the better spend, and it has the additional virtue of needing no side data,
no codebook, and no per-query reconstruction at all.

For remax_kb specifically — which regenerates — the payload frontier stands, and
the right reading is: **remex d=384 @ 2-bit at 100 B/vec beating fp32 d=384 at
1,536 B/vec, on the condition that §6's caching fix lands.** Without it you are
paying 53 ms/query for those savings.

remex beats remax in 30 of 32 cells. remax reaches the frontier once (a8m,
256 B, k=8). Given remax's own framing — a rank-correct precision ladder, not a
byte-optimal codec — that is consistent, not a defect.

### Bit depth: bekko lands with Jina, not SPECTER2

The one-bit-beats-two inversion is embedder-specific. On bekko, **2-bit beats
1-bit in all 8 (variant × dim) cells**, by 0.016–0.129 R@10:

| dim | a8m 1-bit → 2-bit | a25m 1-bit → 2-bit |
|---|---|---|
| 384 | 0.531 → 0.547 | 0.564 → 0.609 |
| 256 | 0.520 → 0.547 | 0.531 → 0.587 |
| 128 | 0.447 → 0.525 | 0.508 → 0.547 |
| 64 | 0.408 → 0.441 | 0.430 → **0.559** |

Unanimous, and largest at the narrowest width. **bekko is a Jina-side
embedder**: the SPECTER2 1-bit result does not generalize to it, and a
`.kb` built on bekko should use 2-bit, not the 1-bit default.

---

## Published follow-up, and what it left out

Muninn (chat wing) published
[*The Compression Result Is on the Model Card, Below the Fold*](https://muninn.austegard.com/blog/compression-result-below-the-fold.html)
(2026-08-04) from this experiment's Part B.

**The post is sound.** It uses the corrected payload figures (52 B / 100 B, not
the pre-correction 48 / 96), avoids all three claims retracted here (the d=12
strawman, the materialized-rotation inversion, the "large codebook"), and states
its own limit honestly: *"random variation runs to about seven queries either
way, and we never ran the paired test."*

**That test has now been run, and its thesis holds:**

| claim | Δ R@10 | disc. | p | verdict |
|---|---|---|---|---|
| remex 2-bit @100 B beats vendor floor d=64 @256 B | **+0.089** | **22/6** | **0.0037** | **SUPPORTED** |
| remex 2-bit @100 B vs uncompressed @1536 B — post calls it *a tie* | +0.011 | 3/1 | 0.625 | tie confirmed |
| remex 1-bit @52 B vs d=64 @256 B | +0.045 | 17/9 | 0.169 | not resolved |

The post's headline is the 2-bit comparison, which survives; the one comparison
that would not have carried a headline (1-bit vs d=64) is not the one it leads
with. Its "±7 queries" estimate also matches the bootstrap CIs.

Every other verifiable number in it reproduces from this repo's artifacts: the
raw query counts (109 / 107 / 93 of 179), the 52 B and 100 B payloads, the 53 ms
rotation regeneration, the 12 B and 20 B frontier points (both d=64, at 1- and
2-bit), "2-bit beat 1-bit in all eight cells by 0.016 to 0.129" (measured 8/8,
0.017–0.128), 26/179 unrecovered, BM25 recovering 14 of 26, RRF at 0.615 / 0.866,
RM3 recovering 3, jina missing 20 with 13 shared, and the 7.3% floor.

One claim is **not verifiable from here**: "we have measured four distributions
now, and the ordering held on each." This experiment measured **two** — the
muninn blog subset and a sklearn code slice, both n=179. The other two come from
prior work in the series (SPECTER2, Jina), which this repo does not hold.

**What it left out: all of Part A.** No mention of code search, scikit-learn,
`rg`, file discovery, or the reversal of the 2026-07 `searching-codebases`
verdict. That omission has one good reason and one bad one:

- **Good:** Part A is n=6 instances with **n=1 on the deciding stratum**. It does
  not meet the evidentiary standard the compression post itself met (n=179 plus a
  paired test). Publishing it at that strength would have been the weaker act.
- **Bad:** Part A is the finding with an **operational consequence**. The
  semantic tier of `searching-codebases` was retired in 2026-07 on evidence that
  falsified *TF-IDF*; this experiment indicates a neural encoder flips that
  result, and on the identifier-poor instance grep scores **0.000**. That is a
  live claim about Oskar's own tooling, and it is currently unpublished and
  unresolved.

There is also an asymmetry worth naming: the published post rests on the
**179-chunk** arm that cost *seconds*, while the **78-minute** arm produced
nothing publishable. Compute and evidentiary weight went to opposite arms.

**The fix is cheap, and it is not compute.** Part A's sample size is limited by
*instance mining* — GitHub search calls to recover (issue body, fixing PR, gold
file set) — not by encoding. The sklearn corpus is a fixed ~8 min (now ~6 with
length-sorted batching) regardless of how many instances are scored against it,
and each additional instance is one query encode plus a `rg` pass. **Going from
n=6 to n=50 costs search-API calls, not GPU-minutes**, and the miner
(`scripts/mine.py` + the date-window recovery procedure) already exists.


## Where the harness should live

It has now been rebuilt three times, and **the code was never the expensive
part** — it is ~3 tool calls. What was lost each time is the **instance set**,
which is why "reuse the same 7 instances" was not satisfiable this run.

Proposal: keep it here, in `oaustegard/experiments/bekko-embedding-bench/`, with
the durable artifact being **`instances.json`** (issue number, PR number, commit
sha, gold file list, issue body) plus `pr2commit.json`. Both are committed; the
embeddings and clones are gitignored and regenerable. A fourth rebuild then
costs a `git clone` and one encode run, and — critically — scores the *same
instances*, which is the only way any of these numbers stay comparable.

Do **not** lift it into `_lib/`: per that repo's rule, code earns a place there
once a *second* experiment needs it, and nothing else does yet.

The mining constraint belongs in `METHODS.md`, because it is not specific to
this experiment: **`add_repo` cannot add a cross-owner repo, so `api.github.com`
403s for it — but `git clone` and `mcp__github__search_*` both still work, and
between them they are sufficient to build a PR-gold benchmark offline.**

## Caveats

- **n = 6**, and n = 1 on the stratum the decision gate turns on.
- Instances are **re-mined, not the prior 7** — cross-run comparison is by
  aggregate difficulty, not per-instance.
- Retrieval is scored against **current main**, not each PR's base commit
  (same caveat as both prior runs).
- Part B numbers are **self-retrieval**, not human-labelled relevance. Fair
  across models (identical splits) but not an absolute quality measure.
- Latency is measured on the **shipped artifacts** (bekko default ONNX, jina
  official q4) — the ones remax_kb would actually load. A different jina export
  might time differently, though the measured ratio matching the FLOPs ratio
  says the gap is architectural rather than an artifact of q4.
- Wall clock: ~81 min of corpus encoding on 4 vCPU for the four cells
  (a8m 11/11 min ast/flat, a25m 33/26 min), partly overlapped with Part B.
