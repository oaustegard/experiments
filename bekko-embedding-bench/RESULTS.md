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

### Results (mean over 6 instances)

| cell | rg r@5 | rg r@10 | bekko r@5 | bekko r@10 | RRF r@5 | RRF r@10 |
|---|---|---|---|---|---|---|
| ast / a8m | 0.667 | 0.778 | 0.806 | **0.889** | **0.889** | **0.889** |
| ast / a25m | 0.667 | 0.778 | 0.806 | 0.806 | **0.889** | **0.889** |
| flat / a8m | 0.667 | 0.778 | **0.833** | **0.889** | 0.833 | 0.889 |
| flat / a25m | 0.667 | 0.778 | 0.806 | **0.889** | **0.889** | **0.889** |

The full 2x2 is reported. Every cell puts dense at r@5 0.806-0.833 and RRF at
0.833-0.889 against grep's flat 0.667, so the headline does not depend on which
cell you pick.

Prior runs, for reference: naive rg 0.57/0.79; retired **TF-IDF** tier 0.36/0.74.

### The decision gate — passes

> Build only if bekko clears rg at r@5 on the identifier-poor stratum **and**
> does not regress on the identifier-rich stratum.

| stratum | n | rg r@5 | bekko r@5 (ast/a8m) |
|---|---|---|---|
| identifier-poor (#22186) | 1 | **0.000** | **0.667** |
| identifier-rich | 5 | 0.800 | 0.833 |

Clears on the poor stratum (grep scores *zero* — it has nothing to grep for);
no regression on the rich stratum (r@10 tied at 0.933). **This is a real
reversal of the TF-IDF result**, and it is exactly the reversal the handoff
predicted: the 2026-07 replication falsified TF-IDF, not dense retrieval.

**But report it as the thin slice it is.** n=1 on the deciding stratum, against
a standing base rate of 2/600 (~0.3%) of merged sklearn PRs fixing
identifier-poor issues. A win here is a win on ~0.3% of traffic.

The StackOverflowQA warning in the handoff (a8m 74.9 vs gte-base 87.1) did not
show up as a disqualifier at this n — but n=6 cannot detect it either.

### Fusion is the actual recommendation

RRF(rg, bekko) is the best arm at r@5 (0.889 vs 0.806 dense-only, 0.667
rg-only) because the two fail on **disjoint** instances. Dense alone regressed
on #18318 (r@5 0.50); fusion recovers it to 1.00 while keeping the #22186 win.
Neither arm alone dominates; the union does.

### Cost — both framings, because only one is flattering

| arm | tokens (6 instances) | wall |
|---|---|---|
| `rg` full line output | 328,230 | 0.8 s |
| **`rg -l` (filenames only)** | **13,336** | 0.8 s |
| bekko dense, top-10 chunks | 25,568 | 1.0 s (+ index) |

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

### Correction — the first pass mis-priced this comparison twice

Prompted by a challenge to check whether the vendor's Matryoshka was being
applied correctly against the codecs. **The truncation itself is correct**:
normalize-full-then-truncate-then-renormalize is bit-identical to
truncate-raw-then-normalize (max |Δ| ≤ 6e-8, i.e. float32 epsilon), which is the
vendor's prescribed procedure. The *accounting* was wrong, twice, and both errors
favoured the codecs.

**Error 1 — the payload itself was undercounted.** Bytes were hand-computed as
`dim*bits/8`, which omits the **float32 norms remex stores separately**. remex's
own `CompressedVectors.nbytes` gives **52 B/vec** at d=384 @ 1-bit, not 48, and
**100 B** at 2-bit, not 96. Negligible at d=384; **+50% at d=64 @ 1-bit**
(12 B, not 8). All tables below now use remex's own accounting.

**Error 2 — and this is the one that matters — shared structures were free.**
**Matryoshka truncation ships nothing**: it is a slice of a vector that already
exists. remex needs a d×d rotation (+ codebook); remax needs *k* of them. Counting
those at zero compares a codec's payload against a baseline's total. This repo
has already been caught by exactly this trap once — `remex-vs-higgs-ablation`
found that "counting the shared codebook, the vector arm costs 52.5 B/vector at
4 bits against a 50 B payload… needs ~350k vectors to amortize." I reproduced the
error rather than the lesson.

With side data materialized, the claims **invert at the corpus actually
benchmarked** (n=179):

| arm | true B/vec @ n=179 | R@10 | baseline | B/vec | R@10 | winner |
|---|---|---|---|---|---|---|
| remex d=384 @ 2-bit | **3,395** | 0.609 | fp32 d=384 | **1,536** | 0.598 | **baseline** |
| remex d=384 @ 1-bit | **3,347** | 0.564 | fp32 d=128 | **512** | 0.564 | **baseline** |
| remex d=64 @ 2-bit | **112** | 0.559 | fp32 d=64 | 256 | 0.520 | codec |

Break-even corpus sizes, materialized:

| comparison | break-even n |
|---|---|
| remex d=64 @ 2-bit vs fp32 d=64 | **69** |
| remex d=384 @ 2-bit vs fp32 d=384 | **411** |
| remex d=384 @ 1-bit vs fp32 d=128 | **1,282** |

So `muninn-subset.kb` (179 chunks) sits *below* two of the three break-evens and
`muninn.kb` (1,238) sits near the third. The wide-and-low-bit spend only pays
once the rotation amortizes.

### But the rotation is seed-derived — bytes and latency are the same object

The thing that makes this more than an erratum: those rotations are
**deterministic from (dim, k, seed)**. A reader can ship them *or* regenerate
them. remax_kb v1 regenerates — which is precisely the **53 ms/query constant**
measured in §6, and why v2's `srht` was chosen to "ship nothing" (§7).

So the two sections were pricing one object in two currencies, and each in
isolation flatters the codec:

| you pay in | cost | measured in |
|---|---|---|
| **bytes** (ship the rotation) | +3,295 B/vec at n=179, +476 at n=1,238, +6 at n=100k | this section |
| **latency** (regenerate per query) | +53 ms on every query, forever | §6 |
| **neither** (ship once, cache in memory) | ~0 | §6's verified fix |

The payload-only frontier is therefore *correct for the deployment remax_kb
actually implements* — it regenerates, so side bytes really are zero — but only
because it is paying in the other currency. Quote payload-only bytes **and** the
per-query constant together, or the compression story is half-told.

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
