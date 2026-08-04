# bekko-embedding-v1 (a8m / a25m) as code-search and remax_kb embedder

Handoff: [claude-workspace#185](https://github.com/oaustegard/claude-workspace/issues/185).
Run 2026-08-04. Box: **4 vCPU / 15 GB** (CCotw). Every throughput number below is
on that box; the claude.ai container (1 vCPU / 3 GB) is not comparable.

**Headline.** Part A reverses the 2026-07-04/05 verdict: a neural code-capable
encoder *does* beat identifier grep on file discovery, including on the
identifier-poor instance where grep scores zero — the decision gate passes.
Part B goes the other way: bekko **loses to the incumbent jina v5 nano q4 at
every byte budget on both distributions**, so the remax_kb default should not
change. These are not in tension; they are different questions.

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

| axis | change | Δ r@5 | Δ r@10 |
|---|---|---|---|
| chunking (encoder fixed a8m) | ast → flat | +0.028 | 0.000 |
| encoder (chunking fixed ast) | a8m → a25m | 0.000 | −0.083 |

The handoff's prior was "chunk boundaries matter more than the encoder here."
**Stated explicitly, as asked: it does not hold.** Both effects are within noise
at n=6, and both are small next to the dense-vs-grep gap (+0.14 r@5) that is the
actual finding. The one place chunking shows up is the identifier-poor instance,
where AST chunking beats flat (r@5 0.667 vs 0.333) — n=1, so suggestive only.

**a25m does not earn its cost.** 3x the encode time (33 min vs 11 min) for
r@10 0.806 vs a8m's 0.889. The 4-layer model is the one to use.

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

**Verdict: do not swap the remax_kb default.** The incumbent is better at every
byte budget, on both distributions, at the same artifact size.

### 3. The strategic note (regime A) — already true, without bekko

The handoff's point was that a ~130 MB encoder becomes shippable alongside the
`.kb`, collapsing regime A from `7cecfd94` (the "light embed endpoint with a
shared secret" existed only because the encoder was too big to distribute).

**That is already the case with the incumbent.** The official jina q4 that
remax_kb *already defaults to* is 131.6 MB. Nothing about bekko is needed to
retire the endpoint-and-secret design; the redistribution question reduces to
whether the *corpus* is redistributable, exactly as the handoff framed it — but
the encoder-size premise was satisfied before this benchmark ran.

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
| 8 | remex d=64 @ 1-bit | 0.430 |
| 16 | remex d=64 @ 2-bit | 0.559 |
| 48 | remex d=384 @ 1-bit | 0.564 |
| 64 | remex d=256 @ 2-bit | 0.587 |
| **96** | **remex d=384 @ 2-bit** | **0.609** |
| 512 | fp32 d=128 (truncation) | 0.564 |
| 1536 | fp32 d=384 (full) | 0.598 |

**The answer to "does remex/remax beat Matryoshka at a fixed byte budget" is
yes, decisively.** remex d=384 @ 2-bit costs **96 B and scores 0.609**, against
full fp32's **1536 B for 0.598** — 16x smaller and not worse. Truncation is the
*worse* way to save bytes on this corpus at every budget compared:
fp32 d=128 (512 B, 0.564) is matched by remex d=384 @ 1-bit at **48 B** — 10.7x
fewer bytes for the same recall.

Rule of thumb this supports: **spend the budget on coordinates, not on bits.**
Keep all 384 dims and drop to 1–2 bits before you truncate to 128 dims at fp32.

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
- `flat/a25m` was still encoding at write-up; the 2x2 is reported as 3 cells.
- Wall clock: ~50 min of encoding on 4 vCPU for the four corpus cells.
