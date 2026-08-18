# Methods ledger — oaustegard/experiments

Portable methods, gotchas, and negative results extracted from the 40
experiments in this repo. Each entry names the experiment that produced it.
**Grep this file before starting a new experiment.**

Convention follows `ms13-campaign/NOGOS.md`, which is already this shape.

## Why this is one file and not forty

The tempting design is a `TECHNIQUES.md` inside each experiment. The repo's
own history argues against it: `te-bridges` repeated `phase-a-bridges`'
Cloudflare-gateway concurrency lesson from the adjacent directory, and
`recall-per-byte` re-derived an ITQ overfitting result that `remax#46` had
already documented. Co-location did not prevent rediscovery, because the
failure mode is not "I found the experiment but missed the reusable bit" —
it is "I never looked". One greppable file at the root is the smallest
thing that answers "has anyone here already hit this?"

## Adding to it

When an experiment ends, add any finding that would change what a *different*
experiment does. Three tests for whether it belongs:

- Would it save someone an hour, a rerun, or a wrong conclusion?
- Is it true outside the experiment that produced it?
- Is it findable by someone who does not know that experiment exists?

If a finding is code rather than prose, put the code in `_lib/` (see below)
and leave a one-line pointer here.

---

## Cross-cutting principles

- **A cached matrix whose encode settings are not recorded is not reusable, no
  matter how exactly its shape and provenance match.** `ttt-embed-quantized`
  needed SciFact under jina-v5-nano; `rotation-decorrelation` already had a
  `jina_scifact_corpus.npy` — same corpus, same embedder, cached in a
  `claude-container-layers` release. It was declined and the 15-minute encode
  paid anyway, because nothing anywhere recorded its `dim` / `max_length` /
  prefix / pooling. The trap is that reuse would have *looked* fine: had it been
  256-dim, every shape, dtype and norm check would have passed and the
  incomparability would have been permanently silent. **"Prior art exists" and
  "prior art is usable" are different findings, and the second one needs
  recorded settings, not a matching filename.** Corollary for producers: an
  embedding artifact ships its full encode configuration next to it or it is
  write-only — `ttt-embed-quantized/data/meta.json` carries encoder repo,
  release tag, asset SHA256, dim, max_length, pooling rule, prefixes and text
  template for exactly this reason. (`ttt-embed-quantized/RESULTS.md`,
  `ERRORS.md`)
- **When the deliverable's job is to be *comparable*, pin the superseded
  artifact on purpose.** `ttt-embed-quantized` encoded with
  jina-v5-nano-mirror's `model.q4.onnx` while both that mirror's
  `PERFORMANCE.md` and this file's own "hand-rolled q4 is dominated by the
  official one" entry say the authors' upstream q4 is smaller *and* more
  faithful. Substituting the better encoder would have silently voided the
  premise — that the 2026-07-08 codec eval's fidelity numbers carry over —
  because comparability constrains **weights**, not just hyperparameters. The
  general rule: "use the better tool" and "match the prior run" are in genuine
  tension, the prior run wins whenever the output feeds a comparison, and the
  decision gets written down rather than made implicitly by whoever reads the
  model card last. (`ttt-embed-quantized/RESULTS.md`)
- **A quantization ladder measured on one corpus does not transfer by
  assumption — but on this pair it did, and got *stronger* on the harder task.**
  The byte/quality ladder here was first run on 179 blog chunks by
  self-retrieval, where remex 1-bit @48 B vs Matryoshka d=64 @256 B was
  directionally right but **not significant** (p=0.169). Re-run over 11,380
  scikit-learn AST chunks and scored on a real NL->file task (n=59), the same
  comparison is **+0.087, 12 wins to 3, p=0.035** — supported. Two things
  generalize: **quantize wide rather than truncate narrow** held on both
  distributions, and **2-bit at 96 B is statistically indistinguishable from an
  uncompressed 1536 B vector** on both. One caveat that only the code run
  exposed: **1-bit is not free** — it loses to full fp32 by a real margin
  (0 wins to 8, p=0.008), so the sweet spot is 2-bit, not 1-bit.
  (`bekko-embedding-bench/RESULTS.md`)
- **A retrieval sidecar for a *code repo* should not carry the corpus text — the
  working tree already is the corpus.** Storing `(path, start-line)` pointers
  instead of chunk bodies took the non-vector half of a scikit-learn index from
  **11.3 MB to 0.04 MB gzipped**, making a whole sidecar **0.56 MB at 1-bit /
  1.08 MB at 2-bit against a 14.1 MB source tree (4.0% / 7.7%)**. Staleness is
  also a non-issue if updates are incremental: measured churn over 197
  code-touching sklearn commits is **median 2 files / mean 3.9 / p90 7**, at
  ~16.9 chunks per file, so a median commit invalidates ~34 chunks — **1.9 s** to
  re-encode with a small model against ~10 min for a full rebuild. Requires a
  mutable index format (`.kb` v1 is immutable; v2 `.kbi`/`.kbc` has tombstones).
  The binding constraint is **not** size or freshness — it is that the dense arm
  only ties `rg`, so build it as a fusion partner, not a replacement.
  (`bekko-embedding-bench/RESULTS.md`)
These were each surfaced independently by separate surveys of disjoint
experiment sets — convergence is the evidence they generalize.

### 1. Verify with a deliberately disjoint code path, not a second implementation

Two independent implementations agree on a false result when they share a
modelling assumption. Vary the assumption and the author, not just the code.

- `ms13-campaign/NOGOS.md` NG-7 — two independently written verifiers both
  assumed a declared path menu instead of the full path set, and jointly
  certified a false counterexample.
- `discrepancy/verify_certificates.py` — reloads only raw stored witnesses and
  recomputes via brute-force `Fraction` enumeration, sharing nothing with the
  search engine; non-zero exit on mismatch. Same convention in `woodall/`.
- `te-bridges` stage 8 — a separately-prompted Opus pass re-verifies each
  claimed "bridge" against source text, deliberately not a re-run of the same
  prompt.
- `optimizing-skills-retro/VALIDATION.md` — ≥2 independent authors when the
  artifact under test is itself agent-compiled, to avoid author-variance
  confound.

**Use when:** any harness where a search proposes and a checker disposes.

### 2. Two-sided calibration gates

A verifier must both find a known-bad instance *and* certify a known-good one
clean. One-sided "found nothing" tests hide inverted signs.

- `ms13-campaign/NOGOS.md` NG-11 (GATE 5) — caught an inverted big-M polarity
  bug and a flipped epsilon sign that would have returned "no counterexample"
  everywhere.
- `optimizing-skills-retro` — the check set must include the failure that
  motivated the edit (held-in), plus a regression guard (held-out).

**Two-sided is necessary, not sufficient — measure the known-bad's *reach*.**
`remex-vs-higgs-ablation/calibrate.py` was two-sided by this principle's own
standard and still had three checks that could not fail. Its single known-bad
(an under-trained grid) exercised exactly one of eight checks; the check axis C
actually rested on *accepted* the same bad grid. Audited with the `gating`
skill's procedure (`audit.py` in that directory):

- **A relative check has no anchor.** "RHT's incoherence is within 25% of
  Haar's" passes when *both* rotations are replaced by the identity — the
  worst possible incoherence — because nothing compares either to an absolute
  reference. Shared failure modes are exactly what a relative check cannot see,
  and two arms of one factorial share almost everything. Fix: bracket against a
  reference computed by a path that touches neither arm.
- **A fallback candidate can make an assertion true by construction.** Adding
  the scalar product grid as a training candidate — itself a good fix, for a
  real bug — made "the vector grid beats the scalar grid" nearly tautological:
  a vector arm doing no vector quantization degrades to the fallback and lands
  on the right side of the inequality by sampling noise. An inequality with no
  stated margin cannot distinguish "worked" from "did nothing." Fix: require a
  margin derived from measured noise (a *paired* estimator, so the shared
  fluctuation cancels), against the degenerate baseline rather than a
  differently-computed one.
- **State the anchor's range, then check past it.** Max (1960) stops at 5 bits;
  the sweep runs to 8. Beyond the table there was no assertion at all, only a
  printed note — and since the scalar MSE is the vector arm's threshold, an
  inflated value *loosens* the check downstream. Fix: Panter–Dite
  (2.7207·2⁻²ᵇ) as a hard upper bracket at the unanchored rates.

**Score the artifact, and also run the code that produces it.** Mutation
testing (`gating/scripts/mutate.py`, 91 mutants over that experiment's two core
modules) found the gate scored codebook *contents* — the MSE of a point set —
but never invoked the *encoder*. Mutating the decision boundaries, the
nearest-neighbour k, and the axis-B norm-mode branch all left it green, and
those functions run on every vector of every corpus. Cheap fixes with real
anchors: idempotence (`Q(Q(x)) == Q(x)`), membership (every output is a
codepoint), and the encoder attaining the point set's own distortion. Two
further survivors were tautologies *in the rebuilt gate* — payload bytes
compared between two arms computed by the same expression, and a `total` field
nothing read. Kills verified individually and permanently in
`remex-vs-higgs-ablation/verify_kills.py`; equivalent mutants are listed there
with the reason each is unobservable, because an unexplained survivor and an
equivalent mutant look identical in a report.

**A statistical margin is not a practical floor, and a paired one gets weaker
the closer the comparison is.** Replacing a bare inequality with "must beat the
baseline by 3 standard errors of a paired estimator" is the right move against
sampling noise, and it is still not a floor: the paired se *shrinks* as the two
things converge, so an arm with a real-but-worthless advantage clears it easily.
Measured in `remex-vs-higgs-ablation`: a product grid perturbed by N(0, 1e-3)
gains +0.0001 dB against a 3-se margin of 1.2e-06 and is accepted, while the
real grids gain 0.35–1.41 dB. "Significant" and "worth having" are different
assertions and need different thresholds — the second one has to come from an
anchor (there, a published E8-ball codebook), not from the estimator.

**Validate a known-bad at the configuration it will run in.** The same
experiment's zero-gain known-bad was built from one Lloyd iteration to avoid a
degenerate zero standard error. It passed in the fast configuration (m=2,
K=16) and failed at full size, because at m=8 with K=65536 a single iteration
relocates ~63,000 empty-cell codepoints toward the mode and earns a genuine
+0.10 dB — it was not a zero-gain arm at all. A known-bad that is only bad at
small scale certifies nothing at large scale.

**Ship a sub-5-minute re-verification fixture, and make it check the prose
against the artifacts.** The expensive parts of an experiment (corpora, trained
models, sweeps) are hours and are usually gitignored, so nobody rebuilds them to
check a one-line edit — which means the writeup and the data drift apart
silently. `remex-vs-higgs-ablation/recheck.py` is the shape: artifact integrity,
then the headline numbers **recomputed from the stored results by a path that
does not import the summariser**, compared against the numbers parsed out of the
prose; then internal consistency of the gate log against the counts the prose
quotes; then the fast gate and a sample of pinned mutants. 90 seconds. Two of
that experiment's logged errors were prose disagreeing with artifacts it
described, and no amount of re-running the science would have found them — the
numbers were right and the sentences about them were wrong. State plainly what
the fixture does *not* cover, which is normally the science itself.

**Keep an `ERRORS.md` per experiment: what was wrong, how it was caught, which
direction it pushed the conclusion.** The base rate is the most useful
calibration number about a body of work and the one nobody records; without it
the only options are "trust the writeup" and "trust nothing." Direction is the
column to read — an error that makes a check more permissive, or a result look
stronger, is far more dangerous than one that makes it look weaker, because the
second announces itself the moment someone tries to use the result.
`remex-vs-higgs-ablation/ERRORS.md` logs 23 errors across three runs, 7 of them
in the flattering direction, and records three things worth generalising: **an
estimate of one's own error count made an hour later was wrong** (seven, versus
eight counted); **not one of the twenty-three was caught by reading code
carefully** — every one came from executing something, comparing two artifacts,
or an outside party; and **a correction that lands in one paragraph is not a
correction** — run 3 #1 found a claim still standing in the reproduction
instructions that run 2 had already logged, diagnosed, and fixed elsewhere in
the same file. Buy execution and comparison, not care. Append, never tidy;
a cleaned log loses the base rate, which is the only thing it is for.

**Register anchors with their covered range, in one place.** See `ANCHORS.md`.
The failure it prevents is specific: Max (1960)'s quantizer table stops at 5
bits, the sweep ran to 8, and for two runs nothing checked past the end of the
table — where a 16%-high value sat, in the direction that *loosened* the check
downstream of it. The gap was discoverable at any moment by writing the range
next to the constant, and nobody did because there was nowhere to write it. Add
a row when you use a constant, not when you audit.

**A gate that does not block is a report.** That experiment's first run listed
the gate as step 3 of a documented command sequence, so nothing stopped the
sweep from running with it red or unrun. `run_ablation.py` now invokes it and
aborts on non-zero exit — including exit 2, INCONCLUSIVE, which the `gating`
harness returns when a gate registers no known-bad or states no coverage limit.

### 3. Fit and evaluate on the same corpus and learned methods look better than they are

- `recall-per-byte/RESULTS.md` — ITQ's apparent win over parameter-free random
  rotation reversed under a transfer split. **This experiment re-derived a
  mistake `remax#46` had already found and documented.**
- `rotation-decorrelation/RESULTS.md` — in-corpus vs transfer protocol split as
  a direct overfit measurement; parameter-free methods show zero gap by
  construction.
- `haiku-assessment/RESULTS.md` — held-out probes for prompt edits.

**Use when:** evaluating any learned transform — PCA, whitening, learned
hashing, PQ codebooks, pruning masks, or a prompt.

### 4. Score against your own uncompressed reference, not human labels

Human qrels conflate "is the base method good" with "did my approximation
damage it", and saturate.

- `jina-remex-vs-remax/score_fidelity.py` — recall@k vs fp32-kNN, per-query
  Spearman ρ, reconstruction cosine. Never touches labels. **Portable code.**

**Score the candidate under the same metric as the reference.** Scoring
`q @ x̂` against a cosine reference without dividing by `‖x̂‖` mixes the
angular question with a reconstruction-norm question, and does it *unevenly
across codecs*: a 1-bit code has constant reconstruction norm by construction
and pays no norm penalty at all, while a multi-bit code carries real norm
error. In `remex-vs-higgs-ablation` this manufactured a false low-rate
reversal (0.663 → 0.689 once renormalised). The same pattern was present in
`score_fidelity.py` above; both are fixed as of 2026-08-01. The failure is
invisible at high bit widths and grows as the codebook coarsens, so it
survives exactly the sanity checks people run.
- `kb-k-sweep/sweep.py::topk_float`/`recall_at_k` — float-cosine top-k as
  ground truth, self-excluded. ~10 lines, numpy only. **Portable code.**

### 5. Do the arithmetic before spending the compute

- `ms13-campaign/NOGOS.md` NG-14 — counted census size × canonicalization cost ×
  measured rate for k=4, concluded ~2,070 h, never launched it.
- `erdos-gyarfas/README.md` — `nauty-geng` + filter ran at 145 graphs/sec;
  `snarkhunter`, which constructs cubic graphs directly, ran the same pipeline
  at 385,000/sec. **2,600×.** Check for a specialised generator before writing
  a filter over a general one.

### 6. A plausible speed-hack filter can silently exclude the target

- `erdos-gyarfas/README.md` — restricting to girth ≥ 5 shrank the order-26
  search 60-fold and returned nothing; all 27 actual extremal graphs have
  girth 3. Before adding a structural pre-filter for speed, check explicitly
  that it cannot exclude the object class you want.

### 7. Checkpoint every stage — runs get reaped

- `phase-a-bridges`, `te-bridges` — numbered idempotent stage scripts, each
  reading the prior stage's JSON and writing its own atomically
  (tmp-then-rename), safe to re-run.
- `q4-official-vs-ours`, `jina-int8-remax_kb` — CCotw silently reaps
  long-running background jobs on idle; multi-minute embedding runs must
  checkpoint to memmap/`.npz` mid-run.

- **Before reporting a ratio between a marginal cost and a price, compute the
  marginal cost's share of total cost — otherwise the ratio is true and decides
  nothing.** `luna-onprem-tco` was asked to compare the electricity cost of
  local inference against an API's per-token price, computed it carefully
  (electricity is **57×** cheaper per million input tokens), and the answer was
  useless: electricity is **3%** of the cost of owning the hardware. What
  actually decides on-prem-vs-API is a **capacity floor** — matching a closed
  model's capability forces a specific open model, whose weights force a
  minimum GPU count, whose capex you pay at 5% utilisation exactly as at 85%.
  The tell was present in the first pass and printed as a footnote *underneath*
  the headline it invalidated ("electricity is 15.7% of hardware TCO"). Generic
  form: **a per-unit comparison is only a decision input when the unit cost is
  a large share of total cost; otherwise report the share first and the ratio
  second, or not at all.** Same shape as `account-routing-tier`'s "cost the
  baseline before optimizing against it", one level up — there the error was
  optimising a cheap path, here it was *measuring* one.
  **What replaces the ratio is duty cycle**: run the same model at two scales
  and it answers oppositely for the same reason. An 800-seat fleet loses by
  2.7× because a $450 k node is bought whole and filled 17%; a single $6,000
  RTX 5090 *wins* above ~4 h/day of flat-out generation. Neither turns on
  $/token. Both turn on what fraction of a fixed capacity is used — so the
  question to ask a buy-vs-rent proposal is "what duty cycle", not "what does
  a token cost". And define the duty cycle in the unit that bills: **hours of
  flat-out generation, not hours the tool is open.** Interactive chat emits
  tokens ~5–10% of wall-clock time, so "I use it 4 hours a day" is off by an
  order of magnitude from "4 hours of generation", and it is off in the
  direction that justifies the purchase.
  (`luna-onprem-tco/RESULTS.md`, `ERRORS.md` "the framing error")

- **Any model that compares "buy capacity" against "rent per unit" must reject
  configurations that cannot serve the load, structurally — every instance of
  that error flatters buying.** `luna-onprem-tco/model.py` cheerfully returned
  `verdict: self-host` at **845% peak utilisation**, comparing one node's cost
  against a token bill that needs nine. It is not a caveat, it is a missing
  constraint: `nodes = max(1, ceil(peak_utilisation))`, with capex *and*
  throughput-derived capacity both scaling on it. Two properties make this
  class dangerous — it is invisible in aggregate annual figures (which balance
  fine), and it is **one-directional**, so it never shows up as an implausible
  result in the direction you are watching for. Assert `per-node peak ≤ 100%`
  in the checker, not in prose. The sibling error in the same file: applying a
  *serving* power floor (idle + 35% of the load delta, correct for a lightly
  loaded GPU) to genuinely **parked** hours, inflating annual kWh 35% on a box
  that is parked three-quarters of the year. (`luna-onprem-tco/ERRORS.md` #3–4)

- **Check a quoted tok/s against the decode roofline before building anything
  on it: `bandwidth / weight_bytes` is a hard single-stream ceiling.** A dense
  model reads every weight once per output token, so no amount of tuning beats
  bandwidth ÷ bytes — only a different decoding scheme (multi-token prediction,
  speculative decoding) or batching. Qwen3.8-27B at ~4.25 effective bits is
  14.8 GB against an RTX 5090's 1,792 GB/s: **121 tok/s at 100% MBU, ~97 at a
  realistic 80%**. A quoted "180–200 TPS" is therefore **1.56× the hard
  ceiling** — true, but only via MTP, speculation or concurrency, which is a
  materially different claim from what the bare number implies. The check is
  one division and it tells you *which mechanism* you are being quoted.
  Corollary for the same box: **prefill and decode contend for one GPU**, so
  sustained output is `1/(1/decode + fresh_ratio/prefill)`, not the decode rate.
  With prefix caching, `fresh_ratio = (1 − hit_rate) × billed_ratio` — the local
  mirror of the API's cache reads. On a 5090 that turned 190 tok/s into 123–167
  depending on input intensity. (`luna-onprem-tco/hourly.py`, `ERRORS.md` #5–6)

- **Prompt caching has a break-even hit rate, and below it caching costs more
  than re-sending.** Providers that price cache *writes* above uncached input
  make this non-obvious. Luna: input $0.20/M, cached read $0.02/M, cache write
  **$0.25/M** (1.25×). Effective price is `h·0.02 + (1−h)·0.25`, which crosses
  plain input at **h = (0.25 − 0.20)/(0.25 − 0.02) = 21.7%**. Any workload with
  a lower reuse rate — one-shot bulk generation, high-cardinality prompts —
  should have caching *off*. Compute the crossing from the provider's own three
  numbers rather than assuming caching is free money; the general form is
  `h* = (write − input)/(write − read)`.
  (`luna-onprem-tco/params.json:api_prices`, `hourly.py::cache_breakeven_hit_rate`)

- **Rack-scale MoE inference benchmarks do not transfer to single nodes —
  decode is off by ~6×, prefill by almost nothing.** Measured on DeepSeek V4
  Pro (1.6 T, 49 B active): a GB200 NVL72 rack does **6,644 tok/s/GPU** decode
  where an 8×B200 node does **976** at the same interactivity, because MoE
  decode is all-to-all bound and scales with NVLink-domain size. In energy
  terms that is **$0.014 vs $0.083 per million output tokens** on identical
  GPUs at the same electricity price. Prefill, being compute-bound, is
  effectively flat between them (**$0.0035 vs $0.0030/M** — the small node is
  *better*, on PUE alone). So: **quote rack numbers for rack deployments only;
  a single node is a fine prefill engine and a poor decode engine**, and anyone
  sizing on-prem inference from published rack benchmarks will overestimate
  decode economics by most of a decimal order.
  (`luna-onprem-tco/params.json:hardware`, `RESULTS.md`)

---

## Portable code (extraction candidates)

| What | Where | Effort |
|---|---|---|
| Jittered exponential backoff + chunking + atomic JSON checkpoint + CF-gateway LLM client | `phase-a-bridges/scripts/common.py` (mirrored as `te-bridges/scripts/te_common.py`) | trivial |
| `RemaxBuilder` stacked-SimHash quantizer, numpy-only, chunked Hamming | `phase-a-bridges/scripts/remax.py` | trivial |
| uint64-view POPCNT Hamming kernel (`xor.view(np.uint64)` + `np.bitwise_count`), ~10× a LUT gather | `remax-hamming-speedup/bench.py` | trivial |
| Fidelity-vs-fp32 quantization eval (`eval_scores`, `recall_vs_gt`) | `jina-remex-vs-remax/score_fidelity.py` | trivial |
| Exact continuous Lloyd-Max for N(0,1) (reproduces Max 1960), plus Gaussian-optimal m-dim VQ grids by KD-tree-accelerated Lloyd | `remex-vs-higgs-ablation/grids.py` | trivial |
| E8 lattice: nearest-point decoder, ball-shaped codebook, normalised second moment | `remex-vs-higgs-ablation/calibrate.py` | trivial |
| Randomized Hadamard rotation for ANY d, no power-of-two padding (rounds of permute + block-diagonal FWHT) | `remex-vs-higgs-ablation/quantizers.py::RHTRotation` | trivial |
| Gate audit harness: probes that demonstrate a check CANNOT FAIL (identity-rotation substitution, degenerate-trainer substitution, anchor-range gap) | `remex-vs-higgs-ablation/audit.py` | small — probes are subject-specific, the four verdicts and the six passes are not |
| Permanent mutation-kill fixture: named mutations + the check claimed to catch each, restored after every run, plus documented equivalent mutants | `remex-vs-higgs-ablation/verify_kills.py` | trivial |
| Paired MSE estimator with an analytic standard error (two codebooks on one sample stream, so the shared sampling fluctuation cancels) — turns a bare inequality into a margin derived from measured noise | `remex-vs-higgs-ablation/gate.py::paired_gain` | trivial |
| Self-retrieval recall@k harness (`topk_float`, `recall_at_k`) | `kb-k-sweep/sweep.py` | trivial |
| Stdlib-only BM25 + RM3 index reader | `lexical-kb/skill_template/search.py` | trivial (already shipped as `creating-kb`) |
| Dependency-free txt/md/html extract + paragraph-respecting chunker | `lexical-kb/build_lexkb.py::extract_text` | trivial |
| Turso-backed append-only message relay CLI (`init`/`post`/`poll`/`wait`) | `session-relay/relay.py` | trivial |
| Stdlib-only LSP JSON-RPC client (Content-Length framing, threaded reader) | `python-lsp-stress/lsp_probe.py` | trivial |
| Exact fixed-length-cycle test, bitmask DFS over graph6, ≤62 vertices | `erdos-gyarfas/src/filt.c` | trivial |
| Straight-through-estimator `FakeQuant` + `QuantLinear`, ~15 lines | `qat-cpu-demo/qat_demo.py` | trivial |
| ONNX dynamic-int8 + blockwise int4 quantization recipe | `jina-int8-remax_kb/quantize.py`, `quantize_lowbit.py` | trivial–moderate |
| Float-screen-then-exact-reverify with rational denominator ladder | `ms13-campaign/sweep.py` | trivial–moderate |
| CEGAR loop for k-way packing feasibility (SAT + exhaustive verify + extract violated constraint) | `woodall/verifier/packing.py` | moderate |
| Certificate re-verification harness (load → recompute independently → diff → non-zero exit) | `discrepancy/verify_certificates.py` | moderate |

---

## Environment gotchas (this container)

- **The Cactus Needle engine holds one global session per process, so two
  `needle.Needle` objects do not coexist — every switch between them re-runs
  `needle_init`.** `_bind()` is a no-op only when the agent is already active
  (`needle/__init__.py`), and a tuned `.cact` cannot be unloaded at all: once
  loaded, constructing a base-weights agent raises rather than silently
  answering with the tuned weights. Any design that alternates agents (a
  two-stage router, an A/B comparison in one loop) pays init per turn or needs
  separate processes.
- **On a 4-core container, a background trainer inflates every concurrent
  latency measurement by an order of magnitude.** A five-tool Needle turn
  measured 284 ms alone and 3,644 ms with a LoRA fine-tune running — same code,
  same schema. Accuracy is unaffected where decoding is deterministic, so
  correctness arms can share the box; timing arms cannot. Record what else was
  running when a latency number was taken. (`needle-bsky/ERRORS.md` #6)

- **`pkill -f '<pattern>'` matches its own command line and kills itself**,
  returning non-zero and aborting the rest of a compound shell command — so a
  "kill the old run, start the new one" one-liner silently never starts the new
  one, and the next poll reads a stale log as if it were fresh. Use
  `pgrep -f '<narrower pattern>' | xargs -r kill`. Related: `setsid`-detached
  jobs in this container do not reliably survive the tool call that launched
  them; two multi-minute fine-tunes died mid-compile with no exit line. Use the
  harness's own `run_in_background` for anything longer than a tool call.
  (`needle-bsky/ERRORS.md` #3)

- **A CCotw session can have `WebSearch` working and `WebFetch` blocked for
  every domain — research is still possible, but every figure becomes
  secondary, and the writeup has to say so.** On 2026-08-15 the agent proxy
  returned `EGRESS_BLOCKED` to `WebFetch` and `CONNECT tunnel failed, response
  403` to `curl` for openai.com, openrouter.ai, artificialanalysis.ai, eia.gov,
  pepco.com, venturebeat.com, together.ai and deepseek.ai — every primary
  source `luna-onprem-tco` needed — while `WebSearch` answered normally.
  `curl -sS "$HTTPS_PROXY/__agentproxy/status"` reported `"selective": false`
  and an empty `recentRelayFailures`, i.e. **the status endpoint does not
  surface the allowlist**, so probing a URL is the only way to learn it is
  blocked. Consequences worth planning for: search-result *summaries* are the
  research channel, so numbers arrive already paraphrased by a model, and two
  of them in that session were visibly wrong (a "27B dense" parameter count for
  a closed model, and a per-seat token figure ~7× any plausible value) —
  reconciling several summaries against each other is the only available
  cross-check. Tag every constant with its provenance at capture time rather
  than reconstructing it later; `luna-onprem-tco/params.json` carries a
  `confidence` field per row and `recheck.py` fails if one is missing.

- **Hugging Face reachability differs between two CCotw containers running the
  same task at the same time — state the environment with any egress claim.**
  Issue #33 warned that HF load-balances its LFS redirect between
  `us.gcp.cdn.hf.co` (allowlisted) and `us.aws.cdn.hf.co` (not). Two encodes of
  that issue ran concurrently on 2026-08-14: one downloaded all three SciFact
  files **first try on `us.aws.cdn.hf.co`**, the supposedly-blocked leg; the
  other found **every** HF host refused — `huggingface.co`, `hf.co`,
  `datasets-server.huggingface.co`, `cdn-lfs.hf.co` *and both CDN legs* —
  answering `connect_rejected / gateway answered 403 to CONNECT`. Same repo,
  same day, same nominal platform. So this is not merely claude.ai-vs-CCotw
  (`bekko-embedding-bench`'s reading, still correct as far as it went): **egress
  policy varies per environment configuration and must be probed, never
  inherited** — from a spec, from a prior writeup, or from a sibling run. Two
  practical corollaries: `curl` alone hides the reason as bare exit 56, so read
  the proxy's own error to distinguish a policy denial from a transient failure;
  and **a policy denial is not to be retried or routed around** — find a
  sanctioned source instead. Keep retry loops for the environments that need
  them, and label which environment any egress claim came from, the same
  discipline `nproc`-with-throughput already gets.
  (`ttt-embed-quantized/RESULTS.md`)
- **Rebuilding BEIR SciFact from the upstream AllenAI release gets you 79.6% of
  the corpus and a green check suite.** When HF is unreachable, BEIR SciFact
  looks reconstructible from `scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`.
  Three mappings are not guessable and all three matter: **BEIR's *test* split is
  AllenAI's *dev* claims** (AllenAI's test claims ship unlabelled); **relevance is
  `cited_doc_ids`, not `evidence`** (`evidence` is the SUPPORT/CONTRADICT subset
  — 209 pairs over 188 queries against BEIR's 339 over 300, which would have
  *inflated* nDCG by shrinking the denominator); and the pair count is **339, not
  the naive 340**, because one dev claim cites the same document twice. Get all
  three right, assert the four published cardinalities, and the rebuild still
  **differs from `BeIR/scifact` on 1,055 of 5,183 documents (20.4%)** — AllenAI's
  `abstract` sentences carry trailing whitespace at structured-abstract section
  boundaries (`"…detected.   \n RESULTS We propose…"`) that BEIR normalised away,
  and `" ".join(abstract)` preserves it. Downstream: per-document cosine 0.9990
  mean / 0.9766 min, **nDCG@10 0.7067 against 0.7152** from real BEIR text, with
  30% of judged-relevant documents affected and the gold doc's rank moving on
  60 of 339 pairs. **Both numbers sit inside the pre-registered 0.60–0.72 sanity
  band**, so neither the counts nor the band detects this. If you must rebuild,
  diff the strings when you next reach a host that serves the real thing —
  `ttt-embed-quantized/crosscheck_allenai.py` does exactly that and is the
  cheapest form of the check. (`ttt-embed-quantized/RESULTS.md`)
- **`xr` needs three pip installs on a cold CCotw container — it is not
  unavailable there.** `scripts/xr.py` imports `remex` to decompress the index
  and `onnxruntime` + `tokenizers` to encode the query, and a bare container has
  none of them, so the first call dies with `ModuleNotFoundError: remex` and the
  second with `ModuleNotFoundError: onnxruntime`. Both are one line:

  ```bash
  python3 -m pip install --break-system-packages remex onnxruntime tokenizers
  ```

  `remex` is on PyPI, published by the same author as the repo (verify via
  `pypi.org/pypi/remex/json` → `project_urls.Repository`) — do not assume a
  bare name on PyPI is the right package without checking. Total cost well
  under a minute; after that the resident server answers in **175 ms** warm, as
  documented. This is worth stating because the failure mode is not the missing
  package, it is concluding from an ImportError that a *mandated* check is
  structurally unavailable and writing that into the record — which happened in
  `lowbit-scan-crossover`, on the one check that would have prevented its
  rediscovery. See the duplication map.

- **A Claude Code session cannot create GitHub releases — the proxy blocks it by
  policy, not by credential.** `POST /repos/{owner}/{repo}/releases` returns
  *"Creating, editing, or deleting releases is not permitted for this session
  type"* even for an in-scope repo with a PAT that can push commits and open PRs.
  Two adjacent gotchas found on the way: `curl -d @file` sends
  `application/x-www-form-urlencoded`, and the proxy rejects that with *"Form-encoded
  request bodies are not accepted"* — use `--data-binary` plus an explicit
  `Content-Type: application/json`; and `uploads.github.com` answering `400` on a
  bare root GET means the host is **reachable**, not blocked.

  **The fix is to move the privileged step into a `workflow_dispatch` Action.**
  The restriction is on the *session*, not on the repo or the token: an Actions
  run holding the repo's own `GITHUB_TOKEN` with `contents: write` creates
  releases fine. So a session can author and commit the workflow, and a human
  fires it with one click (or `gh workflow run <name>`). That generalizes past
  releases to anything the session type withholds but the repo itself permits —
  and it is strictly better than a documented manual procedure, because the steps
  end up version-controlled, reviewable and re-runnable instead of living in a
  README as commands to paste. Do *not* route a blocked capability through a
  side channel (a Worker on a custom domain, say) to defeat the check itself;
  that is a different thing from running it in a context that legitimately has
  the permission.

  **`workflow_dispatch` alone is not enough, though: it only appears once the
  workflow is on the default branch.** So the very first run is blocked behind a
  merge — and if the person who has to merge is on a phone, so is that. Add
  `on: pull_request: types: [labeled]` gated on a specific label name: a
  `pull_request` run uses the workflow file **from the PR itself**, so it is
  triggerable before merging, and adding a label is one tap in the GitHub mobile
  app where the Actions "Run workflow" form is awkward and `gh` does not exist.
  Have a `needs:`-chained `if: always()` job delete the label afterwards and it
  behaves like a button — re-add to re-run, including to retry a failure. Fork
  PRs get a read-only token, so this does not widen who can fire it.
  (`.github/workflows/repo-index-mirror.yml`)
- **`add_repo` cannot add a cross-owner repo, but `git clone` and
  `mcp__github__search_*` still reach it — and that is enough to build a
  PR-gold benchmark.** A session pinned to one owner gets
  `cross-tier adds are not supported in v1`, after which
  `api.github.com/repos/<other-owner>/*` 403s and `mcp__github__issue_read`
  refuses. What still works, unscoped: `git clone` of the public repo (full
  history), and `mcp__github__search_issues` / `search_pull_requests`, which
  return **full issue and PR bodies**. So mine gold from git rather than the
  API: a squash-merging repo (scikit-learn) puts each PR on one commit whose
  subject ends `(#NNNNN)` and whose diff *is* the PR diff. Recover issue
  bodies by bounding the issue number between neighbouring PRs' commit dates
  and date-windowing a search. Large search results are persisted to a file
  by the harness — parse them with a script instead of reading them into
  context, or paging a few hundred PRs will cost more tokens than the
  experiment. (`bekko-embedding-bench`)
- **A model card's OpenVINO-vs-ONNX-Runtime speedup does not survive a
  small-core box.** bekko-embedding-v1's card claims ORT is 5.5x slower than
  OpenVINO on x86 CPU; measured on 4 vCPU they are within 8% (20.1 vs 21.8
  chunks/s). Report `nproc` with any throughput number, and do not pick a
  runtime on a card's benchmark without re-measuring on your own core count.
  (`bekko-embedding-bench`)
- **Cloudflare AI Gateway throttles hard.** Start LLM batch concurrency at **2**,
  not 4 or 12. `phase-a-bridges` learned 12→4→2; `te-bridges` started at 4
  anyway and lost 18–20% of extractions to exhausted retries.
- **Gemini 2.5/3.x thinking models eat the whole output budget.** Set
  `thinkingConfig.thinkingBudget = 0` for structured-extraction calls or you get
  silent empty responses, not errors. (`phase-a-bridges/RESULTS.md`)
- **No apt ffmpeg.** `pip install imageio-ffmpeg` →
  `imageio_ffmpeg.get_ffmpeg_exe()` ships a static ffmpeg 7.0.2.
  (`dc-mall-timelapse/RESULTS.md`)
- **Headless Chromium TLS 1.3 gets RST by the egress proxy.** Launch with
  `--ssl-version-max=tls1.2` and manually load the proxy CA into the NSS DB
  (`certutil -A ... /root/.ccr/agent-proxy-ca.crt`) despite docs claiming it is
  pre-provisioned. (`atproto-pad-login/RESULTS.md`)
- **NFKD does not decompose stroke letters** (ł, ø, đ, þ, ß, œ). `normalize("NFKD")
  .encode("ascii","ignore")` turns "Odrzywołek" into "Odrzywoek" and breaks
  author-name matching. Use the pre-translation table in
  `te-bridges/scripts/te_common.py::ascii_fold`.
- **CCotw reaps idle background jobs.** Checkpoint long runs to disk.
- **GUI apps are testable in this container** — `Xvfb` is preinstalled, and
  `apt-get install -y xdotool libxkbcommon-x11-0` covers the rest. `winit` (and
  anything else on X11) panics with "Library libxkbcommon-x11.so could not be
  loaded" until that package is present; it is a missing runtime dependency, not
  a code bug. (`svgview/`)
- **Screenshot a virtual display with no ImageMagick and no x11grab**: start
  `Xvfb :99 -screen 0 WxHx24 -fbdir /some/dir` and it continuously dumps the
  framebuffer to `/some/dir/Xvfb_screen0` in XWD format. ~60 lines of `struct` +
  `zlib` converts XWD to PNG with no dependencies —
  `svgview/assets/xwd2png.py` is standalone and copyable.
- **`xdotool key` silently goes nowhere without a window manager.** Xvfb alone
  assigns no input focus, so every keystroke lands on the root window and the
  app under test never sees it. Fix: `xdotool windowfocus $(xdotool search
  --name <title> | head -1)` before sending keys. This cost a full round of
  green-but-meaningless test results — see the next entry. (`svgview/`)

**A check that cannot fail is not a check.** The GUI smoke test in `svgview/`
originally asserted "the screenshot contains at least N distinct colours" after
each keypress. It passed for all seven bindings while *none* of the keystrokes
were being delivered, because the assertion was equally true of the unchanged
frame. Two symptoms should have been read as failure and were not: every state
reported the identical colour count, and the pass came on the first try.

The generalizable form: an assertion whose truth does not *depend on* the thing
under test will report success on a no-op. Prefer differential checks — the
frame must change when it should, and a toggle pressed twice must return to the
byte-identical frame it started from — and when a fresh test suite passes
first-try, deliberately break the subject and confirm it goes red. This is
principle 1 (verify with a disjoint code path) applied to the *test* rather than
the result.
- **Semantic Scholar unauthenticated batch** works to ~1–2k papers with backoff;
  intolerable at 1.9M scale. Citations/references GETs are throttled ~50s per
  fail.
- **`sys.path` hardcoded to `/home/user/claude-workspace`** — 32 `.py` files
  across 13 experiments still point at the pre-migration layout, which no longer
  exists. See "Migration breakage" below.

## Numerical / ML gotchas

- **Measure the no-model baseline before concluding a small model earns its
  place.** Twenty ordered regex rules over structural cues routed an 18-tool
  Bluesky catalogue at **0.833** routable against 0.722 for a two-stage Cactus
  Needle router and 0.778 for Needle's *oracle* five-tool ceiling — at 0.022 ms
  against 324 ms, four orders of magnitude. On unseen queries from a different
  template family the rules held at **0.824**, so this is not simply
  overfitting. What the models bought was refusal (0.625 vs the regex's 0.183 on
  unseen data, where a catch-all fallback swallows everything off-topic) and a
  confidence score. Catalogues whose tools are distinguished by argument *shape*
  — handles, URIs, DIDs — hand a regex most of the task; catalogues distinguished
  by intent do not. (`monad-bsky/regex_only.py`)
- **A retry cascade beats any single accept-gate on the coverage/precision
  frontier.** Accepting on Needle's own confidence first, then on
  Needle/Monad agreement, then escalating, gave **0.613 coverage at 0.842
  precision** where the confidence gate alone gave 0.323 at 0.800 and agreement
  alone 0.435 at 0.889. Different tiers accept on different evidence, so the
  cascade keeps queries a single threshold discards. A third tier that re-asks
  with a rewritten query added 1–3 queries out of 24 and lowered precision.
  (`monad-bsky/cascade.py`)
- **Constraining a fine-tuned generative model to score a fixed label set can
  cost more than the hallucinations it prevents.** Scoring Monad's 18 declared
  tool names by `log P(name)` and taking the argmax makes undeclared names
  unreachable and yields a softmax confidence, but scored **0.241** routable
  against **0.481** for letting the same model generate freely — top-3 only
  0.407. Verified against a harness bug with independent full forward passes
  sharing no KV cache. A model trained to emit a derivation then a call does not
  carry its decision in the marginal at one token position; a decode grammar
  constrains the *path* and preserves generation dynamics, label scoring does
  not. (`monad-bsky/classifier.py`)
- **A model that corrupts identifiers cannot be the stage that restates the
  request.** Fine-tuned Monad mangles handles inside its own `<think>` trace
  (`simonwillison.net` → `simanwillander.net`) before any structured output is
  produced, so an english → small-model → router pipeline hands the router a
  corrupted string to copy faithfully. Check whether a proposed rewriter
  preserves literals before designing around it. (`monad-bsky/RESULTS.md`)

- **Two small models agreeing is a better accept-gate than one model's
  calibrated confidence head, at the same coverage.** Cactus Needle's two-stage
  router and a fine-tuned Monad independently naming the same tool were right
  **0.880** of the time over **0.455** coverage; Needle's own confidence head
  reached only 0.741 at 0.491 coverage and needed to drop to 0.236 coverage to
  match 0.846. The signals compose — agreement plus confidence ≥ 0.4 gives 0.929
  at 0.255. Two properties make this worth reaching for: agreement needs **no
  confidence head**, which matters because fine-tuning destroys Needle's, and the
  two signals are close to independent (a post-hoc score over one model's logits
  versus a second model trained on different data under a different objective).
  The cost is running both models — 11x latency here — so it suits batch or
  high-stakes calls, not a phone. (`monad-bsky/synergy.py`)
- **A calibrated confidence score does not transfer to another model's answers
  on the same query.** Needle's head separated its own correctness (mean 0.584
  right / 0.392 wrong) and was flat-to-inverted on Monad's (0.486 / 0.532),
  getting *worse* at higher thresholds: at confidence ≥ 0.8 Needle was 0.867
  accurate and Monad 0.400. Calibration is a property of the model that emitted
  it, not a difficulty score for the input. Do not reuse one model's confidence
  to gate another's output. (`monad-bsky/synergy.py`)
- **Price a hallucinated-name rate in accuracy points before building a grammar
  for it.** Fine-tuned Monad invented undeclared tool names on 14.5% of queries,
  but snapping them to the nearest declared name by edit distance fixed only two
  queries (+3.7 points): most invented names sat on queries that were misrouted
  anyway. A constrained decoder is still the right call for other reasons —
  malformed output, unbounded values — but the accuracy it recovers can be a
  fraction of the violation rate. (`monad-bsky/synergy.py`)

- **A small model's failure to fill tool arguments is usually transcription, not
  routing — measure the two separately before blaming the model's reasoning.**
  Fine-tuned Monad (56M) picked a defensible tool far more often than it
  reproduced the identifier the request contained: over 41 eval arguments that
  appear verbatim in the query it emitted the right string **51%** of the time,
  against **78%** for Cactus Needle's untuned base and **90%** for its LoRA.
  `austegard.com` came back as `afethew.com`, `jetstream` as `jetforek`. Score
  "did it copy the span" as its own metric; a combined arguments-correct number
  hides which half is broken. (`monad-bsky/copy_probe.py`)
- **Do not attribute a copying failure to tokenizer vocabulary without measuring
  both tokenizers.** The obvious story — Monad's 8,192-piece prose vocabulary
  shatters handles and DIDs — was written into a draft and is false: Cactus
  Needle carries **the same 8,192 pieces** and segments the same strings
  identically (`austegard.com` is `['a','ust','eg','ard','.','com']` in both;
  111 vs 109 pieces over ten identifiers). The cause that survives is the
  training objective, and the evidence is that Needle's *base* weights, never
  exposed to the experiment's data, already copy at 0.780.
  (`monad-bsky/ERRORS.md` #1)
- **More fine-tuning made verbatim copying worse while training loss kept
  falling** — 0.561 at one epoch to 0.512 at three, train loss 0.0001. Fitting
  the training set's identifiers is not the same operation as transcribing an
  unseen one, and nothing in the loss distinguishes them.
- **A validation split drawn from your own generator measures template fit, not
  generalisation.** Monad's val loss improved 7.5x across three epochs (0.0128 →
  0.0040 → 0.0017) while eval accuracy went 0.389 → 0.481 → 0.444. If the
  held-out rows come from the same templates as the training rows, the curve
  will look excellent regardless. Judge checkpoints on the task eval.
  (`monad-bsky/ERRORS.md` #5)
- **An unconstrained decoder invents tool names, at a rate worth pricing.**
  Fine-tuned Monad emitted names that were never declared (`get_posts`,
  `get_replies`, once `get_spammer.bsky.social`, assembled from the query's own
  handle) on **6.5% of queries after one epoch and 14.5% after three** — rising
  with training. A grammar compiled over the declared names makes this
  impossible rather than unlikely, which is a concrete reason to prefer a
  constrained decoder over a larger model. (`monad-bsky/RESULTS.md`)

- **A small-model fine-tune can be a wash on accuracy and still cost you the
  calibration head — check whether your serving stack keeps the head before
  spending the compute.** Cactus documents that fine-tuning does not update
  Needle's confidence head, so a tuned agent reports `confidence: None` and warns
  once at construction. Measured: 800 templated rows, LoRA r16, 3 epochs, ~2 h of
  4-core CPU moved routable top-1 from 0.611 to 0.667 (paired McNemar **p=1.0**),
  regressed off-topic refusal from 0.625 to 0.375, and left the two weakest
  categories — both explicitly covered by the training templates — at exactly
  their base scores. The gate it removed was worth more than the accuracy it did
  not add. (`needle-bsky/RESULTS.md`)
- **A confidence head that never clears any usable threshold is the answer, not
  a missing result.** Needle 2's extraction over free-form social text scored
  below 0.05 on all 22 attempts across three schema shapes (max 0.0434), while
  the same model's routing on the same corpus reaches 1.000 precision at 0.9. Two
  capabilities of one model can have completely different operating points; test
  for the operating point rather than eyeballing outputs. Related: fields that
  ask for a summary (`subject`: "the main thing the post is about") fall outside
  a span-copying extraction contract by construction, and get filled with
  arbitrary nearby spans rather than refused. (`needle-bsky/extract_demo.py`)

- **A small tool-calling model is far worse at picking a category than its own
  retrieval head is at picking a tool.** Splitting an 18-tool catalogue into five
  groups of ≤5 and asking Needle 2 to choose the group first scored **0.370**
  routable top-1, 24 points *below* just declaring all 18 (0.611) — its errors
  were systematic, not noisy (9 of 15 account queries landed in one wrong group, 3 were right).
  The same split with ~20 lines of regex over structural cues as stage 1 scored
  **0.722**, above the flat arm and most of the way to the five-tool oracle
  ceiling (0.778). These models are trained to map a concrete request onto a
  concrete callable; a group description is not one. If a design needs a
  pre-filter, make it deterministic, and remember a stage-1 error is
  unrecoverable — the right tool is then absent from stage 2 entirely.
  (`needle-bsky/two_stage.py`)

- **A grammar-constrained router's confidence head scores the whole call, so an
  optional argument you declared but the query never licensed poisons the score
  of an otherwise-correct routing decision.** Cactus Needle 2 emitted
  `get_user_posts(handle='pfrazee.com', limit=10)` for "what has pfrazee.com
  been posting lately" — right tool, right handle, invented `limit` — at
  confidence **0.0004**. Dropping every non-required argument from the same 18
  schemas cut the invented-argument rate from 0.518 to 0.222 and raised mean
  confidence on correct calls from 0.309 to 0.584, which is the difference
  between a gate that can trade coverage for precision (38% coverage at 0.762,
  20% at 0.909, 13% at 1.000) and one that reaches high precision only by
  keeping one call in fifty. Routing accuracy itself barely moved (p=0.11), so
  the reason to trim arity is the **gate**, not the accuracy. Declare only the
  arguments you need the model to fill. (`needle-bsky/RESULTS.md`)
- **`loss 0.0000` at step 5 of a fine-tune is truncation, not an easy task.**
  Training rows that declared all 18 tool schemas tokenized to 1,642 tokens
  against `--max-len 1024`; the target follows the tool block, so every label
  position was cut and masked. Before believing any fine-tune loss, tokenize the
  longest row with the trainer's own tokenizer and compare against `max_len` —
  it is one command and it distinguishes "converged" from "learned nothing".
  (`needle-bsky/ERRORS.md` #1)
- **Match the training context to the decode context when the serving stack does
  retrieval.** Needle renders at most five tools per turn, chosen by a retrieval
  head, so a training row declaring the full catalogue trains on a context the
  model will never see at inference — and is what blew the token budget above.
  Generating rows with the correct tool plus four distractors fixed both at once.
  (`needle-bsky/gen_data.py`)
- **Declaring a sixth tool to Cactus Needle costs a fixed ~750 ms per turn; the
  seventh through eighteenth are free.** Median turn on 4 CPU cores: 284 ms at 5
  tools, 1034 ms at 6, 1109 ms at 18. Retrieval engages above five and embeds
  the query every turn. `tool_index_path` does not help — it caches the *tool*
  embeddings computed once at init, not the per-turn query embed (measured:
  1090–1124 ms with the index, 1150 ms without). Size agent catalogues at five
  or accept the flat penalty; there is no gradient in between.
  (`needle-bsky/results_latency_vs_catalogue.json`)
- **Retrieval, not selection, is where a small router loses most of its
  accuracy.** Giving each query its own five-tool catalogue containing the right
  answer lifted routable top-1 from 0.611 to 0.778 (tuned-min) and 0.704 to
  0.815 (tuned) over the same 54 queries — roughly a third of the remaining
  errors were the right tool never entering the context. If a router
  underperforms, measure the oracle-retrieval arm before rewriting the selector's
  prompts. (`needle-bsky/oracle.py`)

- **Sorting texts by token length before batching a transformer encode is
  bit-identical, not merely close — and worth ~1.37x, not the 2x the intuition
  promises.** Padding is masked out of attention, and last-token pooling indexes
  `mask.sum(-1) - 1`, so batch composition cannot reach the output: measured
  **max absolute difference 0.0** across 48 docs between length-sorted and
  source-order batching, and again across batch size 8 vs 16
  (`ttt-embed-quantized/encode.py --parity-check`). Because it is exact rather
  than approximate, it needs no accuracy budget and can be applied to artifact
  encodes that must stay comparable to earlier runs. But **measure the speedup
  rather than reasoning it out**: "halve the padding, halve the time" predicts
  ~2x; a 240-doc A/B gave **1.37x** (4.3 → 5.8 docs/s), because attention is not
  the whole cost and the sort only helps within a batch. Corollary for progress
  logs: with length-sorted batches the early throughput reading is a large
  overestimate (14.7 docs/s falling to a 5.9 docs/s final average here), so an
  ETA computed in the first minute of a sorted run is systematically optimistic.
  Independently, **plain mini-batching is bit-exact too** — batch 32 vs batch 4
  on this encoder gave `max|Δ| = 0.000e+00` across all 256 dims, upgrading
  `q4-official-vs-ours`' first-principles argument (it mini-batched to dodge a
  ~26 GB attention-mask `Expand` OOM) to a measurement. **Batch size is a pure
  throughput knob on a masked, last-token-pooled encoder**, so an artifact never
  depends on it. Cheapest possible guard: encode one batch two ways and diff.
  (`ttt-embed-quantized/RESULTS.md`)
- **Two independent implementations of this encode produced bit-identical
  vectors wherever the input strings were identical** — 4,128 of 5,183 documents
  and 300 of 300 queries matched to the last bit across two containers, two
  authors and two loaders, with **zero** cases of same-string-different-vector.
  Worth knowing before budgeting for nondeterminism: jina-v5-nano q4 on ONNX
  Runtime CPU is reproducible, so any diff between two runs of it is a diff in
  the *inputs*, and can be chased as one. Do not attribute an embedding
  mismatch to float nondeterminism without checking the strings first.
  (`ttt-embed-quantized/RESULTS.md`)
- **"The RHT is faster" is a claim about a specific implementation at a specific
  d — check which function you are actually calling.** remax's
  `remax.rotation.rht_rotation` at its floored rounds=2 reproduces its documented
  1.5-1.8x over Haar QR (measured 1.70x / 2.06x / 1.65x at d=256/768/1024), but
  `remax_kb.projection.srht_matrix` is a **different** implementation and is
  **1.4-3.0x slower than Haar** at every dim measured — and it is remax_kb v2's
  *default*. That is deliberate, not a defect: srht is seed-only and bit-for-bit
  reproducible by a non-NumPy reader, where Haar's PCG64 + Ziggurat + LAPACK QR
  is not, and a mismatched projection flips ~50% of code bits. Projection choice
  is a **portability** decision; do not reach for it to fix latency. Rounds
  matter too (rht r=2 31 ms vs r=3 43 ms at d=256), and remax floors at 2 for a
  measured reason. Retrieval was neutral across haar/rademacher/srht (R@10 spread
  <=0.022, straddling fp32), so quality does not break the tie either.
  (`bekko-embedding-bench/RESULTS.md` §7)

- **A seed-derived rotation reconstructed per query can dominate query latency —
  and this is the third time it has bitten in this repo.** A hybrid index's query
  measured ~400 ms, of which 270 ms was charged to "decode". The actual decode of
  858 vectors is 22 ms; **200 ms was rebuilding a Haar rotation** (Householder QR,
  O(d^3) at d=384) on every single query. Prior instances: `remax_kb`'s
  `_stacked_simhash_encode` rebuilding k rotations per query (87% of query time,
  `bekko-embedding-bench`), and `repo-index` regenerating its rotation from seed
  (fixed by committing `rotation.npy`). Two fixes, and the cheaper one is not the
  obvious one: persisting a d=384 Haar matrix costs 576 KB, whereas
  `rotation="rht"` **builds in 6.2 ms instead of 171 ms and costs nothing** — and
  remex measured RHT as retrieval-indistinguishable from Haar (-0.0001 +/- 0.0013,
  experiments#11). Prefer the cheap construction over the stored matrix when the
  library offers one. Check where the time actually goes before believing a
  profile label: "decode" was 90% not-decode.
  (`hybrid-code-index/RESULTS.md`)
- **In a hybrid index the lexical arm is usually the bigger artifact.** BM25
  postings for `remax` are 245 KB against 93 KB of remex 2-bit dense codes — 2.6x
  — and postings scale with *vocabulary*, not chunk count, so a corpus with
  hashes, IDs or floats in it explodes them (138,685 terms / 6.36 MB on a
  JSON-heavy corpus where the dense side stayed at 1 MB). If an index is too
  large, look at the lexical side first; the intuition that embeddings are the
  expensive part is wrong here. **Correction to an earlier version of this entry:** BM25 *is*
  incrementalizable. Its idf is derived at query time from `len(postings[t])`
  and `n`, so there is nothing precomputed to invalidate when a document is
  added or dropped; the only reusable work is tokenization, cacheable by content
  hash. The point stands anyway for a different reason — fitting BM25 is 0.3 s
  against a 36 s dense encode, under 1% of build time, so it is not where
  incremental effort pays.
  (`hybrid-code-index/RESULTS.md`)
- **A relocated file is indistinguishable from a deleted one by path — detect
  moves by content.** `git log --diff-filter=D` reports a moved file as deleted
  at its old path. Building a "deleted content" corpus from that put **live**
  content into the gone-corpus: 5 of 21 `remax` files had been relocated
  (`src/remax/bench/*` → `bench/*`), inflating the corpus 27% and manufacturing a
  false positive where a *live* `crossover.py` satisfied a query whose gold was
  its deleted path. Detect by content overlap (>50% of non-trivial lines present
  in some live file), not by path or basename — basename matching would wrongly
  drop `src/remax/bench/__init__.py`, which is genuinely gone. General form:
  **when a control arm scores on a query only the treatment should be able to
  answer, suspect the answer key before believing the arm.** That anomaly, not
  inspection, is what surfaced this.
  (`history-tombstone-index/RESULTS.md`)
- **Cost the baseline before optimizing against it.** A per-repo routing tier
  was built and tuned over three 18-minute encodes so a client could fetch only
  the partitions it needs — then one command showed the *entire* 9-repo account
  index is **8.54 MB** (2.37 MB dense + 6.17 MB postings, 25,904 chunks
  including scikit-learn) against a **157 MB encoder the client already
  downloads**, with a flat scan over everything costing **7.6 ms**. The
  optimization targets something 18x smaller than an existing required
  download. Before measuring how well a shortcut works, measure what the
  un-shortcut path costs; "fetch everything" is often the correct design at
  personal-account scale and only stops being so past ~1M chunks. The failure
  is seductive because the shortcut's *own* metrics look like progress —
  recall@k improved across four card designs while the whole exercise was moot.
  (`account-routing-tier/RESULTS.md`)
- **A README states a project's identity, not its inventory — do not build a
  router from front matter.** Routing queries to the right repo by embedding
  READMEs failed on scikit-learn: its `README.rst` contains **zero** occurrences
  of "gradient boosting", "one-hot", "cross validation", "sparse" or "estimator"
  while its tree has 296 files matching "sparse" and 311 matching "estimator" —
  it is badges, install instructions and links. A card built from repo-level
  tf-idf terms plus module paths beat the front-matter card at every k **with
  2.5x fewer chunks**. The trap is that the front-matter card *looks* obviously
  right, and a related real bug (the card list held only `README.md`, so a repo
  shipping `README.rst` had no README at all) supplies a plausible wrong
  explanation for its failures — fixing that moved recall@1 47% -> 43%.
  (`account-routing-tier/RESULTS.md`)
- **Ranking a group by its best member rewards having more members.** Splitting
  large repos into several routing cards, to fix a genuine 350x capacity
  imbalance, made ranking *worse* at low k (recall@1 53% -> 47% for 3.7x the
  cards) because a repo's score was the max over its cards: more cards is more
  draws from the score distribution, so a split group's maximum rises for
  reasons unrelated to relevance. Any per-group aggregation over a variable
  number of items needs a count correction; `max` has none, and neither does
  "best chunk per file" when files differ wildly in length.
  (`account-routing-tier/RESULTS.md`)
- **Three corpora, three question classes, and they are orthogonal — a corpus
  aimed at the wrong class is inert, not harmful.** Measured on `remax`: the
  working tree answers *what does the code do*, deleted-file tombstones answer
  *how did the removed thing work*, and PR bodies answer *why was it done that
  way*. Adding tombstones to a rationale benchmark moved nothing (6/8 -> 6/8);
  adding PR bodies took it to 8/8. Adding tombstones to a mechanism benchmark
  took 0/6 -> 6/6. So they stack without the arms fighting, and each should be
  justified against the question class it serves rather than a pooled score that
  hides which one is carrying. Cost is small — 43 merged PRs is +12% corpus, and
  17 deleted files +19% — but PR bodies are **not in git**, so indexing them
  makes network and a token a hard dependency of a rebuild that is otherwise
  pure filesystem. Two conditions before believing this transfers: check the
  repo's *median PR body length* (remax's is 2,727 chars because every PR is
  agent-authored; a repo of "fixes #12" bodies gets noise with a title attached),
  and have the build degrade to the offline corpora rather than fail.
  (`pr-decision-log/RESULTS.md`)
- **A corpus that does not come from your file walker must re-implement your
  file walker's exclusions — or it will index exactly what you configured out.**
  Adding a tombstone corpus (deleted files, recovered at their last living
  revision) to the account index first measured **74,822 chunks against a
  13,257-chunk working tree**: 1.7x the entire live index, from three repos. A
  deleted path gets no `stat()` and no `rglob`, so extension, `skip_dirs`,
  `skip_names`, the repo's own `exclude` list and the size cap all silently stop
  applying, and a 767,692-line deleted embedding dump enters a corpus the live
  index refuses at 1 MiB. Filtered, 47x smaller. The same trap is waiting for any
  corpus assembled from git history, an API, or a database rather than from
  `discover()`. Extract the predicate and call it from both.
  (`account-index-corpora/RESULTS.md`)
- **A per-repo corpus result does not survive the move to account scale
  unchanged — the same corpus can invert.** Tombstones measured +19% and 0/6 ->
  6/6 on `remax`, a library whose deletions are hand-written source. Account-wide
  they are +11.8% nominal but **~94% deleted machine-generated data**:
  claude-workspace contributes 1,484 tombstone chunks against a 232-chunk working
  tree, of which ~23 are prose or source. A repo that generates data files and
  later deletes them has a deletion record dominated by them, and no per-repo
  measurement on a library-shaped repo can see that. Two mechanisms only visible
  at account scale: relocation detection must compare **across** repos (a
  migration deletes in one and lands in another — 540 files here), and candidate
  matching must be restricted (by basename) or the guard is O(dead x live).
  PR bodies do transfer: +3.2%, and the median-body-length condition holds.
  (`account-index-corpora/RESULTS.md`)
- **A token that can clone a repo cannot necessarily read its pull requests, and
  the build will not notice.** Indexing merged PR bodies account-wide fetched 53
  of 65 repos; the other 12 returned `HTTPError` on `/pulls` while cloning fine
  in the same run, because the PAT carries contents read but not pull-request
  read on them. The corpus came out ~18% short with every other signal green.
  Two things made it visible rather than silent: the fetch degrades per repo
  instead of failing the run, and it emits a `::warning::` naming the count. Any
  corpus assembled from an API needs both — degrade *and* announce — because a
  smaller corpus still builds, still verifies, and still answers.
  (`account-index-corpora/RESULTS.md`)
- **`--depth N` on a clone buys an arbitrary fraction of the deletion record,
  not a cheap one.** `git log --diff-filter=D` sees only the grafted window, so
  shallow-clone deletion coverage is a function of the repo's commit rate rather
  than of anything chosen: `--depth 50` found **2 of 18** deletions in one repo
  and 640 of 644 in another. It is also not saving anything — depth 1, depth 50
  and a full clone measured 6.5 s / 7.3 s / 7.4 s summed over three repos, with
  no consistent sign, because fixed per-clone overhead dominates the extra
  bytes. If history is needed, take all of it; if it is not, take depth 1. The
  middle is a number that looks tuned and is not.
  (`account-index-corpora/RESULTS.md`)
- **A build gated on "what changed" will not notice that its own configuration
  changed.** The account index skips every step when no repo's `pushed_at` has
  moved, which is what makes a quiet run cheap — so a dispatch that turns a new
  corpus on hits the early exit and reports success having rebuilt nothing. Any
  content-derived skip check has to diff the *build settings* alongside the
  inputs. Same shape one layer down: in a sharded build, three processes rebuild
  the corpus independently and match rows by content hash, so a setting passed
  as a per-job flag rather than carried in the shared plan artifact does not
  error when it diverges — it silently re-encodes everything.
  (`account-index-corpora/RESULTS.md`)
- **Writing your rejections down preserves the verdict, not the mechanism.** A
  repo with an explicit "delete the driver, never the record" convention
  (`remax`) answered 5/6 "was X ever tried?" questions from its surviving prose
  alone — and **0/6** "how was X implemented?". A writeup saying "the encoder,
  its CSR-builder and a BEIR benchmark were all built" tells you the thing
  existed and lost; it does not carry the signature, the batching, or what the
  tests asserted. Indexing deleted files alongside the working tree scored 12/12
  where the tree alone scored 5/12, for +19% corpus. So the discipline is worth
  keeping *and* is not a substitute — and if you want the mechanism to survive a
  deletion, the writeup has to contain the code, not describe it.
  (`history-tombstone-index/RESULTS.md`)
- **More retrieval arms is not monotonically better — RRF is unweighted, so a
  weak arm votes as loudly as a strong one.** Fusing dense + stored-BM25 scored
  24/24; adding ripgrep as a third arm dropped it to **22/24**. Fusion helps only
  when the arms fail on *different* queries and each is individually credible;
  an arm whose ranking is near-noise on one query class (rg on "find a file like
  this one": 3/9) actively drags the consensus. Test each arm alone before
  fusing, and add an arm only if the fused score improves — not because more
  signal sounds better. Related: **ripgrep cannot be a fusion arm for similarity
  queries at all**, because it returns a set rather than a ranking, and counting
  matched terms does not recover a ranking it never produced.
  (`hybrid-code-index/RESULTS.md`)
- **Volume does not predict corpus pollution — similarity to real queries does.**
  Two dilution cases in the same tool, opposite outcomes: 173 `run_NN.md` model
  generations at **20%** of the corpus measurably crowded out real answers
  (keyword agreement 8/10 → 10/10 once excluded), while generated `.json` results
  data at **79%** of the corpus was completely **inert** (24/24 fused, with and
  without). The difference is that the model output was topically on-subject
  *prose* competing directly with real answers, whereas JSON is lexically alien —
  the encoder maps it somewhere no natural query goes. So do not reach for a
  build-time exclusion on size or share; measure whether the content is
  *reachable* by the queries you serve. It is not free either way: the lexical
  arm's postings inflated 1 MB → 6.36 MB (138k terms) on the JSON.
  (`hybrid-code-index/RESULTS.md`)
- **Incremental indexing is exactly equivalent to a full rebuild only for
  components that are not fitted on the corpus.** Content-hash chunk reuse gave a
  **bit-identical** matrix (max abs delta 0.000e+00) at 0.2 s against a 537 s full
  build — 2735x — because the encoder is per-chunk independent and remex
  quantization is data-oblivious. **BM25 breaks this**: IDF shifts for every term
  whenever any document is added, so the lexical arm must be refit wholesale
  (affordable at 5.2 s, but not incrementalizable). Same for PCA, k-means, ITQ,
  and PQ codebooks. Before building an incremental path, ask which components
  depend on corpus statistics — those are the ones where "incremental" silently
  means "approximate". Note also what incremental does *not* fix: a **committed**
  index is a binary blob stored whole per commit (1.00 MB codes + 6.36 MB
  postings here, ~1.5 GB of history at 200 rebuilds), so cheap rebuilds and cheap
  history are separate problems with separate fixes.
  (`hybrid-code-index/hcindex.py::incremental`)
- **Documenting a failure mode does not prevent recurrence — only structure
  does.** The "evaluation harness sits inside the corpus it measures" bug has now
  occurred **four times** in this line of work: `code-index-duplication` (the
  harness retrieved itself for 4 of 9 NL queries, worth 2 points of hit@5),
  `hybrid-code-index`, `pr-decision-log`, and `account-routing-tier` — the last
  two *after* it was diagnosed, written up here, and explicitly guarded against
  in an earlier harness. Each new harness is a fresh opportunity to forget. The
  fix that holds is making self-exclusion the default in the corpus builder
  (`hcindex.build_corpus` excluding the caller's own directory) rather than
  something every harness must remember to pass. Generalise: if a rule has to be
  re-applied by hand at each new call site, expect it to be missed at some of
  them, and move it into the callee.
  (`account-routing-tier/RESULTS.md`)
- **A baseline scoped to a narrower corpus than the system under test reports
  improvements as regressions.** After `repo-index` was extended from `.md` to
  `.md`+`.py`, its keyword benchmark fell 10/10 → 7/10 and looked like a clear
  regression to revert. All three "regressions" were the index returning the
  **definition** instead of a prose mention — `ascii_fold` → `_lib/textnorm.py`,
  `GRID_VERSION` → `remex-vs-higgs-ablation/grids.py` — and the grep arm was
  still pinned to `--glob '*.md'`, so a correct `.py` answer could not score.
  Matched arm: 9/10. Whenever the system's corpus, file types, or candidate set
  changes, re-scope the baseline in the same commit; a frozen baseline silently
  becomes a different question. Related trap in the same tool: a *harness that
  embeds its own queries* joins the corpus it searches — `code-index-duplication/
  run.py` contains all nine NL queries verbatim and retrieved itself in the top 5
  for 4 of 9, worth 2 points of hit@5. (`code-index-duplication/RESULTS.md`)
- **Report the configuration the tool ships, not the one the harness finds most
  flattering.** The same experiment first scored 9/9 excluding only the query
  *file* — but in real CLI use, same-directory neighbours filled every result
  slot, so `--file` excludes the query's whole *directory*. The harness was
  changed to match before any number was written down. Headline was unchanged
  here; the discipline is what matters, because the gap between "what I measured"
  and "what it does" is invisible in the writeup unless you go looking.
  (`code-index-duplication/RESULTS.md`)
- **A semantic index over a repo that stores model output will rank that output
  against real questions — exclude generated directories.** `repo-index` indexed
  `haiku-assessment/**/outputs/` and `**/prompts/`: 173 `run_NN.md`-shaped
  near-duplicate generations, **20% of the corpus**. They were not inert filler.
  Agreement with grep on ten keyword-bearing queries was 8/10, and *both* misses
  were identifier lookups (`ascii_fold`, `GRID_VERSION`) answered with a sample
  of LLM output instead of the file that defines the thing; vague queries would
  return five near-identical `run_0N.md` chunks in a row. Excluding those
  directories: **8/10 → 10/10**, rediscovery cases unchanged at 5/5, index 27%
  smaller. Lexical search never had this failure — nobody greps for a phrase
  that only appears in a sampled generation — so it is easy to carry an
  "index everything" habit across from grep and not notice. Sweep for
  `outputs/`, `prompts/`, `runs/`, `samples/` before building any embedding
  index over an experiments repo. Corollary for evaluation: this was invisible
  to the 5 hand-written rediscovery queries (5/5 both before and after) and only
  showed up on queries whose answer is a *specific* artifact — vague-query
  benchmarks cannot detect corpus pollution. (`repo-index/ask.py::GENERATED`)
- **A stored index that regenerates its transform from a seed inherits every
  upstream default it did not pin — store the transform instead.** `repo-index`
  originally rebuilt its remex rotation from `seed=0` at query time and only
  *fingerprinted* it, justified by "numpy's LAPACK QR drifts across BLAS
  builds". That justification was already stale: remex#40 had replaced
  `np.linalg.qr` with an explicit Householder QR precisely to be bit-reproducible
  across BLAS builds. The real exposure was three other things having to stay
  still — remex's `rotation=` **default** (which remex documents as deliberately
  changeable, and the call site did not pass one), numpy's `default_rng` stream
  (NEP 19 declines to guarantee it across feature releases), and remex's
  construction of the matrix (which #40 is itself proof can change). Decoding
  under a wrong rotation is ~50% of bits different — total and silent, not
  degraded. Two rules: **pass the rotation explicitly, never inherit the
  library's default** (free), and **persist the matrix next to the codes**
  (576 KB at d=384 f32, against a 182 KB index — a 4x artifact increase that is
  still nothing in a git repo). A fingerprint only converts a silent failure into
  a warning plus a full rebuild; storing prevents it. Generalizes to any
  seed-derived artifact — rotations, projections, permutations, codebooks: if
  you felt the need to build a detector for a regeneration mismatch, that is the
  signal to store the thing instead. (`repo-index/README.md`)
- **An iso-byte retrieval comparison prices storage and silently assumes compute
  is free — for a small encoder that is the whole comparison you are missing.**
  bekko-embedding-v1-a8m lost to jina v5 nano q4 in 11 of 12 iso-byte cells,
  which read as "don't switch" until latency was measured: **11.3 ms vs 146.4 ms
  per query on 1 vCPU (12.9x), 11.2x tokens/s**. The gap is architectural
  (4 layers x 384 hidden x 1152 FFN against 12 x 768 x 3072, ~12x the per-token
  FLOPs) and the measured ratio matching the FLOPs ratio is what rules out a
  quantization artifact. A model advertised by *active* parameter count is making
  a compute claim, so benchmark compute or you have not tested its thesis. Report
  both **texts/s and tokens/s** when tokenizers differ (256k vs 128k vocab gives
  different token counts for the same text), and measure **batch=1 at 1 thread**
  separately from batched throughput — the query path and the index-build path
  are different products. (`bekko-embedding-bench/RESULTS.md`)
- **At a fixed byte budget, quantize wide rather than truncate narrow.**
  Matryoshka truncation and scalar quantization are orthogonal axes, and on a
  179-chunk retrieval task they are *not* equally good ways to spend bytes:
  remex d=384 @ 2-bit costs **96 B at R@10 0.609** against full fp32 d=384's
  **1536 B at 0.598** — 16x smaller and not worse — while remex d=384 @ 1-bit
  matches fp32-truncated-to-128 at **48 B vs 512 B**. Truncation lost at every
  budget compared. Keep the coordinates, drop the bits.
  (`bekko-embedding-bench/RESULTS.md`)
- **The one-bit-beats-two inversion is a property of the encoder, so test it
  per encoder — never inherit it.** 1-bit beat 2-bit on SPECTER2 and inverted
  on Jina; on bekko-embedding-v1, **2-bit beats 1-bit in all 8 (variant x dim)
  cells**, by up to 0.129 R@10 and largest at the narrowest width. A `.kb`
  built on a new encoder must re-measure the bit ladder before inheriting the
  1-bit default. (`bekko-embedding-bench/RESULTS.md`)
- **int8-ing only the static token-embedding table is ~free, and that is a
  vocab-size fact, not a quantization insight.** bekko's vocab is 256,000 x 384
  = a 98 M-param table that is essentially the entire 404 MiB fp32 export;
  int8ing it alone gives 404.3 -> 124.1 MiB at per-doc cosine **0.99989** to
  its own fp32, holding on both a prose and a code distribution. Mirror image
  of the `MatMulNBits` gotcha above: there the table stayed fp32 and *inflated*
  naive int4. Check where a model's parameters actually live before choosing a
  quantization recipe. (`bekko-embedding-bench/RESULTS.md`)
- **Mismatched random rotation matrices collapse recall to chance**, not
  graceful degradation. Two *different* valid orthogonal projections on doc vs
  query side flip ~50% of sign bits: recall 0.78 → 0.005. Only int8-rounding of
  the *same* matrix is safe (0.24% bit-flip). Any cross-language/cross-process
  LSH scheme must generate the matrix bit-identically. (`kb-k-sweep/RESULTS.md`
  Part 7)
- **Per-tensor dynamic int8 is domain-fragile.** Fine on tech text (0.83
  cosine-preserving R@5), collapsed on medical abstracts (0.445 per-doc cosine
  to fp32). Blockwise int4 stayed at 0.975 across both *while being smaller*. A
  single-domain smoke test hides this. (`jina-int8-remax_kb/RESULTS.md`)
- **`MatMulNBits` only touches MatMul nodes**, so a large vocab embedding table
  stays fp32 and naive int4 (465MB) ends up *larger* than whole-graph int8
  (212MB). Apply `MatMulNBitsQuantizer` first, then `quantize_dynamic` to mop up.
- **One-shot `encode(all_docs)` OOMs at ~26GB** on the attention-mask `Expand`
  broadcast for a few-thousand-doc corpus. Mini-batch the forward pass.
- **`trust_remote_code` models break on transformers bumps.** LFM2.5-Embedding's
  pinned remote code predates the `seq_idx` kwarg. Monkeypatch the *bound*
  method after model load to swallow unexpected kwargs — editing cached
  remote-code files is not viable. (`lfm25-embedder-remax_kb/lfm25_embedder.py`)
- **Small models follow example central-tendency over stated rules.** Audit
  whether your examples demonstrate the constraint, not just whether you wrote
  the rule. (`haiku-assessment/GUIDE.md`)
- **"Be specific" without "specific about only what's stated" causes confident
  fabrication** — 19/20 in one probe. (`haiku-assessment/GUIDE.md`)
- **Lloyd/LBG from a random init converges to grids WORSE than the scalar
  quantizer at high rate.** Seed it from the product of optimal scalar
  quantizers instead. Trained on N(0,I₂) at 6 bits, random-init Lloyd gave
  held-out MSE/dim 0.000828 against scalar Lloyd-Max's 0.000644 — 29% worse,
  and 87% worse at 8 bits. More samples do not fix it (48 → 1,953 per
  codepoint only moved 0.000854 → 0.000769) and neither does a lattice init: a
  tuned A2 hexagonal ball scored 0.000916, because a *uniform-density* lattice
  is the wrong construction for a Gaussian at fixed rate — optimal point
  density goes as f^(m/(m+2)), which scalar Lloyd-Max already has and a
  lattice ball does not. The fix is one line: `product_init(bits, m)` has
  exactly (2^bits)^m points, so it is a legal starting codebook, and Lloyd
  from there is monotone. Keep the unrefined product grid as a *candidate*
  too, not just as an init — see the next entry.
  (`remex-vs-higgs-ablation/grids.py`)
- **"Lloyd is monotone, so refinement can't be worse" is false as usually
  coded.** Lloyd is monotone in *training* distortion; if you select or report
  on *held-out* distortion, refinement really can land worse than its own
  starting point — the train/held-out gap reaches ~14% at 61 samples per
  codepoint. If you want the bound "never worse than X", X must be in the
  candidate set and scored on the same held-out stream as everything else.
  Select and report on *different* seeds while you are at it.
  (`remex-vs-higgs-ablation/grids.py::train_gaussian_grid`)
- **The Lloyd-Max distortion identity MSE = 1 − Σpᵢyᵢ² is only valid AT the
  fixed point.** Recomputing the cell probabilities from freshly updated levels
  evaluates it slightly off the fixed point. Harmless where Lloyd has
  converged; at 8 bits it has not (20k iterations still leave
  max|level − centroid| ≈ 5e-6) and the identity returned 4.791e-5 against a
  true 4.127e-5 — **16% high**. Integrate the distortion directly. Note the
  failure direction: an inflated scalar baseline makes a "the fancy method
  beats scalar" gate *more permissive*, and Max (1960)'s published table stops
  at 5 bits so a table-comparison check cannot catch it.
  (`remex-vs-higgs-ablation/grids.py::lloyd_max_1d`)
- **Score cosine as cosine: divide by ‖x̂‖.** Ranking quantized documents by
  the bare inner product `q·x̂` silently rewards codecs whose reconstruction
  norm happens to be constant and penalises those with norm spread — which is
  a property of the *codebook shape*, not of retrieval quality. A 1-bit scalar
  quantizer emits ±c on every coordinate, so ‖x̂‖ = c√d is constant **by
  construction** and it pays nothing; an m-dimensional VQ grid's ‖x̂‖ varies
  (CV ≈ 1%). Measured on 750 arXiv abstracts at 1 bit, the VQ arm scored
  recall@10 0.663 against the scalar arm's 0.686 *despite* better MSE (+0.50 dB)
  and better mean reconstruction cosine (0.823 vs 0.799); renormalising the
  reconstruction flipped it to 0.689 and the ordering agreed with the
  distortion numbers again. Left alone this reads as a genuine
  "scalar wins at low rate" reversal. `jina-remex-vs-remax/score_fidelity.py`
  and the first version of `remex-vs-higgs-ablation` both had it, so check any
  harness in this repo that scores `Q @ Xhat.T`. Keep reconstruction MSE on the
  *raw* reconstruction so it stays a property of the codec.
  (`remex-vs-higgs-ablation/run_ablation.py::Reference.score`)
- **Better MSE does not imply better recall, and mean reconstruction cosine
  does not settle it either.** Both were *better* for the arm that lost above.
  When a distortion metric and a ranking metric disagree, look for a
  per-document quantity that shifts scores without shifting fidelity — here the
  spread of ‖x̂‖ across documents. Diagnose it by renormalising and re-scoring:
  if the disagreement vanishes, it was scale, not geometry.
- **…and when renormalising does *not* make it vanish, stop looking for a
  summary statistic.** On `fmnist784` (ANN-benchmarks fashion-mnist-784, raw
  pixels, d=784) at 1 bit, the vector-grid arm wins reconstruction MSE on all
  four corpora in both metrics (≈0.32 vs ≈0.36 relative) **and** wins on the
  variance of the projected score error q·(x̂−x) (0.0359 vs 0.0366) — the
  quantity ranking actually sees — and still loses recall@10 by 0.021 under
  cosine and 0.047 under inner product, one-signed across 5 seeds. Three
  mechanisms were measured and refuted: non-Gaussian rotated marginals (excess
  kurtosis −0.040, KS 0.016 — the rotation Gaussianises even data that is 50%
  exact zeros), norm-noise × corpus norm spread (non-monotone: the largest
  spread of four corpora, CV 31%, gives the largest scalar win), and
  block-correlated VQ residuals (refuted by the score-error variance above).
  The regime is the likely culprit and is worth checking first next time:
  fmnist784's rank-10-to-11 similarity gap is 0.00093 at a mean similarity of
  0.923, so 1-bit score error (std ≈0.037) is **~40× the gap** and both arms
  rank almost entirely inside their own noise. Practical rule: **at rates where
  quantization noise dwarfs the neighbour gap, no distortion summary — MSE,
  cosine, or projected score variance — predicts recall.** Measure recall.
  (`remex-vs-higgs-ablation/RESULTS.md`, "The fmnist784 result")
- **Corpora built from encoders trained under cosine cannot test anything
  about vector norms.** BGE-family raw norms vary by CV 1.4–2.7%, so
  inner-product retrieval on them is nearly the same problem as cosine, and any
  "exact norm vs per-block scale" comparison reads null for a reason that has
  nothing to do with the methods. If an axis touches norms, put a corpus with
  real spread in the design *before* running: GloVe-100 is CV 20%, raw
  fashion-mnist pixels CV 31%, both one `curl` from ann-benchmarks.com and
  needing no encoder. Measured across four corpora spanning CV 1.4%–31%, the
  axis-B effect is flat everywhere (+0.0003 to +0.0017), so the null is real —
  but it took the wide-spread corpora to know that rather than assume it.
  (`remex-vs-higgs-ablation/build_corpora.py::build_fmnist`)
- **Rotated *unit* vectors have Beta-distributed coordinates, not Gaussian**
  (density ∝ (1−x²)^((d−3)/2)); TurboQuant fits its Lloyd-Max to that Beta.
  Using a Gaussian at σ=1/√d instead costs ≤0.007% excess MSE at 2 bits and
  ≤0.43% at 6 bits for d=100, and ~0% for d ≥ 768 — measured, so you can skip
  the Beta with a citation rather than a hope.
  (`remex-vs-higgs-ablation/beta_check.py`)

---

## Cache and measurement hygiene

- **Fit `t = a + b·n` before reporting a crossover — a scale gate and a
  constant term look identical from two data points.** Any linear scan is
  `t(n) = a + bytes_per_candidate·n / (bandwidth·efficiency)`, where `n` enters
  only as a multiplier on the linear term, identically for every kernel. So
  **two kernels can cross only if at least one has `a > 0`**; with `a ≈ 0` on
  both sides the ratio is constant and no corpus size flips the ordering. A
  reported crossover is then an artifact of the two `n` values that bracket it.
  `lowbit-scan-crossover/fit.py` is the check. It found that `dab41dd6`'s
  "compression is SCALE-GATED below ~150k rows" fits `4.108 ms + 32.40 ns·n`
  while the same numpy expression on another machine fits `−0.78 ms +
  33.98 ns·n` — **per-row costs agreeing to 5%, so the entire gate was the
  4.1 ms constant**, and `n* = a/(b_f32 − b_ham)` reproduces the reported gate
  at ~68,000. `n*` is meaningful only inside the fitted range. When `a > 0`, the
  constant *is* the finding, and it belongs to the harness, not the corpus.

- **`.sum(axis=1)` over a narrow inner axis is the bottleneck in packed-code
  scans, not the popcount — store bit planes instead.** In
  `np.bitwise_count(C ^ q).sum(axis=1)` (the kernel `Portable code` records
  above from `remax-hamming-speedup`), `np.bitwise_count` runs at 14.7 GB/s and
  `.sum(axis=1)` over a 4-wide axis at **1.9 GB/s** — 62% of the kernel. numpy's
  per-row pairwise-reduction overhead is fixed, so the damage scales inversely
  with code width: 1.20 GB/s at 4 words/row (k=256) vs 3.33 GB/s at 32 words/row
  (d=512·k=4). Storing the W words as W contiguous **columns** turns the row
  reduction into W−1 whole-array adds: **5.2x at k=256, 2.4x at d=512·k=4**,
  pure numpy. Two consequences: a kernel benchmarked at one code width does not
  transfer to another (this is how `dab41dd6` and `remax-hamming-speedup` reached
  opposite verdicts on the same idiom), and "pure numpy already beats BLAS so a
  compiled kernel is unnecessary" was measured against the wrong numpy — the
  compiled kernel is 37x over BLAS warm and 19x cold, against the idiom's 1.7x.

- **A SIMD instruction you assume is load-bearing usually is not — rebuild at a
  lower `-march` and measure before scoping a result to it.** The obvious
  explanation for a slow popcount scan on one container was a missing AVX-512
  VPOPCNTDQ. Rebuilding the same C source at `-march=x86-64-v3` and `v2`
  (objdump confirming **zero** `vpopcnt` instructions — gcc emits a
  table/Harley-Seal popcount, per Muła/Kurz/Lemire) cost **6%**: 34.7x and
  34.8x over BLAS against 37.0x. Cheap to run, and it kills an entire class of
  "their hardware was different" hand-waving. Practical consequence:
  `remax/_native.py` compiles at import with plain `-O3` and **no `-march`**,
  which is the right call for a cached user-side compile — that portability is
  worth ~6%, now measured rather than assumed.

- **Length-sort before batching, and check the length distribution before
  believing your throughput is compute-bound.** Tokenizer padding is to the
  *batch's longest* sequence, so corpus-order batches of heterogeneous text pay
  for the longest member every time. On an AST-chunked scikit-learn corpus
  (median 322, p90 927 tokens) that is **1.47x** wasted padded tokens; sorting by
  length before batching and restoring the order after measured **1.25-1.38x**
  end-to-end with **bit-identical** output. Two traps around it: (a)
  `encode_batch` over a whole corpus pads *everything* to the global longest, so
  a naive length histogram taken that way reports 100% at the truncation cap and
  hides the distribution entirely — call `no_padding()` first; (b) before
  optimizing, price the work: 3.73M tokens x ~17.4 MFLOP/token = 64.9 TFLOP, and
  a 4-vCPU AVX-512 box sustaining 424 GFLOP/s on 2048^2 sgemm has a 2.6-minute
  floor, so "this should take seconds" was never on the table. Achieved was
  106 GFLOP/s = 25% of peak, which for 384x1152 GEMMs is low but not pathological.
  (`bekko-embedding-bench/RESULTS.md`)
- **Run the power analysis before the compute budget, not after.** 67 of 78
  minutes of encoding went to a 2x2 of axes (encoder size, chunking strategy)
  whose measured effects all landed inside the noise of a 6-instance benchmark.
  Sizing the corpus for rigour was right; replicating it across axes the sample
  could not resolve was not. Ask "what effect size can n detect?" first, then buy
  only the cells that clear it. (`bekko-embedding-bench`)
- **Put the compute where the inference is, and run the paired test before
  writing the headline.** This experiment spent **78 minutes** encoding 41,500
  scikit-learn chunks for a **6-instance** code-search benchmark, while every
  embedding-quality conclusion rode on **179 chunks from 11 blog posts** encoded
  in seconds. At n=179 one query is 0.56 pp of R@10, so the 2-8 query differences
  being reported were unresolvable: exact McNemar on the discordant pairs killed
  **seven of eight** headline claims, including "quantization at 96 B beats the
  uncompressed 1536 B vector" (+0.011, 3 wins to 1, p=0.625). What survived was
  the one comparison with a large effect — jina over bekko on code-distribution
  R@1, +0.168, 31 to 1, p<1e-5. Two habits this buys cheaply: **arms share
  queries, so test paired** (McNemar / paired bootstrap, not two independent
  proportions), and **"wins 11 of 12 cells" is not 12 trials** when the cells are
  nested dimensions of one corpus. Resolving a true 0.01 R@10 gap at 80% power
  needs n>2000 — cheaper than the encode budget already spent.
  (`bekko-embedding-bench/RESULTS.md`)
- **Before charging a codec for "shared structure", check what the structure
  actually is — and whether anything ships it.** I priced remex's side data as a
  materialized dense d x d rotation (590 KB at d=384) plus "the codebook",
  concluded its byte advantage inverted below n~411, and was **wrong on both
  counts**. remex's Lloyd-Max codebook is *scalar*: 2^bits-1 boundaries +
  2^bits centroids for a 1-D N(0,1/d) coordinate, i.e. **28 B at 2 bits**,
  21,065x smaller than the rotation, and *analytic* from (d, bits) so a reader
  recomputes rather than receives it. (The "large shared codebook" intuition came
  from `remex-vs-higgs-ablation`, where it meant a **vector**-quantization grid
  with 2^(m*bits) codepoints — a different object; do not transfer it to a scalar
  quantizer.) And the rotation is seed-derived in every remax_kb configuration
  except v2-`haar`-with-int8-sidecar, so it ships as 4 bytes of seed. Real
  per-vector side cost at n=179: **0.0 B**, not the 3,295 B I charged. Two rules:
  take payload from the codec's own `nbytes` (it includes the separately-stored
  f32 norms — +4 B/vec, i.e. +50% at d=64 @ 1-bit, and that part *was* a real
  bug), and price side data by **what a reader receives**, not by what happens to
  sit in RAM. (`bekko-embedding-bench/RESULTS.md`)
- **An RHT's storage is O(d) bits in operator form; the d^2 floats are a speed
  choice, not a requirement.** A randomized Hadamard transform is a +/-1 diagonal
  plus an FWHT, so its entire state is `rounds * d` bits — 144 B at d=384,
  rounds=3, against 589,824 B for a dense f32 rotation. Both
  `remax.rotation.rht_rotation` and `remax_kb.projection.srht_matrix`
  *materialize* it to keep the apply path a single BLAS matmul, which is why a
  naive `nbytes`-style audit sees no saving. If storage is the constraint, use
  the operator form; if apply latency is, materialize and cache. Removing the
  d^2 term is exactly what the RHT is *for*, and it is the argument for v2's
  `srht` default over `haar`+int8-sidecar (1.2 MB of planes against a 179-chunk
  corpus = 6,590 B/vec). (`bekko-embedding-bench/RESULTS.md`)
- **A per-query constant can eat an order-of-magnitude encoder win — measure the
  whole path, then decompose it.** bekko-a8m is 12.9x faster than jina v5 nano q4
  in isolation, but only **2.3x** through `remax_kb.read.KB.search`, because
  `_stacked_simhash_encode` constructs a `StackedSignBitQuantizer(d, k, seed)` on
  every call and that constructor builds k Haar rotations by QR — from manifest
  parameters that cannot change for an opened index. It was **87% of bekko's
  query** and ~50-60 ms flat for everyone. Caching it per opened index (one line,
  verified to yield identical codes *and* identical hit lists) restores
  **11.6-15.1x**. Two portable lessons: a fixed tax hurts most whoever else is
  cheap, so component benchmarks systematically overstate wins for the fast
  component; and "deterministic from (d, k, seed)" is a construction-time
  invariant that a reader should exploit, not re-derive per call. Decompose
  before concluding — the stage that dominated here was neither the model nor
  the search. (`bekko-embedding-bench/RESULTS.md` §6)
- **Token accounting for a retrieval baseline is a methodology choice — quote
  both, or you are picking the flattering one.** Charging `rg` its full
  matching-line output made a dense arm look **12.8x cheaper**; charging `rg -l`,
  which returns exactly the information a *file*-discovery metric scores, made
  the same dense arm **2x more expensive**. Same runs, same metric, opposite
  conclusions. State which output mode the baseline is charged for.
  (`bekko-embedding-bench/RESULTS.md`)
- **A benchmark's instance set is the artifact worth keeping, not its code.**
  mini-CTXBench has now been rebuilt three times at ~3 tool calls each; what was
  lost every time, and what made "reuse the same 7 instances" unsatisfiable, is
  the *list of instances*. Commit the pinned set (issue number, PR number,
  commit sha, gold files, issue body) even when the harness is throwaway —
  without it no two runs are comparable. (`bekko-embedding-bench/instances.json`)
- **Assign strata by a measured property, not a remembered label.** Splitting
  benchmark instances into identifier-rich/-poor by *running the extractor and
  seeing what it recovers* produced a cleaner split than the hand labels would
  have, and is reproducible: the deciding instance yields literally zero
  code-shaped tokens because it argues its bug in prose and points at code by
  line-number URL. (`bekko-embedding-bench`)
- **Key a cache on the METHOD, not just on the problem.** A grid cache keyed
  `(m, K)` silently served codebooks trained by an older, worse procedure after
  the trainer was fixed — and the stale file was 87% worse than the baseline it
  was supposed to beat, which would have produced a confidently wrong published
  result. It survived a `rm -rf` of the cache because a background job that was
  being killed rewrote it moments after the delete, so it was identifiable only
  by its *schema* (an extra key the new writer does not emit). Put a
  `VERSION` constant in the filename **and** inside the file, and delete on
  mismatch rather than trusting. Bump it whenever the procedure changes.
  (`remex-vs-higgs-ablation/grids.py::GRID_VERSION`)
- **Exonerate the instrument before blaming the subject** — principle 1 applied
  to measurement. When a sampled/empirical metric disagrees with theory, first
  find a *degenerate case where the two must agree by construction* and check
  there. An m-dimensional product of scalar quantizer levels must score exactly
  the closed-form scalar MSE under nearest-neighbour assignment, because NN on
  a product grid decomposes per coordinate; agreement to 3 significant figures
  at 2/4/6 bits proved a KD-tree MSE harness correct and left the trained grid
  as the only suspect. Cheap, and it converts "one of these two things is
  broken" into "this one thing is broken".
  (`remex-vs-higgs-ablation/calibrate.py::g0_measurement_path`)
- **A checked-in number copied from a log line is a number you will get
  wrong.** A migration guard was seeded with an intermediate `lloyd it=…`
  training MSE (0.0796) where the run's *held-out* figure was 0.0887, and
  correctly refused a perfectly good artifact. Training and held-out MSE differ
  by ~11% at 61 samples/codepoint — always take the final reported line, and
  say in the comment which one it is.
  (`remex-vs-higgs-ablation/migrate_grid.py`)
- **`pgrep -f <pattern>` matches the watcher's own command line.** A wait loop
  written as `until ! pgrep -f trainer; do sleep 30; done` never exits, because
  the shell running it contains "trainer" in its own argv. A second attempt
  with a malformed predicate exited *immediately* and reported the job finished
  while it was still running — the more dangerous failure. Wait on a PID
  (`while kill -0 $PID`), which cannot self-match. This is "a check that cannot
  fail is not a check" (see `svgview` below) in wait-loop form.

---

## Negative results — do not re-derive

- **A code-trained encoder did not beat a general text encoder on NL->code file
  discovery — and the obvious confound was ruled out.** On a 59-instance
  scikit-learn bug-report -> gold-file benchmark,
  `jina-embeddings-v2-base-code` (161M, 768-d, 30 languages of code) scored
  r@5 **0.630** against general-text `bekko-a25m`'s **0.656** and a plain `rg`
  baseline's **0.596** — it lost to the general encoder and no comparison was
  significant, at **6x the encode cost** (612 MB / 61.9 min vs 124 MB / ~10 min).
  The confound worth checking first, because it would have explained the result
  trivially: AST chunk headers carry the module path and class name, so a text
  encoder might be matching those and ignoring the code. **Refuted** — retrieving
  against file paths alone, with no code content, scores 0.304/0.370, so code
  content roughly doubles recall and is genuinely used; the specialization just
  adds nothing on top. Likely because file-level discovery from a bug report is
  topical and lexical, not a code-semantics task. Buy a code encoder for
  code-to-code similarity, not for NL->file localization.
  (`bekko-embedding-bench/RESULTS.md`)
- **SPECTER2/citation-trained embedding geometry cannot find cross-disciplinary
  bridge papers.** Four escalating experiments; the twin diagnostic found 0/26
  expected twins in any anchor's ~700-paper candidate union across 9 documented
  bridges. The representation space does not encode mechanism-level cross-domain
  correspondence. **Blocks a paused $435 1.9M-paper production run.**
  (`te-bridges/path_c_cross_domain/RESULTS.md`)
- **RM3 pseudo-relevance feedback does not help on a small corpus** — tied or
  lost to plain BM25 on 73 posts (R@10 1.000→0.900), across two chunk
  granularities. (`muninn-rm3/RESULTS.md`)
- **Plain BM25 on whole documents matches the dense-embedding ceiling for
  in-vocabulary queries** (R@10 = 1.00, mean rank 1.0), and 17× fewer chunks
  caused no recall loss — the "large chunks dilute lexical retrieval"
  assumption is false for BM25. Embeddings win only on vocabulary-divergent
  paraphrase (1 of 5 queries). (`lexical-kb-phase0/RESULTS.md`)
- **Decorrelated (shared-ITQ + random) rotation mixes tie plain random SimHash
  exactly** at every k on two embedders; under an honest transfer protocol plain
  SimHash beats ITQ outright. (`rotation-decorrelation/RESULTS.md`)
- **Offloading a job from claude.ai to CCotw buys wall-clock, not throughput —
  and check what the throughput number already contains before generalising
  from it.** The claude.ai container measures jina-v5-nano q4 at <2 docs/s on
  one core and reaps detached jobs after ~100 s, which is what makes a 5,183-doc
  encode impossible there rather than merely slow. CCotw's 4 vCPU do not encode
  faster *per core* — one measured run got 4.4 docs/s (~1.1/core), i.e. ~2.2x
  whole-machine — so the job becomes possible because 15–20 uninterrupted
  minutes are available, not because the cores are better. **But that per-core
  constant is soft:** a second run on identical hardware got 5.9 docs/s
  (~1.5/core) purely from length-sorted batching, so the 1.1 figure carries a
  1.37x unforced overhead. Both conclusions survive — offload for wall-clock,
  and expect little from more cores — but quote the per-core number with the
  batching strategy attached. (`ttt-embed-quantized/RESULTS.md`)
- **The `jina-v5-nano-mirror` ONNX loader cannot load the q4 asset.**
  `scripts/embed_onnx.py::materialize()` hardcodes the 847 MB fp32 `model.onnx`
  in its `ONNX_ASSETS` table, so it will neither fetch nor open the 170 MB
  `model.q4.onnx` the same release publishes — despite the module docstring
  presenting itself as the torch-free path for ephemeral containers, which is
  exactly where the small asset is wanted. Both encodes of issue #33 hit this
  and both replicated its pooling/prefix/normalize semantics by hand against the
  same `tokenizer.json` and `pad_id=128001`. Reuse `ttt-embed-quantized/encode.py`
  rather than rediscovering it. (`ttt-embed-quantized/encode.py`)
- **Hand-rolled q4 ONNX export is dominated by the model authors' official
  one** — official was better on nDCG, cosine, recall-vs-fp32-kNN and Spearman ρ
  *and* 32MB smaller. Check for an official/Optimum export before building your
  own. (`q4-official-vs-ours/RESULTS.md`)
- **Repo size does not bury small repos in account-wide code search, and the
  three obvious corrections all lose.** On 145 PR-mined file-localization
  queries across 19 repos spanning 278-11,195 chunks, the current RRF ranker
  scored recall@1 **73.8**; a per-repo cap tied at 73.8, a diversified candidate
  pool scored 67.6, and an RRF weight of `1/log(1+chunks_in_repo)` scored
  **48.3** — a 25-point loss, because a third of queries legitimately belong to
  a large repo and the prior pushes those answers down. The premise fails too:
  under the baseline, small repos *out*-perform large ones (80.4 vs 68.3), so
  there is nothing for a size correction to correct. Scoping with `-r` is worth
  3.4-7.6 points and remains the whole real effect. Replicated on
  identifier-poor queries (path tokens stripped) with the same ordering.
  (`xr-repo-crowding/RESULTS.md`)
- **EarthCam's `gethofitems.php` ignores `start`/`date_start`/`date_end`** — always
  returns newest ≤50, ~45-day retention, deep archive is paid with no open API.
  (`dc-mall-timelapse/RESULTS.md`)

---

## Shared code — `_lib/`

Deliberately small. An experiment is self-contained by default; code lands
in `_lib/` only once a second experiment needs it, or once a hardcoded path
has broken across many of them.

| Module | What |
|---|---|
| `_lib/paths.py` | `experiment(name)` for siblings in this repo, `spoke(name)` for checkouts outside it (`EXPERIMENTS_SPOKES_ROOT`) |
| `_lib/pipeline.py` | `retry` (jittered backoff), `chunked`, `save_json`/`load_json` (atomic tmp-then-rename checkpoints) |
| `_lib/textnorm.py` | `ascii_fold` — the NFKD stroke-letter fix |

Tests: `python3 _lib/tests/test_lib.py` (no deps, no network, no creds).

To use from an experiment script — they run directly and are not a package,
so the repo root has to go on the path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke
```

## Migration breakage — fixed

32 `.py` files across 13 experiments hardcoded `/home/user/claude-workspace`,
which stopped existing when this repo split out of the workspace hub. They
were non-runnable as checked in. All now resolve through `_lib/paths.py`.

If you see that prefix again, it came from a script written before the split.

## Duplication map

- `phase-a-bridges/scripts/common.py` ↔ `te-bridges/scripts/te_common.py` —
  were near-identical. Generic parts now extracted to `_lib/pipeline.py` and
  `_lib/textnorm.py`; both files re-export so call sites are unchanged.
- `muninn-embedder-bakeoff/bench.py`, `lfm25-embedder-remax_kb/bench_muninn.py`,
  `jina-int8-remax_kb/bench.py` — three independent reimplementations of the
  same muninn-corpus R@5/R@10 harness (1238 chunks / 73 posts, chunk hits
  collapsed to distinct posts, fixed 5-query topical gold). **Still duplicated.**
  Not consolidated because the corpus is not present here, so a refactor could
  not be run — see "Not done" below.
- `lowbit-scan-crossover/hamkern.c` ↔ `remax/src/remax/_native.py` — an
  independent reimplementation of a C `__builtin_popcountll` Hamming scan that
  **already existed in `remax`**, is already dispatched to by
  `remax.packing.hamming_distances`, and whose docstring already measured
  25-35x over the NumPy path with the same 100k-1M cache falloff. Written
  because the mandated account-wide `xr` check was skipped. **Keep
  `hamkern.c`** as a standalone roofline reference (no ctypes cache, no
  fallback, so it isolates kernel cost) but treat `_native.py` as the
  implementation. Fourth documented rediscovery in this repo, and the least
  excusable: run afterwards, `xr "hamming distance popcount kernel in C over
  packed binary codes"` returns `remax/src/remax/packing.py` at rank 1 and
  `remax/src/remax/_native.py` at rank 8; `-r remax` puts `_native.py` at rank
  2 and `tests/test_native.py` at rank 4. The tool works and answers in 175 ms
  warm. It was skipped, then — when a first attempt raised
  `ModuleNotFoundError: remex` — written up as "unavailable in this container"
  rather than fixed with the `pip install` below. **An ImportError in a
  mandated check is a missing dependency, not a broken check; install it and
  re-run before concluding anything about the tool.**
  `mcp__github__search_code` with `user:oaustegard <terms>` is the no-local-
  index fallback and found `_native.py` in one query.
- `lexical-kb/skill_template/search.py` → `creating-kb` skill →
  `kb-packer-web/vendor/search.py` — deliberate vendoring with
  `kb-packer-web/check_sync.py` guarding drift. **Leave as is.**
- `muninn-rm3/bench.py` imports `QUERIES`/`stem` from `lexical-kb-phase0/sweep.py`
  — good instinct (reuse by import rather than copy), and the path now resolves
  via `experiment("lexical-kb-phase0")`. The query/gold set would be better as
  its own module than as an import from a sibling's `sweep.py`.

## Not done

- **The three-way embedder-bakeoff harness is still duplicated.** Consolidating
  it is a behavioural refactor, and the muninn corpus, spoke checkouts and
  models are not present in this repo — the result could not be run, only
  compiled. Left for a session that has the corpus.
- **Findings below are from a single reading pass** over each experiment's
  results file plus a skim of its code. Numbers are quoted from those writeups,
  not re-measured.

### A closure gap is a property of the context, not an error rate

Before reporting `|S'' \\ S|` as a defect, check whether the closure *is* the
intended answer. `lattice-representation-hypothesis/RESULTS.md` §4.2 — an entire
thesis was built on treating FCA join overshoot as error; the overshot members
are the least upper bound, and in the target task they were the gold label. A
textbook-correct implementation scores 0.60 on that metric.

**Use when:** any metric of the form "operator X admits members that Y does not".

### Never report a normalized rate without checking its denominator

`lattice-representation-hypothesis` Arm A — join overshoot falls 0.60 -> 0.20 as
embedding dimension drops when normalized by |join|, and rises 0.28 -> 0.61 over
the same sweep when normalized by objects at risk (Spearman -1.000). The first
normalization is the natural one and gives the wrong conclusion ("low dimension
is benign"); the denominator was growing 5x. Plot both or state which you chose
and why.

### Compare set-recovery metrics only at matched target size

`lattice-representation-hypothesis` §5.2-5.3 — Jaccard and symmetric-difference
error gave *opposite* meet-vs-join answers on the same runs, purely because one
operator's targets were 12x larger. Bin by constraint count to control it, then
check the residual gap against a random-direction baseline: here the control
reproduced the gap *more strongly* than any real probe, which retired it.

**Use when:** comparing recovery of two set-valued targets of unequal size.

### Trees are not chains — hierarchical data does not have small closure gaps

`lattice-representation-hypothesis` §4.3. A totally nested attribute chain has
zero closure gap. A hypernym *tree* has 0.93 non-closed unions and 0.71
overshoot — worse than iid Bernoulli contexts. Two leaves in different branches
have incomparable extents. Do not accept "the data is hierarchical, so unions
are nearly closed" without measuring it.

### Adversarially review your own thesis before writing it up

`lattice-representation-hypothesis` — a separately-prompted agent tasked with
attacking the critique (not the paper) killed the headline claim, and the
genuinely interesting result only became visible once the wrong one was cleared.
Then re-check the reviewer with your own code: it was right about the central
maths and wrong about two subsidiary claims, both of which mattered.

### Pre-register the null before building a metric to criticise something

State what your measurement reads if the target is **entirely correct**. If that
value is indistinguishable from the "broken" reading, the metric is not a
diagnostic — redesign before running it.

`lattice-representation-hypothesis/RESULTS.md` §4.2 — a textbook-correct FCA join
scored 0.60 on the "overshoot" metric built to show the join was broken. The
metric could not separate right from wrong, and that was knowable from the
definition before any code was written. Six arms and 9.6M measured pairs were
spent downstream of it.

Companion failure modes from the same experiment, all cheap to check:

- **Date the thesis.** Rigor applied after a conclusion is locked measures the
  conclusion, not the world. That thesis was fixed ~2 minutes after reading an
  abstract; nothing downstream ever revisited it.
- **Re-derive your central identity from your own primitives, once, hostilely.**
  The refutation sat eight lines from the false claim in a docstring in this
  experiment's own `fca.py`, read a dozen times as confirmation.
- **Schedule the adversarial pass in the plan, not after a result you like.**
  Both times here the save came from structure — an adversarial subagent and
  another subagent's own controls — never from re-reading.
- **Distrust the most quotable sentence.** "0 meet phantoms across 9,615,370
  pairs" was true, correctly computed, and merely Monte Carlo verification of
  Ganter & Wille's Basic Theorem.

**Use when:** any experiment whose purpose is to show that something is wrong.

**Converges with "A check that cannot fail is not a check"** (`svgview/`, same
week, independently derived): there, a screenshot assertion was equally true of
an unchanged frame, so it passed while no keystroke was being delivered; here, a
metric scored the correct answer identically to the broken one. Same principle
reached from a GUI smoke test and from a formal-methods critique — an assertion
whose truth does not *depend on* the thing under test will report success on a
no-op. Both also share the tell: **the pass came easily and on the first try.**

### Never write up a subagent's data without its interpretation

`lattice-representation-hypothesis` §5 — Arm B's JSON was written up before its
report arrived, producing a narrative the agent's own analysis contradicted:
same numbers, wrong reading. Had the report landed 20 minutes earlier the
writeup would have differed from the start.

Fan-out delivers data and interpretation on separate schedules. Either wait for
the report, or label the reading as your own inference. A conclusion must not
depend on which subagent finished first.

### Validate hidden test suites against a reference before any model sees the task

Author spec, tests, and a reference implementation together; the suite must pass
against the reference before it grades anything else. In
`orchestrated-coding-pareto` this caught authoring bugs in 3 of 14 suites — one
wrong test expectation that would have failed every arm on correct code, one
reference/spec disagreement that would have graded spec-compliant code as wrong,
one mangled-but-accidentally-correct reference. A hidden suite that has never
passed a known-good solution is an unvalidated measuring instrument.
(`orchestrated-coding-pareto/ERRORS.md`)

### Grade early arms before building arms that depend on them

Task-difficulty ceilings are invisible until graded. `orchestrated-coding-pareto`
planned retry/orchestration arms seeded from a cheap model's failures; grading the
first six landed solutions mid-run (rather than after all arms) showed 6/6 passes
and forced two difficulty escalations while generation for later arms had not yet
been paid for. Had grading waited for the full matrix, the whole run would have
produced an undiagnostic null. Corollary: retry-style arms should state up front
what happens when the failure set is empty — "vacuous" is a reportable outcome,
not a broken pipeline.

### Per-token price is not cost — measure tokens-per-task before comparing tiers

A model at 1/5 the per-token price that emits 6.7× the output tokens costs *more*
per solved task. `orchestrated-coding-pareto` measured Haiku 4.5 at 20.1k output
tokens/task vs Opus 5 at 3.0k on identical tasks (ratio worsening with task
difficulty), flipping the expected cost ranking at equal measured quality. Any
cheap-fleet argument made from a price sheet alone is unfounded until the
verbosity ratio is measured — and the ratio may be a harness/effort artifact, so
measure it in the deployment configuration, not in general.

### Workflow-tool token metering: sequential phases, budget.spent() deltas

`budget.spent()` is a turn-global output-token counter shared by all concurrent
workflows. Per-arm/per-phase metering therefore requires strictly sequential
phases (or sequential workflow invocations) with marks recorded between them;
launching two workflows concurrently silently confounds every delta. Input tokens
are not observable per-arm at all — estimate from content sizes and say so.
(`orchestrated-coding-pareto/data/marks.json`)

### Decode latency tracks depth, not parameter count — check before picking a draft model

At batch 1 the marginal cost of a transformer layer is dominated by fixed
overhead (kernel dispatch, normalization, cache handling), not by the width² in
its FLOPs. `monad-specdec` truncated two PleIAs stacks and fitted a line through
the depths: 0.804 ms/layer at width 256, 1.264 ms/layer at width 576 — a ratio of
1.57 where a compute-bound decode would show 2.25² = 5.06. So Monad at 1/5.7 the
parameters of Baguettotron is only 2.1× faster per token, because it carries
64 layers to Baguettotron's 80.

Consequence for speculative decoding: the cost ratio c that sets break-even comes
from **depth**, so a draft model must be *shallow*, not merely *small*. A
parameter-count estimate of c was 2.6× too optimistic here and predicted a win
where the measurement gives 0.90× baseline. Two minutes of layer-truncation
benchmarking answers this before any decoding harness gets written.

Second-order, same experiment: a draft model with a **smaller vocabulary** needs
more steps per target token (3.25 vs 4.12 chars/token → 1.27 draft steps per
target token), which multiplies the effective c. Cross-tokenizer drafting pays
this on top of the acceptance cost.
