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

- **A 34-row eval can report the opposite sign of a real effect — grow the eval
  before you believe the direction, not after.** The retriever `nl2sh-dense`
  recommends raises gold-in-sources on both halves of its eval (0.235 -> 0.382
  on the original 34 rows, 0.269 -> 0.400 on 130 new ones) and moves end-to-end
  routing in opposite directions on the two: **0.206 -> 0.147** on the old rows,
  **0.108 -> 0.169** on the new. Two queries flipping is 0.059 at n=34. Had the
  session kept the inherited eval, its headline would have been that better
  retrieval makes the system worse, with a plausible mechanism available to
  explain it. Extending it cost about 25 minutes: 149 more rows sampled from the
  corpus that eval already used, natural language written by the same model
  through the same prompt. When an eval is small enough that one query is worth
  0.03, treat "extend the eval" as the first experiment, not as diligence after
  the result. (`nl2sh-dense/sample_cyber.py`, `RESULTS.md`)

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

### 2a. Fail-first branching before you call an exact B&B compute-bound

`ms13-k4`: the `ms13-campaign` exact rational branch-and-bound (a
disjunction of `(row, side)` options per rounding, one exact LP per node)
had not finished the first `k = 4` type after 50,000 nodes and 900,000 LPs
in lexicographic rounding order. Choosing the next rounding as the one with
the fewest options still satisfied at the parent LP's optimum point proved
the same type in 909 nodes and 5 s, all 14 types in ~400 s, and re-proved
the campaign's `k = 3` theorem in 82 and 122 nodes against 15,975 and 14,440.
Ordering cannot change the answer of an exhaustive search, so this is free
soundness-wise. Try it before sizing a run as a no-go.

### 2b. Replace a tree census with its split system (Buneman)

`ms13-k4`: the campaign enumerated trees on up to 14 nodes with marked
endpoints (NG-14: ~2,070 h at `k = 4`). Every row is a split of the `2k`
endpoints; compatible split systems are exactly trees; refining a tree by
moving labels to pendant leaves keeps every split and adds trivial ones. If
the objective is monotone in the row-set, only binary trees with labels at
leaves matter: unrooted shapes × perfect pairings of the leaves (4 × 105 at
`k = 4`, 11 × 945 at `k = 5`). Validate by reproducing a smaller case the
old census closed (here `k = 3`: two maximal 6-row types, identical).

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

- **A prompt comparison run zero-shot can be decided by output *shape* rather
  than by the framing you meant to test — anchor the output slot in every
  condition before reading the result.** `nl2sh-instantiate` asked Gemma 3 270M
  to substitute literals into a documented example instead of generating a
  command freely, and the substitution prompt lost 0.146 to 0.500 on routing.
  It was not losing at substitution: on **0.774 of rows** it answered in the
  shape of the source lines it had just been shown — `- go — Go to my home
  directory`, bullet and em-dash included — which the scorer reads as the
  utility `-`. Appending a bare `Command:` cue to *both* conditions separates
  the two questions. One epoch of fine-tuning erases the imitation completely
  (0.774 -> 0.000), which is the same shape as `nl2sh-retrieval`'s 0.026 ->
  0.706 and `monad-bsky`'s 0.000 -> 0.481: **a zero-shot format failure in a
  small instruct model is weak evidence about the task and strong evidence
  about the output slot.** (`nl2sh-instantiate/prompts.py`, `score.py`)
- **When two arms tie on your headline metric, check whether they separate on
  the failure mode — the tie may be the less interesting half of the result.**
  Fine-tuned under either prompt, the two arms route identically (23 wins to
  20, p = 0.76). They separate on degeneracy: token-repeat loops fall
  **0.183 -> 0.049** and *usable* — routes correctly **and** is not a loop —
  goes 31 to 16, p = 0.040. `nl2sh-selfhist/MODELS.md` had named degeneracy the
  real ceiling after `repetition_penalty=1.3` bought a comparable reduction at
  a cost of 0.118 routing; the prompt change buys it for free. Report the
  garbage rate as its own column, always. (`nl2sh-instantiate/RESULTS.md`)

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

- **A repo's `.claude/settings.json` Stop hook only fires in sessions actually
  booted from that repo — cloning it via `add_repo` does not wire the hook
  into your session.** `claude-workspace#233`'s supervisor loop (merged in
  PR #237) depends on `scripts/supervisor_stop.py` being registered as *this
  session's* Stop hook. A `CLAUDE_CODE_CHILD_SESSION=1` session spawned from
  another Claude Code session (via `create_session`, not a CCotw/Muninn hub
  boot) has its own unrelated Stop-hook scaffolding
  (`~/stop-hook-git-check.sh`, `~/stop-hook-reply-gate.py`) and never reads
  `.supervisor/run.json`, no matter how correctly that file is armed. Check
  `~/.claude/settings.json` (or lack of one) and `env | grep
  CLAUDE_CODE_CHILD_SESSION` before assuming a cloned repo's session-level
  hooks are live — arming a hook-dependent loop in the wrong session is a
  silent no-op, not an error. (`avo-supervisor-specter2/RESULTS.md`)

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
- **To disable thinking on gemini-3.5-flash-lite, OMIT `thinkingConfig` — do not
  pass `thinkingBudget: 0`.** The lite model 400s on an explicit `thinkingBudget: 0`
  (`INVALID_ARGUMENT`), which is easy to misread as "thinking is mandatory". It is
  not: with **no `thinkingConfig` field at all**, `thoughtsTokenCount` is **0** and
  a one-command answer returns in ~1.0-1.3 s (gateway round-trip, not thinking).
  Passing `thinkingBudget: -1` turns dynamic thinking ON (213 thought tokens
  measured), which is the opposite of what you want for high-volume work — so the
  earlier claim here that flash-lite carries a mandatory thinking tax was wrong,
  corrected 2026-08-19. Per-model: 3.5-flash-lite rejects budget=0 but defaults to
  no-thinking; 2.5-flash-lite accepts budget=0; both think=0 with no config. In
  `gemini_client.generate`, `thinking_budget=-1` omits the field (the fast path),
  `>=0` sets it. (`nl2sh-selfhist/gemini_direct.py`)
- **Even where `thinkingBudget: 0` is honoured, budget the OUTPUT tokens
  generously — a short answer can still be truncated.** On gemini-3.7-flash with
  `thinkingBudget: 0`, a one-sentence request generated at `maxOutputTokens=120`
  came back as *"Go to my"* / *"Show all files and"* — cut after ~3 words — while
  the identical call at `maxOutputTokens=512` returned the full sentence. So the
  120-token window was consumed by something other than the visible answer even
  with thinking nominally off. The practical rule: for a short structured output,
  set `maxOutputTokens` to several hundred, not to a tight bound around the
  expected length, and re-check the finishReason. Measured 2026-08-19 across 39
  command-to-NL generations, where the tight bound silently truncated the ones
  with the shortest answers. (`nl2sh-selfhist/gen_nl.py`)
- **`thinkingBudget: 0` is rejected outright by some newer models — check before
  relying on the entry below.** On **gemini-3.5-flash-lite** it is a hard
  `HTTP 400 INVALID_ARGUMENT`: thinking cannot be disabled at all. Dynamic
  (`-1`) works, but spends thinking tokens on everything — measured 2026-08-19,
  *"Reply with exactly: ok"* cost **90 thought tokens for a 1-token answer**, and
  at `maxOutputTokens=16` it returned `finishReason=MAX_TOKENS` with empty text
  (13 thought tokens), which is the silent-empty trap below reproduced on a model
  where the documented fix is unavailable. Omitting `thinkingConfig` entirely is
  the workaround. Two consequences: probe the model before assuming budget=0 is
  accepted, and **do not price a "cheap" thinking-mandatory model on its per-token
  rate** — a high-throughput helper on Flash-Lite pays a thinking tax on every
  trivial request. Related: a 4xx that is not 429 will never succeed, so fail fast
  rather than retrying — five backoff attempts turned a 0.8 s answer into minutes
  when probing four model names. (`gh-mcp-regex-fit/gemini_client.py`)
- **Gemini 2.5/3.x thinking models eat the whole output budget.** Where
  `thinkingBudget = 0` *is* accepted, set it for structured-extraction calls or you get
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

- **An execution-scored eval is capped by what the container may run, and no
  wider fixture lifts it — pick the corpus for the scorer.** Building the
  `funceq` fixture from every path the *gold* commands name (113 files) still
  leaves **0.22 coverage** on the cyber corpus: 48 of 164 golds exit 127
  because `nmap`, `john`, `fcrackzip` and `msfconsole` are absent by design, 34
  golds and 19 predictions hit the deny list (`curl`, `ssh`, `scp`, `kill`),
  and 19 more exit 1 in a fixture that cannot hold their real state. Installing
  offensive tooling to score an eval is not a trade worth making. NL2SH-ALFA
  writes its paths under `/testbed` precisely so its rows execute, which is why
  it is the corpus where functional equivalence decides.
  (`nl2sh-instantiate/funceq_ext.py`, `funceq_alfa.py`)
- **A `run_in_background` Bash poll loop suspends the turn instead of freeing
  it, and a reclaimed container takes every uncommitted result with it.** The
  session that produced `nl2sh-instantiate` finished the whole grid, then
  issued `until grep -q DONE …; do sleep 20; done` with `run_in_background:
  true` to wait for the last job. It never woke: seven user messages went
  unanswered over nine hours and the container was reclaimed with the scripts,
  both fine-tuned checkpoints and every `results_*.json` uncommitted. The
  scripts were recoverable from the transcript and re-ran to identical numbers
  (0.427 routing, exactly), the checkpoints were not. Two rules follow:
  **detach long jobs with `nohup` and poll them with short foreground checks**,
  and **commit each artifact as it lands**, not at the end of the pipeline.
  (`nl2sh-instantiate/RESULTS.md`)

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
- **A templated query generator overstates how much structure a real request
  carries, and every regex-router number in this repo is measured on the easy
  side of that gap.** Over the same 79 GitHub MCP routing targets, generated
  queries contained the `owner/repo` their tool requires **61-77%** of the time
  and had every required argument extractable **56-66%** of the time;
  hand-authored queries scored **13.5%** and **14.9%**. A template that renders
  `{pr}` renders it every time, while a person says "go ahead and merge it" —
  the referent was established several turns earlier. `monad-bsky`'s 0.833 was
  scored on 62 queries that all carried their handle, post URI or DID. Before
  trusting any structural router's eval, measure cue presence on requests you
  did not generate; and note the collapse is worse for argument *binding* than
  for tool choice, so a deterministic prefilter belongs on conversation state
  rather than on the current sentence. (`gh-mcp-regex-fit/context_probe.py`)
- **Fitting on structural cues alone is a one-minute test of whether a catalogue
  will take a regex router at all.** Inducing a decision list from the 23
  structural cues with no lexical features covered **4.9%** of training rows on
  the GitHub MCP catalogue and scored 0.029/0.013 held-out, against a Bluesky
  catalogue where the same layer did most of the work. The diagnostic is in the
  schemas: `owner` and `repo` appear in **40 of 50** GitHub tools, so the
  dominant cue discriminates nothing, while `pullNumber`, `path`, `sha` and
  `run_id` reach 8, 5, 4 and 2 tools. Run the cues-only fit before committing to
  a routing design — it distinguishes "signposted by argument shape" from
  "distinguished by intent" quantitatively rather than by eye.
  (`gh-mcp-regex-fit/fit.py --vocab cues`)
- **A fitted rule set loses to a hand-written one because a fitter can only
  learn the surface forms it was shown.** Greedy precision-constrained covering
  (CN2/RIPPER shape, <=2-literal conjunctions, abstain rather than fall back)
  scored **0.984** on the phrasing family it was fitted to and **0.239** on a
  held-out family, where rules written by hand from the same schemas scored
  0.696 and **0.546** (McNemar p=1.7e-44). The learned rules are not memorising
  entities — the entity pools were disjoint — they are sensible rules like
  `tok:diff -> get_diff` that never fire on "what code does this PR actually
  change", where the rule's author writes `\b(diff|patch|changeset)\b` from knowledge of
  the language. Laplace-corrected scoring, min-coverage 8 and dropping bigrams
  each moved coverage and precision without moving accuracy (0.191-0.227). The
  one intervention that helped computes features against the schema at inference
  time instead of learning them. If you fit routing rules, budget for a synonym
  source the training queries do not contain. **And note who that author is:**
  the "hand-written" comparison arm was Claude reading the 50 schemas, not a
  person — so the real contrast is model reasoning compiled once into
  deterministic rules versus statistics fitted from a corpus, and the compiled
  arm won. Budget an offline model pass to author the rules; the artefact still
  runs at 0.04 ms with no inference cost. (`gh-mcp-regex-fit/fit.py`)
- **Price a catch-all fallback rule by ablation before shipping one: it can take
  abstention to zero for no accuracy.** The same 73 hand-written rules with and
  without a catch-all scored abstention 0.925/0.950/0.867 against
  **0.000/0.000/0.000**, for accuracy deltas of +0.000, +0.000 and **+0.014**.
  This isolates `monad-bsky`'s 0.500 -> 0.183 refusal collapse to exactly that
  rule rather than to the rule set. A structural router should abstain and hand
  off; it should never own the decision that no tool applies.
  (`gh-mcp-regex-fit/handwritten.py --fallback`)
- **Two *deterministic* routers agreeing gates as well as two models agreeing,
  at microseconds instead of 11x latency.** A fitted decision list and a
  hand-written rule set naming the same target were right **0.775** (held-out
  templates) and **0.867** (hand-authored) over ~0.20 coverage, against 0.628
  and 0.667 ungated — the same shape as `monad-bsky/synergy.py`'s two-model
  0.880 at 0.455, without a second model in the loop. Weaker independence than
  two models have (shared cue layer and catalogue), so read it as optimistic;
  where they disagreed the hand arm was right 0.518/0.533 against the fitted
  arm's 0.202/0.333. (`gh-mcp-regex-fit/agreement.py`)
- **Dispatcher `method` enums are not where routing accuracy is lost.** Seven of
  the 50 GitHub MCP tools take a required `method` enum (`pull_request_read`
  alone has nine), turning 43 tools into 79 routing targets. Given the right
  tool, the hand-written router picked the right method 0.972/0.927/0.938 across
  three splits; choosing among the 43 tools is where every error lived.
  Collapsing a catalogue into fewer dispatcher tools to keep the tool count down
  looks free on this evidence — relevant given that declaring a sixth tool to
  Cactus Needle costs a fixed ~750 ms. (`gh-mcp-regex-fit/eval.py`)
- **A fallback that can score is worth +0.14 accuracy at zero abstention cost; a
  fallback that cannot is worth +0.014 at all of it.** Same catalogue, same
  precise hand-written rules in front: an unconditional catch-all took abstention
  0.867 -> 0.000 for +0.014 wild accuracy, while a *thresholded* encoder in the
  same slot gave **+0.136** wild accuracy (0.486 -> 0.622) with abstention
  unchanged at 0.867. On the hand-authored split the abstention line is flat for
  every threshold down to the selected one, because the scored arm is simply
  never confident about off-topic rows — a property no catch-all regex can have.
  It is also nearly free: 82% of requests never reach the scorer, so the cascade
  median is 0.088 ms against 0.071 ms for the rules alone, paying the 2-4 ms
  encode only on the tail. Cascade, do not fall back.
  (`gh-mcp-regex-fit/cascade_arms.py`)
- **In an API catalogue grammatical number is semantic, so stemming and
  lemmatisation delete the most discriminative token you have.** Plural names a
  list endpoint, singular a fetch-one. Lemmatising collapsed `branches` (idf
  4.37) to `branch` (2.76) and `issues` (2.98) to `issue` (1.66), dropping
  method-accuracy-given-tool from 1.000 to 0.500; Porter stemming merged 96 stems
  across `tag`/`tags` (`get_tag` vs `list_tags`), `workflow`/`workflows`,
  `commit`/`commits`, for 16 fixed against 83 broken on one split. Two
  independent implementations reached this by different routes. The exception:
  stemming *helps* recall@10 (0.892 -> 0.960), pulling gold into a shortlist
  while pushing it off the top — **stem if you are shortlisting, not if you are
  deciding.** (`gh-mcp-regex-fit/bm25_arms.py`, `spacy_arms.py`)
- **A lexical ranker is a good shortlister and a poor router; measure recall@k
  before judging it by top-1.** BM25 over training queries scored 0.635 top-1 on
  hand-authored GitHub MCP requests but **0.851 at k=5 and 0.919 at k=10** over
  79 targets, at 0.03 ms. That reframes tool-catalogue sizing: `needle-bsky`
  measured a fixed ~750 ms penalty for declaring a sixth tool to Cactus Needle,
  and a 0.03 ms shortlist keeping 85% recall at k=5 is how not to pay it. Also:
  **RRF lost to weighted sum at every weight** (0.554 vs 0.622) because the two
  component arms were of very unequal quality and only a weighted sum can
  down-weight the weak one. (`gh-mcp-regex-fit/bm25_arms.py`)
- **spaCy's general-English vectors do not supply domain synonymy, and the
  zero-lexical-overlap slice is how to prove it in one measurement.** On rows
  sharing no word with any label text — where lexical routing is structurally
  blind, n=77 on one split — an IDF-weighted `en_core_web_md` vector arm scored
  **0.000**, against 0.108 on rows lexical matching can see. Vectors moved "has
  anyone approved it" from lexically unreachable to **rank 21 of 79**: findable,
  not routable. Every spaCy arm lost to a 20-line IDF schema-overlap control with
  no spaCy in the path, at **250x** the latency (8.2 ms vs 0.03 ms). What the rule
  author (an LLM, offline) supplies writing `\b(diff|patch|changeset)\b` is the fact that those name one
  concept *in this API* — not a fact about English, so no general encoder has it.
  (`gh-mcp-regex-fit/spacy_arms.py`)
- **An agreement gate pays in proportion to how independent the second voter is.**
  Hand-written rules agreeing with a fitted decision list — which shares their cue
  layer and catalogue — gated at 0.192/0.775 (held-out) and 0.203/0.867 (wild);
  the same rules agreeing with a sentence encoder, which shares neither, gated at
  **0.355/0.864** and **0.351/0.923**, roughly 1.8x the coverage at higher
  precision. `monad-bsky` established that agreement beats a calibrated
  confidence head; the gradient is the new part, and it says to pick a second
  voter that fails differently rather than one that is merely separate.
  (`gh-mcp-regex-fit/cascade_arms.py`)
- **A held-out query family written to avoid the training family's verbs is
  adversarial toward the schema, not just toward the fitter — check it with a
  zero-parameter ranker.** A BM25 arm over schema text fits nothing, yet dropped
  0.611 -> 0.200 across the two families, a -67% fall against the fitted decision
  list's -76%. Since a zero-parameter ranker cannot overfit, most of that gap is
  a property of the split. Confirmation: **hand-authored queries outscored the
  "held-out" generated ones on every schema-reading arm** (0.405 vs 0.200 BM25,
  0.378 vs 0.260 encoder). Generated held-out families of this construction
  overstate generalisation loss; a zero-parameter baseline separates the split's
  difficulty from the model's failure for the cost of one run.
  (`gh-mcp-regex-fit/bm25_arms.py`)
- **A local module shadowing a PyPI package fails as `AttributeError`, not
  `ImportError`, so dependency guards do not catch it.** spaCy's dependency chain
  imports a package named `catalogue`; a `catalogue.py` in the working directory
  shadows it and `import spacy` dies with "module 'catalogue' has no attribute
  'create'", which reads like a broken spaCy rather than a name collision. Guard
  optional-arm imports with `except Exception`, and hide the directory plus swap
  `sys.modules` around the third-party import if a rename is not possible.
  (`gh-mcp-regex-fit/spacy_arms.py`, `arms.py`)
- **Truncating schema text in a harness silently handicaps every arm that reads
  it — measure the cap before trusting a "the schema lacks the words" finding.**
  A 240-character cap on parameter descriptions cut the `method` enum glosses on
  exactly the three GitHub MCP dispatchers that document them (`pull_request_read`
  kept 3 of 9). Rebuilding at full length doubled the glossed text and moved
  held-out accuracy only 0.200 -> 0.224, so the finding survived — but it was the
  harness, not the catalogue, that was being measured until it was checked.
  (`gh-mcp-regex-fit/catalogue.py --full`)
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

- **A model whose layer stack is an `nn.scan` can be grown to arbitrary depth
  by concatenating along one axis — check for a scanned stack before assuming
  depth is fixed at export.** In Cactus Needle 2 every per-layer tensor carries a
  leading `num_layers` axis (block weights inside the scanned collection, MHC
  lane parameters as explicit `(L, ...)` arrays) and nothing downstream hardcodes
  the count, so growth is `np.concatenate` plus a config bump. Verified
  byte-identical (`max |delta| = 0.0`) at 27 -> 31 and 27 -> 48 layers with
  identity-initialised blocks. The identity recipe is worth stealing for any
  gated-residual architecture: drive the residual gate's pre-sigmoid parameter to
  a value that underflows (a zero-init gate is `sigmoid(0) = 0.5`, **not**
  identity — the most common way to get this wrong), zero the second branch's
  output scale, and set any learned mixing matrix to a logit scale large enough
  that its normaliser returns the identity. Note an init that is *called*
  identity may not be one: this checkpoint's `_res_identity_init` is `4.0 * I`,
  which Sinkhorn maps to ~1.8% off-diagonal mass; 40 is needed for an actual
  identity. (`needle-depth-growth/README.md`)

- **Growing a small model deeper is blocked by the corpus, not the checkpoint.**
  The grown layers contribute exactly nothing until trained, and training them is
  full-parameter pretraining: a LoRA-only trainer cannot express it, and 800
  templated task rows do not fit 4.4M newly-added parameters. Before reaching for
  depth up-scaling on a shipped small model, price the pretraining run — the
  surgery is the cheap half by two or three orders of magnitude. Related: check
  the architecture's own depth ablation first, which for this family is U-shaped
  at iso-param with a 20-layer optimum at d_model 512 while the shipped model is
  already at 27. (`needle-depth-growth/README.md`)

- **A tool schema routes a small model through two near-equal channels — the
  tool's name and its description — and neither alone is enough.** Measured over
  one 18-tool catalogue and 54 queries with a 2x2 of {names opaque
  (`tool_01`..`tool_18`)} x {descriptions replaced by a constant string}: both
  intact **0.611**, names only 0.444, descriptions only 0.407, neither **0.074**
  against a 0.056 chance floor. Deleting either channel is significant (paired
  McNemar **p=0.035** and **p=0.019**); the difference *between* them is 0.037 at
  p=0.82. Against the floor, names are worth 0.370 and descriptions 0.333 while
  both together are worth 0.537, so roughly a quarter of each is carried by the
  other. Budget schema effort across both, and do not assume prose is where the
  routing lives. (`needle-tool-naming/RESULTS.md`)

- **Improving tool names is not a lever; wrecking them is.** A systematic
  rewrite of all 18 names to `<verb>_<distinguishing object>`, descriptions
  unchanged, reproduced the original to three decimals at both catalogue sizes
  (0.611 flat, 0.778 oracle-5, paired **p=1.00** both). Mechanically rotating the
  same names onto confusable neighbours cost 18.5 points. So a name that is
  merely *adequate* is already spending its full value, while a misleading one is
  expensive — spend review effort on catching collisions, not on polish.
  (`needle-tool-naming/RESULTS.md`)

- **Uniform, mutually-parallel tool names flatten a calibrated confidence head
  even when accuracy is unchanged — check the gate after any schema edit, not
  just the accuracy.** The rule-written naming above matched baseline accuracy
  exactly and dropped confidence separation (mean on correct minus mean on wrong)
  from **0.191 to 0.101**, because mean confidence on *wrong* calls rose 0.392 ->
  0.480. For a model whose deployment case is act-above-threshold-escalate-below,
  that is a straight regression delivered by a change that looks free in the
  accuracy column. The mirror case is informative too: when a tool's name and its
  description disagree, the model can still answer correctly off the description
  and its confidence collapses (0.584 -> 0.167 on the same query).
  (`needle-tool-naming/RESULTS.md`)

- **A contrastive tool-retrieval head reads descriptions, and a wrong name hurts
  it more than a missing one.** Retrieval cost (oracle-5 accuracy minus
  full-catalogue accuracy) over the same variants: **0.056** with opaque names and
  real descriptions, **0.185** with real names and no descriptions, **0.278** with
  names rotated onto neighbours. If a catalogue is too big to render and a
  retrieval head is picking the subset, the descriptions are what it selects on —
  but note the decode uses them too, since the +26pp `auto`->`tuned` description
  rewrite in `needle-bsky` survives an oracle catalogue at +20.4pp.
  (`needle-tool-naming/RESULTS.md`)

- **Read the fine-tuner's target list before reasoning from an architecture
  paper to why a fine-tune failed.** Cactus's attention-only paper
  (arXiv:2607.18363) localizes a SAN's content storage to the attention output
  projection, which invites the conclusion that a LoRA leaving two categories
  unmoved must have missed that write path. It does not:
  `needle/model/finetune.py` sets `LORA_TARGETS = ("q_proj", "k_proj", "v_proj",
  "gate_proj", "out_proj")` and, resolved against the shipped checkpoint, reaches
  **28.31M of 45.21M** parameters. What it cannot reach is the two Engram
  n-gram tables (8.39M, **18.6%** of the model), their key/value projections, the
  token embedding, the contrastive head and the confidence head. Ten minutes with
  the checkpoint replaced a plausible story that would have cost a two-hour
  rerun. (`needle-tool-naming/ERRORS.md` #1)

- **A shipped model is usually its architecture paper plus the parts the paper
  concluded it needed.** Cactus Needle 2's model class is literally
  `SimpleAttentionNetwork`, but the 45,211,383-parameter checkpoint also carries
  `Engram` (hash-indexed n-gram lookup tables at two layers, 18.6% of
  parameters — the parametric store the paper concludes an attention-only stack
  lacks) and `HadamardMLP` (Walsh-Hadamard channel mixing, 3x512 diagonal
  parameters per block, nonlinear composition at near-zero parameter cost).
  Reading the paper as a description of the product overstates what was removed.
  (`needle-tool-naming/RESULTS.md`)

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

- **Rewriting the corpus beat every retrieval technique tried against it, and it
  is the cheapest thing on the list.** Documentation is written in the vocabulary
  of the thing documented; requests arrive in the vocabulary of the goal. A
  `gemini-3.5-flash-lite` pass over 6,397 shell-documentation pages — adding a
  normalized summary, a category, 4-8 goal-level phrasings a user would type
  without knowing the tool's name, and a short disambiguation, with the original
  examples kept verbatim — took gold-in-sources from 0.311 to **0.427** on BM25
  alone (24 wins / 5, p = 0.0005) and end-to-end routing from 0.128 to **0.226**
  (p = 0.0052), against 0.201 for a trained query adapter and 0.165 for adding a
  dense arm. **2 hours 5 minutes at concurrency 2, one refusal in 6,397 pages**,
  and it composes with the adapter for 0.555 / 0.250. Reach for the corpus before
  the retriever: Pleias' Redline reports the same ordering for a 321M offline
  legal assistant — "most of the project's effort went not into the model but
  into converting the Guidebook into a corpus a small model can use reliably".
  (`nl2sh-dense/enrich.py`, pleias.ai/blog/local-ai-for-knowledge)

- **When one model writes your corpus and a sibling wrote your eval, the lift is
  unreadable without a human-authored control — and the control needs building
  before the number arrives, not after.** `gemini-3.7-flash` authored the eval's
  natural language and `gemini-3.5-flash-lite` the corpus's query vocabulary, so
  a recall gain could be a better corpus or one model family agreeing with
  itself. A 300-row control whose English came from human annotators settled it:
  the fused gain there (+0.098, 42 wins / 13, p = 0.0001) matched the Gemini
  eval's (+0.116). Two traps in building it, both hit: NL2Bash uncapped is 50%
  `find`, so cap per utility or the control measures the prior; and excluding the
  rows a query adapter trained on leaves **97.7%** `find`, so a control cannot
  serve both an adapter arm and a corpus arm — the adapter arms keep their own
  coverage caveat instead. (`nl2sh-dense/make_control_eval.py`)

- **Split a corpus-rewriting gain by whether the rewrite was reachable from the
  original text; a uniform lift is the signature of an artefact.** Query the
  *plain* corpus with each generated intent line and record whether its own page
  comes back — circular against the enriched corpus, clean against the plain one.
  Utilities whose intents merely echoed their page gained **+0.086**; those whose
  intents added vocabulary the page did not contain gained **+0.224**, starting
  0.126 behind and finishing level. The gain landing exactly on the pages that
  were unreachable is the mechanism showing itself; a phrasing-alignment artefact
  would have lifted both groups by a constant.
  (`nl2sh-dense/check_cards.py`)

- **Key a vector cache on the corpus, not just on the model and the settings.**
  A cache named `chunks_{model}_{granularity}.npy` served an enriched 6,397-row
  corpus and a plain 6,397-row corpus from the same file, and the row-count guard
  passed because the counts match by construction. Caught before it produced a
  number; it would have been silent and permanent if it had not been. This is the
  cached-matrix entry above, one level down: shape, dtype and length checks all
  pass, and the incomparability never surfaces. (`nl2sh-dense/dense_index.py`)

- **Fit a linear adapter on frozen query embeddings before pricing a fine-tune —
  and split the eval by training coverage before believing either.** With the
  gold document at recall@3 = 0.396 and recall@50 = 0.726, the candidate set
  already held the answer and the ordering was wrong, which is the condition a
  trained scorer fixes. One identity-initialized `d x d` matrix on the **query
  side only** — 4,588 NL2Bash pairs capped per utility, in-batch plus 4 mined
  hard negatives, **40 seconds on 4 CPU cores, 4.2 MB** — took gold-in-sources
  from 0.384 to **0.463** (21 wins / 8 losses, p = 0.024) on an independent
  corpus, and end-to-end routing from 0.128 to 0.201 (p = 0.058). Document
  vectors are untouched by construction, so no cached index is invalidated and
  the adapter can be switched off at query time. **But the entire gain sits on
  the 207 utilities the training data covered**: +0.184 on eval rows whose gold
  utility appeared in training (n=87), **−0.039** on rows whose did not (n=77).
  A rank-64 adapter, 16x fewer parameters, reproduces the same split
  (+0.161 / −0.052), so this is not overfitting that capacity control fixes — it
  is training-data coverage, and a full fine-tune on the same pairs would hit the
  same wall with more capacity to memorize. Two rules: run the linear adapter
  first because it costs a minute and answers whether the representation or the
  objective was the problem, and **always report the seen/unseen split**, because
  the aggregate number hides a regression on exactly the long tail retrieval
  exists to serve. (`nl2sh-dense/adapter.py`)

- **Retrieval granularity is a free knob once the consumer's context window
  allows it, and larger units won on both arms.** The shell-documentation corpus
  was chunked to one tldr example per document for Pleias' 4k window. Grouping
  its 31,169 chunks into the **6,397 source pages** they came from lifted BM25
  alone from 0.262 to 0.323 gold-in-sources at no cost in disk, encoder or query
  time, and lifted every hybrid arm by a similar margin (best chunk-level 0.354,
  best page-level 0.390). A whole page carries the vocabulary any one example
  omits, and a mean-pooled vector over a page summarises what a utility is for
  better than a vector over one example line. But **feeding** the whole page to
  the generator did nothing (0.159 vs 0.165 routing, and 0.610 vs 0.640 under
  oracle sources): the index wants pages, the prompt does not. Re-check
  granularity whenever the model downstream changes its context budget.
  (`nl2sh-dense/dense_index.py:page_chunks`)

- **BM25 and a mean-pooled encoder want different document lengths, so fuse
  after aggregating to entities rather than forcing one granularity.** Moving a
  shell-documentation corpus from 31,169 example-level chunks to the 6,397 pages
  they came from moved BM25 **+0.061** gold-in-sources and the three dense arms
  +0.012, **−0.054** and +0.006. The mechanism is not subtle — a longer document
  gives BM25 more terms to match and its length normalization absorbs the cost,
  while a mean-pooled vector averages the one passage that matched into the ten
  that did not. A mixed arm is expressible because fusion happens *after* each
  arm aggregates to the entity being ranked: a page and a chunk are not the same
  object and cannot be fused as documents, but "the score this arm gives utility
  `u`" is. Measured, page-BM25 with chunk-dense was the best cell for the model
  whose dense arm lost from pages (0.366 vs 0.341) and an exact tie for the one
  that was indifferent (0.378) — directionally right, p = 0.45 at n=164, and free
  to adopt because both caches already exist.
  (`nl2sh-dense/coarse_to_fine.py`)

- **A small RAG generator takes a fixed benefit from having an exemplar and no
  graded benefit from a better one — test with a names-only control before
  building a reranker for it.** Gemma 3 270M routing, with the gold utility
  guaranteed present so only the attached text varies: **name only 0.451**, its
  first tldr example **0.640**, the example a query-relevance pass selects
  **0.640** (identical), its whole page 0.610. So the documentation is read — it
  is worth +0.189, most of what fine-tuning bought — and query-relevant selection
  of *which* example moves 4 queries and loses 3 (p = 1.0). One example anchors
  the output format and the utility's argument shape; a better-matched one adds
  nothing the model can use, and more text is slightly worse. The practical
  consequence is that a retrieval tier feeding a model this size should spend its
  effort on producing the right k *entities*, each with any one exemplar, and not
  on selecting passages within them. The names-only arm is the load-bearing
  control: without it "the right example does not matter" reads as "the text does
  not matter", which is false by 0.189. (`nl2sh-dense/fullsystem_dense.py
  --source-form`)

- **A 23.5 MB int8 encoder matched a 164.5 MB one on a documentation corpus, so
  artifact size decides an on-device embedder, not retrieval quality.**
  all-MiniLM-L6-v2 int8 (23.5 MB) and mdbr-leaf-mt int8 (25.6 MB) scored 0.341
  and 0.311 gold-in-sources against bekko-embedding-v1-a8m's 0.354 at 164.5 MB —
  one to two queries apart on 164, and indistinguishable once fused with BM25
  (0.390 for both leaf and bekko at page level). This is the on-device corollary
  to `bekko-embedding-bench` and `mdbr-leaf-mt-bench`, which ranked the same
  models by retrieval on blog and code corpora: on a corpus this size the
  ranking does not reproduce and the footprint is the whole decision. Measure
  the small model before budgeting for the big one. (`nl2sh-dense/encoders.py`)

- **RRF cannot feed an abstention gate — it throws away the magnitudes a
  confidence signal is made of.** Fusing BM25 and a dense arm with reciprocal-rank
  fusion ranks well (0.390 gold-in-sources, best measured) and its top1-minus-top2
  margin has an AUC of **0.47-0.53** against "is the gold answer in the sources"
  — a coin flip — because every leading item's score is a sum of `1/(60+rank)`
  and those are nearly equal by construction. The same fusion as a min-max
  weighted sum reaches AUC 0.59-0.64 on the same queries. Corollary in the other
  direction: min-max normalizing a fused score puts top1 at 1.0, so its margin,
  its `top2/top1` ratio and its relative margin become one scale-free signal for
  free. Pick the fusion by what runs on top of it, not by ranking alone.
  (`nl2sh-dense/calibrate_rel.py`)

- **A threshold written in raw score units does not survive a change of corpus;
  set it as a quantile or normalize the score first.** An abstention gate at
  `BM25 margin >= 5`, fitted where scores ran 11-43, fires on 12% of security
  queries and **0%** of everyday ones where scores run 0.2-2.8. The obvious
  diagnosis — use a ratio, not a difference — is only half right: setting the
  *same absolute margin* by quantile instead of by constant gives 0.50 / 0.46 /
  0.47 coverage across three distributions, as even as `top2/top1` manages, and
  the two signals' AUCs match to three decimals. What the ratio actually buys is
  needing no calibration sample at deployment. Note also that `top2/top1` and
  `(top1-top2)/top1` are the same signal reparameterized, so testing both is
  testing one. (`nl2sh-dense/calibrate_rel.py`)

- **A degeneracy detector built on whitespace tokens misses loops inside a
  single token, and the naive fix fires on IP addresses.** `apt -f -o
  my_my_my_my…` and `user.html.html.html…` are one whitespace-delimited word,
  so a repeated-adjacent-token rule and a bigram-frequency rule both score them
  clean. The intra-token rule that catches them is
  `([A-Za-z][\w.\-]{1,9}?)\1{2,}` — **anchored on a letter**, because
  `(.{2,10}?)\1{2,}` fires on `100.100.100.4` and `8.8.8.8`, where the
  repetition is an address. Adding it moved the arm it was discovered on from
  0.024 to 0.049 and narrowed the gap being claimed from 7.6x to 3.7x, which is
  the direction an honest metric revision usually runs.
  (`nl2sh-instantiate/score.py`)
- **Report the constant-class prior for *external* benchmarks too, not only for
  evals you built.** NL2SH-ALFA's test split is 109 `find` rows out of 300 — an
  always-`find` baseline scores **0.393** on the 270-row leak-free slice, so a
  0.911 headline is mostly skew. NL2Bash leans the same way (0.603). The
  non-`find` slice is the column that carries information, and it moves far
  less: 0.854 vs 0.866 between the two arms.
  (`nl2sh-instantiate/alfa_prep.py`, `RESULTS.md`)
- **A leading-token routing metric can overstate functional correctness by ~8x
  — measure execution once before building on the proxy.** Gemma 3 270M
  zero-shot scores routing 0.427 on the cyber eval and **functional accuracy
  0.055** over the same 164 rows (0.250 over the 36 rows execution can decide).
  That lands on the estimate arrived at by reading every stage-1 output by hand.
  (`nl2sh-instantiate/funceq_ext.py`)

- **An IVF served over HTTP Range wants its cells physically contiguous and its
  cell ids renumbered by centroid proximity.** A hash-partitioned IVF
  (`remex.IVFCoarseIndex`, both `lsh` and `rotated_prefix` modes) scatters the
  `nprobe` visited cells across the file and carries an `int64` permutation of
  8 bytes per vector, which is free for an in-memory scan and expensive for a
  reader that must range-fetch. hypvector does two things instead: it sorts
  rows by cluster id at write time so each cell is one contiguous byte range
  and only per-cluster counts need storing, and it renumbers cluster ids by a
  greedy nearest-neighbour walk in Hamming space so the nearest cells to any
  query land in adjacent id ranges that merge into fewer requests. Both are
  build-time-only and cost nothing per query. (`hyparam-survey/README.md`)

- **In a binary-then-rerank retriever, the candidate budget is the recall
  curve; widening the cluster probe does nothing.** Measured on 50k synthetic
  384-dim vectors against hypvector's own exact scan: scanning all 112 clusters
  instead of the default 28 moved recall@10 from 50% to 49% at
  `rerankFactor: 10`, and from 86% to 82% at 50, at 2.4x the time. Raising
  `rerankFactor` alone walked 50% → 64% → 86% → 93%. Probing wider only admits
  more Hamming ties competing for the same fixed budget. Sizing prior is
  `rerankFactor ≈ max(10, N/3000)`; a constant over-fetch is wrong at any scale
  it was not tuned for. (`hyparam-survey/hypvector_probe.mjs`)

- **Guard a 1-bit index against degenerate sign codes before trusting it.**
  Embeddings whose components are all non-negative produce near-identical sign
  codes, phase-1 ranking becomes near-random, and no amount of rerank repairs
  it. hypvector samples 4096 rows, measures expected pairwise Hamming, and
  refuses the binary path below `dim/16` (healthy mixed-sign embeddings measure
  0.3–0.5 of dimension). Centering before taking signs mitigates this but does
  not detect it. (`hyparam-survey/README.md`)

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

- **transformers.js int8 on `device: "webgpu"` silently returns a collapsed embedding
  space.** Not an error, not a warning — a plausible-looking ranked list. On
  `Xenova/gte-small` q8 over an 860-label taxonomy, every pair of labels came back about
  **0.995** cosine apart, so the order was noise: `Pillow` ranked *Wedding, Drains,
  Fabric, Flags, Candles*. The weights are fine and so was the calling code; only the
  WebGPU path is wrong. Four-way check that isolates it — fp32 PyTorch, the same
  `model_quantized.onnx` under onnxruntime, and the page's own `embed()`/`rank()` under
  transformers.js on Node's CPU backend all rank *Standard Bed Pillows* first at ~0.88,
  and the CPU path reproduces acc@1 0.434 / acc@3 0.588 over 468 queries. Pin the
  encoder to `device: "wasm"`; indexing 860 short labels is ~1s of compute, against the
  16.6s the broken WebGPU run reported, so the fast path was buying nothing.
  **A near-constant pairwise cosine is the signature** — check the spread of the index
  against itself before trusting any ranking off a new backend.
  (`hypothetical-classification/demo/`, oaustegard.github.io#345)

- **A page that ships its own benchmark and wires it only to a button cannot notice it is
  broken.** The taxonomy-snap demo carried 468 gold-labelled queries and a "score it
  yourself" control, and still shipped a wholly broken index, because nothing ran until a
  human pressed something. Moving 24 of those queries to a load-time gate — embed, check
  the gold label lands in the top 3, refuse to render below 0.25 — converts a silent
  wrong answer into a visible red state, and costs one extra batch. **Where an artifact
  already contains its own ground truth, spend it on startup, not on a button.**
  (`hypothetical-classification/demo/index.html`)

- **Quantisation cost is small here but not zero, and the printed target has to match the
  build that runs.** `gte-small` int8 scores acc@1 0.434 / acc@3 0.588 against fp32's
  0.455 / 0.594 on the 468-query WANDS task — about two points at rank 1, almost nothing
  at rank 3, for a 133 MB -> 33 MB download. The demo originally printed the fp32 figure
  as its own target, which the int8 build it runs would have missed every time; a page
  comparing itself against a number it cannot reach reads as broken even when it works.
  (`hypothetical-classification/RESULTS.md`)

- **A 57M-321M model cannot be the cheap generator in hallucinate-and-snap, at any
  interface — and the reason generalises.** Pleias `Monad` (57M) and `Baguettotron`
  (321M) score **0.425 and 0.400** acc@1 as label writers against a **0.500** no-model
  control (snap the raw query), echoing the query or bleeding from few-shot exemplars
  (`chair and a half recliner` -> `Chair & Recycling Bins`). Reframed as likelihood
  rerankers over the encoder's top-10 — asking them for no format compliance at all,
  which removes what tiny models are worst at — they score **0.325 and 0.350** against
  the same 0.500, with the gold label present in that top-10 for 82.5% of queries. The
  mechanism: what a cheap model contributes to this pattern is a **prior over how
  taxonomies name things**, not reasoning. That is world knowledge, it is the first
  thing cut when a model shrinks, and reasoning capacity does not substitute. Before
  swapping a small reasoner into any pipeline, ask whether the step is knowledge-shaped
  or reasoning-shaped; small-reasoner-big-KB does not cover the case where the KB is the
  output vocabulary and is already attached.
  (`hypothetical-classification/RESULTS.md` findings 8-9)

- **For a no-API classifier, the encoder IS the system, and `gte-small` is the knee.**
  Snapping the raw query against 860 labels with no model call: `all-MiniLM-L6-v2` 0.417
  acc@1 at 23 MB int8 ONNX, `bge-small-en-v1.5` 0.427 at 33 MB, **`thenlper/gte-small`
  0.455 at 33 MB**, `bge-base-en-v1.5` 0.462/0.630 at 109 MB. gte-small is +0.038 over
  MiniLM-L6 for 10 MB; bge-base is +0.007 over gte-small for 3.3x the download and is
  worth it only for acc@3. A server round-trip for a register-prompt label buys
  +11.6pp on top (0.455 -> 0.571) — that is the measured price of leaving the browser.
  (`hypothetical-classification/browser_embedders.py`)

- **A seeded sample over a mutable corpus is not a fixture.** Two tag-classification
  measurements drew 250 rows with `random.sample(seed=...)` from a live Turso corpus at
  call time. Writing one memory into that store between the run and its verification
  (3,052 -> 3,053 rows) shifted the draw, so the saved generations zipped against a
  different 250 and every number moved — one by 0.45. The seed pins the draw, not the
  population. Persist the sampled rows next to the generations and have the recheck read
  those. The conclusions survived a pinned re-run; only their verifiability had been
  lost, in numbers already merged into two PRs.
  (`hypothetical-classification/ERRORS.md` #6, `muninn_tags_fixture.json`)

- **Prompt a hallucinate-and-snap classifier on the vocabulary's REGISTER, never on
  novelty — and note that a weak model hides the error.** Doug Turnbull's
  hypothetical-classification pattern (cheap model invents a label, embedder snaps it
  onto the legal set) ships with the prompt *"create a novel, never-seen-before
  classification"*. That instruction is safe only with a model too weak to follow it.
  `gemini-3.5-flash-lite` half-ignores it and writes `Salon & Styling Chairs`; a
  Haiku 4.5 subagent obeys and writes `Hydraulic Styling Thrones`, scoring **0.100
  acc@1 against a 0.500 no-model control** — a fifth of doing nothing. Re-anchored on
  register (*"write the label this vocabulary WOULD file the item under, match the
  examples' register, do not worry whether it already exists"*) the same subagent
  scores 0.525/0.750. The register wording also beat novelty on Gemini across 468
  WANDS queries (0.564 vs 0.489) and by **30 points** on a distinctive 1,273-tag
  vocabulary (0.500 vs 0.200). Two consequences: swapping in a stronger cheap model
  silently breaks a deployment tuned on a weaker one, and any measured "boundary" on
  this pattern must be re-checked under the register prompt before it is believed —
  one was published and withdrawn here. (`hypothetical-classification/RESULTS.md`,
  `ERRORS.md` #2)

- **Shipping the label vocabulary beats hallucinate-and-snap by 14 points whenever you
  can afford the tokens.** On WANDS (860 labels, 468 queries), structured output over
  the full list scored **0.701** acc@1 against the pattern's **0.564**, at 5,265 input
  tokens per query against 6. The pattern still beats every model-free baseline
  (direct MiniLM 0.417, char-ngram TF-IDF 0.316), so it is the right tool when the
  vocabulary does not fit, hits a provider enum cap, or costs too much at volume — and
  the wrong one otherwise. The source post reports the pattern working and being
  cheaper, not the arm it loses to. Batching 40 items per call is free (0.496/0.641 vs
  0.489/0.613 unbatched) at 1/17 the input tokens.
  (`hypothetical-classification/RESULTS.md`)

- **A Claude Code subagent costs ~32,500 tokens before it reads your prompt.** A
  `general-purpose` Haiku 4.5 subagent asked to output the single word `ok`, with zero
  tool calls, spent **32,539 tokens** in 1,143 ms; a 40-item classification batch spent
  36,252. Per item that floor is 813 tokens at batch 40 and 32,500 at batch 1, so
  per-item subagent delegation is never the cheap option it looks like — it is ~65x
  the cost of the same call to `gemini-3.5-flash-lite` through the gateway. Batch, or
  write the output inline in the parent turn.
  (`hypothetical-classification/RESULTS.md` finding 6)

- **Char-ngram TF-IDF is a serious label snapper, not a fallback.** 0.528 acc@1 on
  WANDS against `all-MiniLM-L6-v2`'s 0.564, no download and no GPU — and it *beats*
  MiniLM outright (0.400 vs 0.296) when snapping documents that contain their own
  label words literally, which is the common case for tag vocabularies. Try it before
  paying for an encoder. (`hypothetical-classification/RESULTS.md` finding 7)

- **Query expansion lost again, in both an unsupervised and a
  cross-arm form.** Over 164 shell-documentation requests, RM3 took
  gold-in-sources from 0.262 to **0.226** and dense-PRF (feed the dense arm's
  top hits back as BM25 expansion terms) to **0.232-0.299** depending on the
  encoder. This confirms `muninn-rm3`'s prediction that pseudo-relevance feedback
  cannot bridge a vocabulary-divergent query, and adds that it actively costs.
  Dense-PRF does work on the query that motivated it — *"recover the password for
  invoices2019.zip"* goes from a top-5 with no `fcrackzip` in it to one
  containing `zip2john`, `zipcloak`, `fcrackzip` — and still loses in aggregate,
  because it damages every query whose vocabulary was already right. If a dense
  arm is available, feed its **documents** to the model rather than its
  **vocabulary** to BM25. A gate that expands only on low retrieval confidence
  remains unmeasured. (`nl2sh-dense/reformulate.py`)

- **A failure pattern that is real inside one category can be invisible as a
  main effect — size it before designing around it.** Seven queries in two
  categories all pointed at tool-name capture, and the pre-registered aggregate
  test over 54 queries and 15 categories came back at 0.037, p=0.82. The
  category-level effect was genuine and reproduced on demand (rotating names onto
  neighbours moved that category 0.250 -> 0.750 with descriptions untouched); it
  was simply 4 queries of 54. Pre-registering the prediction is what kept this a
  reported miss rather than a tidy headline built from the one category that
  moved. (`needle-tool-naming/PREREG.md`, `ERRORS.md` #2)

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

### Two metrics that share an algebraic term will correlate — derive before you report

Principle 1 above says two implementations agree falsely when they share a
*modelling assumption*. The metric-level version is sharper and easier to miss:
two scores agree trivially when they share a **term in their definitions**, and
the agreement then says nothing about the system under study.

`tc-interference-weights/RESULTS.md` (a review of external work, not our own
measurement) — Anthropic's interference-weights note scores each virtual weight
by *Fisher effectiveness* (a second-order KL estimate) and by *helpfulness*
(mean loss change under ablation), then reports that the two track each other on
the effective tail as a finding about the trained model: "a model trained to
minimize loss has every reason to get its most consequential weights pointing in
a good direction." Expanding their own helpfulness formula to second order in
`u = sw` gives `helpfulness(w) = -w*dL/dw + fisher(w) + O(u^3)` — the Optimal
Brain Damage saliency they cite in related work. `fisher` is a positive-definite
quadratic form, always >= 0 and growing as `w^2` while the gradient term grows as
`w`, so weights above a crossover in `|sw|` are *forced* to measure as helpful.
Measured on their published figure data: log-log slope 0.89-1.06, r 0.956-0.990,
median ratio 0.58-1.22 in all six weight families over six or more orders of
magnitude. The correlation is largely an identity.

The check is fifteen minutes with a pencil, and it runs **before** the compute:
write both metrics as expressions in the same primitives and see whether one
reduces to the other plus a residual. If it does, the residual is the finding —
here, the first-order term is exactly what separates helpful from harmful, which
also inverts their claim that "a sharper metric has little room to improve."

We are exposed to this wherever a cheap score is validated against an expensive
one: ERA/TWERA against Fisher, `xr` dense score against RRF rank, `utility_ok`
against `funceq`, routing against gold-in-sources. A high correlation between a
proxy and its ground truth is the *expected* reading when the proxy was derived
from the same expansion — not evidence the proxy works.

**Use when:** reporting that metric A predicts metric B, or that a cheap proxy
tracks an expensive ground truth.

### At large N, significance stops discriminating — commit an effect-size floor first

The power entry under "Numerical / ML gotchas" covers being *under*powered. This
is the mirror failure: past some N every difference clears p < 0.05, the
significance test degenerates into a sample-size report, and a headline built on
it survives no honest re-reading.

`tc-interference-weights/RESULTS.md` — with 1B tokens per weight estimate,
Anthropic's note reports 47.6% of virtual weights having positive mean
helpfulness and rests "the model is still dense in this basis" on it. Their own
appendix computes a region of practical equivalence: at their own naive
yardstick (the model's total loss budget spread over its 331M weights,
eps = 1.5e-8 nats/token), **90.1% of weights are practically zero and 2.9% are
positive** — population-weighted, ~9.6M weights against a 2.9M-parameter model.
That converges with their own helpfulness-mass estimate (2.43% density holds 90%
of positive mass) and with Fisher pruning (30% density costs 0.0107 nats). Three
routes land on 2-3%; only the sign-of-the-mean route reaches the Discussion.

The floor has to be chosen from the problem, not from the data: what change in
the outcome would alter a decision? Then report the fraction clearing it in the
main line, next to the p-value, not instead of it. An eps chosen after seeing the
distribution is a post-hoc filter and inherits every objection to one.

**Use when:** n is large enough that the confidence intervals are narrow
relative to the quantity's natural scale — big token counts, exhaustive pair
enumerations, full-corpus sweeps.

### Point `recheck.py` at the artifact that could refute the headline, not only the ones the prose cites

The existing recheck convention verifies **prose against the artifacts it
describes**. That catches sentences that misreport their own numbers. It cannot
catch a headline that is contradicted by a result sitting elsewhere in the same
repo, because nothing in the prose points at it.

`tc-interference-weights/RESULTS.md` — the effect-size analysis that undercuts
the note's central conclusion is not missing, wrong, or hidden. It was computed
correctly, rendered as a figure, and placed in an appendix that the Discussion
never cites. No prose-vs-artifact check would flag it: every sentence in the
Discussion is consistent with the artifacts it quotes.

The addition is one assertion per experiment: name the strongest piece of
evidence *against* the headline, and have the fixture recompute it and print it
next to the headline number. If they are in tension, the writeup either hedges or
explains why the counter-evidence does not bind. Relegating an analysis to an
appendix is a claim that it does not change the conclusion — make that claim
explicitly, where it can be checked.

Companion to "A check that cannot fail is not a check": a check that only reads
the evidence you already believe cannot fail either.

**Use when:** any writeup with a stated headline and an appendix.

### Read small-multiple grids with code, not with your eyes

`tc-interference-weights/RESULTS.md` — the note claims Fisher pruning beats raw
weight magnitude "for every density and individual weight family (except for
negative Features->Logits weights)". Its own `threshold_data.js` says Fisher wins
that family at all 19 sampled densities by 3-10x, and loses on a different one —
Tokens->OV->Logits negative-only, worse at 8/19 densities and up to 2.22x. Six
families times three sign conditions is eighteen small panels of two crossing
curves; the misattribution is what reading that grid by eye produces. Fifteen
lines of interpolation over the stored series settles it exactly.

Generalizes to any faceted comparison we publish: the per-cell claim comes out of
the array, and belongs in `recheck.py`.

**Use when:** a sentence names which facet of a multi-panel figure is the
exception.

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

- **Check whether your "human baseline" is a model.** This repo spent a whole
  experiment comparing "hand-written" regex rules against fitted ones and framing
  it as human knowledge versus machine learning. Claude wrote the rules. The real
  contrast was **model reasoning compiled once into deterministic code** versus
  **statistics fitted from a corpus**, and compiled reasoning won the held-out
  split (0.546 against 0.239 fitted and 0.540 for a supervised encoder) while
  running at 0.04 ms with zero inference cost. Two consequences: an "LLM-as-
  compiler" arm is cheap and belongs in any routing comparison; and if the same
  model authored both the rules and the eval queries, the number is not
  contaminated-but-usable, it is disqualified — re-run the compile in a clean
  room. **Since measured**: the clean-room re-run scored *higher* on realistic
  queries than the contaminated arm (0.540 vs 0.486) and lower on the template
  family (0.504 vs 0.546), so writing rules with the eval in view is worth about
  ±0.05 and points the wrong way. Disqualify the number, but do not assume the
  honest one is worse. (`gh-mcp-regex-fit/handwritten.py`, `gemini_arms.py`)

- **Error-driven rule iteration is model-dependent, and on a frontier compiler it
  is safe but does not transfer.** The regression documented in the entry below
  (round 3 losing 0.123 in-sample) was measured on gemini-3.7-flash and is a
  property of *that model's* revision behaviour: it rewrote the whole ordered
  list each round, so fixes kept shadowing rules that were already right. Run
  under the identical protocol, Claude did not degrade — it **stopped**, returning
  a byte-identical rule file at round 3 rather than changing anything. But the
  gain was in-sample only: family A 0.863 -> 1.000, held-out paraphrase family
  +0.041, and the hand-authored split **0.540 -> 0.540 -> 0.540** across three
  rounds, with round 1 changing 5 of 74 predictions and breaking as many as it
  fixed. So: use a capable compiler, expect **one** useful round, and treat a log
  of real failures as an **evaluation** set rather than a supervision signal —
  not because feedback is destructive, but because it does not carry past the
  rows it was computed on. (`nl2sh-retrieval/results_claude_iteration.json`)

- **Supervise a rule-writing model with its own errors, not with labelled
  examples — and stop at two rounds (see the model-dependence caveat above).** Same model, same corpus, same 79 targets,
  three regimes: shown nothing but the schemas it scores **0.161** on a disjoint
  phrasing family; shown 237 labelled rows it scores **0.050** (*worse than
  nothing*, p=2.8e-16) because it copies the examples' surface forms into its
  patterns; shown its own errors on those same rows it scores **0.219**.
  Labelled examples are a sample of the phrasing distribution and the model
  reads them as the specification. Iteration then peaks fast — round 1 buys
  +0.229 on the hand-authored split, round 2 +0.014, round 3 loses 0.123 *in
  sample* because each revision rewrites the whole ordered list and new rules
  shadow correct ones. Keep the round-2 artefact, not the last.
  (`gh-mcp-regex-fit/compile_variants.py`)

- **Telling a model to "write broad patterns" makes them narrower.** The
  obvious fix for a high-precision/low-coverage rule set is to ask for breadth.
  Holding catalogue, executor and splits fixed and changing only the
  instruction — cover every target, five to ten surface forms each, do not buy
  precision with abstention — coverage went **down** on both held-out splits
  (0.216→0.149 wild, 0.390→0.188). The model wrote more alternations and
  anchored them harder to the schema's own verbs. Coverage of unseen phrasings
  is not something exhortation supplies; error feedback is.
  (`gh-mcp-regex-fit/breadth_arm.py`)

- **Compiling deterministic rules from a schema is a model-capability cliff —
  and budget, procedure and instruction will not substitute for the tier.**
  Identical clean-room prompt, one catalogue of 79 routing targets, accuracy on
  hand-authored queries: gemini-2.5-flash-lite **0.000**, gemini-2.5-flash
  **0.013**, gemini-3.7-flash **0.176**, Claude **0.540**. The small models
  transcribe the schema signature into regex syntax
  (`get workflow (?P<workflow_id>\S+) in (?P<owner>\S+)/(?P<repo>\S+)`) and match
  only requests phrased as the schema names itself. Three interventions were run
  to test whether the gap to Claude was procedure rather than capability, and
  **none of them closed it**: instructing breadth explicitly made coverage *fall*
  (0.216→0.149); splitting the catalogue over 8 calls to cut per-call output
  pressure produced 224 rules — 2.8 per target, more than Claude's 154 — and
  changed nothing (0.176→0.162, p=1.00); two rounds of error supervision reached
  0.419, still below Claude's *zero-shot* 0.540. Price the offline compile on the
  tier you will actually ship, and do not assume a cheaper model plus more calls
  substitutes. Watch *coverage*, not precision, when grading these: a degenerate
  rule set scores 0.918 precision because a pattern matching nothing is never
  wrong. (`gh-mcp-regex-fit/chunked_arm.py`, `breadth_arm.py`)

- **Before believing a split gap, score an arm that has no stake in any split.**
  Every arm in a three-split routing eval spread 3-16x across the splits
  (0.984→0.239 fitted, 0.615→0.161 clean room, 0.696→0.546→0.486 for the eval's
  own author), which reads as generalisation loss. A live LLM answering per
  query — fitted on nothing, reading no schema vocabulary — scored
  **0.532 / 0.532 / 0.568** across the same three. The splits are equally hard;
  every gap was authorship and supervision. The reference arm costs one
  afternoon of inference and reinterprets every other number in the table. It
  also caught the contamination directly: the eval's author's rules were the
  only arm to *beat* live inference, and only on the split whose paraphrases
  that author wrote. (`gh-mcp-regex-fit/live_eval.py`)

- **A microsecond rule tier in front of an LLM router can beat the router, and
  the mechanism is complementary abstention.** Rules first, live model on
  whatever they decline, joined per row on hand-authored queries: the model alone
  scored 0.568, model-compiled rules alone 0.540, and the **cascade 0.770 while
  removing 58% of the model's calls**. The gain is not the rules being more
  accurate — they are slightly worse standalone — it is that the model *declines*
  40% of routable requests at 95.5% precision, and the rules answer a large share
  of exactly those. Two arms with correlated errors cannot do this, so measure the
  abstention overlap before predicting a cascade's value from the two standalone
  numbers. The corollary is a threshold: below a compiler-capability line the
  front tier is purely a cost lever (every weaker rule set landed at 0.635-0.649
  regardless, while calls avoided ranged 19-63%), and above it, it is also an
  accuracy one. (`gh-mcp-regex-fit/cascade_live.py`)

- **Read a model's reference implementation before inferring its prompt protocol
  from its special-token list — the failure looks exactly like incapacity.**
  Pleias-RAG-350M exposes `<|query_start|>`, `<|source_start|>`, `<|source_id|>`
  and friends, and a prompt assembled from those alone produced a **0.000 parse
  rate**: the model kept emitting further `<|source_start|>` blocks and
  degenerated into repetition. That is the same signature as `monad-bsky`'s
  zero-shot result (0 parseable calls in 62), so it reads as "this model cannot
  do the task". It was two missing details, both in the library's own
  `_format_prompt`: every block needs a trailing newline, and **the prompt must
  end with `<|language_start|>\n`**, which is the only signal that the source
  list is closed. Two minutes of reading against a discarded measurement.
  (`nl2sh-retrieval/pleias_gate.py`)

- **Pre-fill a reasoning model's scaffold when you only want its answer: 9x.**
  Pleias-RAG generates `language` → `query_analysis` → `query_report` →
  `source_analysis` → `source_report` → `draft` before `<|answer_start|>`, about
  700 tokens of preamble. Appending a minimal scaffold to the prompt so decoding
  begins at the answer span took a query from **61.2 s to 5.4 s** on 4 CPU cores
  and did not change the content of the answer. Applies to any model whose
  output structure is delimited by special tokens rather than free-form — the
  preamble is prompt, not inference. Check that it does not change the answer
  before relying on it; here it did not. (`nl2sh-retrieval/pleias_gate.py`)

- **A roff `.TP` split is not a general man-page option parser.** Chunking man
  pages by the `.TP` macro looks universal and is not: **32 of 60** man pages on
  a stock container carry zero `.TP`, because DocBook-XSL generated pages spell
  an option as `.PP` / `\fB\-a\fR` / `.br`. A chunk-size statistic computed
  that way silently describes only the subset that uses the macro — it did here,
  and the reported median was wrong. Sample the *parse failures*, not just the
  parse successes, before quoting a distribution over a document corpus.
  (`nl2sh-retrieval/build_corpus.py`, correcting `nl2sh-scoping`)

- **A tiny model that emits nothing usable zero-shot may be one fine-tune away
  from working — and the reason it works may not be the reason you picked it.**
  Pleias-RAG-350M scored **0 usable shell commands in 40** with the correct
  source in its context every time, answering in cited prose. 600 rows, one
  epoch, loss masked to the completion span, **25.5 minutes on 4 CPU cores**
  took it to **0.923 on the slice where a constant answer scores 0.000**. That
  is `monad-bsky`'s 0.000 -> 0.481 replicated on a second model: what these
  models lack zero-shot is an *output shape*, and installing one is cheap. Gate
  zero-shot, but do not conclude from a zero-shot gate.
  **The instructive part is the ablation nobody ran.** The model was chosen
  specifically because it is trained to quote sources literally, to fix a
  documented 51% identifier-copying failure in its 56M sibling. **Verbatim rate
  was 0.000 before fine-tuning and 0.000 after** — it generates the command
  rather than copying it. The result is real, the rationale that selected the
  base model is unsupported, and any other 350M model might do as well. When a
  capability argument picks your model, measure that capability in the *result*,
  not just in the model card. (`nl2sh-retrieval/pleias_gate.py`, `finetune_gate.py`)

- **Score a routing arm against a constant-answer prior before believing it.**
  Two separate measurements in one night were nearly reported against zero when
  the honest baseline was much higher: NL2Bash is **60% `find`**, so "always
  answer `find`" scores **0.675** on a 40-row gate sample, and a
  query-independent list of the 20 commonest utilities scores **0.625@1** on a
  retrieval eval whose real system scored 0.098. Both times the fix was the same
  — report the skew-free slice (non-`find`) beside the headline, where the
  constant scores 0.000 and the arm has to actually route. A corpus with a heavy
  head will flatter any method that learns the head; the prior tells you how
  much of your number is the corpus.
  (`nl2sh-retrieval/score_gate_ft.py`, `verify_retrieval.py`)

- **An eval where neither side is authored by the model under test can be worth
  0.3 accuracy — build one before trusting any number.** A 350M router scored
  **0.923** on a gate whose commands and phrasing both traced to the training
  corpus (NL2Bash), and **0.618** on the same task when the commands were real
  (a public 16k-command corpus) and the natural language was written by an
  *independent* model told not to name the target. The 0.30 gap is the combined
  cost of three flattering properties that are easy to miss: templated phrasing
  from a single author, a head-heavy label distribution (NL2Bash is 60% `find`),
  and the request naming its own answer (34.7% of NL2Bash prompts contained the
  gold utility). The fix is cheap once you have a real artifact corpus — an
  independent model writes the other side — and it is the difference between an
  upper bound and a capability. Public real-command corpora exist and are
  findable in one search (Zenodo/UCI 8136017, CC-BY-4.0).
  (`nl2sh-selfhist/gen_nl.py`, `run_independent_eval.py`)
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

### Verify the watcher, not just the work

Three failures in one session of `monad-specdec`, all in scaffolding, none in the
measurement code. The measurements got full confabulation-cascade discipline —
`lm_head(hidden_states[-1])` checked against logits to 0.0, a KV-context bug
caught that was worth 53% of the headline metric, benchmarks re-run after a
background compile contaminated them. The code that reported *whether the work
was progressing* got no checks at all, and cost a full night of compute.

**An unverified watcher is worse than no watcher: it turns a crash into a silent
stall.**

1. **Unreachable wait condition (~8h lost).** The producer exited on
   `while total < TARGET` (a token count); the waiter blocked on a *shard count*.
   The producer checks its condition at the top of the loop, so on crossing the
   target it exited mid-shard and never wrote the final partial one. That shard
   was unreachable by construction. A producer and its waiter must share **one**
   termination condition — or the waiter must also exit when the producer's pid
   is gone.
2. **`pgrep -f` matches itself (16 min of false confidence).** A liveness check
   run as `bash -c '... pgrep -f "run_sweep|eagle_train2" ...'` matches its own
   command line, because the pattern is *in* that command line. A dead job
   reported ALIVE. Use `ps -C python3`, a pidfile, or a heartbeat file the job
   rewrites every N steps — and check the heartbeat's **age**, not its existence.
3. **2× memory peak on load, killed with no traceback and no OOM line.**
   `H.append(d["h"])` for N shards then `np.concatenate(H)` holds the list *and*
   the result. Preallocate `np.empty((total, dim))` and fill in place. Silent
   death with no error still means OOM; absence of a message is not absence of a
   cause.

Cross-check liveness on a **second axis**. Failure 2 only surfaced because
`free -g` showed 0 GB resident for a job that had to hold 3.2 GB. One signal is
not liveness.
