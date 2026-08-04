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
`remex-vs-higgs-ablation/ERRORS.md` logs 16 errors across two runs, 6 of them
in the flattering direction, and records two things worth generalising: **an
estimate of one's own error count made an hour later was wrong** (seven, versus
eight counted), and **not one of the sixteen was caught by reading code
carefully** — every one came from executing something, comparing two artifacts,
or an outside party. Buy execution and comparison, not care. Append, never tidy;
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
- **Rotated *unit* vectors have Beta-distributed coordinates, not Gaussian**
  (density ∝ (1−x²)^((d−3)/2)); TurboQuant fits its Lloyd-Max to that Beta.
  Using a Gaussian at σ=1/√d instead costs ≤0.007% excess MSE at 2 bits and
  ≤0.43% at 6 bits for d=100, and ~0% for d ≥ 768 — measured, so you can skip
  the Beta with a citation rather than a hope.
  (`remex-vs-higgs-ablation/beta_check.py`)

---

## Cache and measurement hygiene

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
- **Hand-rolled q4 ONNX export is dominated by the model authors' official
  one** — official was better on nDCG, cosine, recall-vs-fp32-kNN and Spearman ρ
  *and* 32MB smaller. Check for an official/Optimum export before building your
  own. (`q4-official-vs-ours/RESULTS.md`)
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
