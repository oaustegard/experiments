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
