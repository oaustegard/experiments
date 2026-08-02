# Error log — remex-vs-higgs-ablation

Every error found in this experiment, both runs, with how it was caught and
which way it pushed the conclusion.

This exists because the base rate is the single most useful calibration number
about a body of work, and it is the one nobody records. Without it the only
available options are "trust the writeup" and "trust nothing", and neither is a
decision procedure. With it you can ask the real question: *is error here
detectable and bounded?*

A worked demonstration, in three escalating steps, of why a log beats a
recollection. Asked how many errors this session had produced, the estimate
given was "roughly seven" — made an hour after the events, by the process that
made them. Counting them properly for this file gave **eight**. Then an outside
party found a **ninth**, and it was the one with the largest effect on the
writeup. Self-assessment was wrong about the count, and wrong again about the
ceiling.

**Direction** is the column to read first. An error that makes a check *more
permissive*, or that makes a conclusion look *stronger*, is far more dangerous
than one that makes it look weaker — the second announces itself when someone
tries to use the result, and the first never does.

---

## Run 2 (2026-08-02, the `gating` rerun) — 9 errors, all mine

| # | error | caught by | direction |
|---|---|---|---|
| 1 | `audit.py` probe 3 claimed to reproduce the historical +16% b=8 scalar-MSE defect. The reconstruction actually converged to the *correct* value; the prose around it contradicted the numbers printed directly above it. | reading the probe's own output | **overstated the evidence.** The conclusion (the anchor's range stops at b=5) was right; the demonstration was not. Rewritten to report a failed reproduction and carry the historical value as a literal. |
| 2 | `gate.py` compared a *paired* standard error against an *unpaired* one. They estimate different quantities; the check was meaningless as written. | the check going red on the first full run | neutral — a meaningless check reporting green |
| 3 | `gate.py` asserted the vector arm always carries more shared bytes. False at d=100 / 2 bits, where remex's 40 KiB Haar matrix outweighs the 20 KiB codebook. | the check going red | **an overclaim that would have shipped.** Narrowed to the codebook term, which does hold everywhere. |
| 4 | `gate.py` bracketed spike incoherence with a *strict* lower bound at an attainable optimum. At power-of-two d a Hadamard transform maps a coordinate spike to exactly ±1/√d, so `rht d=1024` hit the bound exactly and failed. | the gate — **it blocked the sweep** | false red. Cost an aborted sweep; the wiring working as designed. Also revealed that the spike probe is degenerate for the RHT at power-of-two d, so a random-vector probe was added. |
| 5 | `verify_kills.py` listed `grids.py:126` (`k > 1` → `k >= 1`) as a real hole, with a description belonging to a different line. It is an equivalent mutant: `k = 2**bits ≥ 2` at every rate the sweep uses. | the fixture reporting it SURVIVED | **would have claimed a check catches something it cannot.** Moved to the documented equivalent-mutant list. |
| 6 | The zero-gain known-bad was built from one Lloyd iteration (to avoid a degenerate zero standard error) and was **not zero-gain**: at m=8 / K=65536 with 2,000 samples, one iteration relocates ~63,000 empty-cell codepoints toward the mode and earns a real +0.10 dB. | the gate, at full size — **it passed in fast mode (m=2, K=16) first** | **a known-bad that certified nothing.** The lesson generalised into METHODS.md: validate a known-bad at the configuration it will run in. |
| 7 | `RESULTS.md` and `README.md` stated 13 coverage limits after a 14th was added. | re-reading `gate.log` against the prose | prose drift. Now caught mechanically by `recheck.py` phase 3. |
| 8 | Repeated an inherited claim that `uvx ruff check .` passes from the repo root. It reports 1,089 errors (582 in `ms13-campaign`), none in this directory. | running the command instead of trusting the sentence | **an unverified inherited claim**, propagated from the first run's writeup. |

**Detection attribution.** Gate went red: 4. Standing fixture: 1. Re-reading
output against artifacts: 2. Running a command rather than trusting prose: 1.

### 9 — caught by someone else, after the branch was already open

| # | error | caught by | direction |
|---|---|---|---|
| 9 | **"RHT is 11–24× slower to apply in numpy."** Reported as a result across `RESULTS.md`, `README.md` and the PR body. It measured a butterfly running two full-array copies per stage in interpreted numpy against one tuned `sgemm` — the implementation, not the transform. Corrected upstream in PR #10: ~1.2× slower at d=768, parity at d=1024, **3–4× faster at d=4096–8192**, crossover near d≈1024. | **an outside party**, in a review of the merged first run, while this branch was open | **the flattering direction for remex**, and the writeup said so out loud without acting on it: *"This is a fact about numpy, not about the algorithm... A fused FWHT would change this entirely."* The caveat was correct and the number was still printed as a finding. |

This one is the most instructive entry in the file, for three reasons.

**The disclaimer was not a substitute for the fix.** Flagging that a measurement
is implementation-bound does not make it safe to report as a result. If a claim
is known to be an artifact of the harness, the options are to fix the harness or
to drop the claim — not to publish it with a footnote. Both runs published it.

**The `gating` work did not touch it.** The whole rerun was about whether checks
can fail, and the gate grew from 8 checks to 139 — none of which looked at axis
A's timing, because the gate certifies *codebooks and rotations for
correctness*, not *performance claims*. A performance number has no anchor in
that gate at all. The confidence grading added later called this claim
**MEASURED** — "one machine, one BLAS, a contended 4-core box, one run" — which
was the right tier and still understated it: the problem was not variance, it
was that the two arms were not comparably implemented.

**It is the first entry here found by someone outside the process.** The other
eight in run 2 were found by machinery the same process built. That machinery is
good at "does this code do what its author thinks", and structurally blind to
"is this comparison fair" — which is a question about intent, not behaviour.
Anchors and mutants cannot supply it.

**Consequence for this branch:** the sweep was re-run in full against the merged
code (a changed RHT and a fourth corpus), and every axis-A number in the writeup
was replaced rather than patched.

Note what did *not* catch anything: careful reading of my own code before
running it. Every one of these was found by execution, by comparing two
artifacts, or by an outside party — none by inspection alone.

Note also #6: it passed the fast configuration and failed the full one. A
cheap check that runs often is not a substitute for running the real one at
least once.

---

## Run 1 (2026-08-02, the original) — 8 errors

| # | error | caught by | direction |
|---|---|---|---|
| 1 | Lloyd seeded from a random sample of the source (textbook LBG init) converged to grids **worse than the scalar quantizer** at 6 and 8 bits. | the calibration gate, G3 | **would have shipped "scalar wins axis C at high rate"** — the exact wrong conclusion the commissioning issue warned about, and directionally plausible enough to survive review. |
| 2 | `grid_m2_K65536.npz` was a stale artifact of the *pre-fix* trainer, held-out MSE 7.71e-5 against scalar's 4.13e-5. It survived a cache wipe because a dying background trainer rewrote it after the delete. Root cause: the cache was keyed on `(m, K)` — the *problem* — not on the *method*. | scheduled adversarial review | **would have reproduced the same fake result** the gate had already caught once. Fixed with a `GRID_VERSION` stamp; a file whose stamp does not match is deleted rather than trusted. |
| 3 | `lloyd_max_1d` returned distortion via the fixed-point identity `MSE = 1 − Σpᵢyᵢ²`, which holds only at the fixed point. At b=8 Lloyd has not fully converged, and it returned **4.791e-5 against a true 4.127e-5, +16%**. | scheduled adversarial review | **the worst direction available.** The scalar MSE is the threshold the vector arm must beat, so inflating it made the axis-C check *more permissive* exactly where the vector arm was weakest. Max (1960) stops at 5 bits, so the gate was structurally incapable of seeing it. |
| 4 | The claim that seeding Lloyd from the scalar product grid makes the vector arm "provably no worse than scalar" was argued, not enforced: Lloyd is monotone in *training* distortion, selection happens on *held-out*, and the largest grids get ~61 samples/codepoint where the gap reaches ~14%. | scheduled adversarial review | **a guarantee the writeup leaned on that did not hold.** Fixed by making the unrefined product grid an actual candidate. |
| 5 | At d=100 / 4 bits the block size (50) was not a multiple of the sub-vector dimension (4), so 2 of every 25 sub-vectors straddled a block boundary and had their halves scaled by different fp16 factors. | scheduled adversarial review | **confounds axes B and C**, and corrupts *only* the HIGGS-like arm rather than degrading anything uniformly. |
| 6 | G3 hard-coded `pick_m(b, 768)`, so the m=5 grids behind **every** `glove100` number at 1, 2 and 3 bits were never certified against scalar or against Shannon. | scheduled adversarial review | unchecked — a gate reporting green over grids it never looked at. |
| 7 | BGE ships a `Normalize` module that overrides `encode(normalize_embeddings=False)`, so `arxiv768` came out at exactly ‖x‖ = 1.0000, σ = 0. | noticing the cosine and inner-product tables were **byte-identical in every cell** | **made axis B untestable** — the difference between "store the exact norm" and "fold in a scale" is vacuous when the norm is a constant. |
| 8 | Cosine scoring omitted the division by ‖x̂‖, silently penalising any arm whose reconstruction has norm spread. | investigating an implausible 1-bit axis-C inversion | **manufactured a fake axis-C result** that was an artifact of the scoring, not a property of the codebook. |

**Detection attribution.** Gate went red: 1. Scheduled adversarial review: 5.
Investigating an anomaly in output: 2.

---

## What the two runs together say

**17 errors across two runs of one experiment.** Both runs were done carefully
by a process that believed it was being careful.

The distribution is the useful part:

- **Run 1's errors were mostly in the science** (a wrong trainer, a wrong
  constant, a confound). Five of eight needed a scheduled adversarial reviewer
  to find, because the gate could not see them.
- **Run 2's errors were mostly in the checking apparatus** (a meaningless
  comparison, an overclaim, a false red, a toothless known-bad). Four of nine
  were found by the gate going red on the author's own work — and the one the
  machinery could not reach (#9, an unfair performance comparison) needed a
  reviewer, because fairness is a question about intent, not behaviour.

That shift is what the `gating` skill bought. It did not reduce the error
count — it moved errors from the class "found by a scheduled human-equivalent
review, or not at all" into the class "found by machinery, immediately, at the
moment of the mistake."

**Seven of the seventeen pushed a conclusion in the flattering direction**
(run 1 #1, #2, #3, #8; run 2 #1, #3, #9). That is the number to watch. Errors that
make results look weaker get caught when someone tries to use them; errors that
make results look stronger are load-bearing until something external
contradicts them.

**Nothing here was caught by reading code carefully.** Every one was caught by
executing something, comparing two artifacts, or by an outside party — and
exactly one, run 2 #9, needed the outside party. Plan
accordingly: buy execution and comparison, not care.

---

## Convention

Every experiment in this repo should carry an `ERRORS.md` with this shape:
what was wrong, how it was caught, and which direction it pushed the
conclusion. Append rather than rewrite — a log that gets tidied loses the base
rate, which is the only thing it is for.
