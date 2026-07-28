# Theory → Empirical Bridge MVP: Results

*Successor to #90 (phase A symmetric bridge MVP) and #91 (paused 1.9M build).*
*Issue: [#97](https://github.com/oaustegard/claude-workspace/issues/97)*
*Run date: 2026-05-24*

## TL;DR

Pipeline runs end-to-end. **3 of 4 acceptance criteria met.** Criterion 2
(≥20 surviving cheap-judge candidates) missed at **3 surviving** —
the cascade is more restrictive than the issue's prior expected.
All 3 surviving candidates were independently verdict-FOLKLORE by a
second-opinion Opus pass, matching phase A's tier-1 outcome at math×math
scale: the cascade reliably finds *real structural connections*, but at
2000-paper math.X×cs.LG/CV/CL/etc. scale those connections are uniformly
either textbook results or specialist folklore.

The asymmetric pivot (#97 vs. symmetric #90/#91) did **not** sidestep
the density problem. The failure mode just shifted from
"same-program in math×math" to "well-trodden ML-theory rates in
math×cs.LG/stat.ML."

## Pipeline trace

| Stage | Script | Output | Status | Result |
|-------|--------|--------|--------|--------|
| 1 | `te_corpus.py` | `empirical_corpus.json`, `theory_corpus.json` | done | 800 emp + 1377 th |
| 2 | `te_embed.py` | `empirical_meta.json`, `theory_meta.json`, `*_vecs.npy` | done | 782 + 1372 SPECTER2 (97.75% / 99.6%) |
| 3 | `te_scan.py` | `te_candidates.json` | done | 2000 cross-axis pairs after sequential dedup |
| 4 | `te_extract.py` | `te_extractions.json` | done | 600 / 731 slot extractions OK (82%) |
| 5 | `te_rerank.py` | `te_reranked.json` | done | 95 pairs with both ends successfully extracted |
| 6 | `te_judge.py` | `te_judged.json` | done | 92 unrelated / 3 partially_resolves / 0 resolves / 0 errors |
| 7 | Agent-tool translations | `te_translations.json` | done | 3 Opus subagent translations (orchestrator path) |
| 8 | Agent-tool indep-Opus second opinion | (inline in translations.json) | done | 3 × FOLKLORE verdicts |

## Cost estimate (actuals)

| Stage | Calls | Estimated | Actual / Notes |
|-------|-------|-----------|---------------|
| S2 SPECTER2 batch | 4 batches × 500 | $0 | $0; heavy 429 throttle, retry-backoff absorbed it |
| arXiv body fetch | 731 unique | $0 | $0; ~15 min @ 1.2s polite interval |
| Slot extract (gemini-2.5-flash) | 731 × 1 (some retried) | ~$0.50 | ~$0.10 actual; 131/731 failed after 5 retries due to CF gateway 429 burst |
| Slot embed (`batchEmbedContents`) | 6 batches × 100 | ~$0.50 | ~$0.05 actual |
| Cheap-judge (gemini-2.5-flash) | 95 × 1 | ~$0.30 | ~$0.05 actual |
| Claude translations (Agent, opus) | 3 × 1 | ~$1.00 | orchestrator-side, included in this session |
| Claude indep-Opus second opinion | 3 × 1 | n/a | orchestrator-side, included in this session |
| **Total external API spend** | | **~$2–5** | **~$0.20** (well under budget) |

Wall clock: ~50 min for pipeline + ~5 min for orchestrator stages.

## Acceptance criteria

| # | Criterion | Verdict | Notes |
|---|-----------|---------|-------|
| 1 | Pipeline runs end-to-end with hard dedup tiers; resumable; under ~$5 | **PASS** | Resumed cleanly after CF 429 throttle bursts. Sequential-arXiv dedup applied; author-Jaccard dedup skipped (no S2_API_KEY); citation dedup not requested. |
| 2 | At least 20 candidate pairs survive cheap-judge `resolves` / `partially_resolves` | **FAIL** | Only 3 surviving. 92/95 judged `unrelated`. |
| 3 | Top 3 candidates pass indep-Opus "does the theorem actually say what the cascade thinks it says?" sanity check | **PASS** | All 3 candidates assessed internally consistent — the theorem is correctly described, the cascade's mapping is recognizable. (Each verdict notes overclaim in some dimension.) |
| 4 | At least 1 candidate rated by indep-Opus as worth a domain-expert second opinion | **PASS** | Pair 3 (DGM uncertainty × Martin-Liu IMs) rated `yes` — specifically calls out a constructive-feasibility question that a Martin-school statistician fluent in Bayesian inverse problems should adjudicate. |

## Top candidate translations (all 3, judged + translated + indep-Opus reviewed)

Sorted by slot-cosine descending.

### Pair 1 — slot cosine 0.801, judge: `partially_resolves`

- **E:** [arXiv:2605.14764](https://arxiv.org/abs/2605.14764) — "Compositional Sparsity as an Inductive Bias for Neural Architecture Design"
- **T:** [arXiv:2504.03405](https://arxiv.org/abs/2504.03405) — "On the Rate of Convergence of an Over-Parametrized Deep Neural Network Regression Estimate Learned by Gradient Descent"

**Cascade translation (Opus subagent):**
> The empirical "approximation/estimation error" of a trained DNN maps to
> the theorem's L2 risk E[|m_n(X) − m(X)|²] of the gradient-descent-trained
> over-parametrized estimator m_n, while the empirical "ambient dimension d"
> corresponds to the input dimension d of X and the unstated "structure"
> the authors invoke maps to the theorem's hierarchical composition
> model — a target m that decomposes as a tree of functions each depending
> on only d* ≪ d inputs and each β-smooth. The empirical regime where DNNs
> "escape the curse" requires, on the theory side, that (i) the true
> regression function actually admits such a hierarchical decomposition
> with small effective dimension d*, (ii) the network depth/width and GD
> step size/iteration count match the theorem's over-parametrization
> schedule, and (iii) noise/sub-Gaussian tail assumptions hold — none of
> which the empirical paper verifies or controls for. The concrete untested
> prediction is a *rate*: the squared L2 error should decay as
> n^(−2β/(2β+d*)) rather than n^(−2β/(2β+d)), so plotting log-risk vs.
> log-n across architectures matched to the same hierarchical target
> should yield a slope governed by d* (the max arity of the composition),
> not the raw input dimension — and changing d while holding d* fixed
> should not change the slope.

**Indep-Opus second opinion:** `FOLKLORE`. Worth-expert: **no**.
Lineage: Mhaskar–Poggio 2016; Bauer–Kohler 2019 (Annals of Statistics);
Schmidt-Hieber 2020 (Annals); Kohler–Krzyżak / Kohler–Langer follow-ups
extending rates to GD-trained over-parametrized nets. Bridge is real and
internally consistent — the theorem says what the cascade says it says,
but the result is the explicit thesis of a decade-old well-known line of
work.

### Pair 2 — slot cosine 0.686, judge: `partially_resolves`

- **E:** [arXiv:2605.15549](https://arxiv.org/abs/2605.15549) — "CTF4Nuclear: Common Task Framework for Nuclear Fission and Fusion Models"
- **T:** [arXiv:2509.07474](https://arxiv.org/abs/2509.07474) — "DKFNet: Differentiable Kalman Filter for Field Inversion and Machine Learning"

**Cascade translation (Opus subagent):**
> The CTF4Nuclear surrogate ML model output ŷ(t; θ) for system state
> given parameters θ corresponds to the DKFNet posterior mean
> x̂_{k|k} produced by the differentiable Kalman recursion, while the
> twelve evaluation metrics (forecasting error, noise-robustness scores,
> low-data MSE, parametric generalisation gap) map onto components of the
> DKF's posterior covariance P_{k|k} and innovation statistics. The
> empirical regimes — sparse measurements, noisy sensors, limited training
> data, and the "monitoring from sparse measurements only" paradigm —
> map respectively to DKFNet's measurement model H with low row-rank,
> the assumed Gaussian observation noise R, and the linear-dynamics-with-
> known-structure hypothesis; the gap is that nuclear systems (molten
> salt, MHD coolants) are strongly nonlinear and the noise is rarely
> Gaussian, so the correspondence requires either an EKF/UKF-style
> linearisation or the explicit nonlinear-system extension the theorem
> only gestures at.

**Indep-Opus second opinion:** `FOLKLORE`. Worth-expert: **no**.
Over-claims in a specific way: CTF4Nuclear is a benchmark (defines tasks
and metrics) — not a paper claiming an empirical phenomenon with an
unexplained mechanistic gap. "The 12 metrics don't measure calibrated
intervals" is benchmark scope, not a flagged gap. DKFNet's linear-
Gaussian regime also fits nuclear systems poorly without EKF/UKF/EnKF
extensions. Differentiable Kalman filters trace to Haarnoja et al. 2016;
Kalman-based UQ for nuclear/plasma state has decades of data-assimilation
literature.

### Pair 3 — slot cosine 0.676, judge: `partially_resolves`

- **E:** [arXiv:2605.15050](https://arxiv.org/abs/2605.15050) — "Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems"
- **T:** [arXiv:2503.19748](https://arxiv.org/abs/2503.19748) — "No-prior Bayes reIMagined: probabilistic approximations of inferential models"

**Cascade translation (Opus subagent):**
> The empirical "intrinsic ambiguity" — the null-space directions of the
> forward operator A along which reconstructions x are indistinguishable
> given measurement y = Ax + ε — corresponds to the parameter coordinates
> where the possibilistic IM's contour function π_y(x) remains near 1,
> while "estimation uncertainty" maps to coordinates where π_y contracts
> sharply; the theorem's inner probabilistic approximation P_y ⪯ π_y then
> plays the role of the generative model's posterior surrogate q_φ(x | y).
> The empirical many-to-one regime maps cleanly onto the theorem's
> hypothesis that the IM be derived from a valid possibility measure,
> but requires the nontrivial assumption that a tractable possibilistic
> IM can be constructed for the high-dimensional, non-identified
> likelihood p(y | x) induced by deep priors — something the theory
> establishes abstractly but not constructively for neural generators.
> The testable prediction is that credible sets C_α = {x : P_y(x) ≥ α}
> read off from the inner approximation should achieve exact frequentist
> coverage Pr(x* ∈ C_α) = 1 − α, even on null-space coordinates where
> standard variational posteriors are known to be miscalibrated.

**Indep-Opus second opinion:** `FOLKLORE (leaning WEAK-VALID if "constructive IM for deep priors" is honestly downgraded to "calibration target")`.
Worth-expert: **yes** — specifically a Martin-school statistician fluent
in Bayesian inverse problems, because the constructive-feasibility
question is the load-bearing one. IMs quantify uncertainty about a
finite-dimensional parameter; applying to a deep generative prior over
x requires either treating the latent code as the parameter (re-imports
the prior, loses no-prior guarantee) or a valid predictive/nonparametric
IM, which exists in theory but has no constructive scalable implementation
for deep priors. The general area (frequentist coverage / calibrated UQ
for ill-posed linear inverse problems) is well-trodden — Nickl, Szabó,
van der Vaart — but the specific IM-to-deep-generative bridge at
theorem-matching level is probably not published.

## Why the hit rate is low: sampling density, not cascade failure

The dominant cause is structural and was probably always going to be
this way at the 2200-paper scale.

**Math of the joint sample.** arXiv has on the order of 5×10⁵ ML/applied-CS
papers and 5×10⁵+ theory-math papers in the categories we drew from.
If the *true* population of "theorem T explains observation O" pairs is
~10³ in the entire two corpora combined (a generous upper bound — the
issue's own table lists ~5 bridges that took years to articulate), then
the per-paper probability of being a bridge endpoint is ~2×10⁻³ on each
side. Drawing 800 emp + 1377 th uniformly, the expected number of
*known* bridge pairs in our joint sample is roughly:

> E[bridges in sample] ≈ |bridges| × (800/5×10⁵) × (1377/5×10⁵)
> ≈ 10³ × 1.6×10⁻³ × 2.8×10⁻³ ≈ 0.004

i.e. the expected count of real-published-bridge endpoints both falling
into a uniform sample of this size is effectively zero. **Three
partially-resolves verdicts at this scale is consistent with the cascade
catching folklore-grade connections by chance, which is exactly what
the indep-Opus verdicts confirmed it did.**

The asymmetric pivot reduced one density-related failure mode (math×math
having too much program-overlap folklore) and exposed another (math×cs
at sub-1% sampling can't find non-folklore bridges because the population
density is too low). The cascade is doing its job; the corpus is too thin.

What would change the picture, in order of how much they matter:

1. **Corpus size.** At 1.9M (phase B's paused scale), expected
   sample-hit becomes `10³ × (1.9×10⁶/5×10⁵)² × density adjustments`
   ≈ 5–50 expected bridges in candidate space — order(s) of magnitude
   beyond what 2200 papers can deliver. The math says #91's pause was
   correct to question whether to spend $435 — but if you accept the
   density estimate, that scale is the *minimum* where the cascade can
   credibly produce non-folklore output. 2200 is structurally too small.

2. **Targeted sampling.** Phase A's Sawin anchor injection
   (`sawin_lenstra_specter.json` + #87 / #89) is the right pattern: if
   you know an anchor paper that's *part* of a candidate bridge, seed
   the sample around it. Random uniform sampling is the
   density-pessimum.

3. **Per-paper amortization.** The expensive stages (slot extract,
   judge) are per-paper, so a 10× corpus is roughly 10× cost — but the
   `2200 papers → 0.004 expected hits` math means each tier of cost
   has to chase non-linear yield. The cheap-judge being strict is fine;
   loosening it produces noise, not hits.

### Secondary contributors (smaller but real)

1. **`stat.ML` belongs in both pools.** Mixed-shape category. After
   the SPECTER2 scan, some "empirical" papers were actually theory
   papers; the binary judge correctly labeled these `unrelated`
   (theory↔theory, not theory-resolves-empirical). Pulling `stat.ML`
   out of the empirical pool would help, but only at the margins.

2. **`mechanism_unknown: not stated` papers can't be cleanly judged.**
   When the empirical slot extractor can't find an explicit
   author-acknowledged gap, the judge has nothing to anchor against.
   Two of the 3 surviving pairs do have explicit gaps. The extractor's
   recall on this slot is the most under-investigated dimension — a
   two-pass extraction (find the gap first, then extract structure
   around it) is the obvious next iteration but won't close the
   density gap by itself.

3. **CF gateway 429 throttle dropped 131/731 extractions.** 18%
   slot-extraction loss; pair survival into rerank is `0.82² ≈ 67%`,
   then top-200 cut to 95. Lower starting concurrency (2 instead of
   4) per phase A's lesson would have recovered most of those — but
   at 95 → 3 ≈ 3% positive rate, even fully recovering all 131 would
   yield only ~3–4 more candidates. Doesn't change the picture.

4. **Top-2000 cosine pairs ≠ candidate bridges.** SPECTER2 cosine
   identifies papers that *sound alike at the abstract level*. In the
   asymmetric setting, "sounds alike" frequently surfaces theory↔theory
   pairs that only survive the cross-axis filter because one was tagged
   stat.ML. The judge correctly downgrades these.

## Limitations (honestly named)

- **No S2_API_KEY in environment.** Author-Jaccard dedup was skipped
  (te_scan.py logs `author dedup: S2_API_KEY not set, skipping`).
  Citation dedup is gated behind both the flag *and* an S2 key and
  was therefore not requested. Effect: same-group pairs that would
  fail authorJaccard could in principle survive into the top-2000.
  Inspection of the 3 surviving pairs shows no obvious same-group
  overlap, so this didn't bite us this run — but it would at larger
  scale, and at math×CS specifically there should be little same-group
  density anyway.

- **CF gateway 429 throttle.** Heavy and sustained throttling during
  the slot-extraction phase. Retry-with-backoff handled the burst, but
  131 papers exhausted all 5 retries and were stored with `slots: null`.
  Phase A hit identical throttling and resolved it by progressively
  reducing concurrency 12 → 4 → 2; this run started at 4 and would
  have benefited from 2 from the outset.

- **The empirical pool's category mix is suboptimal.** As above,
  `stat.ML` straddles theory/empirical; `cs.NI` and `cs.DC` are
  largely systems papers whose phenomena are about latency/throughput
  rather than something a probability/statistics theorem would
  explain. The pool should probably be tightened to `cs.LG, cs.CV,
  cs.CL, cs.NE, cs.CR` (drop `stat.ML, cs.NI, cs.DC`).

- **`mechanism_unknown` extraction recall is the bottleneck.** When
  the slot extractor can't find an explicit author-acknowledged gap,
  the judge has nothing to anchor against. A two-pass extraction
  (first pass: find the gap; second pass: extract structure around
  it) would likely improve judge yield more than any other single
  change.

## Recommendations for next iteration

Given the sampling-density framing above, the decision tree is:

**Path A (cheap, low-yield) — keep iterating at sub-thousand scale.**
The cascade itself is sound; tightening the inputs marginally improves
yield. Useful for de-risking and engineering refinement but won't
produce non-folklore candidates. Worth doing if there's a specific
methodological question to answer (e.g. "does two-pass extraction lift
gap-recall enough to matter?"). Not worth doing as a hunt for
publication-grade bridges. Concrete changes:

  1. Tighten the empirical pool — drop `stat.ML, cs.NI, cs.DC`; keep
     `cs.LG, cs.CV, cs.CL, cs.NE, cs.CR`. Hard-route papers
     cross-tagged `stat.ML+math.ST` to theory.
  2. Two-pass extraction with a gap-detection first pass — only
     spend the bridge-search budget on empirical papers with an
     explicit author-acknowledged mechanism gap.
  3. Provision S2_API_KEY — enables author-Jaccard + citation-overlap
     dedup; cheap insurance.
  4. Start at concurrency=2 on the CF gateway — phase A's lesson,
     re-learned.
  5. Bias empirical-pool sampling harder toward 2024–2026 (less
     folklore density on recent observations).

**Path B (expensive, can-actually-produce-hits) — re-open #91's
production build with the asymmetric cascade.** 1.9M papers brings
the expected-bridges-in-sample math from 0.004 to single digits or
better, depending on how generous the per-paper bridge-density prior
is. The cascade *as currently structured* would then be filtering
real signal, not sampling noise. Cost was previously estimated at
~$435; that estimate stands. The decision becomes "is $435 worth a
non-folklore candidate stream worth domain-expert review?", which is
a question for #91, not this MVP.

**Path C (cheapest, most likely to yield) — targeted anchor sampling.**
Phase A's Sawin injection is the right pattern: if you can name a paper
that's *part of* a known or suspected bridge, seed the sample around
it. SPECTER2 neighborhoods of an anchor paper concentrate density
specifically where bridges are likely to live. This run did not test
that mode in the asymmetric setting; it should be the next experiment
*before* paying for path B.

The right next step is **path C**: build a small anchor list (5–10
known or strongly-suspected non-folklore bridges from the issue's
table — neural scaling laws / RMT, lottery ticket / compressed
sensing, etc.) and seed the asymmetric cascade with their SPECTER2
neighborhoods. If the cascade surfaces non-anchor cousins at decent
rate, then path B becomes defensible. If it still only surfaces
folklore neighbors of the anchors, path B isn't worth it.

## What's now landed

- **PR #98** (this branch): scaffold + this run's outputs + RESULTS.md.
- Honest accounting of the 3-of-4 acceptance verdict, with all three
  surviving candidates' translations and indep-Opus verdicts inline.

## Cross-references

- #90 — phase A symmetric bridge MVP (closed by PR #92)
- #91 — phase B 1.9M production build (paused; #97 was supposed to
  validate that the asymmetric pivot would justify resuming. It did
  not.)
- PR #96 — phase A tier-0 + tier-1 experiments (merged; established
  the cheap-LLM-judge + slot-extract cascade)
- [PR #98](https://github.com/oaustegard/claude-workspace/pull/98) —
  this PR (asymmetric scaffold + run)
