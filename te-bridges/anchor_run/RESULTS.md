# Theory → Empirical Bridge MVP: Anchor-Mode Run (Path C)

*Successor experiment to #97 (uniform-sampling MVP). Same cascade,
anchor-seeded candidate assembly per path C in #97's RESULTS.*
*Run date: 2026-05-24*

## TL;DR

Anchor mode raised the cheap-judge positive rate from **3.2% → 4.8%**
(uniform: 3/95; anchor: 7/145) but **did not lift the quality ceiling**.
Indep-Opus second-opinion on the top 3 candidates: 2× FOLKLORE, 1×
DIRECT-DOWNSTREAM citation, 0× novel cross-disciplinary bridge worth
expert review.

Two empirically validated findings:

1. **The cascade correctly detects direct downstream citations.**
   Pair #2 (arXiv:2604.13873 "Evaluating the EML Sheffer Operator for
   Battery Characterization" × Odrzywolek arXiv:2603.21852) was flagged
   `partially_resolves`. Verified: the battery paper's abstract opens
   *"Odrzywolek (2026) recently introduced the Exp-Minus-Log (EML)
   operator…"* — it's a 1-month-later direct application of the
   anchor. Useful for citation-tracking / related-work discovery; not
   what path C was supposed to test.

2. **Path C with EML and PySR anchors did not validate path B.** The
   asymmetric pivot reduced one density problem (uniform sampling),
   the anchor pivot reduced another (random-vs-targeted), but the
   surviving candidates are all in the same research conversation as
   the anchors (Mhaskar-Poggio depth-beats-width line; SR
   library-of-operators folklore since Schmidt-Lipson 2009).
   1.9M-paper production scale (#91) would amplify the folklore
   signal, not surface novel bridges, unless anchors are
   *genuinely* cross-disciplinary.

The corvid version of the answer: between-the-spokes is real, but
EML↔ML-theory and PySR↔SR-theory aren't between spokes — they're at
the center of a well-trodden spoke.

## Pipeline trace

| Stage | Script | Output | Status | Result |
|-------|--------|--------|--------|--------|
| 0 | `te_anchor.py` (new) | `te_candidates.json` | done | 160 pairs (80 per anchor) from anchor-seeded assembly |
| 4 | `te_extract.py` | `te_extractions.json` | done | 152/160 extractions OK (**95%**, vs uniform 82%) |
| 5 | `te_rerank.py` | `te_reranked.json` | done | 145 pairs with both ends extracted (vs uniform 95) |
| 6 | `te_judge.py` | `te_judged.json` | done | 138 unrelated / 7 partially_resolves / 0 resolves |
| 7 | Agent-tool translations | `te_translations.json` | done | 7 Opus subagent translations |
| 8 | Agent-tool indep-Opus second opinion | inline below | done | 2× FOLKLORE, 1× DIRECT-DOWNSTREAM |

## Cost (actuals)

| Stage | Calls | Estimated | Actual |
|-------|-------|-----------|--------|
| S2 lookup + recs + bulk-search | ~15 calls | $0 | $0 (free tier) |
| LLM reformulations (gemini-2.5-flash) | 2 × 1 | $0.001 | $0.001 |
| S2 SPECTER2 fetch | 2 × 1 batch | $0 | $0 |
| arXiv body fetch | 160 unique | $0 | $0 (~3 min @ 1.2s) |
| Slot extract (gemini-2.5-flash) | 160 × 1 | ~$0.05 | ~$0.04 |
| Slot embed (batchEmbedContents) | 2 batches | ~$0.01 | ~$0.01 |
| Cheap-judge (gemini-2.5-flash) | 145 × 1 | ~$0.05 | ~$0.04 |
| Claude translations (Agent, opus) | 7 × 1 | orchestrator | included in session |
| Claude indep-Opus second-opinion | 3 × 1 | orchestrator | included in session |
| **Total external API spend** | | **~$0.10** | **~$0.10** |

Wall clock: ~20 min for stage 0; ~25 min for stages 4-6.

## Anchor configuration

| Anchor | Pool | Rationale |
|--------|------|-----------|
| [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek "All elementary functions from a single binary operator" | theory | Recent theorem; eml(x,y) = exp(x) − ln(y) generates the elementary functions. Searches empirical neighborhood (cs.LG/CV/NE) for ML papers that observe single-operator expressivity without invoking the theorem. |
| [arXiv:2305.01582](https://arxiv.org/abs/2305.01582) — Cranmer "PySR / SymbolicRegression.jl" | empirical | Symbolic regression engine. Searches theory neighborhood (math.FA/CO/IT) for function-class decomposition theorems PySR's authors might benefit from citing. |

LLM-crafted query reformulations per anchor (Gemini 2.5-flash):

**Odrzywolek → empirical:**
- `symbolic regression exact formula recovery`
- `neural network function approximation basis`
- `gradient-based symbolic optimization`
- `universal function approximators minimal basis`
- `computational graph elementary functions`
- `learnable mathematical expression discovery`

**PySR → theory:**
- `operator basis function decomposition`
- `evolutionary algorithm convergence theory`
- `symbolic expression complexity bounds`
- `universal approximation Stone-Weierstrass`
- `sparse regression information theory`
- `empirical process theory statistics`

Reformulations were used as S2 `/paper/search/bulk` queries with
`fieldsOfStudy` filtered to the opposite pool. S2 `/recommendations`
was queried per anchor as a small bonus arm.

Union sizes: Odrzywolek 273 unique arxiv candidates; PySR 360. After
SPECTER2 cosine rank against the anchor, top 80 per anchor were kept
(160 total candidate pairs).

## Anchor mode vs. uniform mode (head-to-head)

| Metric | Uniform run | Anchor run | Delta |
|--------|-------------|------------|-------|
| Total papers ingested | 800 emp + 1377 th = 2177 | ~2 anchors + 633 unique cands = 635 | **−71%** |
| Candidate pairs entering judge | 95 | 145 | **+53%** |
| Slot-extract success rate | 600/731 = 82% | 152/160 = 95% | **+13 pp** |
| Judge `resolves` + `partially_resolves` | 3/95 = 3.2% | 7/145 = 4.8% | **+1.6 pp** |
| Indep-Opus FOLKLORE on top 3 | 3/3 | 2/3 | — |
| Indep-Opus DIRECT-DOWNSTREAM on top 3 | 0/3 | 1/3 | — |
| Indep-Opus WORTH-EXPERT on top 3 | 1/3 (pair 3, Bayesian IMs) | 0/3 | **−1** |
| External API spend | ~$0.20 | ~$0.10 | **−50%** |

**Anchor mode is cheaper and faster but lands in the same place on
quality.** The improvement in hit rate is small (+1.6 pp) and the
improvement in extract success (+13 pp) reflects cleaner upstream
candidate assembly, not better cascade output. Most tellingly:
uniform run produced *one* candidate (DGM uncertainty × Martin-Liu
IMs) rated WORTH-EXPERT by indep-Opus; anchor run produced **zero**.

## Top 7 candidate translations (all judged + translated)

Sorted by slot-cosine descending. Full slots and translations in
`te_translations.json`.

### Pair 1 — slot 0.714, judge: `partially_resolves`
- **E:** [arXiv:2305.01582](https://arxiv.org/abs/2305.01582) — PySR
- **T:** [arXiv:2604.12806](https://arxiv.org/abs/2604.12806) — COSINE: LLM-Guided Symbolic Dynamics Modeling
- **Cascade prediction:** Augmenting PySR with an LLM-guided proposal step that prunes operator/feature subsets per iteration should measurably increase EmpiricalBench recovery rate, with the largest gains on targets with sparse interactions among many candidate features.
- **Indep-Opus:** *(not in top-3 reviewed; pair is empirical×empirical — COSINE is itself an empirical ML system, not a theorem)*

### Pair 2 — slot 0.674, judge: `partially_resolves` ⚠️ DIRECT-DOWNSTREAM
- **E:** [arXiv:2604.13873](https://arxiv.org/abs/2604.13873) — "Evaluating the Exp-Minus-Log Sheffer Operator for Battery Characterization"
- **T:** [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek
- **Cascade prediction:** EML/classical wall-clock ratio should track minimum-node-count tree ratio; battery paper reports ratio but does not measure against theoretical lower bound.
- **Indep-Opus verdict:** **DIRECT-DOWNSTREAM | WORTH-EXPERT: NO.** The battery paper's abstract opens *"Odrzywolek (2026) recently introduced the Exp-Minus-Log (EML) operator…"* — confirmed verbatim via arxiv fetch. 1-month delta, same operator name, explicit citation. Co-discovery, not a bridge.
- *Useful methodological data point:* the cascade correctly surfaced a downstream paper as `partially_resolves`, which validates the assembly-and-judge machinery on a known-positive case.

### Pair 3 — slot 0.646, judge: `partially_resolves`
- **E:** [arXiv:2305.01582](https://arxiv.org/abs/2305.01582) — PySR
- **T:** [arXiv:2602.20550](https://arxiv.org/abs/2602.20550) — "The Finite Primitive Basis Theorem for Computational Imaging: OperatorGraph"
- **Cascade prediction:** Extending PySR's operator library to a superset of the 11 imaging primitives should raise recovery rate; recovered expressions should respect the theorem's depth/complexity bounds D(ε), C(ε).
- **Indep-Opus verdict:** **FOLKLORE | WORTH-EXPERT: NO.** Domain-mismatch dressed in theorem language: the 11-primitive theorem is for *imaging* operators (Lipschitz, bounded, finite-stage), not physics/chemistry/astronomy equations on EmpiricalBench. "Library matters" is bedrock SR folklore since Schmidt-Lipson (2009).

### Pair 4 — slot 0.645, judge: `partially_resolves`
- **E:** [arXiv:2505.10762](https://arxiv.org/abs/2505.10762) — Deep Symbolic Optimization (DSO)
- **T:** [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek
- **Cascade prediction:** DSO retrained with action space reduced to `{eml, const, x_i}` should converge in fewer episodes on Nguyen 1-12 and match/exceed baseline recovery rate; search-space cardinality drops from |V|^(2^d) to 3^(2^d).

### Pair 5 — slot 0.642, judge: `partially_resolves`
- **E:** [arXiv:2308.03230](https://arxiv.org/abs/2308.03230) — "Tractability of approximation by general shallow networks" (Mhaskar lineage)
- **T:** [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek
- **Cascade prediction:** For elementary-class targets, a depth-d network with width-2 channels and a single shared bivariate activation initialized near `eml` should match (up to constants) the sample complexity of a network with d distinct learned activations.
- **Indep-Opus verdict:** **FOLKLORE | WORTH-EXPERT: NO.** Kolmogorov–Arnold (1957) already gives "all continuous multivariate functions decompose into superpositions of univariate + one binary operator"; Mhaskar–Poggio (2016), Poggio et al. (2017), Yarotsky, Telgarsky all use bivariate compositional trees. Cascade rebrands this with `eml` substituted for `+`. The theorem is representability, not sample complexity — the prediction is unsupported by what the theorem actually proves.

### Pair 6 — slot 0.631, judge: `partially_resolves`
- **E:** [arXiv:2304.08915](https://arxiv.org/abs/2304.08915) — Differentiable Genetic Programming
- **T:** [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek
- **Cascade prediction:** DGP with operator library reduced to `{eml, 1}` should match or beat full-library version on elementary targets; isolates joint-tuning advantage from library-size advantage.

### Pair 7 — slot 0.615, judge: `partially_resolves`
- **E:** [arXiv:2602.21307](https://arxiv.org/abs/2602.21307) — SymTorch: Symbolic Distillation
- **T:** [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) — Odrzywolek
- **Cascade prediction:** Restricting SymTorch search to `{eml, 1, x_i}` and full binary trees should recover the same expressions; any target requiring depth beyond Odrzywolek's construction bound is evidence the target is non-elementary, not that distillation failed.

**Pattern across pairs 4-7:** all four are SR-line empirical papers
(DSO, shallow nets, DGP, SymTorch) paired with the EML theorem.
Cascade is correctly surfacing the right neighborhood — symbolic
regression / function approximation papers that could plausibly be
restructured around a single-operator basis. But the indep-Opus
verdict on the representative pair (#5) shows this neighborhood is
folklore-saturated by Kolmogorov-Arnold and Mhaskar-Poggio: the
theorem provides existence not learnability, so no novel testable
prediction survives.

## Why anchor mode hit the same ceiling

The path-C design assumed that anchor-seeded candidate density would
overcome uniform-sampling's structural problem (expected hits ≈ 0.004
in a 2200-paper uniform sample). It does — assembly is cleaner, hit
rate is 50% higher. But the candidates anchor mode surfaces are still
inside the same research conversation as the anchors themselves.

EML is at the math-CS intersection (functional completeness, calculator
operations); its semantic neighbors are SR engines (PySR, DSO, DGP,
SymTorch) and shallow-network approximation theory (Mhaskar). These
are well-trodden — the depth-vs-width / compositional-tractability
program has been active for a decade and the "library matters" framing
for SR is older still. Anchor-seeded SPECTER2 + LLM-crafted queries
land squarely inside that program.

The diagnosis is sharper than the uniform-run RESULTS suggested. It's
not just "sampling density too low at 2200 papers." It's:

> **Cross-disciplinary semantic neighborhoods exist where the
> "neighborhood of an anchor" is a single research conversation,**
> regardless of how the neighborhood is assembled. For these anchors,
> path C ≈ path A in candidate quality.

The bridges Oskar's "between the spokes" hypothesis wants — neural
scaling laws × RMT, lottery ticket × compressed sensing, in-context
learning × Solomonoff — *are* between research conversations. EML × SR
is not. Both anchors live at the heart of one conversation, so their
"opposite-pool" neighbors are descendants/parallels of the same
conversation.

## What would actually test path C

Anchors whose two communities don't share a citation graph. Concretely:

1. **A recent empirical observation from a domain that doesn't read
   math arxiv.** Genomics, climate, materials, condensed-matter,
   social-science computation. Anchor: a 2024-2026 paper that
   observes a scaling or universality result without citing
   probability/statistics theorems.
2. **An older theory result that has not been re-popularized.**
   Anything pre-2010 from math.PR / math.OC / math.FA where the
   modern ML re-derivation either hasn't happened or happened
   independently. Anchor: e.g. a Levy concentration result, an
   information-geometry theorem, a stochastic-approximation lemma.

The eml-sr work is genuinely interesting *as engine development*, and
the indep-Opus verdict on pair #2 (DIRECT-DOWNSTREAM) means the
cascade does correctly identify same-conversation downstream papers —
that's useful for citation-tracking / related-work surfacing. But it's
not the cross-disciplinary discovery use case path C was supposed to
validate.

## Recommendations

1. **Don't open #91 / path B (1.9M-paper production build) on the
   strength of this run.** Anchor mode showed the same folklore
   ceiling as uniform mode. 1.9M scale amplifies signal at every
   tier including folklore — the cost is real ($435), the yield is
   not improved relative to per-paper amortization.
2. **Run path C again with genuinely cross-domain anchors.** Pick
   2-3 from the issue's table (neural scaling × RMT, lottery ticket
   × compressed sensing, in-context learning × Solomonoff). These
   anchors are far apart in citation graph by construction, so their
   opposite-pool neighborhoods will not be the same conversation.
   If even *those* anchors only surface folklore, the cascade
   is structurally unable to find non-folklore bridges and path B
   should stay closed.
3. **Keep the downstream-detection finding as a side product.** The
   cascade's correct flag on the EML-Sheffer paper is a real signal
   for "find downstream applications of this theorem" as a workflow
   — separate from cross-disciplinary discovery, but useful.

## Cross-references

- #97 — uniform-sampling MVP (closed by PR #98; this run was its path C follow-up)
- #91 — paused 1.9M production build (this run is evidence to keep it paused)
- #90 — phase A symmetric bridge MVP (closed by PR #92)

## Data artifacts

- `anchor_run/data/anchors.json` — anchor configuration
- `anchor_run/data/anchor_meta.json` — anchor SPECTER2 + metadata
- `anchor_run/data/anchor_reformulations.json` — Gemini-crafted queries per anchor
- `anchor_run/data/anchor_candidates_raw.json` — union of recs + bulk-search per anchor
- `anchor_run/data/anchor_neighbor_specter.json` — SPECTER2 for all union candidates
- `anchor_run/data/te_candidates.json` — top-K per anchor in downstream schema
- `anchor_run/data/te_translations.json` — 7 Opus subagent translations
- Indep-Opus second-opinion verdicts: inline in this document (verdicts were generated as Agent calls in the orchestrator session and not persisted to disk)
