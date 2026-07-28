# Theory → Empirical Bridge MVP: Path C Cross-Domain Run

*Direct follow-up to `anchor_run/RESULTS.md` and `anchor_run_filtered/RESULTS.md`
(PR #99 + #100), implementing the explicit recommendation from both:*
*"Re-test path C with genuinely cross-domain anchors — neural scaling × RMT,
lottery ticket × compressed sensing, in-context learning × Solomonoff."*
*Run date: 2026-05-24*

## What this run decides

Per BTS post 2 ("Between the Spokes: What the Embeddings Can't See") and the
between-the-spokes health assessment (memory `0c0c8afc`), Path C is the
~$1 experiment that gates the $435 Path B (1.9M production build).

Two diagnostic failure modes:

1. **TWIN-COUSIN majority**: SPECTER2 surfaces twins or close cousins of
   known cross-domain bridges. Bridge signal is present in the geometry
   at small scale → Path B becomes defensible at scale.
2. **TWIN-MISS majority**: SPECTER2 only returns same-conversation
   folklore even when given an anchor with a known theoretical twin in
   a *disjoint* community. The Singular Learning Theory objection
   (memory `a8b97f70`) is doing real work — novel bridges manifest as
   new directions in representation space, not as points between
   existing clusters. Path B does not improve the situation, and the
   cascade mechanism itself needs to change before the corpus does.

## Anchor configuration

Nine anchors drawn from issue #97's exemplar table plus grokking. Each
declares `expected_twin_arxiv` — known arXiv IDs in the theoretical (or
empirical) twin conversation — used by the twin diagnostic to ask:
"does SPECTER2 put twin cousins in the top-K, or only same-conversation
papers?"

| # | Anchor | Pool | Bridge | Expected twin conversation |
|---|--------|------|--------|---------------------------|
| 1 | Kaplan 2001.08361 | empirical | Scaling laws × RMT | Pennington-Worah, Couillet-Liao |
| 2 | Madry 1706.06083 | empirical | Adversarial × isoperimetric | Bubeck-Sellke, Gilmer-spheres, Mahloujifar |
| 3 | Soudry 1710.10345 | empirical | Implicit bias × max-margin | Telgarsky, Bartlett |
| 4 | Belkin 1812.11118 | empirical | Double descent × stat mech | Hastie-Montanari, Mei-Montanari |
| 5 | Power 2201.02177 | empirical | Grokking × SLT/phase transitions | Watanabe SLT, Murfet-Wei devinterp |
| 6 | Frankle 1803.03635 | empirical | Lottery ticket × compressed sensing | Candes-Tao, Candes-Romberg-Tao |
| 7 | math/0502327 | theory | (Reverse of #6) | Frankle, Liu pruning |
| 8 | Garg 2208.01066 | empirical | ICL × Solomonoff | Hutter, Xie implicit Bayes |
| 9 | cs/0701125 | theory | (Reverse of #8) | Garg, Brown GPT-3, Xie |

Five empirical-only (#1–5) plus two both-direction pairs (#6+7, #8+9).

## Pipeline trace

| Stage | Script | Output | Result |
|-------|--------|--------|--------|
| 0 | `te_anchor.py --skip-citation-filter` | `te_candidates.json` | 720 pairs (9 anchors × top-80; 5645 candidates pre-filter, 5608 post, 5244 unique paperIds) |
| diag-pre | `twin_diagnostic.py` | `01_twin_diagnostic.log` | **9/9 TWIN-MISS (strict); 1/9 loose-TWIN-COUSIN (false positive on inspection)** |
| 4 | `te_extract.py` | `te_extractions.json` | 659/707 papers extracted (93% success; 16 hit CF gateway 429 post-retry, 32 misc errors) |
| 5 | `te_rerank.py` | `te_reranked.json` | top-200 by slot-cosine across the 720 candidate pool |
| 6 | `te_judge.py` | `te_judged.json` | **43/200 positive (1 resolves + 42 partially_resolves), 141 unrelated, 16 errors. 21.5% positive rate.** |
| diag-post | `twin_diagnostic.py` | `07_twin_diagnostic_post.log` | unchanged: 9/9 TWIN-MISS strict; the judge surfacing positives doesn't add twin papers because no twin was in the union |
| 7 | Agent-tool indep-Opus | inline below | **0/3 WORTH-EXPERT** (2× DIRECT-DOWNSTREAM, 1× FOLKLORE) |

`--skip-citation-filter` was used because the free-tier S2 API rate-limits
individual `/citations` and `/references` GETs heavily (~50s per fail
with 8-retry exponential backoff), and for cross-domain anchors the
citation-graph filter is mostly irrelevant — bridge candidates are by
definition outside the anchor's citation neighborhood. The abstract-mention
filter (which uses the batch endpoint and is fast) still runs.

## Twin diagnostic results

**Strict reading: 9/9 TWIN-MISS. Loose reading: 8/9 TWIN-MISS, 1/9 TWIN-COUSIN (false positive on inspection).**

For each anchor, the diagnostic asks: are the `expected_twin_arxiv` papers
(known canonical members of the theoretical-or-empirical twin conversation)
present in the union of recommendations + LLM-reformulated bulk-search
hits — not in top-K, but in the union *at all*?

| Anchor | Bridge | Twin papers in union? | Top-40 keyword density |
|--------|--------|----------------------:|-----------------------:|
| Kaplan 2001.08361      | Scaling × RMT                | 0/3 found | 30% — *all* "scaling law" empirical followups, **zero RMT theory** |
| Madry 1706.06083       | Adversarial × isoperimetric  | 0/3 found | 15% — Lipschitz-flavored, no pure isoperimetric |
| Soudry 1710.10345      | Implicit bias × max-margin   | 0/2 found | 10% — one near-bridge, no canon |
| Belkin 1812.11118      | Double descent × stat mech   | 0/3 found | 8% — one bridge-paper hit |
| Power 2201.02177       | Grokking × SLT/phase trans   | 0/3 found | 10% — all hits are grokking-followups, **zero Watanabe/Murfet** |
| Frankle 1803.03635     | Lottery × compressed sensing | 0/3 found | 2% — one bridge-paper hit |
| math/0502327 (CRT)     | (reverse)                    | 0/3 found | 0% |
| Garg 2208.01066        | ICL × Solomonoff             | 0/3 found | 0% |
| cs/0701125 (Hutter)    | (reverse)                    | 0/3 found | 0% |

**Twin-papers-in-union: 0 / 26.** Not "low rank". *Not in the candidate
union at all*, after a candidate assembly that already includes S2
`/recommendations` for the anchor's paperId + S2 `/paper/search/bulk` for
six Gemini-reformulated opposite-pool queries per anchor — together
yielding 244–918 unique candidates per anchor (5645 total post-filter).

**The Kaplan "TWIN-COUSIN" verdict is a false positive on inspection.**
All 12 keyword matches are *other empirical* scaling-law papers — "Practical
Scaling Laws", "Tokens-per-Parameter Coverage", "InfoLaw", "Scaling Laws for
Weather Emulation", "Parcae". Same conversation as the anchor, not the
cross-disciplinary RMT twin. Tightening the keyword bag would drop this
to TWIN-MISS as well. Strict reading: **all 9 anchors miss.**

### Why the Gemini reformulations missed too

The LLM-crafted queries for each anchor mostly stayed inside the anchor's
own research conversation, with a few partial reaches toward the twin:

- Madry → "Lipschitz continuity function approximation", "contraction
  mapping fixed point theorems" — reaches toward concentration/isoperimetric
  language, but not via the canonical terms
- Belkin → "random matrix theory spectral properties" — *actual* RMT term,
  yet still failed to surface Hastie-Montanari et al.
- Power → no "singular learning theory", "Watanabe", "phase transition"
  in any of the 6 reformulations
- Garg → no "Solomonoff", "algorithmic probability", "MDL" in any
  reformulation
- Hutter → did include "PAC-learning sample complexity", "algorithmically
  random sequences", but Garg / Brown / Xie still missed

Two layers of selection both failed to cross the bridge: (1) the LLM's
ability to *name* the opposite community's terminology, (2) the SPECTER2
geometry's willingness to put the twin within the top several hundred
neighbors even when queried with the right terms.

## Cascade results (head-to-head)

| Run | Anchors | Total judged | Judge positive rate | Indep-Opus WORTH-EXPERT | Twin-papers in union |
|-----|---------|-------------:|--------------------:|------------------------:|--------------------:|
| Uniform (#98)                       | (none)  | 95  | 3.2% | 1/3 | n/a (no anchors) |
| Anchor unfiltered (#99 anchor_run)  | 2 EML/PySR | 145 | 4.8% | 0/3 | n/a (single-conversation anchors) |
| Anchor + citation filter (#100)     | 2 EML/PySR | 145 | 2.8% | 0/3 | n/a (single-conversation anchors) |
| **Path C cross-domain (this)**      | **9 cross-domain** | **200** | **21.5%** | **0/3** | **0 / 26** |

### Why the positive rate jumped to 21.5%

This is *not* an improvement signal — it's a calibration finding. With
cross-domain anchors, the cheap-judge sees pairs that share the anchor's
vocabulary (which is now richer because the anchor itself is well-known
and well-cited) but doesn't share mechanistic correspondence. The judge
labels them `partially_resolves` on slot-vocabulary overlap rather than
on bridge structure. Examples from the top-10 by slot-cosine, all judged
`partially_resolves` (the lone `resolves` is also same-conversation):

- Garg (ICL) × `2310.10616` "How Do Transformers Learn In-Context **Beyond** Simple Functions? A Case Study..." — literal Garg sequel paper, indep-Opus: **DIRECT-DOWNSTREAM**
- Garg (ICL) × `2604.25858` "Investigation into In-Context Learning Capabilities of Transformers" — same conversation
- Soudry (impl-bias) × `1810.08727` "Condition Number Analysis of Logistic Regression…" — Freund-lineage refinement of Soudry's own separable-loss setup, indep-Opus: **DIRECT-DOWNSTREAM**
- Kaplan (scaling) × `2605.02364` "InfoLaw: Information Scaling Laws for LLMs…" — same conversation
- Madry (advers) × `2009.06202` "Risk Bounds for Robust Deep Learning" — PAC-Bayes-for-adv-loss canon, indep-Opus: **FOLKLORE**
- Belkin (DD) × `2106.04003` "Double Descent and Other Interpolation Phenomena in GANs" — same conversation
- Madry (advers) × `2202.13216` "Adversarial robustness of sparse local Lipschitz predictors" — same conversation
- (… 36 more in `te_judged.json`, all same conversation by inspection)

**Positives per anchor:** Madry 21, Belkin 9, Kaplan 6, Garg 4, Soudry 3, Power/Frankle/CRT/Hutter 0. The four anchors that produce zero positives are precisely the ones where the candidate union's vocabulary is most distinct from the anchor (math-prefix arXiv ID; pre-arXiv Solomonoff terminology that didn't ingest well). The judge can't even produce false-positive resemblance there.

### Indep-Opus verdicts on top survivors

| Rank | Anchor | Pair | Judge | Indep-Opus |
|---:|--------|------|-------|------------|
| #1 (resolves) | Garg ICL | 2208.01066 × 2310.10616 | resolves, cos=0.826 | **DIRECT-DOWNSTREAM** — sequel paper, title is literal callback ("Beyond Simple Functions" answers Garg's "Simple Function Classes") |
| Soudry top | Soudry impl-bias | 1710.10345 × 1810.08727 | partially_resolves, cos=0.826 | **DIRECT-DOWNSTREAM** — Freund-lineage condition-number refinement, same separable-loss conversation |
| Madry top | Madry advers | 1706.06083 × 2009.06202 | partially_resolves, cos=0.798 | **FOLKLORE** — PAC-Bayes-for-adv-loss is the adversarial-ML/learning-theory canon, not a cross-disciplinary bridge |

**0 / 3 WORTH-EXPERT.** Two of three are direct citation-graph downstream; one is in-canon folklore. Same pattern as #99 / #100 — the cascade catches downstream papers reliably (the EML-Sheffer battery paper finding survives as a real product), but does not surface cross-disciplinary bridges, *even when seeded with anchors from known cross-disciplinary bridges.*

## Verdict

**The SLT objection wins. Path B at \$435 does not become defensible
on the strength of this run.**

The diagnostic question Oskar posed in BTS post 2:

> If non-anchor cousins surface at non-trivial rate, the SPECTER2
> geometry holds enough signal at small scale that Path B becomes
> defensible. If the cascade still returns only folklore neighbors
> of the anchors, the Singular Learning Theory objection — that
> novel bridges manifest as new directions in representation space,
> not as points between existing clusters — is doing real work,
> and the mechanism itself needs to change before the corpus does.

Has a clean answer: **0/26 expected twins** appear in any anchor's
candidate union, *not just outside top-K but absent from the
~700-paper-per-anchor union assembled via S2 recommendations + six
Gemini-reformulated bulk-search queries*. The cascade's positives
are all same-conversation refinements, downstream papers, or
folklore. Scaling the corpus from 5645 to 1.9M won't change this —
the bridge points aren't more sparsely sampled, they're *outside
the geometry's reach*.

**What would actually change the answer:**

1. **A different representation space.** Not SPECTER2 (or any
   scientific-document embedder fine-tuned on citation prediction),
   but a space that explicitly models mechanism / formal structure
   correspondence. Promising directions: prompt-based embeddings
   that ask "what mechanism does this paper provide?" rather than
   "what topic is this paper about"; structured proof-graph
   embeddings; or — per the SLT framing — embedding into a space
   that *allows new dimensions* (singularity-resolved
   representations à la blowing-up).
2. **Citation-graph-blind LLM bridge proposal.** Skip the
   neighborhood-search entirely; let a strong LLM propose
   mechanism-correspondence hypotheses from the anchor's slots,
   then verify against the corpus. Inverts the search: from
   "find similar papers" to "verify hypothesized bridges".
3. **Author-graph diversification.** If anchor's union is
   dominated by the anchor's own citation neighborhood, force
   the union to span >K disjoint author components before
   ranking. Cheaper than #1 but doesn't address the core SLT
   issue — just makes the failure mode visible.

**What remains useful from the cascade:**

- **Downstream-citation detection.** The cascade reliably labels
  direct-downstream papers `partially_resolves` (cf. EML-Sheffer
  battery paper in #99, two of three indep-Opus verdicts here).
  This is a usable product for "find recent papers that build on
  this theorem" workflows.
- **Same-conversation density mapping.** Cheap way to characterize
  how tightly a paper's community is clustered in SPECTER2 — useful
  signal for the diagnostic question itself, separate from
  bridge-discovery.

## Cost (actuals)

| Stage | Calls | External spend |
|-------|------:|---------------:|
| S2 anchor batch fetch                       | 1     | $0 (free tier) |
| S2 recommendations + bulk-search            | ~63   | $0 (free tier, ~10 min walked through 429s) |
| Gemini reformulations (gemini-2.5-flash)    | 9     | ~$0.001 |
| S2 abstract batch (5244 paperIds)           | ~12   | $0 (free tier) |
| S2 SPECTER2 batch (5244 paperIds)           | ~12   | $0 (free tier) |
| arXiv body fetch                            | 707   | $0 (~10 min @ 1.2s politeness) |
| Slot extract (gemini-2.5-flash, parallel=2) | 707   | ~$0.18 |
| Slot embed (batchEmbedContents)             | ~14   | ~$0.01 |
| Cheap-judge (gemini-2.5-flash, parallel=2)  | 200   | ~$0.06 |
| Indep-Opus second-opinion (Agent tool)      | 3     | orchestrator (free for accounting) |
| **Total external API spend**                |       | **~$0.25** |

Well under the $1 budget — the SLT-objection-wins finding has been
purchased for the cost of a coffee, and removes the burden of paying
\$435 for the same finding at 350x larger scale.

Wall clock: ~80 min end-to-end (~10 min candidate assembly, ~12 min
body fetch, ~50 min Gemini extract under rate-limiting, ~5 min rerank
+ judge, ~30 sec Opus verdicts in parallel).

## Cross-references

- `anchor_run/RESULTS.md` — PR #99 (unfiltered EML/PySR baseline)
- `anchor_run_filtered/RESULTS.md` — PR #100 (citation-filtered EML/PySR baseline)
- `mvp/theory-empirical-bridges/RESULTS.md` — PR #98 (uniform-mode baseline)
- BTS post 2: https://muninn.austegard.com/blog/between-the-spokes-what-the-embeddings-cant-see.html
- Methods page: https://muninn.austegard.com/scratch/between-the-spokes-data.html
- Issue #97 exemplar table — source of 5 of the 9 anchors

## Data artifacts

- `path_c_cross_domain/data/anchors.json` — 9 anchors + `expected_twin_arxiv` per anchor
- `path_c_cross_domain/data/anchor_meta.json` — anchor SPECTER2 + metadata (batch fetch)
- `path_c_cross_domain/data/anchor_reformulations.json` — 6 Gemini queries per anchor
- `path_c_cross_domain/data/anchor_candidates_raw.json` — union of recs + bulk-search per anchor
- `path_c_cross_domain/data/anchor_candidates_filtered.json` — post abstract-mention filter
- `path_c_cross_domain/data/anchor_neighbor_specter.json` — SPECTER2 for union
- `path_c_cross_domain/data/te_candidates.json` — top-K per anchor in downstream schema
- `path_c_cross_domain/data/te_judged.json` — judge verdicts (43/200 positive)
- `path_c_cross_domain/logs/*.log` — full per-stage logs (gitignored)
- `path_c_cross_domain/twin_diagnostic.py` — twin-cousin / twin-miss verdict generator
- `path_c_cross_domain/data/anchor_neighbor_specter.json` — 100MB SPECTER2 cache (gitignored; regenerable from S2 batch)
- `path_c_cross_domain/data/candidate_abstracts.json` — 7MB abstract cache for the mention filter (gitignored)
- `path_c_cross_domain/data/te_bodies.json` — arXiv body cache for extracted papers (gitignored)
- `path_c_cross_domain/data/te_extractions.json` — Gemini slot extractions (gitignored)
- `path_c_cross_domain/data/te_reranked.json` — top-200 by slot-cosine (gitignored)
