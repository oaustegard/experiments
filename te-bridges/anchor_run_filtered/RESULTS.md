# Theory → Empirical Bridge MVP: Anchor + Citation-Filter Run

*Follow-up to `anchor_run/RESULTS.md`. Adds explicit citation-graph and
abstract-mention filters to address the direct-downstream defect found
in the first anchor run (EML-Sheffer battery paper at rank #2, verified
direct citer of Odrzywolek).*
*Run date: 2026-05-24*

## TL;DR

Citation filter does what it's supposed to do. EML-Sheffer correctly
excluded (`drop 2604.13873 (author:'odrzywolek')`), confirmed via fall
of the cascade's top-2 hit. **But filtering doesn't surface hidden
non-folklore bridges — the freed-up top-K slots were filled by
candidates that the judge labeled `unrelated`.**

Compared to the prior runs:

| Run | Positive rate | True positive rate¹ | Indep-Opus WORTH-EXPERT |
|-----|--------------:|--------------------:|------------------------:|
| Uniform (#98)                       | 3/95 = 3.2% | 3/95 = 3.2% | 1/3 |
| Anchor unfiltered (anchor_run, #99) | 7/145 = 4.8% | 6/145 = 4.1% | 0/3 |
| Anchor + citation filter (this)     | 4/145 = 2.8% | 4/145 = 2.8% | 0/3 (carried) |

¹ "True positive rate" = positives that are NOT direct citation-graph
neighbors of the anchor. For the unfiltered anchor run, subtracting the
EML-Sheffer direct-citer leaves 6 — still folklore-saturated per indep-Opus
on the Tractability representative.

The sharper diagnosis: **once you remove citation-graph noise, anchor
mode and uniform mode have indistinguishable positive rates** (2.8% vs.
3.2%) at this scale. Anchor mode's apparent advantage in the prior run
was largely the direct-downstream paper inflating the count.

## What the filter caught

The filter has three tiers, applied to the candidate union BEFORE
SPECTER2 cosine ranking:

1. **Citation-graph** (`S2 /paper/{id}/citations` + `/references`):
   exclude candidates whose paperId or arXiv id appears in the anchor's
   citation graph.
2. **Author overlap**: exclude candidates whose author set intersects
   the anchor's own authors plus all authors in the citation-graph
   neighborhood.
3. **Abstract mention** (`S2 /paper/batch` for abstracts): exclude
   candidates whose abstract literally mentions the anchor's title
   fragments or any anchor author surname. This catches S2 citation-index
   lag — for newly-uploaded arXiv papers, S2 takes weeks to months to
   ingest citations, so direct citers won't appear in `/citations`
   until then.

Per-anchor exclusion stats (this run):

| Anchor | citation_graph dropped | author dropped | abstract-mention dropped |
|--------|-----------------------:|---------------:|-------------------------:|
| Odrzywolek (2603.21852) | 0 | 1 | **1** (EML-Sheffer 2604.13873) |
| Cranmer / PySR (2305.01582) | 0 | 7 | 0 |

Citation graph alone caught **zero** for the recent anchor (Odrzywolek
is 2 months old; only 3 citers indexed in S2 so far, none of them was
the battery paper). The abstract-mention filter is the one that
actually catches direct downstream citations for recent anchors.

## The implementation detail that almost lost the catch

The abstract-mention filter initially dropped zero candidates. Debug
trace revealed a Unicode normalization bug:

- S2 returns Odrzywolek's name as `"Andrzej Odrzywołek"` in `/paper/{id}`
  (with Polish ł, U+0142) but the candidate's abstract text reads
  `"Odrzywolek (2026) recently introduced..."` (without ł).
- `unicodedata.normalize("NFKD", "ł")` does NOT decompose the stroked
  l — it has no canonical decomposition — so `.encode("ascii", "ignore")`
  silently drops it: `"Odrzywołek"` → `"Odrzywoek"` (missing the l
  entirely).
- Substring match: `"odrzywoek" in "...odrzywolek (2026)..."` → False.

Fixed by adding a `_STROKED_LETTER_FOLD` translation table covering
the precomposed stroke letters (Polish ł, Nordic ø, Croatian đ,
Icelandic þ/ð, Latin æ/œ, German ß) before NFKD decomposition.

After the fix: `drop 2604.13873 (author:'odrzywolek')` confirms catch.

This is a quiet failure mode worth flagging: ASCII-folding via NFKD
alone is **not** sufficient for European-language author names. Worth
adding to `te_common.py` if other parts of the pipeline ever do name
matching.

## Surviving 4 positives (all SR-line × Odrzywolek family)

Sorted by slot cosine, all `partially_resolves`:

1. **slot 0.714** — PySR × COSINE (`2305.01582` × `2604.12806`)
   - Empirical×empirical pairing. COSINE is itself a symbolic-dynamics ML system, not a theorem.
2. **slot 0.642** — Tractability of shallow nets × Odrzywolek
   - Indep-Opus (previous run): **FOLKLORE**. Mhaskar-Poggio + Kolmogorov-Arnold already cover compositional bivariate nets.
3. **slot 0.631** — Differentiable Genetic Programming × Odrzywolek
   - Same family as #2. Same family of folklore.
4. **slot 0.615** — SymTorch × Odrzywolek
   - Same family. Same verdict.

All 4 are SR-line empirical papers paired with the EML theorem. The
indep-Opus verdict on the representative pair (Tractability of shallow
nets × Odrzywolek) was FOLKLORE: the theorem is about representability,
not learnability or sample complexity; the cascade rebrands
Kolmogorov-Arnold + Mhaskar-Poggio with `eml` substituted for `+`. The
other three in the family inherit the same verdict by family
membership — no need to spend more Opus calls on near-duplicates.

(Translations for all 4 are in `te_translations.json`, carried forward
from the unfiltered run.)

## What this confirms about path C

The diagnosis from `anchor_run/RESULTS.md` stands and sharpens:

> **Cross-disciplinary semantic neighborhoods exist where the
> "neighborhood of an anchor" is a single research conversation,**
> regardless of how the neighborhood is assembled.

Adding citation filters cleans the noise but does not surface
hidden bridges. There are no hidden bridges *because the anchor
itself sits inside a well-trodden conversation*. Cleaning the noise
just reveals that the rest of the neighborhood is also part of the
same conversation, judged `unrelated` because there's no
empirical-resolves-theorem structure.

The right next experiment is unchanged: test path C with anchors
whose two communities don't share a citation graph (neural scaling
× RMT, lottery ticket × compressed sensing, in-context learning ×
Solomonoff). Until that's tested, the cascade has not been shown
to find non-folklore bridges at any scale.

## Cost (actuals)

Citation graph fetches: ~150 S2 calls (free tier, throttled), ~3 min.
Abstract batch fetch: 2 batch calls covering 625 paperIds, ~1 min after backoff.
Re-run of extract → rerank → judge: ~5 min (most extractions reused from prior run cache).

Total incremental spend over the unfiltered run: **~$0.02** (Gemini judge re-runs only).

## Cross-references

- `anchor_run/RESULTS.md` — unfiltered baseline (PR #99)
- `mvp/theory-empirical-bridges/RESULTS.md` — uniform-mode baseline (PR #98, merged)
- te_anchor.py — citation + author + abstract-mention filters now baked in as default
- te_common.py — could promote `_ascii_fold` here if other modules ever need name matching

## Data artifacts (this run)

- `anchor_run_filtered/data/anchor_exclusions.json` — per-anchor citation + author exclude sets
- `anchor_run_filtered/data/candidate_abstracts.json` — abstract cache for the 625-candidate union
- `anchor_run_filtered/data/anchor_candidates_filtered.json` — candidates surviving all three filter tiers
- `anchor_run_filtered/data/te_candidates.json` — top-K pairs, downstream-schema-compatible
- `anchor_run_filtered/data/te_judged.json` — 145 judged, 4 partially_resolves, 0 resolves
- `anchor_run_filtered/data/te_translations.json` — 4 translations (carried from anchor_run)
