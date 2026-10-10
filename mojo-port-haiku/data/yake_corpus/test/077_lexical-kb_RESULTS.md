# lexical-kb — embedding-free portable KB skill

**Status:** prototype validated here in Python; **canonical implementation
shipped as the `creating-kb` skill (JavaScript) in claude-skills — PR
oaustegard/claude-skills#712** (renamed from the initial `packing-lexical-kb` /
#711). The toolchain was ported to pure Node so one implementation serves both
the Claude-chat builder and the future in-browser SPA packer; the JS index was
verified bit-identical to the Python here (df / doclen / postings match, scores
to 6 decimals), so the Python in this directory is retained only as the
**parity oracle / reference**, not the shipped artifact. Each bundle ships both
`search.js` and `search.py` (parity-pinned) so it runs in node-only or
python-only consumers.
Quantitative ranking-lift study deferred to Phase 0 on the real muninn corpus
(task #2) — see "Why the tiny corpus can't show lift."

## What this is

A portable knowledgebase that does retrieval with **no embedding model and no
semantic search** — pure BM25 over a precomputed inverted index. The semantic
work (bridging the query↔document vocabulary gap) is moved onto the *consuming
agent*, which expands the query into search terms at query time. The whole thing
ships as a `.skill` bundle (an ordinary zip): `SKILL.md` + `search.py` +
`index.json` + `chunks.jsonl`, pure Python stdlib, no install, no network,
no model download.

## Pieces

| File | Role |
|---|---|
| `skill_template/search.py` | shipped runtime: BM25 + RM3 + metadata filter. Owns the tokenizer (single source of truth). |
| `skill_template/SKILL.md` | the query-expansion protocol the consuming agent follows. |
| `build_lexkb.py` | builder: structural chunker → inverted index → bundle dir → `.skill` zip. Imports the tokenizer from `search.py` so builder/searcher can't drift. |
| `test_lexkb.py` | three-way comparison (raw BM25 / RM3 / agent-expansion) + score-margin metric. |

## Build + use

```bash
# build a .skill from a directory of .txt/.md/.html files
python3 build_lexkb.py CORPUS_DIR --target-chars 1200 --zip --name mykb

# query from inside the bundle (zero deps)
python3 out/kb/search.py --query "the question" \
  --core term1 --core term2 --expand synonym1 --expand variant2 --k 5
```

## Test results (tiny corpus: 4 docs — Federalist 10×2, Federalist 51, Gettysburg)

Rank of the gold document under each mode (lower = better), plus a normalized
score margin (gold − best distractor, over gold) which has resolution even when
rank saturates:

```
=== target_chars=1200 (8 chunks) ===
case                    rank raw/rm3/exp    margin raw→exp
plural/synonym                     1/1/1     +1.00 → +0.27
separation of powers               1/1/1     +0.25 → +0.25
property/inequality                1/1/1     +0.70 → +0.92
paraphrase (hard)                  1/1/1     +0.59 → +0.84
```

(Full sweep over target_chars=500/1200/whole-doc in `test_lexkb.py` output — the
pattern is stable across chunk sizes.)

### What the test established

1. **Correctness.** Build, BM25 scoring, RM3 pseudo-relevance feedback, and
   metadata filtering (`section=`, `source_path~`, `date>=`) all work end to
   end, including the full `.skill` zip → unzip → query round trip.

2. **Expansion widens the margin on vocabulary-gap cases** — `property/inequality`
   (+0.70→+0.92) and the hard paraphrase `honoring soldiers who died in battle`
   → Gettysburg (+0.59→+0.84). This is the intended effect: agent-supplied
   synonyms/variants add discriminative signal the literal query lacked.

3. **Over-generic expansion can *narrow* the margin** — `plural/synonym`
   (+1.00→+0.27). Adding broad terms ("majority", "rights") leaks score into
   unrelated docs. Gold still leads (rank 1), but the lesson is real and is now
   in the SKILL.md: keep expansion targeted.

### The bug testing caught (and the fix)

First run, the `separation of powers` case under agent-expansion returned the
gold doc at **score 0 / not retrieved** while raw BM25 had it at rank 1. Cause:
the hand-authored expansion *substituted* curated synonyms (checks, balances,
departments) for the user's words — and that short Federalist-51 excerpt
contains none of them; it keys on "government" and "power" (singular). The raw
query matched on "government"; the substitutive expansion threw that signal away.

**Fix:** expansion is now strictly **additive**. `search.py` carries the user's
original `--query` terms at a low floor weight (0.25) beneath `--core` (1.0) and
`--expand` (0.4), so a curated synonym can only *lift* a result, never drop a doc
the literal question would have matched. The SKILL.md protocol mandates passing
`--query` alongside the term groups.

### Why the tiny corpus can't show ranking lift

The four documents are topically disjoint, so raw BM25 already wins at rank 1 on
almost any shared term — there's no ranking competition for expansion to improve.
That's why every mode shows rank 1 and we fall back to the margin metric. A
real corpus with many topically-overlapping chunks (the muninn blog) is where
rank-lift becomes measurable — that's **Phase 0 (task #2)**: sweep chunk size
500/1500/4000/whole-section, re-tune BM25 `b` per size, run the issue-#76
acceptance queries, and diff hits against the embedding `muninn.kb` to quantify
what the embeddings were actually buying.

## Design decisions

- **Index format:** stdlib-JSON inverted index, not bm25s-CSC. Rationale: the
  `.skill` is unpacked on the consumer and queried by the bundled `search.py`,
  so zero-dep portability beats reader-compat. The bm25s-CSC format (for the
  hosted-`.kbi` / JS-reader path) is a separate convergence question — task #3.
- **Metadata stays structured** (not folded into indexed text). No embedding
  means no need to inject title/description for "semantic grounding"; keeping
  meta as fields enables `--filter`. Title-term *boosting* (BM25F) is deferred —
  for now the agent can add title words to `--expand` if it wants the lift.
- **Structural chunking** (`--target-chars`, whole paragraphs, never split
  mid-paragraph; `--target-chars 0` = whole document). Lexical BM25 tolerates
  bigger chunks than embeddings (no centroid to dilute), so chunk size is a free
  knob here — quantified in Phase 0.
