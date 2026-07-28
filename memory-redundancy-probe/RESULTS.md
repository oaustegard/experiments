# Memory redundancy probe + synthesis-process review

**Date:** 2026-06-04
**Trigger:** While reviewing arxiv 2606.03787 (surprise-gated robot episodic
memory), the question arose: should Muninn move episodic-event storage off
hand-judged "this seems worth keeping" onto a measured signal (surprise /
prediction-error)? That presumes a storage-quality problem. This probe checks
whether one exists, and reviews the existing curation/synthesis pipeline.

## Method

- Pulled all 1,747 active memories (`deleted_at IS NULL AND is_superseded=0`)
  from Turso via `scripts.memory._exec`.
- Embeddings were removed from the memory store (v0.13.0 dropped OpenAI embed
  generation; v2.0.0 dropped the column), so redundancy was measured
  **lexically** via `muninn_utils.memory_tfidf.MemoryIndex` (TF-IDF, 1–2 grams,
  cosine). This is a **floor** on redundancy — it misses semantic paraphrase.
- Read the `consolidate()` and `curate()` source to review the synthesis path.

## Findings

### 1. Lexical redundancy is a small tail (but see the blind spot in §5)

Near-duplicate tail (distinct memories involved in ≥1 pair at each threshold):

| TF-IDF cosine | pairs | distinct mems | % of store |
|---|---|---|---|
| ≥0.95 | 23 | 19 | 1.0% |
| ≥0.90 | 24 | 21 | 1.2% |
| ≥0.80 | 64 | 40 | 2.2% |
| ≥0.70 | 75 | 56 | 3.1% |
| ≥0.60 | 166 | 140 | 7.7% |

By **lexical** measure the store is not a landfill of redundant episodes. A
fancier storage *gate* (surprise, prediction-error) would address a problem that
isn't present at this measure. The marginal value of a measured episodic gate is
low. **Caveat: this is a floor — TF-IDF is known-inadequate for the running-topic
families in this store; see §5.**

### 1b. The real growth driver is session-log scaffolding, not redundancy

Active session-log families: **92 `SLEEP SESSION` + 93 `FLY SESSION` = 185
memories (10.6% of the store)**, plus 44 zeitgeist digests. The store grows by
session/operational logs far faster than by redundant episodes. The sleep-session
maintenance itself flags `experience` (501) as dominant "session-log
accumulation" and defers to `prune_by_age`. This — not redundancy — is where
volume comes from.

### 2. Three write-side leaks (the actual problems)

- **`"Valid"` stub memories (6):** `type=world`, empty tags, summary literally
  `"Valid"`, dated 2026-04-11 → 2026-05-10. A write path is persisting a
  validation token as a memory.
- **Exact write-twice duplicates:** identical summaries stored twice
  (install-manifest v0.3 for muninn-bsky-card; the PySR/Julia Containerfile
  note; the RAG-Match-paper thought). Double-call / retry-without-idempotency.
- **`"Skipped zeitgeist …"` no-op logs (14):** a background process writing its
  own skips ("last was N.Nd ago, floor is 5d") into permanent memory. Some at
  **priority 1**, typed `decision`/`experience`/`procedure`. Operational
  telemetry masquerading as durable memory.

### 3. Curation tooling is mismatched to the problems

- `consolidate()` groups by **shared tag** and "synthesizes" by **concatenation**
  (bullet list + header). No semantics, no compression. Catches none of the
  three leaks: `"Valid"` stubs have empty tags (invisible to tag-clustering);
  exact dups needn't share tags; concatenating the zeitgeist logs preserves the
  noise.
- `curate()` docstring advertises strategy 3, *"Duplicate detection: memories
  with high textual overlap"* — **not implemented in the code**. Only
  tag-consolidation and stale-detection run. The tool that would implement it
  (`memory_tfidf.MemoryIndex.duplicates`) already exists in `muninn_utils` but
  is not wired into `curate()`.

### 4. `access_count` is not a usable utility signal

Intended to measure dead weight via never-recalled memories. Unusable:
median `access_count`=10, only 7/1747 (0.4%) never recalled, max=1327. Boot-time
bulk recalls inflate it. Cannot distinguish a useful memory from a dead one.

### 5. TF-IDF blind spot — documented in the store's own memory

Memory `517a2f07` (2026-05-04) records that when the zeitgeist pipeline needed
dedup, **TF-IDF failed**: it gave 0.05–0.13 cosine for *obviously* duplicative
running-topic content (Iran/Hormuz sections) because IDF crushes the shared
vocabulary that makes them duplicates. The fix was Gemini embeddings via the CF
gateway (dim=256, dup≥0.93). Implication for this probe: the §1 numbers
**undercount** semantic redundancy in the 44 zeitgeist-digest + 3
weekly-snapshot family — exactly the content most likely to be near-identical
across entries. Re-measuring properly needs embeddings, and the Gemini-via-CF
path was logged broken on 2026-06-01 (`68159beb`). So the semantic redundancy of
the digest family is currently **unmeasured**, not low.

### 6. No embeddings in the memory store (and a confabulation caught)

The Turso memory store has **no embeddings** (removed v0.13.0 / v2.0.0).
`recall` is FTS5+LIKE (lexical); `consolidate` is tag-bucket concatenation;
`curate`'s advertised textual-overlap dedup is unimplemented. Embeddings exist
only *outside* the store (zeitgeist_delta, remax_kb via CF gateway). During this
session Muninn **hallucinated** that `consolidate` does "embedding-cluster dedup"
and began acting on it before reading the code; ground-truth reading caught it.
Correction stored as a priority-2 scar-tissue memory (`4bfb05fa`) so future
sessions don't re-derive the false capability.

### Note on index scope

`MemoryIndex.build()` filters only `deleted_at IS NULL` — it includes superseded
rows (indexed 1810 vs 1747 active), slightly inflating apparent dups. Minor.

## Recommendations

### Curate tooling (the actual fix)

1. **Implement the dedup `curate()` already advertises.** Wire
   `MemoryIndex.duplicates()` into `curate()` for exact/near-exact lexical dups
   (catches the `"Valid"` stubs and write-twice dups). Lexical is the right tool
   for *exact* duplicates. **Do not** rely on it for running-topic semantic dups
   — §5 shows it fails there. For that family, gate on embeddings when the CF
   path is restored, or skip semantic dedup entirely rather than pretend.
2. **Remove the false docstring claim** in the interim so a future session
   doesn't trust a no-op. (The scar-tissue memory `4bfb05fa` also covers this.)
3. **Stop the maintenance routine reporting false-clean.** Sleep sessions report
   "No duplicates identified" because they trust the phantom check — while 6
   `"Valid"` stubs + exact dups + 14 zeitgeist-skip logs sit active.

### Write-side leaks

4. **Fix the `"Valid"` write path** and add **write idempotency** (dedup on
   identical summary+type within a short window) to stop exact double-writes.
5. **Stop persisting no-op telemetry as memory.** The zeitgeist-skip logger
   should write at priority ≤ -1, use a TTL/ephemeral type, or not store at all
   — and never type a skip as a `decision`/`procedure`.
6. **Address session-log accumulation (§1b)** — `prune_by_age` on old
   `SLEEP/FLY SESSION` experience logs, since they are the dominant growth term.

### Telemetry

7. If utility telemetry is wanted, exclude boot-time bulk recalls from
   `access_count`, or track query-attributed access separately (§4).

## Bottom line

The episodic-storage **gate** is not the weak point; **write hygiene**, the
**curation pass**, and **session-log accumulation** are. Lexical redundancy is
~1% (exact) to 7.7% (loose) — but that floor undercounts the zeitgeist/snapshot
family, whose semantic redundancy is currently unmeasured (§5). Fix the write
leaks, implement the dedup `curate()` already advertises (lexical for exact;
embeddings-when-available for running-topic), prune session-log scaffolding, and
do **not** re-architect storage onto a measured surprise gate — it solves a
problem the store doesn't have.

The store has **no embeddings** (§6); a confabulation that it did was caught
mid-session by reading the code.

## Files

- `memories.json` — snapshot of 1,747 active memories (gitignored; regenerable)
- `dups_80.json` — near-duplicate pairs at ≥0.80 (gitignored; regenerable)
- `probe.py` — combined probe script
