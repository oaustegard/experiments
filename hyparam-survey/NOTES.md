# Reading notes

Raw notes behind `README.md`, kept at full detail because the README compresses
most of this to one line each. Every claim carries a file and line so it can be
checked against the source rather than against my paraphrase.

Read on 2026-08-24. Repos cloned anonymously from `github.com/hyparam` at the
commits their default branches pointed to that day; `hypvector` read from the
npm tarball pinned in `extract_evidence.py` (the GitHub repo is private).

## Access

`git clone https://github.com/hyparam/<repo>` works anonymously for every repo
in `repos.json`. It prompts for auth on `hypvector`, `hypgrep` and `hypstore`,
all three of which exist as directories in `hyparam/demos` and as npm packages.
`hypvector@0.2.2` ships the full unminified `src/` (20 files, 160 KB), so the
private repo costs nothing here.

The GitHub API path `orgs/hyparam/repos` returns 403 through the CCotw agent
proxy — sessions are bound to their configured repositories. `mcp__github__
search_repositories` with `org:hyparam` works globally and is what produced
`repos.json`.

## hypvector source files

`src/` is 1,826 lines across 14 files.

| File | Lines | Contents |
|---|---:|---|
| `writeVectors.js` | 343 | `writeVectors`, `streamVectors`, schema, KV metadata |
| `cluster.js` | 241 | `binaryKMeans`, `reorderClustersByHamming`, `hammingDistanceBytes` |
| `utils.js` | 237 | metrics, `l2Normalize`, `packFloat32`, `packBinary`, `expectedHamming` |
| `searchVectors.js` | 176 | option validation, algorithm dispatch, multi-source merge |
| `search/rerank.js` | 171 | the three-phase binary+cluster+rerank path |
| `search/chunks.js` | 157 | SWAR Hamming kernels over chunks and flat buffers |
| `search/heap.js` | 112 | bounded top-K heaps, metric direction, tie-breaks |
| `search/ranges.js` | 101 | `selectClusterRowRanges`, `mergeRanges`, `coalesceRuns` |
| `constants.js` | 65 | every default, each with the measurement that set it |
| `search/exact.js` | 67 | single-pass float32 scan |
| `readVectors.js` | 62 | stream `{id, vector}` back out |
| `prefetch.js` | 57 | pull the whole binary column into RAM |
| `asyncBufferFactory.js` | 31 | URL / path / AsyncBuffer resolution, `cachedAsyncBuffer` |

`constants.js` is the file to read first. Every default carries the experiment
that produced it in a comment — `defaultClusterProbeCap = 48` cites a WildChat
1024-dim sweep at 1M and 3.2M, `defaultBinaryPageSize = 32 * 1024` cites the
384-dim wiki benchmark, `defaultBinaryMinSpread = 1/16` cites measured spreads
of 0.3–0.5 of dimension for healthy embeddings. Excerpted in
`evidence/constants-probe-cap.js` and `evidence/constants-binary-spread.js`.

## hypvector — details that did not fit the README

**Two write paths.** `writeVectors.js:60-79` takes a streaming path when
`binary` is set explicitly and no clustering is requested: the schema is known
up front, rows emit in input order, peak memory is one row group. Auto-binary
needs N before it can choose the column set, and clustering needs a global
k-means plus a row reorder, so both buffer the whole corpus. Anyone packing a
large corpus with clustering pays O(N) memory at build time.

**`binaryKMeans` refuses non-multiple-of-4 byte counts** (`cluster.js:26-30`).
The SWAR popcount loop iterates u32 words and would silently skip tail bytes.
dim=384 gives 48 bytes and is fine; an arbitrary dim is not.

**Early termination in the assignment loop** (`cluster.js:71`): `if (d >=
bestDist) break` inside the per-word popcount, so a centroid that is already
worse than the incumbent stops accumulating.

**Empty clusters reseed from a random row** (`cluster.js:111-115`) rather than
being dropped, so cluster count stays fixed across iterations.

**Six k-means iterations by default**, and the last iteration skips the
centroid update since nothing consumes it (`cluster.js:82-83`).

**The contiguity check before the flat view** (`chunks.js:55-70`) is the kind of
bug that would be silent. A row range read from a clustered file can be
assembled from several page buffers; building a `Uint32Array` span over
`rows[0].buffer` would either throw `RangeError` or, if the first buffer happens
to be large enough, score the wrong bytes. They verify in O(1) by checking the
last row's `buffer` identity and `byteOffset` against the expected stride.

**Ties are broken by `rowIndex`** (`heap.js:36-49`, `rerank.js:121-123`) so
results do not depend on the completion order of parallel range reads. Without
it the same query returns different top-K on different runs.

**Phase 1 deliberately does not use `useOffsetIndex`** (`rerank.js:52-64`): the
binary column is `dim/8` bytes per row, so per-page seeking costs a round trip
without saving meaningful bytes. Phase 2's float32 column is ~32x larger per
row and does use it.

**`probe` overloads its type**: in `(0, 1]` it is a fraction of clusters, above
1 it is an absolute count (`ranges.js:38-45`). The cap of 48 applies only when
`probe` is left undefined; an explicit value is honored literally.

**Multi-source search** (`searchVectors.js:71-99`) queries an array of sources
in parallel and heap-merges. It asserts that all sources agree on metric
direction. That covers sharding if one file gets too large.

## hyparquet — the option surface

From `src/types.d.ts:28-45`, which the README does not document. This is the
part worth knowing even if we never write JavaScript, because it is a compact
statement of what a range-request Parquet reader can prune on:

| Option | Effect |
|---|---|
| `columns` | project columns |
| `rowStart` / `rowEnd` | physical row range |
| `filter` | Mongo-shaped predicate: `$gt $gte $lt $lte $eq $ne $in $nin $not`, composed with `$and $or $nor` |
| `useOffsetIndex` | limit column-chunk reads to covering pages (default false) |
| `useBloomFilters` | fetch bloom filters, skip row groups on `$eq`/`$in` (default false) |
| `usePageIndex` | fetch column + offset index for filter columns, skip non-matching pages (default false) |
| `onChunk` / `onPage` | column-oriented callbacks as data arrives; skips the row transpose entirely |

`parquetScan` (`types.d.ts:58-81`) returns `{ metadata, ranges, readColumn }` —
a prepared scan with physical row ranges and a lazy per-column read, taking a
`pruningFilter` used only to prune ranges. hypvector's phase 2 sits on top of
this.

Bloom filters live in `src/bloom.js` (split-block bloom filter: `blockIndex`,
`blockMask`, `sbbfInsert`, `sbbfContains`, `hashParquetValue`), and
`hyparquet-writer/src/bloom.js` writes them with `optimalNumBytes`. A lexical
membership prune over a corpus column is available without inventing anything.

`src/filter.js` carries `canSkipRowGroup`, `canSkipStats`, `matchFilter` — the
statistics-based row-group pruning the blog post describes as the thing pandas
and pyarrow default readers skip.

## squirreling

Streaming SQL engine, `src/` 59 files / 398 symbols, 13 KB minified, zero
dependencies. Rows are `AsyncGenerator`s and cells are `() => Promise<T>`
thunks, so an expensive UDF runs only for cells that survive to the result set.
Their example is `SELECT name, AI_SCORE(description) FROM products` where
`AI_SCORE` is an LLM call.

The part worth borrowing is the data-source contract (`README.md:96-137`):

```typescript
interface ScanResults {
  rows(): AsyncIterable<AsyncRow>
  appliedWhere: boolean        // did the source apply the WHERE?
  appliedLimitOffset: boolean  // did the source apply LIMIT/OFFSET?
}
```

A source pushes down what it can and reports what it did; the engine applies
the rest. Any retrieval backend with partial predicate support wants exactly
this handshake, and returning a flag beats returning a silently-unfiltered
stream.

`src/backend/` holds only `batch.js`, `batchAdapters.js`, `dataSource.js` — no
Parquet binding. The Parquet source lives in the demo and in the `hyperparam`
npm package, so the engine itself stays source-agnostic.

Also present: a `spatial/` directory (11 files, 63 symbols) implementing
`ST_Intersects`, `ST_Contains`, `ST_DWithin` and friends, and `syntax.md`,
a log of SQL syntax failures they hit in production.

## icebird

Iceberg client, `src/` 50 files / 378 symbols. Relevant pieces if a corpus ever
wants versioning without a catalog service: `metadata.js` (version listing and
time travel), `manifest.js`, `prune.js` (`partitionMightMatch`,
`fileMightMatch`, `columnMightMatch` — conservative pruning at three levels),
`delete.js` (position and equality deletes), `sigv4.js` + `s3.js` (request
signing from the browser), `json.js` (a hand-written JSON parser: Iceberg writes 64-bit snapshot ids as
bare JSON numbers, and `JSON.parse` truncates them to lossy doubles).

## hysnappy

Snappy decompression as hand-written WASM, under 4 KB. The reason for the size
target, from the blog: a WASM module under 4 KB can be compiled synchronously
by the browser, which removes the extra round trip that normally makes WASM
slower to start than JavaScript. No emscripten, no `memcpy`.

## Blog: design rules and their numbers

`_posts/2025-07-24-quest-for-instant-data.md`, Kenny Daniel. The claims that
generalize past Parquet:

- The target is 500 ms time-to-first-interactivity, from Liu & Heer 2014.
  DuckDB took ~5 s and pandas ~57 s to show 10 rows of a cloud-hosted dataset.
- Metadata fetch: skip the HEAD → footer-length → footer three-step and
  optimistically fetch the last 512 KB, which covers the metadata 99% of the
  time. Over the internet an 8-byte fetch costs nearly what a 512 KB fetch
  costs.
- DuckDB-WASM issues exponentially-increasing request sizes in series, which
  suits local disk and is pathological over a network. Browsers give you 6+
  concurrent connections; use them.
- Engine size is latency. Their core is ~10 KB gzipped against several MB of
  WASM.
- The stated inversion: thin server, client does the work; round trips matter
  more than total bandwidth; time-to-first-byte is the optimization target.

Six months to parse every Parquet file in the wild — 8 physical types, 22
converted types, 17 logical types, 8 codecs, 2 major versions.

## remax_kb and remex — where the comparison anchors

For the README's claim that remax_kb downloads the whole dense index:

- `js/kb-reader.js:539` lists the required `.kbi` entries; `:650` is
  `const vecBytes = zip.read("vectors.bin")`, the whole dense column into
  memory; `:652-654` validates its length against `total * this._rowBytes`.
- `js/kb-reader.js:864-880` is `fetchChunks`, the only HTTP Range use. It
  fetches chunk *text* per hit; the vectors never travel this path.
- `SPEC_v2.md:33-34` states the split as designed: the `.kbi` is "hot — agent
  downloads on startup" and the `.kbc/` is "cold — fetched per-hit via HTTP
  Range".

For the claim about `remex.IVFCoarseIndex`:

- `remex/ivf.py:1-8` — "Partitions a compressed corpus into `2**n_bits` cells
  via a **data-oblivious** hash — no k-means, no training, no fitting."
- `remex/ivf.py:29-31` — the memory accounting: `cell_ids` uint16 at 2 bytes
  per vector, `sorted_idx` int64 at 8 bytes per vector, `cell_offsets` int64
  per cell.
- `remex/ivf.py:86-89` — `sorted_idx` is a CSR-style permutation; cell `c`
  occupies `sorted_idx[cell_offsets[c]:cell_offsets[c+1]]`. The corpus itself
  is never reordered, which is why the cells are not contiguous on disk.

Both properties are correct for an in-memory scan and wrong for a reader that
must range-fetch. That is the whole of borrow #1 in the README.
