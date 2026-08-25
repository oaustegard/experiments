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

Snappy codec in C, compiled to WebAssembly by clang with no emscripten. Read
in full on 2026-08-24 and benchmarked against `snappyjs`; `hysnappy_bench.mjs`
and `hysnappy_results.json` are the artifacts.

### Build

`Makefile` is 45 lines and the whole toolchain is one clang invocation:

```
clang --target=wasm32 -O3 -nostdlib -Wl,--export-all -Wl,--no-entry \
      -o uncompress.wasm c/uncompress.c
```

Then `base64 -w 0` the `.wasm` and `sed` the string into
`const wasm64 = '...'` in `js/uncompress.js`. The `.wasm` files are gitignored;
what ships on npm is `js/` only, with the module inlined as base64. Nothing
in the published package fetches anything.

`-nostdlib` means no libc, which is why `c/uncompress.c:5-37` defines its own
`memcpy` and `memmove` as byte loops. The blog's "not even memcpy" means no
libc `memcpy`; the functions exist, hand-written. The only includes are
`stdbool.h`, `stddef.h` and `stdint.h`, all header-only.

The C is adapted from `andikleen/snappy-c` (credited in the README's
references), 476 lines for uncompress and 198 for compress.

### The 4 KB rule stopped applying in 2023

`js/uncompress.js:64-72` instantiates with `new WebAssembly.Module(byteArray)`
followed by `new WebAssembly.Instance(mod)`, with the comment "only works for
payload less than 4kb", and the README explains the size target the same way.
Measured from the base64 in the published package:

| Module | Bytes |
|---|---:|
| `uncompress.wasm` | 3,458 |
| `compress.wasm` | 2,017 |

Chrome raised that limit from 4 KB to 8 MB in Chrome 115, June 2023
([chromestatus 5099433642950656](https://chromestatus.com/feature/5099433642950656),
[intent to ship](https://groups.google.com/a/chromium.org/g/blink-dev/c/nJw2zwaiJ2s/m/EYPgC5D3LwAJ)).
The 4 KB rule dated from when V8 compiled WASM eagerly with an optimizing
compiler; lazy compilation made synchronous compile cheap enough that they
re-measured on a Pixel 1 and picked 8 MB.

Verified in Chromium 141 by padding the real module past 4 KB with a valid
custom section, so the only rule that could reject it was the size rule:

| Module size | `new WebAssembly.Module` on the main thread |
|---:|---|
| 4,093 B | accepted |
| 7,997 B | accepted |
| 999,998 B | accepted |
| 8,388,607 B | accepted |
| 8,388,699 B | rejected: "disallowed on the main thread, if the buffer size is larger than 8MB" |

So the ceiling is exactly 8 MiB, and hysnappy sits three orders of magnitude
under a constraint that has not bound since 2023. The `-nostdlib` build still
buys a small module, which is worth having, but it is no longer buying
synchronous instantiation. Anything under 8 MiB gets that.

An earlier draft of this file went further and said the 4 KB budget was *why*
the `memcpy` is a byte loop "rather than anything vectorised". I made that up.
`wasm-variants/` builds the alternatives and measures them: the whole spread
from byte loop to `-msimd128` is 150 to 674 bytes, so even the old 4 KB rule
would not have forced the byte loop.

The useful version of this rule for us: a hot kernel we compile to WASM can be
up to 8 MiB and still instantiate synchronously on the main thread, with no
`await` and no second network request. web.dev's "Loading WebAssembly" article
still documents the old 4 KB figure, which is where hysnappy's comment and my
first reading of it both came from.

### Calling convention

There is no allocator. `js/uncompress.js:23-56` writes the input into WASM
linear memory at a hardcoded offset of 68,000 bytes ("clang uses some wasm
memory, so we need to skip past that"), puts the output immediately after it,
grows memory by whole 64 KiB pages if the total exceeds the current buffer,
calls `uncompress(inputStart, inputLength, outputStart)`, and slices the
result back out. Errors come back as negative integers that the JS maps to
messages: `-1` invalid length header, `-2` missing eof marker, `-3` premature
end of input.

`outputLength` is a required argument because there is no allocator to grow
into. For Parquet that is free — the page header carries the uncompressed
size — which is the sense in which this library was built for hyparquet rather
than as a general codec.

`snappyUncompressor()` instantiates once and returns a closure, so repeated
calls skip compilation. `snappyUncompress()` is the convenience wrapper that
instantiates per call, and hyparquet uses the former.

### Measured against snappyjs

Node 22 on a CCotw container, hysnappy 1.1.1 vs snappyjs 0.7.0, medians of
three sweeps each timed after two warm calls. hysnappy's figures include the
copy into WASM memory and the slice back out, which is what a caller actually
pays. Full numbers in `hysnappy_results.json`.

| Corpus | Ratio | Decompress MB/s | Compress MB/s |
|---|---:|---|---|
| json-ish, highly repetitive | 5.2% | 4,373 vs 521 = 8.39x | 5,658 vs 420 = 13.47x |
| pseudo-random, incompressible | 100% | 2,703 vs 586 = 4.61x | 345 vs 372 = 0.93x |
| already-compressed bytes | 99.7% | 3,034 vs 590 = 5.14x | 224 vs 303 = 0.74x |

This container is noisy. Within the committed run the three sweeps of the
json-ish decompress ratio came out 5.85x, 8.4x and 8.83x, a 1.51x spread;
the incompressible corpora are steadier at 1.14x and 1.07x. Read the ratios to
about one significant figure. Every row's direction held in every sweep and
every run.

Decompression wins on every corpus, by 4x to 9x. The blog claims "40% faster"
than standard JavaScript Snappy decompression, which understates it by roughly
an order of magnitude. That post is from 2025-07 and both libraries and V8 have
moved since; it may also have been measuring end-to-end Parquet parsing rather
than the codec alone.

Compression is where the shape changes. On compressible input hysnappy is
13.47x, and on input that does not compress it is *slower* than snappyjs:
0.93x on random bytes and 0.74x on an already-compressed file, and that
ordering held in every run. When almost every byte is emitted as a literal the
compressor is a copy loop, and hysnappy additionally pays the JS-to-WASM copy
in and the slice out. This costs nothing on the hyparquet read path, which only
calls the decompressor. It is worth knowing before reaching for `hysnappy` in
`hyparquet-writer` on a column that will not compress.

### Their benchmark's missing warm-up costs it 35%

`benchmark.js` reports absolute MB/s with no comparison library and no warm-up
loop. `hysnappy_cold.mjs` reproduces that shape in a fresh process per trial:
15 trials with no warm-up gave a median of 3,201 MB/s, against 4,334 MB/s for
the same loop after 200 warm-up iterations. That is **1.35x**, Mann-Whitney
one-sided z=3.9, p≈7e-05 (U=207, matching `scipy.stats.mannwhitneyu` exactly;
the p differs in the third figure because the script uses a normal
approximation). Read the shipped number as a floor about a third below the warm
steady state.

Two false starts are worth recording, because both produced a confident wrong
answer:

- **Measuring in-process does not work.** The first attempt allocated a fresh
  `snappyUncompressor()` per trial inside the main benchmark and saw almost no
  difference, because V8 had already optimized the same code paths during the
  sweeps above and a new closure does not undo that. The subprocess is the
  measurement.
- **Five trials per arm is underpowered here.** At n=5 two consecutive runs
  gave opposite orderings of the cold and warm spreads. n=15 separates them
  cleanly and reproduced across two independent runs. The max/min spread
  statistic never stabilised at any n, which is what an extreme-value
  statistic does under this much noise; the median shift is the number to
  quote. `ERRORS.md` carries the run-by-run detail.

### Could the decoder be vectorised

`wasm-variants/build_and_bench.py`. Four source variants against the pinned
v1.1.1 `c/uncompress.c`, each built with and without `-msimd128`, each
(variant, corpus) timed in a fresh node process with no allocation in the
timing loop. Three corpora that separate the two copy paths: 4 MiB
incompressible (every byte a literal copy), 4 MiB of a repeated 26-byte
alphabet (almost all back-references), and 4 MiB of tiled API JSON.

| Variant | Bytes | literal | match | json |
|---|---:|---:|---:|---:|
| `base` (as shipped) | 3,608 | 3,971 MB/s | 3,564 | 8,277 |
| `base` + `-msimd128` | 4,085 | **2.86x** | 0.99x | 0.85x |
| `b_i64` | 3,560 | 1.00x | 1.00x | **1.22x** |
| `b_i64` + `-msimd128` | 4,020 | 2.67x | **0.64x** | 1.18x |
| `c_widecpy` | 3,824 | **2.74x** | 1.00x | 1.01x |
| `c_widecpy` + `-msimd128` | 4,282 | 2.84x | 1.00x | 0.84x |
| `e_both` | 3,772 | 2.67x | 1.02x | 1.20x |
| `e_both` + `-msimd128` | 4,213 | 2.77x | **0.64x** | 1.05x |

Every variant decodes correctly, and all eight instantiate in Chromium 141,
which reports WASM SIMD as supported.

**Yes, and clang will do it for you.** Adding `-msimd128` to the existing
source auto-vectorises the byte-loop `memcpy` and gets 2.86x on incompressible
input for 477 more bytes. No intrinsics, no hand-written SIMD.

**SIMD is not where the speedup comes from.** Widening the hand-written
`memcpy` to 8-byte chunks, eight lines of C with no SIMD, lands at 2.74x on the
same corpus.
Adding SIMD on top moves it to 2.84x. Clang's auto-vectoriser and a manual
8-byte loop arrive at the same place, so what matters is that the copy stops
being one byte wide.

**SIMD costs speed on the other two corpora.** `-msimd128` takes realistic JSON
to 0.85x, and combined with the i64 fix it takes the back-reference corpus to
0.64x, consistently across five trials in each of two independent runs. A blanket
`-msimd128` would trade a large win on incompressible pages for a large loss on
compressible ones.

**The best single change is neither.** `unaligned_copy64` (`c/uncompress.c:198`)
guards its 64-bit path on `sizeof(void *) == 8`. On wasm32 that is false, so the
shipped build emits two 32-bit stores where one i64 store would do — WASM has
i64 natively regardless of pointer width. Removing the guard gives 1.22x on
realistic JSON and 1.00x elsewhere, and the module gets 48 bytes *smaller*.

Which of these is worth anything depends on the pages hyparquet actually reads.
The literal-heavy case is not exotic: a snappy-compressed Parquet page of
high-entropy float data is mostly literals, and that is where the 2.7x sits. A
dictionary-encoded string column is match-heavy, where none of this moves.

Two caveats. All of it ran under Node. The browser decode figures elsewhere in
this file are much lower, so these ratios may not carry to a page. And clang 18
here produces a 3,608-byte baseline against the published 3,458, so the absolute
sizes are not hyparam's; only the deltas between my own builds mean anything.


### Practical as a general in-browser compression library: no

`practicality.mjs` and `practicality_results.json`. Four findings, each of
which is about fitness rather than speed.

**It speaks the block format.** Compressed output starts `0x2b 0x28 0x68 ...`;
the framed `.sz` stream identifier `0xff 0x06 0x00 0x00 sNaPpY` is absent. So
there is no streaming decode, no chunking, and no CRC32C. The framed format
checksums every chunk. The block format has no checksum at all.

**Corruption is usually silent.** Flipping one bit at an even stride through a
6,680-byte compressed buffer, 400 trials: 163 threw, **237 returned wrong bytes
with no error**, none returned correct bytes. So 59% of single-bit corruptions
produce plausible-looking garbage. Anything crossing a network needs its own
checksum on top.

**`outputLength` is unchecked in both directions.** Pass it too small and you
get a short buffer of wrong bytes; too large and you get the right bytes
zero-padded to the length you asked for. Neither throws.

| `outputLength` error | Result |
|---|---|
| -1000, -100, -1 | returns that many bytes, wrong content, no error |
| exact | correct |
| +1, +100 | correct prefix, zero-padded, no error |

Parquet makes this a non-issue because the page header carries the uncompressed
size. For a general library it is a trap, and combined with the missing
checksum there is no layer that will catch a mistake.

**The browser already ships a better-compressing codec.** `CompressionStream`
with `gzip` / `deflate` / `deflate-raw` is native, costs zero shipped bytes, has
framing, and compresses far harder:

| Corpus | snappy | gzip | snappy is |
|---|---:|---:|---|
| API JSON, 37 KB | 6,680 B (17.9%) | 3,859 B (10.4%) | 1.73x larger |
| English prose, 49 KB | 2,423 B (4.9%) | 304 B (0.6%) | 7.97x larger |

Snappy is not a browser transport format — there is no `Content-Encoding:
snappy` — so for bytes you control, gzip wins on the axis you are actually
paying for.

What snappy buys is CPU. On an 8 MiB payload, decode 1,208 vs 181 MB/s and
encode 4,287 vs 91 MB/s. If you are decoding hundreds of megabytes of
already-snappy data client-side, that gap is what you came for. If you are
choosing a format for your own payloads, you are trading 1.7x to 8x more bytes
for CPU you probably were not short of.

### It runs in a real browser

Chromium 141 headless, hysnappy loaded from its published ESM files over HTTP:

| | |
|---|---|
| round trip | correct |
| first `snappyUncompressor()` | 0.9 ms |
| snappy decode | 136 MB/s |
| gzip decode via `DecompressionStream` | 58 MB/s |
| `CompressionStream` present | yes |

The browser decode figure is well below Node's on the same corpus, and the
gzip figure at this size is dominated by per-call `CompressionStream` plus
`Response` setup rather than by inflate. Read the 37 KB row as an API-shape
comparison and the 8 MiB row above as the codec comparison.


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
