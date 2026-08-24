# hyparam — survey

Hyparam (Kenny Daniel's company, [hyperparam.app](https://hyperparam.app))
ships a browser-native data stack: a from-scratch JavaScript Parquet reader
and writer, an Iceberg client, a streaming SQL engine, and — the reason for
this survey — a vector index that lives inside a Parquet file and is queried
by HTTP range request with no server in the middle.

Everything is MIT, zero-to-three dependencies per package, and written by
people who care about time-to-first-byte more than throughput. The Parquet
reader is funded by a Hugging Face open-source grant.

Surveyed 2026-08-24 from the public repos plus the `hypvector` npm tarball.

## Public repos

27 public repos. The ones that matter:

| Repo | Stars | What it is |
|---|---:|---|
| [`hyparquet`](https://github.com/hyparam/hyparquet) | 859 | Parquet reader. Zero dependencies, ~10 KB gzipped core, every physical type / encoding / codec. Range-request-first. |
| [`hightable`](https://github.com/hyparam/hightable) | 277 | React virtual-scroll table over async row sources. |
| [`icebird`](https://github.com/hyparam/icebird) | 126 | Apache Iceberg client — manifests, snapshots, time travel, partition pruning, position deletes, SigV4 signing. |
| [`hyparquet-writer`](https://github.com/hyparam/hyparquet-writer) | 60 | Parquet writer. Dictionary/delta/RLE encodings, bloom filters, column + offset indexes, page-size control. |
| [`hyllama`](https://github.com/hyparam/hyllama) | 50 | GGUF metadata parser. Reads llama.cpp model headers without downloading the weights. |
| [`squirreling`](https://github.com/hyparam/squirreling) | 38 | Streaming SQL engine, 13 KB, zero deps. Rows are AsyncGenerators, cells are async thunks. |
| [`hysnappy`](https://github.com/hyparam/hysnappy) | 27 | Snappy decoder in hand-written WASM, under 4 KB so the browser can compile it synchronously. |
| [`hyparquet-compressors`](https://github.com/hyparam/hyparquet-compressors) | 18 | gzip/brotli/zstd/lz4/lzo for hyparquet. |
| [`demos`](https://github.com/hyparam/demos) | 10 | Ten runnable Vite apps, one per library. |
| [`parquet-grep`](https://github.com/hyparam/parquet-grep) | 2 | `grep` over local or remote Parquet, CLI. |

`hypvector`, `hypgrep` and `hypstore` appear as demo directories and npm
packages but their GitHub repos are not public — `git clone` returns an auth
prompt. `hypvector`'s npm tarball ships the full unminified `src/`, which is
what this survey read.

The blog names the design rule the whole stack follows: *round trips matter
more than total bandwidth, and time-to-first-byte is the query optimization
target.* Hyparquet fetches the last 512 KB of a file optimistically rather
than doing the HEAD → footer-length → footer three-step, because over the
internet an 8-byte fetch costs nearly what a 512 KB fetch costs.

## hypvector

A Parquet file with three columns is the vector database.

| Column | Type | Bytes/row |
|---|---|---|
| `id` | `STRING` | variable |
| `vector` | `FIXED_LEN_BYTE_ARRAY(4 × dim)`, raw float32, `UNCOMPRESSED` | `4 × dim` |
| `vector_bin` | `FIXED_LEN_BYTE_ARRAY(dim/8)`, one sign bit per dim | `dim/8` |

Dimension, metric, normalization flag, cluster count, base64 centroids and
per-cluster row counts all live in the Parquet key-value footer metadata, so
the file is self-describing and a reader needs no sidecar.

### Write path

`writeVectors` L2-normalizes, packs float32 and sign bits, then runs k-means
on the **1-bit codes** using Hamming distance and bit-majority-vote centroid
updates. Rows are physically reordered so each cluster is one contiguous run,
and each cluster becomes its own row group. Cluster count defaults to
`sqrt(N)/2`; the binary column switches on above 10k rows.

Two details in that path are worth stealing on their own:

**Greedy Hamming renumbering.** After k-means, `reorderClustersByHamming`
walks the centroids nearest-neighbour-first and renumbers them so adjacent
cluster ids are close in Hamming space. Since rows are sorted by cluster id,
the top-N nearest clusters to any query then tend to fall in *adjacent* row
ranges, which `mergeRanges` collapses into fewer reads. The renumbering costs
nothing at query time and buys fewer HTTP round trips.

**No permutation array.** Because rows are physically reordered at write
time, the file stores only per-cluster row counts; a cumulative sum recovers
each cluster's `[start, end)`. No `sorted_idx` rides along.

**A degeneracy guard on the sign codes.** Before enabling the binary column
automatically, `writeVectors` samples 4096 rows and measures the expected
Hamming distance between random pairs. If it falls below `dim/16`, the sign
codes are near-all-ties, which happens when an embedding's components are all
non-negative, and phase-1 ranking would then be near-random. In that case it
writes no binary column and search stays an exact scan.

### Query path

1. Rank clusters by Hamming distance from the query's sign code, take the
   nearest `probe` fraction (default 0.25, capped at 48 clusters), merge
   their row ranges, and Hamming-scan only those ranges of `vector_bin`.
2. Take the top `topK × rerankFactor` candidates, coalesce their row indices
   into runs (merging gaps ≤ 64 rows), and issue one ranged `parquetRead` per
   run for the `vector` column only, with `useOffsetIndex` so only the
   covering pages are fetched. Score under the exact metric.
3. Fetch the `id` column for the top-K winners only. Ids are variable-length,
   and reading them for every candidate roughly doubles phase-2 cost.

Everything runs on typed-array views with a SWAR popcount, no per-row
allocation on the hot path, and a contiguity check before taking the flat
`Uint32Array` view — a clustered row range can be assembled from several page
buffers, and spanning it blindly would read out of bounds.

`prefetchBinary` pulls the whole `vector_bin` column into RAM
(`N × dim/8` bytes) so phase 1 runs with no network at all. On the 156k wiki
dataset that is 7.5 MB.

`searchVectors` also takes an array of sources, querying them in parallel and
heap-merging to a global top-K.

### Their published numbers

156k × 384-dim wiki embeddings, 249 MB on S3: one top-10 query reads ~6 MB
across ~160 ranged fetches at ~91% recall. Over localhost with 20 ms injected
per-request latency, 139 ms/query against 362 ms for an exact full scan.

Their WildChat comparison table, 838k OpenAI 1024-dim vectors:

| Engine | Storage | Recall@10 | Query | All-in / mo |
|---|---:|---:|---:|---:|
| hypvector | 3.58 GB | 0.975 | 46 ms (local) | ~$0.08 |
| pgvector | 11.5 GB | 0.965 | ~1 ms | $94 |
| Qdrant | 3.6 GB | 0.965 | 2 ms | $62 |
| turbopuffer | 3.43 GB | 0.93 | 60 ms (cloud) | $16 min |
| Pinecone | 3.43 GB | 0.97 | 125 ms (cloud) | $50 min |

The latency columns mix local compute against live cloud round-trips, which
their footnotes say. Treat the dollar column as the honest one.

## A 50k-vector sweep

`hypvector_probe.mjs` in this directory: 50,000 synthetic 384-dim vectors,
64 Gaussian centers with σ=0.9 noise, L2-normalized; 20 queries, each a
perturbed corpus vector. Local file, node 22, no network. Recall is measured
against hypvector's own exact scan, so it isolates the approximation and not
the data.

The written file is 79,969,680 bytes against 76,800,000 bytes of raw float32,
104.1% of raw, in 112 clusters of one row group each.

| Search | ms/query | recall@10 |
|---|---:|---:|
| exact full scan (`rerankFactor: 0`) | 198 | 100% |
| default (`rerankFactor: 10`, `probe: 0.25`) | 19 | 50% |
| `rerankFactor: 17` (their `N/3000` rule) | 21 | 64% |
| `rerankFactor: 50` | 41 | 86% |
| `rerankFactor: 100` | 63 | 93% |
| `probe: 1.0`, `rerankFactor: 10` | 41 | 49% |
| `probe: 1.0`, `rerankFactor: 50` | 100 | 82% |

**The claim in `constants.js` reproduces.** Their comment says residual
misses are a `rerankFactor` limit and not a `probe` limit. Scanning every
cluster instead of a quarter of them moved recall from 50% to 49% at
`rerankFactor: 10`, and from 86% down to 82% at 50, at 2.4x the time. The
candidate budget is the whole curve; probing wider only pulls in more
Hamming ties competing for the same fixed budget.

**The default is not a default you can ship on.** 50% recall@10 at
`rerankFactor: 10` on this corpus. Their own README shows 18% at that setting
on a 1M synthetic set, and their `N/3000` rule of thumb gives 17 here, which
measured 64%. Reaching 93% costs `rerankFactor: 100` and a third of the exact
scan's time. Synthetic Gaussian clusters at σ=0.9 are a harder case than real
embeddings — their wiki number is 91% at the default — so read this as a
warning about tuning per corpus rather than a refutation of their figure.
Anyone adopting this measures recall on their own vectors before choosing
`rerankFactor`.

## Borrows for remax_kb

remax_kb v2 and hypvector are the same architecture with a different
container. Both quantize to 1 bit per dimension, scan by Hamming, and rerank
the survivors at higher precision. The difference is what the reader has to
download.

remax_kb's `.kbi` is a zip the reader opens whole: `kb-reader.js` calls
`zip.read("vectors.bin")` and scans every row. HTTP Range is used only for
the `.kbc` chunk text, per hit. So the dense index must fit in memory and
every query is O(N). At the xr corpus scale — 42.5k chunks — that is correct
and fast. At a million chunks it stops working in a browser.

hypvector never downloads the index. It reads a footer, picks 28 of 112
clusters, and range-fetches only those clusters' pages. That is the change
that takes their story from 156k vectors to 3.2M.

Ranked by value:

1. **Cluster the codes at pack time and physically reorder rows.** Store
   centroids and per-cluster counts in `manifest.json`; the reader then
   range-fetches only the probed clusters' slices of `vectors.bin`. This is
   the single change that lifts remax_kb's ceiling from "index fits in RAM"
   to "index fits in object storage." `remex.IVFCoarseIndex` already
   partitions a corpus into cells, but it is deliberately data-oblivious
   (LSH or rotated-prefix, no k-means) and it keeps a `sorted_idx`
   permutation of 8 bytes per vector — it was built for an in-memory scan
   where cell contiguity is free. For an artifact fetched over the network,
   physical reordering removes the permutation array *and* makes each cell
   one contiguous byte range.

2. **Renumber cells so nearby cells get nearby ids.** remex's cell ids come
   from a hash, so the `nprobe` visited cells are scattered across the
   file. hypvector's greedy Hamming walk makes them cluster, and merged
   ranges mean fewer requests. Applies to any IVF served over range
   requests, remex's included. Cost is one k×k Hamming pass at build time.

3. **The degeneracy guard.** Sample a few thousand rows, measure expected
   pairwise Hamming, and refuse to enable the 1-bit path when it falls below
   `dim/16`. remax_kb centers before taking signs, which mitigates the
   all-non-negative case but does not detect it. An encoder that quietly costs 40 points of recall is worth four
   thousand rows of sampling to detect.

4. **Defer the id/text fetch to the winners.** remax_kb already does this via
   `.kbc` Range fetches. Their phase-3 measurement puts a number on it:
   fetching ids alongside vectors instead of after cost 50 ms against 22 ms,
   more than double.

5. **Scale `rerankFactor` with N.** remax_kb's `defaultOverFetch(k)` is the
   same knob under a different name, and their rule of thumb is
   `max(10, N/3000)`. Both their measurements and mine say a constant is
   wrong.

### Parts that do not transfer

**The container.** Parquet buys hypvector a self-describing footer, offset
indexes, page-level range reads and every tool in the ecosystem for free.
remax_kb's `.kbi`/`.kbc` split already has the properties that matter and
carries BM25 postings, which Parquet has no place for. Swapping containers is
a rewrite that buys interoperability we are not asking for.

**`UNCOMPRESSED` float32 vectors.** hypvector stores raw fp32 and pays 4×dim
bytes per row because rerank precision is the product. remex's whole point is
that a Lloyd-Max 4-bit code reranks nearly as well at a quarter of the bytes.
Our two-tier arrangement is stronger here; hypvector's is 1-bit-then-fp32
with nothing in between.

**Dense-only.** hypvector has no lexical leg. remax_kb's BM25 + RRF fusion
catches the proper nouns and identifiers a sign-bit code cannot.

## The rest of the stack

- **`hyparquet`'s option surface is worth reading even if you never write
  JavaScript.** `parquetRead` takes a Mongo-shaped `filter`
  (`$gt/$in/$and/$or/$nor`), `useOffsetIndex` for page-level range reads,
  `useBloomFilters` for row-group skipping on `$eq`/`$in`, `usePageIndex`
  for page-level skipping, and `onChunk`/`onPage` callbacks that emit
  column-oriented data as it arrives instead of transposing to rows. There is
  also a `parquetScan` returning physical row ranges plus a lazy
  `readColumn`, which is the primitive hypvector is built on.
- **`squirreling`** is 13 KB of streaming SQL where cells are async thunks,
  so an expensive UDF — an LLM call, an API fetch — only runs for cells that
  survive to the result. `AsyncDataSource.scan()` returns `appliedWhere` and
  `appliedLimitOffset` flags so a source can push down what it can and let
  the engine handle the rest. That handshake is a good pattern for any
  retrieval backend with partial predicate support.
- **`hysnappy`** exists because a WASM blob under 4 KB can be compiled
  synchronously by the browser, which saves the extra round trip that
  normally makes WASM slower to start than JavaScript. Useful constraint to
  remember for any hot kernel we might want to ship to a page.
- **`icebird`** does partition pruning, manifest-level `fileMightMatch`
  filtering, position deletes and SigV4 signing from the browser. If a
  corpus ever wants versioning and time travel without a catalog service,
  this is the reference.
- **`hightable`** is the virtual-scroll table the demos all use, over async
  row sources with late-materialized cells.
- **`parquet-grep`** is a small CLI that greps local or remote Parquet.
- **`hyllama`** reads GGUF headers over range requests without pulling the
  weights, which is how you inspect a 40 GB model file in a second.

## Reproduce

```bash
cd hyparam-survey
npm install
node hypvector_probe.mjs
```

Writes an 80 MB `v.parquet` (gitignored), runs the sweep, prints the table.
Takes about five minutes. `node node_modules/hypvector/bin/cli.js v.parquet`
prints the format header.
