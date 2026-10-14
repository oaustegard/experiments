# EmbeddingGemma 2 in a vector database: Matryoshka truncation against quantization

Gus (@gusthema.bsky.social, EmbeddingGemma PM) replied on Bluesky to the 16× smaller index
post on muninn.austegard.com that 2-bit quantization
shrinks storage but leaves search cost scaling with all 768 dimensions, while a Matryoshka-truncated
vector has fewer dimensions to compare and so searches faster. This directory measures that claim
on real EmbeddingGemma 2 vectors in FAISS and Qdrant.

Setup: `google/embeddinggemma-2` text-only, fp32 768d, normalized. Base set is the first
100,000 BEIR Quora corpus documents; 1,000 test queries. Quality is recall@10 against the
exact fp32 768d top-10, so a truncated or quantized index is charged for every neighbour it
loses. Timings are single-thread search on a 4 vCPU Xeon (260 MiB shared L3), one benchmark
at a time on an otherwise idle box. The vectors are in the R2 embedding store at
`embeddings/quora/embeddinggemma-2/fp32-768-first100k/`.

## Matryoshka ladder ceilings

Exact search on truncated, renormalized vectors (`remex_flat.py`):

| dims | bytes/vec | recall@10 |
|---|---|---|
| 768 | 3072 | 1.000 |
| 512 | 2048 | 0.926 |
| 256 | 1024 | 0.850 |
| 128 | 512 | 0.715 |

No index setting raises these. A d=128 index returns 71.5% of the true neighbours at best.

## Quantized codes at matched quality

Flat search, remex 2.0.0 centred (mean of 3 rotation seeds) and FAISS 1.15.1
`IndexRaBitQ`, recall@10:

| code | remex 2.0 | bytes | FAISS RaBitQ | bytes |
|---|---|---|---|---|
| 1-bit d=768 | 0.837 | 104 | 0.833 | 104 |
| 2-bit d=768 | 0.914 | 200 | 0.912 | 212 |
| 4-bit d=768 | 0.973 | 392 | 0.970 | 404 |
| 1-bit d=512 | 0.796 | 72 | 0.796 | 72 |
| 2-bit d=512 | 0.877 | 136 | 0.875 | 148 |
| 4-bit d=256 | 0.847 | 136 | 0.845 | 148 |

Every rung of the ladder has a quantized code that matches it in a fraction of the bytes:
1-bit d=768 (104 B) beats d=128 (512 B); 4-bit d=256 (136 B) ties d=256 (1024 B); 2-bit
d=768 (200 B) comes within 0.012 of d=512 (2048 B), and 4-bit d=768 (392 B) passes it.
remex 2.0 and RaBitQ differ by +0.004 to −0.002 across all nine bit-width × dimension
cells, inside remex's seed range. remex 1.2.0 trailed by 0.02–0.03 at 1 bit; 2.0 adopted
RaBitQ's inner-product estimator (`results_remex_flat.json`, `results_remex2_flat.json`,
`RESULTS_remex_flat.md`).

## HNSW search speed

FAISS HNSW, M=32, efConstruction=200, efSearch swept 16–512. Best QPS at which each
config reaches the target recall (— = not reached anywhere in the sweep). "+ rescore" re-ranks 4× the
candidates with the full-precision vectors (`results_hnsw.json`, `results_hnsw_ladder.json`):

| config | bytes/vec | best recall | QPS @0.8 | QPS @0.9 | QPS @0.95 |
|---|---|---|---|---|---|
| fp32 d=128 | 512 | 0.715 | — | — | — |
| fp32 d=256 | 1024 | 0.850 | 18,811 | — | — |
| fp32 d=512 | 2048 | 0.926 | 10,933 | 10,933 | — |
| fp32 d=768 | 3072 | 1.000 | 8,664 | 8,664 | 8,664 |
| RaBitQ 2-bit d=512 | 148 | 0.873 | 16,683 | — | — |
| RaBitQ 2-bit d=768 | 212 | 0.911 | 9,507 | 6,357 | — |
| RaBitQ 1-bit d=768 + rescore | 104 | 0.996 | 11,617 | 11,617 | 11,617 |
| RaBitQ 2-bit d=768 + rescore | 212 | 0.999 | 8,603 | 8,603 | 8,603 |

Without rescoring, fewer dimensions are faster at a given recall while the index fits in
cache: fp32 d=512 runs 10.9k QPS at recall 0.9 against 6.4k for 2-bit d=768, at ten times
the bytes, and fp32 d=256 runs 18.8k at recall 0.8. With rescoring, 1-bit d=768 runs 11.6k
QPS at recall 0.9, ahead of d=512, and the rescored codes are the only configs besides full
fp32 that pass 0.95. FAISS keeps the rescoring
vectors in RAM, so the rescore rows measure speed, not memory; the RAM-cap section below
measures the memory side.

## Brute-force scan cost

Synthetic unit vectors, since scan cost depends on bytes and code format and not on what
the vectors encode. Nanoseconds per vector scanned, 1 thread / 4 threads
(`results_flat.json`):

| format | bytes/vec | 16 MB (in cache) | 1 GB (beyond L3) |
|---|---|---|---|
| RaBitQ 2-bit d=768, FastScan | 212 | 20.8 / 5.0 | 13.8 / 4.0 |
| RaBitQ 1-bit d=768 | 104 | 21.2 / 8.1 | 38.1 / 8.1 |
| RaBitQ 2-bit d=512 | 148 | 17.2 / 5.7 | 37.9 / 7.5 |
| fp32 d=128 | 512 | 21.7 / 6.0 | 47.2 / 11.7 |
| RaBitQ 2-bit d=768 | 212 | 29.1 / 7.2 | 53.4 / 10.3 |
| fp32 d=768 | 3072 | 127.5 / 43.5 | 280.3 / 65.9 |

Past the cache, a scan is bound by memory bandwidth. 2-bit codes at all 768 dimensions
in FAISS's SIMD FastScan layout scan 3.4× faster than fp32 d=128 single-threaded; the
plain RaBitQ layouts run at about the d=128 rate.

## Qdrant and the RAM cap

Qdrant 1.19.1, HNSW m=32, original vectors on disk for quantized collections
(`results_qdrant.json`). With one sequential client every collection runs 477–646 QPS,
about 2 ms per query, so the client round trip hides the engine's distance cost; these
runs measure recall and memory, not engine speed. Recall matches FAISS: 2-bit d=768
without rescore peaks at 0.77, with rescore at 0.988.

For the memory test the server ran inside a cgroup-v1 memory limit of 235 MB, launched in
the group so all its pages are charged there, with page cache dropped before each phase.
235 MB holds the 2-bit index; fp32 768d needs about 460 MB.

| collection | ef | uncapped QPS / p99 | capped QPS / p99 | capped major faults |
|---|---|---|---|---|
| fp32 d=768 | 64 | 360 / 4.9 ms | 26 / 65.4 ms | 3.1M |
| fp32 d=768 | 256 | 239 / 6.8 ms | 9 / 185.4 ms | 9.8M |
| 2-bit d=768 | 64 | 414 / 3.7 ms | 404 / 4.7 ms | 0 |
| 2-bit d=768 + rescore | 64 | 401 / 3.8 ms | 267 / 6.1 ms | 91K |
| 2-bit d=768 + rescore | 256 | 320 / 4.9 ms | 252 / 7.5 ms | 74K |

Recall is unchanged under the cap (0.997/0.999 fp32, 0.768 2-bit, 0.993/0.995 with
rescore). Under the cap the fp32 index pages its vectors from disk and slows 14–27×. The
2-bit index stays resident and does not fault. Rescoring reads the candidates' originals
from disk and costs 21–33% of its throughput.

## Limits

- 100k vectors, one corpus (Quora duplicate questions), one machine. The HNSW timings are
  cache-resident; a corpus large enough to push the compact indexes past L3 would need
  about 5M vectors.
- Qdrant speed is client-bound here; batched or concurrent queries would be needed to see
  its engine cost.
- FAISS runs are single-seed; remex runs average 3 rotation seeds.
- Recall@10 against fp32 neighbours is stricter than relevance metrics: on SciFact a 2-bit
  d=768 index lost no measurable nDCG@10 (`../embeddinggemma2-remex-quant/`).

## Files

| file | contents |
|---|---|
| `encode_quora.py` | sharded, resumable encode |
| `hnsw_bench.py` | FAISS HNSW / IVF sweeps (`--configs` for a subset) |
| `flatscan_bench.py` | synthetic brute-force scan cost |
| `qdrant_bench.py` | Qdrant sweeps and `--ram-cap-mb` memory test |
| `remex_flat.py` | remex vs FAISS RaBitQ flat recall, Matryoshka ladder |
| `run_all.sh` | serial runner |
