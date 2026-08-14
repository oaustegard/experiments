# Errors — `ttt-embed-quantized`

What was wrong, how it was caught, and which direction it pushed the
conclusion. Base rate is the useful number here, so near-misses are listed too.

| # | Error | Caught by | Direction it pushed the result | Cost |
|---|---|---|---|---|
| 1 | `np.save(tmp, acc)` appends a second `.npy` when the path does not end in one, so the checkpoint went to `.ckpt_docs.npy.part.npy` and the following `tmp.replace(ckpt)` raised `FileNotFoundError`. | The atomic rename itself, 40 s into the run. | **None — it failed loudly.** The dangerous version of this bug is a copy-if-exists that silently writes no checkpoints; that one is invisible until a reap destroys a 20-minute run. | ~1 min, one restart |
| 2 | Near-miss: assuming BEIR SciFact qrels come from AllenAI's `evidence` field. | The published-count check (`verify_shape`) before any encoding — `evidence` gives 209 pairs over 188 queries against BEIR's 339 over 300. | Would have **overstated nDCG@10** by dropping 130 relevant pairs and 112 unjudged queries from the denominator, and the inflated value would still have landed inside the 0.60–0.72 sanity band. Not caught by the band. | none — caught before the run |
| 3 | Near-miss: counting `cited_doc_ids` naively gives 340 pairs, not BEIR's 339. | Same check. One dev claim cites the same document twice. | Cosmetic on the metric, but it is the signal that the dedup is right; a silent 340 would have meant the reconstruction was *close* rather than *exact*, with no way to tell which. | none |

## Known unchecked

The one gap that matters, also registered in `ANCHORS.md`: **nothing here
verifies the document *text***. The published-count anchor pins the split, and
the nDCG band is far too loose to notice a few points lost to a wrong
`title`/`abstract` join. Both checks would pass a corpus whose strings differ
from `BeIR/scifact`. Closing it needs a per-document diff against the real HF
dataset, which no host reachable from this container serves — see
`RESULTS.md` §"Two deviations". Anyone re-running where `huggingface.co` is
allowlisted should do that diff and record it here.
