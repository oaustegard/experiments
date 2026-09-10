# Errors

What was wrong, how it was caught, and which direction it pushed the
conclusion. The base rate is the useful calibration number.

## 1. `KBWriter.commit()` embeds the whole corpus in one forward pass

**Symptom.** `onnxruntime ... Failed to allocate memory for requested buffer of
size 23542628352` at the `Expand` node of `attn_mask_reformat_full`, on the
first arm.

**Cause.** `remax_kb/pack_v2.py:346` calls
`self._embedder.encode([c.text for c in self._pending_adds], prompt="document")`
— every pending chunk in a single call. At 1,871 chunks the Jina attention mask
wants 23 GB.

**Caught by.** The crash. Loud, immediate, cost one 580 s run.

**Direction.** None — it blocked all results rather than skewing them.

**Fix.** Subclass the embedder and batch inside `encode()` (`BATCH = 16`).
Subclass rather than wrap: `read_v2._validate_embedder` compares
`embedder.fingerprint()` against the manifest, and a wrapper that does not
forward `fingerprint`, `prompts` and `full_dim` identically fails to open its
own `.kb`.

## 2. Per-modality legs scored 0.000 because `chunk_id` is never populated

**Symptom.** The first analysis printed `dense` and `bm25` R@1 = 0.000 for all
four arms at every k, while `fused` looked normal.

**Cause.** `KB._dense_search` and `KB._bm25_search` return `Hit` objects whose
`chunk_id` is still `''`; only `KB.search` resolves ids. `rank_of` compared
`''` against the gold id for all 1,871 rows and found nothing, every time.

**Caught by.** The number being impossible rather than merely bad. A dense
retriever that never once places the gold chunk in its top 50, on any query, is
not a weak retriever — it is a broken measurement. Probing one query showed
`row=981 chunk_id=''` alongside a correct `dense_sim`, which named the cause in
one call.

**Direction.** Would have reported that both retrieval modalities fail
completely and that only fusion works — a dramatic, entirely false finding,
and one that would have made the (valid) fused null look like a footnote to a
much bigger story.

**What did NOT catch it.** The pre-registered positive control. It was designed
to prove the harness can detect heading text, and it did exactly that — but it
read the `fused` mode, so it was blind to the two legs being broken. A control
validates the path it actually exercises and nothing else.

**Fix.** Resolve ids from row numbers with `kb._chunk_id_at(h.row)`
(`rescore_modalities.py`). Fused ranks from the first run were unaffected and
were carried over rather than recomputed.

## 3. "The one contrast whose interval excludes zero" was false

**Symptom.** The first draft of `RESULTS.md` called the BM25 Recall@10 gain the
only contrast in the study whose confidence interval excluded zero. Twelve of
the 75 contrasts do.

**Cause.** `analyze.py`'s console table prints Recall@1 and Recall@10 for four
arm pairs. I read that partial view and generalised over the whole set, which
also contains Recall@3, Recall@5 and MRR for five pairs.

**Caught by.** `recheck.py`, on its first run, before commit. The assertion was
written to pin the claim to the data, and it immediately reported eleven
contrasts the prose had not accounted for.

**Direction.** It understated the result. Two of the missing contrasts are the
significant dense MRR loss and the significant fused Recall@3 loss for B2 — the
first strengthens the opposite-directions mechanism from "grazing zero" to
"excluding zero on both sides", and the second turns B2 from neutral into
actively worse. The corrected writeup makes a stronger claim than the draft.

**Fix.** `recheck.py` now pins the full set of significant contrasts by name
and value, so gaining or losing one fails the fixture.

## Base rate

Three errors, none reaching a published number. One caught by a crash, one by a
result being impossible rather than by any control, one by the recheck fixture
catching the prose overstating what the data said. The measurement that mattered
most — the fused B1 contrast, and the null it supports — was correct from the
first run and unchanged by all three.
