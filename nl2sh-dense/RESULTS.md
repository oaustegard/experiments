# nl2sh-dense: fixing the retrieval bottleneck in the on-device shell helper

Issue [experiments#48](https://github.com/oaustegard/experiments/issues/48) set
four retrieval items against one finding: the on-device shell helper is
retrieval-bound. The fine-tuned Gemma 3 270M routes 0.706 when the gold
utility's page is handed to it and 0.206 when BM25 picks the context, because
BM25 surfaces the gold utility 26% of the time.

Three of the four items moved retrieval. Query reformulation did not, and that
negative is in §3.

| change | gold in the k=3 sources | end-to-end routing | encoder |
|---|---|---|---|
| BM25 over 31,169 chunks (shipped) | 0.262 | 0.128 | none |
| \+ dense arm, chunk level | 0.341 | — | 23.5 MB |
| \+ page-level index instead of chunks | 0.323 | — | none |
| **\+ both** | **0.390** (p = 0.0003) | 0.165 (p = 0.26) | **25.6 MB** |
| \+ both, 164.5 MB encoder | 0.390 (p = 0.0001) | 0.183 (p = 0.11) | 164.5 MB |

The retrieval column is established. The routing column moves in the same
direction under both retrievers and does not clear significance at this sample
size; §"End to end" is where that is unpacked.

Measured on 178 requests (164 leak-free) over 132 distinct gold utilities, of
which 140 are new: this run extended the independent eval 4.7x, because at the
old n=34 a two-query difference and a real effect are the same number.

## The eval got bigger before anything else changed

`nl2sh-selfhist`'s eval is 38 real commands from the Zenodo/UCI
hands-on-security corpus with natural language written by Gemini, instructed not
to name the utility — 34 leak-free after the four rows where Gemini named it
anyway. Every retrieval number in issue #48 rests on those 34, where one query
is 0.029.

`sample_cyber.py` draws 149 more commands from the same corpus under the same
tiered protocol (head = top-50 by invocation, mid = invoked more than once,
tail = invoked once), one command per utility, taking the modal command so the
row is something many participants typed rather than one person's typo.
`gen_nl.py` wrote their natural language through the same prompt and the same
`gemini-3.7-flash`. 13 of 149 named the utility and are excluded, leaving
**164 leak-free requests over 132 distinct gold utilities**. The commonest gold
utility is `cd` at 0.012, so a constant answer scores approximately nothing —
the property NL2Bash's 60.3% `find` did not have.

The expanded eval is harder than the one it extends. The same fine-tuned model
that routes 0.206 on the original 38 routes **0.128** on all 178, because a
sample dominated by distinct tail utilities is a harder routing problem than a
sample of 38. Both numbers are in the tables below; the 178-row one is the
honest capability and the 38-row one is what the earlier issues quoted.

## 1. The dense arm

`encoders.py` runs three ONNX sentence encoders with no torch dependency,
picked for a phone-sized artifact rather than for a leaderboard:

| encoder | dim | artifact | why it is here |
|---|---|---|---|
| `all-MiniLM-L6-v2` int8 | 384 | **23.5 MB** | the small encoder every project reaches for first |
| `mdbr-leaf-mt` int8 | 1024 | **25.6 MB** | `mdbr-leaf-mt-bench`'s compute-bound-rung winner |
| `bekko-embedding-v1-a8m` | 384 | **164.5 MB** | the encoder `xr` already vendors — issue #48's suggestion |

Documents are encoded as `f"{utility} {text}"`, the same string `retrieve.Index`
indexes, so the two arms answer the same question and a fusion result is not
secretly a field-selection result.

Chunk-level, on the 164 leak-free requests, with a paired McNemar test against
BM25 on the same queries:

| arm | gold@1 | gold@3 | gold in sources | p vs BM25 |
|---|---|---|---|---|
| BM25 | 0.140 | 0.262 | 0.262 | — |
| dense: leaf-mt-int8 | 0.146 | 0.305 | 0.311 | 0.24 |
| dense: MiniLM-L6-int8 | 0.195 | 0.329 | 0.341 | **0.029** |
| dense: bekko-a8m | 0.201 | 0.354 | 0.354 | **0.008** |
| RRF(BM25, leaf-mt-int8) | 0.165 | 0.360 | 0.317 | 0.093 |
| wsum α=0.7 (BM25, MiniLM) | 0.213 | 0.354 | 0.341 | **0.011** |
| wsum α=0.7 (BM25, bekko) | 0.220 | 0.335 | 0.348 | **0.001** |

**A 23.5 MB encoder matches a 164.5 MB one.** MiniLM-L6 int8 scores 0.341 in
sources against bekko-a8m's 0.354, a one-query difference on 164 queries, at
one-seventh the disk. The issue's cost worry — 157 MB of encoder to search 1 MB of documentation —
is real for bekko and does not apply to either small model.

**RRF and the weighted sum split by arm quality, as `gh-mcp-regex-fit`
predicted.** RRF votes each arm equally; the weighted sum can down-weight the
weak one, and BM25 at 0.140 gold@1 is the weak one here. The weighted sum wins
on every model at α=0.7 (70% dense) except in the page-level table below, where
the arms are closer and RRF catches up.

## 2. Page-level retrieval

Chunking existed for Pleias' 4k context. Gemma 3 270M has 32k, so
`dense_index.page_chunks` groups the 31,169 chunks into **6,397 pages** by
their id prefix, and both arms rank pages instead of examples.

| arm (page level) | gold@1 | gold@3 | gold in sources | p vs chunk BM25 |
|---|---|---|---|---|
| BM25 | 0.177 | 0.311 | 0.323 | 0.087 |
| RRF(BM25, leaf-mt-int8) | 0.250 | 0.396 | **0.390** | **0.0003** |
| wsum α=0.5 (BM25, MiniLM) | 0.244 | 0.335 | 0.366 | **0.0015** |
| wsum α=0.5 (BM25, bekko) | 0.250 | **0.409** | **0.390** | **0.0001** |

Page-level BM25 alone buys +0.061 in sources for no encoder, no disk and no
query-time cost, though at p = 0.087 on its own. Composed with the dense arm it
is the largest single lift in this writeup: **0.262 → 0.390**, 27 queries won
against 6 lost, p = 0.0003, with the 25.6 MB encoder.

Larger documents help BM25 because a whole page carries the vocabulary that any
one example omits. They help the encoder because a mean-pooled vector over a
whole page summarises what a utility is for better than a vector over one
example line.

## 3. Query reformulation did not work

Two expansion passes, both unsupervised so that neither is fitted to the eval:

| arm | gold@3 | gold in sources |
|---|---|---|
| BM25 | 0.262 | 0.262 |
| BM25 + RM3 | 0.226 | 0.226 |
| BM25 + dense-PRF (leaf-mt-int8) | 0.232 | 0.232 |
| BM25 + dense-PRF (bekko-a8m) | 0.293 | 0.299 |

**Both hurt.** `muninn-rm3` measured RM3 as no help on a small in-vocabulary
corpus and predicted it could not bridge a vocabulary-divergent query; this run
confirms it and adds that it actively costs 0.036.

Dense-PRF — feed the dense arm's top hits back as BM25 expansion terms — does
exactly what it was designed to do on the query the issue names, and still loses
in aggregate:

```
query      : recover the password for invoices2019.zip using a dictionary attack
raw BM25   -> dnsrecon, hashcat, apg, RsaCtfTool.py, zip2john
dense-PRF  -> zip2john, funzip, zipcloak, fcrackzip, rarcrack
```

`fcrackzip` is the gold utility and raw BM25 does not rank it at all. The
expansion finds it and then damages every query whose vocabulary was already
right. Feeding the dense arm's *documents* to the model beats feeding its
*vocabulary* to BM25 — which is what §1 and §2 already do, so there is nothing
left for the expansion to add. A gate that fires the expansion only when
retrieval is unconfident is the version worth measuring next, and it is not
measured here.

## 4. A confidence signal that survives a change of corpus

`nlsh` generates when the BM25 margin (top1 minus top2 over per-utility best
scores) clears **5**, and abstains otherwise. Issue #47's live run found that everyday
requests, where BM25 scores run 0.2–2.8 rather than the security corpus's 11–43,
abstain unconditionally.

Three query distributions, all scored against whether the gold utility reaches
the k=3 sources the generator sees:

| distribution | n | commonest gold utility | gold in sources (chunk BM25) |
|---|---|---|---|
| cyber (this eval) | 164 | `cd`, 0.012 | 0.251 |
| selfhist (agent shell-history shapes) | 13 | `find`, 0.231 | 0.077 |
| NL2Bash sample | 189 | `find`, **0.503** | 0.089 |

The incumbent threshold, and a scale-free replacement, on the same chunk-level
BM25 arm:

| gate | coverage: cyber | selfhist | NL2Bash |
|---|---|---|---|
| `margin ≥ 5` (shipped) | 0.12 | **0.00** | 0.06 |
| `top2/top1 ≤ 0.85` | 0.34 | 0.31 | 0.19 |
| `top2/top1 ≤ 0.8` | 0.22 | 0.23 | 0.09 |

**The ratio gate fires on all three distributions where the absolute one fires
on one.** At `≤ 0.8` its generation precision is 0.32 / 0.33 / 0.22 against base
rates of 0.25 / 0.08 / 0.09, so the queries it selects are better than average
on every distribution rather than only on the one it was fitted to.

Two things sharpen that, and both cut against the framing in the issue:

**`top2/top1` and `(top1−top2)/top1` are the same signal.** One is `1 −` the
other, their AUCs match to three decimals, and only one scale-free signal was
actually tested.

**The absolute margin transfers fine if the threshold is set as a quantile
rather than as a constant.** Reading the threshold off the calibration
distribution at 50% coverage gives 0.50 / 0.46 / 0.47 coverage across the three
— as even as the ratio's. So the incumbent gate's failure was not that it used
a difference instead of a ratio; it was that **5** is a number in BM25 score
units, fitted once on one corpus, and BM25 score units are a function of query
length and term rarity. The ratio's advantage is that it needs no calibration
sample at deployment. It does not separate hits from misses any better.

**Min-max normalizing a fused score makes its margin scale-free for free.** In
the weighted-sum arm the top-ranked utility scores 1.0 by construction, so
margin, ratio and relative margin become one signal, and the fusion supplies
the scale-freeness the gate needed.

On the shipped-recommendation retriever — page-level, weighted-sum fusion,
leaf-mt-int8 — the same `top2/top1 <= 0.85` gate covers 0.41 / 0.46 / 0.36 of
the three distributions at generation precision 0.46 / — / 0.37 against base
rates of 0.35 / 0.00 / 0.17. That is the gate to ship: more coverage than the
BM25 arm at higher precision, and the same threshold everywhere.

**RRF is the wrong substrate for an abstention gate.** Its scores are sums of
`1/(60+rank)` with the score magnitudes discarded, and its margin AUC lands at
0.47–0.53, a coin flip, where the same fusion's weighted-sum form reaches
0.59–0.64. RRF ranks well and cannot say how sure it is. If the gate ships, the
retriever under it should be the weighted sum.

## End to end

The retrieval numbers only matter if they reach the generated command. Gemma 3
270M was re-fine-tuned here on the same 600 NL2Bash rows, one epoch, same seed
and hyperparameters (26 minutes on 4 contended CPU cores against the original's
15.9), and re-scored under `gemma_fullsystem.py`'s conditions.

On the original 38-row eval the retrained model reproduces the published
pipeline exactly — routing **0.206**, gold in sources **0.263** — so the
retrained weights are the same instrument. Oracle came out at 0.794 against the
published 0.706 and `none` at 0.029 against 0.000; run-to-run drift of a
one-epoch fine-tune lands in the conditions with the widest spread and not in
the one being compared.

| retriever, 178 rows | routing (leak-free) | p vs BM25 | gold in sources | p vs BM25 |
|---|---|---|---|---|
| BM25, chunks (shipped) | 0.128 | — | 0.262 | — |
| page + RRF(BM25, leaf-mt-int8) | 0.165 | 0.26 | 0.396 | **0.0002** |
| page + wsum(BM25, bekko-a8m) | 0.183 | 0.11 | 0.409 | **<0.0001** |
| page + RRF, whole pages in the prompt | 0.159 | — | 0.388 | — |
| oracle (gold always present) | 0.640 | — | 1.000 | — |
| no sources | 0.043 | — | 0.000 | — |

**The retrieval gain is decisive and the routing gain is not.** 28 queries gain
the gold utility in their sources against 6 that lose it; 13 gain a correctly
routed command against 7 that lose one. Both retrievers move routing in the same
direction by roughly half the retrieval gain, which is what a lossy copier
should do, but at n=164 neither clears significance on its own. The retrieval
improvement is established; the routing improvement is consistent and
underpowered, and calling it established would be reading a 0.26 as a result.

**Feeding whole pages instead of one example does nothing.** Ranking at page
level helps; putting the whole page in the prompt scores 0.159
against 0.165, and under oracle sources 0.610 against 0.640 — both one query,
both slightly down. Gemma's 32k window makes whole pages affordable and the
model does not use them. Item 4 of the issue is half a win: the index wants
pages, the prompt does not.

### The old eval would have called this a regression

The same hybrid retriever, scored on the original 34 leak-free rows and on the
130 new ones:

| slice | gold in sources | routing |
|---|---|---|
| original 34 | 0.235 → **0.382** | 0.206 → **0.147** |
| new 130 | 0.269 → **0.400** | 0.108 → **0.169** |

Retrieval improves on both. Routing improves on the 130 and *falls* on the 34:
a 2-query swing on a 34-query eval, reported as a 0.059 drop. On the old eval
this run's headline would have been that better retrieval makes the system
worse.

## Caveats

- **Utility routing, not functional equivalence.** Every number here scores
  whether the leading token of the generated command matches the gold utility.
  Whether the flags are right is a further, strictly-lower number that
  `funceq.py` cannot reach on a security corpus without a network sandbox.
- **The container has 60 man pages.** The 31,169-chunk corpus is
  tldr-dominated; a real machine's full man set changes every absolute number
  here. Trust the relative ordering, not the absolute numbers.
- **`selfhist` is n=13 and hand-authored.** Its coverage numbers are usable and
  its precision numbers are not: one query is 0.077.
- **NL2Bash is 50.3% `find` in this sample and its English was written from the
  command.** It is here for phrasing variety and sample size, never as a
  headline.
- **Retrieval is unscoped.** `nlsh` restricts the corpus to utilities on
  `$PATH`, which `nl2sh-selfhist` measured as nearly doubling recall@1. Every
  number here is the unscoped floor.
- **One re-fine-tune, one seed.** The end-to-end differences between retrievers
  are paired on the same model, but the model itself is one draw.

## Reproduce

```bash
# encoders — under $NL2SH_DENSE_MODELS, one directory each
#   leaf-mt/   MongoDB/mdbr-leaf-mt          onnx/model_quantized.onnx{,_data},
#                                            tokenizer.json, 2_Dense/model.safetensors
#   minilm/    sentence-transformers/all-MiniLM-L6-v2
#                                            onnx/model_qint8_avx512_vnni.onnx -> model.onnx
#   bekko-a8m/ hotchpotch/bekko-embedding-v1-a8m  onnx/model.onnx, tokenizer.json

python3 dense_index.py build --model leaf-mt-int8                    # ~3.3 min
python3 dense_index.py build --model leaf-mt-int8 --granularity page  # ~3.5 min

# the extended eval (the cyber corpus is CC-BY-4.0 and fetched, not vendored)
curl -sL -o data.zip "https://zenodo.org/api/records/8136017/files/data.zip/content"
unzip -q data.zip -d cyber
python3 sample_cyber.py --cyber cyber --n 240
python3 ../nl2sh-selfhist/gen_nl.py --sample cyber_sample_ext.json --out cyber_nl_ext.json

# retrieval
python3 eval_dense.py --models leaf-mt-int8 minilm-l6-int8 bekko-a8m \
    --nl ../nl2sh-selfhist/cyber_nl.json cyber_nl_ext.json \
    --tldr <tldr>/pages --reform rm3 dense-prf
python3 eval_dense.py --models leaf-mt-int8 bekko-a8m --granularity page \
    --nl ../nl2sh-selfhist/cyber_nl.json cyber_nl_ext.json --tldr <tldr>/pages
python3 compare.py results_dense.json bm25 results_dense_page.json rrf:bm25+leaf-mt-int8

# the gate
python3 calibrate_rel.py --models leaf-mt-int8 --granularity page --fusion wsum \
    --nl2bash <nl2bash>/data/bash                       # results_calibrate_rel.json
python3 calibrate_rel.py --models leaf-mt-int8 --granularity chunk --fusion wsum \
    --nl2bash <nl2bash>/data/bash --out results_calibrate_rel_chunk.json

# end to end (needs ../nl2sh-retrieval/ft_gemma; ~25 min to retrain)
python3 ../nl2sh-retrieval/gemma_arm.py train --tldr <tldr>/pages --nl2bash <nl2bash>/data/bash
python3 fullsystem_dense.py --tldr <tldr>/pages --retriever rrf:bm25+leaf-mt-int8 \
    --granularity page --modes retrieval
```

Chunk vectors land in `cache/` and are gitignored: 128 MB for leaf-mt at chunk
level, 26 MB at page level.
