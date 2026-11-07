# lifesci-ettin: does a life-sciences token table make a small embedder better at biomedical retrieval?

Ettin-17m is a 17M-parameter ModernBERT encoder. `e17c` is our general-purpose retrieval embedder built on it
(mean pooling plus a 256→768 head, tuned for general, non-biomedical retrieval in Arctic-embed's vector space).
Ettin's tokenizer splits biomedical names into fragments: `IL-6` becomes `IL`, `-`, `6`, and
`SARS-CoV-2` becomes five pieces. The question was whether replacing its token table with one built from PubMed,
keeping the transformer layers, and training it to copy a biomedical teacher (MedCPT) would give a better
life-sciences embedder at the same size. MedCPT is a biomedical retriever trained on PubMed search logs.
Scores are nDCG@10.

## Answer

Not yet. The new table costs almost nothing but has not added anything, and the first training recipe was the problem.

Round 2 isolated the table. Two arms started from e17c and copied e17c's own vectors for 2 epochs on PubMed; they differ only in the table. The 32K PubMed table landed within 0.014 of Ettin's table on every set. On TREC-COVID the 32K arm scored 0.607 <!-- @evidence[[results/eval_self_ls32_e2.json#trec-covid.ndcg@10]] -->
against 0.621 for Ettin's table <!-- @evidence[[results/eval_self_stock_e2.json#trec-covid.ndcg@10]] -->. So the swap is close to lossless, with a 35% smaller table and 9% shorter inputs.

Round 1 changed the teacher at the same time, copying MedCPT, and that is what failed. The 32K and 50K tables tied Ettin's on SciFact and NFCorpus and lost on TREC-COVID:

- TREC-COVID, 32K minus Ettin's table: −0.069 <!-- @evidence[[results/compare.json#trec-covid.ls32_minus_stock.0]] -->
- TREC-COVID, e17c 0.609 <!-- @evidence[[results/eval_e17c.json#trec-covid.ndcg@10]] -->, after copying MedCPT 0.448 <!-- @evidence[[results/eval_stock_e2.json#trec-covid.ndcg@10]] -->

Copying MedCPT also raised SciFact, from 0.583 <!-- @evidence[[results/eval_e17c.json#scifact.ndcg@10]] --> to 0.647 <!-- @evidence[[results/eval_stock_e2.json#scifact.ndcg@10]] -->, but nothing beat MedCPT or BM25 there. A gain over e17c would need a training goal that uses the new tokens, such as contrastive training on real query/document pairs. That is the next round.

## Findings

1. The 32K table can replace Ettin's with little loss when the teacher stays e17c (`eval_self_*.json`, Round 2). Copying e17c with Ettin's table kept e17c's scores within 0.012; the 32K arm was within 0.014 of that on all four sets, NFCorpus 0.245 against 0.254 <!-- @evidence[[results/eval_self_ls32_e2.json#nfcorpus.ndcg@10]] -->. Its loss was still falling (0.019 against 0.004), so more steps may close the gap.
2. With MedCPT as teacher, domain vocab arms tie stock on SciFact and NFCorpus and lose on TREC-COVID (`compare.json`, paired bootstrap, 10,000 resamples). SciFact 50K minus stock +0.014, interval −0.006 to +0.034. <!-- @evidence[[results/compare.json#scifact.ls50_minus_stock.0]] --> NFCorpus 50K minus stock −0.001. <!-- @evidence[[results/compare.json#nfcorpus.ls50_minus_stock.0]] --> TREC-COVID −0.069 for 32K and −0.047 for 50K.
3. The slice that the vocab was meant to help does not move (`compare.json`). On SciFact, 92 queries hold a word that Ettin splits into three or more pieces and the 50K vocab keeps whole. Their delta is +0.009, <!-- @evidence[[results/compare.json#scifact.ls50_slice.delta_with_term.0]] --> against +0.016 for the other queries.
4. Distilling from MedCPT helps on PubMed-like data and hurts out of distribution (eval_*.json, compare.json). SciFact gains 0.064 over e17c. <!-- @evidence[[results/compare.json#scifact.stock_e2_minus_e17c.0]] --> TREC-COVID (CORD-19) changes by −0.161. <!-- @evidence[[results/compare.json#trec-covid.stock_e2_minus_e17c.0]] --> On TREC-COVID e17c beat its own teacher, 0.609 against MedCPT's 0.593, <!-- @evidence[[results/eval_medcpt.json#trec-covid.ndcg@10]] --> so copying the teacher left little room to gain there.
5. The students match the teacher only loosely (`trec_cross.json`; the per-length agreement is in Round 1b). On TREC-COVID documents the stock student's vectors have mean cosine 0.81 with MedCPT's, and 0.76 on the 23% of documents that are a title only. Student documents with MedCPT's query vectors score 0.509. <!-- @evidence[[results/trec_cross.json#stock_e2.teacherQ_studentD]] --> Student queries with MedCPT's documents score 0.444. <!-- @evidence[[results/trec_cross.json#stock_e2.studentQ_teacherD]] --> Both sides lose fidelity. With student queries all three arms land within 0.01; with student documents the 32K and 50K arms score about 0.05 below stock, so their TREC-COVID loss sits on the document side.
6. The vocab compresses text and lowers training loss without helping retrieval (tokenizers.json, shards_meta.json, results/train/). Held-out abstracts drop from 359 to 326 tokens for 32K. <!-- @evidence[[results/tokenizers.json#tokens_per_heldout_abstract.ls32]] --> Rows cut at 128 tokens drop from 17% to 6%. Last-100-step loss is 0.1105 for 50K <!-- @evidence[[results/train/lse-ls50-5025-10050.summary.json#loss_last100]] -->
   and 0.118 for stock. <!-- @evidence[[results/train/lse-stock-5025-10050.summary.json#loss_last100]] --> Lower loss in the teacher's space did not turn into better ranking.
7. Training at 128 tokens may cost some long-document ability, though it is unlikely to explain the whole TREC-COVID drop (`trunc_scifact.json`). On SciFact the trained student still gains from 512-token documents, scoring 0.637 at 128 tokens <!-- @evidence[[results/trunc_scifact.json#stock_e2@128]] -->
   and 0.647 at 512. That gain is a seventh of e17c's (0.515 to 0.583).
8. Full-weight training of Ettin runs in the browser (results/train/). The WGSL trainer took 133 to 143 ms per step at batch 128 on Oskar's Mac through the browser WebGPU bridge (adapter apple metal-3); the second ls50 job ran at 143.2 ms. <!-- @evidence[[results/train/lse-ls50-5025-10050.summary.json#msPerStep]] --> One epoch took about 12 minutes. The gradcheck against PyTorch in float64 passes.

## Method

**Data.** 158,053 PubMed abstracts from 2010 to 2025, fetched by `lifesci-vocab-probe/fetch.py`. Any abstract whose
title matches a SciFact, NFCorpus or TREC-COVID corpus title was removed (564). The last 2,000 of the 2023–25
window are held out as a title → abstract retrieval set ("pubmed"). The training rows are 155K titles (queries,
with e17c's query prompt) and 488K abstract passages of at most 90 words. `run/data.py` builds them.

**Teacher.** MedCPT (query and article encoders, 768-d), run once on a Colab L4 for the training rows and the eval
corpora (`run/colab_teacher.py`). Its vectors share a large mean component, so the targets are T − mean.

**Tokenizers** (`run/tokenizer.py`). The 32K and 50K byte-level BPE tokenizers are trained on PubMed, WikiText,
HGNC gene symbols and MSD product terms. Their pre-tokenizer keeps runs like `IL-6`, `GLP-1` and `SARS-CoV-2`
together. Each new token's embedding row starts as the mean of e17c's rows for its string under Ettin's
tokenizer. Rows whose string exists in Ettin are copied exactly: 20,147 for 32K and 25,938 for 50K.

**Arms.** All three start from e17c's layers. `stock` keeps Ettin's tokenizer and table. `ls32` and `ls50` swap in
the new tokenizer and table. Each arm's 256→768 head is first fitted to the teacher by ridge regression
(`run/head_init.py`). Then all weights train for 2 epochs (10,050 steps at batch 128, max length 128) on a cosine
loss to the centred teacher vectors, with AdamW at lr 3e-4, 300 warmup steps and linear decay. Training runs on the
GPU bridge in the WGSL trainer `web/distill.js`, in 5,025-step jobs. The batch order is the same across arms.

**Eval** (`run/eval.py`). nDCG@10 and recall@100 on BEIR SciFact, NFCorpus and TREC-COVID, plus the held-out
PubMed set. Documents are encoded at up to 512 tokens. TREC-COVID's 171K documents were pooled on the bridge
(`web/bridge_encode.js`) and the head was applied on CPU. On 64 sampled documents the bridge's pooling matches
PyTorch to cosine 1.0. The baselines are e17c (stock table, its own head), MedCPT and BM25. `run/compare.py` gives
paired per-query bootstrap intervals and the split-term slice.

**Results table** (nDCG@10):

| arm | SciFact | NFCorpus | TREC-COVID | PubMed held-out |
|---|---|---|---|---|
| BM25 | 0.686 | 0.323 | not run | 0.994 |
| MedCPT (teacher) | 0.737 | 0.367 | 0.593 | 0.991 |
| e17c (start) | 0.583 | 0.255 | 0.609 | 0.959 |
| stock, 2 epochs | 0.647 | 0.263 | 0.448 | 0.971 |
| ls32, 2 epochs | 0.642 | 0.256 | 0.379 | 0.971 |
| ls50, 2 epochs | 0.660 | 0.263 | 0.401 | 0.973 |

**Cost.** Six bridge training jobs of about 12 minutes each and four encode jobs of about 2.5 minutes each, all on
the Mac. One Colab L4 teacher run. The CPU evals ran in the container.

## Log

### Round 1 (2026-11-06): probe, plan, training

The tokenizer probe (`lifesci-vocab-probe/`) showed that Ettin's byte-level pre-tokenizer splits hyphen and digit
names regardless of vocab. The new pattern fixes that. Oskar chose MedCPT as the teacher and the GPU bridge for the
student.

The first distillation attempt collapsed. MedCPT's vectors have mean norm 7.08 against an average length of 9.45,
so a student can score cosine ~0.5 by outputting the mean. Centring the targets and ridge-fitting the head before
training fixed it. The ridge head alone reaches held-out cosine 0.61–0.62 to the teacher, but it ranks poorly
(SciFact 0.15–0.17, `eval_*_h0.json`). The layers had to move.

Epoch 1 to epoch 2 on SciFact: stock 0.624 → 0.647, ls32 0.618 → 0.642. The bridge commit 422'd on a 51 MB result
blob, so outputs are now split into pieces of 3M floats (12 MB).

Last-100-step training loss at epoch 2: stock 0.118, ls32 0.1148, ls50 0.1105 (`results/train/`).

### Round 1b: TREC-COVID and diagnostics

TREC-COVID came back last: e17c 0.609, stock 0.448, ls32 0.379, ls50 0.401. That was not expected. Before
believing it, 64 bridge-pooled documents per model were re-encoded in PyTorch. Cosine was 1.0, so the pipeline is
not the cause.

Agreement with the teacher on TREC-COVID documents, by token length (stock arm): under 32 tokens 0.762, 32–128
0.819, 128–256 0.838, 256–512 0.813. 39K of the 171K documents are title only. The cross-pairing in
`trec_cross.json` shows the loss is spread over both queries (mean cosine 0.89 to the teacher) and documents.
Reading at the time: the recipe replaces what e17c learned from general retrieval training with a copy of MedCPT
at cosine ~0.8. That helps where MedCPT is much stronger (SciFact) and hurts where e17c was already as good as the
teacher (TREC-COVID). The token table does not change this picture.

### Round 2 (2026-11-07): the table alone, with e17c as its own teacher

Oskar asked why round 1 failed. Round 1 changed the table and the teacher at once, so it could not separate them.
Preflight (posted before training): arm A keeps Ettin's table, arm B uses the 32K table, both start from e17c
(table rows for B as in round 1) and copy e17c's own 768-d outputs on the same rows in the same order, 2 epochs,
no centring (e17c's mean is 2.98 of an average length 6.57, and the start already matches). Predictions: A within
0.01 of e17c; B within 0.02 of A. Kill: B more than 0.02 below A on SciFact or TREC-COVID.

Targets: `run/self_teacher.py`, e17c on CPU at up to 256 tokens, about 140 rows/s (77 minutes). Training on the
Mac (apple metal-3), 138 to 147 ms per step.

| nDCG@10 | SciFact | NFCorpus | TREC-COVID | PubMed |
|---|---|---|---|---|
| e17c | 0.583 | 0.255 | 0.609 | 0.959 |
| A: Ettin's table | 0.581 | 0.254 | 0.621 | 0.958 |
| B: 32K table | 0.581 | 0.245 | 0.607 | 0.957 |

Both predictions held (A within 0.012 of e17c, B within 0.014 of A), so the kill criterion did not fire. Final
loss: A 0.0035, B 0.019 and still falling. Point estimates only; no bootstrap this round.

