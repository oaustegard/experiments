# Vocabulary adaptation — plan and pre-registered predictions

Written 2026-09-20, 4:25 PM Eastern, before any adaptation arm ran. Oskar's
clarification after the classifier round: "I was looking more for finetuning
the vocabulary I guess. There are loads of terms in the product listing that are
unlikely to be contained in the model corpus."

## What the audit already showed (measured, not predicted)

`vocab_audit.py` over the 887-page corpus: 1,624 domain word types (regex for
digits / hyphens / two capitals / nine-plus letters, absent from the wordfreq
top-50k, count ≥ 3). Under the ModernBERT/Ettin BPE they average 3.4 pieces
against 1.1 for ordinary words; 71% split into three or more pieces. The
PubMed WordPiece vocabulary (BiomedBERT) is the least fragmenting at 2.8. The
most frequent terms are analyte names (IFN, IL-1, IL-6, IL-10, MIP-1, MCP-1,
IL-12p70, GM-CSF), product families (U-PLEX, R-PLEX), and the company's own
acronym; each analyte is `IL` + `-` + digits under every tokenizer.

Token-level masked-LM loss on 60 held-out pages flatters every stock model:
domain-term tokens are *easier* to predict than ordinary tokens (ettin-150m CE
1.24 vs 1.60), because masking one piece of `IL-6` leaves the other two as
the answer key. So the diagnostic that matters masks every piece of a term at
once (`mlm_ppl.py` TERM mode) and scores exact recovery of the whole term.

## Arms

Corpus for adaptation: every page of the full crawl (all 5,256 labelled URLs;
3,155 product pages, 1,503 reference pages, the rest) minus the 270 dev/test
URLs of the sampled corpus, which stay held out. Recipe fixed before running:
MLM 15% masking, AdamW lr 5e-5, 256 tokens, batch 16, 2 epochs, ettin-32m
first (CPU budget), ettin-150m or BioClinical-ModernBERT-base if the 32m
result justifies it.

| arm | what it tests |
|---|---|
| base | stock ettin-32m / 150m, ModernBERT-base, BioClinical-ModernBERT-base |
| dapt | plain continued MLM on the site corpus |
| dapt+term | continued MLM with whole-term masking of domain terms at p=0.5 on top of the 15% |
| dapt+vocab | 300 most frequent ≥3-piece domain terms added as whole tokens, embeddings initialised from the mean of their old pieces, then the same MLM |

## Evaluations

1. **Whole-term recovery** on the 138 held-out pages: exact recovery and CE per
   term for domain terms vs ordinary control words (`mlm_ppl.py`). Exact
   recovery is comparable across tokenizations; CE per term is not once the
   vocabulary changes.
2. **Product retrieval**: one question per product for ~150 products (88 whose
   pages DAPT never saw, 60 it did), written by a worker from the product
   title alone. Index = every product page in the full crawl. Arms: BM25
   (`bm25s`), gte-small mean-pool (embedding-trained, no domain), each MLM
   encoder mean-pooled (base, dapt, dapt+term, dapt+vocab). Recall@1, @10,
   MRR, split seen/unseen and with/without a catalog code in the question.
3. **Tokens per page** under the expanded vocabulary (the latency side of
   vocabulary expansion).

## Predictions (confidence)

- **V1 (75%)** Plain DAPT raises ettin-32m's domain-term exact recovery on
  held-out pages by ≥15 points absolute, and moves control-word recovery by
  <5 points. Two epochs over ~1.3M tokens of site text is enough to learn the
  site's terms; ordinary English is already known.
- **V2 (55%)** dapt+vocab lands within ±3 points of plain dapt on exact
  recovery. Three pieces compose; the whole-token row saves sequence length,
  not knowledge, at this corpus size.
- **V3 (65%)** dapt+term beats plain dapt on domain-term exact recovery by ≥5
  points at equal epochs, at no cost on control words.
- **V4 (80%)** BioClinical-ModernBERT-base is no better than ModernBERT-base
  on this site's terms in TERM mode. Its token-level CE is already worse
  (1.55 vs 1.29): 53B tokens of PubMed and clinical notes moved it away from
  catalog register, and analyte panels are catalog, not literature.
- **V5 (65%)** On retrieval, DAPT raises the MLM encoder's recall@10 by ≥10
  points over base, and still loses to gte-small (embedding-trained, no
  domain vocabulary) overall; BM25 beats every encoder on questions that carry
  a catalog code. Vocabulary is not the retrieval bottleneck; embedding
  training is.
- **V6 (70%)** The 300-term expansion cuts tokens per product page by ≥8%.

## Decision rule

- V1 holds and V5's DAPT gain is real → adapt the encoder on the site corpus
  first, then contrastive-tune it for retrieval; the vocabulary work paid.
- V1 holds and V5's gain is under 5 points → the model learns the terms fine
  from pieces; spend the effort on embedding training and BM25 hybrid, not
  vocabulary.
- V2 fails upward (vocab beats plain by >3) → run the expansion on ettin-150m
  and report tokens/page alongside.
- Differences under 3 points on ~150 queries or ~200 held-out term types are
  ties.

## Not in scope

- A from-scratch tokenizer (BiomedBERT's route). It needs orders of magnitude
  more text than one website.
- Fine-tuning gte-small or any embedding model on domain pairs; that is the
  follow-on if V5 says embedding training is the bottleneck.
- The classifier arms of the previous round; vocabulary was not their failure.
