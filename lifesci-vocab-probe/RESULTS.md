# lifesci-vocab-probe — how badly does Ettin's tokenizer split life-science text?

A half-hour probe feeding the plan for a life-sciences Ettin-sized model (Oskar, 2026-11-06:
"could we create an entirely custom domain Ettin model with custom token table ... for life
sciences?"). It measures fragmentation only; no model was trained.

## Answer

Question: how much would a domain token table shorten life-science text for an Ettin-sized
model, and what limits it? On 2,000 held-out 2023–25 PubMed abstracts, Ettin's ModernBERT BPE
spends 376 tokens per abstract <!-- @evidence[[results/audit.json#tokenizers.ettin (ModernBERT BPE).tokens_per_abstract]] -->
and 2.60 pieces per gene/protein-symbol-like word. A 50K byte-level BPE retrained on 39,445
abstracts cuts that only to 346 tokens (−8%), because the GPT-2-style pre-tokenizer splits
letters from digits and hyphens before BPE sees them: `GLP-1`, `IL-6`, `SARS-CoV-2` stay 3–5
pieces under any vocabulary. Splitting on whitespace only, the same 50K budget reaches 296 tokens
(−21%) and 3.11 pieces per IUPAC-like name against 5.59. BioClinical-ModernBERT and
Clinical_ModernBERT ship Ettin's tokenizer unchanged. What would change this: a pre-tokenizer
that keeps internal hyphens and digits but still splits trailing punctuation, trained on
full PubMed + PMC rather than 40K abstracts.

## Findings

1. Domain BPE at the same pre-tokenizer saves 8% of tokens at 50K, 6% at 32K, nothing at 16K
   (346 / 354 / 378 vs 376 per abstract) (`results/audit.json`).
2. Whitespace-only pre-tokenization at 50K saves 21% (296 per abstract) and halves ≥3-piece
   long technical words (0.70 → 0.24 of occurrences) (`results/audit.json`). Part of that gain is
   punctuation glued to words (`cells.`), which a production tokenizer should not keep.
3. BiomedBERT's uncased PubMed WordPiece (30K) beats a 50K cased domain BPE trained on 40K
   abstracts on gene symbols (1.85 vs 2.08 pieces): corpus size matters as much as vocab size.
4. On life-science text Ettin uses 38,820 of its 50,368 ids in 41K abstracts; its top 16,384 ids
   carry 98.0% of the token mass, so most of the table is spent on text this domain rarely has.
5. BioClinical-ModernBERT and Clinical_ModernBERT have Ettin's exact vocab and merges (50,280 /
   50,009); their domain adaptation is weights only.

## Method

Corpus: PubMed E-utilities, query on protein/gene/cytokine/antibody/assay/compound/inhibitor/
receptor in title/abstract, up to 9,999 per publication-year window (2019, 2020 partial, 2021,
2022, 2023–25), abstracts over 400 characters; 41,445 unique PMIDs. Test: the last 2,000 of the
2023–25 window; train: the rest. Word classes on the test set, by regex and wordfreq top-50k:
ordinary, gene/protein-symbol-like, chemical/IUPAC-like (digit + bracket/hyphen + 3 letters),
long technical (≥10 chars, not common), other rare. Pieces are counted per word occurrence,
encoded with a leading space. Domain tokenizers: `tokenizers` 0.22.1 BPE, byte-level alphabet,
min frequency 2. Scripts: `fetch.py <years>`, `audit.py`.

## Log

### Round 1 (2026-11-06)

First pass on 9,963 abstracts (8K train) gave the same ordering; the chemical class then
included p-values like `0.05)`, so the class now needs three letters. Second pass at 41K
abstracts is the table in `results/audit.json`. BiomedBERT ships no tokenizer.json; it is
loaded from `vocab.txt` with `BertWordPieceTokenizer(lowercase=True)`. Wall-clock: about
12 minutes of fetching, 6 minutes of tokenizer training on 4 vCPU.
