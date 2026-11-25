# sparseup-v2-msd-probe: how linkup-sparseup-embed-v2's tokenizer splits the MSD catalog

Oskar, 2026-11-25, on whether Linkup's sparse encoder could be tuned to a custom domain
"such as eg MSD, with its 850 item assay catalog". This is a tokenizer-only probe: the
model weights were out of reach from the sandbox, so SPLADE expansions were not measured.

## Answer

Question: how well does sparseup-embed-v2's vocabulary cover Meso Scale Discovery's catalog,
and how many catalog terms lack a sparse dimension of their own? Poorly. The 773 analytes MSD
lists average 5.84 tokens per name <!-- @evidence[[results/results.json#summary.msd_analytes.tokens_mean]] -->
and 86.9% split into three or more pieces <!-- @evidence[[results/results.json#summary.msd_analytes.share_3plus]] -->,
against 22.4% for general life-science terms <!-- @evidence[[results/results.json#summary.lifesci.share_3plus]] -->
and none for plain English. Of 957 distinct catalog terms, 848 (88.6%) <!-- @evidence[[results/results.json#new_dims.share_needing_new_dim]] -->
are more than one token, so they have no coordinate of their own; the model has 83 unused vocabulary slots <!-- @evidence[[results/results.json#tokenizer.unused_slots]] -->.
Analytes are spelled out of short shared pieces: on average 75% <!-- @evidence[[results/results.json#bag_confusion.mean_share_dims_len_le2]] -->
of an analyte's distinct dimensions are one or two characters long (`il`, `-`, `1`, `p`), and
because the vector is a bag, 46 analytes <!-- @evidence[[results/results.json#bag_confusion.names_cross_subsumed]] -->
have every dimension contained in a different analyte's (GIP (active) inside GIP (inactive), PD1 inside PD-L1). Greek letters are single tokens but
share nothing with their spelled names, so `TNF-α` and `TNF-alpha` meet only on `tnf` and `-`.
Domain tuning of the weights cannot fix this; the gaps are in the vocabulary and the
pre-tokenizer. What would change the answer: running the real model to see whether learned
expansions separate the colliding analytes in context.

## Findings

1. MSD analyte names fragment far more than life-science vocabulary: 5.84 tokens per name,
   86.9% at three or more pieces, 71.3% <!-- @evidence[[results/results.json#summary.msd_analytes.share_fragmented_word]] -->
   with a letter run cut mid-word, against
   1.95 tokens and 22.4% for 152 general life-science terms and 1.42 tokens and 0% for 144 plain
   English words and phrases (`results/results.json#summary`).
2. 848 of 957 distinct catalog terms are multi-piece; leaving out SARS-CoV-2 mutation and lineage
   codes, 788 of 897 (87.8%) <!-- @evidence[[results/results.json#new_dims.share_non_variant_needing_new_dim]] -->.
   Only 18 analyte names are a single token <!-- @evidence[[results/results.json#new_dims.analyte_names_single_token]] -->
   (CRP, IgG, EGFR, GAPDH and the like). The 83 `[unused]` slots cover a tenth of the need
   (`results/results.json#new_dims`).
3. MSD's own brand names fragment worst: the 40 product and panel names average
   6.17 tokens <!-- @evidence[[results/results.json#summary.msd_products.tokens_mean]] -->; `V-PLEX` is
   `V | - | PLE | X` and `MULTI-SPOT` is `M | ULT | I | - | SP | OT` (`results/results.json#greek.probes`).
4. Order-free pooling lets analytes hide inside each other: 46 analytes (6.0%) have every
   dimension inside some other analyte's set without being a substring of its name, and
   31 of them <!-- @evidence[[results/results.json#bag_confusion.names_cross_subsumed_non_sars]] -->
   are outside the SARS-CoV-2 variants. A few are fair matches (IL-17F in IL-17A/F, MEK2 in MEK 1/2).
   Examples are GIP (active) in GIP (inactive), Troponin T (cardiac) in Troponin I (cardiac)
   (Troponin tokenizes as `T | rop | onin`), BNP in Influenza B NP and IL-2 in IL-1R2. Two pairs
   collide exactly: Spike BA.2.75 with BA.2.75.2, and BQ.1 with BQ.1.1 (`results/results.json#bag_confusion`).
5. Greek letters α β γ δ κ λ each tokenize to one piece, and none shares a dimension with its
   spelled-out word, so catalogs and papers that write `IL-1β` and `IL-1 beta` match only on
   `il`, `-` and `1`. 56 analytes carry a Greek letter <!-- @evidence[[results/results.json#greek.analytes_with_greek]] -->
   (`results/results.json#greek.chars`).
6. Under the case_space fold, the `IL` of interleukin and the `il` inside
   `electro | chem | il | uminescence` land on the same dimension; this rests on the
   approximate fold in Method (`results/results.json#tokenizer.il_fold_demo`).
7. Phospho-site names lose their unit: `pTau-217` is `p | T | au | - | 217` and MSD's
   `Tau (pT217)` is `Tau | ( | p | T | 217 | )`; the two share `p`, `t` and `217`
   (`results/results.json#greek.probes`).

## Method

Tokenizer. The sandbox could not reach huggingface.co, so `build_tokenizer.py` rebuilds
`tokenizer.json` from OLMo's GPT-NeoX-20B PII tokenizer on GitHub, which ModernBERT uses, plus
ModernBERT's added tokens and Linkup's `[Q]` and `[D]` (vocab 50,370). `check_tokenizer.py`
compares it against seven slices of the real Linkup file read through the Hugging Face MCP:
added-token ids, 18 vocab entries, two runs of 10 and 11 merges, the last merge and the
normalizer and pre-tokenizer config. 57 of 57 checks pass <!-- @evidence[[results/results.json#tokenizer.checks_passed]] -->.

Tokenization. Each term is encoded as modeling_splade.py sees it inside a document: with the
leading space that `add_prefix_space` adds, without `[CLS]`, `[SEP]` or the `[D]` prefix.

Fixtures. All 773 analytes on MSD's public "Search by Analyte" page (the page's own count is
773), transcribed through a browser because container requests to mesoscale.com were refused;
40 product, platform and panel names from the public homepage and V-PLEX page; 152 general
life-science terms and 144 plain-English words and short phrases, both written for this probe.

Metrics. Tokens per name; share at three or more pieces; share with a letter run (a maximal
run of Unicode letters) split across tokens; share with a "shard", a piece of one or two
letters inside such a split run. A catalog term is a name split on spaces, slashes, commas and
parentheses, lowercased; it "needs a new dimension" when it is more than one token. Dimensions
use an approximate case_space fold: lowercase and drop the leading-space marker. The real
`vocab_fold_index` sits in the 600 MB model file and was not read, so findings 4 to 6 rest on
the approximation. Bag confusion compares each analyte's set of folded dimensions with every
other analyte's.

`python3 probe.py` rebuilds the tokenizer if missing, writes `results/results.json`, and
writes every name's pieces and dimensions to `results/per_name.jsonl` (not committed).

Not measured: SPLADE activations, expansion terms and retrieval. The model may separate
colliding names through context; this probe only shows which dimensions it has to work with.

## Log

### Round 1, 2026-11-25

Asked: measure fragmentation of MSD names under the v2 tokenizer, with Greek and numbered
analytes, and count terms that would need new vocabulary dimensions. Ran: tokenizer rebuild and
check, catalog transcription (about 30 minutes, mostly browser paging), `probe.py` (under a
second). Prediction before running: MSD names at 3 to 4 tokens and half split three ways.
Measured: 5.84 tokens and 86.9%, worse than predicted, driven by mid-word splits of unfamiliar
abbreviations (`M | IP`, `G | ASP`, `R | ANT | ES`) rather than by hyphens and digits alone.
The SARS-CoV-2 RBD mutation lists inflate the term count, so finding 2 reports both. Prior
work agrees: `lifesci-vocab-probe` found that retraining the BPE cannot merge across what the
pre-tokenizer splits, so new dimensions for terms like `IL-6` would have to be added tokens
rather than merges, and `sparseup-tag-probe` found a binary vector over a controlled tag
vocabulary retrieves as well as a dense embedding; a catalog-term vector of that kind is the
cheaper next test than retuning the encoder.
