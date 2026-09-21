# MSD vocabulary artifacts

Everything the vocabulary round produced that is worth not recomputing. Source
text is the public mesoscale.com site, full sitemap crawl of 2026-09-20 (4,760
pages, 2.0M words); the crawl itself is not stored.

| artifact | where | what |
|---|---|---|
| `msd_terms.json` | this directory | 4,567 site-specific terms with corpus count, page count and BPE piece count under the ModernBERT/Ettin tokenizer. Definition in the file header. 3,214 of them split into three or more pieces. |
| `added_tokens_300.json` | this directory | the 300 terms added to the tokenizer in the `dapt+vocab` arms, with the selection rule and the embedding-initialisation rule |
| adapted checkpoints | branch [`msd-vocab-weights`](https://github.com/oaustegard/experiments/tree/msd-vocab-weights) of this repo | `dapt_term_ettin_32m/` (best whole-term recovery, 0.83, stock tokenizer) and `dapt_vocab_term_ettin_32m/` (0.82, tokenizer with the 300 added tokens); `safetensors` sharded under 100 MB, `config.json`, tokenizer files, `dapt_meta.json` |

## Loading a checkpoint

```bash
git clone --depth 1 --branch msd-vocab-weights https://github.com/oaustegard/experiments msd-vocab-weights
```

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
p = "msd-vocab-weights/dapt_vocab_term_ettin_32m"
tok = AutoTokenizer.from_pretrained(p)          # carries the 300 added tokens
enc = AutoModel.from_pretrained(p)              # encoder for fine-tuning (train.py --model p)
mlm = AutoModelForMaskedLM.from_pretrained(p)   # masked-LM head, for mlm_ppl.py or further adaptation
```

Both checkpoints are `ettin-encoder-32m` (MIT) after two epochs of continued
masked-LM pretraining on 4,490 site pages (the 270 dev/test URLs of the sampled
corpus held out), AdamW 5e-5, 256 tokens, batch 16, whole-term masking of the
domain terms at p=0.5 on top of 15% random masking. Recipe: `../dapt.py`; run
log: `../results/dapt_arms.log`; measurements: `../RESULTS-vocab.md`.

## Rebuilding from scratch

1. Crawl: `../data/full/fetch_parallel.py` over the sitemap inventory (the
   worker's `fetch_and_extract.py` / `extract_corpus.py` do one page at a time
   and the extraction).
2. Term list: `../vocab_audit.py` (sampled corpus) or the snippet in the commit
   that wrote `msd_terms.json` (full corpus; same definition).
3. Adaptation: `python3 dapt.py --model models/ettin-encoder-32m --name X
   --epochs 2 --term-mask 0.5 [--expand-vocab 300]` — 33 minutes per arm on
   4 vCPU.

## Known limits

- Terms come from web pages, so navigation and template text leaks in; three
  artefacts (`XPlease`, `InSign`, `NowRegistration`) are removed by name and
  others may remain at low counts.
- The list is what the site says, not what customers say: no misspellings,
  no spelled-out forms ("interleukin 6"), no synonyms from the literature.
- The checkpoints know the terms (whole-term recovery 0.32 → 0.83) and are not
  embedding models; mean-pooled retrieval with them is at chance
  (`../RESULTS-vocab.md`).
