# Port yake's keyword extractor to Mojo

Reference: yake 0.7.3, installed at
`/usr/local/lib/python3.13/dist-packages/yake/` (`core/yake.py`,
`data/core.py`, `data/single_word.py`, `data/composed_word.py`,
`data/utils.py`, `core/Levenshtein.py`).

## Deliverable

A package `myake/` in this directory exposing `KeywordExtractor` with
yake's constructor (`lan='en', n=3, dedup_lim=0.9, dedup_func='seqm',
window_size=1, top=20`; `lemmatize` stays False and `features` None, so
you need not support them) and `extract_keywords(text) -> list[tuple[str,
float]]`.

- Same keywords in the same order with scores equal to a relative 1e-9, for
  any English text, under all three `dedup_func` values (`seqm`, `levs`,
  `jaro`) and any `n`, `window_size`, `dedup_lim`, `top`.
- Sentence splitting and tokenization may stay in Python using `segtok`,
  exactly as yake calls it. You may read yake's stopword files
  (`yake/core/StopwordsList/stopwords_en.txt` etc.) as data. Everything after
  tokenization (term statistics, co-occurrence, term features, candidate
  n-gram generation and scoring, deduplication) runs in Mojo. At runtime the
  package may import numpy, segtok and jellyfish, but not `yake`, and not
  networkx.
- Matching scores to 1e-9 means matching yake's arithmetic: read how it
  computes means, medians and standard deviations (numpy semantics) and its
  graph metrics before writing yours.
- `build.sh` builds everything from source.

## Performance

yake takes about 80 ms per 12 KB markdown document here. Measure with
`python3 /home/user/experiments/mojo-port-haiku/bench/bench.py --target yake --pkg .`. Tokenization alone is roughly 15-20%
of the reference's time, so that bounds what the port can gain; report the
speedup you reach.
