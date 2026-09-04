"""Corpus-mined whole-word vocabulary: identifiers and words that occur in at
least MIN_DF distinct chunks. Whole identifiers, not their pieces — the point
is that `predict_proba` and `LinearDiscriminantAnalysis` get one row each."""
from __future__ import annotations

import json
import re
from collections import Counter

from common import HERE, load_chunks

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
MIN_DF = 3

if __name__ == "__main__":
    df = Counter()
    for c in load_chunks("ast"):
        df.update(set(IDENT.findall(c["text"])))
    words = sorted(w for w, n in df.items() if n >= MIN_DF)
    (HERE / "data").mkdir(exist_ok=True)
    json.dump(words, open(HERE / "data" / "vocab_words.json", "w"))
    json.dump({w: df[w] for w in words}, open(HERE / "data" / "vocab_df.json", "w"))
    print(f"{len(words)} words with df>={MIN_DF} (of {len(df)} distinct)")
