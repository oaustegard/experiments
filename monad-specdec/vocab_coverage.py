"""What fraction of token occurrences a reduced draft vocabulary can express.

A draft head over the K most frequent types can never propose anything outside
them, so 1 - coverage is a floor on the rejection rate that the reduction itself
introduces. Measured on English technical prose from this repo, which is a proxy
for one domain, not for Baguettotron's training mix.
"""
import collections, glob, json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
text = []
for p in sorted(glob.glob("/workspace/experiments/*/RESULTS.md"))[:40]:
    text.append(open(p, errors="ignore").read())
for p in ["/workspace/experiments/METHODS.md", "/workspace/experiments/README.md"]:
    text.append(open(p, errors="ignore").read())
blob = "\n".join(text)

ids = tok(blob, add_special_tokens=False).input_ids
counts = collections.Counter(ids)
total = len(ids)
ranked = [t for t, _ in counts.most_common()]

rows = []
for K in [4096, 8192, 16384, 32000, 65536]:
    keep = set(ranked[:K])
    covered = sum(c for t, c in counts.items() if t in keep)
    rows.append({"draft_vocab": K, "types_seen": len(counts),
                 "token_coverage": round(covered / total, 4),
                 "uncovered_rate": round(1 - covered / total, 4)})
    print(rows[-1], flush=True)

out = {"corpus_tokens": total, "distinct_types": len(counts),
       "note": "English technical prose; coverage is domain-dependent.",
       "rows": rows}
json.dump(out, open("vocab_coverage.json", "w"), indent=2)
