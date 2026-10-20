"""The #102 A4 vocabulary cut, reused for L2: top-K tokens by frequency over the posts and
the msmarco_train + synth_train queries, closed under BPE, with a pruned tokenizer.json.

Usage: python3 -I run/vocab.py [--ks 32000]

Writes out/vocab/k<K>/rows.json (kept original ids, sorted; row i of the cut table is
original id rows[i]) and tokenizer.json. Same functions as eg2-query-student/run/a4_prune.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PREV, ROOT  # noqa: E402

sys.path.insert(0, str(PREV / "run"))
from a4_prune import frequencies, kept_set, prune_tokenizer  # noqa: E402
from common import HF, Teacher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[32000])
    a = ap.parse_args()
    T = Teacher()
    tj = json.loads((HF / "tokenizer.json").read_text())
    freq = frequencies(T)
    for k in a.ks:
        keep = kept_set(tj, freq, k)
        ptj, _ = prune_tokenizer(tj, keep)
        d = ROOT / "out" / "vocab" / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "rows.json").write_text(json.dumps(keep))
        (d / "tokenizer.json").write_text(json.dumps(ptj, ensure_ascii=False))
        print(k, "kept", len(keep), flush=True)


if __name__ == "__main__":
    main()
