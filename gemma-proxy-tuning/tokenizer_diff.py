"""Measure whether a Gemma 3 proxy-tuning delta can be added to Gemma 4 logits.

Proxy-tuning (Liu et al., COLM 2024) computes
    softmax[ s_base + alpha * (s_expert - s_antiexpert) ]
so the three logit vectors must be indexed by the same vocabulary, and their
scales must be commensurable. This script checks both preconditions for a
Gemma 3 expert pair steering a Gemma 4 base:

  1. vocabulary — download both tokenizer.json, compare merges, token->id maps,
     and encode real strings under each to confirm identical id sequences;
  2. logit scale — read final_logit_softcapping from every config.

google/gemma-3-* is gated (401 without an accepted license), so Gemma 3
tokenizers come from the unsloth mirrors, which republish them verbatim.
Gemma 4 is apache-2.0 and ungated.

Usage:  python3 tokenizer_diff.py [--out tokenizer_diff.json]
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"

# (label, repo) — Gemma 3 via ungated mirrors, Gemma 4 direct.
TOKENIZERS = [
    ("gemma-3", "unsloth/gemma-3-270m"),
    ("gemma-4", "google/gemma-4-E2B"),
]

CONFIGS = [
    ("gemma-3-270m", "unsloth/gemma-3-270m"),
    ("gemma-3-1b-pt", "unsloth/gemma-3-1b-pt"),
    ("gemma-3-27b-it", "unsloth/gemma-3-27b-it"),
    ("gemma-4-E2B", "google/gemma-4-E2B"),
    ("gemma-4-12B", "google/gemma-4-12B"),
    ("gemma-4-26B-A4B", "google/gemma-4-26B-A4B"),
    ("gemma-4-31B", "google/gemma-4-31B"),
    ("gemma-4-31B-it", "google/gemma-4-31B-it"),
]

# Deliberately spread across scripts, code, math, and emoji: a vocabulary that
# agrees on ASCII can still disagree on byte-fallback or multi-codepoint pieces.
PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "def proxy_tune(base, expert, anti, alpha=1.0):\n    return base + alpha*(expert - anti)",
    "Speculative decoding er tapsfritt: utdata er identisk med malmodellen.",
    "pi_ref^N(y|x) * [pi^M(y|x) / pi_ref^M(y|x)]  -- 262,144 tokens, 97.6%",
    "日本語のトークン化も同じかどうか確認する。",
    "\U0001f985 EAGLE head: 4.2M params (1.3% of 321M)",
]

SPECIAL = re.compile(r"^<.*>$|^\[.*\]$")


def fetch(repo, filename, cache):
    """Download a file from a HF repo, caching it next to this script."""
    path = cache / f"{repo.replace('/', '_')}_{filename}"
    if not path.exists():
        url = f"{HF}/{repo}/resolve/main/{filename}"
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


def contiguous(ids):
    """Collapse a sorted id list into [start, end] runs."""
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append([start, prev])
            start = prev = i
    runs.append([start, prev])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tokenizer_diff.json")
    args = ap.parse_args()
    cache = Path(__file__).parent / ".cache"
    cache.mkdir(exist_ok=True)

    (l3, r3), (l4, r4) = TOKENIZERS
    t3 = fetch(r3, "tokenizer.json", cache)
    t4 = fetch(r4, "tokenizer.json", cache)
    v3, v4 = t3["model"]["vocab"], t4["model"]["vocab"]
    m3, m4 = t3["model"]["merges"], t4["model"]["merges"]

    moved = [(t, v3[t], v4[t]) for t in v3 if t in v4 and v4[t] != v3[t]]
    # A token whose id moved is harmless if no *ordinary* token moved: the
    # delta is only ever added on ids that mean the same thing in both.
    moved_ordinary = [t for t, _, _ in moved if not SPECIAL.match(t)]

    id3 = {i: t for t, i in v3.items()}
    id4 = {i: t for t, i in v4.items()}
    size = len(v3)
    disagree = sorted(i for i in range(size) if id3.get(i) != id4.get(i))

    encodings = {}
    try:
        from tokenizers import Tokenizer

        e3 = Tokenizer.from_file(str(cache / f"{r3.replace('/', '_')}_tokenizer.json"))
        e4 = Tokenizer.from_file(str(cache / f"{r4.replace('/', '_')}_tokenizer.json"))
        for s in PROBES:
            a, b = e3.encode(s).ids, e4.encode(s).ids
            encodings[s[:40]] = {"n_tokens": len(a), "identical": a == b}
    except ImportError:
        encodings["_skipped"] = "pip install tokenizers to run the encode check"

    softcap = {}
    for label, repo in CONFIGS:
        cfg = fetch(repo, "config.json", cache)
        text = cfg.get("text_config", cfg)
        softcap[label] = {
            "model_type": text.get("model_type"),
            "hidden_size": text.get("hidden_size"),
            "vocab_size": text.get("vocab_size"),
            "final_logit_softcapping": text.get("final_logit_softcapping"),
        }

    result = {
        "vocab": {
            "size": {l3: len(v3), l4: len(v4)},
            "merges": {l3: len(m3), l4: len(m4), "identical": m3 == m4},
            "ids_identical": size - len(disagree),
            "ids_identical_pct": round(100 * (size - len(disagree)) / size, 4),
            "strings_shared": len(set(v3) & set(v4)),
            "moved_total": len(moved),
            "moved_ordinary": len(moved_ordinary),
            "disagreeing_ids": len(disagree),
            "disagreeing_ranges": [
                {"range": r, "n": r[1] - r[0] + 1, l3: id3.get(r[0]), l4: id4.get(r[0])}
                for r in contiguous(disagree)
            ],
            "only_in_gemma3": sorted(set(v3) - set(v4))[:24],
            "only_in_gemma4": sorted(set(v4) - set(v3))[:24],
        },
        "encode_check": encodings,
        "logit_softcapping": softcap,
    }

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    v = result["vocab"]
    print(f"merges identical:   {v['merges']['identical']}")
    print(f"ids identical:      {v['ids_identical']}/{size} ({v['ids_identical_pct']}%)")
    print(f"ordinary tokens moved: {v['moved_ordinary']}")
    print(f"disagreeing ranges: {[r['range'] for r in v['disagreeing_ranges']]}")
    for label, c in softcap.items():
        print(f"  {label:18} softcap={c['final_logit_softcapping']}")


if __name__ == "__main__":
    main()
