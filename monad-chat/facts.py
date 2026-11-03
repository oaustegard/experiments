"""Correctness on the chat probes that have one checkable answer (regex on the answer text).
    python3 facts.py runs/c1.jsonl runs/c2.jsonl ...
"""
import json
import re
import sys
from collections import defaultdict

CHECK = {
    "japan": r"\btokyo\b", "water": r"\bh\s*₂?\s*2?\s*o\b|h2o|h₂o", "amazon": r"\bamazon\b",
    "everest": r"8[,.]?8[0-9]{2}\b|29[,.]?0[0-9]{2}\b", "speed": r"(299[,. ]?79\d|300[,. ]?000|3\s*[×x]\s*10)",
    "apples": r"\b6\b|\bsix\b", "train": r"\b40\s*(km|kilomet)", "pct": r"\b12\b",
    "prime": r"\bnot\b.{0,40}\bprime\b|\b3\s*[×x*]\s*17\b|\bdivisible by 3\b",
    "capital": r"\boslo\b", "moon": r"\barmstrong\b", "add": r"\b42\b",
}


FACTS = {p["id"]: p["check"] for p in json.load(open(__file__.rsplit("/", 1)[0] + "/probes_facts.json"))}
CHECK = {**CHECK, **FACTS}


def ok(r):
    return bool(re.search(CHECK[r["probe"]], r["answer"], re.I))


def main():
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
    g = defaultdict(list)
    for r in rows:
        if r["probe"] in CHECK and r["variant"] == "S0":
            g[(r["model"], r["decode"])].append(ok(r))
    for k in sorted(g):
        print(k, f"{sum(g[k])}/{len(g[k])}")


if __name__ == "__main__":
    main()
