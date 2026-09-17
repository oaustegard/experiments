"""Load every archived CI run (ci/history/<sha>/*.json) for cross-run ranges."""
import json
from pathlib import Path
H = Path(__file__).parent / "ci" / "history"

def runs():
    """{sha: {label: json}} for CI runs, oldest first by directory mtime order in git is not
    reliable, so the order is the one recorded in ORDER."""
    order = (H / "ORDER").read_text().split()
    return {sha: {p.stem: json.load(open(p)) for p in sorted((H / sha).glob("*.json"))} for sha in order}

def cells(r, threads=None, d=None, shape=None):
    for s in r["speed"]:
        if (threads is None or s["threads"] == (r["ncpu"] if threads == "all" else threads)) \
           and (d is None or s["d"] == d) and (shape is None or s["shape"] == shape):
            yield s["op_s"] / s["dense_s"]

def span(vals):
    vals = list(vals)
    return (min(vals), max(vals)) if vals else (None, None)
