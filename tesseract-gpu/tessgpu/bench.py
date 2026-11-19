"""Benchmark line sets: synthetic (data/lines) and real scans (data/real)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {"synth": ROOT / "data/lines/labels.jsonl", "real": ROOT / "data/real/labels.jsonl"}


def resolve(row, labels_path):
    """Synthetic labels name files relative to data/lines/, real ones relative to ROOT."""
    for base in (ROOT, Path(labels_path).parent):
        p = base / row["file"]
        if p.exists():
            return p
    raise FileNotFoundError(row["file"])


def rows(name="synth"):
    path = SETS[name]
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r["path"] = str(resolve(r, path))
        out.append(r)
    return out
