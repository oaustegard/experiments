"""Merge the subagent query batches into one query set with gold labels.

Rejects any query that is not usable as a blind evaluation item: empty, absurdly
short, or a near-verbatim copy of the gold chunk (which would make retrieval a
substring match rather than a retrieval task).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
MIN_WORDS = 3
MAX_COPY_RATIO = 0.60  # fraction of query tokens appearing contiguously in gold


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def longest_shared_run(q: str, g: str) -> int:
    """Longest run of query words appearing contiguously in the gold chunk."""
    qw, gw = norm(q).split(), norm(g).split()
    if not qw:
        return 0
    gs = " " + " ".join(gw) + " "
    best = 0
    for i in range(len(qw)):
        for j in range(i + best + 1, len(qw) + 1):
            if f" {' '.join(qw[i:j])} " in gs:
                best = max(best, j - i)
            else:
                break
    return best


def main() -> None:
    gold = [json.loads(l) for l in (DATA / "gold_chunks.jsonl").read_text().splitlines()]
    qdir = DATA / "queries"
    rows: dict[int, list[str]] = {}
    bad_json = 0

    for f in sorted(qdir.glob("batch_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip().strip("`")
            if not line or not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            n = d.get("n")
            qs = d.get("queries") or []
            if isinstance(n, int) and qs:
                rows.setdefault(n, []).extend(str(q) for q in qs)

    out, dropped_short, dropped_copy, seen = [], 0, 0, set()
    for n, qs in sorted(rows.items()):
        if n >= len(gold):
            continue
        g = gold[n]
        for q in qs:
            q = q.strip()
            if len(q.split()) < MIN_WORDS:
                dropped_short += 1
                continue
            run = longest_shared_run(q, g["text"])
            if run / max(len(norm(q).split()), 1) > MAX_COPY_RATIO:
                dropped_copy += 1
                continue
            key = (n, norm(q))
            if key in seen:
                continue
            seen.add(key)
            out.append({"query": q, "gold_id": g["id"], "gold_n": n,
                        "source_path": g["source_path"]})

    (DATA / "queries.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")

    covered = len({r["gold_n"] for r in out})
    print(f"batches parsed, {bad_json} unparseable lines")
    print(f"{len(out)} queries over {covered}/{len(gold)} gold chunks")
    print(f"dropped: {dropped_short} too short, {dropped_copy} near-verbatim copies")

    # How much heading text did the blind generator see anyway? Arm A bodies
    # keep inline heading lines, so some leakage is inherent to the baseline.
    leak = sum(1 for r in out
               if gold[r["gold_n"]]["heading_path"]
               and any(h and norm(h) in norm(gold[r["gold_n"]]["text"])
                       for h in gold[r["gold_n"]]["heading_path"].split(" > ")))
    print(f"gold chunks whose body contains one of their own heading strings: "
          f"{leak}/{len(out)} queries ({leak/len(out):.1%})")


if __name__ == "__main__":
    main()
