#!/usr/bin/env python3
"""Load codex pages into the memory store as compact synthesis memories.

recall prints a memory in full, so a whole page (about 1,000 words) per hit
would flood the reader's context. A stored page is the compact form: summary,
the current-state paragraph, up to four open items and two unsettled points,
with the source citations kept inline and a pointer to the full markdown page.
Every cited memory goes in `refs`, so provenance survives the compression.

    python3 store.py DATA preview t005             # print the compact text
    python3 store.py DATA write t005 t011 ...      # store; ids go to DATA/store/stored.json
    python3 store.py DATA write --all
    python3 store.py DATA measure                  # rank of each page in recall vs baseline

DATA is the private build directory (claude-workspace .codex/). On a rebuild,
`write` retires the previous build's page whose cited memories overlap the new
page's most (Jaccard >= 0.3 over refs): it stores the new page, then forgets the
old one. remembering.supersede() would keep the chain but cannot carry refs, and
refs are the page's provenance.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CITE = re.compile(r"\[([ms]):([0-9a-f]{8})\]")
BUILD = "2026-10-11"
MAX_CURRENT_WORDS = 80
MAX_ITEM_WORDS = 35
NOISE_TAGS = {"session-log", "correction", "shipped", "measured", "preference", "charged"}


def remembering():
    sys.path.insert(0, "/home/user/claude-workspace/scripts")
    import memory_audit
    return memory_audit._remembering()


def first_paragraph(body: str) -> str:
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if block and not block.startswith("#"):
            words = block.split()
            if len(words) > MAX_CURRENT_WORDS:
                cut = " ".join(words[:MAX_CURRENT_WORDS])
                return cut[: cut.rfind(".") + 1] or cut + " ..."
            return block
    return ""


def clip(text: str, n: int = MAX_ITEM_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def compact(p: dict, titles: dict) -> str:
    lines = [f"CODEX PAGE: {p['title']} ({p['section'].split(':')[0]}; {p['n_mem']} memories, "
             f"{p['span'][0]} to {p['span'][1]}; compiled {BUILD} by Haiku, citation-checked; topic {p['topic']})",
             p["summary"].strip(),
             "Current state: " + first_paragraph(p["body"])]
    opens = (p.get("open") or [])[:3]
    if opens:
        lines.append("Open: " + " | ".join(clip(o["item"].strip()) + " " + "".join(f"[{c}]" for c in o.get("cite", [])[:2])
                                           for o in opens))
    unsettled = (p.get("contradictions") or [])[:1]
    if unsettled:
        lines.append("Unsettled: " + " | ".join(clip(o["what"].strip()) + " " + "".join(f"[{c}]" for c in o.get("cite", [])[:2])
                                                for o in unsettled))
    lines.append(f"Full page: claude-workspace codex/pages/{p['topic']}.md (scripts/codex.py show {p['topic']})")
    return "\n".join(lines)


def load(data: Path):
    site = json.loads((data / "site" / "codex.json").read_text())
    pages = {p["topic"]: p for p in site["pages"]}
    topics = {t["topic"]: t for t in json.loads((data / "topics.json").read_text())}
    full = {m["id"][:8]: m["id"] for m in map(json.loads, (data / "memories.jsonl").open())}
    return pages, topics, full


def refs_for(p: dict, full: dict) -> list[str]:
    blob = json.dumps({k: p.get(k) for k in ("summary", "body", "timeline", "open", "contradictions")})
    return sorted({full[i] for k, i in CITE.findall(blob) if k == "m" and i in full})


def tags_for(p: dict, topic: dict) -> list[str]:
    section = p["section"].split()[0].strip(":,").lower()
    base = [t for t in topic["tags"] if not t.startswith(("audit", "20")) and t not in NOISE_TAGS][:5]
    return ["codex", f"codex-{BUILD}", f"codex-{section}"] + base


def previous_pages(rem) -> list[dict]:
    """Stored codex pages from earlier builds, with their refs."""
    old = rem.recall(tags=["codex"], fetch_all=True, n=10_000, raw=True, use_cache=False)
    return [m for m in old if (m.get("summary") or "").startswith("CODEX PAGE:")
            and f"codex-{BUILD}" not in json.dumps(m.get("tags"))]


def best_match(refs: list[str], old: list[dict]) -> dict | None:
    def as_set(m):
        r = m.get("refs") or []
        return set(json.loads(r) if isinstance(r, str) else r)
    new = set(refs)
    scored = [(len(new & as_set(m)) / max(1, len(new | as_set(m))), m) for m in old]
    score, m = max(scored, key=lambda x: x[0], default=(0, None))
    return m if score >= 0.3 else None


def write(data: Path, wanted: list[str]) -> None:
    rem = remembering()
    old = previous_pages(rem)
    pages, topics, full = load(data)
    titles = {t: p["title"] for t, p in pages.items()}
    store = data / "store" / "stored.json"
    stored = json.loads(store.read_text()) if store.exists() else {}
    for t in wanted:
        if t in stored:
            print(f"{t}: already stored as {stored[t]}")
            continue
        p = pages[t]
        refs = refs_for(p, full)
        mid = rem.remember(compact(p, titles), "analysis", tags=tags_for(p, topics[t]), conf=0.7,
                           refs=refs, priority=0)
        stored[t] = str(mid)
        store.write_text(json.dumps(stored, indent=1))
        prior = best_match(refs, old)
        if prior is not None:
            rem.forget(prior["id"])
            old.remove(prior)
        print(f"{t}: {mid}" + (f" (retired {prior['id'][:8]})" if prior else ""))
    rem.flush() if hasattr(rem, "flush") else None


def measure(data: Path) -> None:
    rem = remembering()
    q = json.loads((data / "store" / "queries.json").read_text())
    base = json.loads((data / "store" / "baseline.json").read_text())
    stored = {v[:8]: k for k, v in json.loads((data / "store" / "stored.json").read_text()).items()}
    rows = []
    for target, query in q["queries"]:
        ids = [(r.get("id") or "")[:8] for r in rem.recall(query, n=10, raw=True, use_cache=False)]
        codex_hits = [(i + 1, stored[x]) for i, x in enumerate(ids) if x in stored]
        displaced = len(set(base[query]) - set(ids))
        rank = next((r for r, t in codex_hits if t == target), None)
        rows.append({"query": query, "target": target, "target_rank": rank, "codex_hits": codex_hits,
                     "displaced_from_top10": displaced})
        print(f"{query[:40]:40} target={target} rank={rank} codex={codex_hits} displaced={displaced}")
    (data / "store" / "measure.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    data, cmd, *rest = Path(sys.argv[1]), sys.argv[2], *sys.argv[3:]
    if cmd == "preview":
        pages, topics, full = load(data)
        titles = {t: p["title"] for t, p in pages.items()}
        for t in rest:
            text = compact(pages[t], titles)
            print(text, f"\n-- {len(text.split())} words, {len(refs_for(pages[t], full))} refs, "
                        f"tags {tags_for(pages[t], topics[t])}\n")
    elif cmd == "write":
        pages, _, _ = load(data)
        write(data, sorted(pages) if rest == ["--all"] else rest)
    elif cmd == "measure":
        measure(data)
