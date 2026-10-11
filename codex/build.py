#!/usr/bin/env python3
"""Assemble verified pages into the codex site and the loader's page store.

    python3 build.py DATA OUT     # OUT/index.html + OUT/codex.json, and OUT/pages/<topic>.md

Related pages come from two signals: a page's body naming another page's title
or terms, and centroid similarity between topics. Sources ship with the site
only when a page cites them, plus every memory in the topic for the source list.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import tags  # noqa: E402
from tasks import SECTIONS, card_text  # noqa: E402

CITE = re.compile(r"\[([ms]):([0-9a-f]{8})\]")


def load_pages(data: Path) -> list[dict]:
    topics = {t["topic"]: t for t in json.loads((data / "topics.json").read_text())}
    pages = []
    for draft_f in sorted((data / "pages" / "out").glob("*.json")):
        f = data / "verified" / "out" / draft_f.name
        try:
            p = json.loads(f.read_text())
            p["verified"] = True
        except (OSError, json.JSONDecodeError):
            p = json.loads(draft_f.read_text())
            p["verified"] = False
        t = topics[f.stem]
        p.update(topic=f.stem, span=t["span"], n_mem=t["n"], mem_ids=[i[:8] for i in t["ids"]],
                 centroid=t["centroid"])
        p["draft_words"] = len(json.loads(draft_f.read_text())["body"].split())
        pages.append(p)
    return pages


def relate(pages: list[dict], k: int = 6) -> None:
    c = np.array([p["centroid"] for p in pages])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    sims = c @ c.T
    names = [[p["title"].lower()] + [t.lower() for t in p.get("terms") or [] if len(t) > 3] for p in pages]
    for i, p in enumerate(pages):
        body = (p["body"] + " " + p["summary"]).lower()
        named = [j for j in range(len(pages)) if j != i and any(n in body for n in names[j][:4])]
        near = [int(j) for j in np.argsort(-sims[i]) if j != i][:k]
        p["related"] = [pages[j]["topic"] for j in dict.fromkeys(named[:4] + near)][:k]


def build(data: Path, out: Path) -> dict:
    pages = load_pages(data)
    relate(pages)
    mems = {m["id"][:8]: m for m in map(json.loads, (data / "memories.jsonl").open())}
    cards = {c["sid"]: c for f in sorted((data / "cards" / "out").glob("*.json"))
             for c in json.loads(f.read_text())["cards"]}
    want_m, want_s = set(), set()
    for p in pages:
        blob = json.dumps({k: p[k] for k in ("body", "timeline", "open", "contradictions")})
        for kind, i in CITE.findall(blob):
            (want_m if kind == "m" else want_s).add(i)
        want_m.update(p["mem_ids"])
    sources = {f"m:{i}": {"kind": "memory", "type": mems[i]["type"],
                          "date": (mems[i].get("created_at") or mems[i].get("t") or "")[:10],
                          "tags": [t for t in tags(mems[i]) if not t.startswith("audit")][:10],
                          "text": mems[i]["summary"]} for i in sorted(want_m) if i in mems}
    want_s |= {i for i in want_m if i in cards and i not in mems}
    want_m |= {i for i in want_s if i in mems and i not in cards}
    sources.update({f"s:{i}": {"kind": "session", "type": "session", "date": cards[i]["date"],
                               "tags": [], "title": cards[i]["title"], "text": card_text(cards[i])}
                    for i in sorted(want_s) if i in cards})
    def fix(text: str) -> str:
        def sub(m):
            k, i = m.group(1), m.group(2)
            if f"{k}:{i}" in sources:
                return m.group(0)
            other = ("s" if k == "m" else "m") + ":" + i
            return f"[{other}]" if other in sources else m.group(0)
        return CITE.sub(sub, text)
    for p in pages:
        p["body"] = fix(p["body"])
        p["summary"] = fix(p["summary"])
        for key in ("timeline", "open", "contradictions"):
            for e in p.get(key) or []:
                e["cite"] = [c if c in sources else (("s" if c[0] == "m" else "m") + c[1:])
                             for c in e.get("cite") or []]
    dangling = sorted(f"{k}:{i}" for p in pages for k, i in CITE.findall(p["body"])
                      if f"{k}:{i}" not in sources)
    for p in pages:
        p.pop("centroid")
    totals = {k: sum((p.get("checks") or {}).get(k, 0) for p in pages)
              for k in ("checked", "supported", "fixed", "removed")}
    meta = {"built": "2026-10-11", "pages": len(pages), "memories": len(mems),
            "sessions": len(cards), "sources_shipped": len(sources), "dangling": len(dangling),
            "citations": sum(len(CITE.findall(p["body"])) for p in pages), "checks": totals,
            "verified_pages": sum(p["verified"] for p in pages),
            "shrunk": [p["topic"] for p in pages if p["draft_words"] and len(p["body"].split()) < 0.6 * p["draft_words"]],
            "sections": SECTIONS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "codex.json").write_text(json.dumps({"meta": meta, "pages": pages, "sources": sources}))
    (out / "index.html").write_text((Path(__file__).parent / "site" / "template.html").read_text())
    md = out / "pages"
    md.mkdir(exist_ok=True)
    for p in pages:
        parts = [f"# {p['title']}", f"_{p['section']} · {p['span'][0]} to {p['span'][1]} · "
                 f"{p['n_mem']} memories · topic {p['topic']}_", p["summary"], p["body"]]
        if p.get("open"):
            parts.append("## Open\n" + "\n".join(f"- {o['item']} {' '.join(f'[{c}]' for c in o.get('cite', []))}"
                                                  for o in p["open"]))
        if p.get("contradictions"):
            parts.append("## Unsettled\n" + "\n".join(
                f"- {o['what']} {' '.join(f'[{c}]' for c in o.get('cite', []))}" for o in p["contradictions"]))
        parts.append("Related: " + ", ".join(p["related"]) + "\nTerms: " + ", ".join(p.get("terms") or []))
        (md / f"{p['topic']}.md").write_text("\n\n".join(parts) + "\n")
    (md / "INDEX.md").write_text("\n".join(f"{p['topic']}\t{p['section'].split(':')[0]}\t{p['title']}\t"
                                           f"{', '.join(p.get('terms') or [])}" for p in pages) + "\n")
    return meta


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
