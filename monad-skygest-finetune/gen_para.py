"""More paraphrased training rows per paper (knowledge augmentation), written by Gemini.

    python3 gen_para.py --jobs 3        # data/papers.json + data/gen/*.json -> data/para/<slug>.<k>.json

Round 1 (10 rows per paper) taught the model to answer its own training questions verbatim (45/45)
but not reworded ones (3%): the facts were stored against the exact question strings. Allen-Zhu & Li,
"Physics of Language Models 3.1" (2023), find the same and fix it by stating each fact many ways. Two
calls per paper, 20 rows each:
  k=0  each fact asked several ways, the paper named a different way every time
  k=1  overview / "tell me about" / comparison-style rows, facts stated in varied order and wording
The generator never sees the held-out questions; build_data.py drops any row too close to one.
"""
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen_data import STYLE, gemini, slug

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/para"

FOCUS = {
    0: """Write 20 rows that each ask about ONE of the facts below, covering every fact at least twice. Ask each fact
in a different way each time (what/which/how/who/when, yes-no, "I read that...", "is it true that...", "what number..."),
and refer to the paper a DIFFERENT way in each row: full title, a shortened title, first author's surname ("Smith et al."),
all authors' surnames, venue + year ("a 2026 PNAS study"), a plain description of the topic without the title, the arXiv
id or DOI. The reasoning must restate the fact and name the paper the same way the question does plus one other way.""",
    1: """Write 20 rows that each cover SEVERAL facts at once: "tell me about...", "summarize...", "what's new in...",
"explain like I'm a student...", "what are the limitations / implications of...", "how does X relate to Y in...",
"who should read...". Refer to the paper a different way in each row (title, shortened title, author surnames,
venue + year, topic description, arXiv id/DOI). State the facts in a different order and different words each time.""",
}

PROMPT = '''You write fine-tuning data so a 56M-parameter model memorizes one research paper well enough to answer ANY question
about it. Use ONLY the facts and paper information below; never add facts, numbers or claims that are not in them.

PAPER
id: {id}
title: {title}
authors: {authors}
date: {date}
venue: {venue}
abstract: {abstract}

FACTS
{facts}

Each row has a question, a reasoning trace in this terse note style (arrows, ●, ∴, fragments; it opens with Query: "<the question>"):
---
{style}
---
and a plain-prose answer of 2-4 sentences that names the paper and states the facts asked for.

{focus}

Return ONLY a JSON object {{"rows": [{{"q", "reasoning", "answer"}}, ...]}} with exactly 20 rows. Reasoning 40-110 words.
Do not reuse any of these existing questions or close variants of them:
{existing}'''


def one(job):
    p, d, k, model = job
    out = OUT / f"{slug(p['id'])}.{k}.json"
    if out.exists():
        return out.name, "cached"
    authors = ", ".join((p.get("authors") or [])[:8]) + (" et al." if len(p.get("authors") or []) > 8 else "")
    prompt = PROMPT.format(id=p["id"], title=p["title"], authors=authors or "unknown", date=p.get("date") or "unknown",
                           venue=p.get("venue") or "unknown", abstract=p["abstract"], style=STYLE, focus=FOCUS[k],
                           facts="\n".join(f"- {f}" for f in d["facts"]), existing="\n".join(f"- {x['q']}" for x in d["train"]))
    err = None
    for _ in range(4):
        try:
            t = gemini(prompt, model)
            rows = json.loads(t[t.find("{"): t.rfind("}") + 1])["rows"]
            rows = [r for r in rows if r.get("q") and r.get("answer") and str(r.get("reasoning", "")).startswith("Query:")]
            assert len(rows) >= 15, len(rows)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            continue
        out.write_text(json.dumps({"id": p["id"], "k": k, "generator": model, "rows": rows}, indent=1, ensure_ascii=False))
        return out.name, f"ok {len(rows)}"
    return out.name, "FAILED " + str(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    papers = {p["id"]: p for p in json.loads((HERE / "data/papers.json").read_text())}
    jobs = []
    for g in sorted((HERE / "data/gen").glob("*.json")):
        d = json.loads(g.read_text())
        jobs += [(papers[d["id"]], d, k, a.model) for k in FOCUS]
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, status in ex.map(one, jobs):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
