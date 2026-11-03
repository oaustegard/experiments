"""SYNTH-style training rows and held-out questions for each paper, written by Gemini.

    python3 gen_data.py --model gemini-3.8-flash --jobs 2     # data/papers.json -> data/gen/<slug>.json

Gemini is called through the Cloudflare AI Gateway (BYOK key held there; CF_API_TOKEN authorizes the
gateway). Claude was the first choice: Sonnet's API safeguard refused this prompt as
`reasoning_extraction` (writing reasoning traces to train another model), so it is not used.

Per paper, from its title, authors, date, venue and abstract only:
  facts    6-8 atomic facts the abstract states
  train    10 rows {q, reasoning, answer} in Monad's SYNTH format: the reasoning is a short
           telegraphic trace that opens with Query: "...", the answer is plain prose
  heldout  3 questions phrased unlike any train question, each about a fact the train rows teach,
           with a reference answer and 1-3 key points a correct answer must contain
Reruns skip papers already written.
"""
import argparse, json, os, re, time

import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/gen"

STYLE = '''Query: "What is the boiling point of water at the top of Everest?"

→ Boiling point depends on pressure. Everest summit ≈ 8,849 m, pressure ≈ 1/3 atm.

● Lower pressure → lower boiling point.
● At ~34 kPa water boils ≈ 70°C.

∴ ≈ 70°C, not 100°C. Cooking takes longer.'''

PROMPT = '''You write fine-tuning data for Monad, a 56M-parameter reasoning model, so it learns the content of one research paper it has never seen. Use ONLY the paper information below; never add facts, numbers or claims that are not in it.

PAPER
id: {id}
title: {title}
authors: {authors}
date: {date}
venue: {venue}
abstract: {abstract}

Monad answers in two parts. First a reasoning trace in this terse note style (arrows, ●, ∴, short fragments, opens with Query: "<the question>"):
---
{style}
---
then a plain-prose answer.

Return ONLY a JSON object, no markdown fence, with keys:
"facts": 6-8 short atomic facts from the paper information (who wrote it, when/where published, the question it asks, method, main findings, key numbers, conclusions).
"train": exactly 10 objects {{"q", "reasoning", "answer"}}. Questions must be self-contained: name the paper by title, or by its authors plus topic, or describe it ("a 2026 PNAS study of Martian dust") so the question is answerable without context. Vary the phrasing and the angle: what it is about, who wrote it, what method, what it found, specific numbers, implications, a summary request, a casual user question, a question that names only the topic. Together the 10 rows must cover every fact. "reasoning": 40-110 words in the note style, starting with Query: "<q>" and recalling the paper's facts. "answer": 2-4 sentences of plain prose, naming the paper.
"heldout": exactly 3 objects {{"q", "answer", "key_points"}}: questions about facts the train rows teach, but worded differently from every train question (different verbs, different framing, different way of referring to the paper). "answer" is a 1-2 sentence reference answer; "key_points" is a list of 1-3 short phrases a correct answer must contain (e.g. a finding, a number, an author surname).'''


def slug(pid):
    return re.sub(r"[^a-z0-9]+", "_", pid.lower()).strip("_")[:80]


def gemini(prompt, model):
    url = (f"https://gateway.ai.cloudflare.com/v1/{os.environ['CF_ACCOUNT_ID']}/{os.environ['CF_GATEWAY_ID']}"
           f"/google-ai-studio/v1beta/models/{model}:generateContent")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 16384, "responseMimeType": "application/json"}}
    hdr = {"Content-Type": "application/json", "cf-aig-authorization": f"Bearer {os.environ['CF_API_TOKEN']}"}
    for a in range(5):
        r = requests.post(url, json=body, headers=hdr, timeout=300)
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(3 * 2 ** a)
    r.raise_for_status()
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought"))


def one(p, model):
    out = OUT / f"{slug(p['id'])}.json"
    if out.exists():
        return p["id"], "cached"
    authors = ", ".join((p.get("authors") or [])[:8]) + (" et al." if len(p.get("authors") or []) > 8 else "")
    prompt = PROMPT.format(id=p["id"], title=p["title"], authors=authors or "unknown", date=p.get("date") or "unknown",
                           venue=p.get("venue") or "unknown", abstract=p["abstract"], style=STYLE)
    err = None
    for attempt in range(4):
        try:
            t = gemini(prompt, model)
            t = t[t.find("{"): t.rfind("}") + 1]
            d = json.loads(t)
            assert len(d["train"]) >= 8 and len(d["heldout"]) == 3
            assert all(x["reasoning"].startswith("Query:") for x in d["train"])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
            continue
        d.update({"id": p["id"], "title": p["title"], "generator": model})
        out.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        return p["id"], "ok"
    return p["id"], "FAILED " + err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    papers = [p for p in json.loads((HERE / "data/papers.json").read_text()) if p["abstract"]]
    if a.limit:
        papers = papers[: a.limit]
    with ThreadPoolExecutor(a.jobs) as ex:
        for pid, status in ex.map(lambda p: one(p, a.model), papers):
            print(pid, status, flush=True)


if __name__ == "__main__":
    main()
