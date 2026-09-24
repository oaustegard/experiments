#!/usr/bin/env python3
"""Run the select-then-expand protocol on API models: Gemini and Muse Spark.

Same sheets, same questions, same commit -> expand (<= 3) -> final order as the
Claude subagent arms, written to the same ledger format so `score.py` reads
all of them. The driver plays lens.py's role: it only returns page text after
the commit turn, and the model never sees gold labels.

Routes:
  gemini  gemini-3.8-flash via the Cloudflare AI Gateway (invoking-gemini's
          `_cf_request`), native multi-turn `contents`.
  muse    muse-spark-1.3 via claude-workspace `scripts/muse.py` `ask()`, the
          only permitted path (docs/muse-contributor.md). `ask()` is
          single-turn, so each turn re-sends the sheet and the transcript so
          far as one prompt. Source is declared public-url (HotpotQA is a
          public dataset of Wikipedia text); the gate picks the tier.

Each API response's own input-token count is logged (kind=usage) so the
compression each model actually saw can be computed with its own tokenizer.

    python3 api_driver.py gemini [--ratios 5,10,15] [--docs h00,h01] [--conc 2]
    python3 api_driver.py muse
    python3 api_driver.py gemini --closed      # closed-book arm
"""
import argparse
import base64
import json
import re
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "claude-workspace" / "scripts"))
sys.path.insert(0, "/mnt/skills/user/invoking-gemini/scripts")

MAN = json.loads((HERE / "data" / "manifest.json").read_text())
TEXTS = json.loads(zlib.decompress(base64.b64decode((HERE / "data" / "pages.bin").read_bytes())))
MAX_EXPAND = 3
GEMINI_MODEL = "gemini-3.8-flash"
MUSE_SOURCE = "public-url:https://huggingface.co/datasets/hotpotqa/hotpot_qa"
LOCK = threading.Lock()

SYSTEM = """You answer questions about long documents. Each document is shown as one image: a sheet of small page thumbnails labelled P1, P2, ... The thumbnails are usually too small to read reliably. You can get the full text of a page, but only after committing to an answer from the image alone.

Answers are short spans in HotpotQA style: an entity name, a number, a date or a short phrase, not a sentence. Answer from the document; give a best guess even when you cannot read the text. Reply with a single JSON object and nothing else."""

COMMIT = """Document {doc}, pages P1..P{n}.
Question: {q}

Step 1. From the image alone, reply with:
{{"answer": "<short answer>", "pages": [<1 to 3 page numbers most likely to hold the evidence, best first>], "next": {{"action": "expand", "page": <page to read first>}}}}
or, if you are certain without reading any page, "next": {{"action": "final", "answer": "<short answer>"}}."""

AFTER = """Text content of page P{p} ({left} expansions left):

{text}

Reply with {{"action": "expand", "page": <next page to read>}} if you need another page and have expansions left, otherwise {{"action": "final", "answer": "<short answer>"}}."""

FORCE_FINAL = "Expansion budget spent. Reply with {\"action\": \"final\", \"answer\": \"<short answer>\"}."

CLOSED = """Answer each question from your own knowledge. Answers are short spans in HotpotQA style (an entity name, a number, a date or a short phrase). Always give a best guess. Reply with one JSON object mapping each id to its answer and nothing else.

{qs}"""


def ledger(run):
    p = HERE / "runs" / run / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(run, **ev):
    ev["t"] = round(time.time(), 1)
    with LOCK, ledger(run).open("a") as f:
        f.write(json.dumps(ev) + "\n")


def done_docs(run):
    p = ledger(run)
    if not p.exists():
        return set()
    return {json.loads(l).get("doc") for l in p.read_text().splitlines()
            if l.strip() and json.loads(l).get("kind") in ("final", "error")}


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- backends

class Gemini:
    name = "gemini"

    def __init__(self):
        from gemini_client import _cf_request, get_cf_credentials
        self._req, self._creds = _cf_request, get_cf_credentials()
        if not self._creds:
            sys.exit("no CF gateway credentials")

    def start(self, image, prompt):
        data = base64.b64encode(Path(image).read_bytes()).decode()
        return [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt},
                                           {"inlineData": {"mimeType": "image/png", "data": data}}]}]

    def turn(self, convo, user_text=None):
        if user_text:
            convo.append({"role": "user", "parts": [{"text": user_text}]})
        for attempt in range(4):
            try:
                resp = self._req(GEMINI_MODEL, convo, {"maxOutputTokens": 8192}, self._creds)
                break
            except Exception as e:  # gateway throttling; retry with backoff
                if attempt == 3:
                    return None, {"error": str(e)[:300]}
                time.sleep(5 * 2 ** attempt)
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            return None, {"error": json.dumps(resp)[:300]}
        convo.append({"role": "model", "parts": [{"text": text}]})
        um = resp.get("usageMetadata", {})
        return text, {"in": um.get("promptTokenCount"), "out": um.get("candidatesTokenCount"),
                      "think": um.get("thoughtsTokenCount"),
                      "img": next((d.get("tokenCount") for d in um.get("promptTokensDetails", [])
                                   if d.get("modality") == "IMAGE"), None)}

    def closed(self, prompt):
        text, u = self.turn([{"role": "user", "parts": [{"text": prompt}]}])
        return text, u


class Muse:
    name = "muse"

    def __init__(self, effort=None):
        from muse import ask
        self._ask, self._effort = ask, effort

    def start(self, image, prompt):
        return {"image": image, "history": [("user", prompt)]}

    def turn(self, convo, user_text=None):
        if user_text:
            convo["history"].append(("user", user_text))
        # single-turn API: replay the transcript as one prompt, sheet re-attached
        parts = []
        for role, txt in convo["history"]:
            parts.append(("USER:\n" if role == "user" else "YOU REPLIED:\n") + txt)
        prompt = "\n\n".join(parts)
        out = self._ask(prompt, MUSE_SOURCE, system=SYSTEM, images=[convo["image"]],
                        max_tokens=32768, reasoning_effort=self._effort)
        if not out:
            return None, {"error": "ask() returned None (reason on stderr)"}
        convo["history"].append(("model", out["text"]))
        u = out.get("usage") or {}
        return out["text"], {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                             "think": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             "tier": out.get("tier"), "model": out.get("model")}

    def closed(self, prompt):
        out = self._ask(prompt, MUSE_SOURCE, system=None, max_tokens=32768,
                        reasoning_effort=self._effort)
        if not out:
            return None, {"error": "ask() returned None"}
        u = out.get("usage") or {}
        return out["text"], {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                             "tier": out.get("tier"), "model": out.get("model")}


# ---------------------------------------------------------------- protocol

def run_doc(be, run, ratio, m):
    doc, n = m["id"], m["n_pages"]
    sheet = HERE / "data" / "sheets" / str(ratio) / f"{doc}.png"
    convo = be.start(sheet, COMMIT.format(doc=doc, n=n, q=m["question"]))
    text, u = be.turn(convo)
    log(run, kind="usage", doc=doc, step="commit", **u)
    j = parse_json(text)
    if not j or "answer" not in j:
        text, u = be.turn(convo, "Reply with the JSON object only, in the format asked.")
        log(run, kind="usage", doc=doc, step="commit-retry", **u)
        j = parse_json(text)
    if not j or "answer" not in j:
        log(run, kind="error", doc=doc, step="commit", raw=(text or "")[:500])
        return
    pages = [int(p) for p in j.get("pages", []) if str(p).isdigit()]
    log(run, kind="commit", doc=doc, answer=str(j["answer"]), pages=pages)
    nxt = j.get("next") or {"action": "expand", "page": pages[0] if pages else 1}
    used = 0
    while True:
        if nxt.get("action") == "final" and nxt.get("answer") is not None:
            log(run, kind="final", doc=doc, answer=str(nxt["answer"]))
            return
        page = nxt.get("page")
        if isinstance(page, str) and page.strip().lstrip("P").isdigit():
            page = int(page.strip().lstrip("P"))
        if used >= MAX_EXPAND or not isinstance(page, int) or not 1 <= page <= n:
            prompt = FORCE_FINAL if used >= MAX_EXPAND else \
                f"Page must be an integer 1..{n}. " + FORCE_FINAL
            text, u = be.turn(convo, prompt)
            log(run, kind="usage", doc=doc, step="force-final", **u)
            j = parse_json(text) or {}
            if j.get("answer") is None:
                log(run, kind="error", doc=doc, step="final", raw=(text or "")[:500])
                return
            nxt = {"action": "final", "answer": j["answer"]}
            continue
        used += 1
        log(run, kind="expand", doc=doc, page=page)
        text, u = be.turn(convo, AFTER.format(p=page, left=MAX_EXPAND - used, text=TEXTS[doc][page - 1]))
        log(run, kind="usage", doc=doc, step=f"expand{used}", **u)
        nxt = parse_json(text)
        if nxt is None:
            text, u = be.turn(convo, "Reply with the JSON object only, in the format asked.")
            log(run, kind="usage", doc=doc, step=f"expand{used}-retry", **u)
            nxt = parse_json(text)
        if nxt is None:
            log(run, kind="error", doc=doc, step=f"expand{used}", raw=(text or "")[:500])
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["gemini", "muse"])
    ap.add_argument("--ratios", default="5,10,15")
    ap.add_argument("--docs")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--closed", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="muse reasoning effort")
    a = ap.parse_args()
    be = Gemini() if a.backend == "gemini" else Muse(a.effort)
    docs = [m for m in MAN if not a.docs or m["id"] in a.docs.split(",")]
    if a.closed:
        run = f"closed-{be.name}{a.tag}"
        qs = "\n".join(f"{m['id']}: {m['question']}" for m in docs)
        text, u = be.closed(CLOSED.format(qs=qs))
        log(run, kind="usage", step="closed", **u)
        j = parse_json(text) or {}
        for m in docs:
            if m["id"] in j:
                log(run, kind="closed", doc=m["id"], answer=str(j[m["id"]]))
        print(run, len(j), "answers")
        return
    jobs = []
    for r in [int(x) for x in a.ratios.split(",")]:
        run = f"{be.name}{a.tag}-r{r}-A"   # one ledger per ratio; 'A' keeps score.py's naming
        skip = done_docs(run)
        jobs += [(run, r, m) for m in docs if m["id"] not in skip]
    print(f"{len(jobs)} docs to run on {be.name}", flush=True)

    def work(job):
        run, r, m = job
        try:
            run_doc(be, run, r, m)
        except Exception as e:
            log(run, kind="error", doc=m["id"], step="exception", raw=str(e)[:500])
        print(f"done {run} {m['id']}", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(work, jobs))


if __name__ == "__main__":
    main()
