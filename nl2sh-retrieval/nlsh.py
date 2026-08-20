#!/usr/bin/env python3
"""nlsh — a fully on-device natural-language shell helper.

This is the local path: everything runs on the machine, nothing leaves it. It
trades accuracy and the cloud's safety layer for privacy and offline operation,
and it is assembled from the pieces this repo measured rather than assumed:

* **Retrieval** — BM25 over tldr + man chunks (`retrieve.py`), scoped to the
  utilities actually on `$PATH`, which nearly doubled recall@1 in
  `nl2sh-selfhist`. Top **k=3** chunks: the fine-tuned model degrades past that.
* **Generation** — the fine-tuned Pleias-350M (`ft/`), which routes real
  independent requests at ~0.62 (`nl2sh-selfhist`). Runs on CPU at ~12 s/query
  unquantised; loaded once and reused in the REPL.
* **Parameter grounding** — `extract_params.py` pulls literal values from the
  request (paths, ports, sizes). It does NOT splice them in (only 53% of
  extractions are whole command tokens); it **audits** the generated command and
  warns when a literal from the request is missing from it — the transcription
  failure `monad-bsky` measured a small model make 49% of the time.
* **A confirmation gate** — the command is printed and never runs until you
  press enter. On the local path there is no upstream safety model, so this gate
  and the destructive-command guard below are the only safety there is.

    python3 nlsh.py "find files bigger than 100MB under src"   # one-shot
    python3 nlsh.py                                            # REPL
    python3 nlsh.py --explain "..."                            # show retrieval + audit, don't run

Design choices that follow from the measurements, not taste:
- never auto-execute; --yes still shows the command and requires one keypress.
- a destructive-command guard refuses to even offer to run rm -rf /, mkfs, dd,
  fork bombs, etc. — it will print them for you to copy, but nlsh will not run them.
- if torch or the model is absent, degrade to showing the top retrieved
  examples: a worse tool, but still useful, and honest about which mode it is in.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import retrieve as R
import extract_params as EP
import pleias_gate as G

CMDLINE = re.compile(r"`([^`\n]{2,200})`|^\s*([a-z][a-z0-9_.+-]{1,20}\s+[^\n]{2,200})$", re.M)
DESTRUCTIVE = re.compile(r"""
    \brm\s+(-[a-z]*f|-[a-z]*r)[a-z]*\s+/(?!\w) | \bmkfs | \bdd\s+if=.*of=/dev
  | :\(\)\s*\{ | \bshutdown\b | \breboot\b | >\s*/dev/sd | \bchmod\s+-R\s+000\s+/
""", re.X)


def path_utilities() -> set[str]:
    """Every executable name on $PATH — the corpus is scoped to these."""
    out = set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            for f in os.scandir(d):
                if f.is_file() and os.access(f.path, os.X_OK):
                    out.add(f.name)
        except OSError:
            continue
    return out


class Helper:
    def __init__(self, chunks_path: Path, model_path: Path | None, scope_path: bool = True):
        chunks = R.load_chunks(chunks_path)
        if scope_path:
            on_path = path_utilities()
            scoped = [c for c in chunks if c.utility in on_path]
            # keep the full corpus only if scoping would leave us with almost nothing
            chunks = scoped if len(scoped) > 50 else chunks
            self.scoped = len(scoped) > 50
        else:
            self.scoped = False
        self.index = R.Index(chunks)
        self.model_path = model_path
        self._tok = self._model = None

    def _load_model(self):
        if self._model is not None or self.model_path is None:
            return self._model is not None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch.float32).eval()
            self._torch = torch
            return True
        except Exception as e:
            print(f"[model unavailable: {type(e).__name__}: {e}]", file=sys.stderr)
            self.model_path = None
            return False

    def retrieve(self, query: str, k: int = 3):
        hits = self.index.search(query, k=12)
        # one example per utility, top k utilities
        seen, srcs = [], []
        for chunk, _ in hits:
            if chunk.utility in seen:
                continue
            seen.append(chunk.utility)
            srcs.append(f"{chunk.utility} — {chunk.text.splitlines()[0] if chunk.text else ''}")
            if len(seen) >= k:
                break
        return srcs, seen

    def generate(self, query: str, sources: list[str], max_new_tokens: int = 140) -> str:
        if not self._load_model():
            return ""
        ids = self._tok(G.build_prompt(query, sources) + G.PREFILL, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_new_tokens,
                                       do_sample=False, pad_token_id=self._tok.pad_token_id)
        gen = self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        ans = re.split(r"<\|answer_end\|>", gen)[0].strip()
        m = CMDLINE.search(ans)
        return (m.group(1) or m.group(2) or "").strip() if m else ""

    def audit(self, query: str, command: str) -> list[str]:
        """Warn when a literal value the user named is missing from the command."""
        warnings = []
        for kind, spans in EP.extract(query).items():
            for s in spans:
                v = s["value"]
                if v and v not in command:
                    warnings.append(f"request said {kind} '{v}' — not in the command")
        return warnings

    def suggest(self, query: str, k: int = 3) -> dict:
        sources, utils = self.retrieve(query, k=k)
        command = self.generate(query, sources) if self.model_path else ""
        return {"query": query, "candidates": utils, "sources": sources,
                "command": command, "warnings": self.audit(query, command) if command else [],
                "destructive": bool(command and DESTRUCTIVE.search(command))}


def confirm_and_run(command: str, destructive: bool) -> None:
    if destructive:
        print("\n⚠  refusing to run this — it looks destructive. Copy it yourself if you mean it.")
        return
    try:
        resp = input("\nrun it? [enter = yes, anything else = no] ")
    except EOFError:
        return
    if resp.strip() == "":
        subprocess.run(["bash", "-c", command])


def render(res: dict, explain: bool) -> None:
    if not res["command"]:
        print("no command produced. closest documented utilities:")
        for u, s in zip(res["candidates"], res["sources"]):
            print(f"  {s}")
        return
    print(f"\n  $ {res['command']}")
    for w in res["warnings"]:
        print(f"  ⚠  {w}")
    if explain:
        print(f"\n  retrieved: {', '.join(res['candidates'])}")
        for s in res["sources"]:
            print(f"    · {s}")


def main() -> int:
    ap = argparse.ArgumentParser(description="on-device natural-language shell helper")
    ap.add_argument("query", nargs="*", help="the request; omit for a REPL")
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("--model", type=Path, default=HERE / "ft")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--explain", action="store_true", help="show retrieval + audit, do not run")
    ap.add_argument("--no-scope", action="store_true", help="do not restrict to $PATH utilities")
    a = ap.parse_args()

    model = a.model if a.model.exists() else None
    if model is None:
        print("[no model at ./ft — running in retrieval-only mode]", file=sys.stderr)
    helper = Helper(a.chunks, model, scope_path=not a.no_scope)
    print(f"[corpus: {helper.index.n} chunks{' ($PATH-scoped)' if helper.scoped else ''}]",
          file=sys.stderr)

    if a.query:
        res = helper.suggest(" ".join(a.query), k=a.k)
        render(res, a.explain)
        if res["command"] and not a.explain:
            confirm_and_run(res["command"], res["destructive"])
        return 0

    print("nlsh — describe what you want; ctrl-d to quit")
    while True:
        try:
            q = input("\n» ").strip()
        except EOFError:
            print()
            return 0
        if not q:
            continue
        res = helper.suggest(q, k=a.k)
        render(res, a.explain)
        if res["command"] and not a.explain:
            confirm_and_run(res["command"], res["destructive"])


if __name__ == "__main__":
    raise SystemExit(main())
