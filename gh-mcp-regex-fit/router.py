#!/usr/bin/env python3
"""Apply a fitted decision list, then bind arguments by extraction.

Two properties are deliberate and both come out of `monad-bsky`:

* **No fallback label.** The rules that fire, fire; anything else abstains. The
  catch-all `search_posts` in `regex_only.py` is what took its refusal accuracy
  from 0.500 on the fitted set to 0.183 on unseen queries.
* **Arguments are copied, never generated.** `repair.py` recovered most of a
  56M-parameter model's argument gap by refilling structurally extractable
  values from the request. Here there is no model to refill *for* — the binder
  is the only thing that ever produces an argument.
"""

from __future__ import annotations

import json
from pathlib import Path

from catalogue import load as load_catalogue
from cues import extract
from fit import featurise, label_text, schema_vocab

HERE = Path(__file__).resolve().parent


class Router:
    def __init__(self, rules_path: Path):
        blob = json.loads(Path(rules_path).read_text())
        self.rules = [(tuple((f, bool(p)) for f, p in r["literals"]), r["label"])
                      for r in blob["rules"]]
        self.catalogue = load_catalogue("session")
        self.vocab = {"schema": schema_vocab(self.catalogue),
                      "open": None, "cues": set()}[blob["vocab"]]
        self.bigrams = blob.get("bigrams", True)
        self.top = blob.get("overlap", 0)
        self.texts = label_text(self.catalogue) if self.top else None
        self.meta = blob

    def route(self, query: str) -> str | None:
        f = featurise(query, self.vocab, self.bigrams, self.texts, self.top)
        for lits, label in self.rules:
            if all((lit in f) == positive for lit, positive in lits):
                return label
        return None

    def call(self, query: str) -> dict | None:
        """A full call: tool name, method where the tool dispatches, bound args."""
        label = self.route(query)
        if label is None:
            return None
        tool, _, method = label.partition("::")
        spec = self.catalogue[tool]
        args = {k: v for k, v in extract(query).items() if k in spec["params"]}
        if method:
            args["method"] = method
        return {"tool": tool, "method": method or None, "args": args,
                "missing_required": [k for k in spec["required"] if k not in args]}


def main() -> int:
    import sys
    r = Router(HERE / "rules_schema.json")
    for q in sys.argv[1:] or ["is the CI green on oaustegard/experiments#412",
                              "what time is it in oslo"]:
        print(f"{q}\n  -> {r.call(q)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
