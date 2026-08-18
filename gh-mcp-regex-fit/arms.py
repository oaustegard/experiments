#!/usr/bin/env python3
"""The interface every routing arm implements, and the registry `eval.py` reads.

An arm answers one question: given a request, which of the 79 routing targets is
it, or none. Anything that can do that — twenty regex rules, a fitted decision
list, a BM25 ranker, an encoder — is comparable on the same three splits.

    class Arm:
        def route(self, query: str) -> str | None: ...
        # optional, for gated and cascaded arms:
        def score(self, query: str) -> list[tuple[str, float]]: ...

`score` returns (label, score) descending. An arm that has one gets a threshold
sweep for free; an arm that does not is measured at its single operating point.

Register by name so nothing has to edit this file's callers:

    register("bm25-schema", lambda: BM25Arm(source="schema"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

REGISTRY: dict[str, Callable[[], object]] = {}
UNAVAILABLE: dict[str, str] = {}


def register(name: str, factory: Callable[[], object]) -> None:
    REGISTRY[name] = factory


def build(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown arm {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]()


def load_all() -> None:
    """Import every arm module that is present, skipping ones whose deps are absent."""
    import importlib
    for mod in ("baseline_arms", "bm25_arms", "spacy_arms", "encoder_arms", "cascade_arms"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            # Catch everything, not just ImportError: an optional arm whose
            # dependency is absent, or which is mid-edit, must not take the
            # whole evaluation down with it.
            UNAVAILABLE[mod] = f"{type(e).__name__}: {e}"


class ArmBase:
    """Gives any arm with a `route` the argument binding and `call` shape for free.

    Arguments are always bound by extraction from the request, never generated —
    the one thing from `monad-bsky` that transferred unconditionally.
    """

    def route(self, query: str) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def catalogue(self):
        from catalogue import load
        if not hasattr(self, "_cat"):
            self._cat = load("session")
        return self._cat

    def call(self, query: str) -> dict | None:
        from cues import extract
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


def labels(catalogue: dict | None = None) -> list[str]:
    """The 79 routing targets: a tool, or `tool::method` where it dispatches."""
    if catalogue is None:
        from catalogue import load
        catalogue = load("session")
    out = []
    for name, tool in sorted(catalogue.items()):
        m = tool["params"].get("method")
        out += [f"{name}::{e}" for e in m["enum"]] if m and m.get("enum") else [name]
    return out
