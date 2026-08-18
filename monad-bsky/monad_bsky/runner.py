"""Load Monad, generate one call per query, score it like the Needle arms.

Greedy decoding, single turn, CPU. `MonadRouter.route(query)` returns the same
`Decision`-shaped record the Needle arms produce, minus `confidence`: Monad has
no calibrated confidence head, which is one of the two things it gives up
relative to a purpose-built tool-caller (the other is the decode grammar).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompt import build_prompt, parse_call, thinking

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "model")


@dataclass
class Decision:
    query: str
    tool: str | None
    arguments: dict
    status: str
    confidence: None = None
    decision: str = "act"
    reasoning: str | None = None
    latency_ms: float = 0.0
    completion: str = ""
    prompt_tokens: int = 0
    new_tokens: int = 0
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("raw", None)
        return d


class MonadRouter:
    def __init__(self, model_dir: str = DEFAULT_MODEL, schemas: list[dict] | None = None, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        self.model.eval()
        self.schemas = schemas or []
        self.max_new_tokens = max_new_tokens
        # `<|im_end|>` is not a special token in this tokenizer, so generation
        # cannot stop on it directly; the completion is truncated at the first
        # occurrence in text instead.
        self.eos_id = self.tok.eos_token_id

    def complete(self, query: str) -> tuple[str, float, int, int]:
        prompt = build_prompt(self.schemas, query)
        ids = self.tok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.eos_id,
            )
        dt = (time.perf_counter() - t0) * 1000.0
        n_prompt = ids["input_ids"].shape[1]
        new = out[0][n_prompt:]
        text = self.tok.decode(new, skip_special_tokens=False)
        return text, dt, n_prompt, int(new.shape[0])

    def route(self, query: str) -> Decision:
        text, dt, n_prompt, n_new = self.complete(query)
        name, args, status = parse_call(text)
        return Decision(
            query=query,
            tool=name,
            arguments=args,
            status=status,
            reasoning=thinking(text)[:300],
            latency_ms=dt,
            completion=text[:600],
            prompt_tokens=n_prompt,
            new_tokens=n_new,
        )


def load_eval_schemas(arm: str = "tuned-min") -> list[dict]:
    """The same 18 schemas the Needle arms declare, from the sibling experiment."""
    import sys

    sys.path.insert(0, str(_needle_dir()))
    from needle_bsky.router import load_schemas

    return load_schemas(arm)


def _needle_dir() -> Path:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _lib.paths import experiment

    return experiment("needle-bsky")


def load_eval_items() -> list[dict]:
    import json

    path = _needle_dir() / "evalset.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score_one(item: dict, d: Decision) -> dict:
    """Same scoring rules as `needle-bsky/eval.py`, plus a parse status."""
    import sys

    sys.path.insert(0, str(_needle_dir()))
    from eval import norm

    accepted = item["tool"]
    refuse_expected = not accepted
    refused = d.tool is None

    if refuse_expected:
        tool_ok = refused
        args_ok = tool_ok
        invented: list[str] = []
    else:
        tool_ok = (not refused) and d.tool in accepted
        expected_args = item.get("args", {})
        args_ok = tool_ok and all(
            k in d.arguments and norm(d.arguments[k]) == norm(v) for k, v in expected_args.items()
        )
        licensed = set(expected_args) | set(item.get("evidenced", []))
        invented = sorted(
            k for k, v in d.arguments.items() if k not in licensed and v not in (None, "", [], {})
        )

    row: dict[str, Any] = {
        "id": item["id"],
        "cat": item["cat"],
        "query": item["query"],
        "expected": accepted,
        "got": d.tool,
        "arguments": d.arguments,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "invented": invented,
        "status": d.status,
        "latency_ms": round(d.latency_ms, 1),
        "prompt_tokens": d.prompt_tokens,
        "new_tokens": d.new_tokens,
        "reasoning": d.reasoning,
    }
    return row
