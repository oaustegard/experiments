"""Needle 2 as a routing layer in front of the Bluesky read tools.

`Router.route(query)` is one `complete()` turn: text in, a tool name plus
arguments out, or the empty call for anything the catalogue cannot serve. No
network, no Bluesky credentials — this is the layer that is being measured.

`Router.call(query)` routes, applies the confidence gate, and executes the
chosen tool through `catalogue.executors()`. Below the threshold it returns a
decision of `escalate` and executes nothing, which is the whole point of the
gate: a 14 MB model that is unsure hands the query up instead of guessing at a
network call.

Arguments the model is not allowed to fill (`transcribe`, `stopwords`, and the
thread shape knobs) never reach the executor: `EXECUTOR_ONLY` strips them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from . import catalogue

# Default gate. Chosen from the measured separation on evalset.jsonl, not by
# taste; see RESULTS.md "Where to put the gate".
DEFAULT_THRESHOLD = 0.60

# Arguments a router must never supply, whatever it emits. Model-selection and
# pagination knobs that cost money or change the shape of the answer.
EXECUTOR_ONLY = {"transcribe", "stopwords", "exclude_patterns"}


@dataclass
class Decision:
    query: str
    tool: str | None
    arguments: dict[str, Any]
    confidence: float | None
    decision: Literal["act", "escalate", "refuse"]
    reasoning: str | None = None
    latency_ms: float = 0.0
    prefill_tps: float | None = None
    decode_tps: float | None = None
    peak_ram_mb: float | None = None
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("raw", None)
        return d


class Router:
    """One Needle agent bound to one schema arm."""

    def __init__(
        self,
        arm: str = "tuned",
        threshold: float = DEFAULT_THRESHOLD,
        system: str | None = None,
        weights: str | None = None,
        tool_index_path: str | None = None,
    ):
        import needle

        self.arm = arm
        self.threshold = threshold
        self.schemas = load_schemas(arm)
        kwargs: dict[str, Any] = {"tools": self.schemas}
        if system:
            kwargs["system"] = system
        if weights:
            kwargs["weights"] = weights
        if tool_index_path:
            kwargs["tool_index_path"] = tool_index_path
        self.agent = needle.Needle(**kwargs)

    # -- routing -----------------------------------------------------------

    def route(self, query: str, reset: bool = True) -> Decision:
        if reset:
            self.agent.reset()
        t0 = time.perf_counter()
        r = self.agent.complete(query)
        dt = (time.perf_counter() - t0) * 1000.0

        calls = r.get("function_calls") or []
        conf = r.get("confidence")
        if not calls:
            # Needle's whole contract for "no declared tool serves this".
            return Decision(
                query=query,
                tool=None,
                arguments={},
                confidence=conf,
                decision="refuse",
                reasoning=r.get("reasoning"),
                latency_ms=dt,
                prefill_tps=r.get("prefill_tps"),
                decode_tps=r.get("decode_tps"),
                peak_ram_mb=r.get("peak_ram_mb"),
                raw=r,
            )

        call = calls[0]
        args = {k: v for k, v in (call.get("arguments") or {}).items() if k not in EXECUTOR_ONLY}
        gate = "act" if (conf is None or conf >= self.threshold) else "escalate"
        return Decision(
            query=query,
            tool=call.get("name"),
            arguments=args,
            confidence=conf,
            decision=gate,
            reasoning=r.get("reasoning"),
            latency_ms=dt,
            prefill_tps=r.get("prefill_tps"),
            decode_tps=r.get("decode_tps"),
            peak_ram_mb=r.get("peak_ram_mb"),
            raw=r,
        )

    # -- execution ---------------------------------------------------------

    def call(self, query: str) -> tuple[Decision, Any]:
        d = self.route(query)
        if d.decision != "act" or not d.tool:
            return d, None
        fn = catalogue.executors().get(d.tool)
        if fn is None:
            d.decision = "escalate"
            return d, None
        return d, fn(**d.arguments)


ARMS = ("auto", "auto-min", "tuned", "tuned-min")


def minimal(schemas: list[dict]) -> list[dict]:
    """Drop every optional argument, keeping only what `required` names.

    The `-min` arms. An argument the caller never needs the model to fill is an
    argument the model can fill wrong.
    """
    import copy

    out = []
    for s in schemas:
        s = copy.deepcopy(s)
        params = s.get("parameters", {})
        req = set(params.get("required", []))
        params["properties"] = {k: v for k, v in params.get("properties", {}).items() if k in req}
        out.append(s)
    return out


def load_schemas(arm: str) -> list[dict]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    base = arm.split("-")[0]
    if base == "tuned":
        from .tools_tuned import SCHEMAS as s
    else:
        from .tools_auto import build

        s = build()
    return minimal(s) if arm.endswith("-min") else s
