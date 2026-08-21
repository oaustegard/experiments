#!/usr/bin/env python3
"""The two prompt shapes stage 2 compares, over identical sources.

Stage 1 asked the model to write a command from a request plus a few documented
utilities. Issue #52 argues the job is narrower than that: §6 of
`nl2sh-dense/RESULTS.md` measured an exemplar worth +0.189 routing under oracle
sources while the *choice* of exemplar was worth zero, which reads as the model
copying a template it was handed rather than composing one. If that is what it
is doing, asking for the copy explicitly — here is the template, here are the
values, substitute — should beat asking for free generation on the same weights.

`generate` is stage 1's prompt, byte for byte (`nl2sh-retrieval/gemma_arm.py`
`build_user`), so the comparison holds the model, the sources, the distractors
and the seed fixed and varies only the instruction.

`instantiate` names the substitution and hands over the literals that
`nl2sh-retrieval/extract_params.py` lifted out of the request at 0.971
precision. `instantiate_bare` drops the literal list and keeps the framing, so
a gain can be attributed to one half or the other rather than to the pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

RETRIEVAL = Path(__file__).resolve().parent.parent / "nl2sh-retrieval"
sys.path.insert(0, str(RETRIEVAL))

import extract_params  # noqa: E402

GEN_SYS = ("Translate the request into a single shell command. "
           "Use the documented utilities below when relevant. Output only the command.")

INST_SYS = ("Below are documented shell commands, one per utility, each with a "
            "description and an example. Pick the example that does what the request "
            "asks, then rewrite it with the request's own values in place of the "
            "example's arguments. Keep the utility and the option style of the "
            "example. Output only the command.")


def build_generate(nl: str, sources: list[str]) -> str:
    """Stage 1's prompt. Identical to gemma_arm.build_user."""
    src = "\n".join(f"- {s}" for s in sources)
    return f"{GEN_SYS}\n\nUtilities:\n{src}\n\nRequest: {nl}"


def literals(nl: str) -> list[str]:
    """Values the request already contains, sliced from it, never retyped."""
    seen, out = set(), []
    for v in extract_params.values(nl):
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_instantiate(nl: str, sources: list[str], with_literals: bool = True) -> str:
    src = "\n".join(f"- {s}" for s in sources)
    parts = [INST_SYS, "", "Examples:", src, ""]
    if with_literals:
        vals = literals(nl)
        parts += ["Values taken from the request: " + (", ".join(vals) if vals else "(none)"), ""]
    parts += [f"Request: {nl}"]
    return "\n".join(parts)


ANCHOR = "\nCommand:"


def anchored(builder):
    """Same prompt, ending on a `Command:` cue.

    The first zero-shot run needed this control. Under `instantiate` the model
    answered `- go — Go to my home directory` — the shape of the source lines it
    had just been shown, bullet included — on most rows, which is a format
    imitation and says nothing about whether substitution is the easier task.
    Anchoring both conditions the same way separates the two questions: the
    plain pair measures the prompts as written, the anchored pair measures the
    framings once the output slot is unambiguous in both.
    """
    return lambda nl, srcs: builder(nl, srcs) + ANCHOR


BUILDERS = {
    "generate": lambda nl, srcs: build_generate(nl, srcs),
    "instantiate": lambda nl, srcs: build_instantiate(nl, srcs, True),
    "instantiate_bare": lambda nl, srcs: build_instantiate(nl, srcs, False),
    "generate_anchored": anchored(lambda nl, srcs: build_generate(nl, srcs)),
    "instantiate_anchored": anchored(lambda nl, srcs: build_instantiate(nl, srcs, True)),
    "instantiate_anchored_bare": anchored(lambda nl, srcs: build_instantiate(nl, srcs, False)),
}
