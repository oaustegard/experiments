"""Needle Retention episodes, after ContextBench (CLM paper, arXiv 2609.37725, Appendix D).

An episode is a stream of ~3.5k-token chunks. Each chunk holds 140 filler lines and
2-8 needle lines. Every line carries an `[id#hash]` prefix, so a retyped line that
drops or alters a character no longer matches.

Two variants:
  labeled    the paper's format: needles sit under a NEEDLES header, filler inside a
             FILLER-BLOCK the agent is told to delete.
  unlabeled  needles are shuffled into the filler. A needle is any line recording a
             change in who is responsible for an asset, in one of several phrasings;
             filler includes near-misses that name the same people and assets.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PEOPLE = ["Abara", "Brandt", "Castillo", "Dimitrova", "Eze", "Fournier", "Gupta", "Haugen",
          "Ishikawa", "Jovanovic", "Kowalski", "Lindqvist", "Mbeki", "Novak", "Okafor", "Petrov",
          "Quispe", "Rasmussen", "Sato", "Tanaka", "Ueda", "Valdez", "Wojcik", "Xu", "Yilmaz", "Zielinski"]
KINDS = ["pump", "valve", "chiller", "boiler", "compressor", "relay", "turbine", "filter-bank"]

NEEDLE_T = [
    "custody of {a} passed from {p} to {q} at the {t} shift change.",
    "{q} now maintains {a}, taking over from {p}.",
    "{a} was handed over to {q} by {p}; sign-off recorded.",
    "responsibility for {a} moves to {q} effective {t}; {p} is released from it.",
    "{p} transferred ownership of {a} to {q} after the review.",
    "from {t} onward {a} belongs to {q}'s crew rather than {p}'s.",
]
FILLER_T = [
    "{a} pressure read {x:.1f} bar during the {t} round.",
    "{p} inspected {a} and found no leaks.",
    "{q} asked {p} about the maintenance schedule for {a}.",
    "{a} vibration within tolerance at {x:.1f} mm/s.",
    "{p} logged a routine lubrication of {a}.",
    "{a} remains with {p}; no change this week.",
    "{q} shadowed {p} on {a} for training only.",
    "temperature at {a} outlet steady near {x:.1f} C.",
    "{p} and {q} discussed whether {a} should change hands next quarter; no decision.",
    "spare parts for {a} ordered by {q}.",
]
TIMES = ["morning", "evening", "night", "06:00", "14:00", "22:00", "Monday", "Thursday"]


def _hash(rng: random.Random) -> str:
    return "%08x" % rng.getrandbits(32)


def _fill(rng: random.Random, tmpl: str) -> str:
    p, q = rng.sample(PEOPLE, 2)
    a = f"{rng.choice(KINDS)}-{rng.randint(1, 99):02d}{rng.choice('ABCDEFGH')}"
    return tmpl.format(a=a, p=p, q=q, t=rng.choice(TIMES), x=rng.uniform(0.5, 90))


def episode(n_chunks: int, seed: int, variant: str) -> dict:
    rng = random.Random(f"{seed}:{n_chunks}:{variant}")
    chunks, needles = [], []
    for c in range(n_chunks):
        k = rng.randint(2, 8)
        nl = [f"[n{c:03d}i{i:02d}#{_hash(rng)}] {_fill(rng, rng.choice(NEEDLE_T))}" for i in range(k)]
        fl = [f"[f{c:03d}x{i:03d}#{_hash(rng)}] {_fill(rng, rng.choice(FILLER_T))}" for i in range(140)]
        needles += nl
        head = f"=== chunk {c + 1}/{n_chunks} ({n_chunks - c - 1} remaining after this) ==="
        if variant == "labeled":
            body = ["NEEDLES (keep these lines verbatim):", *nl,
                    f"<<<FILLER-BLOCK {c:05d} START -- discard this entire block>>>", *fl,
                    f"<<<FILLER-BLOCK {c:05d} END>>>"]
        else:
            lines = nl + fl
            rng.shuffle(lines)
            # unlabeled ids must not leak the class: renumber every line uniformly
            body = []
            for i, ln in enumerate(lines):
                rest = ln.split("] ", 1)[1]
                h = ln.split("#", 1)[1].split("]", 1)[0]
                new = f"[c{c:03d}L{i:03d}#{h}] {rest}"
                if ln.startswith("[n"):
                    needles[needles.index(ln)] = new
                body.append(new)
        chunks.append("\n".join([head, *body]))
    return {"n_chunks": n_chunks, "seed": seed, "variant": variant, "chunks": chunks, "needles": needles}


TASK = {
    "labeled": ("You are processing a long stream of chunks. Keep every line listed under a NEEDLES "
                "header verbatim, exactly as written including its [id#hash] prefix. Everything inside "
                "a FILLER-BLOCK is noise. At the end, what you kept is graded: each needle line must "
                "appear exactly; kept noise wastes budget."),
    "unlabeled": ("You are processing a long stream of chunks of an equipment log. Keep verbatim every "
                  "line that records a change in who is responsible for an asset (custody, ownership or "
                  "maintenance handed from one person to another), exactly as written including its "
                  "[id#hash] prefix. Lines where responsibility stays put, is only discussed, or is "
                  "shadowed for training are noise. At the end, what you kept is graded: each handover "
                  "line must appear exactly; kept noise wastes budget."),
}

if __name__ == "__main__":
    n, s, v = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    e = episode(n, s, v)
    print(json.dumps({"needles": len(e["needles"]), "chars": sum(map(len, e["chunks"]))}))
    Path(sys.argv[4]).write_text(e["chunks"][0]) if len(sys.argv) > 4 else None
