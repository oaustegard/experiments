"""Custody Register episodes: a running result that must be updated in place, not just retained.

40 assets start with known holders. Each chunk carries ~16 handover lines (same six phrasings as
gen.py) shuffled among ~120 filler lines, with the handovers kept in chronological order. At the
end the agent must report, for every asset, its current holder and how many times it changed
hands. The whole stream of handover lines outgrows the carry cap well before 96 chunks; the
register itself is ~40 short rows. Keeping lines verbatim is not enough once the cap binds.
"""
from __future__ import annotations

import random

from gen import FILLER_T, NEEDLE_T, PEOPLE, TIMES, _hash

N_ASSETS, EVENTS_PER_CHUNK, FILLER_PER_CHUNK = 40, 16, 120


def assets(rng: random.Random) -> list[str]:
    kinds = ["pump", "valve", "chiller", "boiler", "compressor", "relay", "turbine", "filter-bank"]
    out = set()
    while len(out) < N_ASSETS:
        out.add(f"{rng.choice(kinds)}-{rng.randint(1, 99):02d}{rng.choice('ABCDEFGH')}")
    return sorted(out)


def episode(n_chunks: int, seed: int) -> dict:
    rng = random.Random(f"tally:{seed}:{n_chunks}")
    A = assets(rng)
    holder = {a: rng.choice(PEOPLE) for a in A}
    initial = dict(holder)
    count = {a: 0 for a in A}
    chunks = []
    for c in range(n_chunks):
        events = []
        for _ in range(EVENTS_PER_CHUNK):
            a = rng.choice(A)
            p = holder[a]
            q = rng.choice([x for x in PEOPLE if x != p])
            events.append(rng.choice(NEEDLE_T).format(a=a, p=p, q=q, t=rng.choice(TIMES)))
            holder[a], count[a] = q, count[a] + 1
        filler = []
        for _ in range(FILLER_PER_CHUNK):
            a, t = rng.choice(A), rng.choice(FILLER_T)
            # "remains with" must name the true holder, or the filler would contradict the register
            p = holder[a] if "remains with" in t else holder_or(rng, holder)
            q = rng.choice([x for x in PEOPLE if x != p])
            filler.append(t.format(a=a, p=p, q=q, t=rng.choice(TIMES), x=rng.uniform(0.5, 90)))
        # interleave: filler goes anywhere, handovers keep their order
        slots = sorted(rng.sample(range(len(events) + len(filler)), len(events)))
        lines, ei, fi = [], 0, 0
        for i in range(len(events) + len(filler)):
            if ei < len(events) and i == slots[ei]:
                lines.append(events[ei]); ei += 1
            else:
                lines.append(filler[fi]); fi += 1
        body = [f"[c{c:03d}L{i:03d}#{_hash(rng)}] {t}" for i, t in enumerate(lines)]
        chunks.append("\n".join([f"=== chunk {c + 1}/{n_chunks} ({n_chunks - c - 1} remaining after this) ===",
                                 *body]))
    return {"n_chunks": n_chunks, "seed": seed, "assets": A, "initial": initial, "chunks": chunks,
            "final_holder": holder, "final_count": count}


def holder_or(rng: random.Random, holder: dict) -> str:
    # filler mentions real current holders half the time, so "remains with X" lines look plausible
    return rng.choice(list(holder.values())) if rng.random() < 0.5 else rng.choice(PEOPLE)


def task(ep: dict) -> str:
    reg = "\n".join(f"{a}: {p}" for a, p in ep["initial"].items())
    return ("You are processing a long stream of chunks of an equipment log, in chronological order "
            "(within a chunk, earlier lines happened first). Track the custody register: for each of the "
            "40 assets below, who holds it now and how many times it has changed hands since the start. "
            "A handover is any line recording that custody, ownership or maintenance responsibility passed "
            "from one person to another. Lines where responsibility stays put, is only discussed, or is "
            "shadowed for training do not change the register. At the end you will be asked for the final "
            "holder and handover count of every asset.\n\nInitial register (count 0 for each):\n" + reg)


if __name__ == "__main__":
    import sys
    e = episode(int(sys.argv[1]), int(sys.argv[2]))
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ev = sum(e["final_count"].values())
    print({"events": ev, "chunk0_tokens": len(enc.encode(e["chunks"][0])),
           "max_count": max(e["final_count"].values())})
    print(e["chunks"][0][:900])
