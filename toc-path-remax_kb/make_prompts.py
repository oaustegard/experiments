"""Sample gold chunks and emit blind query-writing prompts.

The prompts carry ONLY arm A's indexed body text — the same bytes production
indexes today. No heading path, no source filename, no statement of what is
being tested. See PLAN.md for why both omissions are load-bearing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
N_CHUNKS = 400
PER_BATCH = 40
MIN_CHARS = 220
SEED = 20260909


def load_a() -> list[dict]:
    return [json.loads(l) for l in (DATA / "chunks_A.jsonl").read_text().splitlines()]


def sample(chunks: list[dict]) -> list[dict]:
    """Stratified by source file so no skill dominates the query set."""
    rng = random.Random(SEED)
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        if len(c["text"]) >= MIN_CHARS:
            by_file.setdefault(c["source_path"], []).append(c)

    files = sorted(by_file)
    picked: list[dict] = []
    round_i = 0
    while len(picked) < N_CHUNKS:
        added = False
        for f in files:
            pool = by_file[f]
            if round_i < len(pool):
                picked.append(pool[round_i])
                added = True
                if len(picked) >= N_CHUNKS:
                    break
        if not added:
            break
        round_i += 1
    rng.shuffle(picked)
    # de-dup by id, keep order
    seen, out = set(), []
    for c in picked:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out[:N_CHUNKS]


INSTRUCTIONS = """\
Below are {n} numbered passages from a collection of technical documentation.

For each passage, write exactly 2 questions that a person could plausibly type
into a documentation search box, where THIS passage is the answer they want.

Rules:
- Write the question the way a real user types it: natural, specific, usually
  under 15 words. Not a quiz question, not a restatement of the passage.
- Do not copy long phrases verbatim from the passage. Use the vocabulary a user
  would bring, which is often different from the passage's own wording.
- The question must be answerable from the passage, and specific enough that
  this passage answers it better than a generic page on the same broad topic.
- Two questions for the same passage should approach it differently, not be
  rewordings of each other.

Output JSONL, one line per passage, nothing else:
{{"n": <passage number>, "queries": ["<question 1>", "<question 2>"]}}

Passages:

"""


def main() -> None:
    chunks = sample(load_a())
    (DATA / "gold_chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n"
    )
    prompt_dir = DATA / "prompts"
    prompt_dir.mkdir(exist_ok=True)

    n_batches = 0
    for start in range(0, len(chunks), PER_BATCH):
        batch = chunks[start:start + PER_BATCH]
        body = INSTRUCTIONS.format(n=len(batch))
        for i, c in enumerate(batch):
            body += f"\n--- passage {start + i} ---\n{c['text']}\n"
        (prompt_dir / f"batch_{start:04d}.txt").write_text(body)
        n_batches += 1

    files = sorted({c["source_path"] for c in chunks})
    print(f"{len(chunks)} gold chunks across {len(files)} files")
    print(f"{n_batches} prompt batches of <= {PER_BATCH} -> {prompt_dir}")
    leaked = [c for c in chunks if c["heading_path"] and c["heading_path"] in
              (prompt_dir / f"batch_0000.txt").read_text()]
    print(f"sanity: heading paths appearing verbatim in batch 0 prompt: {len(leaked)}")


if __name__ == "__main__":
    main()
