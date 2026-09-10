"""Arm D: S4-style comparative rewrites, one per query.

The container has no ANTHROPIC_API_KEY by design, so this emits batched prompts
for the parent session to dispatch and then consolidates the returned JSON.

    python3 make_rewrites.py --emit 10        # writes prompts/batch_XX.txt
    python3 make_rewrites.py --consolidate    # reads prompts/out_XX.json
"""
import argparse, glob, json, os
import pyarrow.parquet as pq

DATA = "/home/user/erb-data"
OUT = os.path.dirname(os.path.abspath(__file__))
P = f"{OUT}/prompts"

INSTRUCTION = """\
You are reproducing strategy S4 ("Comparative") from arXiv:2609.05637, applied
to an enterprise-search benchmark about a fictional company called Redwood
Inference that sells AI model inference.

For each numbered question below, write ONE comparative rewrite: a single
question that reframes the original as a comparison against alternatives,
siblings, or neighbouring cases. The paper's own example is
"What are the side effects of Tylenol?" -> "How do Tylenol's side effects
compare to Advil's?".

Rules:
- Exactly one rewrite per question, on one line.
- Keep the original's entities and jargon verbatim. Do not expand acronyms, do
  not invent product names, do not add facts.
- Stay a question. Do not answer it.
- Return ONLY a JSON object mapping question id to rewrite, no prose around it.

Questions:
"""


def load():
    q = pq.read_table(f"{DATA}/questions_test.parquet").to_pydict()
    return [dict(qid=i, text=x) for i, x, g in
            zip(q["question_id"], q["question"], q["expected_doc_ids"])
            if g is not None and len(g) > 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", type=int, default=0)
    ap.add_argument("--consolidate", action="store_true")
    a = ap.parse_args()
    qs = load()

    if a.emit:
        os.makedirs(P, exist_ok=True)
        size = -(-len(qs) // a.emit)
        for b in range(a.emit):
            part = qs[b * size:(b + 1) * size]
            if not part:
                continue
            body = "\n".join(f'{x["qid"]}: {x["text"]}' for x in part)
            open(f"{P}/batch_{b:02d}.txt", "w").write(INSTRUCTION + body + "\n")
        print(f"wrote {len(glob.glob(f'{P}/batch_*.txt'))} batches, {size} questions each")

    if a.consolidate:
        rw = {}
        for f in sorted(glob.glob(f"{P}/out_*.json")):
            rw.update(json.load(open(f)))
        want = {x["qid"] for x in qs}
        missing = sorted(want - set(rw))
        extra = sorted(set(rw) - want)
        json.dump({k: rw[k] for k in sorted(want & set(rw))},
                  open(f"{OUT}/results/rewrites_S4.json", "w"), indent=1)
        print(f"consolidated {len(want & set(rw))}/{len(want)} | missing {len(missing)} | extra {len(extra)}")
        if missing:
            print("missing:", missing[:10])
