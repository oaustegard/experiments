"""Greedy generation on an 8-bit page export with onnxruntime CPU (monad-weight-edit/ort_gen.py's loop,
the reference the browser runtime matches), for three probe sets:

    python3 evalgen.py export/r2_A8p_q8 facts  runs/A8p_q8__facts.jsonl    # monad-chat/probes_facts.json (30)
    python3 evalgen.py export/r2_A8p_q8 papers runs/A8p_q8__papers.jsonl   # held-out paper questions (135)
    python3 evalgen.py export/r2_A8p_q8 dev    runs/A8p_q8__dev.jsonl      # monad-weight-edit dev fixture (192)

Appends one line per prompt to <out>.part and skips ids already written there, so an interrupted run
resumes; the file is renamed to <out> only when the set is complete. (A committed output file that a
later `git rebase` rewrote while this was appending to it lost every row: the process kept writing to
the replaced inode.)
"""
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MWE = experiment("monad-weight-edit")
sys.path.insert(0, str(MWE))
from ort_gen import generate, pieces, session  # noqa: E402
from common import loop_period, split_think  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def items(kind):
    """Prompts of one set from data/eval_prompts.jsonl (written by build_data.py)."""
    return [r for r in map(json.loads, open(HERE / "data/eval_prompts.jsonl")) if r["set"] == kind]


class Tok:
    def __init__(self, path):
        self.t = Tokenizer.from_file(str(path))

    def encode(self, s):
        return self.t.encode(s, add_special_tokens=False).ids

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=False)


def main():
    model, kind, final = sys.argv[1], sys.argv[2], HERE / sys.argv[3]
    if final.exists():
        print("complete:", final)
        return
    out = final.with_name(final.name + ".part")
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    mdir = HERE / model
    tok = Tok(mdir / "tokenizer.json")
    piece = [tok.decode([i]) for i in range(8192)]
    sess = session(str(mdir), threads=4)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [x for x in items(kind) if x["id"] not in done]
    todo.sort(key=lambda x: x["max_new"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("a") as f:
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            r = generate(sess, [tok.encode(prompt(x["q"])) for x in chunk], max(x["max_new"] for x in chunk), piece)
            for x, tk, st in zip(chunk, r["tokens"], r["stopped"]):
                text = tok.decode(tk)
                trace, ans, closed = split_think(text)
                row = {"set": kind, "id": x["id"], "q": x["q"], "stopped": st, "closed": closed, "tokens": len(tk),
                       "loop": (not st) and loop_period(tk) is not None, "answer": ans, "trace": trace}
                if "check" in x:
                    row["ok"] = bool(re.search(x["check"], ans, re.I))
                if "task" in x:
                    row["task"] = x["task"]
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{k + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    out.rename(final)


if __name__ == "__main__":
    main()
