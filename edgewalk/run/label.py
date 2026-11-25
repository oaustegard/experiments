"""Label sandwich cases with Jev: P(this food is a sandwich). Appends to data/labels.jsonl and resumes.

    python3 edgewalk/run/label.py --n 6000 [--workers 8]
Needs TYPESAFE_API_KEY (or the Cloudflare gateway pair) in the environment; the transport is
claude-workspace's scripts/model_advisor.py `_jev`.
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from _lib.paths import spoke  # noqa: E402
import sandwich  # noqa: E402

sys.path.insert(0, str(spoke("claude-workspace") / "scripts"))
import model_advisor  # noqa: E402

OUT = HERE.parent / "data" / "labels.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cases, seen = [], set()
    for name, vals in sandwich.CLASSICS.items():   # the classics first, unchanged
        c = dict(zip(sandwich.KEYS, vals)); cases.append(c); seen.add(json.dumps(c, sort_keys=True))
    while len(cases) < a.n:
        c = sandwich.sample(rng); k = json.dumps(c, sort_keys=True)
        if k not in seen:
            seen.add(k); cases.append(c)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            done.add(json.dumps(json.loads(line)["case"], sort_keys=True))
    todo = [c for c in cases if json.dumps(c, sort_keys=True) not in done]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    lock, fails = threading.Lock(), [0]
    OUT.parent.mkdir(exist_ok=True)

    def one(c):
        for attempt in range(3):
            try:
                ans = model_advisor._jev(sandwich.state(c), sandwich.QUESTION)
                p = float(ans["is_sandwich"]["noul"])
                with lock, OUT.open("a") as f:
                    f.write(json.dumps({"case": c, "p": p}) + "\n")
                return
            except (model_advisor.AdvisorError, KeyError, TypeError, ValueError) as e:
                err = e
        with lock:
            fails[0] += 1
            print("fail", str(err)[:120], flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 500 == 0:
                print(i, flush=True)
    print("failures", fails[0], flush=True)


if __name__ == "__main__":
    main()
