"""Replay rows from SYNTH, Monad's own training set, mixed into fine-tuning to limit forgetting.

    python3 fetch_replay.py --n 3000     # -> data/replay.jsonl

Random offsets through the Hugging Face datasets-server rows API (seeded), English rows only, and
reasoning + answer under 3,000 characters so a row fits the 1,024-token training window.
"""
import argparse, json, random, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://datasets-server.huggingface.co"


def get(url):
    for t in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"}), timeout=60))
        except Exception as e:  # noqa: BLE001
            print("retry", t, e, flush=True)
            time.sleep(3 * 2 ** t)
    raise SystemExit("datasets-server unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--out", default="data/replay.jsonl")
    a = ap.parse_args()
    total = get(f"{API}/size?dataset=PleIAs/SYNTH")["size"]["splits"][0]["num_rows"]
    rng = random.Random(a.seed)
    out, seen = [], set()
    while len(out) < a.n:
        off = rng.randrange(total - 100)
        d = get(f"{API}/rows?dataset=PleIAs/SYNTH&config=default&split=train&offset={off}&length=100")
        for r in (x["row"] for x in d["rows"]):
            if r["language"] != "en" or r["synth_id"] in seen:
                continue
            n = len(r["synthetic_reasoning"]) + len(r["synthetic_answer"])
            if not a.min_chars <= n <= a.max_chars or not r["synthetic_answer"].strip():
                continue
            seen.add(r["synth_id"])
            out.append({"id": r["synth_id"], "exercise": r["exercise"], "q": r["query"],
                        "reasoning": r["synthetic_reasoning"], "answer": r["synthetic_answer"]})
        print(len(out), flush=True)
    with open(HERE / a.out, "w") as f:
        for r in out[: a.n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total rows", total)


if __name__ == "__main__":
    main()
