"""Label accounts with Claude Sonnet through headless `claude -p` children (the oracle).

    python3 -I run/label.py [--pass 1|2] [--workers 6] [--limit N]

Pass 1 labels every account in data/accounts.jsonl. Pass 2 relabels a fixed random 150 in fresh batches
(different neighbours and order), to measure the oracle's agreement with itself. Each call sees 5
accounts, exactly the state text the model will read (run/state.py, fitted to 830 Ettin tokens), and
returns, per account, a probability for every option of every question in run/questions.json plus a
probability that each numbered item is about AI. Replies are validated (every key, numbers in [0, 1])
and normalized; a batch that fails twice is logged and skipped. One file per batch under
data/labels<pass>/, so a rerun resumes. The children run with no settings, MCP or session files, from a
scratch directory.
"""
import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state import fit_account_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "run" / "questions.json").read_text())
BATCH = 5

SYSTEM = """You annotate Bluesky accounts. Each account is shown as its profile, counts, recent activity summary and numbered recent items (posts, replies, reposts; text cut to 140 characters).

For every account answer every question below with a probability for each option (numbers that sum to 1), reflecting your real uncertainty: use values near 1 only when the evidence is clear. Also give, for each numbered item, the probability that the item is ABOUT AI (artificial intelligence, machine learning, LLMs or chatbots, AI products, companies, research, policy, safety, art, jobs or energy use, or using AI tools). An item that only mentions tech, computing or science without AI is not about AI; a short reply whose topic cannot be told from its text is not about AI.

QUESTIONS
""" + "\n".join(
    f"- {name} ({q['type']}): {q['instructions']} Options: " + "; ".join(
        f"{k} = {v}" for k, v in (q["criteria"].items() if isinstance(q["criteria"], dict)
                                  else ((str(i), d) for i, d in enumerate(q["criteria"]))))
    for name, q in Q.items()) + """

Reply with ONLY a JSON object, no prose, of this shape:
{"<account id>": {"account_type": {"human": 0.0, ...}, "ai_share": {"0": 0.0, ...}, "ai_role": {...}, "ai_stance": {...}, "original_content": {...}, "items_about_ai": {"1": 0.0, "2": 0.0, ...}}, ...}
Include every account id given, every option of every question, and every item number of that account."""


def options(q):
    c = q["criteria"]
    return list(c) if isinstance(c, dict) else [str(i) for i in range(len(c))]


def n_items(state):
    return len(re.findall(r"^\d+\. \[", state, re.M))


def validate(reply, batch):
    out = {}
    for aid, st in batch:
        r = reply.get(aid)
        if not isinstance(r, dict):
            raise ValueError(f"missing account {aid}")
        rec = {}
        for name, q in Q.items():
            d = r.get(name)
            opts = options(q)
            if not isinstance(d, dict) or set(d) != set(opts):
                raise ValueError(f"{aid} {name}: keys {sorted(d) if isinstance(d, dict) else d}")
            v = [float(d[o]) for o in opts]
            if min(v) < 0 or sum(v) <= 0:
                raise ValueError(f"{aid} {name}: values {v}")
            rec[name] = {o: x / sum(v) for o, x in zip(opts, v)}
        items = r.get("items_about_ai")
        want = [str(i) for i in range(1, n_items(st) + 1)]
        if not isinstance(items, dict) or set(items) != set(want):
            raise ValueError(f"{aid} items: {sorted(items) if isinstance(items, dict) else items} vs {len(want)}")
        rec["items_about_ai"] = {k: min(1.0, max(0.0, float(items[k]))) for k in want}
        out[aid] = rec
    return out


def call(batch, scratch):
    prompt = "\n\n".join(f"=== ACCOUNT {aid} ===\n{st}" for aid, st in batch)
    cmd = ["claude", "-p", "--model", "sonnet", "--output-format", "json", "--setting-sources", "",
           "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--system-prompt", SYSTEM, prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=scratch, timeout=600)
    env = json.loads(r.stdout)
    text = env.get("result", "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return validate(json.loads(m.group(0)), batch), {"model": list((env.get("modelUsage") or {}).keys()),
                                                     "cost_usd": env.get("total_cost_usd"), "ms": env.get("duration_ms")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pas", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(snapshot_download("jhu-clsp/ettin-encoder-17m", allow_patterns=["*.json"])) / "tokenizer.json"))
    tok.no_truncation()
    count = lambda t: len(tok.encode(t, add_special_tokens=False).ids)
    accts = [json.loads(l) for l in (ROOT / "data" / "accounts.jsonl").open()]
    rng = random.Random(a.pas)
    if a.pas == 2:
        accts = random.Random(42).sample(accts, 150)
    rng.shuffle(accts)
    if a.limit:
        accts = accts[: a.limit]
    states = [(x["did"], fit_account_state(x, count)) for x in accts]
    outdir = ROOT / "data" / f"labels{a.pas}"
    outdir.mkdir(parents=True, exist_ok=True)
    batches = [states[i: i + BATCH] for i in range(0, len(states), BATCH)]
    scratch = tempfile.mkdtemp(prefix="label-")
    log = (ROOT / "data" / f"label{a.pas}.log").open("a")

    def run(ib):
        i, b = ib
        f = outdir / f"b{i:04d}.json"
        if f.exists():
            return "skip"
        for attempt in range(3):
            try:
                labels, meta = call(b, scratch)
                f.write_text(json.dumps({"labels": labels, "meta": meta, "states": dict(b)}))
                log.write(f"{time.strftime('%H:%M:%S')} b{i:04d} ok {meta}\n")
                log.flush()
                return "ok"
            except Exception as e:  # noqa: BLE001  logged, retried, then skipped
                log.write(f"{time.strftime('%H:%M:%S')} b{i:04d} attempt {attempt} {type(e).__name__}: {str(e)[:300]}\n")
                log.flush()
        return "fail"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, enumerate(batches)))
    print({k: res.count(k) for k in set(res)}, flush=True)


if __name__ == "__main__":
    main()
