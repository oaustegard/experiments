"""Score a run file (gpu_eval.py or evalgen.py output) on all three probe sets.

    python3 score.py runs_gpu/A8p.jsonl [more ...] [--judge] [--json results/x.json]

papers  held-out questions about the 45 papers. key_recall: share of each reference key point whose
        content words (3+ letters) are at least 60% present in the answer, averaged. With --judge,
        Gemini (gateway) grades each answer against the reference: correct / partial / wrong, and
        `judged` is correct + half partial. Split by post-cutoff (dated 2025-11-01 or later) vs older.
        Verdicts are cached in data/judge_cache.json by (question, answer).
facts   monad-chat's 30 one-line facts, its regex check on the post-think answer.
dev     monad-weight-edit's dev fixture: loop rate (hit the token limit inside a repetition),
        unclosed think rate, MMLU and GSM8K accuracy with that experiment's parsers.
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("monad-weight-edit")))
from common import mmlu_parse  # noqa: E402
from summarize import gsm8k_answer  # noqa: E402

CACHE = HERE / "data/judge_cache.json"
JUDGE = """You grade answers from a very small language model about recent research papers.
For each item, compare the MODEL ANSWER with the REFERENCE ANSWER and KEY POINTS. Verdict:
"correct": states the key facts asked for, with no contradicting claim about them.
"partial": gets some of the asked-for facts right but misses or garbles others.
"wrong": does not give the asked-for facts, gives different ones, or is empty.
Ignore style, length and extra unrelated text; judge only whether the asked-for facts are there and right.
Return a JSON list of {{"i": <item number>, "verdict": "correct"|"partial"|"wrong"}} for every item.

{items}"""


def words(s):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


def key_hit(kp, ans):
    w = words(kp)
    return bool(w) and len(w & words(ans)) / len(w) >= 0.6


def gemini(prompt):
    import requests
    url = (f"https://gateway.ai.cloudflare.com/v1/{os.environ['CF_ACCOUNT_ID']}/{os.environ['CF_GATEWAY_ID']}"
           f"/google-ai-studio/v1beta/models/gemini-3.8-flash:generateContent")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": 8192}}
    hdr = {"Content-Type": "application/json", "cf-aig-authorization": f"Bearer {os.environ['CF_API_TOKEN']}"}
    for a in range(6):
        r = requests.post(url, json=body, headers=hdr, timeout=300)
        if r.status_code == 200:
            break
        time.sleep(3 * 2 ** a)
    r.raise_for_status()
    return json.loads("".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought")))


def judge(rows, ho):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = lambda r: hashlib.sha256((ho[r["id"]]["q"] + "\x00" + r["answer"]).encode()).hexdigest()[:24]  # noqa: E731
    todo = [r for r in rows if key(r) not in cache and r["answer"].strip()]
    for i in range(0, len(todo), 15):
        ch = todo[i:i + 15]
        items = "\n\n".join(f"ITEM {j}\nQUESTION: {ho[r['id']]['q']}\nREFERENCE ANSWER: {ho[r['id']]['answer']}\n"
                            f"KEY POINTS: {'; '.join(ho[r['id']]['key_points']) or '(the facts in the reference answer)'}\nMODEL ANSWER: {r['answer'][:1500]}"
                            for j, r in enumerate(ch))
        for v in gemini(JUDGE.format(items=items)):
            cache[key(ch[int(v["i"])])] = v["verdict"]
        CACHE.write_text(json.dumps(cache, indent=0))
    return {r["id"]: (cache[key(r)] if r["answer"].strip() else "wrong") for r in rows}


def score(rows, use_judge):
    out = {}
    ho = {r["id"]: r for f in ("heldout", "seen") for r in map(json.loads, open(HERE / f"data/{f}.jsonl"))}
    ev = {r["id"]: r for r in map(json.loads, open(HERE / "data/eval_prompts.jsonl"))}
    for setname in ("papers", "seen"):
        out.update(score_papers([r for r in rows if r.get("set") == setname], ho, use_judge, setname))
    return out | score_rest(rows, ev)


def score_papers(P, ho, use_judge, setname):
    out = {}
    if P:
        verdicts = judge(P, ho) if use_judge else {}
        for name, sub in (("all", P), ("post", [r for r in P if ho[r["id"]]["post_cutoff"]]),
                          ("pre", [r for r in P if not ho[r["id"]]["post_cutoff"]])):
            kr = [sum(key_hit(k, r["answer"]) for k in ho[r["id"]]["key_points"]) / max(1, len(ho[r["id"]]["key_points"])) for r in sub]
            s = {"n": len(sub), "key_recall": sum(kr) / len(sub), "loop": sum(r["loop"] for r in sub) / len(sub),
                 "closed": sum(r["closed"] for r in sub) / len(sub)}
            if verdicts:
                v = [verdicts[r["id"]] for r in sub]
                s.update({"correct": v.count("correct") / len(v), "partial": v.count("partial") / len(v),
                          "judged": (v.count("correct") + 0.5 * v.count("partial")) / len(v)})
            if not any(ho[r["id"]]["key_points"] for r in sub):
                s.pop("key_recall")
            out[f"{setname}_{name}"] = s
    return out


def score_rest(rows, ev):
    out = {}
    F = [r for r in rows if ev.get(r["id"], {}).get("set") == "facts"]
    if F:
        out["facts"] = {"n": len(F), "ok": sum(bool(re.search(ev[r["id"]]["check"], r["answer"], re.I)) for r in F),
                        "loop": sum(r["loop"] for r in F) / len(F)}
    D = [r for r in rows if ev.get(r["id"], {}).get("set") == "dev"]
    if D:
        s = {"n": len(D), "loop": sum(r["loop"] for r in D) / len(D), "open": sum(not r["closed"] for r in D) / len(D)}
        for t in ("mmlu", "gsm8k", "ifeval"):
            T = [r for r in D if ev[r["id"]]["task"] == t]
            s[f"{t}_loop"] = sum(r["loop"] for r in T) / len(T)
            if t == "mmlu":
                s["mmlu_acc"] = sum(r["closed"] and mmlu_parse(r["answer"]) == ev[r["id"]]["gold"] for r in T) / len(T)
            if t == "gsm8k":
                s["gsm8k_acc"] = sum(r["closed"] and gsm8k_answer(r["answer"]) == ev[r["id"]]["gold"] for r in T) / len(T)
        out["dev"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for f in a.files:
        rows = [json.loads(l) for l in open(f)]
        res[Path(f).stem] = score(rows, a.judge)
        print(Path(f).stem, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                        for k, v in res[Path(f).stem].items()}), flush=True)
    if a.json:
        p = HERE / a.json
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(res)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
