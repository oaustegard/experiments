"""How small a token table can ship if rows it lacks are fetched from the full table when needed.

    python3 -I run/vocab_fallback.py collect      # two Bluesky account pools, A then B (data/bsky/)
    python3 -I run/vocab_fallback.py measure      # results/vocab_fallback.json

The fallback is exact by construction: the page keeps the full tokenizer, so token ids never change,
and a row it lacks is read byte-for-byte from the shipped weights file with an HTTP range request
(GitHub Pages answers 206). The only questions are bytes and how often a request needs a fetch.

Core tables are the K most frequent token ids in pool A's account text plus the benchmark's training
split, always including the special tokens and the page's own question text. Pool B is collected
afterwards from a fresh firehose window and other accounts, so it is held out from the ranking. Each
account is tokenized the way the account decider reads it: one account state and one state per item.

Pool texts are other people's public posts: data/ is gitignored and only counts are written.
"""
import asyncio
import brotli
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "bsky-account-decider" / "run"))
from common import WORKFLOWS, examples, load_cases  # noqa: E402
from weight_probe import MD, Enc  # noqa: E402

DATA = ROOT / "data" / "bsky"
KS = [2000, 4000, 8000, 12000, 16000, 24000, 32000]


def collect():
    import collect as C  # bsky-account-decider/run/collect.py, functions only
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    members = C.list_members()
    taken = set()
    for name in ("A", "B"):
        out = DATA / f"pool_{name}.jsonl"
        if out.exists():
            taken |= {json.loads(l)["did"] for l in out.open()}
            continue
        sm = rng.sample(members, 40)
        with ThreadPoolExecutor(8) as ex:
            fl = [d for ds in ex.map(C.follows_of, sm) for d in ds]
        disc = list(dict.fromkeys(C.discover_authors(pages=4)))
        fire = list(asyncio.run(C.firehose_authors(seconds=150)))
        for L in (fl, disc, fire):
            rng.shuffle(L)
        plan, seen = [], set(taken)
        for src, L, n in (("list", rng.sample(members, 150), 150), ("follows", fl, 150),
                          ("discover", disc, 150), ("firehose", fire, 250)):
            k = 0
            for d in L:
                if k >= n:
                    break
                if d not in seen:
                    seen.add(d)
                    plan.append((src, d))
                    k += 1
        prof = C.profiles([d for _, d in plan])
        kept = 0
        with out.open("w") as f, ThreadPoolExecutor(8) as ex:
            for (src, d), items in zip(plan, ex.map(lambda x: C.feed_items(x[1]), plan)):
                if d not in prof or len(items) < 5:
                    continue
                f.write(json.dumps({"did": d, "source": src, "profile": prof[d], "items": items}) + "\n")
                kept += 1
        taken |= {d for _, d in plan}
        print(name, kept, time.strftime("%H:%M:%SZ", time.gmtime()), flush=True)


def account_ids(tok, acct):
    from state import account_state, post_state
    ids = list(tok.ids(account_state(acct)))
    for it in acct["items"][:12]:
        ids += tok.ids(post_state(it))
    return ids


def measure():
    tok = Enc(MD / "tokenizer.json")
    man = json.loads((MD / "manifest.json").read_text())
    data = (MD / "weights.bin").read_bytes()
    T = man["table"]
    V, H = T["V"], T["H"]
    codes = np.frombuffer(data, np.uint8, T["codes"]["length"], T["codes"]["offset"]).reshape(V, H // 2)
    scales = np.frombuffer(data, np.uint8, T["scales"]["length"], T["scales"]["offset"]).reshape(V, -1)
    row = codes.shape[1] + scales.shape[1]

    load = lambda n: [json.loads(l) for l in (DATA / f"pool_{n}.jsonl").open()]
    A, B = load("A"), load("B")
    qs = json.loads((ROOT.parent / "bsky-account-decider" / "run" / "questions.json").read_text())
    always = {tok.cls, tok.sep, tok.mask, tok.tok.token_to_id("[UNK]"), tok.tok.token_to_id("[PAD]")}
    always |= set(tok.ids(json.dumps(qs)))
    tr = examples([c for w in WORKFLOWS for c in load_cases(w, "train")])
    te = examples([c for w in WORKFLOWS for c in load_cases(w, "test")])
    cnt = Counter()
    for a in A:
        cnt.update(account_ids(tok, a))
    for e in tr:
        cnt.update(tok.build(e)[0])
    ranked = [i for i, _ in cnt.most_common() if i not in always]
    B_ids = [(a["source"], account_ids(tok, a)) for a in B]
    td_cases = {}
    for e in te:
        td_cases.setdefault(e["case"], set()).update(tok.build(e)[0])

    full_br = len(brotli.compress(codes.tobytes() + scales.tobytes(), quality=11))
    res = {"V": V, "row_bytes": row, "full_table_bytes": V * row, "full_table_brotli": full_br,
           "pool_A": len(A), "pool_B": len(B), "B_by_source": dict(Counter(s for s, _ in B_ids)),
           "distinct_ids_A_plus_train": len(cnt), "always": len(always), "K": []}
    for K in KS:
        keep = always | set(ranked[: K - len(always)])
        idx = np.array(sorted(keep))
        core = codes[idx].tobytes() + scales[idx].tobytes()
        r = {"K": len(keep), "core_bytes": len(core), "core_brotli": len(brotli.compress(core, quality=11)),
             "id_map_bytes": 2 * V}
        for label, sel in (("B_all", B_ids), ("B_firehose", [x for x in B_ids if x[0] == "firehose"])):
            miss = [len({i for i in ids if i not in keep}) for _, ids in sel]
            occ = sum(1 for _, ids in sel for i in ids if i not in keep) / sum(len(ids) for _, ids in sel)
            r[label] = {"n": len(sel), "share_needing_fetch": round(float(np.mean([m > 0 for m in miss])), 4),
                        "rows_fetched_median": int(np.median(miss)), "rows_fetched_p90": int(np.percentile(miss, 90)),
                        "rows_fetched_max": int(max(miss)), "token_share_missing": round(occ, 5)}
        tm = [len(s - keep) for s in td_cases.values()]
        r["TD_test_cases"] = {"n": len(tm), "share_needing_fetch": round(float(np.mean([m > 0 for m in tm])), 4),
                              "rows_fetched_max": int(max(tm))}
        res["K"].append(r)
        print(json.dumps(r), flush=True)
    (ROOT / "results" / "vocab_fallback.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"collect": collect, "measure": measure}[sys.argv[1]]()
