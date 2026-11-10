"""Round 3: Answer Search's index at 4 bits x 256 folded numbers against the 1 bit x 768 it shipped with.

Usage: SITE=/path/to/oaustegard.github.io python3 -I run/answer_search.py SNAPSHOT_BIN

The harness and metric are answer-search-1bit's (that experiment's run.py), on the live snapshot given:
posts are embedded with the site's Ettin model by refresh.py's own Ettin class, the 768-dim vectors are folded
to 256 with refresh.fold_basis (the same C the builder uses), and each arm is a remex 2.1.0 index scored by
search_adc, the estimate the page's GPU scan computes.

Arms: 768 at 1 bit (what shipped), 768 at 2 bits, 256 folded at 3 and 4 bits (4 is the new index). All are
centred on the snapshot's mean (folded for the 256 arms). Reference: exact fp32 search, which is the same at
768 and 256. Queries: answer-search-1bit's 60 keyword queries, and 200 snippet queries (the first six words of
200 random posts of at least eight words, random.Random(42), from this snapshot). Per query: recall@10 and
@40 of the post ranking against fp32, and how many of the page's 10 quoted sentences (the best sentence of each
of the top 40 posts, ranked by fp32 sentence vectors) are the ones the fp32 ranking quotes. Paired bootstrap
over queries, 5,000 resamples. Writes results/answer_search.json; vectors are cached in out/answer_search/.
"""
import gzip, json, os, random, sys, time
from pathlib import Path

import numpy as np
import remex

HERE = Path(__file__).resolve().parent.parent
EXP = HERE.parent
SITE = Path(os.environ.get("SITE", "/home/user/oaustegard.github.io"))
sys.path.insert(0, str(SITE / "scripts" / "answer-search"))
from refresh import MAGIC, Ettin, fold_basis  # noqa: E402

QPROMPT = "Represent this sentence for searching relevant passages: "
POSTS, PICKS = 40, 10
OUT = HERE / "out" / "answer_search"


def load(path):
    buf = Path(path).read_bytes()
    assert buf[:4] == MAGIC
    hl = int.from_bytes(buf[4:8], "little")
    idx = json.loads(buf[8:8 + hl])
    o, n = idx["sections"]["posts"]
    return idx, json.loads(gzip.decompress(buf[8 + hl + o: 8 + hl + o + n]))["texts"]


def sentences(text):
    """The page's sentence splitter (answers.js sentences()), as answer-search-1bit/run.py has it."""
    import re
    out = []
    for s in re.split(r"(?<=[.!?…])\s+|\n+", text):
        s = s.strip()
        if len(re.findall(r"[^\W\d_]{2,}", s)) >= 4 and not re.fullmatch(r"\S+\.\S+/\S*", s):
            out.append(s)
    return out


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": round(float(x.mean()), 4), "lo": round(float(np.percentile(m, 2.5)), 4), "hi": round(float(np.percentile(m, 97.5)), 4)}


def main():
    idx, texts = load(sys.argv[1])
    model = SITE / "bsky" / "embedders" / "models" / "ettin"
    enc = Ettin(model)
    C, fold = fold_basis(model)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"X_{idx['built'].replace(':', '')}.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        t0 = time.time()
        X = enc.embed(texts)
        np.save(cache, X)
        print(f"embedded {len(X)} posts in {time.time() - t0:.0f} s", flush=True)
    Y = (X.astype(np.float64) @ C).astype(np.float32)
    mu768 = np.array(idx["mu"], np.float32)
    assert len(mu768) == 768, "give the 768-dim snapshot: its mean is the one the 1-bit index used"
    mu256 = (mu768.astype(np.float64) @ C).astype(np.float32)

    t0 = time.time()
    qz = {"768x1": remex.Quantizer(768, bits=1, seed=42, mean=mu768),
          "768x2": remex.Quantizer(768, bits=2, seed=42, mean=mu768),
          "256x3": remex.Quantizer(256, bits=3, seed=42, rotation="none", mean=mu256),
          "256x4": remex.Quantizer(256, bits=4, seed=42, rotation="none", mean=mu256)}
    cv = {k: q.encode(X if k.startswith("768") else Y) for k, q in qz.items()}
    print(f"encoded {len(cv)} arms in {time.time() - t0:.0f} s", flush=True)
    arms = list(qz)
    bytes_per_post = {k: int(k.split("x")[0]) * int(k.split("x")[1]) // 8 + 8 for k in arms}

    kw = json.loads((EXP / "answer-search-1bit" / "queries.json").read_text())["keyword"]
    rng = random.Random(42)
    pool = [i for i, t in enumerate(texts) if len(t.split()) >= 8]
    sets = {"keyword": kw, "snippet": [" ".join(texts[i].split()[:6]) for i in rng.sample(pool, 200)]}

    sent_cache = {}

    def picks(post_ids, qv):
        todo = [i for i in post_ids if i not in sent_cache]
        flat = [(i, s) for i in todo for s in sentences(texts[i])]
        for i in todo:
            sent_cache[i] = ([], [])
        if flat:
            V = enc.embed([s for _, s in flat])
            for (i, s), v in zip(flat, V):
                sent_cache[i][0].append(s)
                sent_cache[i][1].append(v)
        cands = sorted(((float(v @ qv), i, s) for i in post_ids for s, v in zip(*sent_cache[i])), reverse=True)
        seen, out = set(), []
        for _, i, s in cands:
            if i not in seen:
                seen.add(i)
                out.append((i, s))
                if len(out) == PICKS:
                    break
        return out

    res = {"snapshot": {k: idx[k] for k in ("n", "built", "oldest_t", "remex_version")}, "fold_sha256": fold["sha256"],
           "bytes_per_post": bytes_per_post, "sets": {}}
    for name, qs in sets.items():
        Q = enc.embed([QPROMPT + q for q in qs])
        rows = []
        t0 = time.time()
        for q, qv in zip(qs, Q):
            exact = np.argsort(-(X @ qv), kind="stable")[:POSTS]
            ref = set(picks(list(exact), qv))
            qy = (qv.astype(np.float64) @ C).astype(np.float32)
            row = {}
            for k in arms:
                got = qz[k].search_adc(cv[k], qv if k.startswith("768") else qy, k=POSTS)[0]
                row[k] = {"r10": len(set(got[:10]) & set(exact[:10])) / 10, "r40": len(set(got) & set(exact)) / POSTS,
                          "picks": len(set(picks(list(got), qv)) & ref)}
            rows.append(row)
        s = {k: {m: boot([r[k][m] for r in rows]) for m in ("r10", "r40", "picks")} for k in arms}
        for a in ("256x4", "256x3", "768x2"):
            s[f"{a}-768x1"] = {m: boot([r[a][m] - r["768x1"][m] for r in rows]) for m in ("r10", "r40", "picks")}
        s["n_queries"] = len(rows)
        res["sets"][name] = s
        print(name, f"{time.time() - t0:.0f} s", json.dumps({k: {m: v["mean"] for m, v in s[k].items()} for k in arms}), flush=True)
    (HERE / "results" / "answer_search.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
