"""1-bit vs 2-bit remex index for bsky/answer-search, against fp32 exact search. See PLAN.md.

    python3 run.py embed      # fp32 Ettin vectors of the 100,000 snapshot posts -> data/X.npy (~3 min CPU)
    python3 run.py eval       # every arm, every query -> results/*.json

Needs numpy, onnxruntime, tokenizers, remex==2.1.0 and a checkout of oaustegard/oaustegard.github.io (for
the Ettin model and the builder's Ettin class) at SITE (env, default /home/user/oaustegard.github.io).
"""
import gzip, json, os, random, sys, time
from pathlib import Path

import numpy as np
import remex

HERE = Path(__file__).resolve().parent
SITE = Path(os.environ.get("SITE", "/home/user/oaustegard.github.io"))
sys.path.insert(0, str(SITE / "scripts" / "answer-search"))
from refresh import Ettin, MAGIC  # noqa: E402

QPROMPT = "Represent this sentence for searching relevant passages: "
POSTS, PICKS = 40, 10


def load_snapshot():
    buf = (HERE / "data" / "snapshot.bin").read_bytes()
    assert buf[:4] == MAGIC
    hl = int.from_bytes(buf[4:8], "little")
    idx = json.loads(buf[8:8 + hl])
    sec = lambda k: buf[8 + hl + idx["sections"][k][0]: 8 + hl + idx["sections"][k][0] + idx["sections"][k][1]]
    c = json.loads(gzip.decompress(sec("posts")))
    return idx, c["texts"]


def sentences(text):
    """The page's sentence splitter (answers.js sentences())."""
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
    return [round(float(x.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main(cmd):
    idx, texts = load_snapshot()
    enc = Ettin(SITE / "bsky" / "embedders" / "models" / "ettin")
    if cmd == "embed":
        t0 = time.time()
        X = enc.embed(texts)
        np.save(HERE / "data" / "X.npy", X)
        print(f"embedded {len(X)} in {time.time() - t0:.0f} s")
        return
    X = np.load(HERE / "data" / "X.npy")
    mu = np.array(idx["mu"], np.float32)
    (HERE / "results" / "fixture.json").write_text(json.dumps(
        {k: idx[k] for k in ("n", "built", "newest", "oldest_t", "remex_version", "model")}, indent=1))

    kw = json.loads((HERE / "queries.json").read_text())["keyword"]
    rng = random.Random(42)
    pool = [i for i, t in enumerate(texts) if len(t.split()) >= 8]
    snip = [" ".join(texts[i].split()[:6]) for i in rng.sample(pool, 200)]
    sets = {"keyword": kw, "snippet": snip}

    q2, q1 = remex.Quantizer(768, bits=2, seed=42, mean=mu), remex.Quantizer(768, bits=1, seed=42, mean=mu)
    t0 = time.time()
    c2, c1 = q2.encode(X), q1.encode(X)
    print(f"encoded both widths in {time.time() - t0:.0f} s")
    arms = {
        "2bit": lambda q, k: q2.search_adc(c2, q, k=k)[0],
        "1bit": lambda q, k: q1.search_adc(c1, q, k=k)[0],
        "2bit@1": lambda q, k: q2.search_adc(c2, q, k=k, precision=1)[0],
    }

    sent_cache = {}

    def picks(post_ids, qv):
        cands = []
        todo = [i for i in post_ids if i not in sent_cache]
        flat = [(i, s) for i in todo for s in sentences(texts[i])]
        for i in todo:
            sent_cache[i] = ([], [])                     # a post may have no sentence worth quoting
        if flat:
            V = enc.embed([s for _, s in flat])
            for (i, s), v in zip(flat, V):
                sent_cache[i][0].append(s)
                sent_cache[i][1].append(v)
        for i in post_ids:
            for s, v in zip(*sent_cache[i]):
                cands.append((float(v @ qv), i, s))
        cands.sort(reverse=True)
        seen, out = set(), []
        for sc, i, s in cands:
            if i in seen:
                continue
            seen.add(i)
            out.append((i, s))
            if len(out) == PICKS:
                break
        return out

    summary = {}
    for name, qs in sets.items():
        Q = enc.embed([QPROMPT + q for q in qs])
        rows = []
        t0 = time.time()
        for qi, (q, qv) in enumerate(zip(qs, Q)):
            exact = np.argsort(-(X @ qv))[:POSTS]
            ref = picks(list(exact), qv)
            row = {"q": q}
            for arm, fn in arms.items():
                got = fn(qv, POSTS)
                p = picks(list(got), qv)
                row[arm] = {"r10": len(set(got[:10]) & set(exact[:10])) / 10,
                            "r40": len(set(got) & set(exact)) / POSTS,
                            "picks": len(set(p) & set(ref))}
            rows.append(row)
            if qi % 50 == 49:
                print(f"  {name}: {qi + 1}/{len(qs)} ({time.time() - t0:.0f} s)", flush=True)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(rows, indent=0))
        s = {}
        for arm in arms:
            s[arm] = {m: boot([r[arm][m] for r in rows]) for m in ("r10", "r40", "picks")}
        for a, b in (("1bit", "2bit"), ("2bit@1", "2bit"), ("2bit@1", "1bit")):
            s[f"{a}-{b}"] = {m: boot([r[a][m] - r[b][m] for r in rows]) for m in ("r10", "r40", "picks")}
        summary[name] = s
        print(name, json.dumps(s, indent=1), flush=True)
    summary["bytes_per_post"] = {"2bit": 192 + 8, "1bit": 96 + 8}
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
