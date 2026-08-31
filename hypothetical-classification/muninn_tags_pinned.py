"""Both tag arms, with the sampled rows PINNED to disk.

The earlier scripts re-drew `random.sample` from a live Turso corpus each run. Writing
one memory into the store between a run and its recheck shifts the draw, so the saved
hallucinations get zipped against a different 250 rows and every number moves. The rows
are the fixture; they belong on disk next to the generations.
"""
import sys, json, random, collections, re, concurrent.futures as cf
sys.path.insert(0, '/mnt/skills/user'); sys.path.insert(0, '/home/user/muninn-utilities')
import numpy as np
from muninn_utils.memory_tfidf import MemoryIndex
from muninn_utils.hypothetical_classifier import Vocabulary, _invoke, _MODEL

MIN_USES, N_EVAL, SEED, K = 3, 250, 20260831, 5

NOVELTY = """Invent {k} novel, never-seen-before topic tags for the engineering-memory entry below.
Tags are single words or hyphenated phrases, like: ccotw / correction / atproto /
paper-insight / architecture / perch / session-log.

Invent freely - do not try to recall real tags, do not explain. Output {k} tags on one
line, comma separated, nothing else.

ENTRY:
{entry}"""

REGISTER = """You are writing entries for a topic-tag vocabulary used by an engineering memory store.

Write the {k} tags this vocabulary WOULD file the entry below under. Match the register
exactly: single words or hyphenated phrases, lowercase, plain - like ccotw / correction /
atproto / paper-insight / architecture / perch / session-log.

Do not worry about whether a tag already exists. Write the obvious ones. Do not invent
novel or creative wording, do not explain. Output {k} tags on one line, comma separated.

ENTRY:
{entry}"""


def tags_of(m):
    t = m.get("tags") or []
    if isinstance(t, str): t = [x.strip() for x in t.split(",") if x.strip()]
    return set(t)


def gen(prompt_tpl, summaries):
    def one(s):
        r = _invoke(prompt_tpl.format(k=K, entry=s[:1500]), _MODEL, 300) or ""
        return [t.strip().strip('"').lower() for t in re.split(r"[,\n]", r) if t.strip()][:K]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        return list(ex.map(one, summaries))


if __name__ == "__main__":
    idx = MemoryIndex(); idx.build()
    counts = collections.Counter(t for m in idx.meta for t in tags_of(m))
    labels = sorted(t for t, c in counts.items() if c >= MIN_USES)
    VS = set(labels)
    rows = [(idx.summaries[i], sorted(tags_of(idx.meta[i]) & VS)) for i in range(len(idx.ids))
            if len(idx.summaries[i]) > 300 and tags_of(idx.meta[i]) & VS]
    random.seed(SEED); rows = random.sample(rows, min(N_EVAL, len(rows)))
    S = [s for s, _ in rows]; G = [set(g) for _, g in rows]
    fixture = {"vocab": labels, "rows": [{"summary": s, "gold": g} for s, g in rows],
               "corpus_size_at_capture": len(idx.ids), "min_uses": MIN_USES, "seed": SEED}
    print(f"vocab={len(labels)} rows={len(rows)} corpus={len(idx.ids)} "
          f"mean_gold={np.mean([len(g) for g in G]):.1f}", flush=True)

    fixture["novelty"] = gen(NOVELTY, S);  print("novelty done", flush=True)
    fixture["register"] = gen(REGISTER, S); print("register done", flush=True)
    json.dump(fixture, open("muninn_tags_fixture.json", "w"), indent=1)

    print(f"\n{'arm':44} {'@1':>6} {'@3':>6} {'@5':>6}")
    print("-" * 66)
    res = {}
    for bk in ("tfidf", "minilm"):
        V = Vocabulary(labels, backend=bk)
        ctl = V.snap(S, k=5)
        per = {}
        for arm in ("novelty", "register"):
            flat = [t for tags in fixture[arm] for t in tags]
            snapped = V.snap(flat, k=1)
            out, c = [], 0
            for tags in fixture[arm]:
                out.append([snapped[c + i][0][0] for i in range(len(tags))]); c += len(tags)
            per[arm] = out
        def hc(k): return float(np.mean([any(l in g for l, _ in r[:k]) for r, g in zip(ctl, G)]))
        def hh(arm, k): return float(np.mean([any(l in g for l in p[:k]) for p, g in zip(per[arm], G)]))
        def hu(k): return float(np.mean([any(l in g for l, _ in r[:k]) or any(l in g for l in p[:k])
                                         for r, p, g in zip(ctl, per["register"], G)]))
        for nm, fn in ((f"{bk}: summary -> tag  [no LLM control]", hc),
                       (f"{bk}: 5 tags, novelty prompt", lambda k: hh("novelty", k)),
                       (f"{bk}: 5 tags, register prompt", lambda k: hh("register", k)),
                       (f"{bk}: control + register, interleaved", hu)):
            vals = [fn(k) for k in (1, 3, 5)]
            res[nm] = vals
            print(f"{nm:44} {vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:6.3f}")
    json.dump(res, open("muninn_tags_pinned_results.json", "w"), indent=1)
