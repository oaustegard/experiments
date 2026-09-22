"""Round 4: gemini-3.5-flash-lite writes tags for 300 memories; round-3 retrieval re-scored.

  python3 flashlite_tags.py tag        # 300 calls through the gateway -> data/flashlite_tags.json
  python3 flashlite_tags.py score      # snap, substitute, retrieve -> results/flashlite.json
"""
import json, re, sys, time, concurrent.futures as cf
from collections import Counter
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from common import DATA, load_fixture

RESULTS = DATA.parent / "results"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "hint"
RAW = DATA / ("flashlite_tags.json" if VARIANT == "hint" else f"flashlite_tags_{VARIANT}.json")
N_DOCS, K_HINT, SNAP = 300, 150, 0.85
PROMPT = """You are writing topic tags for an entry in an engineering memory store.

Write four to seven tags for the entry below. Match the register of the store's existing
tags exactly: lowercase, single words or hyphenated phrases, specific, no explanations.
The store's most used tags, as examples of that register (reuse them when they fit, write
new ones in the same shape when they do not):
{hint}

Output the tags on one line, comma separated, nothing else.

ENTRY:
{entry}"""

PROMPT_CONTEXT = """You are writing topic tags for a new entry in an engineering memory store.

Write four to seven tags for the entry below: lowercase, single words or hyphenated phrases,
no explanations. At least half of the tags must be the specific things the entry is about --
the project, repo, tool, model, paper, person, place or issue it names -- written as they
appear. The rest may be general topics. Prefer a specific name over a generic category.

RELATED ENTRIES ALREADY IN THE STORE (their tags, then a snippet). When one of these tags
fits the new entry, reuse it exactly rather than writing a variant:
{context}

Output the tags on one line, comma separated, nothing else.

NEW ENTRY:
{entry}"""

# Post-hoc variant (not pre-registered): no hint list; asks for the specific names the entry
# is about. Written after the first run showed flash-lite reusing hint tags 62% of the time.
PROMPT_SPECIFIC = """You are writing topic tags for an entry in an engineering memory store.

Write four to seven tags for the entry below: lowercase, single words or hyphenated phrases,
no explanations. At least half of the tags must be the specific things the entry is about --
the project, repo, tool, model, paper, person, place or issue it names -- written as they
appear (e.g. remex, pliny, ettin-32m, muninn-utilities-137, arxiv-2609-01807). The rest may
be general topics. Prefer a specific name over a generic category every time.

Output the tags on one line, comma separated, nothing else.

ENTRY:
{entry}"""


def select_docs(fx, links):
    pos = {m["id"]: i for i, m in enumerate(fx["memories"])}
    rel = {}
    for q, t in links["pairs"]:
        rel.setdefault(q, set()).add(t)
    chosen, queries = set(), []
    for q in sorted(rel, key=lambda q: len(rel[q])):
        need = {q} | rel[q]
        if len(chosen | need) <= N_DOCS:
            chosen |= need; queries.append(q)
    return sorted(chosen), queries, rel, pos


def older_neighbours(fx, links, docs, k=5):
    """Top-k gte-small neighbours among strictly older memories: the write-time recall view."""
    from datetime import datetime
    ids = [m["id"] for m in fx["memories"]]; pos = {m: i for i, m in enumerate(ids)}
    def ts(s):
        try: return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception: return 0.0
    t = np.array([ts(links["t"][m]) for m in ids]); D = np.load(DATA / "dense.npy")
    out = {}
    for d in docs:
        i = pos[d]; s = D @ D[i]; s[t >= t[i]] = -np.inf; s[i] = -np.inf
        out[d] = [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j])]
    return out


def tag():
    from muninn_utils.hypothetical_classifier import _invoke, _MODEL
    fx = load_fixture(); links = json.loads((DATA / "links.json").read_text())
    docs, queries, _, _ = select_docs(fx, links)
    if VARIANT == "context":
        nb = older_neighbours(fx, links, docs); tags_of = {m["id"]: m["tags"] for m in fx["memories"]}
        (DATA / "context_neighbours.json").write_text(json.dumps(nb))
    hint = ", ".join(t for t, _ in Counter(t for m in fx["memories"] for t in m["tags"]).most_common(K_HINT))
    tj = json.loads((DATA / "texts.json").read_text()); tmap = dict(zip(tj["ids"], tj["texts"]))
    done = json.loads(RAW.read_text()) if RAW.exists() else {}
    todo = [d for d in docs if d not in done]
    print(f"{len(docs)} docs ({len(queries)} queries), {len(todo)} to tag", file=sys.stderr, flush=True)
    t0 = time.time()
    def one(d):
        if VARIANT == "context":
            ctx = "\n".join(f"- [{', '.join(tags_of[x])}] {tmap[x][:150].replace(chr(10), ' ')}" for x in nb[d]) or "- (none)"
            pr = PROMPT_CONTEXT.format(context=ctx, entry=tmap[d][:6000])
        else:
            pr = PROMPT.format(hint=hint, entry=tmap[d][:6000]) if VARIANT == "hint" else PROMPT_SPECIFIC.format(entry=tmap[d][:6000])
        r = _invoke(pr, _MODEL, 200) or ""
        return d, [t.strip().strip('"').strip("'").lower() for t in re.split(r"[,\n]", r) if t.strip()][:7]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        for i, (d, tags) in enumerate(ex.map(one, todo)):
            done[d] = tags
            if i % 25 == 0:
                RAW.write_text(json.dumps(done)); print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    RAW.write_text(json.dumps(done))
    print(f"tagged {len(done)} in {time.time()-t0:.0f}s; empty: {sum(1 for v in done.values() if not v)}", file=sys.stderr)


def score():
    import retrieval as R
    from sentence_transformers import SentenceTransformer
    fx = load_fixture(); mems = fx["memories"]; ids = [m["id"] for m in mems]; n = len(ids)
    links = json.loads((DATA / "links.json").read_text())
    docs, queries, rel, pos = select_docs(fx, links)
    raw = json.loads(RAW.read_text())
    vocab_all = sorted({t for m in mems for t in m["tags"]})
    st = SentenceTransformer("thenlper/gte-small", device="cpu")
    Ev = st.encode(vocab_all, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    written = sorted({t for v in raw.values() for t in v})
    Ew = st.encode(written, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    sims = Ew @ Ev.T; vset = set(vocab_all)
    snap = {}
    for i, w in enumerate(written):
        if w in vset: snap[w] = (w, "exact")
        else:
            j = int(np.argmax(sims[i])); snap[w] = (vocab_all[j], "snapped") if sims[i, j] >= SNAP else (w, "new")
    snapped = {d: sorted({snap[t][0] for t in v}) for d, v in raw.items()}
    kinds = Counter(snap[t][1] for v in raw.values() for t in v)
    hint = set(t for t, _ in Counter(t for m in mems for t in m["tags"]).most_common(K_HINT))
    mine = {m["id"]: set(m["tags"]) for m in mems}
    jacc = [len(set(snapped[d]) & mine[d]) / len(set(snapped[d]) | mine[d]) for d in docs if snapped[d] or mine[d]]
    hint_share = np.mean([t in hint for v in raw.values() for t in v])

    def tagmat(tags_of):
        vocab = sorted({t for v in tags_of.values() for t in v}); vi = {t: j for j, t in enumerate(vocab)}
        r, c = zip(*[(pos[d], vi[t]) for d, v in tags_of.items() for t in v])
        return normalize(sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(vocab))))
    qidx = [pos[q] for q in queries]; rels = [{pos[t] for t in rel[q]} for q in queries]
    arms = {}
    base = {m["id"]: list(m["tags"]) for m in mems}
    subs = [("my tags", None), ("flash-lite snapped", snapped), ("flash-lite raw", raw)]
    if VARIANT == "context":
        nb = json.loads((DATA / "context_neighbours.json").read_text())
        inherit = {d: sorted({t for x in nb[d] for t in mine[x]}) or ["__none__"] for d in docs}
        subs.append(("neighbour-tag inheritance (no model)", inherit))
        ctx_tags = {d: {t for x in nb[d] for t in mine[x]} for d in docs}
        out_extra = {"share_written_tags_from_context": float(np.mean([t in ctx_tags[d] for d, v in raw.items() for t in v]))}
    else:
        out_extra = {}
    for name, sub in subs:
        tags_of = dict(base)
        if sub: tags_of.update({d: sub[d] for d in docs})
        M = tagmat(tags_of); arms[name] = R.rank_rows((M[qidx] @ M.T).toarray(), qidx)
    D = np.load(DATA / "dense.npy"); arms["gte-small"] = R.rank_rows(D[qidx] @ D.T, qidx)
    metrics = {k: R.evaluate(v, rels) for k, v in arms.items()}
    m10, mm = R.per_query(arms["my tags"], rels)
    boots = {}
    for k, v in arms.items():
        if k == "my tags": continue
        a10, am = R.per_query(v, rels); boots[k] = {"recall@10": R.boot_diff(a10, m10), "mrr": R.boot_diff(am, mm)}
    out = {"n_docs": len(docs), "n_queries": len(queries), "n_pairs": sum(len(r) for r in rels),
           "tags_per_memory_written": float(np.mean([len(v) for v in raw.values()])),
           "tags_per_memory_mine_on_these": float(np.mean([len(mine[d]) for d in docs])),
           "snap_kinds": dict(kinds), "share_new_after_snap": kinds["new"] / sum(kinds.values()),
           "share_written_tags_in_hint": float(hint_share), "jaccard_snapped_vs_mine": float(np.mean(jacc)),
           "metrics": metrics, "bootstrap_vs_my_tags": boots, **out_extra}
    (RESULTS / ("flashlite.json" if VARIANT == "hint" else f"flashlite_{VARIANT}.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("metrics", "bootstrap_vs_my_tags")}, indent=1))
    print(f"\n{'arm':22s} R@10   R@50   MRR")
    for k, v in metrics.items(): print(f"{k:22s} {v['recall@10']:.3f}  {v['recall@50']:.3f}  {v['mrr']:.3f}")
    print("\nvs my tags, R@10:")
    for k, b in boots.items(): print(f"  {k:20s} {b['recall@10']['mean']:+.3f} [{b['recall@10']['lo']:+.3f}, {b['recall@10']['hi']:+.3f}]")
    ex = [d for d in docs if d in queries][:3]
    for d in ex: print(f"\n{d[:8]} mine: {sorted(mine[d])}\n         flash: {snapped[d]}")


if __name__ == "__main__":
    {"tag": tag, "score": score}[sys.argv[1]]()
