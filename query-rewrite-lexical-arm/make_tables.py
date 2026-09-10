"""Emits the markdown tables RESULTS.md embeds, straight from results/*.json.

recheck.py asserts the generated text appears verbatim in RESULTS.md, so the
prose and the artifacts cannot drift apart between rebuilds.
"""
import glob, json, os

OUT = os.path.dirname(os.path.abspath(__file__))
RES = f"{OUT}/results"

LABEL = {
    "A2_whole_doc":       "A2  BM25, whole document, full corpus",
    "A1_chunked":         "A1  BM25, 380-word chunks, full corpus",
    "A2s_whole_doc_sub":  "A2s BM25, whole document, 20% subcorpus",
    "A1s_chunked_sub":    "A1s BM25, 180-word chunks, 20% subcorpus",
    "B_dense_sub":        "B   MiniLM-L6-v2 dense, 180-word chunks, 20% subcorpus",
    "C_rrf_A1s_B":        "C   rrf(A1s, B)",
    "D_rewrite_hybrid":   "Dr  rrf of the S4 rewrite's lexical and dense legs",
    "D_rrf_all":          "D   rrf(A1s, B, A1s-rewrite, B-rewrite)",
}
ORDER = list(LABEL)


def arms():
    out = {}
    for p in sorted(glob.glob(f"{RES}/*.json")):
        n = os.path.basename(p)[:-5]
        if n in ("analysis", "paper_numbers") or n.endswith(".smoke"):
            continue
        d = json.load(open(p))
        if "overall" in d:
            out[n] = d
    a = f"{RES}/analysis.json"
    if os.path.exists(a):
        for n, m in json.load(open(a))["arms"].items():
            out.setdefault(n, {"overall": m, "config": {}})
            out[n]["overall"] = m
    return out


def table_main(A):
    rows = ["| arm | HIT@10 | HIT@1 | Recall@10 | MRR@10 | NDCG@10 |",
            "|---|---:|---:|---:|---:|---:|"]
    for n in ORDER:
        if n not in A:
            continue
        o = A[n]["overall"]
        rows.append(f"| {LABEL[n]} | {o['hit10']:.2f} | {o['hit1']:.2f} | "
                    f"{o['recall10']:.2f} | {o['mrr10']:.2f} | {o['ndcg10']:.2f} |")
    return "\n".join(rows)


def table_rescue(an):
    if not an:
        return ""
    rows = ["| miss set | rescued by | misses | rescued | rate |",
            "|---|---|---:|---:|---:|"]
    for k, v in an["rescue"].items():
        if v["rate"] is None:
            continue
        helper, _, missed = k.partition("_rescues_")
        if helper not in LABEL or missed not in LABEL:
            continue
        rows.append(f"| {missed} | {helper} | {v['miss_n']} | {v['rescued']} | {v['rate']}% |")
    return "\n".join(rows)


def table_bytype(A, names):
    have = [n for n in names if n in A and A[n].get("by_type")]
    if not have:
        return ""
    types = sorted(A[have[0]]["by_type"])
    rows = ["| question type | n | " + " | ".join(n for n in have) + " |",
            "|---|---:|" + "---:|" * len(have)]
    for t in types:
        n0 = A[have[0]]["by_type"][t]["n"]
        cells = " | ".join(f"{A[n]['by_type'][t]['hit10']:.2f}" for n in have)
        rows.append(f"| {t} | {n0} | {cells} |")
    return "\n".join(rows)


def table_bootstrap(an):
    if not an or not an.get("paired_bootstrap_vs"):
        return ""
    rows = ["| contrast | delta HIT@10 | 95% CI | p |", "|---|---:|---|---:|"]
    for k, v in an["paired_bootstrap_vs"].items():
        rows.append(f"| {k.replace('_vs_', ' vs ')} | {v['delta']:+.2f} | "
                    f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {v['p']} |")
    return "\n".join(rows)


def render():
    A = arms()
    an = json.load(open(f"{RES}/analysis.json")) if os.path.exists(f"{RES}/analysis.json") else None
    blocks = {
        "MAIN": table_main(A),
        "RESCUE": table_rescue(an),
        "BYTYPE": table_bytype(A, ["A1s_chunked_sub", "B_dense_sub", "C_rrf_A1s_B", "D_rrf_all"]),
        "BOOTSTRAP": table_bootstrap(an),
    }
    return {k: v for k, v in blocks.items() if v}


if __name__ == "__main__":
    import sys
    if "--inject" in sys.argv:
        print(f"injected {inject()} table spans into RESULTS.md")
    else:
        for k, v in render().items():
            print(f"<!-- TABLE:{k} -->")
            print(v)
            print(f"<!-- /TABLE:{k} -->")
            print()


def inject(path="RESULTS.md"):
    """Replace each <!-- TABLE:X --> ... <!-- /TABLE:X --> span with a fresh table."""
    import re
    md = open(path).read()
    n = 0
    for name, block in render().items():
        pat = re.compile(rf"(<!-- TABLE:{name} -->\n).*?(\n<!-- /TABLE:{name} -->)", re.S)
        md, k = pat.subn(lambda m: m.group(1) + block + m.group(2), md)
        n += k
    open(path, "w").write(md)
    return n
