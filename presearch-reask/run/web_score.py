"""Score round 2 (results/web_raw.jsonl): WEB vs PSWEB, all questions and leak-free ones.

    python3 run/web_score.py     # writes results/web_scores.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import best, boot, em, f1, mean  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main():
    qs = {q["id"]: q for q in map(json.loads, (HERE / "data" / "eval.jsonl").open())}
    recs, errors = {}, {}
    for r in map(json.loads, (HERE / "results" / "web_raw.jsonl").open()):
        if "error" in r:
            errors[r["arm"]] = errors.get(r["arm"], 0) + 1
            continue
        recs[(r["arm"], r["id"])] = r
    round1 = {(r["arm"], r["id"]): r for r in map(json.loads, (HERE / "results" / "raw.jsonl").open())}
    ids = [i for i in qs if ("WEB", i) in recs and ("PSWEB", i) in recs]
    leaked = {i for i in ids if recs[("WEB", i)]["leak"] or recs[("PSWEB", i)]["leak"]
              or recs[("PSWEB", i)].get("presearch_leak")}
    F = {(a, i): best(f1, recs[(a, i)]["answer"], qs[i]["answers"]) for a in ("WEB", "PSWEB") for i in ids}
    E = {(a, i): best(em, recs[(a, i)]["answer"], qs[i]["answers"]) for a in ("WEB", "PSWEB") for i in ids}
    splits = {"all": ids, "no_leak": [i for i in ids if i not in leaked],
              "2hop": [i for i in ids if qs[i]["hops"] == "2hop"],
              "3-4hop": [i for i in ids if qs[i]["hops"] != "2hop"]}
    out = {"n_paired": len(ids), "errors_dropped": errors, "leaked": len(leaked),
           "presearch_leaks": sum(bool(recs[("PSWEB", i)].get("presearch_leak")) for i in ids),
           "agent_leaks": {a: sum(recs[(a, i)]["leak"] for i in ids) for a in ("WEB", "PSWEB")}, "arms": {}}
    for a in ("WEB", "PSWEB"):
        rs = [recs[(a, i)] for i in ids]
        out["arms"][a] = {
            **{s: {"n": len(m), "f1": mean([F[(a, i)] for i in m]), "em": mean([E[(a, i)] for i in m])}
               for s, m in splits.items()},
            "searches_per_q": mean([len(r["searches"]) for r in rs]),
            "fetches_per_q": mean([len(r["fetches"]) for r in rs]),
            "no_search_share": mean([not r["searches"] and not r["fetches"] for r in rs]),
            "no_answer": sum(r["answer"] is None for r in rs),
            "cost_per_q": mean([r["cost"] for r in rs]),
            "wall_s_per_q": mean([r["wall_s"] for r in rs]),
            "extended_mode_searches": sum(m == "extended" for r in rs for m in r["search_modes"]),
        }
    out["diff"] = {s: {"diff": mean([F[("PSWEB", i)] - F[("WEB", i)] for i in m]),
                       "ci": boot([F[("PSWEB", i)] - F[("WEB", i)] for i in m])} for s, m in splits.items() if m}
    # the same questions in round 1 (BM25 over the closed set), for reference
    out["round1_same_questions"] = {a: mean([best(f1, round1[(a, i)]["answer"], qs[i]["answers"]) for i in ids])
                                    for a in ("CB", "AG3", "PSAG3")}
    (HERE / "results" / "web_scores.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
