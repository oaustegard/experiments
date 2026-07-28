"""Structure-guided random search over small DAGs for a Woodall counterexample.

Per issue #163 generator priority list, item 3 ("structure-guided random").
Items 1 (null-arc resolutions of D1/D2/D3) and 2 (Williams' catalog) are
NOT implemented this pass -- see README.md "Deferred work". This is a
lightweight, honestly-bounded random search, not a systematic campaign.

Hard pruning filters applied (per issue, all proven-good classes excluded):
  - DAG (WLOG, contract SCCs -- random_dag() only ever emits DAGs by
    construction, arcs always go from earlier to later in a random order)
  - NOT source-sink-connected (every counterexample must have some
    source/sink pair with no directed path -- Schrijver/Feofiloff-Younger)
  - NOT single-source or single-sink
  - tau >= 3 (tau <= 2 is proven safe)

No isomorphism dedup (no nauty/pynauty available in this environment this
pass -- flagged as deferred, not silently skipped).
"""

import random
import sys
import time

sys.path.insert(0, "..")

from verifier.digraph import Digraph
from verifier.packing import nu


def random_dag(n_vertices, p_edge, seed):
    rng = random.Random(seed)
    order = list(range(n_vertices))
    rng.shuffle(order)
    arcs = []
    lab = 0
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < p_edge:
                arcs.append((order[i], order[j], f"e{lab}", 1))
                lab += 1
    return Digraph(list(range(n_vertices)), arcs)


def is_source_sink_connected(D):
    sources = [v for v in D.vertices if not D.in_arcs(v)]
    sinks = [v for v in D.vertices if not D.out_arcs(v)]
    if not sources or not sinks:
        return False
    adj = {v: set(v2 for _, v2, *_ in D.out_arcs(v)) for v in D.vertices}

    def reach(v):
        seen = {v}
        stack = [v]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    for s in sources:
        r = reach(s)
        if not all(t in r for t in sinks):
            return False
    return True


def run_search(n_vertices, p_edge, trials, time_budget_s, log_path):
    stats = {
        "trials": 0,
        "not_dag_or_empty": 0,
        "source_sink_connected": 0,
        "single_source_or_sink": 0,
        "tau_lt_3": 0,
        "checked": 0,
        "counterexamples": [],
    }
    start = time.time()
    seed = 0
    with open(log_path, "w") as log:
        log.write(f"# search: n={n_vertices} p_edge={p_edge} trials={trials}\n")
        while stats["trials"] < trials:
            if time.time() - start > time_budget_s:
                log.write(f"# time budget ({time_budget_s}s) exhausted, stopping early\n")
                break
            seed += 1
            stats["trials"] += 1
            D = random_dag(n_vertices, p_edge, seed)
            if not D.is_dag() or not D.arcs:
                stats["not_dag_or_empty"] += 1
                continue
            sources = [v for v in D.vertices if not D.in_arcs(v)]
            sinks = [v for v in D.vertices if not D.out_arcs(v)]
            if len(sources) <= 1 or len(sinks) <= 1:
                stats["single_source_or_sink"] += 1
                continue
            if is_source_sink_connected(D):
                stats["source_sink_connected"] += 1
                continue
            tau, _ = D.tau()
            if tau < 3:
                stats["tau_lt_3"] += 1
                continue
            stats["checked"] += 1
            n = nu(D)
            log.write(f"seed={seed} n={n_vertices} tau={tau} nu={n}\n")
            if n < tau:
                stats["counterexamples"].append((seed, tau, n, list(D.arcs)))
                log.write(f"  *** CANDIDATE COUNTEREXAMPLE *** seed={seed} tau={tau} nu={n}\n")
                log.write(f"  arcs={D.arcs}\n")
        log.write(f"# final stats: {stats}\n")
    return stats


if __name__ == "__main__":
    import json

    all_stats = []
    for n_vertices in (7, 8, 9, 10):
        for p_edge in (0.25, 0.4):
            log_path = f"logs/search_n{n_vertices}_p{p_edge}.log"
            stats = run_search(
                n_vertices=n_vertices,
                p_edge=p_edge,
                trials=2000,
                time_budget_s=25,
                log_path=log_path,
            )
            summary = {k: v for k, v in stats.items() if k != "counterexamples"}
            summary["n_vertices"] = n_vertices
            summary["p_edge"] = p_edge
            summary["num_candidates"] = len(stats["counterexamples"])
            all_stats.append(summary)
            print(json.dumps(summary))

    with open("logs/search_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)
