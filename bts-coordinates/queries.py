#!/usr/bin/env python3
"""Query texts for the retro-eval cases.

P1_PRE vs P1_POST is the load-bearing distinction of this experiment.

Issue #179's P1 test case is SUMMARY.md Lemma 5 *as written after the
reduction was found* — it contains the strings "totally unimodular" and
"lindisc". Those are the target's own vocabulary. Any keyword or dense method
scores well on it, so it measures nothing about cross-field search. The
campaign did not have that text until phase 7; phases 1-6 had only flow
vocabulary.

P1_PRE reconstructs the pre-reduction statement from SUMMARY.md sections 1
and 2 with every discrepancy-side term removed. It is the query the campaign
actually held while failing to find Doerr.

P1_POST is kept verbatim as a leak control: a method that cannot hit on
P1_POST is broken, and the P1_POST - P1_PRE gap measures how much of the
difficulty is vocabulary.
"""

P1_PRE = """An acyclic single-source unsplittable-flow instance: a DAG with source s
and terminals t_1..t_k with demands d_i > 0, d_max = max_i d_i, together with a
fractional flow x decomposed into per-terminal path flows. A routing P assigns each
terminal entirely to one s->t_i path; f^P(a) is the induced load on arc a. Question:
is there always a routing P with |f^P(a) - x(a)| <= d_max on every arc?

Restrict to instances where each terminal has exactly two s->t_i paths and the path
set is closed (each terminal's declared path set is all of its s->t_i graph paths).
Then sharing between a terminal's two paths is prefix-only and the paths never
reconverge, so the instance is an arborescence rooted at s whose terminals are sinks
of in-degree 2. For terminal i let u_i and v_i be its two attachment nodes. Write
z_i = 1 if terminal i routes via u_i, and w_i for the fractional weight on that path.
For a tree arc a with head-subtree S_a, the deviation of the routing on a is the
signed sum over terminals whose attachment nodes straddle a, weighted by demand and
by (z_i - w_i).

We want to know the worst-case deviation, over all fractional weights w in [0,1]^k,
of the best integral choice z in {0,1}^k, measured in the maximum norm over arcs and
in units of d_max. Is it bounded by d_max? What is the sharp constant for small k?"""

P1_POST = """Let T be the tree, and for terminal i let u_i, v_i be its two attachment
nodes ("chords"). Write z_i = 1 if i routes via u_i, and w_i for the fractional weight
on that path. For a tree arc a with head-subtree S_a put
C[a,i] = [u_i in S_a] - [v_i in S_a] in {-1,0,+1}.
Lemma 5. f^P(a) - x(a) = sum_i C[a,i]*d_i*(z_i - w_i) for every tree arc a, and
terminal arcs have deviation d_i w_i or d_i(1-w_i), hence never exceed d_max.
C[a,i] != 0 exactly when a lies on the tree path between u_i and v_i, signed by side
- so C is the network matrix of T with respect to the terminal chords, and is
therefore totally unimodular. Consequently, with
lindisc(A) = max_{w in [0,1]^k} min_{z in {0,1}^k} ||A(z-w)||_inf,
Conjecture 1.3 restricted to 2-path instances is equivalent to
lindisc(C*diag(d)) <= d_max for all network matrices C and demands d."""

P2 = """In a path-closed instance with exactly two paths per terminal, sharing is
prefix-only and paths never reconverge; consequently the instance is an arborescence
rooted at s whose terminals are sinks of in-degree 2. Conversely every such "out-tree
instance" is path-closed by construction."""

N3 = """Conjecture: for a totally unimodular matrix arising from a "braided" path
system - a terminal permitted to revisit an already-covered chord twice before
terminating - is the resulting linear discrepancy bounded by (2/3)*d_max in the limit
of unbounded chord multiplicity?"""

CASES = {
    "P1_PRE":  {"text": P1_PRE,  "target": "Doerr 2004, Linear Discrepancy of Totally Unimodular Matrices",
                "target_doi": "10.1007/s00493-004-0007-x", "on_arxiv": False, "kind": "positive"},
    "P1_POST": {"text": P1_POST, "target": "Doerr 2004, Linear Discrepancy of Totally Unimodular Matrices",
                "target_doi": "10.1007/s00493-004-0007-x", "on_arxiv": False, "kind": "positive-leakcontrol"},
    "P2":      {"text": P2,      "target": "MSW25, Integer and Unsplittable Multiflows in Series-Parallel Digraphs",
                "target_arxiv": "2412.05182", "on_arxiv": True, "kind": "positive"},
    "N3":      {"text": N3,      "target": None, "on_arxiv": False, "kind": "negative"},
}

if __name__ == "__main__":
    for k, v in CASES.items():
        print(f"{k:9s} {v['kind']:22s} target={v['target']}")
        print(f"          chars={len(v['text'])}")
