"""General ring-of-length-2i construction generalizing Schrijver's (D1,u1).

Derived by decomposing the verified D1 (i=3, see calibration/schrijver_d1.py)
into its 3-fold rotational-symmetry orbits and writing the per-orbit arc
pattern parametrically in i. NOT transcribed from Figure 8 of the survey
(that figure was not read pixel-by-pixel the way Figure 6 was) -- this is
a reconstruction from D1's own verified structure, checked here against
the survey's stated claim (section 7, page 9) that the construction is a
counterexample for i=5,7,9,... and NOT a counterexample for i=2,4,6,8,...
That parity check is the calibration gate (gate 3); if it fails, this
construction is not the one Figure 8 depicts and must not be trusted for
search.

Orbit structure (indices mod i):
  source_k  (in-degree 0)   -- k = 0..i-1
  plain_k   (both in & out) -- target of source_k's "long" active arc
  sink_k    (out-degree 0)  -- target of source_k's "short" active arc
  other_k   (both in & out) -- source_k's null-arc target

Per k, active arcs (u=1):
  source_k -> plain_k
  source_k -> sink_k
  other_k  -> sink_{k+1}

Per k, null arcs (u=0):
  source_k -> other_k
  source_k -> plain_{k-1}
  plain_k  -> sink_{k+1}
  other_k  -> sink_k

At i=3 this reproduces D1 exactly (isomorphic, verified in
calibration/test_calibration.py gate1 x gate3 cross-check).
"""

from verifier.digraph import Digraph


def build_ring_counterexample(i: int) -> Digraph:
    if i < 2:
        raise ValueError("i must be >= 2")

    vertices = []
    for k in range(i):
        vertices += [f"source_{k}", f"plain_{k}", f"sink_{k}", f"other_{k}"]

    arcs = []
    lab = 0

    def L():
        nonlocal lab
        lab += 1
        return f"e{lab}"

    for k in range(i):
        kp1 = (k + 1) % i
        km1 = (k - 1) % i
        arcs.append((f"source_{k}", f"plain_{k}", L(), 1))
        arcs.append((f"source_{k}", f"sink_{k}", L(), 1))
        arcs.append((f"other_{k}", f"sink_{kp1}", L(), 1))
        arcs.append((f"source_{k}", f"other_{k}", L(), 0))
        arcs.append((f"source_{k}", f"plain_{km1}", L(), 0))
        arcs.append((f"plain_{k}", f"sink_{kp1}", L(), 0))
        arcs.append((f"other_{k}", f"sink_{k}", L(), 0))

    return Digraph(vertices, arcs)


if __name__ == "__main__":
    from verifier.packing import nu

    # brute-force closed_sets() is 2^(4i); i=5 -> n=20 -> 2^20 ~1M, fine.
    # i=6+ (n>=24) needs a smarter tau/dicut algorithm -- not attempted here.
    for i in range(2, 6):
        D = build_ring_counterexample(i)
        dag_ok = D.is_dag()
        tau, _ = D.tau()
        n = nu(D)
        print(f"i={i}: DAG={dag_ok} tau={tau} nu={n} counterexample={n < tau}")
