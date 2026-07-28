"""Schrijver's counterexample (D1, u1) to the Edmonds-Giles conjecture.

Transcribed directly from Figure 6 of Feofiloff's survey
(https://www.ime.usp.br/~pf/dijoins/woodall/survey1-en.pdf), pages 8-9,
by rendering the PDF page to a 600dpi PNG and reading vertex positions and
arrowhead directions off the image -- NOT from any secondary/memorized
source, per issue #163's explicit warning not to trust a transcribed arc
list without reproducing nu and tau.

Vertex naming: the paper's Figure 6 labels only the 9 ACTIVE (capacity-1)
arcs (a..i); the 12 vertices themselves are unlabeled dots/circles/squares.
We name vertices by their position in the figure:
  TR, L, BR  = the 3 source vertices (circle marker, in-degree 0)
  P, S, U    = the 3 sink vertices (square marker, out-degree 0)
  TL, Q, R, M, W, BL = the 6 plain (both in- and out-degree > 0) vertices

Degree-sum cross-check performed by hand before writing this file:
  sum(out-degree) = sum(in-degree) = 21 = 9 active + 12 null arcs. Matches.

CALIBRATION NOTE (2026-07-23): the first transcription attempt read the
W->Q dashed arc wrong (should be W->U) -- it produced tau=2 and all four
"special join" checks passed, but only 3 of the paper's stated 4 critical
cuts existed and nu(D1,u1) came out as 2, contradicting Fact 7.1 (nu=1).
Diagnosed by independently deriving the graph's 3-fold rotational symmetry
(source vertices TR/L/BR and their orbits under L->TR->BR) by hand from the
correctly-read arcs, which predicted W->U where the image had been read as
W->Q; a targeted re-crop of that one region confirmed the correction, and
retargeting n12 to W->U restored tau=2, exactly 4 critical cuts, and
nu(D1,u1)=1. This is exactly the failure mode issue #163 warned about --
the fix came from cross-checking derived structure (symmetry, critical-cut
count) against the paper's stated facts, not from trusting the first read.
"""

from verifier.digraph import Digraph

VERTICES = ["TR", "TL", "L", "BR", "BL", "R", "P", "Q", "U", "S", "M", "W"]

# (source, target, label, capacity). Active (solid, u=1) arcs use the
# paper's own letters a..i; null (dashed, u=0) arcs are numbered n1..n12.
ARCS = [
    # active arcs (capacity 1) -- the letters a..i as printed in Figure 6
    ("TR", "R", "a", 1),
    ("TR", "P", "b", 1),
    ("M", "P", "c", 1),
    ("L", "TL", "d", 1),
    ("L", "S", "e", 1),
    ("W", "S", "f", 1),
    ("BR", "BL", "g", 1),
    ("BR", "U", "h", 1),
    ("Q", "U", "i", 1),
    # null arcs (capacity 0), dashed in the figure
    ("TR", "TL", "n1", 0),
    ("TL", "P", "n2", 0),
    ("TR", "Q", "n3", 0),
    ("Q", "P", "n4", 0),
    ("R", "U", "n5", 0),
    ("L", "M", "n6", 0),
    ("L", "BL", "n7", 0),
    ("M", "S", "n8", 0),
    ("BL", "S", "n9", 0),
    ("BR", "W", "n10", 0),
    ("BR", "R", "n11", 0),
    ("W", "U", "n12", 0),
]


def build() -> Digraph:
    return Digraph(list(VERTICES), list(ARCS))


# The four "special joins" from section 7.1 of the survey (the fractional
# packing of size 2), given directly in the text -- an independent check
# on the transcription: if these aren't all valid dijoins under our arc
# list, the transcription is wrong.
SPECIAL_JOINS = {
    "J1": {"a", "c", "d", "f", "h"},
    "J2": {"d", "f", "g", "i", "b"},
    "J3": {"g", "i", "a", "c", "e"},
    "J4": {"b", "h", "e"},
}

ACTIVE_ARCS = {"a", "b", "c", "d", "e", "f", "g", "h", "i"}


if __name__ == "__main__":
    D = build()
    print("DAG:", D.is_dag())
    print("n =", D.n, " |A| =", len(D.arcs))

    tau, witness = D.tau()
    print("tau =", tau, " witness S =", sorted(witness))

    for name, J in SPECIAL_JOINS.items():
        ok1 = D.is_dijoin(J)
        ok2 = D.contract_and_check_strong(J)
        print(f"{name} = {sorted(J)}  is_dijoin={ok1}  contract_strong={ok2}")

    print("B1 (all active arcs) is_dijoin:", D.is_dijoin(ACTIVE_ARCS))
