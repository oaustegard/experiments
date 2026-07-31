"""Formal Concept Analysis core + geometric (half-space) realization.

Shared foundation for the Lattice Representation Hypothesis experiments.

Notation follows Ganter & Wille, *Formal Concept Analysis: Mathematical
Foundations* (1999).

A formal context is K = (G, M, I) with G objects, M attributes, and
I subset of G x M the incidence ("object g has attribute m").

Derivation operators:
    A'  = {m in M : g I m for all g in A}      (A subset of G)
    B'  = {g in G : g I m for all m in B}      (B subset of M)

A *formal concept* is a pair (A, B) with A' = B and B' = A.
A is the *extent*, B the *intent*.

Concept lattice operations (Ganter & Wille Thm 3):
    (A1,B1) /\ (A2,B2) = (A1 & A2,        (B1 | B2)'')     [meet]
    (A1,B1) \/ (A2,B2) = ((A1 | A2)'',    B1 & B2 )        [join]

Note the asymmetry that motivates this whole experiment: the meet's extent
is a bare intersection, while the join's extent needs a closure ('').
That closure is exactly where a half-space geometry has to over-approximate.

Sets are represented as numpy boolean masks throughout, so ' and '' are
matrix ops and everything vectorizes.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Core context + derivation operators
# --------------------------------------------------------------------------


class Context:
    """A formal context backed by a boolean incidence matrix.

    Parameters
    ----------
    I : (n_objects, n_attributes) bool array
        ``I[g, m]`` is True iff object ``g`` has attribute ``m``.
    """

    def __init__(self, I: np.ndarray):
        I = np.asarray(I, dtype=bool)
        if I.ndim != 2:
            raise ValueError(f"incidence must be 2-D, got shape {I.shape}")
        self.I = I
        self.n_obj, self.n_att = I.shape

    # -- derivation operators ---------------------------------------------

    def attrs_of(self, A: np.ndarray) -> np.ndarray:
        """A' — attributes shared by *every* object in mask ``A``.

        The empty object set derives all of M (vacuous truth), which is what
        ``np.all`` over an empty axis already gives us.
        """
        return np.all(self.I[A], axis=0) if A.any() else np.ones(self.n_att, bool)

    def objs_of(self, B: np.ndarray) -> np.ndarray:
        """B' — objects carrying *every* attribute in mask ``B``.

        The empty attribute set derives all of G, again by vacuous truth.
        """
        return np.all(self.I[:, B], axis=1) if B.any() else np.ones(self.n_obj, bool)

    # -- closures ----------------------------------------------------------

    def close_objs(self, A: np.ndarray) -> np.ndarray:
        """A'' — the smallest concept extent containing ``A``.

        This is the operation the concept-lattice join needs and that a
        half-space geometry realizes by intersecting the surviving
        half-spaces. Everything ``A''`` contains beyond ``A`` is overshoot.
        """
        return self.objs_of(self.attrs_of(A))

    def close_attrs(self, B: np.ndarray) -> np.ndarray:
        """B'' — the smallest concept intent containing ``B``."""
        return self.attrs_of(self.objs_of(B))

    # -- concept enumeration ----------------------------------------------

    def concepts(self, max_concepts: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """All formal concepts, via Ganter's NextClosure over intents.

        Returns a list of ``(extent_mask, intent_mask)`` in lectic order.
        NextClosure enumerates each closed set exactly once with no
        duplicate check, which is what keeps this tractable.
        """
        n = self.n_att
        out: list[tuple[np.ndarray, np.ndarray]] = []

        B = self.close_attrs(np.zeros(n, bool))  # bottom intent
        while True:
            out.append((self.objs_of(B), B.copy()))
            if max_concepts is not None and len(out) >= max_concepts:
                break
            nxt = self._next_closure(B)
            if nxt is None:
                break
            B = nxt
        return out

    def _next_closure(self, B: np.ndarray) -> np.ndarray | None:
        """Ganter's NextClosure successor of intent ``B`` in lectic order."""
        n = self.n_att
        for i in range(n - 1, -1, -1):
            if B[i]:
                B = B.copy()
                B[i] = False
                continue
            cand = B.copy()
            cand[i] = True
            cand[i + 1 :] = False
            closed = self.close_attrs(cand)
            # lectic test: the closure may not have added anything below i
            if not closed[:i][~B[:i]].any():
                return closed
        return None


# --------------------------------------------------------------------------
# Lattice operations, with the quantity we actually care about
# --------------------------------------------------------------------------


def meet_extent(ctx: Context, A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Extent of the lattice meet: a bare intersection, no closure needed.

    If ``A1`` and ``A2`` are genuine concept extents then ``A1 & A2`` is
    already closed, so this is exact by construction. ``assert_meet_closed``
    is the calibration gate that checks we did not fool ourselves.
    """
    return A1 & A2


def join_extent(ctx: Context, A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Extent of the lattice join: ``(A1 | A2)''`` — the closure of the union."""
    return ctx.close_objs(A1 | A2)


def join_overshoot(ctx: Context, A1: np.ndarray, A2: np.ndarray) -> dict:
    """Quantify how far the lattice join exceeds the plain set union.

    ``phantoms`` are objects the geometry places inside the join that belong
    to neither input extent — the concrete price of closing a union under
    half-space intersection.

    Returns a dict with:
        union      : |A1 | A2|
        closed     : |(A1 | A2)''|
        phantoms   : |closure \\ union|
        overshoot  : phantoms / closed   (0 when the union is already closed)
    """
    union = A1 | A2
    closed = ctx.close_objs(union)
    n_union = int(union.sum())
    n_closed = int(closed.sum())
    n_phantom = int((closed & ~union).sum())
    return {
        "union": n_union,
        "closed": n_closed,
        "phantoms": n_phantom,
        "overshoot": n_phantom / n_closed if n_closed else 0.0,
    }


# --------------------------------------------------------------------------
# Geometric realization: the Lattice Representation Hypothesis's model
# --------------------------------------------------------------------------


def realize(E: np.ndarray, V: np.ndarray, tau: np.ndarray) -> Context:
    """Build the context a half-space geometry induces.

    Object ``g`` has attribute ``m`` iff ``<v_m, e_g> > tau_m`` — the
    linear-direction-plus-threshold model the Lattice Representation
    Hypothesis posits.

    Parameters
    ----------
    E   : (n_obj, d)  object embeddings
    V   : (n_att, d)  attribute directions
    tau : (n_att,)    separating thresholds
    """
    return Context((E @ V.T) > tau[None, :])


def coherence(V: np.ndarray) -> float:
    """Mutual coherence: max |cos| between distinct attribute directions.

    0 means mutually orthogonal. The paper's canonical form assumes the
    directions are linearly independent, and coherence is the continuous
    dial that measures how badly that assumption is strained.
    """
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = np.abs(Vn @ Vn.T)
    np.fill_diagonal(C, 0.0)
    return float(C.max()) if C.size > 1 else 0.0


# --------------------------------------------------------------------------
# Calibration gates (METHODS.md principle 2: two-sided, not "found nothing")
# --------------------------------------------------------------------------


def assert_meet_closed(ctx: Context, concepts: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """KNOWN-GOOD side: intersections of concept extents must already be closed.

    This is a theorem, so any violation means the FCA core is wrong rather
    than that we found something. Raises on failure; returns the number of
    pairs checked.
    """
    checked = 0
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            A = concepts[i][0] & concepts[j][0]
            if not np.array_equal(ctx.close_objs(A), A):
                raise AssertionError(
                    f"meet of concepts {i},{j} is not closed — FCA core is broken"
                )
            checked += 1
    return checked


def assert_join_overshoots(ctx: Context, concepts: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """KNOWN-BAD side: at least one join must strictly exceed its union.

    A context where every union is already closed would make the whole
    experiment vacuous, so we check that the phenomenon under study is
    actually present. Raises if no overshoot exists anywhere; returns the
    number of strictly-overshooting pairs.
    """
    n_over = 0
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if join_overshoot(ctx, concepts[i][0], concepts[j][0])["phantoms"] > 0:
                n_over += 1
    if n_over == 0:
        raise AssertionError(
            "no join overshoots anywhere — context is degenerate for this study"
        )
    return n_over
