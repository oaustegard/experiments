"""Decoders over an item-position score matrix, and ranking metrics.

Every decoder maps M (N x K, higher = better affinity of item i for rank j)
to a ranking: an array of item indices, best first.

D1 is the paper's (arXiv:2609.01807 Eq. 1, solved by the Hungarian algorithm).
D2-D5 ignore the assignment constraint in different ways.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _tail_order(M, assigned):
    """Items with no assigned position, ordered by their best score."""
    unassigned = np.setdiff1d(np.arange(M.shape[0]), assigned, assume_unique=False)
    if unassigned.size == 0:
        return unassigned
    return unassigned[np.argsort(-M[unassigned].max(axis=1), kind="stable")]


def d1_hungarian(M):
    """Optimal bipartite assignment: argmax_pi sum_i M[i, pi(i)]."""
    rows, cols = linear_sum_assignment(M, maximize=True)
    order = rows[np.argsort(cols, kind="stable")]
    return np.concatenate([order, _tail_order(M, rows)])


def d2_row_greedy(M):
    """Each item takes its best free position; items in descending confidence.

    The 'repair' strategy the paper says its decoder replaces.
    """
    _, K = M.shape
    slot = np.full(K, -1, dtype=np.int64)
    order_items = np.argsort(-M.max(axis=1), kind="stable")
    placed = []
    for i in order_items:
        prefs = np.argsort(-M[i], kind="stable")
        for j in prefs:
            if slot[j] < 0:
                slot[j] = i
                placed.append(i)
                break
    filled = slot[slot >= 0]
    return np.concatenate([filled, _tail_order(M, np.asarray(placed, dtype=np.int64))])


def d3_col_greedy(M):
    """Each position takes its best free item, positions in order."""
    N, K = M.shape
    free = np.ones(N, dtype=bool)
    out = []
    for j in range(K):
        if not free.any():
            break
        col = np.where(free, M[:, j], -np.inf)
        i = int(np.argmax(col))
        out.append(i)
        free[i] = False
    out = np.asarray(out, dtype=np.int64)
    return np.concatenate([out, _tail_order(M, out)])


def d4_expected_position(M):
    """Sort ascending by the softmax-expected rank position of each item."""
    z = M - M.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    exp_pos = p @ np.arange(M.shape[1], dtype=np.float64)
    return np.argsort(exp_pos, kind="stable")


def d5_col0(M):
    """Sort descending by affinity for rank 1 alone."""
    return np.argsort(-M[:, 0], kind="stable")


DECODERS = {
    "D1_hungarian": d1_hungarian,
    "D2_row_greedy": d2_row_greedy,
    "D3_col_greedy": d3_col_greedy,
    "D4_expected_pos": d4_expected_position,
    "D5_col0": d5_col0,
}


# ---------------------------------------------------------------- diagnostics

def row_argmax_collisions(M):
    """How many items lose a tie for their single preferred position.

    Returns (n_colliding_items, n_distinct_positions_claimed).
    If this is 0, D2's repair never fires and D1 == D2 by construction.
    """
    pref = np.argmax(M, axis=1)
    _, counts = np.unique(pref, return_counts=True)
    return int((counts - 1)[counts > 1].sum()), int(counts.size)


def rank1_energy(M):
    """Share of Frobenius energy in M's top singular value.

    If this is 1, M[i,j] = a_i * b_j exactly. With b monotone decreasing in j
    the rearrangement inequality makes the optimal assignment equal to sorting
    by a_i, so the Hungarian solve is a fixed-cost identity on a sort.
    """
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


# ------------------------------------------------------------------- metrics

def dcg(labels_in_order, k):
    g = (2.0 ** labels_in_order[:k]) - 1.0
    d = 1.0 / np.log2(np.arange(g.size) + 2.0)
    return float((g * d).sum())


def ndcg(labels, ranking, k):
    ideal = dcg(np.sort(labels)[::-1], k)
    if ideal <= 0:
        return np.nan
    return dcg(labels[ranking], k) / ideal


def recall_at_k(labels, ranking, k, thresh=1.0):
    rel = labels >= thresh
    n_rel = int(rel.sum())
    if n_rel == 0:
        return np.nan
    return float(rel[ranking[:k]].sum()) / n_rel


def mrr(labels, ranking, thresh=1.0):
    hit = np.nonzero(labels[ranking] >= thresh)[0]
    return 1.0 / (hit[0] + 1.0) if hit.size else 0.0


def kendall_tau_vs(ranking, reference):
    """Kendall tau between two rankings given as item-index orders."""
    from scipy.stats import kendalltau
    pos_a = np.empty(ranking.size, dtype=np.int64)
    pos_a[ranking] = np.arange(ranking.size)
    pos_b = np.empty(reference.size, dtype=np.int64)
    pos_b[reference] = np.arange(reference.size)
    return float(kendalltau(pos_a, pos_b).statistic)
