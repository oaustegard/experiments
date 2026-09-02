#!/usr/bin/env python3
"""Four ranking arms over one fixed candidate pool.

All arms share ONE encoder (bge-small-en-v1.5). A feature is a short natural
language string NAMED by an LLM and EVALUATED by that same encoder:

    Phi[j, f] = cos( embed_doc(candidate_j), embed_query(feature_f) )

That is the point of the design. If a coordinate-based arm beats flat
similarity, the gain cannot be attributed to a stronger model, a bigger index,
or extra text -- the encoder, the pool and the document text are identical
across arms. The only thing that changes is the basis the ranking is computed
in. This is the narrowest test of the SLT objection (memory a8b97f70) that we
can actually run: the objection says valuable structure lies on a direction
the flat space does not expose as an axis, and a named feature IS such a
direction.

Oracle: binary, target-only. read(doc) = 1 iff doc is the case's target.
Everything else returns 0. This simulates the expensive expert read and
removes verifier noise as a confound. Metric is reads-to-hit.
"""
import json
import numpy as np
import embed


def doc_text(p, mode):
    t = (p.get("title") or "").strip()
    if mode == "title":
        return t
    a = (p.get("abstract") or "").strip()
    return (t + ". " + a).strip() if a else t


def build_phi(pool, features, mode):
    D = embed.encode([doc_text(p, mode) for p in pool])
    F = embed.encode(features, is_query=True)
    return D @ F.T, D


def bayes_ridge(Phi, idx, y, alpha=1.0, noise=0.25):
    """Bayesian linear regression posterior over observed rows -> (mu, sd) for all rows."""
    X = Phi[idx]
    d = Phi.shape[1]
    A = (X.T @ X) / noise + alpha * np.eye(d)
    Ainv = np.linalg.inv(A)
    w = Ainv @ (X.T @ np.asarray(y, float)) / noise
    mu = Phi @ w
    sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Phi, Ainv, Phi), 1e-12))
    return mu, sd


def run_sequential(Phi_rounds, target_idx, budget=60, batch=5, beta=1.5, grow=True):
    """Phi_rounds: list of feature-matrices, one per available round (growing).

    grow=True  -> arm B: on a plateau, widen to the next matrix in the list.
    grow=False -> arm C: stay on Phi_rounds[0] forever.
    Returns (reads_to_hit or None, trace).
    """
    n = Phi_rounds[0].shape[0]
    Phi = Phi_rounds[0]
    stage = 0
    read, y = [], []
    trace = []
    # cold start: no labels yet, so rank by mean similarity across the named axes
    order = np.argsort(-Phi.mean(axis=1))
    pending = [int(i) for i in order]
    reads = 0
    while reads < budget and pending:
        if read:
            mu, sd = bayes_ridge(Phi, read, y)
            ucb = mu + beta * sd
            pending.sort(key=lambda i: -ucb[i])
        take = pending[:batch]
        pending = pending[batch:]
        for i in take:
            reads += 1
            hit = (i == target_idx)
            read.append(i)
            y.append(1.0 if hit else 0.0)
            if hit:
                trace.append({"reads": reads, "event": "HIT", "stage": stage})
                return reads, trace
        trace.append({"reads": reads, "event": "batch_all_negative", "stage": stage,
                      "n_features": Phi.shape[1]})
        # plateau: every read this round came back negative
        if grow and stage + 1 < len(Phi_rounds):
            stage += 1
            Phi = Phi_rounds[stage]
            trace.append({"reads": reads, "event": "GROW", "stage": stage,
                          "n_features": Phi.shape[1]})
    return None, trace


def rank_of(sim, target_idx):
    return int((np.argsort(-sim) == target_idx).nonzero()[0][0]) + 1
