"""Compare a tail implementation's outputs against the remex fixtures.

Usage: python3 -I oracle/check_tail.py FIXTURES OUTDIR [--json]

OUTDIR holds (all little-endian, doc-major):
  packed.u8   [n, 192]  remex byte order
  scale.f32 bias.f32 len.f32 mdot.f32   [n]
  scores.f32  [q, n]
  topk.i32    [q, 10]
  pooled.f32  [pool_b, d]   (optional; pool kernel)
Exit status 0 only if every check passes.
"""
import json
import os
import sys

import numpy as np
from remex.packing import unpack

MARGIN = 1e-5   # coordinates closer than this to a boundary may flip
RTOL = 1e-4
ATOL = 5e-6


def load(path, dtype, shape):
    a = np.fromfile(path, dtype)
    if a.size != int(np.prod(shape)):
        raise SystemExit(f"{path}: {a.size} values, expected {shape}")
    return a.reshape(shape)


def emax(err):
    return float(err.max()) if err.size else 0.0


def close(got, want, rtol=RTOL, atol=ATOL):
    err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    lim = atol + rtol * np.abs(want.astype(np.float64))
    return err <= lim, err


def main():
    fx, out = sys.argv[1], sys.argv[2]
    as_json = "--json" in sys.argv
    p = json.load(open(os.path.join(fx, "params.json")))
    n, q, d, k = p["n_docs"], p["n_queries"], p["d"], p["k"]
    E = lambda f: os.path.join(fx, "expect", f)
    O = lambda f: os.path.join(out, f)
    report, ok = {}, True

    # ---- codes
    want_codes = load(E("codes.u8"), np.uint8, (n, d))
    margin = load(E("margin.f32"), np.float32, (n, d))
    got_packed = load(O("packed.u8"), np.uint8, (n, p["bytes_per_vec"]))
    got_codes = unpack(got_packed.ravel(), p["bits"], n * d).reshape(n, d)
    diff = got_codes != want_codes
    excused = diff & (margin < MARGIN)
    hard = diff & ~excused
    report["codes"] = {"mismatch": int(diff.sum()), "excused": int(excused.sum()),
                       "excusable_coords": int((margin < MARGIN).sum()),
                       "unexcused": int(hard.sum())}
    if hard.any():
        ok = False
        i, j = np.argwhere(hard)[0]
        report["codes"]["first_unexcused"] = {
            "doc": int(i), "coord": int(j), "got": int(got_codes[i, j]),
            "want": int(want_codes[i, j]), "margin": float(margin[i, j])}
    flipped_rows = diff.any(1)

    # ---- per-doc floats vs float64 recomputation from the GPU's own codes
    c = np.asarray(p["centroids"], np.float64)
    Rmu = load(os.path.join(fx, "Rmu.f32"), np.float32, (d,)).astype(np.float64)
    mu = load(os.path.join(fx, "mu.f32"), np.float32, (d,)).astype(np.float64)
    R = load(os.path.join(fx, "R.f32"), np.float32, (d, d)).astype(np.float64)
    want_len = load(E("len.f32"), np.float32, (n,))
    want_mdot = load(E("mdot.f32"), np.float32, (n,))
    v = c[got_codes]
    vlen = np.sqrt((v * v).sum(1))
    scale64 = want_len.astype(np.float64) / (np.maximum(vlen, 1e-12) * p["kappa"])
    bias64 = want_mdot.astype(np.float64) - scale64 * (v @ Rmu)

    for name, want, rtol, atol in [
        ("len", want_len, 1e-5, 1e-6), ("mdot", want_mdot, 1e-5, 1e-6),
        ("scale", scale64, RTOL, ATOL), ("bias", bias64, RTOL, ATOL),
    ]:
        got = load(O(f"{name}.f32"), np.float32, (n,))
        good, err = close(got, want, rtol, atol)
        report[name] = {"max_abs_err": emax(err), "fail": int((~good).sum())}
        if not good.all():
            ok = False
    for name in ("scale", "bias"):
        got = load(O(f"{name}.f32"), np.float32, (n,))
        want = load(E(f"{name}.f32"), np.float32, (n,))
        good, err = close(got[~flipped_rows], want[~flipped_rows])
        report[name]["vs_remex_fail_unflipped_rows"] = int((~good).sum())
        report[name]["rows_with_flips"] = int(flipped_rows.sum())
        if not good.all():
            ok = False

    # ---- scores
    queries = load(os.path.join(fx, "queries_unit.f32"), np.float32, (q, d)).astype(np.float64)
    got_scores = load(O("scores.f32"), np.float32, (q, n))
    qr = queries @ R.T
    s64 = (qr @ v.T) * scale64[None, :] + bias64[None, :] + (queries @ mu)[:, None]
    good, err = close(got_scores, s64)
    report["scores_vs_own_codes"] = {"max_abs_err": emax(err),
                                     "fail": int((~good).sum())}
    if not good.all():
        ok = False
    want_scores = load(E("scores.f32"), np.float32, (q, n))
    good, err = close(got_scores[:, ~flipped_rows], want_scores[:, ~flipped_rows])
    report["scores_vs_remex_unflipped"] = {"max_abs_err": emax(err),
                                           "fail": int((~good).sum())}
    if not good.all():
        ok = False

    # ---- top-k
    got_top = load(O("topk.i32"), np.int32, (q, k))
    want_top = load(E("topk.i32"), np.int32, (q, k))
    # The returned list must be the top k of the float64 own-code scores up to
    # ties: descending within tol, and nothing left out scores above its tail.
    bad, differs_from_remex = [], 0
    for qi in range(q):
        s = s64[qi]
        g = got_top[qi]
        tol = 2 * (ATOL + RTOL * np.abs(s[g]).max())
        desc = np.all(np.diff(s[g]) <= tol)
        rest = np.ones(n, bool)
        rest[g] = False
        complete = s[rest].max() <= s[g].min() + tol
        if not (desc and complete and len(set(g.tolist())) == k):
            bad.append(qi)
        if not np.array_equal(g, want_top[qi]):
            differs_from_remex += 1
    report["topk"] = {"bad_queries": bad,
                      "queries_ordered_differently_from_remex": differs_from_remex}
    if bad:
        ok = False

    # ---- pool (optional)
    if os.path.exists(O("pooled.f32")):
        want = load(E("pooled.f32"), np.float32, (p["pool_b"], d))
        got = load(O("pooled.f32"), np.float32, (p["pool_b"], d))
        good, err = close(got, want, 1e-5, 1e-5)
        report["pool"] = {"max_abs_err": emax(err), "fail": int((~good).sum())}
        if not good.all():
            ok = False
    else:
        report["pool"] = "absent"

    report["PASS"] = bool(ok)
    print(json.dumps(report, indent=None if as_json else 1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
