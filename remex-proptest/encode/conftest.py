"""Mutation hook for the calibration runs.

``ENCODE_MUTANT=<name>`` plants one plausible defect into remex *before* the
harness imports it; no variable, no mutation. The mutants are source-level
edits of the real functions (inspect.getsource + one string replacement +
re-exec in the module's namespace), so the planted bug is a one-token change of
the shipped code rather than a hand-written look-alike. The remex checkout
itself is never modified. See run_mutants.py and mutation_results.md.
"""
import hashlib
import inspect
import os
import textwrap

MUTANT = os.environ.get("ENCODE_MUTANT", "")


def _patch(module, name, old, new, cls=None):
    holder = getattr(module, cls) if cls else module
    fn = holder.__dict__[name] if cls else getattr(module, name)
    src = textwrap.dedent(inspect.getsource(fn))
    assert src.count(old) == 1, f"{name}: {old!r} occurs {src.count(old)} times"
    ns = dict(module.__dict__)
    exec(compile(src.replace(old, new), f"<mutant {MUTANT}>", "exec"), ns)
    setattr(holder, name, ns[name])
    return ns[name]


def _patch_c(old, new):
    from remex import _native
    assert _native._C_SOURCE.count(old) == 1
    assert not _native._loaded, "kernel already loaded; cannot mutate the C source"
    _native._C_SOURCE = _native._C_SOURCE.replace(old, new)
    _native._SOURCE_HASH = hashlib.sha256(
        (_native._C_SOURCE + " ".join(_native._FLAGS)).encode()).hexdigest()[:16]


def apply(name):
    import remex.codebook as cb
    import remex.core as core
    import remex.rotation as rot
    if name == "float64_midpoints":
        # boundaries from the unrounded float64 centroids (the pre-fix behaviour)
        f = _patch(cb, "lloyd_max_codebook", "c64 = c32.astype(np.float64)", "c64 = centroids")
        core.lloyd_max_codebook = f
    elif name == "rht_one_round":
        _patch(rot, "rht_rounds", "return max(2, math.ceil(math.log(d) / math.log(B)))",
               "return 1 if B == d else max(2, math.ceil(math.log(d) / math.log(B)))")
    elif name == "kernel_ties_go_up":
        _patch_c("v > b[c - 1]", "v >= b[c - 1]")
    elif name == "numpy_assign_side_right":
        _patch(cb, "assign_codes", "return np.searchsorted(boundaries, X_rot).astype(np.uint8)",
               "return np.searchsorted(boundaries, X_rot, side='right').astype(np.uint8)")
    elif name == "nested_unweighted_mean":
        _patch(cb, "nested_codebooks", "np.average(\n                    centroids_max[start:end], weights=group_probs\n                )",
               "centroids_max[start:end].mean()")
    elif name == "length_floor_1e-3":
        _patch(core, "encode", "np.maximum(lengths, 1e-8)", "np.maximum(lengths, 1e-3)", cls="Quantizer")
    elif name == "thread_chunk_drops_last_row":
        _patch(cb, "assign_codes", "min(n, lo + step), bp, bits", "min(n, lo + step) - 1, bp, bits")
    elif name == "scalar_encode_float32":
        _patch(core, "_encode_scalar", "X = np.asarray(X, dtype=np.float64)",
               "X = np.asarray(X, dtype=np.float32)", cls="Quantizer")
    elif name == "numpy_rht_sign_index":
        _patch(rot, "_apply_numpy", "Z[:, perm] = Y * s", "Z[:, perm] = Y * s[perm]", cls="RHTOperator")
    elif name == "lloyd_30_iterations":
        f = _patch(cb, "lloyd_max_codebook", "n_iter: int = 300", "n_iter: int = 30")
        core.lloyd_max_codebook = f
    elif name == "encode_block_drops_row":
        _patch(core, "encode", "hi = min(n, lo + rows)",
               "hi = min(n, lo + rows) - (1 if lo + rows < n else 0)", cls="Quantizer")
    elif name == "haar_no_sign_correction":
        _patch(rot, "haar_rotation", "if sign_flip.any():", "if False:")
    elif name == "rht_plan_draw_order":
        _patch(rot, "rht_plan", "perms.append(rng.permutation(d))\n        signs.append(rng.choice(np.array([-1.0, 1.0], np.float32), size=d))",
               "signs.append(rng.choice(np.array([-1.0, 1.0], np.float32), size=d))\n        perms.append(rng.permutation(d))")
    elif name == "scalar_boundary_float32_window":
        # scalar codes through float32: sub-float32 distinctions vanish
        _patch(cb, "assign_codes", "f64 = X_rot.dtype == np.float64", "f64 = False")
    elif name == "linspace_init_asymmetric":
        f = _patch(cb, "lloyd_max_codebook", "centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)",
                   "centroids = np.linspace(-3 * sigma, 3.0001 * sigma, n_levels)")
        core.lloyd_max_codebook = f
    elif name == "anisotropy_uses_max_norm":
        import remex
        remex.anisotropy = _patch(core, "anisotropy", "mean_len = float(np.sqrt(np.sum(X64 ** 2, axis=1)).mean())",
                                  "mean_len = float(np.sqrt(np.sum(X64 ** 2, axis=1)).max())")
    elif name == "encode_ratio_drops_zero_guard":
        # the in-encode ratio loses the norm_total > 0 guard: all-zero corpora
        # divide 0/0 and warn about a "common direction"
        _patch(core, "encode", "if judge and norm_total > 0.0:", "if judge:", cls="Quantizer")
    else:
        raise SystemExit(f"unknown mutant {name!r}")


if MUTANT:
    apply(MUTANT)
