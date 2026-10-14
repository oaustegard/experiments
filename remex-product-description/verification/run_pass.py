#!/usr/bin/env python3
"""Automated verification pass for verification/persistence.md.

Usage: python3 verification/run_pass.py [--json results.json]

Runs every checklist item that a script can observe against the installed remex
and prints `ID  pass|fail  note`. A `fail` means the product did not do what the
document says; for items marked (suspected bug) the document already says the
behavior is wrong, so `pass` there confirms the defect. Exits 1 if any item fails.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import remex
from remex import CompressedVectors, Quantizer

ITEMS: dict[str, Callable[[Path], tuple[bool, str]]] = {}


def item(item_id: str):
    def wrap(fn):
        ITEMS[item_id] = fn
        return fn
    return wrap


def data(n: int = 200, d: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def refuses(fn: Callable[[], object], fragment: str) -> tuple[bool, str]:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the type is part of what is recorded
        msg = f"{type(e).__name__}: {str(e)[:90]}"
        return fragment in str(e) or fragment == type(e).__name__, msg
    return False, "returned without raising"


def mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)))


# --- persistence/npz-files.md -------------------------------------------------

@item("NPZ-01")
def npz_roundtrip(tmp: Path):
    X, q = data(), Quantizer(64, 4)
    c = q.encode(X)
    c.save(tmp / "a.npz")
    lo = CompressedVectors.load(tmp / "a.npz")
    i1, s1 = q.search(c, X[0], k=10)
    i2, s2 = q.search(lo, X[0], k=10)
    ok = (np.array_equal(c.indices, lo.indices) and np.array_equal(c.norms, lo.norms)
          and np.array_equal(i1, i2) and np.allclose(s1, s2))
    return ok, "codes, norms, top-10 ids and scores identical"


@item("NPZ-02")
def npz_suffix_appended(tmp: Path):
    Quantizer(64, 4).encode(data()).save(str(tmp / "index"))
    return sorted(p.name for p in tmp.iterdir()) == ["index.npz"], str(sorted(p.name for p in tmp.iterdir()))


@item("NPZ-03")
def npz_load_without_suffix(tmp: Path):
    Quantizer(64, 4).encode(data()).save(str(tmp / "index"))
    return refuses(lambda: CompressedVectors.load(str(tmp / "index")), "FileNotFoundError")


@item("NPZ-04")
def npz_suffix_case(tmp: Path):
    Quantizer(64, 4).encode(data()).save(str(tmp / "INDEX.NPZ"))
    names = sorted(p.name for p in tmp.iterdir())
    return names == ["INDEX.NPZ.npz"], str(names)


@item("NPZ-05")
def npz_missing_dir(tmp: Path):
    ok, note = refuses(lambda: Quantizer(64, 4).encode(data()).save(str(tmp / "nodir" / "x.npz")),
                       "FileNotFoundError")
    return ok and not any(tmp.iterdir()), note


@item("NPZ-06")
def npz_foreign_archive(tmp: Path):
    np.savez(tmp / "other.npz", a=np.arange(3))
    return refuses(lambda: CompressedVectors.load(tmp / "other.npz"), "KeyError")


@item("NPZ-07")
def npz_unreadable(tmp: Path):
    good = tmp / "good.npz"
    Quantizer(64, 4).encode(data()).save(good)
    raw = good.read_bytes()
    (tmp / "empty.npz").write_bytes(b"")
    (tmp / "half.npz").write_bytes(raw[: len(raw) // 2])
    (tmp / "junk.npz").write_bytes(b"not a zip at all")
    cases = [refuses(lambda: CompressedVectors.load(tmp / "empty.npz"), "No data left in file"),
             refuses(lambda: CompressedVectors.load(tmp / "half.npz"), "BadZipFile"),
             refuses(lambda: CompressedVectors.load(tmp / "junk.npz"), "pickled")]
    return all(ok for ok, _ in cases), " | ".join(n for _, n in cases)


@item("NPZ-08")
def npz_kill_during_overwrite(tmp: Path):
    target = tmp / "big.npz"
    Quantizer(384, 8).encode(data(300_000, 384)).save(target)
    before = target.stat().st_size
    child = (
        "import numpy as np, sys\nfrom remex import Quantizer\n"
        "c = Quantizer(384, 8).encode(np.random.default_rng(1).standard_normal((300000, 384)).astype(np.float32))\n"
        f"open({str(tmp / 'ready')!r}, 'w').close()\n"
        f"c.save({str(target)!r})\n")
    p = subprocess.Popen([sys.executable, "-c", child])
    while not (tmp / "ready").exists():
        time.sleep(0.01)
    time.sleep(0.15)
    p.send_signal(signal.SIGKILL)
    p.wait()
    after = target.stat().st_size
    ok, note = refuses(lambda: CompressedVectors.load(target), "")
    return ok, f"{before} bytes before, {after} after; load: {note}"


@item("NPZ-09")
def npz_keys(tmp: Path):
    Quantizer(64, 4, seed=7).encode(data()).save(tmp / "a.npz")
    keys = sorted(np.load(tmp / "a.npz").files)
    return keys == ["bits", "d", "n", "norms", "packed_indices", "rotation"], str(keys)


@item("NPZ-10")
def npz_scalar(tmp: Path):
    q = Quantizer(64, 4, normalize=False, rotation="none")
    q.encode(data()).save(tmp / "s.npz")
    lo = CompressedVectors.load(tmp / "s.npz")
    return "norms" not in np.load(tmp / "s.npz").files and lo.norms is None, "no norms entry; loads with norms None"


@item("NPZ-11")
def npz_centred(tmp: Path):
    X = data()
    q = Quantizer(64, 4, mean=X.mean(0))
    c = q.encode(X)
    c.save(tmp / "c.npz")
    lo = CompressedVectors.load(tmp / "c.npz")
    keys = set(np.load(tmp / "c.npz").files)
    return {"mean", "mean_dots"} <= keys and np.allclose(q.decode(lo), q.decode(c)), "mean, mean_dots written; decode equal"


@item("NPZ-12")
def npz_legacy_rotation(tmp: Path):
    c = Quantizer(64, 4).encode(data())
    c.save(tmp / "a.npz")
    fields = dict(np.load(tmp / "a.npz"))
    fields.pop("rotation")
    np.savez_compressed(tmp / "legacy.npz", **fields)
    lo = CompressedVectors.load(tmp / "legacy.npz")
    ok, note = refuses(lambda: Quantizer(64, 4).decode(lo), "rotation mismatch")
    return ok and lo.rotation == "haar", f"loaded as {lo.rotation!r}; {note}"


@item("NPZ-13")
def npz_pre2_centred(tmp: Path):
    X = data()
    q = Quantizer(64, 4, mean=X.mean(0))
    q.encode(X).save(tmp / "c.npz")
    fields = dict(np.load(tmp / "c.npz"))
    fields.pop("mean_dots")
    np.savez_compressed(tmp / "c1.npz", **fields)
    lo = CompressedVectors.load(tmp / "c1.npz")
    return refuses(lambda: q.decode(lo), "mean_dots")


@item("NPZ-14")
def npz_no_cache_after_load(tmp: Path):
    X, q = data(), Quantizer(64, 4)
    c = q.encode(X)
    q.search(c, X[0], k=3)
    c.save(tmp / "a.npz")
    lo = CompressedVectors.load(tmp / "a.npz")
    return c._x_hat_rot is not None and lo._x_hat_rot is None, "source had a cache, loaded container has none"


@item("NPZ-15")
def npz_empty(tmp: Path):
    Quantizer(64, 4).encode(data()[:0]).save(tmp / "e.npz")
    n = CompressedVectors.load(tmp / "e.npz").n
    return n == 0, f"n={n}"


@item("NPZ-16")
def npz_unknown_rotation(tmp: Path):
    c = Quantizer(64, 4).encode(data())
    c.rotation = "bogus"
    c.save(tmp / "b.npz")
    return refuses(lambda: CompressedVectors.load(tmp / "b.npz"), "unknown rotation")


@item("NPZ-17")
def npz_path_kinds(tmp: Path):
    c = Quantizer(64, 4).encode(data())
    buf = io.BytesIO()
    c.save(buf)
    buf.seek(0)
    c.save(Path(tmp / "p.npz"))
    return CompressedVectors.load(buf).n == 200 and CompressedVectors.load(Path(tmp / "p.npz")).n == 200, "BytesIO and Path both round-trip"


@item("NPZ-18")
def npz_small_file_size(tmp: Path):
    c = Quantizer(64, 4).encode(data(50))
    c.save(tmp / "a.npz")
    size = (tmp / "a.npz").stat().st_size
    return size > c.nbytes, f"nbytes {c.nbytes}, file {size}"


@item("NPZ-19")
def npz_env_no_effect(tmp: Path):
    script = ("import numpy as np, sys\nfrom remex import Quantizer, CompressedVectors\n"
              "c = CompressedVectors.load(sys.argv[1])\n"
              "c.save(sys.argv[2])\n")
    Quantizer(64, 4).encode(data()).save(tmp / "src.npz")
    for name, env in (("plain", {}), ("nonative", {"REMEX_NO_NATIVE": "1"})):
        subprocess.run([sys.executable, "-c", script, str(tmp / "src.npz"), str(tmp / f"{name}.npz")],
                       check=True, env={**os.environ, **env})
    a, b = (dict(np.load(tmp / f"{n}.npz")) for n in ("plain", "nonative"))
    same = a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    return same, "entries identical with and without REMEX_NO_NATIVE"


# --- foundations/the-encoding-contract.md -------------------------------------

def _paths(q: Quantizer, c, X: np.ndarray, fragment: str) -> tuple[bool, str]:
    results = [refuses(lambda: q.decode(c), fragment),
               refuses(lambda: q.search(c, X[0], k=3), fragment),
               refuses(lambda: q.search_adc(c, X[0], k=3), fragment)]
    return all(ok for ok, _ in results), results[0][1]


@item("CONTRACT-01")
def contract_rotation(tmp: Path):
    X = data()
    return _paths(Quantizer(64, 4, rotation="haar"), Quantizer(64, 4).encode(X), X, "rotation mismatch")


@item("CONTRACT-02")
def contract_mode(tmp: Path):
    X = data()
    return _paths(Quantizer(64, 4, normalize=False), Quantizer(64, 4).encode(X), X, "mode mismatch")


@item("CONTRACT-03")
def contract_bits(tmp: Path):
    X = data()
    return _paths(Quantizer(64, 8), Quantizer(64, 4).encode(X), X, "bits mismatch")


@item("CONTRACT-04")
def contract_mean(tmp: Path):
    X = data()
    c_plain = Quantizer(64, 4).encode(X)
    c_centred = Quantizer(64, 4, mean=X.mean(0)).encode(X)
    a = _paths(Quantizer(64, 4, mean=X.mean(0)), c_plain, X, "mean mismatch")
    b = _paths(Quantizer(64, 4, mean=X.mean(0) + 0.1), c_centred, X, "mean mismatch")
    return a[0] and b[0], f"{a[1]} | {b[1]}"


@item("CONTRACT-05")
def contract_seed_silent(tmp: Path):
    X = data()
    c = Quantizer(64, 4, seed=1).encode(X)
    try:
        wrong = Quantizer(64, 4, seed=2).decode(c)
    except Exception as e:  # noqa: BLE001
        return False, f"raised {type(e).__name__}"
    right = Quantizer(64, 4, seed=1).decode(c)
    cw, cr = mean_cos(wrong, X), mean_cos(right, X)
    return abs(cw) < 0.1 and cr > 0.9, f"no error; mean cosine {cw:.3f} wrong seed vs {cr:.3f} right seed"


@item("CONTRACT-06")
def contract_scale_silent(tmp: Path):
    X = data()
    c = Quantizer(64, 4, normalize=False, scale=2.0, rotation="none").encode(X)
    try:
        wrong = Quantizer(64, 4, normalize=False, scale=1.0, rotation="none").decode(c)
    except Exception as e:  # noqa: BLE001
        return False, f"raised {type(e).__name__}"
    right = Quantizer(64, 4, normalize=False, scale=2.0, rotation="none").decode(c)
    ratio = float(np.abs(right).max() / np.abs(wrong).max())
    return abs(ratio - 2.0) < 1e-3, f"no error; decoded values off by a factor of {ratio:.3f}"


@item("CONTRACT-07")
def contract_d_unguarded(tmp: Path):
    X = data()
    notes, silent = [], False
    for rot, norm in (("rht", True), ("haar", True), ("none", False)):
        c = Quantizer(64, 4, rotation=rot, normalize=norm).encode(X)
        try:
            out = Quantizer(32, 4, rotation=rot, normalize=norm).decode(c)
            notes.append(f"{rot}: returned {out.shape}")
            silent = silent or rot == "none"
        except Exception as e:  # noqa: BLE001
            notes.append(f"{rot}: {type(e).__name__}")
    return silent, "; ".join(notes)


@item("CONTRACT-08")
def contract_construction_refusals(tmp: Path):
    checks = [refuses(lambda: Quantizer(64, 5), "negligible benefit"),
              refuses(lambda: Quantizer(63, 4, rotation="rht"), "odd"),
              refuses(lambda: Quantizer(64, 4, scale=2.0), "only meaningful with normalize=False")]
    return all(ok for ok, _ in checks), " | ".join(n for _, n in checks)


@item("CONTRACT-09")
def contract_default_rotation(tmp: Path):
    a, b = Quantizer(64, 4).rotation, Quantizer(63, 4).rotation
    return (a, b) == ("rht", "haar"), f"d=64 -> {a}, d=63 -> {b}"


@item("CONTRACT-10")
def contract_renorm_not_guarded(tmp: Path):
    c = Quantizer(64, 4).encode(data())
    a, b = Quantizer(64, 4, renorm=False).decode(c), Quantizer(64, 4).decode(c)
    return not np.allclose(a, b), "renorm=False reads the codes without error and returns different vectors"


@item("CONTRACT-11")
def contract_refusal_builds_no_cache(tmp: Path):
    X = data()
    c = Quantizer(64, 4).encode(X)
    refuses(lambda: Quantizer(64, 4, rotation="haar").search(c, X[0], k=3), "")
    return c._x_hat_rot is None, "no decode cache after a refused search"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write results to this file")
    args = ap.parse_args()
    results = {}
    for item_id, fn in ITEMS.items():
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ok, note = fn(Path(tmp))
            except Exception as e:  # noqa: BLE001 - a crashed item is a blocked item
                ok, note = None, f"blocked: {type(e).__name__}: {e}"
        results[item_id] = {"result": {True: "pass", False: "fail", None: "blocked"}[ok], "note": note}
        print(f"{item_id:12} {results[item_id]['result']:8} {note}")
    meta = {"remex": remex.__version__, "python": sys.version.split()[0], "numpy": np.__version__}
    if args.json:
        Path(args.json).write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")
    bad = [k for k, v in results.items() if v["result"] != "pass"]
    print(f"{len(results)} items, {len(results) - len(bad)} pass, {len(bad)} not pass: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
