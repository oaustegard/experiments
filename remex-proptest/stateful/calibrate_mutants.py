"""Mutation calibration for the stateful harness.

Each mutant is a small source-level change to remex, applied at runtime by recompiling
one function (the remex checkout is never edited).  A mutant is *caught* when the
named machine raises under it.  Runs use ``derandomize=True`` (so a result is
reproducible), no example database, and no shrinking (we only need detection and the
number of examples it took).

    python3 calibrate_mutants.py [--examples N] [--steps N] [only-substring ...]

Writes results/mutation_results.json and results/mutation_results.md.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest.mock as mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("REMEX_PROP_NOSHRINK", "1")
sys.path.insert(0, str(HERE))

import warnings  # noqa: E402

from hypothesis import HealthCheck, Phase, settings  # noqa: E402
from hypothesis.stateful import run_state_machine_as_test  # noqa: E402

import test_stateful_props as T  # noqa: E402
from remex import AnisotropyWarning, CompressedVectors, Quantizer  # noqa: E402
from remex.gpu import GPUSearcher  # noqa: E402
from remex.ivf import IVFCoarseIndex  # noqa: E402
from remex.st import RemexIndex  # noqa: E402


def mutated(func, *pairs):
    """Recompile ``func`` with each (old, new) text replacement applied; returns the new function."""
    src = inspect.getsource(func)
    for old, new in pairs:
        assert old in src, f"mutation anchor not found in {func.__qualname__}: {old!r}"
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("@"))
    ns: dict = {}
    exec(compile(src, f"<mutant {func.__qualname__}>", "exec"), func.__globals__, ns)
    return ns[func.__name__]


def patch_method(cls, name, *pairs):
    return mock.patch.object(cls, name, mutated(getattr(cls, name), *pairs))


def patch_setter(cls, name, *pairs):
    prop = getattr(cls, name)
    return mock.patch.object(cls, name, property(prop.fget, mutated(prop.fset, *pairs)))


MUTANTS = [
    dict(id="M01", machine="RemexIndexMachine",
         what="RemexIndex.add forgets to reset the decoded-token cache (stale _decoded after add)",
         patch=lambda: patch_method(RemexIndex, "add", ("        self._decoded = None\n", ""))),
    dict(id="M02", machine="RemexIndexMachine",
         what="RemexIndex.add returns ids counted from 0 instead of from len(index)",
         patch=lambda: patch_method(RemexIndex, "add", ("return np.arange(start, len(self))", "return np.arange(0, len(self) - start)"))),
    dict(id="M03", machine="RemexIndexMachine",
         what="RemexIndex.add re-derives the centring mean from every batch instead of the first",
         patch=lambda: patch_method(RemexIndex, "add", ("        if self.quantizer is None:\n            mean = corpus_mean(X)", "        if True:\n            mean = corpus_mean(X)"))),
    dict(id="M04", machine="RemexIndexMachine",
         what="RemexIndex.load drops the corpus mean when rebuilding the quantizer",
         patch=lambda: mock.patch.object(
             RemexIndex, "load", classmethod(mutated(RemexIndex.load.__func__, ("rotation=codes.rotation, mean=codes.mean,", "rotation=codes.rotation,"))))),
    dict(id="M05", machine="RemexIndexMachine",
         what="MaxSim block loop forgets to rebase reduceat offsets (breaks once the corpus spans several blocks)",
         patch=lambda: patch_method(RemexIndex, "_search_tokens", ("off[d0:d1] - lo", "off[d0:d1]"))),
    dict(id="M06", machine="CacheMachine",
         what="CompressedVectors.drop_cache does not clear _dir_len (answer-equivalent: only the structural postcondition sees it)",
         patch=lambda: patch_method(CompressedVectors, "drop_cache", ("        self._dir_len = {}\n", ""))),
    dict(id="M07", machine="CacheMachine",
         what="CompressedVectors.subset copies the parent's whole float32 cache instead of the selected rows",
         patch=lambda: patch_method(CompressedVectors, "subset", ("self._x_hat_rot[idx]", "self._x_hat_rot"))),
    dict(id="M08", machine="CacheMachine",
         what="Quantizer._direction_lengths caches under one key regardless of precision",
         patch=lambda: patch_method(Quantizer, "_direction_lengths", ("compressed._dir_len.get(precision)", "compressed._dir_len.get(None)"),
                                    ("compressed._dir_len[precision] = lengths", "compressed._dir_len[None] = lengths"))),
    dict(id="M09", machine="CacheMachine",
         what="Quantizer.R setter does not drop the RHT operator (assigned matrix ignored)",
         patch=lambda: patch_setter(Quantizer, "R", ("        self._op = None  # an explicitly assigned matrix is what gets applied", "        pass"))),
    dict(id="M10", machine="CacheMachine",
         what="Quantizer._get_x_hat_rot serves the full-precision cache to every precision",
         patch=lambda: patch_method(Quantizer, "_get_x_hat_rot", ("if precision is None and compressed._x_hat_rot is not None:", "if compressed._x_hat_rot is not None:"))),
    dict(id="M11", machine="CacheMachine",
         what="CompressedVectors.subset does not slice mean_dots (centred containers)",
         patch=lambda: patch_method(CompressedVectors, "subset", ("None if self.mean_dots is None else self.mean_dots[idx],", "self.mean_dots,"))),
    dict(id="M12", machine="CacheMachine",
         what="Quantizer._row_bias caches the centred bias under one key regardless of precision",
         patch=lambda: patch_method(Quantizer, "_row_bias", ('key = ("bias", precision)', 'key = ("bias", None)'))),
    dict(id="M13", machine="GpuIvfMachine",
         what="GPUSearcher caches the device scale/bias under one key regardless of precision",
         patch=lambda: patch_method(GPUSearcher, "_device_norms", ("if precision not in self._norm_cache:", "if None not in self._norm_cache:"),
                                    ("self._norm_cache[precision] = (", "self._norm_cache[None] = ("),
                                    ("return self._norm_cache[precision]", "return self._norm_cache[None]"))),
    dict(id="M14", machine="GpuIvfMachine",
         what="GPUSearcher.drop_cache does not release the device cache (answer-equivalent)",
         patch=lambda: patch_method(GPUSearcher, "drop_cache", ("        self._x_hat_rot_gpu = None", "        pass"))),
    dict(id="M15", machine="GpuIvfMachine",
         what="IVF probe order breaks Hamming ties by descending cell id",
         patch=lambda: patch_method(IVFCoarseIndex, "probe_cells", ("np.lexsort((all_cells, hd))", "np.lexsort((-all_cells.astype(np.int64), hd))"))),
    dict(id="M16", machine="GpuIvfMachine",
         what="IVF candidate_indices drops the last row of every visited cell",
         patch=lambda: patch_method(IVFCoarseIndex, "candidate_indices", ("self.cell_offsets[c + 1]", "(self.cell_offsets[c + 1] - 1 if self.cell_offsets[c + 1] > self.cell_offsets[c] else self.cell_offsets[c + 1])"))),
    dict(id="M17", machine="CacheMachine",
         what="Quantizer._effective_norms skips the per-search guards when the container is warm (renorm shortcut)",
         patch=lambda: patch_method(Quantizer, "_effective_norms", ("        self._check_rotation(compressed)\n        self._check_bits(compressed)\n        self._check_mean(compressed)\n        lengths", "        lengths"))),
]

MACHINES = {"RemexIndexMachine": T.RemexIndexMachine, "CacheMachine": T.CacheMachine, "GpuIvfMachine": T.GpuIvfMachine}


def run_machine(cls, examples, steps):
    counter = {"n": 0}

    class Counted(cls):
        def __init__(self):
            counter["n"] += 1
            super().__init__()

    cfg = settings(max_examples=examples, stateful_step_count=steps, deadline=None, database=None, derandomize=True,
                   suppress_health_check=list(HealthCheck), phases=[Phase.generate])
    t0 = time.time()
    try:
        run_state_machine_as_test(Counted, settings=cfg)
    except BaseException as e:  # noqa: BLE001
        return dict(caught=True, examples=counter["n"], seconds=round(time.time() - t0, 1),
                    error=f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    return dict(caught=False, examples=counter["n"], seconds=round(time.time() - t0, 1), error="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=150)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("only", nargs="*")
    args = ap.parse_args()
    T._TMP_ROOT = Path(tempfile.mkdtemp(prefix="mutants_"))
    warnings.simplefilter("ignore", AnisotropyWarning)
    # baseline: the unmutated code must pass with the same settings
    results = []
    todo = [("baseline", None)] + [(m["id"], m) for m in MUTANTS]
    for mid, m in todo:
        if args.only and mid != "baseline" and not any(o in mid or o in m["machine"] for o in args.only):
            continue
        machines = list(MACHINES) if m is None else [m["machine"]]
        for mname in machines:
            ctx = contextlib.nullcontext() if m is None else m["patch"]()
            with ctx:
                r = run_machine(MACHINES[mname], args.examples, args.steps)
            row = dict(id=mid, machine=mname, what="unmutated remex" if m is None else m["what"], **r)
            results.append(row)
            print(f"{mid} {mname:18s} caught={r['caught']!s:5} examples={r['examples']:4d} {r['seconds']:6.1f}s  {r['error'][:110]}", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "mutation_results.json").write_text(json.dumps(results, indent=1))
    lines = ["| id | machine | defect | caught | examples run | seconds | first failure |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['machine']} | {r['what']} | {'yes' if r['caught'] else 'NO'} | {r['examples']} | {r['seconds']} | {r['error'].replace('|', '/')[:120]} |")
    (out / "mutation_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
