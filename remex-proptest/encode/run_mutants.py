"""Run the harness once per planted defect and record which tests catch it.

    python3 run_mutants.py [mutant ...]      # default: all
Writes mutation_results.md next to this file. Each run: ENCODE_MUTANT=<name>,
fresh Hypothesis database off, fixed seed, --maxfail=4, 0.4x example budget, 15 minute cap, kernel
cache in a throwaway directory so mutated C does not pollute ~/.cache/remex.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = {
    "float64_midpoints": "boundaries = midpoints of the unrounded float64 centroids (pre-fix behaviour, codebook.py:84-86)",
    "rht_one_round": "rht_rounds returns 1 when B == d (the pre-1.0 rule behind issue #89)",
    "kernel_ties_go_up": "C quant kernel uses v >= b instead of v > b (ties go to the upper cell)",
    "numpy_assign_side_right": "NumPy fallback of assign_codes uses side='right'",
    "nested_unweighted_mean": "nested centroids use a plain mean instead of the probability-weighted mean",
    "length_floor_1e-3": "encode floors the norm at 1e-3 instead of 1e-8",
    "thread_chunk_drops_last_row": "threaded assign_codes drops the last element of every chunk",
    "scalar_encode_float32": "scalar mode casts input to float32 instead of float64",
    "numpy_rht_sign_index": "NumPy RHT path indexes the sign vector with perm (s[perm])",
    "lloyd_30_iterations": "lloyd_max_codebook default n_iter 300 -> 30",
    "encode_block_drops_row": "encode block loop drops the last row of every full block",
    "haar_no_sign_correction": "Haar rotation without the Mezzadri sign correction",
    "rht_plan_draw_order": "rht_plan draws signs before permutations (operator vs dense matrix drift)",
    "scalar_boundary_float32_window": "assign_codes compares float64 haystacks in float32",
    "linspace_init_asymmetric": "Lloyd-Max initial centroids linspace(-3s, 3.0001s)",
    "anisotropy_uses_max_norm": "anisotropy() divides by max ||x|| instead of mean ||x||",
    "encode_ratio_drops_zero_guard": "encode's anisotropy judge loses its norm_total > 0 guard",
}


def run(name):
    env = dict(os.environ, ENCODE_MUTANT=name, ENCODE_NO_DB="1", ENCODE_EXAMPLES_SCALE="0.4",
               REMEX_CACHE_DIR=tempfile.mkdtemp(prefix="remex_mut_"))
    t = time.time()
    try:
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "test_encode_props.py", "-q", "-p", "no:cacheprovider", "-W", "ignore",
             "--maxfail=4", "-rf", "--hypothesis-seed=0", "--tb=line"],
            cwd=HERE, env=env, capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return False, [], time.time() - t, "TIMEOUT"
    failed = re.findall(r"^FAILED test_encode_props.py::(\S+)", out, re.M)
    failed = [f.split(" ")[0] for f in failed]
    # an xfail(strict) that *passes* shows as FAILED too; drop those (they are not the mutant)
    summary = out.strip().splitlines()[-1] if out.strip() else ""
    return bool(failed), failed, time.time() - t, summary


def main():
    names = sys.argv[1:] or list(MUTANTS)
    rows = []
    for n in names:
        caught, failed, secs, summary = run(n)
        print(f"{n:34s} caught={caught!s:5s} {secs:6.1f}s  {', '.join(failed[:4])}", flush=True)
        rows.append((n, caught, failed, secs, summary))
    with open(os.path.join(HERE, "mutation_results.md"), "w") as f:
        f.write("# Mutation calibration of test_encode_props.py\n\n")
        f.write("Each mutant is a one-token edit of the real remex function applied by `conftest.py` "
                "(`ENCODE_MUTANT=<name>`); remex itself is untouched. `--maxfail=4`, 0.4x example budget, Hypothesis seed 0, "
                "database off. 'Caught' = at least one test failed (xfail-strict tests that start passing "
                "also count, and are listed).\n\n")
        f.write("| mutant | planted defect | caught | first failing tests | wall s |\n|---|---|---|---|---|\n")
        for n, caught, failed, secs, summary in rows:
            f.write(f"| `{n}` | {MUTANTS[n]} | {'YES' if caught else '**NO (survived)**'} | "
                    f"{', '.join('`'+x+'`' for x in failed[:4]) or '-'} | {secs:.0f} |\n")
        f.write(f"\nbaseline (no mutant) is the unmodified harness: all green with 8 xfails, see coverage_events.txt.\n")


if __name__ == "__main__":
    main()
