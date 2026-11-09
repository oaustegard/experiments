"""Paths and the jail boundary shared by both drivers.

Everything that executes code nobody here wrote runs under the hub's
scripts/jail.sh: OpenEvolve itself, OpenEvolve's evaluator, and every program
a model writes. The jail root is /tmp/jail/hsp, laid out as

  .deps/       OpenEvolve 0.4.1 and its dependencies (wheels, installed outside the jail)
  problem/     initial_program.py and evaluator.py from OpenEvolve examples/circle_packing
  jailside/    this experiment's jail-side scripts
  runs/<id>/   one directory per run, writable by nobody
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAIL = Path(os.environ.get("HSP_JAIL", "/tmp/jail/hsp"))
JAIL_SH = Path(os.environ.get("HSP_JAIL_SH", "/home/user/claude-workspace/scripts/jail.sh"))
OE_EXAMPLE = Path(os.environ.get("HSP_OE_EXAMPLE", "/tmp/src/openevolve/examples/circle_packing"))
BEST_KNOWN = 2.635983
NOBODY = 65534
HEX_MIN_SIDE = 11 ** 0.5      # area bound for 11 unit hexagons inside one regular hexagon
HEX_MAX_SCORE = 3.930092 / HEX_MIN_SIDE   # ~1.185: any higher hexagon score is impossible


def setup_jail():
    """Copy the problem and jail-side scripts in; idempotent."""
    if not (JAIL / ".deps" / "openevolve").exists():
        raise SystemExit("missing OpenEvolve in the jail: see README.md, Setup")
    import problems
    for name, spec in problems.PROBLEMS.items():
        dst = JAIL / spec["jail_dir"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("initial_program.py", "evaluator.py", spec["config"]):
            shutil.copy(spec["src"] / f, dst / f)
        os.chmod(dst, 0o755)
    shutil.copytree(HERE / "jailside", JAIL / "jailside", dirs_exist_ok=True)
    (JAIL / "runs").mkdir(exist_ok=True)
    for p in (JAIL, JAIL / "problem", JAIL / "jailside", JAIL / "runs"):
        os.chmod(p, 0o755)
    os.chown(JAIL / "runs", NOBODY, NOBODY)


def run_dir(run_id: str) -> Path:
    d = JAIL / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    os.chown(d, NOBODY, NOBODY)
    return d


def give_to_nobody(path: Path):
    for root, dirs, files in os.walk(path):
        os.chown(root, NOBODY, NOBODY)
        for f in files:
            os.chown(os.path.join(root, f), NOBODY, NOBODY)


def jail_cmd(*cmd, env: dict | None = None) -> list[str]:
    args = ["sh", str(JAIL_SH), "-C", str(JAIL),
            "-e", f"PYTHONPATH={JAIL / '.deps'}",
            "-e", f"CIRCLE_PACKING_TARGET_VALUE={BEST_KNOWN}"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    return args + ["--", *cmd]


def jail_eval(prog_dir: Path, timeout: float = 60, workers: int = 4, problem_dir: str = "problem") -> dict:
    """Score every *.py in prog_dir inside the jail; returns {name: {metrics, t, dt}}."""
    give_to_nobody(prog_dir)
    rel = prog_dir.relative_to(JAIL)
    p = subprocess.run(jail_cmd("python3", "jailside/evalkit.py", str(rel),
                                "--timeout", str(timeout), "--workers", str(workers),
                                "--problem", problem_dir),
                       capture_output=True, text=True, timeout=timeout * 4 + 120)
    res = prog_dir / "results.json"
    if not res.exists():
        raise RuntimeError(f"jail eval failed rc={p.returncode}: {p.stderr[-800:]}")
    return json.loads(res.read_text())


def score_of(metrics: dict, problem: str = "circle") -> float:
    """Circle packing: sum_radii / best known, 0 for an invalid packing,
    recomputed rather than read from combined_score so the normalisation cannot
    drift with an env var. Other problems: the evaluator's combined_score."""
    if problem == "hexagon":
        # The stock hexagon evaluator tests containment with an absolute tolerance on an
        # unnormalised cross product, so a tiny outer hexagon (side 1e-9) passes every
        # check and scores ~3.9e9 (hx-L-s2, 2026-11-09). Eleven unit hexagons have
        # area 11 x unit, so the outer side can never be below sqrt(11).
        inv = (metrics or {}).get("inv_outer_hex_side_length")
        if inv and 1.0 / float(inv) < HEX_MIN_SIDE:
            return 0.0
    if problem != "circle":
        return float((metrics or {}).get("combined_score", 0) or 0)
    if not metrics or float(metrics.get("validity", 0) or 0) < 1:
        return 0.0
    return float(metrics.get("sum_radii", 0) or 0) / BEST_KNOWN
