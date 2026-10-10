"""Register a non-lite Commit0 library as a task, build its env and any compiled artifacts.

    python harness/full_setup.py statsmodels

statsmodels: Commit0's recipe is `pip install -e .[develop]`, which runs setup.py.
Dependencies come from its requirements files instead (read with git show), and
its 27 Cython modules, which the stub left byte-identical, are compiled once
inside the jail from a stub checkout (`setup.py build_ext --inplace`, no network,
build tools preinstalled in the env). The resulting .so files and generated
sources go to ENVS/_statsmodels-ext and are copied into every checkout
(workspace.place_artifacts). Test files live inside the package, so they are
excluded from every patch (run.save_patch, task["patch_exclude"]).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ENVS, PARQUET, WORK, git, jailed  # noqa: E402

FULL = DATA / "full_tasks.json"

RECIPES = {
    "statsmodels": {
        "packages": ["requirements.txt"],
        "pip_packages": ["numpy<2", "scipy<1.14", "pandas<2.3", "patsy", "packaging", "matplotlib",
                         "joblib", "colorama", "cython>=0.29.33,<4", "setuptools_scm[toml]~=8.0",
                         "setuptools>=63.4.3", "pytest>=7.3.0", "pytest-xdist", "pytest-timeout"],
        "patch_exclude": [":(exclude,glob)statsmodels/**/tests/**"],
        "artifacts": "_statsmodels-ext",
        "test_timeout": 9000,
        "test_extra": ["--timeout=300"],
        "build": ["python", "setup.py", "build_ext", "--inplace", "-j", "4"],
        "build_env": {"SETUPTOOLS_SCM_PRETEND_VERSION": "0.15.0.dev0"},
    },
}


def register(name: str) -> dict:
    import pandas as pd
    d = pd.read_parquet(PARQUET)
    r = d[d.repo == f"commit-0/{name}"].iloc[0]
    setup = {k: (list(v) if hasattr(v, "tolist") else v) for k, v in r["setup"].items()}
    rec = RECIPES[name]
    setup.update(install="(deps from requirements; see full_setup.py)", packages=rec["packages"],
                 pip_packages=rec["pip_packages"])
    test = dict(r["test"])
    # -p no:randomly: pytest-randomly reorders and reseeds; grading wants one order.
    test["test_cmd"] = test["test_cmd"] + " -p no:randomly"
    task = {"name": name, "repo": r["repo"], "original_repo": r["original_repo"],
            "base_commit": r["base_commit"], "reference_commit": r["reference_commit"],
            "setup": setup, "test": test, "src_dir": r["src_dir"].rstrip("/"),
            "patch_exclude": rec["patch_exclude"], "artifacts": rec["artifacts"],
            "test_timeout": rec["test_timeout"], "test_extra": rec.get("test_extra", []), "full": True}
    out = json.loads(FULL.read_text()) if FULL.exists() else {}
    out[name] = task
    FULL.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return task


def build_artifacts(task: dict) -> Path:
    import workspace
    rec = RECIPES[task["name"]]
    dest = ENVS / rec["artifacts"]
    if (dest / ".ok").exists():
        return dest
    tree = WORK / "build" / task["name"]
    t0 = {**task, "artifacts": None}
    workspace.checkout(t0, tree, history=False)
    before = set(git("ls-files", cwd=tree).split())
    cmd = jailed(t0, tree, rec["build"], 3600, env=rec["build_env"])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    (ENVS / f"{task['name']}-build.log").write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f"build failed ({p.returncode}); see {ENVS / (task['name'] + '-build.log')}:\n"
                           + p.stdout[-2000:])
    new = git("ls-files", "--others", "--exclude-standard", "--ignored", "--exclude=*", cwd=tree).split()
    keep = [f for f in new if f.endswith((".so", ".pyx", ".pxd", "_version.py")) and f not in before
            and not f.startswith("build/")]
    if dest.exists():
        shutil.rmtree(dest)
    for f in keep:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tree / f, dest / f)
    subprocess.run(["chmod", "-R", "a+rX", str(dest)], check=True)
    (dest / ".ok").write_text(f"{len(keep)} files\n")
    shutil.rmtree(tree)
    return dest


def main(name: str):
    import envs
    task = register(name)
    envs.build(task)
    print("env ok", flush=True)
    d = build_artifacts(task)
    print("artifacts", d, sum(1 for _ in d.rglob("*.so")), "extension modules", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
