"""Build one dependency venv per (repo, version) under ENVS.

    python harness/envs.py [KEY ...]          # default: every key in tasks.jsonl

Dependencies come from PyPI through uv, outside the jail. The task repo itself is
never installed (installing runs its setup.py); checkouts go on PYTHONPATH.
A venv is finished when ENV/.ok exists; its log is ENV.log.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENVS, UVPY, env_key, git, load_tasks, mirror, python_for  # noqa: E402

# Optional extras from Django's test requirements that need a C toolchain or a
# system library, or that no Django test in the pool imports unconditionally.
DJANGO_SKIP = {"pylibmc", "mysqlclient", "psycopg2", "psycopg", "cx_oracle", "oracledb",
               "gdal", "selenium", "pywatchman", "black"}


def interpreter(version: str) -> str:
    hits = sorted(UVPY.glob(f"cpython-{version}.*/bin/python{version}"))
    if not hits:
        raise FileNotFoundError(f"no python {version} under {UVPY}; run "
                                f"UV_PYTHON_INSTALL_DIR={UVPY} uv python install {version}")
    return str(hits[-1])


def django_requirements(task: dict) -> list[str]:
    commit = task["environment_setup_commit"]
    m = mirror(task["repo"])

    def read(path):
        return git("--git-dir", str(m), "show", f"{commit}:{path}", check=False)

    out = []
    for path in ("tests/requirements/py3.txt", "tests/requirements/base.txt"):
        text = read(path)
        if not text:
            continue
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-e"):
                continue
            if line.startswith("-r"):
                sub = read("tests/requirements/" + line[2:].strip())
                out += [s.split("#")[0].strip() for s in sub.splitlines() if s.split("#")[0].strip()]
                continue
            out.append(line)
        break
    name = lambda s: s.split(";")[0].replace(">", "=").replace("<", "=").replace("!", "=").split("=")[0].split("[")[0].strip().lower()
    return [r for r in dict.fromkeys(out) if name(r) not in DJANGO_SKIP]


def requirements(task: dict) -> list[str]:
    if task["repo"] == "django/django":
        return django_requirements(task)
    if task["repo"] == "sympy/sympy":
        return ["mpmath==1.3.0", "flake8", "flake8-comprehensions"]
    raise KeyError(task["repo"])


def uv(*args, log) -> int:
    p = subprocess.run(["uv", *args], stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def build(task: dict, force: bool = False) -> Path:
    key = env_key(task)
    env = ENVS / key
    if (env / ".ok").exists() and not force:
        return env
    ENVS.mkdir(parents=True, exist_ok=True)
    py = interpreter(python_for(task["spec_python"]))
    with open(ENVS / f"{key}.log", "w") as log:
        if uv("venv", "-q", "--clear", "-p", py, str(env), log=log):
            raise RuntimeError(f"venv failed for {key}")
        reqs = requirements(task)
        pip = ["pip", "install", "-q", "-p", str(env / "bin" / "python")]
        if reqs and uv(*pip, *reqs, log=log):
            # One unresolvable optional extra should not sink the env: retry singly.
            for r in reqs:
                if uv(*pip, r, log=log):
                    print(f"[{key}] skipped requirement {r}", file=log, flush=True)
        if task["repo"] == "django/django":
            # Django's own runtime deps (asgiref, sqlparse, pytz) at the release
            # matching this version, then drop Django itself: the checkout supplies it.
            series = task["version"]
            if uv(*pip, f"django=={series}.*", log=log) == 0:
                uv("pip", "uninstall", "-q", "-p", str(env / "bin" / "python"), "django", log=log)
    subprocess.run(["chmod", "-R", "a+rX", str(env)], check=True)
    (env / ".ok").write_text("ok\n")
    return env


def main(keys: list[str]):
    tasks = load_tasks()
    by_key = {}
    for t in tasks.values():
        by_key.setdefault(env_key(t), t)
    for key in keys or sorted(by_key):
        try:
            build(by_key[key])
            print("ok", key, flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue with the next env
            print("FAIL", key, type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
