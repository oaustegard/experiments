"""Build one dependency venv per library under ENVS.

    python harness/envs.py [NAME ...]          # default: all 16

Commit0's recipe is `pip install -e .[extras]`, which runs the library's own build.
Instead the published distribution is installed from PyPI with the same extras
(pulling the same dependency set), then uninstalled so the checkout supplies the
code. Requirements files are read from the reference commit with `git show`, never
executed. A venv is finished when ENV/.ok exists; its log is ENV.log.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENVS, UVPY, git, load_tasks, mirror  # noqa: E402

PYPI = {"simpy": "simpy", "tinydb": "tinydb", "marshmallow": "marshmallow", "parsel": "parsel",
        "pyjwt": "PyJWT", "wcwidth": "wcwidth", "imapclient": "IMAPClient", "chardet": "chardet",
        "babel": "babel", "voluptuous": "voluptuous", "jinja": "Jinja2", "deprecated": "Deprecated",
        "cachetools": "cachetools", "cookiecutter": "cookiecutter", "portalocker": "portalocker",
        "minitorch": None}
# Dev extras that need a toolchain or are linters/docs no test imports.
SKIP = {"pre-commit", "sphinx", "sphinx-rtd-theme", "sphinx_rtd_theme", "coveralls", "tox",
        "mypy", "ruff", "black", "flake8", "pylint", "twine", "build", "wheel", "furo",
        "sphinx-issues", "autodocsumm", "alabaster", "pallets-sphinx-themes", "pip-tools",
        "pip-compile-multi", "sphinx-tabs", "sphinxcontrib-log-cabinet"}


def interpreter(version: str) -> str:
    hits = sorted(UVPY.glob(f"cpython-{version}.*/bin/python{version}"))
    if not hits:
        raise FileNotFoundError(f"no python {version} under {UVPY}")
    return str(hits[-1])


def _name(req: str) -> str:
    return re.split(r"[<>=!~;\[ @]", req.strip(), maxsplit=1)[0].lower()


def requirement_lines(task: dict, path: str, seen: set | None = None) -> list[str]:
    seen = seen or set()
    if path in seen:
        return []
    seen.add(path)
    text = git("--git-dir", str(mirror(task["name"])), "show", f"{task['reference_commit']}:{path}", check=False)
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith(("-e", "--", ".")):
            continue
        if line.startswith(("-r", "-c")):
            sub = str(Path(path).parent / line[2:].strip())
            out += requirement_lines(task, sub, seen)
            continue
        out.append(line)
    return out


def pyproject_requirements(task: dict, extras: list[str]) -> list[str] | None:
    """Dependencies and extras as the reference commit's pyproject declares them
    (PyPI's latest metadata drifts: marshmallow's latest [dev] dropped pytz).
    None when the reference has no PEP 621 table."""
    import tomllib
    text = git("--git-dir", str(mirror(task["name"])), "show",
               f"{task['reference_commit']}:pyproject.toml", check=False)
    if not text:
        return None
    proj = tomllib.loads(text).get("project")
    if not proj or "dependencies" not in proj and "optional-dependencies" not in proj:
        return None
    self_name = proj.get("name", "").lower()
    opt = proj.get("optional-dependencies", {})
    out, todo, seen = list(proj.get("dependencies", [])), list(extras), set()
    while todo:
        e = todo.pop()
        if e in seen:
            continue
        seen.add(e)
        for r in opt.get(e, []):
            sub = re.match(rf"{re.escape(self_name)}\[(.*)\]", r.lower())
            if sub:
                todo += sub.group(1).split(",")
            else:
                out.append(r)
    return out


def requirements(task: dict) -> list[str]:
    s = task["setup"]
    reqs = list(s.get("pip_packages") or [])
    for f in s.get("packages") or []:
        reqs += requirement_lines(task, f)
    m = re.search(r"\.\[(.*)\]", s["install"])
    extras = m.group(1).split(",") if m else []
    declared = pyproject_requirements(task, extras)
    dist = PYPI[task["name"]]
    if declared is not None:
        reqs += declared
    elif dist:
        reqs.append(f"{dist}[{','.join(extras)}]" if extras else dist)
    # pkg_resources: removed from setuptools 81; several suites still import it.
    reqs += ["pytest", "setuptools<81"]
    return [r for r in dict.fromkeys(reqs) if _name(r) not in SKIP]


def uv(*args, log) -> int:
    return subprocess.run(["uv", *args], stdout=log, stderr=subprocess.STDOUT).returncode


def build(task: dict, force: bool = False) -> Path:
    env = ENVS / task["name"]
    if (env / ".ok").exists() and not force:
        return env
    ENVS.mkdir(parents=True, exist_ok=True)
    with open(ENVS / f"{task['name']}.log", "w") as log:
        if uv("venv", "-q", "--clear", "-p", interpreter(task["setup"]["python"]), str(env), log=log):
            raise RuntimeError(f"venv failed for {task['name']}")
        py = str(env / "bin" / "python")
        pip = ["pip", "install", "-q", "-p", py]
        reqs = requirements(task)
        print("requirements:", reqs, file=log, flush=True)
        if uv(*pip, *reqs, log=log):
            for r in reqs:  # one unresolvable extra should not sink the env
                if uv(*pip, r, log=log):
                    print(f"[{task['name']}] skipped requirement {r}", file=log, flush=True)
        dist = PYPI[task["name"]]
        if dist:
            uv("pip", "uninstall", "-q", "-p", py, dist, log=log)
    subprocess.run(["chmod", "-R", "a+rX", str(env)], check=True)
    (env / ".ok").write_text("ok\n")
    return env


def main(names: list[str]):
    tasks = load_tasks()
    for n in names or sorted(tasks):
        try:
            build(tasks[n])
            print("ok", n, flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue
            print("FAIL", n, type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
