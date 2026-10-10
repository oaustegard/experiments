"""Turn post-cutoff libraries into Commit0-shaped tasks: the contamination control.

    python harness/control_setup.py          # every candidate in data/control_candidates.json

For each library: bare mirror from GitHub, then a stub commit on top of HEAD in which
every function and method body under the package directory is replaced by its
docstring (if any) and `pass`, via ast.unparse, the way Commit0 built its stubs.
HEAD is the reference commit, the stub commit is the base commit; the task joins
data/control_tasks.json, which common.load_tasks merges in. Only git and Python's
ast module touch the repos here; no repo code runs.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MIRRORS, WORK, git  # noqa: E402

CONTROL = DATA / "control_tasks.json"
# Test-time packages the candidates' notes name; installed into every control env.
TEST_DEPS = ["pytest", "hypothesis", "jsonschema", "pyyaml", "rich", "textual", "Pillow"]
PY = {"epub-blocks": "3.13"}


class Stub(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        body = node.body
        doc = [body[0]] if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else []
        node.body = doc + [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def stub_source(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    return ast.unparse(ast.fix_missing_locations(Stub().visit(tree))) + "\n"


def package_dir(root: Path, name: str) -> str | None:
    """The importable package: src/<pkg> or <pkg>, matched against the project name."""
    want = {name.replace("-", "_").lower(), name.split("-")[0].lower()}
    py = root / "pyproject.toml"
    if py.exists():
        try:
            proj = tomllib.loads(py.read_text())
            pk = proj.get("tool", {}).get("setuptools", {}).get("packages")
            if isinstance(pk, list) and pk:
                want.add(pk[0].lower())
            want.add(proj.get("project", {}).get("name", "").replace("-", "_").lower())
        except tomllib.TOMLDecodeError:
            pass
    for base in ("src", ""):
        d = root / base if base else root
        if not d.is_dir():
            continue
        for c in sorted(d.iterdir()):
            if c.is_dir() and (c / "__init__.py").exists() and c.name.lower() in want:
                return str(c.relative_to(root))
    for base in ("src",):  # single package under src/
        d = root / base
        if d.is_dir():
            pk = [c for c in d.iterdir() if c.is_dir() and (c / "__init__.py").exists()]
            if len(pk) == 1:
                return str(pk[0].relative_to(root))
    return None


def setup_one(c: dict) -> dict:
    name = c["name"]
    m = MIRRORS / f"{name}.git"
    if not m.exists():
        git("clone", "-q", "--bare", c["github"], str(m))
    work = WORK / "control-stub" / name
    if work.exists():
        shutil.rmtree(work)
    git("clone", "-q", "--shared", str(m), str(work))
    ref = git("rev-parse", "HEAD", cwd=work).strip()
    src = package_dir(work, name)
    if src is None:
        return {"name": name, "error": "package dir not found"}
    n_stubbed = 0
    for f in (work / src).rglob("*.py"):
        s = stub_source(f.read_text(errors="replace"))
        if s is not None:
            f.write_text(s)
            n_stubbed += 1
    git("-c", "user.email=c0@harness", "-c", "user.name=c0", "commit", "-qam",
        "Commit0-style stub: function bodies removed", cwd=work)
    base = git("rev-parse", "HEAD", cwd=work).strip()
    git("push", "-q", str(m), f"HEAD:refs/heads/c0-stub", cwd=work)
    test_dir = "tests/" if (work / "tests").is_dir() else "test/" if (work / "test").is_dir() else "."
    shutil.rmtree(work)
    return {"name": name, "repo": c["github"].split("github.com/")[1], "original_repo": c["github"].split("github.com/")[1],
            "base_commit": base, "reference_commit": ref, "src_dir": src, "control": True,
            "setup": {"install": "pip install -e .", "packages": None, "pip_packages": TEST_DEPS,
                      "pre_install": None, "python": PY.get(name, "3.12"), "specification": c["github"]},
            "test": {"test_cmd": "pytest", "test_dir": test_dir}, "files_stubbed": n_stubbed}


def main():
    cands = json.loads((DATA / "control_candidates.json").read_text())["candidates"]
    out = json.loads(CONTROL.read_text()) if CONTROL.exists() else {}
    for c in cands:
        try:
            t = setup_one(c)
        except Exception as e:  # noqa: BLE001
            t = {"name": c["name"], "error": f"{type(e).__name__}: {e}"[:300]}
        print(t["name"], t.get("src_dir"), t.get("files_stubbed"), t.get("error", ""), flush=True)
        if "error" not in t:
            out[t["name"]] = t
    CONTROL.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
