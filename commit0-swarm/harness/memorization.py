"""How much of an agent's code is the original library, verbatim?

    python harness/memorization.py RUN [RUN ...]   -> data/runs/RUN/memorization.json

For each graded library: the lines the agent added (stub -> agent) are compared
with the lines the reference has over the stub (stub -> reference), after
stripping whitespace. Only "informative" lines count: at least 25 characters,
not a comment, and not present anywhere in the stub tree (tests, docs, the
skeleton itself), so the test suite or a docstring cannot have dictated them.

  verbatim_rate   share of the agent's informative added lines that occur
                  verbatim among the reference's added lines
  recall          share of the reference's informative added lines the agent
                  reproduced verbatim

Independent reimplementation from tests produces a low rate; recall from
training data produces a high one. Read it beside the cross-run agreement
(two runs of the same library compared with each other).
Only git and text comparison run here; no library code executes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, git, load_tasks, mirror  # noqa: E402

MIN = 25


def norm(line: str) -> str:
    return " ".join(line.split())


def informative(line: str) -> bool:
    return len(line) >= MIN and not line.startswith(("#", '"""', "'''"))


def tree_lines(name: str, commit: str, prefix: str = "") -> dict[str, list[str]]:
    out = {}
    files = git("--git-dir", str(mirror(name)), "ls-tree", "-r", "--name-only", commit).split()
    for f in files:
        if f.endswith((".py", ".rst", ".md", ".txt", ".cfg", ".toml", ".json", ".yaml", ".yml")) and f.startswith(prefix):
            text = git("--git-dir", str(mirror(name)), "show", f"{commit}:{f}", check=False)
            out[f] = [norm(l) for l in text.splitlines()]
    return out


def added(before: list[str], after: list[str]) -> list[str]:
    b = set(before)
    return [l for l in after if l not in b and informative(l)]


def agent_tree(task: dict, patch: Path, dest: Path) -> dict[str, list[str]]:
    git("clone", "-q", "--shared", "--no-checkout", str(mirror(task["name"])), str(dest))
    git("checkout", "-q", task["base_commit"], cwd=dest)
    if patch.read_text().strip():
        git("apply", "--binary", str(patch), cwd=dest)
    out = {}
    for f in (dest / task["src_dir"]).rglob("*.py"):
        out[str(f.relative_to(dest))] = [norm(l) for l in f.read_text(errors="replace").splitlines()]
    return out


def compare(task: dict, patch: Path) -> dict:
    src = task["src_dir"]
    stub = tree_lines(task["name"], task["base_commit"])
    stub_all = {l for ls in stub.values() for l in ls}
    ref = tree_lines(task["name"], task["reference_commit"], prefix=src)
    with tempfile.TemporaryDirectory() as d:
        agent = agent_tree(task, patch, Path(d) / "t")
    ref_add, ag_add = set(), set()
    for f, lines in ref.items():
        if f.endswith(".py"):
            ref_add |= {l for l in added(stub.get(f, []), lines) if l not in stub_all}
    for f, lines in agent.items():
        ag_add |= {l for l in added(stub.get(f, []), lines) if l not in stub_all}
    hit = ag_add & ref_add
    return {"agent_lines": len(ag_add), "ref_lines": len(ref_add), "verbatim": len(hit),
            "verbatim_rate": round(len(hit) / max(1, len(ag_add)), 3),
            "recall": round(len(hit) / max(1, len(ref_add)), 3),
            "_agent_set": ag_add}


def main(runs: list[str]):
    tasks = load_tasks()
    sets = {}
    for run in runs:
        d = DATA / "runs" / run
        out = {}
        for p in sorted((d / "patches").glob("*.diff")):
            r = compare(tasks[p.stem], p)
            sets[(run, p.stem)] = r.pop("_agent_set")
            out[p.stem] = r
            print(f"{run:8s} {p.stem:13s} agent {r['agent_lines']:5d}  ref {r['ref_lines']:5d}  "
                  f"verbatim {r['verbatim']:5d}  rate {r['verbatim_rate']:.2f}  recall {r['recall']:.2f}", flush=True)
        (d / "memorization.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    # Cross-run agreement: two independent runs of one library, compared with each other.
    libs = {lib for _, lib in sets}
    for lib in sorted(libs):
        have = [(r, s) for (r, l), s in sets.items() if l == lib]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                (ra, a), (rb, b) = have[i], have[j]
                print(f"agree {lib:13s} {ra} vs {rb}: {len(a & b) / max(1, min(len(a), len(b))):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
