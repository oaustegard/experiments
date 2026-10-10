"""Drive one arm over Commit0-lite libraries, wrapping claude-workspace's fanout.py.

    run.py stage RUN --arm solo  --libs tinydb voluptuous ...
    run.py stage RUN --arm swarm --libs ... [--kmax 8]
    run.py stage RUN --arm solo-cont --from RUN1     # continue RUN1's trees, one agent each
    run.py stage RUN --arm fixer --from RUN1 [--k N] # K concurrent fixers on RUN1's trees
    run.py next RUN --slots N        # JSON lines: one Agent call per line
    run.py done RUN ID [ID ...]      # on each completion; saves a lib's patch once all its agents finish
    run.py grade RUN [--workers 2]
    run.py status RUN

Agents of one library share one tree, WORK/RUN/LIB. The patch (src_dir only,
against the base commit) is saved when the library's last agent reports; the
grade comes from a fresh base checkout with that patch applied, never from the tree.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, JAIL, WORK, git, load_tasks  # noqa: E402
import prompts  # noqa: E402
import workspace  # noqa: E402

sys.path.insert(0, str(JAIL.parent))
import fanout  # noqa: E402  (claude-workspace/scripts/fanout.py)

RUNS = DATA / "runs"
SCHEMA = {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}}
EXCLUDE = (":(exclude)babel/locale-data", ":(exclude)babel/global.dat")


def rdir(run: str) -> Path:
    return RUNS / run


def tree(run: str, lib: str) -> Path:
    return WORK / run / lib


def lib_of(unit: str) -> str:
    return unit.split(".")[0]


def stub_weights(task: dict, root: Path) -> dict[str, int]:
    """src file -> 1 + number of stubbed bodies (docstring and/or pass only), for every .py.

    Every file gets an owner: some stubs delete members outright and leave no empty
    body (voluptuous error.py lost its properties), so a stub count of 0 does not
    mean a file is complete. The pilot's partition skipped such files."""
    out = {}
    for f in sorted((root / task["src_dir"]).rglob("*.py")):
        try:
            mod = ast.parse(f.read_text(errors="replace"))
        except SyntaxError:
            continue
        n = 0
        for node in ast.walk(mod):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                     and isinstance(b.value.value, str))]
                if all(isinstance(b, ast.Pass) for b in body):
                    n += 1
        out[str(f.relative_to(root))] = n + 1
    return out


def partition(weights: dict[str, int], k: int) -> list[list[str]]:
    """Longest-processing-time greedy: k bins of files, balanced by stub count."""
    bins = [[0, []] for _ in range(k)]
    for f, w in sorted(weights.items(), key=lambda x: -x[1]):
        b = min(bins, key=lambda x: x[0])
        b[0] += w
        b[1].append(f)
    return [sorted(b[1]) for b in bins if b[1]]


def shard(failing: list[str], k: int) -> list[list[str]]:
    """k contiguous runs of the sorted failing ids, so related tests stay together.
    The pilot's per-test-file claims let one fixer hold voluptuous's only test file
    (89 failures) while the other idled."""
    failing = sorted(failing)
    n = math.ceil(len(failing) / k)
    return [failing[i:i + n] for i in range(0, len(failing), n)]


def junit_to_node(key: str, root: Path | None = None) -> str:
    """'tests.test_x.TestC::test_m' -> 'tests/test_x.py::TestC::test_m'.

    The module is the longest dotted prefix that is a file under root; without a
    root, the prefix up to the first capitalised (class) part."""
    cls, name = key.split("::", 1)
    parts = cls.split(".")
    cut = next((j for j, p in enumerate(parts) if p[:1].isupper()), len(parts))
    if root is not None:
        for j in range(len(parts), 0, -1):
            if (root / ("/".join(parts[:j]) + ".py")).is_file():
                cut = j
                break
    return "::".join(["/".join(parts[:cut]) + ".py", *parts[cut:], name])


def save_patch(run: str, lib: str, task: dict) -> Path:
    t = tree(run, lib)
    git("add", "-A", "--", task["src_dir"], *EXCLUDE, cwd=t)
    # Agent trees have one root commit (the skeleton); pilot trees carried history.
    has_base = subprocess.run(["git", "cat-file", "-e", task["base_commit"]], cwd=t,
                              stderr=subprocess.DEVNULL).returncode == 0
    base = task["base_commit"] if has_base else git("rev-list", "--max-parents=0", "HEAD", cwd=t).split()[-1]
    diff = git("diff", "--cached", "--binary", base, "--", task["src_dir"], *EXCLUDE, cwd=t)
    (rdir(run) / "patches").mkdir(exist_ok=True)
    p = rdir(run) / "patches" / f"{lib}.diff"
    p.write_text(diff)
    return p


def tree_from(run: str, src_run: str, lib: str, task: dict) -> Path:
    t = workspace.checkout(task, tree(run, lib), marker={"name": lib}, history=False)
    patch = rdir(src_run) / "patches" / f"{lib}.diff"
    if patch.read_text().strip():
        git("apply", "--binary", str(patch), cwd=t)
    workspace.opener(t)
    return t


def last_result(src_run: str, lib: str) -> str:
    r = results(src_run).get(lib)
    if not r or r.get("passed") is None:
        return "not recorded"
    return f"{r['passed']} passed, {r['failed']} failed or errored (of {r['n_target']} graded tests)"


def stage(a) -> None:
    d = rdir(a.run)
    if d.exists():
        sys.exit(f"{d} exists; use a fresh run name")
    tasks = load_tasks()
    libs = a.libs or (sorted(results(a.from_run)) if a.from_run else [])
    if not libs:
        sys.exit("no libraries: pass --libs, or --from a graded run")
    d.mkdir(parents=True)
    (d / "schema.json").write_text(json.dumps(SCHEMA))
    units, plan = [], {}
    for lib in libs:
        task = tasks[lib]
        if a.from_run:
            t = tree_from(a.run, a.from_run, lib, task)
        else:
            t = workspace.checkout(task, tree(a.run, lib), marker={"name": lib}, history=False)
        if a.arm == "solo":
            units.append((f"{lib}.solo", prompts.solo(task, t)))
        elif a.arm == "solo-cont":
            units.append((f"{lib}.cont", prompts.solo_cont(task, t, last_result(a.from_run, lib))))
        elif a.arm == "swarm":
            w = stub_weights(task, t)
            k = max(1, min(a.kmax, len(w), math.ceil(sum(w.values()) / a.per_worker)))
            bins = partition(w, k)
            plan[lib] = {"k": len(bins), "stubs": sum(w.values()) - len(w), "bins": bins}
            for i, files in enumerate(bins, 1):
                units.append((f"{lib}.w{i}", prompts.swarm(task, t, i, len(bins), files)))
        elif a.arm == "fixer":
            failing = json.loads((rdir(a.from_run) / "failing" / f"{lib}.json").read_text())
            if not failing:
                continue
            prev = json.loads((rdir(a.from_run) / "meta.json").read_text()).get("plan", {})
            k = min(a.k or prev.get(lib, {}).get("k", 1), math.ceil(len(failing) / a.min_shard))
            shards = shard(failing, k)
            plan[lib] = {"k": len(shards), "failing": len(failing)}
            for i, sh in enumerate(shards, 1):
                f = t / f".c0-shard-{i}.txt"
                f.write_text("\n".join(junit_to_node(x, t) for x in sh) + "\n")
                units.append((f"{lib}.f{i}", prompts.fixer(task, t, i, len(shards),
                                                          last_result(a.from_run, lib), f, len(sh))))
            workspace.opener(t)
    lines = [json.dumps({"id": u, "prompt": p, "out": str(d / "out" / f"{u}.json"),
                         "description": f"{a.run} {u}", "model": a.model,
                         "subagent_type": "general-purpose"}) for u, p in units]
    (d / "tasks.jsonl").write_text("\n".join(lines) + "\n")
    (d / "meta.json").write_text(json.dumps({"arm": a.arm, "model": a.model, "from": a.from_run,
                                             "libs": libs, "units": len(units), "plan": plan}, indent=1))
    fanout.init(d / "fanout", d / "tasks.jsonl", d / "schema.json", retries=0)
    (d / "out").mkdir()
    print(f"staged {len(units)} agents over {len(libs)} libraries in {d}")
    for lib, p in plan.items():
        print(f"  {lib}: k={p['k']}" + (f" stubs={p['stubs']}" if "stubs" in p else ""))


def next_(a) -> None:
    d = rdir(a.run)
    state = fanout._load(d / "fanout")
    batch = fanout.next_batch(state, a.slots)
    fanout._save(d / "fanout", state)
    for t in batch:
        print(json.dumps(fanout.agent_call(t, d / "prompts")))
    print(f"# {fanout.counts(state)}", file=sys.stderr)


def done(a) -> None:
    d = rdir(a.run)
    state = fanout._load(d / "fanout")
    print("\n".join(fanout.mark_done(state, a.ids)))
    fanout._save(d / "fanout", state)
    tasks = load_tasks()
    finished = {t["id"] for t in state["tasks"] if t.get("status") in ("done", "failed")}
    for lib in sorted({lib_of(i) for i in a.ids}):
        mine = [t["id"] for t in state["tasks"] if lib_of(t["id"]) == lib]
        if all(u in finished for u in mine) and not (d / "patches" / f"{lib}.diff").exists():
            p = save_patch(a.run, lib, tasks[lib])
            print(f"saved {p.name} ({p.stat().st_size} bytes): all {len(mine)} agents of {lib} finished")
    print(f"# {fanout.counts(state)}")


def cycle(a) -> None:
    """done IDS, then hand out as many agents as the session cap allows, counting
    agents still running in every run under data/runs."""
    if a.ids:
        done(argparse.Namespace(run=a.run, ids=a.ids))
    busy = 0
    for d in RUNS.iterdir():
        if (d / "fanout").exists():
            busy += fanout.counts(fanout._load(d / "fanout")).get("running", 0)
    free = max(0, a.cap - busy)
    own = fanout.counts(fanout._load(rdir(a.run) / "fanout")).get("running", 0)
    next_(argparse.Namespace(run=a.run, slots=own + free))


def results(run: str) -> dict[str, dict]:
    f = rdir(run) / "results.jsonl"
    if not f.exists():
        return {}
    return {r["lib"]: r for r in map(json.loads, f.open())}


_PARAM = re.compile(r"\[(.*)\]$")


def param_key(test_id: str) -> str:
    """Digit runs inside a parametrize id become '#'. Some ids embed the clock
    (marshmallow: test_invalid_datetime_deserialization[00:15:10 2026-10-10]), so
    certification and grading name the same test differently."""
    m = _PARAM.search(test_id)
    if not m:
        return test_id
    return test_id[:m.start()] + "[" + re.sub(r"\d+", "#", m.group(1)) + "]"


def normalize_outcomes(oc: dict[str, str]) -> dict[str, str]:
    """Collapse by param_key; a collapsed group passes only if all its members pass."""
    out: dict[str, str] = {}
    for k, v in oc.items():
        nk = param_key(k)
        out[nk] = v if nk not in out else ("pass" if out[nk] == v == "pass" else "fail")
    return out


def grade_one(run: str, lib: str, task: dict) -> dict:
    g = workspace.checkout(task, WORK / f"grade-{run}" / lib, marker={"name": lib})
    patch = rdir(run) / "patches" / f"{lib}.diff"
    out = {"lib": lib}
    try:
        if patch.read_text().strip():
            git("apply", "--binary", str(patch), cwd=g)
        r = workspace.run_tests(task, g)
    except Exception as e:  # noqa: BLE001
        return {**out, "passed": None, "error": f"{type(e).__name__}: {e}"[:500]}
    target = json.loads((DATA / "targets" / f"{lib}.json").read_text())
    stubpass_f = DATA / "targets" / f"{lib}.stubpass.json"
    stubpass = set(json.loads(stubpass_f.read_text())) if stubpass_f.exists() else None
    oc = r["outcomes"]
    if oc is None:
        out.update(passed=0, failed=len(target), n_target=len(target), score=0.0, note="no junit (collection crash)",
                   tail=r["tail"][-1500:])
    else:
        oc = normalize_outcomes(oc)
        passed = [k for k in target if oc.get(param_key(k)) == "pass"]
        failing = [k for k in target if oc.get(param_key(k)) != "pass"]
        (rdir(run) / "failing").mkdir(exist_ok=True)
        (rdir(run) / "failing" / f"{lib}.json").write_text(json.dumps(failing, indent=0) + "\n")
        out.update(passed=len(passed), failed=len(target) - len(passed), n_target=len(target),
                   score=round(len(passed) / len(target), 4), all_pass=len(passed) == len(target))
        if stubpass is not None:
            head = [k for k in target if k not in stubpass]
            out["headroom_score"] = round(sum(oc.get(param_key(k)) == "pass" for k in head) / max(1, len(head)), 4)
            out["n_headroom"] = len(head)
    (rdir(run) / "logs").mkdir(exist_ok=True)
    (rdir(run) / "logs" / f"{lib}.log").write_text(r["tail"])
    shutil.rmtree(g, ignore_errors=True)
    return out


_lock = threading.Lock()


def grade_run(a) -> None:
    d = rdir(a.run)
    have = results(a.run) if not a.regrade else {}
    todo = sorted(p.stem for p in (d / "patches").glob("*.diff") if p.stem not in have)
    tasks = load_tasks()
    if a.regrade and (d / "results.jsonl").exists():
        (d / "results.jsonl").unlink()

    def one(lib):
        r = grade_one(a.run, lib, tasks[lib])
        with _lock, (d / "results.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(lib, r.get("passed"), "/", r.get("n_target"), r.get("score"), r.get("error", ""), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


def status(a) -> None:
    d = rdir(a.run)
    state = fanout._load(d / "fanout")
    res = results(a.run)
    print(json.dumps({"fanout": fanout.counts(state),
                      "patches": len(list((d / "patches").glob("*.diff"))) if (d / "patches").exists() else 0,
                      "graded": {k: v.get("score") for k, v in sorted(res.items())}}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("run")
    s.add_argument("--arm", choices=("solo", "solo-cont", "swarm", "fixer"), required=True)
    s.add_argument("--libs", nargs="*")
    s.add_argument("--from", dest="from_run")
    s.add_argument("--model", default="haiku")
    s.add_argument("--kmax", type=int, default=8)
    s.add_argument("--per-worker", type=int, default=25, help="stubbed bodies per swarm builder")
    s.add_argument("--k", type=int, help="fixers per library (default: the source run's k)")
    s.add_argument("--min-shard", type=int, default=5, help="fewest failing tests per fixer")
    n = sub.add_parser("next")
    n.add_argument("run")
    n.add_argument("--slots", type=int, default=10)
    dn = sub.add_parser("done")
    dn.add_argument("run")
    dn.add_argument("ids", nargs="+")
    g = sub.add_parser("grade")
    g.add_argument("run")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--regrade", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("run")
    cy = sub.add_parser("cycle", help="done IDS then refill up to the session cap")
    cy.add_argument("run")
    cy.add_argument("ids", nargs="*")
    cy.add_argument("--cap", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "cycle":
        cycle(a)
        return
    {"stage": stage, "next": next_, "done": done, "grade": grade_run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
