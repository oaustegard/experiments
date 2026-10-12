"""Federation: run the swarm across several machines that share one git remote.

One hub owns main. Siblings (other containers) each work a partition of the
suites and send their gated commits back as candidates. Transport is branches
of one GitHub repo (default oaustegard/experiments), all append-only:

    hcc/main          the hub's main, pushed (fast-forward) as it advances
    hcc/state         orphan chain; federation/{known,baseline,ingest,meta}.json
                      = the hub's test baseline + what it did with each candidate
    hcc/cand/<NAME>   sibling NAME's local main (its candidates on top of an
                      hcc/main commit). After a rebase it is re-stacked with a
                      bookkeeping merge whose first parent is the new local main
                      and whose second parent is the old branch tip, so the
                      branch only ever fast-forwards.

Sibling (`run.py run --sibling NAME`): seeds swarm/state from hcc/state, runs
the normal loop on its --only partition with ids prefixed `NAME-`, gates
locally, stamps each accepted commit with a `Fixed:` trailer (gate-verified
newly passing tests), pushes its main to hcc/cand/NAME, and every
--rebase-every seconds rebases onto hcc/main: commits the hub merged or
rejected are dropped, the rest are cherry-picked; one that no longer applies is
marked a conflict in the sibling ledger, which makes the next worker on that
cluster a REBASE worker (run.launch).

Hub (`run.py run --federate`): every --fed-poll seconds fetches hcc/cand/*,
lists each branch's candidate stack (first-parent, non-merge commits not
reachable from main), and turns each new commit into a ledger record with
status `finished`, `ingest: True` and its diff as patch.diff. The normal merge
gate takes it from there, lighter for ingests: targets outside --suites are not
run and no target has to flip (the sibling's gate verified that); its claimed
fixes are recorded as PASS. A conflict is final for that commit sha; the
sibling's rebase produces a new sha that is tried again. After main moves the
hub pushes hcc/main, and hcc/state when the baseline or ingest outcomes change.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from swarm import testrun as tr
from swarm.common import (AUTHOR_EMAIL, AUTHOR_NAME, Ctx, head, ledger, ledger_put, log, read_json,
                          write_json)

DEFAULT_URL = "https://github.com/oaustegard/experiments"
REMOTE = "fed"
MAIN_BRANCH = "hcc/main"
STATE_BRANCH = "hcc/state"
CAND_PREFIX = "hcc/cand/"
NET_TIMEOUT = 600
STATE_PUSH_EVERY = 300      # seconds between hcc/state pushes (when changed)


# ---------------------------------------------------------------- git plumbing

def fgit(cwd: Path, *args: str, input: str | None = None, timeout: int = NET_TIMEOUT,
         check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_AUTHOR_NAME=AUTHOR_NAME, GIT_AUTHOR_EMAIL=AUTHOR_EMAIL,
               GIT_COMMITTER_NAME=AUTHOR_NAME, GIT_COMMITTER_EMAIL=AUTHOR_EMAIL, GIT_TERMINAL_PROMPT="0")
    try:
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="surrogateescape", input=input,
                           env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        p = subprocess.CompletedProcess(args, 124, "", f"git {args[0]} timed out after {timeout}s")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[-400:]}")
    return p


def out(cwd: Path, *args: str) -> str:
    return fgit(cwd, *args).stdout.strip()


def rev(cwd: Path, ref: str) -> str | None:
    p = fgit(cwd, "rev-parse", "-q", "--verify", f"{ref}^{{commit}}", check=False)
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def is_ancestor(cwd: Path, a: str, b: str) -> bool:
    return fgit(cwd, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def ensure_remote(ctx: Ctx, url: str) -> None:
    cur = fgit(ctx.root, "remote", "get-url", REMOTE, check=False)
    if cur.returncode != 0:
        fgit(ctx.root, "remote", "add", REMOTE, url)
    elif cur.stdout.strip() != url:
        fgit(ctx.root, "remote", "set-url", REMOTE, url)


def fetch(ctx: Ctx, *refspecs: str) -> bool:
    p = fgit(ctx.root, "fetch", "-q", "--no-tags", REMOTE, *refspecs, check=False)
    if p.returncode != 0:
        log(f"fed    fetch failed: {p.stderr.strip()[-300:]}")
    return p.returncode == 0


def push(ctx: Ctx, sha: str, branch: str) -> bool:
    p = fgit(ctx.root, "push", "-q", REMOTE, f"{sha}:refs/heads/{branch}", check=False)
    if p.returncode != 0:
        log(f"fed    push {branch} failed: {p.stderr.strip()[-300:]}")
    return p.returncode == 0


def fstate(ctx: Ctx) -> dict:
    return read_json(ctx.state / "federation.json", {}) or {}


def fstate_put(ctx: Ctx, st: dict) -> None:
    write_json(ctx.state / "federation.json", st)


# ---------------------------------------------------------------- commit messages

def trailer(msg: str, key: str) -> str | None:
    m = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*)$", msg)
    return m.group(1).strip() if m else None


def id_list(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def task_ids_in(cwd: Path, rng: str) -> set[str]:
    """Swarm-Task ids named in the commit messages of a rev range."""
    p = fgit(cwd, "log", "--format=%B%x00", rng, check=False)
    return {m.group(1) for m in re.finditer(r"(?m)^Swarm-Task:[ \t]*(\S+)", p.stdout)}


def add_fixed_trailer(msg: str, fixed: list[str]) -> str:
    """Insert `Fixed: a, b` after the Tests: line (or before Co-Authored-By)."""
    if not fixed or trailer(msg, "Fixed") is not None:
        return msg
    line = f"Fixed: {', '.join(fixed)}"
    if re.search(r"(?m)^Tests:.*$", msg):
        return re.sub(r"(?m)^(Tests:.*)$", lambda m: m.group(1) + "\n" + line, msg, count=1)
    if "Co-Authored-By:" in msg:
        return msg.replace("Co-Authored-By:", line + "\n\nCo-Authored-By:", 1)
    return msg.rstrip("\n") + "\n\n" + line + "\n"


def restamp(cwd: Path, parent: str, shas: list[str], fixed: list[list[str]]) -> list[str]:
    """Recreate a stack of commits on `parent` with Fixed: trailers (same trees)."""
    new = []
    for sha, f in zip(shas, fixed):
        msg = fgit(cwd, "log", "-1", "--format=%B", sha).stdout
        tree = out(cwd, "rev-parse", f"{sha}^{{tree}}")
        parent = fgit(cwd, "commit-tree", tree, "-p", parent, input=add_fixed_trailer(msg, f)).stdout.strip()
        new.append(parent)
    return new


# ---------------------------------------------------------------- hub: publish

def ingest_outcomes(ctx: Ctx) -> dict:
    return {r["id"]: {"status": r["status"], "reason": str(r.get("reason", ""))[:240],
                      "merged_commit": r.get("merged_commit"), "cand_sha": r.get("cand_sha"),
                      "sibling": r.get("sibling")}
            for r in ledger(ctx).values() if r.get("ingest")}


def build_state_commit(ctx: Ctx, files: dict[str, str], parent: str | None, msg: str) -> str:
    entries = []
    for name, text in sorted(files.items()):
        blob = fgit(ctx.root, "hash-object", "-w", "--stdin", input=text).stdout.strip()
        entries.append(f"100644 blob {blob}\t{name}")
    sub = fgit(ctx.root, "mktree", input="\n".join(entries) + "\n").stdout.strip()
    root = fgit(ctx.root, "mktree", input=f"040000 tree {sub}\tfederation\n").stdout.strip()
    args = ["commit-tree", root] + (["-p", parent] if parent else [])
    return fgit(ctx.root, *args, input=msg + "\n").stdout.strip()


def hub_publish(ctx: Ctx, force_state: bool = False) -> None:
    """Push main to hcc/main when it moved; push hcc/state when its content changed
    (at most every STATE_PUSH_EVERY seconds unless force_state)."""
    import json
    st = fstate(ctx)
    main = head(ctx)
    if st.get("pushed_main") != main and push(ctx, main, MAIN_BRANCH):
        st["pushed_main"] = main
        log(f"fed    pushed {MAIN_BRANCH} = {main[:10]}")
    due = force_state or time.time() - st.get("state_pushed_at", 0) >= STATE_PUSH_EVERY
    if not due:
        fstate_put(ctx, st)
        return
    known = json.dumps(tr.known(ctx), sort_keys=True, indent=0)
    ingest = json.dumps(ingest_outcomes(ctx), sort_keys=True, indent=1)
    baseline = json.dumps(read_json(ctx.state / "baseline.json", {}) or {}, sort_keys=True, indent=1)
    import hashlib
    digest = hashlib.sha1((known + ingest + baseline).encode()).hexdigest()
    if digest == st.get("state_digest"):
        st["state_pushed_at"] = time.time()     # unchanged: look again after the interval
    else:
        if "state_parent" not in st:          # chain onto an existing remote state branch
            fetch(ctx, f"+refs/heads/{STATE_BRANCH}:refs/fed/state")
            st["state_parent"] = rev(ctx.root, "refs/fed/state")
        meta = json.dumps({"main": main, "time": time.time()}, indent=1)
        sha = build_state_commit(ctx, {"known.json": known, "ingest.json": ingest,
                                       "baseline.json": baseline, "meta.json": meta},
                                 st.get("state_parent"), f"federation state at main {main[:10]}")
        if push(ctx, sha, STATE_BRANCH):
            st.update(state_parent=sha, state_digest=digest, state_pushed_at=time.time())
            log(f"fed    pushed {STATE_BRANCH} = {sha[:10]}")
    fstate_put(ctx, st)


# ---------------------------------------------------------------- hub: ingest

def _is_conflict(r: dict) -> bool:
    return r.get("status") == "rejected" and str(r.get("reason", "")).startswith(("conflict", "main kept moving"))


def candidate_stack(ctx: Ctx, tip: str, main: str) -> list[str]:
    p = fgit(ctx.root, "rev-list", "--reverse", "--first-parent", "--no-merges", tip, f"^{main}", check=False)
    return p.stdout.split()


def poll_candidates(ctx: Ctx) -> int:
    """Fetch hcc/cand/* and add each new candidate commit to the ledger as `finished`."""
    from swarm import merge
    if not fetch(ctx, f"+refs/heads/{CAND_PREFIX}*:refs/fed/cand/*"):
        return 0
    st = fstate(ctx)
    sibs = st.setdefault("siblings", {})
    refs = fgit(ctx.root, "for-each-ref", "--format=%(objectname) %(refname)", "refs/fed/cand/").stdout.split("\n")
    main = head(ctx)
    added = 0
    for line in filter(None, refs):
        tip, ref = line.split(" ", 1)
        name = ref[len("refs/fed/cand/"):]
        s = sibs.setdefault(name, {})
        if s.get("tip") == tip and s.get("main") == main:
            continue
        led = ledger(ctx)
        for c in candidate_stack(ctx, tip, main):
            msg = fgit(ctx.root, "log", "-1", "--format=%B", c).stdout
            tid = trailer(msg, "Swarm-Task") or f"{name}-{c[:10]}"
            r = led.get(tid)
            if r and (r["status"] in ("merged", "finished", "launched", "reverted")
                      or (r["status"] == "rejected" and (not _is_conflict(r) or r.get("cand_sha") == c))):
                continue
            parents = out(ctx.root, "rev-list", "--parents", "-n1", c).split()[1:]
            if len(parents) != 1:
                continue
            base = parents[0]
            changed = [f for f in out(ctx.root, "diff", "--no-renames", "--name-only", base, c).splitlines() if f]
            tests = id_list(trailer(msg, "Tests"))
            fixed = id_list(trailer(msg, "Fixed"))
            cluster = trailer(msg, "Cluster") or f"{name}: {msg.splitlines()[0] if msg else c[:10]}"
            rec = {"id": tid, "key": cluster, "signature": cluster, "tests": tests[:3] or fixed[:3],
                   "targets": fixed or tests, "base": base, "status": "finished", "cost_usd": 0.0,
                   "result": {"status": "fixed", "root_cause": msg.splitlines()[0] if msg else tid,
                              "test_ids_fixed": fixed or tests, "files_changed": changed, "notes": ""},
                   "merged_commit": None, "reason": "", "t_launch": time.time(), "model": f"sibling:{name}",
                   "ingest": True, "sibling": name, "cand_sha": c}
            tdir = ctx.task_dir(tid)
            for f in ("patch.diff", "patch_files.json"):
                (tdir / f).unlink(missing_ok=True)
            bad = [f for f in changed if f.startswith(merge.FORBIDDEN)]
            files = [f for f in changed if f.startswith(merge.ALLOWED) and not any(j in f for j in merge.JUNK)]
            if bad or not any(f.startswith(merge.CODE) for f in files):
                ledger_put(ctx, rec)
                merge.reject(ctx, rec, f"edited forbidden paths: {', '.join(bad[:5])}" if bad
                             else "empty diff (no changes under hcc/)")
                continue
            (tdir / "patch.diff").write_text(
                fgit(ctx.root, "diff", "--no-renames", "--binary", base, c, "--", *files).stdout, errors="surrogateescape")
            write_json(tdir / "patch_files.json", files)
            ledger_put(ctx, rec)
            log(f"INGEST {tid} from {name} {c[:10]}: {rec['result']['root_cause'][:80]}")
            added += 1
        s.update(tip=tip, main=main, polled=time.time())
    fstate_put(ctx, st)
    return added


# ---------------------------------------------------------------- sibling

def sibling_fetch(ctx: Ctx, name: str) -> bool:
    return fetch(ctx, f"+refs/heads/{MAIN_BRANCH}:refs/fed/main", f"+refs/heads/{STATE_BRANCH}:refs/fed/state")


def remote_json(ctx: Ctx, name: str):
    import json
    p = fgit(ctx.root, "show", f"refs/fed/state:federation/{name}", check=False, timeout=120)
    try:
        return json.loads(p.stdout) if p.returncode == 0 else None
    except ValueError:
        return None


def seed(ctx: Ctx) -> bool:
    """Copy the hub's known/baseline into an empty swarm/state. True if seeded."""
    if tr.known(ctx):
        return False
    known, base = remote_json(ctx, "known.json"), remote_json(ctx, "baseline.json")
    if not known:
        return False
    write_json(ctx.state / "known.json", known)
    write_json(ctx.state / "baseline.json", base or {"commit": "seed", "time": time.time(), "suites": {}})
    log(f"fed    seeded known.json from {STATE_BRANCH}: {len(known)} tests")
    return True


def sibling_rebase(ctx: Ctx, name: str) -> bool:
    """Rebase local main onto refs/fed/main. Returns True if main changed."""
    from swarm import merge
    hub = rev(ctx.root, "refs/fed/main")
    if not hub:
        return False
    local = head(ctx)
    if is_ancestor(ctx.root, hub, local):
        return False
    if is_ancestor(ctx.root, local, hub):
        stack = []
    else:
        stack = candidate_stack(ctx, local, hub)
    outcomes = remote_json(ctx, "ingest.json") or {}
    base = fgit(ctx.root, "merge-base", local, hub, check=False).stdout.strip()
    hub_took = task_ids_in(ctx.root, f"{base}..{hub}" if base else hub)
    g = merge.prepare_gate(ctx, hub)
    led = ledger(ctx)
    kept, dropped = 0, []
    for c in stack:
        msg = fgit(ctx.root, "log", "-1", "--format=%B", c).stdout
        tid = trailer(msg, "Swarm-Task") or ""
        o = outcomes.get(tid) or {}
        rec = led.get(tid)
        if tid in hub_took or o.get("status") == "merged":
            continue                              # the hub has it
        if o.get("status") in ("rejected", "reverted") and not _is_conflict(o):
            if rec:
                dropped.append(rec)
                rec.update(status="rejected", reason=f"hub rejected: {o.get('reason', '')}"[:300])
                ledger_put(ctx, rec)
            log(f"fed    drop {tid or c[:10]}: hub {o.get('status')}: {o.get('reason', '')[:120]}")
            continue
        p = fgit(g, "cherry-pick", "--allow-empty", "--keep-redundant-commits", c, check=False)
        if p.returncode != 0:
            fgit(g, "cherry-pick", "--abort", check=False)
            fgit(g, "reset", "--hard", "-q", check=False)
            if rec:
                dropped.append(rec)
                rec.update(status="rejected", reason=f"conflict applying onto main: rebase onto {MAIN_BRANCH} {hub[:10]}")
                ledger_put(ctx, rec)
            log(f"fed    drop {tid or c[:10]}: conflicts with {MAIN_BRANCH} {hub[:10]} (cluster re-queued as rebase)")
            continue
        kept += 1
    new = out(g, "rev-parse", "HEAD")
    fgit(ctx.root, "reset", "--hard", "-q", new)
    log(f"fed    rebased onto {MAIN_BRANCH} {hub[:10]}: kept {kept}, dropped {len(dropped)}, main {new[:10]}")
    redo = sorted({t for r in dropped for t in (r.get("gate_fixed") or [])})
    if redo:                                     # the dropped fixes are gone: re-measure their tests
        tr.record(ctx, new, tr.run_ids(ctx, g, redo))
    return True


def publish_cand(ctx: Ctx, name: str) -> bool:
    st = fstate(ctx)
    local = head(ctx)
    if st.get("pushed_local") == local:
        return False
    branch = CAND_PREFIX + name
    fetch(ctx, f"+refs/heads/{branch}:refs/fed/cand/{name}")
    tip = rev(ctx.root, f"refs/fed/cand/{name}")
    sha = local
    if tip and tip != local and not is_ancestor(ctx.root, tip, local):
        tree = out(ctx.root, "rev-parse", f"{local}^{{tree}}")
        sha = fgit(ctx.root, "commit-tree", tree, "-p", local, "-p", tip,
                   input=f"federation: {name} candidates re-stacked on {local[:10]}\n").stdout.strip()
    if tip == sha:
        ok = True
    else:
        ok = push(ctx, sha, branch)
    if ok:
        st["pushed_local"] = local
        st["pushed_cand"] = sha
        fstate_put(ctx, st)
        log(f"fed    pushed {branch} = {sha[:10]} (main {local[:10]})")
    return ok


def status(ctx: Ctx) -> None:
    st = fstate(ctx)
    print(f"federation: pushed main {str(st.get('pushed_main'))[:10]}  state {str(st.get('state_parent'))[:10]}"
          f"  cand {str(st.get('pushed_cand'))[:10]}")
    from collections import Counter
    by = Counter((o["sibling"], o["status"]) for o in ingest_outcomes(ctx).values())
    for (sib, s), n in sorted(by.items()):
        print(f"  ingested from {sib}: {s} {n}")
    for name, s in sorted((st.get("siblings") or {}).items()):
        print(f"  {name}: tip {s.get('tip', '')[:10]} polled {time.strftime('%H:%M:%S', time.localtime(s.get('polled', 0)))}")
