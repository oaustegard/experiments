"""Serial merge gate: worker worktree -> tested commit on main, or rejection.

    python3 swarm/merge.py TASK_ID [TASK_ID ...]     gate these finished workers
    python3 swarm/merge.py --refresh                 full-suite baseline (+ bisect/revert)

Per candidate: diff its worktree against its base commit (only hcc/ and
docs/notes/ are taken; any edit under tests/ rejects the candidate), 3-way
apply onto current main in the gate worktree (wt_root/_gate), commit it with
the worker's root_cause as message. Up to ctx.batch candidates are stacked and
tested together:
    smoke suite + every candidate's target tests + a fixed sample of
    ctx.sample_size known-passing tests (seeded by the baseline commit).
Zero regressions against state/known.json are required, and each candidate
must flip at least one of its targets to PASS. A failing batch is retried one
candidate at a time. Accepted commits fast-forward main.

refresh() runs every suite at main HEAD. Tests that were PASS in known.json and
now fail are bisected over swarm merge commits (state/merges.jsonl) and the
culprit is reverted.
"""

from __future__ import annotations

if __name__ == "__main__":
    import _boot  # noqa: F401

import argparse
import sys
import time
from pathlib import Path

from swarm import testrun as tr
from swarm.common import (Ctx, GitError, add_worktree, append_jsonl, git, head, ledger, ledger_put,
                          link_untracked_suites, log, read_json, read_jsonl, write_json)

ALLOWED = ("hcc/", "docs/notes/")       # paths a worker may change
FORBIDDEN = ("tests/", "swarm/")        # touching these rejects the candidate
JUNK = ("__pycache__/", ".pyc")
MAX_MAIN_RACES = 3


# ---------------------------------------------------------------- candidate diff

def candidate_patch(ctx: Ctx, rec: dict) -> tuple[str, list[str], str | None]:
    """(patch, files, error). Cached in the task dir so a removed worktree can still be merged."""
    tdir = ctx.task_dir(rec["id"])
    cached = tdir / "patch.diff"
    if cached.exists():
        files = read_json(tdir / "patch_files.json", [])
        return cached.read_text(), files, None
    wt = ctx.worktree(rec["id"])
    if not wt.exists():
        return "", [], "worktree missing"
    git(wt, "add", "-A")
    changed = [f for f in git(wt, "diff", "--cached", "--no-renames", "--name-only", rec["base"]).splitlines() if f]
    # symlinks are the orchestrator's links to untracked suites (common.link_untracked_suites)
    bad = [f for f in changed if f.startswith(FORBIDDEN) and not (wt / f).is_symlink()]
    if bad:
        return "", changed, f"edited forbidden paths: {', '.join(bad[:5])}"
    files = [f for f in changed if f.startswith(ALLOWED) and not any(j in f for j in JUNK)]
    if not files:
        return "", changed, "empty diff (no changes under hcc/)"
    # --no-renames: a rename would list only the new path and drop the deletion
    patch = git(wt, "diff", "--cached", "--no-renames", "--binary", rec["base"], "--", *files)
    cached.write_text(patch)
    write_json(tdir / "patch_files.json", files)
    return patch, files, None


# ---------------------------------------------------------------- gate tree

def prepare_gate(ctx: Ctx, commit: str) -> Path:
    g = ctx.gate_tree
    if not (g / ".git").exists():
        add_worktree(ctx, g, commit)
    else:
        git(g, "reset", "--hard", "-q")
        git(g, "checkout", "--detach", "--force", "-q", commit)
        git(g, "clean", "-fdq")             # untracked, but keeps gitignored suites/tools
        link_untracked_suites(ctx, g)
    return g


def commit_patch(ctx: Ctx, g: Path, rec: dict, patch: str) -> str | None:
    """Apply + commit in the gate tree; returns the new sha or None on conflict."""
    try:
        git(g, "apply", "--3way", "--index", "--whitespace=nowarn", "-", input=patch)
    except GitError as e:
        git(g, "reset", "--hard", "-q")
        errs = [x for x in str(e).splitlines() if x.startswith("error:")]
        rec["reason"] = f"conflict applying onto main: {'; '.join(errs[:3])[:300] or str(e).split('): ', 1)[-1][:300]}"
        return None
    res = rec.get("result") or {}
    subject = (res.get("root_cause") or rec["signature"]).strip().splitlines()[0][:72]
    body = (f"{subject}\n\nSwarm-Task: {rec['id']}\nCluster: {rec['signature']}\n"
            f"Tests: {', '.join(res.get('test_ids_fixed') or rec['tests'])}\n\n"
            f"Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>\n")
    git(g, "commit", "-q", "--no-verify", "-m", body)
    return git(g, "rev-parse", "HEAD").strip()


def ff_main(ctx: Ctx, expected_parent: str, sha: str) -> bool:
    """Fast-forward main to sha iff main is still at expected_parent."""
    if head(ctx) != expected_parent:
        return False
    try:
        git(ctx.root, "merge", "--ff-only", "-q", sha)
        return True
    except GitError as e:
        log(f"ff of main failed: {e}")
        return False


# ---------------------------------------------------------------- evaluation

def gate_ids(ctx: Ctx, recs: list[dict]) -> list[str]:
    k = tr.known(ctx)
    base = read_json(ctx.state / "baseline.json", {}) or {}
    ids = {t for t in k if tr.suite_of(t) == "smoke"}
    for r in recs:
        ids.update(r.get("targets") or r["tests"])
    ids.update(tr.sample_ids(ctx, ctx.sample_size, base.get("commit", "seed")))
    return sorted(ids)


def evaluate(ctx: Ctx, tree: Path, ids: list[str]) -> tuple[dict, list[str]]:
    """Run ids in tree; returns (results, regressions vs known). Regressions are re-run
    once at low parallelism so a timeout under CPU load does not reject a good patch."""
    k = tr.known(ctx)
    res = tr.run_ids(ctx, tree, ids)
    regs = [t for t in ids if tr.is_pass(k.get(t)) and not tr.is_pass(res.get(t))]
    if regs:
        again = tr.run_ids(ctx, tree, regs, jobs=max(1, ctx.jobs // 2))
        res.update(again)
        regs = [t for t in regs if not tr.is_pass(again.get(t))]
    return res, regs


def gains(ctx: Ctx, res: dict, targets: list[str]) -> list[str]:
    k = tr.known(ctx)
    return [t for t in targets if tr.is_pass(res.get(t)) and not tr.is_pass(k.get(t))]


# ---------------------------------------------------------------- outcomes

def reject(ctx: Ctx, rec: dict, reason: str) -> None:
    rec.update(status="rejected", reason=reason, t_gate=time.time())
    patch = ctx.task_dir(rec["id"]) / "patch.diff"
    body = patch.read_text() if patch.exists() else ""
    (ctx.rejected / f"{rec['id']}.diff").write_text(
        f"# task {rec['id']}\n# cluster {rec['signature']}\n# rejected: {reason}\n{body}")
    ledger_put(ctx, rec)
    log(f"REJECT {rec['id']}: {reason}")


def accept(ctx: Ctx, rec: dict, sha: str, parent: str, fixed: list[str]) -> None:
    rec.update(status="merged", merged_commit=sha, reason="", gate_fixed=fixed, t_gate=time.time())
    ledger_put(ctx, rec)
    append_jsonl(ctx.state / "merges.jsonl", {"commit": sha, "parent": parent, "task": rec["id"],
                                              "fixed": fixed, "time": time.time()})
    log(f"MERGE  {rec['id']} -> {sha[:10]} fixed {len(fixed)}: {', '.join(fixed[:4])}")


def bump_merge_counter(ctx: Ctx, n: int) -> int:
    c = read_json(ctx.state / "counters.json", {}) or {}
    c["merges_since_full"] = c.get("merges_since_full", 0) + n
    c["merges_total"] = c.get("merges_total", 0) + n
    write_json(ctx.state / "counters.json", c)
    return c["merges_since_full"]


# ---------------------------------------------------------------- gate

def merge_batch(ctx: Ctx, recs: list[dict]) -> int:
    """Gate and merge `recs` (ledger records with status finished). Returns merged count."""
    pending: list[tuple[dict, str]] = []
    for r in recs:
        patch, _files, err = candidate_patch(ctx, r)
        if err:
            reject(ctx, r, err)
        else:
            pending.append((r, patch))
    merged = 0
    for _race in range(MAX_MAIN_RACES):
        if not pending:
            return merged
        main = head(ctx)
        g = prepare_gate(ctx, main)
        stacked: list[tuple[dict, str, str]] = []
        for r, patch in pending:
            sha = commit_patch(ctx, g, r, patch)
            if sha is not None:
                stacked.append((r, patch, sha))
            elif stacked:
                log(f"defer  {r['id']}: conflicts with another candidate in this batch")
            else:
                reject(ctx, r, r["reason"])
        if not stacked:
            return merged
        res, regs = evaluate(ctx, g, gate_ids(ctx, [r for r, _, _ in stacked]))
        if regs and len(stacked) > 1:
            log(f"batch of {len(stacked)} regressed {len(regs)} tests; gating one by one")
            return merged + sum(merge_batch(ctx, [r]) for r, _, _ in stacked)
        if regs:
            r = stacked[0][0]
            reject(ctx, r, f"regressions ({len(regs)}): {', '.join(regs[:6])}")
            return merged
        no_gain = [(r, p) for r, p, _ in stacked if not gains(ctx, res, r.get("targets") or r["tests"])]
        if no_gain:
            for r, _ in no_gain:
                reject(ctx, r, "no target test flipped to PASS")
            pending = [(r, p) for r, p, _ in stacked if (r, p) not in no_gain]
            continue
        top = stacked[-1][2]
        if not ff_main(ctx, main, top):
            log("main moved during gate; re-gating on new HEAD")
            pending = [(r, p) for r, p, _ in stacked]
            continue
        fixed = {r["id"]: gains(ctx, res, r.get("targets") or r["tests"]) for r, _, _ in stacked}
        tr.record(ctx, top, res)
        parent = main
        for r, _, sha in stacked:
            accept(ctx, r, sha, parent, fixed[r["id"]])
            parent = sha
        bump_merge_counter(ctx, len(stacked))
        return merged + len(stacked)
    for r, _ in pending:
        reject(ctx, r, "main kept moving during gate")
    return merged


# ---------------------------------------------------------------- refresh / bisect

def refresh(ctx: Ctx, do_bisect: bool = True) -> dict:
    """Full suites at main HEAD -> new baseline. Regressions vs known are bisected and reverted."""
    main = head(ctx)
    g = prepare_gate(ctx, main)
    log(f"full refresh at {main[:10]}")
    res = tr.run_full(ctx, g)
    prev = tr.known(ctx)
    regs = [t for t in res if tr.is_pass(prev.get(t)) and not tr.is_pass(res[t])]
    if regs:
        again = tr.run_ids(ctx, g, regs, jobs=max(1, ctx.jobs // 2))
        res.update(again)
        regs = [t for t in regs if not tr.is_pass(again.get(t))]
    for _round in range(3):
        if not (regs and do_bisect):
            break
        log(f"refresh found {len(regs)} regressions: {', '.join(regs[:6])}")
        culprit = bisect(ctx, regs)
        if not culprit or not revert(ctx, culprit, regs):
            log("could not locate/revert culprit; accepting the regression into the baseline")
            break
        main = head(ctx)
        g = prepare_gate(ctx, main)
        lost = [t for m in read_jsonl(ctx.state / "merges.jsonl") if m["commit"] == culprit for t in m["fixed"]]
        again = tr.run_ids(ctx, g, regs + lost)   # the culprit's fixes are undone too
        res.update(again)
        regs = [t for t in regs if not tr.is_pass(again.get(t))]
    main = head(ctx)
    tr.record(ctx, main, res, replace_known=True)
    c = read_json(ctx.state / "counters.json", {}) or {}
    c["merges_since_full"] = 0
    write_json(ctx.state / "counters.json", c)
    counts = tr.suite_counts(res)
    write_json(ctx.state / "baseline.json", {"commit": main, "time": time.time(), "suites": counts})
    append_jsonl(ctx.state / "history.jsonl", {"time": time.time(), "commit": main,
                                               "merges_total": c.get("merges_total", 0), "suites": counts})
    log("baseline: " + ", ".join(f"{s} {v['pass']}/{v['total']}" for s, v in sorted(counts.items())))
    return res


def bisect(ctx: Ctx, regs: list[str]) -> str | None:
    """First swarm merge commit since the last baseline at which any of `regs` fails."""
    base = (read_json(ctx.state / "baseline.json", {}) or {}).get("time", 0)
    reverted = {r.get("merged_commit") for r in ledger(ctx).values() if r.get("status") == "reverted"}
    commits = [m["commit"] for m in read_jsonl(ctx.state / "merges.jsonl")
               if m["time"] >= base and m["commit"] not in reverted]
    ancestors = set(git(ctx.root, "rev-list", head(ctx)).split())
    commits = [c for c in commits if c in ancestors]
    if not commits:
        return None

    def bad(c: str) -> bool:
        g = prepare_gate(ctx, c)
        r = tr.run_ids(ctx, g, regs)
        return any(not tr.is_pass(r.get(t)) for t in regs)

    if not bad(commits[-1]):
        return None                      # regression came from outside the swarm commits
    lo, hi = 0, len(commits) - 1         # invariant: commits[hi] is bad
    while lo < hi:
        mid = (lo + hi) // 2
        if bad(commits[mid]):
            hi = mid
        else:
            lo = mid + 1
    log(f"bisect: culprit {commits[hi][:10]}")
    return commits[hi]


def revert(ctx: Ctx, culprit: str, regs: list[str]) -> bool:
    main = head(ctx)
    g = prepare_gate(ctx, main)
    try:
        git(g, "revert", "--no-edit", culprit)
    except GitError as e:
        git(g, "revert", "--abort", check=False)
        log(f"revert of {culprit[:10]} conflicts: {e}")
        return False
    sha = git(g, "rev-parse", "HEAD").strip()
    if not ff_main(ctx, main, sha):
        return False
    for r in ledger(ctx).values():
        if r.get("merged_commit") == culprit:
            r.update(status="reverted", reason=f"reverted by {sha[:10]}: regressed {', '.join(regs[:4])}")
            ledger_put(ctx, r)
    append_jsonl(ctx.state / "reverts.jsonl", {"culprit": culprit, "revert": sha, "regs": regs, "time": time.time()})
    log(f"REVERT {culprit[:10]} as {sha[:10]}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--root", type=Path, default=Ctx().root)
    ap.add_argument("--wt-root", type=Path, default=Ctx().wt_root)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    ctx = Ctx(root=a.root.resolve(), wt_root=a.wt_root, jobs=a.jobs)
    if a.refresh:
        refresh(ctx)
    if a.tasks:
        led = ledger(ctx)
        merge_batch(ctx, [led[t] for t in a.tasks])
    return 0


if __name__ == "__main__":
    sys.exit(main())
