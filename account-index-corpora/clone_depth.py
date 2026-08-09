"""What does `--depth 50` actually buy the account index over `--depth 1`?

`account.py cmd_clone` clones every repo at depth 50 with this comment:

    --depth 50 keeps enough history for the tombstone corpus to see recent
    deletions without paying for the full clone

There is no tombstone corpus in the account build, so the extra history is
currently paid for nothing -- and the sharded rebuild pays it 6 times in
parallel. claude-workspace#197 asks not to change it blind.

Two numbers decide it, and they are different questions:

  cost      seconds and bytes for depth 1 vs depth 50 vs full. If depth 50 is
            free, dropping it is not worth a line of diff either way.
  coverage  IF tombstones land, does depth 50 even work? `git log
            --diff-filter=D` sees only the grafted window, so a shallow
            tombstone corpus is a *sliding recent window*, not the record. The
            experiment that measured 0/6 -> 6/6 ran against a full clone.

Run against whatever repos this session has in scope; the ratios are what
transfer, not the absolute seconds (this box clones through an agent proxy).

    python3 clone_depth.py owner/repo [owner/repo ...]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("DEPTH_WORK", "/tmp/clone-depth"))


def clone(url: str, dest: Path, depth: int | None) -> tuple[float, int]:
    """Wall seconds and received bytes for one clone. Bytes come from git's own
    progress line, which is the transfer the depth flag actually changes."""
    token = os.environ.get("GH_TOKEN", "")
    helper = f"!f() {{ echo username=x-access-token; echo password={token}; }}; f"
    cmd = ["git", "-c", f"credential.helper={helper}", "clone", "--progress"]
    if depth:
        cmd += ["--depth", str(depth)]
    cmd += [url, str(dest)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode:
        raise SystemExit(f"clone failed: {r.stderr[-400:]}")
    m = re.findall(r"Receiving objects:\s*100%.*?\|?\s*([\d.]+)\s*([KM])iB", r.stderr)
    by = 0
    if m:
        v, u = m[-1]
        by = int(float(v) * (1 << (10 if u == "K" else 20)))
    return dt, by


def git(d: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(d), *a],
                          capture_output=True, text=True).stdout


def deleted_files(d: Path) -> set[str]:
    """Files deleted and never restored -- the tombstone corpus's input set."""
    out = git(d, "log", "--diff-filter=D", "--name-only", "--format=")
    return {f for f in out.split("\n") if f.strip() and not (d / f).exists()}


def main() -> None:
    repos = sys.argv[1:]
    if not repos:
        raise SystemExit(__doc__.strip().split("\n")[-1])
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for full in repos:
        name = full.split("/")[1]
        url = f"https://github.com/{full}"
        row: dict = {"repo": full}
        by_depth = {}
        for label, depth in (("depth1", 1), ("depth50", 50), ("full", None)):
            dest = WORK / f"{name}-{label}"
            shutil.rmtree(dest, ignore_errors=True)
            dt, by = clone(url, dest, depth)
            n_commits = len(git(dest, "log", "--format=%H").split())
            row[label] = {"secs": round(dt, 1), "bytes": by, "commits": n_commits}
            by_depth[label] = dest
        # Coverage: what the tombstone corpus would see at each depth.
        full_dead = deleted_files(by_depth["full"])
        for label in ("depth1", "depth50"):
            dead = deleted_files(by_depth[label])
            row[label]["deleted_visible"] = len(dead)
        row["deleted_full"] = len(full_dead)
        span = git(by_depth["depth50"], "log", "--format=%ad", "--date=short")
        days = span.split()
        row["depth50_window"] = f"{days[-1]}..{days[0]}" if days else ""
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
        for d in by_depth.values():
            shutil.rmtree(d, ignore_errors=True)

    (HERE / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'repo':28s} {'d1':>8s} {'d50':>8s} {'full':>8s}   "
          f"{'deleted d50/full':>18s}")
    for r in rows:
        print(f"{r['repo']:28s} {r['depth1']['secs']:>7.1f}s "
              f"{r['depth50']['secs']:>7.1f}s {r['full']['secs']:>7.1f}s   "
              f"{r['depth50']['deleted_visible']:>8d}/{r['deleted_full']:<9d}")


if __name__ == "__main__":
    main()
