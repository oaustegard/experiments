"""Tests for the corpora account.py builds beyond the working tree.

No network, no encoder, no credentials — a git repo is built in a temp dir and
files are deleted in it, which is the only fixture the tombstone path needs.

    python3 hybrid-code-index/test_account.py

The cases here are the ones that cost something when they were wrong:

  admissibility  a deleted file gets no stat() and no rglob, so every filter
                 `discover` applies to the tree has to be reapplied by hand.
                 Missing them made one repo contribute 74,736 tombstone chunks
                 against a 232-chunk working tree, almost all of it deleted
                 machine-generated JSON the live index also refuses.
  relocation     a moved file looks deleted at its old path. Account-wide the
                 move often crosses repos, so a per-repo guard does not see it.
  path prefix    `xr -r <repo>` filters on startswith(repo + "/"), so a label
                 that does not lead with the repo silently vanishes from every
                 scoped query rather than erroring.
  plan-carried   shard and merge rebuild the corpus independently and match
                 rows by content hash, so the corpus decision has to travel in
                 the artifact both of them read.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as Args

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import account as A  # noqa: E402
import hcindex as H  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def git(d: Path, *a: str) -> None:
    subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True)


def make_repo(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@example.com")
    git(d, "config", "user.name", "t")

    def commit(msg: str) -> None:
        git(d, "add", "-A")
        git(d, "-c", "commit.gpgsign=false", "commit", "-q", "-m", msg)

    body = "\n".join(f"def step_{i}(): return {i} * 7  # a substantial line here"
                     for i in range(40))
    (d / "keeper.py").write_text(body)
    (d / "gone.py").write_text("# a deleted module\n" + body)
    (d / "moved.py").write_text("# relocated later\n" + body)
    # over the 1 MiB cap: the live index would never carry this, so neither
    # should its tombstone
    (d / "huge.json").write_text(json.dumps([{"v": i} for i in range(120_000)]))
    (d / "node_modules").mkdir()
    (d / "node_modules" / "dep.js").write_text("// vendored\n" + body)
    (d / "package-lock.json").write_text(json.dumps({"lockfileVersion": 3}))
    commit("seed")

    for f in ("gone.py", "moved.py", "huge.json", "package-lock.json"):
        (d / f).unlink()
    (d / "node_modules" / "dep.js").unlink()
    commit("Drop the module, keep the conclusion")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "work"
        repo = root / "alpha"
        make_repo(repo)
        # `moved.py` reappears in a *different* repo, which is what the
        # 2026-07-28 experiments migration did 37 times over
        other = root / "beta"
        other.mkdir(parents=True)
        (other / "moved.py").write_text((repo / "keeper.py").read_text()
                                        .replace("keeper", "moved"))
        src = subprocess.run(["git", "-C", str(repo), "show", "HEAD~1:moved.py"],
                             capture_output=True, text=True).stdout
        (other / "moved.py").write_text(src)

        cfg = dict(H.DEFAULT_CFG)
        names = {"alpha": "x", "beta": "x"}

        print("deletion scan")
        dels = A._deletions(repo, set(cfg["extensions"]))
        check("finds deleted files", "gone.py" in dels, str(sorted(dels)))
        check("carries the removing commit's subject",
              dels["gone.py"][2] == "Drop the module, keep the conclusion")
        check("ignores files that still exist", "keeper.py" not in dels)

        print("admissibility mirrors discover()")
        check("keeps ordinary source", A.admissible("gone.py", cfg))
        check("drops skip_dirs", not A.admissible("node_modules/dep.js", cfg))
        check("drops skip_names", not A.admissible("package-lock.json", cfg))
        check("drops unindexed extensions", not A.admissible("a.bin", cfg))
        excl = dict(cfg); excl["exclude"] = ["data/*"]
        check("honours the repo's own exclude list",
              not A.admissible("data/embeddings.json", excl)
              and A.admissible("src/embeddings.json", excl))

        print("test fixtures never enter the corpus")
        # Gutenberg novels shipped as fuse-test input are prose, so they win
        # prose-shaped queries outright: a vague debugging question returned
        # alice_in_wonderland.txt and pride_and_prejudice.txt as its top two
        # account-wide hits, ahead of every real document. 438 chunks.
        for p in ["test/data/alice_in_wonderland.txt", "src/test/data/big.txt",
                  "tests/data/sample.json", "testdata/golden.txt",
                  "pkg/testdata/x.json", "fixtures/user.json",
                  "spec/fixtures/reply.json", "app/__fixtures__/state.json",
                  "internal/golden/out.txt", "ui/snapshots/home.txt"]:
            check(f"drops {p}", not A.admissible(p, cfg))
        # `match()` wraps a metacharacter-free pattern as *pat*, so the globs
        # are anchored on separators. Unanchored, "test/data" eats "latest/data"
        # and "fixtures" eats "src/fixtures.py" — both real source paths.
        for p in ["latest/data/real.py", "src/test/data_loader.py",
                  "manifests/data/spec.md", "src/fixtures.py", "goldens.md",
                  "app/snapshots.ts", "datatest/run.py"]:
            check(f"keeps {p}", A.admissible(p, cfg))
        # Same protection as `.git` in skip_dirs: load_cfg unions list values,
        # so a repo config cannot quietly re-admit its fixtures.
        merged = dict(cfg); merged["fixture_globs"] = []
        check("a repo cannot switch fixtures back on by emptying the key",
              sorted(set(H.DEFAULT_CFG["fixture_globs"]) | set(merged["fixture_globs"]))
              == sorted(H.DEFAULT_CFG["fixture_globs"]),
              "load_cfg unions rather than replaces")

        print("tombstone corpus")
        by_base = A.live_index(names, root, cfg)
        tombs = A.tombstone_chunks("alpha", repo, cfg, by_base)
        files = {c.f for c in tombs}
        check("indexes a genuinely deleted file",
              any("gone.py" in f for f in files), str(sorted(files)))
        check("drops a file relocated to ANOTHER repo",
              not any("moved.py" in f for f in files),
              "a per-repo guard sees only the deletion, and re-imports as "
              "'gone' content the account still has")
        check("drops a deleted file over the size cap",
              not any("huge.json" in f for f in files),
              "the live index refuses it at 1 MiB; importing it via history "
              "is how one repo produced 74,736 tombstone chunks")
        check("drops deleted vendored paths",
              not any("dep.js" in f for f in files))
        check("labels lead with the repo, so -r still scopes them",
              all(f.startswith("alpha/") for f in files), str(sorted(files)))
        check("marks them as deleted", all("[deleted]" in f for f in files))
        check("header names the removing commit",
              all("DELETED" in c.text for c in tombs))

        print("PR corpus")
        prs = [{"n": 7, "t": "Why the rounds are floored at two", "at": "2026-08-01",
                "b": "x" * 5000},
               {"n": 8, "t": "fixes typo", "at": "2026-08-02", "b": ""}]
        pc = A.pr_chunks("alpha", prs, cfg)
        check("splits a long body at max_chars",
              sum(1 for c in pc if c.f.startswith("alpha/[PR #7]")) == 3,
              f"got {sum(1 for c in pc if 'PR #7' in c.f)}")
        check("keeps a bodyless PR as one title chunk",
              sum(1 for c in pc if "PR #8" in c.f) == 1)
        check("labels lead with the repo", all(c.f.startswith("alpha/") for c in pc))
        check("offsets are distinct within a PR",
              len({c.s for c in pc if "PR #7" in c.f}) == 3)

        print("the corpus decision travels in the plan")
        base = {"repos": {"alpha": "x"}}
        n_tree = len(A.corpus_for(base, root, cfg))
        check("off by default", n_tree == len(A.corpus_for(
            dict(base, corpora={"tombstones": False, "prs": False}), root, cfg)))
        with_t = A.corpus_for(dict(base, corpora={"tombstones": True}), root, cfg)
        check("tombstones on adds chunks", len(with_t) > n_tree,
              f"{n_tree} -> {len(with_t)}")
        with_p = A.corpus_for(
            dict(base, corpora={"prs": True}, pr_bodies={"alpha": prs}), root, cfg)
        check("PRs on adds chunks", len(with_p) > n_tree,
              f"{n_tree} -> {len(with_p)}")
        check("PRs on with no bodies fetched degrades to the tree",
              len(A.corpus_for(dict(base, corpora={"prs": True}), root, cfg)) == n_tree,
              "a failed fetch must shrink the corpus, not fail the build")

        print("plan reacts to a corpus change")
        # `changed` drives the whole build: every downstream step is gated on
        # it, which is what makes a quiet run cheap. Turning a corpus on moves
        # no repo's pushed_at, so without an explicit check the flag would be
        # accepted and rebuild nothing.
        real_api, real_env = A.api, os.environ.get("GH_TOKEN")
        A.api = lambda path, token: [
            {"full_name": "oaustegard/alpha", "pushed_at": "2026-08-01T00:00:00Z"},
            {"full_name": "oaustegard/beta", "pushed_at": "2026-08-02T00:00:00Z"}]
        os.environ["GH_TOKEN"] = "x"
        try:
            prev = Path(tmp) / "prev.json"
            out = Path(tmp) / "plan.json"

            def run(prev_meta: dict, **flags) -> dict:
                prev.write_text(json.dumps(prev_meta))
                A.cmd_plan(Args(owner="oaustegard", prev=str(prev), out=str(out),
                                with_tombstones=flags.get("t", False),
                                with_prs=flags.get("p", False)))
                return json.loads(out.read_text())

            settled = {"repos": {"alpha": "2026-08-01T00:00:00Z",
                                 "beta": "2026-08-02T00:00:00Z"},
                       "corpora": {"tombstones": False, "prs": False}}
            check("quiet account with no corpus change stays quiet",
                  run(settled)["changed"] == [])
            check("turning a corpus on rebuilds every repo",
                  run(settled, p=True)["changed"] == ["alpha", "beta"])
            check("turning one back off also rebuilds",
                  run(dict(settled, corpora={"tombstones": False, "prs": True}))
                  ["changed"] == ["alpha", "beta"])
            check("a pre-corpora manifest is read as 'both off'",
                  run({"repos": settled["repos"]})["changed"] == [],
                  "otherwise every existing index forces one free full rebuild")
            check("tombstones force a full-history clone",
                  run(settled, t=True)["clone_depth"] == 0)
            check("without them the clone is depth 1",
                  run(settled)["clone_depth"] == 1,
                  "the old hardcoded --depth 50 carried history nothing read")
        finally:
            A.api = real_api
            if real_env is None:
                os.environ.pop("GH_TOKEN", None)
            else:
                os.environ["GH_TOKEN"] = real_env

        print("corpus is stable across independent rebuilds")
        # shard and merge each rebuild this and match rows by content hash; a
        # difference here does not error, it silently re-encodes the account
        p = dict(base, corpora={"tombstones": True, "prs": True},
                 pr_bodies={"alpha": prs})
        a1 = A.corpus_for(p, root, cfg)
        a2 = A.corpus_for(p, root, cfg)
        check("same chunks, same order",
              [(c.f, c.s, H.chunk_hash(c.text)) for c in a1]
              == [(c.f, c.s, H.chunk_hash(c.text)) for c in a2])

    print()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    print("all passed")


if __name__ == "__main__":
    main()
