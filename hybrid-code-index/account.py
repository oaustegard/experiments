"""Account-wide index: plan / clone / build / verify.

Triggering is by **state comparison**, not events. Each run lists every owned
repo with its `pushed_at` and diffs against the manifest published with the
previous index. Nothing is configured in the indexed repos, a skipped or failed
run costs nothing (the next run compares state rather than replaying events),
and a new repo is picked up simply by appearing in the list.

Cloning is all-or-nothing on purpose. Selective cloning looks like the obvious
optimization, but the whole account clones in ~60 s and skipping repos means
their chunk *texts* are unavailable, so their rows cannot be content-hashed for
reuse and would have to be spliced positionally — a second, subtler consistency
problem. Clone everything, encode only what changed: measured 44.5 s -> 1.0 s
for a one-file delta.

Measured before this was written (`account-routing-tier`, `hybrid-code-index`):

    ~49,600 chunks account-wide      cold encode ~34 min     index 16.3 MB
    clone all repos ~60 s            flat query scan 7.6 ms

Two corpora beyond the working tree are available and **off by default**:
tombstones (deleted files, from `history-tombstone-index`) and merged PR bodies
(from `pr-decision-log`). Both measured as clear wins per-repo -- 0/6 -> 6/6 on
mechanism questions about deleted code, 6/8 -> 8/8 on rationale -- and neither
has been measured at account scale, where the known weakness is crowding rather
than coverage. `account.py corpora` reports the chunk-count delta without
encoding anything, which is the cheap half of that measurement; the answer-
quality half needs a full build. Turn them on in `plan`, not downstream: the
decision is recorded in plan.json so every shard and the merge agree by
construction rather than by matching CLI flags.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "repo-index"))
import hcindex as H  # noqa: E402

DIM, BITS, SEED, ROT = 384, 2, 0, "rht"


def tokenizer_sha256() -> str:
    """Hash the *logic* of `hcindex.tokens`, ignoring comments and formatting.

    BM25 postings are keyed by this function's output, so a query-side copy that
    drifts from it does not error -- the lexical arm just stops matching, while
    the dense arm keeps returning plausible-looking results. Recording the hash
    in the manifest lets a consumer refuse rather than silently degrade;
    `claude-workspace/scripts/xr.py` vendors `tokens` and checks it.

    Parsed and re-emitted rather than hashed raw, so a reflowed comment does not
    invalidate every published index, and with the docstring stripped so the two
    copies may document themselves differently.

    `ast.unparse`, NOT `ast.dump`. dump serializes the AST's internal field
    layout, which grows between releases -- the same source hashed three
    different ways across interpreters:

        3.10 / 3.11   0a95c45ce7fa46ed
        3.12          2faf90cb782a1c60      <- what the runner published
        3.13          6e147ade19dc28ae

    That makes the pin a Python-version detector rather than a logic detector,
    and it would refuse a perfectly good index in any session whose interpreter
    differs from the runner's. `unparse` re-emits canonical *source*, which is
    identical on 3.10 through 3.13.
    """
    import ast
    import hashlib
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(H.tokens)))
    body = tree.body[0].body  # type: ignore[attr-defined]
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body.pop(0)
    return hashlib.sha256(ast.unparse(tree).encode()).hexdigest()[:16]


def api(path: str, token: str) -> list:
    out, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://api.github.com/{path}&per_page=100&page={page}",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json"})
        batch = json.load(urllib.request.urlopen(req))
        if not batch:
            return out
        out += batch
        page += 1
        if page > 10:
            return out


def cmd_plan(a) -> None:
    """List owned repos and diff pushed_at against the previous manifest."""
    token = os.environ["GH_TOKEN"]
    repos = api("user/repos?affiliation=owner&sort=pushed", token)
    keep = [r for r in repos
            # forks are upstream code, not this account's decisions; archived
            # repos never change so they cost nothing to keep
            if not r.get("fork") and not r.get("disabled")]
    now = {r["full_name"].split("/")[1]: r["pushed_at"] for r in keep}
    # repo size in KB, only ever used as a shard-balancing fallback for repos
    # the previous manifest has never seen
    sizes = {r["full_name"].split("/")[1]: r.get("size", 0) for r in keep}
    prev_meta = json.loads(Path(a.prev).read_text())
    prev = prev_meta.get("repos", {})
    changed = sorted(n for n, t in now.items() if prev.get(n) != t)
    corpora = {"tombstones": a.with_tombstones, "prs": a.with_prs}
    # Turning a corpus on or off changes what every repo contributes, and no
    # repo's pushed_at moves to say so. Without this, dispatching the build with
    # --with-prs on a quiet account hits the `changed == 0` early exit and
    # reports success having rebuilt nothing — the flag would look supported and
    # do nothing, which is worse than not having it.
    if corpora != prev_meta.get("corpora", {"tombstones": False, "prs": False}):
        print(f"corpora changed {prev_meta.get('corpora', {})} -> {corpora}; "
              f"rebuilding every repo", flush=True)
        changed = sorted(now)
    unchanged = sorted(set(now) - set(changed))
    plan = {"repos": now, "changed": changed, "unchanged": unchanged,
            "owner": a.owner, "sizes": sizes,
            # measured chunk counts from the last build; the good weights
            "repo_chunks": prev_meta.get("repo_chunks", {}),
            # Which corpora this run builds, and how deep the clone has to be to
            # support them. Decided once, here, and carried in the artifact every
            # downstream job already reads -- see corpus_for() for why a flag on
            # those jobs would be a silent-divergence bug rather than an error.
            "corpora": corpora,
            # `git log --diff-filter=D` sees only the grafted window, so a
            # shallow clone yields a sliding recent slice of the deletion record
            # rather than the record. Measured on this account
            # (`account-index-corpora`): depth 50 saw 2 of muninn-utilities' 18
            # deletions, and cost the same wall time as depth 1 and as a full
            # clone -- the difference is inside run-to-run noise on a step that
            # is 89 s against a 22 min encode. So there is no cheap middle to
            # buy: take the whole history when tombstones are on, and depth 1
            # when they are not.
            "clone_depth": 0 if a.with_tombstones else 1}
    if a.with_prs:
        print("fetching merged PR bodies ...", flush=True)
        plan["pr_bodies"] = fetch_pr_bodies(a.owner, now, token)
        n = sum(len(v) for v in plan["pr_bodies"].values())
        print(f"  {n} merged PRs across {len(plan['pr_bodies'])} repos", flush=True)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(now)} repos: {len(changed)} changed, {len(unchanged)} unchanged",
          flush=True)
    if changed:
        print("  changed: " + ", ".join(changed[:12])
              + (" ..." if len(changed) > 12 else ""))
    on = [k for k, v in corpora.items() if v]
    print("  extra corpora: " + (", ".join(on) if on else "none (working tree only)"))


def cmd_clone(a) -> None:
    plan = json.loads(Path(a.plan).read_text())
    token = os.environ["GH_TOKEN"]
    root = Path(a.dir); root.mkdir(parents=True, exist_ok=True)
    helper = ("!f() { echo username=x-access-token; "
              f'echo password={token}; }}; f')
    # Set by `plan`: 1 for the working tree alone, 0 (full history) when the
    # tombstone corpus is on and needs the deletion record. The old hardcoded
    # `--depth 50` was a middle that bought neither -- it carried history no
    # corpus read, and had it been read it would have been an arbitrary window.
    # Legacy plans predate the field; depth 1 matches what they actually used.
    depth = plan.get("clone_depth", 1)
    t0 = time.time()
    for name in plan["repos"]:
        dest = root / name
        if dest.exists():
            continue
        cmd = ["git", "-c", f"credential.helper={helper}", "clone", "-q"]
        if depth:
            cmd += ["--depth", str(depth)]
        cmd += [f"https://github.com/{plan['owner']}/{name}.git", str(dest)]
        subprocess.run(cmd, check=False, capture_output=True)
    got = sum(1 for p in root.iterdir() if p.is_dir())
    print(f"cloned {got}/{len(plan['repos'])} repos at depth "
          f"{depth or 'full'} in {time.time()-t0:.0f}s", flush=True)


# ── corpora beyond the working tree ─────────────────────────────────────────

def _git(d: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(d), *a],
                          capture_output=True, text=True).stdout


def _deletions(d: Path, exts: set) -> dict[str, tuple[str, str, str]]:
    """path -> (commit, date, subject) for the commit that removed it.

    One `git log` pass rather than three calls per file. The per-file form in
    `history-tombstone-index/run.py` is fine for one repo with a dozen
    deletions; account-wide it is thousands of subprocess spawns (claude-
    workspace alone has 644 deleted files after the 2026-07-28 migration).

    Walks newest-first and keeps the FIRST commit seen for a path, which is the
    one that removed it last -- a file deleted, restored and deleted again is
    tombstoned at its most recent death, not its first.
    """
    out: dict[str, tuple[str, str, str]] = {}
    raw = _git(d, "log", "--diff-filter=D", "--name-only",
               "--format=%x00%H%x1f%ad%x1f%s", "--date=short")
    for rec in raw.split("\x00")[1:]:
        head, _, files = rec.partition("\n")
        parts = head.split("\x1f")
        if len(parts) != 3:
            continue
        commit, date, subject = parts
        for f in files.split("\n"):
            f = f.strip()
            if f and f not in out and Path(f).suffix in exts:
                out[f] = (commit, date, subject)
    return out


def admissible(rel: str, cfg: dict) -> bool:
    """Would `hcindex.discover` have kept this path if the file still existed?

    A deleted file gets no `stat()` and no `rglob`, so every filter `discover`
    applies to the working tree has to be re-applied here by hand -- extension,
    skip_dirs, skip_names, and the repo's own `exclude` list -- with the size cap
    checked against the retrieved body at the call site.

    Measuring first is what found this. Without the filters, claude-workspace
    contributed **74,736** tombstone chunks against 232 from its working tree:
    the 2026-07-28 experiments migration deleted embedding dumps like
    `phase-a-bridges/data/full_body_embeddings.json` at 767,692 lines, and
    `.json` is an indexed extension. Those files were never in the live index
    either -- `discover` drops them at the 1 MiB cap -- so the tombstone corpus
    was not surfacing lost knowledge, it was importing the exact rows the tree
    already refuses. That is 1.7x the entire account index, from one repo, for
    deleted machine-generated data.
    """
    p = Path(rel)
    if p.suffix not in set(cfg["extensions"]) or p.name in set(cfg["skip_names"]):
        return False
    if any(s in p.parts for s in cfg["skip_dirs"]):
        return False
    return not any(H.match(rel, pat) for pat in cfg["exclude"])


def live_index(names, root: Path, cfg: dict) -> dict[str, list[set]]:
    """basename -> line-sets of every live file with that name, account-wide.

    Used to drop relocated files from the tombstone corpus. A moved file looks
    deleted at its old path, so indexing it as a tombstone puts content that
    still exists into the "gone" corpus -- measured at 27% of one repo's
    tombstones, and one of them answered a mechanism query from the working
    tree, which is what exposed it.

    Two changes from the per-repo version this generalizes:

    - The comparison is **account-wide**, not within one repo. The 37 research
      projects that moved from claude-workspace to experiments on 2026-07-28
      are deleted in one repo and live in another; a per-repo guard sees only
      the deletion and would tombstone ~640 files whose content is indexed
      three feet away.
    - Candidates are restricted **by basename** instead of comparing every dead
      file against every live one. That is O(dead x live) set intersections
      account-wide -- tens of millions. Relocation keeps the filename in every
      case observed (`src/remax/bench/x.py` -> `bench/x.py`,
      `experiments/foo/RESULTS.md` -> `foo/RESULTS.md`), so a same-basename
      restriction costs nothing real and makes the check linear. A rename that
      also changes the basename is missed and gets tombstoned; that is the
      conservative direction, and it is bounded.
    """
    exts = set(cfg["extensions"])
    by_base: dict[str, list[set]] = {}
    for name in sorted(names):
        d = root / name
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in exts:
                continue
            if any(s in p.relative_to(d).parts for s in cfg["skip_dirs"]):
                continue
            try:
                if p.stat().st_size > cfg["max_bytes"]:
                    continue
                lines = p.read_text(errors="ignore").split("\n")
            except OSError:
                continue
            by_base.setdefault(p.name, []).append(
                {ln for ln in lines if len(ln.strip()) > 20})
    return by_base


def tombstone_chunks(name: str, d: Path, cfg: dict,
                     by_base: dict[str, list[set]]) -> list:
    """Chunks for every file deleted and never restored, at its last living revision.

    The chunk header carries the removing commit and its subject, because the
    message is often the highest-signal part of a deletion -- remax's own read
    "Delete the ten Nemotron/NVFP4 bench drivers, keep every conclusion".

    Paths are `<repo>/[deleted] <path>` so `xr -r <repo>` still scopes them:
    that filter is a `startswith(repo + "/")` on the stored path, so a label
    that does not lead with the repo silently drops out of every scoped query.
    """
    exts = set(cfg["extensions"])
    out, moved, filtered = [], 0, 0
    for f, (commit, date, subject) in sorted(_deletions(d, exts).items()):
        if (d / f).exists() or not admissible(f, cfg):
            continue
        body = _git(d, "show", f"{commit}^:{f}")
        if not body.strip():
            continue
        if len(body.encode("utf-8", "ignore")) > cfg["max_bytes"]:
            filtered += 1
            continue
        want = {ln for ln in body.split("\n") if len(ln.strip()) > 20}
        if want and any(len(want & have) / len(want) > 0.5
                        for have in by_base.get(Path(f).name, ())):
            moved += 1
            continue
        head = f"{f} [DELETED {date} in {commit[:8]}: {subject}]"
        lines = body.split("\n")
        for s in range(0, max(1, len(lines)), cfg["stride"]):
            seg = "\n".join(lines[s:s + cfg["win"]]).strip()
            if len(seg) >= cfg["min_chars"]:
                out.append(H.Chunk(f"{name}/[deleted] {f}", s + 1,
                                   f"{head}\n{seg[:cfg['max_chars']]}"))
            if s + cfg["win"] >= len(lines):
                break
    if moved or filtered:
        print(f"    {name}: skipped {moved} relocated, {filtered} oversized",
              flush=True)
    return out


def fetch_pr_bodies(owner: str, names, token: str) -> dict[str, list[dict]]:
    """Merged PR number/title/date/body for each repo.

    Fetched in `plan` and carried in plan.json rather than fetched where it is
    used, for two reasons. One is cost: the sharded rebuild would otherwise make
    this many API calls once per shard, in parallel, for identical data. The
    other is correctness, and it is the load-bearing one -- shard and merge each
    rebuild the corpus independently and match rows by content hash, so a PR
    body edited between the two would change the chunk text, miss its hash, and
    trip the merge's coverage check. Freezing the bodies in the shared artifact
    makes them the same bytes for every job in the run.
    """
    out: dict[str, list[dict]] = {}
    failed = []
    for name in sorted(names):
        try:
            prs = api(f"repos/{owner}/{name}/pulls?state=all", token)
        except Exception as e:  # noqa: BLE001 - one repo must not fail the plan
            failed.append(f"{name} ({type(e).__name__})")
            continue
        out[name] = [{"n": p["number"], "t": p["title"] or "",
                      "b": p.get("body") or "",
                      "at": (p.get("merged_at") or p["created_at"])[:10]}
                     for p in prs if p.get("merged_at")]
    # PR bodies are the one corpus not in git, so this step makes the network
    # and a token a dependency of a rebuild that is otherwise pure filesystem.
    # Degrading to the offline corpora is the right behaviour and it is what
    # happens above -- but degrading *quietly* would publish a smaller index
    # that still verifies and still answers, which is the failure this whole
    # line of work keeps hitting. So say it, loudly, in the runner's log.
    if failed:
        print(f"::warning::PR bodies unavailable for {len(failed)}/{len(names)} "
              f"repos; building without them: {', '.join(failed[:6])}")
    return out


def pr_chunks(name: str, prs: list[dict], cfg: dict) -> list:
    """One chunk per ~2000 chars of merged PR body, headed with number/title/date.

    A PR body has no line semantics worth preserving, so `s` is the character
    offset -- enough to point a reader at the right part of the thread.

    **What this is worth depends on how the PRs were written.** Measured 6/8 ->
    8/8 on rationale questions against remax, whose PR bodies are Claude-authored
    with a median of 2,727 chars and none empty. A repo whose PRs say "fixes #12"
    contributes noise at the same chunk cost, which is why bodies shorter than
    `min_chars` are stored as a bare title line rather than a body chunk.
    """
    out = []
    for p in prs:
        head = f"PR #{p['n']} [{p['at']}]: {p['t']}"
        label = f"{name}/[PR #{p['n']}] {p['t'][:60]}"
        body = p["b"].strip()
        if len(body) < cfg["min_chars"]:
            out.append(H.Chunk(label, 0, head))
            continue
        for i in range(0, len(body), cfg["max_chars"]):
            out.append(H.Chunk(label, i, f"{head}\n{body[i:i + cfg['max_chars']]}"))
    return out


def corpus_for(plan: dict, root: Path, cfg: dict) -> list:
    """Chunk every repo in `plan` under `root`, namespacing each path with its repo.

    Shared by build/shard/merge so all three produce byte-identical chunk text
    for the same file -- which is what makes a shard's rows reusable by the
    merge. Any divergence here silently becomes a re-encode.

    Which extra corpora are included is read from the plan, never from a flag on
    this command. Three jobs build this corpus independently in a sharded
    rebuild, and a flag that reached one of them and not the others would not
    error -- the merge would just re-encode everything it could not hash-match.
    """
    names = plan["repos"]
    extra = plan.get("corpora", {})
    chunks = []
    by_base = live_index(names, root, cfg) if extra.get("tombstones") else {}
    for name in sorted(names):
        d = root / name
        if not d.is_dir():
            print(f"  missing clone: {name}")
            continue
        c = dict(cfg); c.update(H.load_cfg(d)); c["extensions"] = cfg["extensions"]
        for ch in H.build_corpus(d, c):
            chunks.append(H.Chunk(f"{name}/{ch.f}", ch.s, ch.text))
        if extra.get("tombstones"):
            chunks += tombstone_chunks(name, d, c, by_base)
        if extra.get("prs"):
            chunks += pr_chunks(name, plan.get("pr_bodies", {}).get(name, []), c)
    return chunks


def cmd_shard(a) -> None:
    """Encode every n-th chunk of the whole corpus. Emits codes + hashes only.

    Sharding is by **chunk**, not by repo. Repo-level sharding was written first
    and measured worse: this account's largest repo is 11,155 chunks of ~42,500,
    so any split that keeps repos whole has a floor at ~26% of the serial build
    no matter how many runners you add. `chunks[i::n]` has no floor -- doubling
    the shard count halves the critical path until per-job overhead dominates.

    The cost is that every shard clones and chunks the whole account rather than
    its own slice. That is much cheaper than it sounds and it happens in
    parallel: clone measured 89 s, chunking ~10 s, against an encode measured in
    tens of minutes.

    BM25 needs the whole corpus (postings index into a single chunk list), so
    the lexical arm is built once in `merge`. That costs nothing either: fitting
    BM25 measured 0.3 s against a 36 s dense encode, under 1% of build time.
    """
    import remex
    plan = json.loads(Path(a.plan).read_text())
    allc = corpus_for(plan, Path(a.dir), dict(H.DEFAULT_CFG))
    chunks = allc[a.i::a.n]
    print(f"shard {a.i}/{a.n}: {len(chunks)} of {len(allc)} chunks", flush=True)

    from ask import Encoder
    enc = Encoder()
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROT)
    t0 = time.time()
    codes, hashes, n_enc, _ = H.incremental(
        chunks, lambda ts: qz.encode(enc(ts, batch=16)).indices)
    print(f"  encoded {n_enc} in {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(a.out, codes=codes, hashes=np.array(hashes))
    print(f"  wrote {Path(a.out).stat().st_size/2**20:.1f} MB")


def cmd_merge(a) -> None:
    """Fold every shard's rows into one index, then fit BM25 over the whole corpus.

    Re-chunks all repos rather than shipping chunk *texts* between jobs: the
    texts are ~100 MB of artifact traffic, while re-chunking measured ~10 s.
    Every row then matches a shard row by content hash, so the encode count
    here should be 0 -- and if it is not, the shards and the merge disagreed
    about chunking, which is exactly the bug worth failing on.
    """
    import remex
    plan = json.loads(Path(a.plan).read_text())
    chunks = corpus_for(plan, Path(a.dir), dict(H.DEFAULT_CFG))
    print(f"corpus: {len(chunks)} chunks from {len(plan['repos'])} repos", flush=True)

    codes_in, hashes_in = [], []
    for p in sorted(Path(a.shards).glob("**/shard-*.npz")):
        z = np.load(p, allow_pickle=False)
        codes_in.append(z["codes"]); hashes_in += z["hashes"].tolist()
        print(f"  {p.name}: {len(z['hashes'])} rows", flush=True)
    if not codes_in:
        raise SystemExit("no shard artifacts found; refusing to build an empty index")
    prev_codes = np.concatenate(codes_in)
    print(f"  {len(hashes_in)} shard rows total", flush=True)

    # Check coverage BEFORE encoding, not after. A lost shard artifact used to
    # surface as `encoded 663` only once the merge had already spent that
    # encode -- the diagnosis was right but arrived after paying for the thing
    # it was diagnosing. Hashing is seconds.
    want = {H.chunk_hash(c.text) for c in chunks}
    missing = want - set(hashes_in)
    if missing and not a.allow_reencode:
        raise SystemExit(
            f"{len(missing)} of {len(want)} distinct chunks are in no shard "
            f"({len(codes_in)} shard files loaded). A shard artifact is missing, "
            f"or the shards and this merge disagreed about chunking. Fix that "
            f"rather than papering over it with --allow-reencode.")

    from ask import Encoder
    enc = Encoder()
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROT)
    t0 = time.time()
    codes, hashes, n_enc, n_reused = H.incremental(
        chunks, lambda ts: qz.encode(enc(ts, batch=16)).indices,
        prev_codes=prev_codes, prev_hashes=hashes_in)
    print(f"  encoded {n_enc}, reused {n_reused} in {time.time()-t0:.0f}s", flush=True)

    bm = H.BM25().fit(chunks)
    repo_chunks: dict[str, int] = {}
    for c in chunks:
        r = c.f.split("/")[0]
        repo_chunks[r] = repo_chunks.get(r, 0) + 1
    meta = {"dim": DIM, "bits": BITS, "seed": SEED, "rotation": ROT,
            "n_chunks": len(chunks), "repos": plan["repos"],
            "repo_chunks": repo_chunks,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # which corpora beyond the working tree this index contains, so a
            # consumer can tell an index without tombstones from one whose
            # tombstones happen to miss -- absent means working tree only
            "corpora": plan.get("corpora", {}),
            "encoder_sha256": __import__("ask").MODEL_SHA256,
            # lets a consumer refuse a drifted tokenizer instead of silently
            # losing the lexical arm -- see tokenizer_sha256() above
            "tokenizer_sha256": tokenizer_sha256()}
    H.save(Path(a.out), chunks, codes, bm, hashes, meta)
    Path(a.manifest).write_text(json.dumps(meta, indent=1))
    print(f"  wrote {Path(a.out).stat().st_size/2**20:.1f} MB")


def cmd_build(a) -> None:
    import remex
    plan = json.loads(Path(a.plan).read_text())
    root = Path(a.dir)

    chunks = corpus_for(plan, root, dict(H.DEFAULT_CFG))
    print(f"corpus: {len(chunks)} chunks from {len(plan['repos'])} repos",
          flush=True)

    prev_codes = prev_hashes = None
    p = Path(a.prev)
    if p.exists():
        try:
            _, prev_codes, _, prev_hashes, _ = H.load(p)
            print(f"  reusing {len(prev_hashes)} prior rows", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  previous index unreadable ({type(e).__name__}); full encode")

    from ask import Encoder
    enc = Encoder()
    qz = remex.Quantizer(d=DIM, bits=BITS, seed=SEED, rotation=ROT)
    t0 = time.time()
    codes, hashes, n_enc, n_reused = H.incremental(
        chunks, lambda ts: qz.encode(enc(ts, batch=16)).indices,
        prev_codes=prev_codes, prev_hashes=prev_hashes)
    print(f"  encoded {n_enc}, reused {n_reused} in {time.time()-t0:.0f}s",
          flush=True)

    bm = H.BM25().fit(chunks)
    repo_chunks: dict[str, int] = {}
    for c in chunks:
        r = c.f.split("/")[0]
        repo_chunks[r] = repo_chunks.get(r, 0) + 1
    meta = {"dim": DIM, "bits": BITS, "seed": SEED, "rotation": ROT,
            "n_chunks": len(chunks), "repos": plan["repos"],
            # measured per-repo counts: the weights the next shard split uses
            "repo_chunks": repo_chunks,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # which corpora beyond the working tree this index contains, so a
            # consumer can tell an index without tombstones from one whose
            # tombstones happen to miss -- absent means working tree only
            "corpora": plan.get("corpora", {}),
            "encoder_sha256": __import__("ask").MODEL_SHA256,
            # lets a consumer refuse a drifted tokenizer instead of silently
            # losing the lexical arm -- see tokenizer_sha256() above
            "tokenizer_sha256": tokenizer_sha256()}
    H.save(Path(a.out), chunks, codes, bm, hashes, meta)
    Path(a.manifest).write_text(json.dumps(meta, indent=1))
    print(f"  wrote {Path(a.out).stat().st_size/2**20:.1f} MB")


def cmd_corpora(a) -> None:
    """Report what each extra corpus would add, without encoding anything.

    The risk named for tombstones and PR bodies at account scale is not answer
    quality -- an unrelated corpus measured inert rather than harmful -- it is
    size, in an index whose known weakness is already crowding. Size is the half
    of that question that costs seconds instead of a 22 minute sharded encode,
    so it is worth answering on its own before committing a runner to the other
    half.

    Reads the same plan every other command reads, and ignores its `corpora`
    setting: the point is to compare against what is currently shipped.
    """
    plan = json.loads(Path(a.plan).read_text())
    root = Path(a.dir)
    both = dict(plan); both["corpora"] = {"tombstones": True, "prs": True}
    # Built once through the same path a real build uses, then partitioned by
    # label. Chunking three separate corpora and subtracting lengths would be
    # wrong as well as slower: corpus_for interleaves per repo, so the extra
    # chunks are not a suffix to slice off.
    rows: dict[str, list] = {"tree": [], "tombstones": [], "prs": []}
    for c in corpus_for(both, root, dict(H.DEFAULT_CFG)):
        rest = c.f.partition("/")[2]
        rows["tombstones" if rest.startswith("[deleted] ")
             else "prs" if rest.startswith("[PR #") else "tree"].append(c)

    n = len(rows["tree"])
    print(f"\n{'corpus':14s} {'chunks':>9s} {'% of tree':>10s} {'chars':>12s}")
    print("-" * 48)
    out = {}
    for k, cs in rows.items():
        chars = sum(len(c.text) for c in cs)
        print(f"{k:14s} {len(cs):>9,} {100*len(cs)/max(1, n):>9.1f}% {chars:>12,}")
        out[k] = {"chunks": len(cs), "chars": chars,
                  "pct_of_tree": round(100 * len(cs) / max(1, n), 1)}
    added = len(rows["tombstones"]) + len(rows["prs"])
    print(f"{'tree + both':14s} {n + added:>9,} {100*(n+added)/max(1, n):>9.1f}%")
    out["added"] = {"chunks": added, "pct_of_tree": round(100 * added / max(1, n), 1)}

    # Per-repo, because the decision may not be uniform: a repo whose PR bodies
    # are "fixes #12" pays the same chunk cost as one whose are 2,727 chars.
    per: dict[str, dict[str, int]] = {}
    for k, cs in rows.items():
        for c in cs:
            per.setdefault(c.f.split("/")[0], {}).setdefault(k, 0)
            per[c.f.split("/")[0]][k] += 1
    print(f"\n{'repo':30s} {'tree':>7s} {'tombs':>7s} {'PRs':>7s}")
    for r, d in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:20]:
        print(f"{r:30s} {d.get('tree', 0):>7,} {d.get('tombstones', 0):>7,} "
              f"{d.get('prs', 0):>7,}")
    if a.out:
        Path(a.out).write_text(json.dumps({"totals": out, "per_repo": per}, indent=1))


def cmd_verify(a) -> None:
    """Refuse to publish an index that would silently replace a working one.

    A rebuild that fails quietly is the failure class this whole line of work
    kept hitting, so the checks are about *plausibility*, not just readability.
    """
    chunks, codes, bm, hashes, meta = H.load(Path(a.index))
    fails = []
    if len(chunks) != meta["n_chunks"]:
        fails.append(f"chunk count {len(chunks)} != manifest {meta['n_chunks']}")
    if len(chunks) < 1000:
        fails.append(f"only {len(chunks)} chunks — suspiciously small")
    if codes.shape != (len(chunks), meta["dim"]):
        fails.append(f"codes {codes.shape} != ({len(chunks)}, {meta['dim']})")
    if len(set(hashes)) < 0.5 * len(hashes):
        fails.append("over half the chunk hashes are duplicates")
    repos_seen = {c.f.split("/")[0] for c in chunks}
    missing = set(meta["repos"]) - repos_seen
    if len(missing) > 0.2 * len(meta["repos"]):
        fails.append(f"{len(missing)} repos contributed no chunks: "
                     f"{sorted(missing)[:8]}")
    if not bm.postings:
        fails.append("BM25 postings are empty")
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        raise SystemExit("index failed verification; refusing to publish")
    print(f"ok: {len(chunks)} chunks, {len(repos_seen)} repos, "
          f"{len(bm.postings)} terms, built {meta['built_at']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan");  p.add_argument("--owner", required=True)
    p.add_argument("--prev", required=True); p.add_argument("--out", required=True)
    # Both off by default: measured wins per-repo, unmeasured at account scale
    # where crowding is the known weakness. `corpora` sizes them for free.
    p.add_argument("--with-tombstones", action="store_true",
                   help="index deleted files too (forces a full-history clone)")
    p.add_argument("--with-prs", action="store_true",
                   help="index merged PR bodies too (fetched here, frozen in the plan)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("clone"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.set_defaults(fn=cmd_clone)
    p = sub.add_parser("build"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.add_argument("--prev", required=True)
    p.add_argument("--out", required=True); p.add_argument("--manifest", required=True)
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("corpora"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.add_argument("--out")
    p.set_defaults(fn=cmd_corpora)
    p = sub.add_parser("verify"); p.add_argument("--index", required=True)
    p.add_argument("--manifest", required=True); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("shard"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.add_argument("-i", type=int, required=True)
    p.add_argument("-n", type=int, required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_shard)
    p = sub.add_parser("merge"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--allow-reencode", action="store_true",
                   help="tolerate chunks no shard produced (normally a chunking bug)")
    p.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    a.fn(a)
