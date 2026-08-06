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
    now = {r["full_name"].split("/")[1]: r["pushed_at"] for r in repos
           # forks are upstream code, not this account's decisions; archived
           # repos never change so they cost nothing to keep
           if not r.get("fork") and not r.get("disabled")}
    prev = json.loads(Path(a.prev).read_text()).get("repos", {})
    changed = sorted(n for n, t in now.items() if prev.get(n) != t)
    unchanged = sorted(set(now) - set(changed))
    plan = {"repos": now, "changed": changed, "unchanged": unchanged,
            "owner": a.owner}
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"{len(now)} repos: {len(changed)} changed, {len(unchanged)} unchanged",
          flush=True)
    if changed:
        print("  changed: " + ", ".join(changed[:12])
              + (" ..." if len(changed) > 12 else ""))


def cmd_clone(a) -> None:
    plan = json.loads(Path(a.plan).read_text())
    token = os.environ["GH_TOKEN"]
    root = Path(a.dir); root.mkdir(parents=True, exist_ok=True)
    helper = ("!f() { echo username=x-access-token; "
              f'echo password={token}; }}; f')
    t0 = time.time()
    for name in plan["repos"]:
        dest = root / name
        if dest.exists():
            continue
        # --depth 50 keeps enough history for the tombstone corpus to see recent
        # deletions without paying for the full clone
        subprocess.run(["git", "-c", f"credential.helper={helper}", "clone", "-q",
                        "--depth", "50",
                        f"https://github.com/{plan['owner']}/{name}.git", str(dest)],
                       check=False, capture_output=True)
    got = sum(1 for p in root.iterdir() if p.is_dir())
    print(f"cloned {got}/{len(plan['repos'])} repos in {time.time()-t0:.0f}s",
          flush=True)


def cmd_build(a) -> None:
    import remex
    plan = json.loads(Path(a.plan).read_text())
    root = Path(a.dir)

    cfg = dict(H.DEFAULT_CFG)
    chunks: list[H.Chunk] = []
    for name in sorted(plan["repos"]):
        d = root / name
        if not d.is_dir():
            print(f"  missing clone: {name}")
            continue
        c = dict(cfg); c.update(H.load_cfg(d)); c["extensions"] = cfg["extensions"]
        for ch in H.build_corpus(d, c):
            # namespace the path so results say which repo they came from
            chunks.append(H.Chunk(f"{name}/{ch.f}", ch.s, ch.text))
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
    meta = {"dim": DIM, "bits": BITS, "seed": SEED, "rotation": ROT,
            "n_chunks": len(chunks), "repos": plan["repos"],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "encoder_sha256": __import__("ask").MODEL_SHA256}
    H.save(Path(a.out), chunks, codes, bm, hashes, meta)
    Path(a.manifest).write_text(json.dumps(meta, indent=1))
    print(f"  wrote {Path(a.out).stat().st_size/2**20:.1f} MB")


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
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("clone"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.set_defaults(fn=cmd_clone)
    p = sub.add_parser("build"); p.add_argument("--plan", required=True)
    p.add_argument("--dir", required=True); p.add_argument("--prev", required=True)
    p.add_argument("--out", required=True); p.add_argument("--manifest", required=True)
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("verify"); p.add_argument("--index", required=True)
    p.add_argument("--manifest", required=True); p.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    a.fn(a)
