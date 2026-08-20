#!/usr/bin/env python3
"""Fill the two template families from disjoint entity pools.

Family A entities and family B entities share no owner, repo, path, branch, tag,
sha or free-text term. That matters because the failure mode a fitter falls into
is memorising an entity: if `cli/cli` only ever appeared in `list_branches` rows,
an open-vocabulary feature set would happily learn it.

    python3 gen_data.py --n 12 --out data/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from templates import OFF_TOPIC, T

HERE = Path(__file__).resolve().parent

POOLS = {
 "a": {
  "or": ["oaustegard/experiments", "github/github-mcp-server", "anthropics/claude-code",
         "pallets/flask", "psf/requests"],
  "path": ["src/router.py", "pkg/github/actions.go", "docs/architecture.md",
           "internal/parser.ts", "lib/encode.rs"],
  "branch": ["feature/retry", "main", "dev", "fix/timeout", "release/v2"],
  "base": ["main", "develop", "trunk"],
  "tag": ["v2.3.1", "v0.9.0", "v14.2.0"],
  "wf": ["ci.yml", "build-layers.yml", "release.yml"],
  "q": ["retry backoff", "token refresh", "memory leak", "flaky test", "rate limit"],
  "title": ["fix the retry loop", "drop the legacy shim", "add pagination"],
  "org": ["anthropics", "github", "pallets"],
  "team": ["platform", "infra", "security"],
  "user": ["oaustegard", "simonw", "kentcdodds"],
 },
 "b": {
  "or": ["torvalds/linux", "rust-lang/cargo", "vercel/next.js", "astral-sh/ruff",
         "denoland/deno"],
  "path": ["cmd/serve/main.go", "app/models/user.rb", "test/helpers.js",
           "crates/core/lib.rs", "README.md"],
  "branch": ["hotfix/auth", "next", "canary", "chore/deps", "wip/parser"],
  "base": ["master", "stable", "release"],
  "tag": ["v1.0.4", "v3.11.2", "v0.4.7"],
  "wf": ["test.yaml", "publish.yml", "nightly.yml"],
  "q": ["websocket handshake", "unicode normalisation", "cold start", "index corruption",
        "signature mismatch"],
  "title": ["support nested configs", "remove the deprecated flag", "speed up startup"],
  "org": ["rust-lang", "denoland", "vercel"],
  "team": ["compiler", "docs", "release-eng"],
  "user": ["dtolnay", "rakyll", "sindresorhus"],
 },
}

HEX = "0123456789abcdef"


def _fill(tmpl: str, fam: str, rng: random.Random) -> tuple[str, dict]:
    """Render one template, returning the query and the values it actually used."""
    p = POOLS[fam]
    used: dict[str, object] = {}

    def take(slot: str):
        if slot in used:
            return used[slot]
        if slot == "or":
            v = rng.choice(p["or"])
        elif slot == "pr":
            n = rng.randint(3, 4800)
            style = rng.random()
            repo = used.get("or") or rng.choice(p["or"])
            used["or"] = repo
            v = (f"#{n}" if style < 0.4 else
                 f"PR {n}" if style < 0.7 else
                 f"https://github.com/{repo}/pull/{n}")
            used["_pullNumber"] = n
        elif slot == "issue":
            n = rng.randint(3, 4800)
            style = rng.random()
            repo = used.get("or") or rng.choice(p["or"])
            used["or"] = repo
            v = (f"#{n}" if style < 0.4 else
                 f"issue {n}" if style < 0.7 else
                 f"https://github.com/{repo}/issues/{n}")
            used["_issue_number"] = n
        elif slot == "sha":
            n = 40 if rng.random() < 0.5 else 8
            v = "".join(rng.choice(HEX) for _ in range(n - 1)) + rng.choice("abcdef")
        elif slot in ("run", "job"):
            v = str(rng.randint(10_000_000, 99_999_999_9))
        elif slot == "thread":
            v = "PRRT_" + "".join(rng.choice("kwABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                                  for _ in range(12))
        elif slot == "q":
            v = f'"{rng.choice(p["q"])}"' if rng.random() < 0.5 else rng.choice(p["q"])
        elif slot == "user":
            v = "@" + rng.choice(p["user"])
        else:
            v = rng.choice(p[slot])
        used[slot] = v
        return v

    out, i = [], 0
    while i < len(tmpl):
        if tmpl[i] == "{":
            j = tmpl.index("}", i)
            out.append(str(take(tmpl[i + 1:j])))
            i = j + 1
        else:
            out.append(tmpl[i])
            i += 1
    q = "".join(out)
    # An owner/repo is nearly always present in real usage even when the phrasing
    # omits it; leave it out where the template did, so the fitter meets both.
    return q, used


def generate(fam: str, per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    idx = 0 if fam == "a" else 1
    rows = []
    for label, fams in sorted(T.items()):
        tmpls = fams[idx]
        for i in range(per_label):
            q, used = _fill(rng.choice(tmpls), fam, rng)
            args = {}
            if "or" in used:
                o, r = str(used["or"]).split("/", 1)
                args["owner"], args["repo"] = o, r
            for k in ("_pullNumber", "_issue_number"):
                if k in used:
                    args[k[1:]] = used[k]
            if "sha" in used:
                args["sha"] = used["sha"]
            if "path" in used:
                args["path"] = used["path"]
            if "tag" in used:
                args["tag"] = used["tag"]
            rows.append({"id": f"{fam}-r{len(rows)}", "query": q,
                         "label": label, "args": args})
    # Off-topic rows are used once each — no duplication, so the abstention
    # denominator is a count of distinct requests rather than of resamples.
    for i, q in enumerate(OFF_TOPIC[idx]):
        rows.append({"id": f"{fam}-off-{i}", "query": q, "label": None, "args": {}})
    rng.shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="rows per label per family")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args()

    a.out.mkdir(exist_ok=True)
    for fam in ("a", "b"):
        rows = generate(fam, a.n, a.seed + (0 if fam == "a" else 1))
        path = a.out / f"family_{fam}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        n_off = sum(r["label"] is None for r in rows)
        print(f"{path.name}: {len(rows)} rows ({len(rows) - n_off} routable, {n_off} off-topic)")

    # Overlap audit: the two pools must share no entity string.
    shared = {k: sorted(set(POOLS['a'][k]) & set(POOLS['b'][k])) for k in POOLS['a']}
    bad = {k: v for k, v in shared.items() if v}
    print("pool overlap:", bad or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
