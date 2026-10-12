"""Build the work queue from test results: cluster failures by signature.

    python3 swarm/queue.py RESULTS.json [RESULTS.json ...] [--top 30] [--json]

A signature is "<stage>: <normalised reason>", e.g.
    "parse: expected <id> in struct declarator"
Numbers, file:line:col, identifiers in quotes and hex constants are masked so
the same compiler bug in different tests lands in one cluster. Short
punctuation, C keywords and CamelCase names (AST node classes, e.g. a
"KeyError: 'StructRef'") in quotes are kept ("expected ';'" stays specific).

A work item is one cluster (or one shard of a big cluster) with 1-3
representative tests. Priority = ladder level of the suite, then cluster size,
minus a penalty for earlier failed attempts. Items whose key or tests are
already in flight are skipped, so two live workers never get the same cluster.
"""

from __future__ import annotations

if __name__ == "__main__":
    import _boot  # noqa: F401  (must precede stdlib imports that use `queue`)

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ladder level per suite-name prefix: higher runs first.
# Real programs rank with the basics: once the conformance suites saturate, a whole-program
# build failure (one root cause, hundreds of tests) is the highest-yield work in the queue.
LADDER = [("smoke", 6), ("c-testsuite", 5), ("chibicc", 5), ("sqlite", 5), ("lua", 5), ("quickjs", 5),
          ("doom", 5), ("zlib", 5), ("bzip2", 5), ("cjson", 5), ("tcc", 5), ("mbedtls", 5), ("redis", 5),
          ("cpython", 5), ("torture", 4), ("gcc", 4), ("csmith", 4), ("llvm", 4), ("linux", 3)]
DEFAULT_LEVEL = 2
SHARD_SIZE = 10        # a cluster of n tests gets min(MAX_SHARDS, n // SHARD_SIZE) shards
MAX_SHARDS = 6
MAX_TEST_ATTEMPTS = 3  # a test tried this many times without a merge is parked
REPS = 3               # representative tests per item
GATE_TARGETS = 20      # cluster members the gate checks for improvement

C_KEYWORDS = set("""auto break case char const continue default do double else enum extern float for
goto if inline int long register restrict return short signed sizeof static struct switch typedef union
unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn
_Static_assert _Thread_local __attribute__ __asm__ __extension__ __typeof__ __int128 __builtin_va_list
__builtin_va_arg __label__ typeof asm""".split())


def level(suite: str) -> int:
    for prefix, lv in LADDER:
        if suite.startswith(prefix):
            return lv
    return DEFAULT_LEVEL


def _quoted(m: re.Match) -> str:
    q, s = m.group(1), m.group(2)
    keep = (s in C_KEYWORDS or (len(s) <= 3 and not re.search(r"\w", s))
            or re.fullmatch(r"[A-Z][a-z]+(?:[A-Z][a-z]*)+|[A-Z][a-z]{3,}", s) is not None)  # AST class names
    return f"{q}{s}{q}" if keep else f"{q}<id>{q}"


def normalise(reason: str) -> str:
    r = (reason or "").strip().splitlines()[-1:] or [""]   # last line: exception / message
    r = r[0]
    r = re.sub(r"(?<![\w.])(?:/[\w.+-]+){2,}/?", "<path>", r)          # absolute paths
    r = re.sub(r"(?<![\w<])[\w.+-]+\.(?:c|h|s|S|o|i)\b", "<file>", r)   # foo.c
    r = re.sub(r"^\s*(?:<path>|<file>)(?::\d+)*:\s*(?:(?:fatal )?error:\s*)?", "", r)  # "f.c:1:2: error:"
    r = re.sub(r"\bline \d+", "line <n>", r)
    r = r.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    r = re.sub(r"(['\"`])(.*?)\1", _quoted, r)
    r = re.sub(r"\b0x[0-9a-fA-F]+\b", "<hex>", r)
    r = re.sub(r"\b\d+(\.\d+)?\b", "<n>", r)
    r = re.sub(r"(<path>|<file>)(\s*<(?:path|file)>)+", r"\1", r)
    r = re.sub(r"\s+", " ", r).strip(" :")
    r = re.sub(r"^\[\w+\]\s*", "", r)                 # "[parse] msg": stage is already in the signature
    if r.endswith("..."):                               # tests/run.py truncates at varying lengths:
        words = r[:-3].rstrip().split(" ")[:-1]         # drop the cut word, keep a stable prefix
        r = " ".join(words[:3]) + " ..."
    return r[:110] or "<no reason>"


def signature(rec: dict) -> str:
    stage = rec.get("stage") or "unknown"
    reason = normalise(rec.get("reason", ""))
    if reason.startswith(stage + ":"):
        reason = reason[len(stage) + 1:].strip()
    return f"{stage}: {reason}"


@dataclass
class WorkItem:
    key: str                  # unique cluster/shard key used for in-flight exclusion
    signature: str
    priority: float
    size: int                 # tests in the whole cluster (all shards)
    tests: list[str]          # 1-3 representatives the worker must fix
    targets: list[str]        # cluster members the gate checks (includes tests)
    reasons: dict[str, str] = field(default_factory=dict)   # test -> raw reason


def _shard(test_id: str, n: int) -> int:
    return int(hashlib.sha1(test_id.encode()).hexdigest(), 16) % n


def clusters(results: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tid, rec in results.items():
        if rec.get("status") in ("PASS", "SKIP"):
            continue
        out.setdefault(signature(rec), []).append(tid)
    return out


def build(results: dict, ledger: dict | None = None, source_size=None, limit: int = 0) -> list[WorkItem]:
    """Work items, best first. `ledger` is common.ledger(ctx): used for in-flight
    exclusion and attempt penalties. `source_size(test_id) -> int` ranks
    representatives (smallest source first, easiest for a small model)."""
    ledger = ledger or {}
    busy_keys, busy_tests = set(), set()
    attempts: dict[str, int] = {}
    sig_fails: dict[str, int] = {}
    for r in ledger.values():
        if r.get("status") in ("launched", "finished"):
            busy_keys.add(r.get("key"))
            busy_tests.update(r.get("tests", []))
        elif r.get("status") == "rejected" and str(r.get("reason", "")).startswith(("conflict", "main kept moving")):
            continue   # a good fix that lost a merge race: the next worker rebases it, no penalty
        elif r.get("status") in ("rejected", "failed", "reverted"):
            sig_fails[r.get("signature")] = sig_fails.get(r.get("signature"), 0) + 1
            # a worker that died without a result (rate limit, timeout) counts half
            for t in r.get("tests", []):
                attempts[t] = attempts.get(t, 0) + (1 if r.get("result") else 0.5)
    size_of = source_size or (lambda t: 0)
    items: list[WorkItem] = []
    for sig, tids in clusters(results).items():
        n = len(tids)
        # A fully concrete signature (nothing masked, e.g. one undefined symbol) is one root cause:
        # shards would only make several workers write the same fix and conflict.
        concrete = not re.search(r"<(id|n|file|path)>", sig)
        shards = 1 if concrete else max(1, min(MAX_SHARDS, n // SHARD_SIZE))
        for k in range(shards):
            members = [t for t in tids if shards == 1 or _shard(t, shards) == k]
            key = sig if shards == 1 else f"{sig} #{k + 1}/{shards}"
            if key in busy_keys:
                continue
            free = [t for t in members if t not in busy_tests and attempts.get(t, 0) < MAX_TEST_ATTEMPTS]
            if not free:
                continue
            lv = max(level(t.split("/", 1)[0]) for t in free)
            free.sort(key=lambda t: (-level(t.split("/", 1)[0]), attempts.get(t, 0), size_of(t), t))
            reps = free[:REPS]
            targets = reps + [t for t in members if t not in reps][:GATE_TARGETS - len(reps)]
            prio = 10 * lv + 3 * math.log2(1 + n) - 4 * sig_fails.get(sig, 0) / shards
            items.append(WorkItem(key, sig, round(prio, 2), n, reps, targets,
                                  {t: results[t].get("reason", "")[:300] for t in reps}))
    items.sort(key=lambda w: (-w.priority, w.key))
    return items[:limit] if limit else items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path, help="tests/run.py --json outputs")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="print items as JSON lines")
    a = ap.parse_args(argv)
    results: dict = {}
    for p in a.results:
        results.update(json.loads(p.read_text()))
    items = build(results, limit=a.top)
    for w in items:
        if a.json:
            print(json.dumps(asdict(w)))
        else:
            print(f"{w.priority:7.2f} n={w.size:<4} {w.key}\n         e.g. {', '.join(w.tests)}")
    fails = sum(1 for r in results.values() if r.get("status") not in ("PASS", "SKIP"))
    print(f"{len(results)} results, {fails} failing, {len(clusters(results))} signatures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
