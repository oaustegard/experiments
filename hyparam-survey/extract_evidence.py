#!/usr/bin/env python3
"""Extract the hypvector source passages this survey cites, from the pinned npm tarball.

The hypvector GitHub repo is private; the npm tarball is the only public source.
This downloads hypvector@PINNED_VERSION, verifies its sha256, and writes each
cited passage to evidence/ with its file and line range, so every claim in
README.md can be checked without npm and without trusting my paraphrase.

    python3 extract_evidence.py            # download, verify, extract
    python3 extract_evidence.py --check    # verify extracted files still match

Prints the tarball hash it saw. If it differs from PINNED_SHA256, npm republished
or the pin is stale — do not silently accept it, update the pin deliberately.
"""
import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

PINNED_VERSION = "0.2.2"
PINNED_SHA256 = "5ac1eb31e4e81b19a663349a285b95be7bbcbf6ad283559882b2f62360a27f62"
TARBALL_URL = f"https://registry.npmjs.org/hypvector/-/hypvector-{PINNED_VERSION}.tgz"
HERE = Path(__file__).parent
EVIDENCE = HERE / "evidence"

# Each excerpt: output filename -> (source path in package, anchor, lines_after)
# The anchor is a literal substring; the excerpt starts at the line containing it.
# Anchoring on content rather than line numbers means a version bump relocates
# the excerpt instead of silently grabbing the wrong bytes.
EXCERPTS = [
    (
        "cluster-reorder.js",
        "package/src/cluster.js",
        "export function reorderClustersByHamming",
        30,
        ("Greedy nearest-neighbour renumbering of cluster ids so adjacent ids are "
        "close in Hamming space. README: 'Greedy Hamming renumbering'."),
    ),
    (
        "cluster-kmeans-head.js",
        "package/src/cluster.js",
        "export function binaryKMeans",
        40,
        ("Binary k-means over the 1-bit codes: Hamming assignment, bit-majority-vote "
        "centroid update. README: 'runs k-means on the 1-bit codes'."),
    ),
    (
        "ranges-select.js",
        "package/src/search/ranges.js",
        "export function selectClusterRowRanges",
        40,
        ("Phase-1 cluster selection: rank centroids by Hamming, take the top probe "
        "fraction, merge their row ranges. README: query path step 1."),
    ),
    (
        "ranges-coalesce.js",
        "package/src/search/ranges.js",
        "export function coalesceRuns",
        22,
        ("Phase-2 run coalescing with maxGap. Called with 64 in rerank.js. "
        "README: 'merging gaps <= 64 rows'."),
    ),
    (
        "constants-probe-cap.js",
        "package/src/constants.js",
        "// Upper bound on clusters probed",
        18,
        ("THE CLAIM THIS SURVEY TESTS: 'Residual misses are a rerankFactor limit, "
        "not a probe limit.' Also the defaultClusterProbeCap = 48 rationale."),
    ),
    (
        "constants-binary-spread.js",
        "package/src/constants.js",
        "// Auto-binary also requires the sign-bit codes to discriminate",
        16,
        ("Degeneracy guard rationale: below dimension * 1/16 expected Hamming, "
        "phase-1 ranking degenerates. README: 'A degeneracy guard on the sign codes'."),
    ),
    (
        "write-auto-defaults.js",
        "package/src/writeVectors.js",
        "  // Resolve auto defaults now that we know N",
        20,
        ("Auto-binary threshold, the expectedHamming spread check, and "
        "clusterCount = round(sqrt(N)/2). README: write path."),
    ),
    (
        "rerank-phases.js",
        "package/src/search/rerank.js",
        "  // Phase 1: Hamming scan over selected ranges",
        60,
        ("Phases 1 and 2 in full: range scan, candidate set, coalesced runs, "
        "useOffsetIndex float32 fetch. README: query path steps 1-2."),
    ),
    (
        "rerank-defer-ids.js",
        "package/src/search/rerank.js",
        "  // Phase 3: fetch ids for just the top-K winners",
        4,
        "Phase 3 id deferral. README: 'Fetch the id column for the top-K winners only'.",
    ),
]


def fetch_tarball() -> bytes:
    with urllib.request.urlopen(TARBALL_URL, timeout=120) as resp:
        return resp.read()


def excerpt(text: str, anchor: str, lines_after: int, source: str) -> tuple[str, int, int]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if anchor in line:
            # include a preceding JSDoc/comment block if one is attached
            start = i
            while start > 0 and lines[start - 1].strip().startswith(("*", "/**", "//")):
                start -= 1
            end = min(len(lines), i + lines_after)
            return "\n".join(lines[start:end]), start + 1, end
    raise SystemExit(f"anchor not found in {source}: {anchor!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify evidence/ matches the pin")
    args = ap.parse_args()

    raw = fetch_tarball()
    got = hashlib.sha256(raw).hexdigest()
    print(f"hypvector@{PINNED_VERSION} tarball sha256 {got}")
    if got != PINNED_SHA256:
        print(f"MISMATCH: pinned {PINNED_SHA256}", file=sys.stderr)
        return 1

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        sources = {m.name: tar.extractfile(m).read().decode()
                   for m in tar.getmembers() if m.isfile()}

    EVIDENCE.mkdir(exist_ok=True)
    manifest = {
        "package": f"hypvector@{PINNED_VERSION}",
        "tarball_url": TARBALL_URL,
        "tarball_sha256": PINNED_SHA256,
        "note": "The hypvector GitHub repo is private. This npm tarball is the public source.",
        "excerpts": [],
    }
    failures = []

    for name, path, anchor, after, why in EXCERPTS:
        if path not in sources:
            raise SystemExit(f"missing from tarball: {path}")
        body, start, end = excerpt(sources[path], anchor, after, path)
        header = (
            f"// {path.removeprefix('package/')}:{start}-{end}\n"
            f"// hypvector@{PINNED_VERSION}, sha256 {PINNED_SHA256[:16]}...\n"
            f"// why cited: {why}\n\n"
        )
        out = EVIDENCE / name
        content = header + body + "\n"
        if args.check:
            if not out.exists() or out.read_text() != content:
                failures.append(name)
        else:
            out.write_text(content)
        manifest["excerpts"].append({
            "file": name,
            "source": path.removeprefix("package/"),
            "lines": f"{start}-{end}",
            "why_cited": why,
        })

    mpath = EVIDENCE / "MANIFEST.json"
    mtext = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        if not mpath.exists() or mpath.read_text() != mtext:
            failures.append("MANIFEST.json")
        if failures:
            print("STALE: " + ", ".join(failures), file=sys.stderr)
            return 1
        print(f"OK  {len(EXCERPTS)} excerpts match the pinned tarball")
    else:
        mpath.write_text(mtext)
        print(f"wrote {len(EXCERPTS)} excerpts + MANIFEST.json to evidence/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
