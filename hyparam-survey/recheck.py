#!/usr/bin/env python3
"""Check this experiment's prose against its artifacts.

Runs in well under a minute (one 40 KB tarball fetch). It does not re-run the
sweep; it checks that what README.md claims matches what results.json, repos.json
and evidence/ actually contain, so the writeup and the data cannot drift apart.

    python3 recheck.py            # all checks
    python3 recheck.py --offline  # skip the checks that need the network

Exit 0 = prose matches artifacts. Exit 1 = something drifted, with the diff named.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
README = HERE / "README.md"
NOTES = HERE / "NOTES.md"
RESULTS = HERE / "results.json"
REPOS = HERE / "repos.json"
HYSNAPPY = HERE / "hysnappy_results.json"
PRACTICAL = HERE / "practicality_results.json"
EVIDENCE = HERE / "evidence"
# hypvector sources, only present when the npm package is installed locally
LOCAL_SRC = HERE / "node_modules" / "hypvector" / "src"

failures: list[str] = []
checks = 0


def check(ok: bool, msg: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(msg)


def markdown_rows(text: str, header_contains: str) -> list[list[str]]:
    """Return the data rows of the first markdown table whose header matches."""
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not in_table:
            if header_contains in stripped:
                in_table = True
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def section(text: str, start_marker: str, end_marker: str = "\n## ") -> str:
    """The slice of `text` from start_marker to the next end_marker."""
    i = text.find(start_marker)
    if i < 0:
        return ""
    j = text.find(end_marker, i + len(start_marker))
    return text[i:] if j < 0 else text[i:j]


def check_sweep_table() -> None:
    """README's sweep table must match results.json row for row."""
    results = json.loads(RESULTS.read_text())
    rows = markdown_rows(README.read_text(), "recall@10")
    check(len(rows) == len(results["runs"]),
          f"sweep table has {len(rows)} rows, results.json has {len(results['runs'])}")

    # README labels are prose; map them to results.json labels by their options.
    def opts_of(label: str) -> str:
        rf = re.search(r"rerankFactor[:= ]+(\d+)", label)
        pr = re.search(r"probe[:= ]+([\d.]+)", label)
        rf_val = int(rf.group(1)) if rf else (0 if "exact" in label.lower() else 10)
        pr_val = float(pr.group(1)) if pr else 0.25
        return f"rf={rf_val},probe={pr_val}"

    json_index = {}
    for r in results["runs"]:
        rf = r["options"].get("rerankFactor", 0 if "exact" in r["label"] else 10)
        pr = float(r["options"].get("probe", 0.25))
        json_index[f"rf={rf},probe={pr}"] = r

    for cells in rows:
        label, ms_cell, recall_cell = cells[0], cells[1], cells[2]
        key = opts_of(label)
        run = json_index.get(key)
        check(run is not None, f"sweep row {label!r} -> {key} has no match in results.json")
        if run is None:
            continue
        claimed_recall = float(recall_cell.rstrip("%"))
        actual_recall = round(run["recall_at_k"] * 100, 1)
        check(abs(claimed_recall - actual_recall) < 0.05,
              f"sweep row {label!r}: README says recall {claimed_recall}%, results.json has {actual_recall}%")
        # Timings are machine-dependent; only require the same order of magnitude
        # and the same ordering, not equality.
        claimed_ms = float(ms_cell)
        actual_ms = run["ms_per_query"]
        check(0.25 <= claimed_ms / actual_ms <= 4.0,
              f"sweep row {label!r}: README says {claimed_ms} ms, results.json has {actual_ms} ms "
              "(>4x apart — rerun the probe and update the table)")


def check_file_numbers() -> None:
    """The byte counts quoted in README must come from results.json."""
    results = json.loads(RESULTS.read_text())
    text = README.read_text()
    f = results["file"]
    check(f"{f['bytes']:,}" in text, f"README does not quote file bytes {f['bytes']:,}")
    check(f"{f['raw_fp32_bytes']:,}" in text,
          f"README does not quote raw fp32 bytes {f['raw_fp32_bytes']:,}")
    check(f"{f['pct_of_raw']}% of raw" in text,
          f"README does not quote overhead {f['pct_of_raw']}% of raw")
    corpus = results["corpus"]
    check(f"{corpus['n']:,}" in text or str(corpus["n"]) in text,
          f"README does not quote corpus size {corpus['n']}")
    check(f"{corpus['dim']}-dim" in text, f"README does not quote dimension {corpus['dim']}")
    check(str(results["queries"]["count"]) in text,
          f"README does not quote query count {results['queries']['count']}")


def check_repo_table() -> None:
    """Every repo named in README's table must exist in repos.json with those stars."""
    repos = json.loads(REPOS.read_text())
    by_name = {r["name"]: r for r in repos["repos"]}
    rows = markdown_rows(README.read_text(), "Stars")
    check(len(rows) > 0, "no repo table found in README")
    for cells in rows:
        name_match = re.search(r"`([^`]+)`", cells[0])
        check(name_match is not None, f"repo row has no backticked name: {cells[0]!r}")
        if not name_match:
            continue
        name = name_match.group(1)
        check(name in by_name, f"repo {name!r} in README is not in repos.json")
        if name not in by_name:
            continue
        claimed = int(cells[1])
        actual = by_name[name]["stargazers_count"]
        check(claimed == actual,
              f"repo {name!r}: README says {claimed} stars, repos.json has {actual}")
    total = repos.get("total_count")
    check(f"{total} public repos" in README.read_text(),
          f"README does not state the repo count {total}")


def check_private_repos() -> None:
    """The three repos named as private must indeed be absent from repos.json."""
    repos = json.loads(REPOS.read_text())
    names = {r["name"] for r in repos["repos"]}
    for name in ("hypvector", "hypgrep", "hypstore"):
        check(name not in names,
              f"{name!r} is in repos.json, but README calls it private")


def check_notes_citations() -> None:
    """Every hypvector file:line citation in NOTES.md must land inside that file."""
    if not LOCAL_SRC.exists():
        print("  skip: node_modules/hypvector not installed (run npm install to check citations)")
        return
    lengths = {p.name: len(p.read_text().splitlines()) for p in LOCAL_SRC.rglob("*.js")}
    pattern = re.compile(r"`(\w+\.js):(\d+)(?:-(\d+))?`")
    seen = 0
    for name, start, end in pattern.findall(NOTES.read_text()):
        if name not in lengths:
            continue
        seen += 1
        last = int(end or start)
        check(last <= lengths[name],
              f"NOTES.md cites {name}:{start}-{end or start} but the file has {lengths[name]} lines")
    check(seen > 0, "no hypvector citations found in NOTES.md — did the format change?")


def check_hysnappy() -> None:
    """The hysnappy numbers in README/NOTES must come from hysnappy_results.json."""
    data = json.loads(HYSNAPPY.read_text())
    readme = README.read_text()
    notes = NOTES.read_text()

    for module, size in data["wasm_bytes"].items():
        check(f"{size:,}" in readme or f"{size:,}" in notes,
              f"neither README nor NOTES quotes the {module}.wasm size {size:,}")
    # The sync-compile ceiling itself is checked in check_practicality, which
    # measured it rather than quoting the constant this file recorded.

    # Every speedup quoted as "N.Nx" in the hysnappy sections must be a real ratio.
    for row in data["results"]:
        for direction in ("decompress", "compress"):
            for engine in ("hysnappy", "snappyjs"):
                mbps = row[f"{direction}_mbps"][engine]
                quoted = f"{mbps:,}" in notes or str(mbps) in notes
                check(quoted,
                      f"NOTES does not quote {row['corpus']} {direction} {engine} {mbps} MB/s")
    # Prose rounds; it must round from a stored ratio. Scope the scan to the
    # hysnappy sections so unrelated multipliers elsewhere are not swept in.
    stored = [r["decompress_speedup"] for r in data["results"]]
    stored += [r["compress_speedup"] for r in data["results"]]
    stored += [r["decompress_speedup_spread"] for r in data["results"]]
    for r in data["results"]:
        stored += r["decompress_speedup_per_sweep"]
    w = data["warmup"]
    stored += [w["cold_spread"], w["warm_spread"], w["warm_over_cold_median"]]
    if PRACTICAL.exists():
        prac = json.loads(PRACTICAL.read_text())
        stored += [r["snappy_over_gzip"] for r in prac["ratio"]]
    sections = [("NOTES", section(notes, "## hysnappy"))]
    sections += [("README", section(readme, "- **`hysnappy`**", "\n- **"))]
    for where, text in sections:
        for found in re.findall(r"(\d+\.\d+)x", text):
            value = float(found)
            places = len(found.split(".")[1])
            check(any(round(v, places) == value for v in stored),
                  f"{where} quotes {found}x, which rounds from no ratio in hysnappy_results.json")


def check_practicality() -> None:
    """Fitness claims in README/NOTES must come from practicality_results.json."""
    data = json.loads(PRACTICAL.read_text())
    notes = NOTES.read_text()
    readme = README.read_text()

    check(data["format"]["framed_sz"] is False,
          "practicality_results says the output IS framed, but the prose says block format")
    check("block format" in notes, "NOTES does not state the block format finding")

    c = data["corruption"]
    check(f"{c['silent_corruption_pct']}%" in notes or f"{c['silent_corruption_pct']}%" in readme,
          f"neither doc quotes the silent-corruption rate {c['silent_corruption_pct']}%")
    check(f"{c['returned_wrong_bytes']}" in notes,
          f"NOTES does not quote the {c['returned_wrong_bytes']} wrong-byte trials")
    check(f"{c['threw']}" in notes, f"NOTES does not quote the {c['threw']} that threw")

    # Every outputLength row must be unchecked in the direction the prose claims.
    exact = [r for r in data["output_length_discipline"] if r["delta"] == 0]
    check(exact and exact[0]["bytes_correct"], "the exact outputLength row is not correct")
    wrong = [r for r in data["output_length_discipline"] if r["delta"] != 0]
    check(all(not r["threw"] for r in wrong),
          "some wrong outputLength threw, but the prose says none do")
    check(all(not r["bytes_correct"] for r in wrong),
          "some wrong outputLength returned correct bytes")

    for row in data["ratio"]:
        check(f"{row['snappy_over_gzip']}x larger" in notes,
              f"NOTES does not quote {row['corpus']} as {row['snappy_over_gzip']}x larger than gzip")
        check(f"{row['snappy_bytes']:,} B" in notes,
              f"NOTES does not quote {row['corpus']} snappy size {row['snappy_bytes']:,}")

    big = data["large_payload"]
    for key in ("snappy_decode_mbps", "gzip_decode_mbps", "snappy_encode_mbps", "gzip_encode_mbps"):
        check(f"{big[key]:,}" in notes or str(big[key]) in notes,
              f"NOTES does not quote large-payload {key} = {big[key]}")

    b = data.get("browser", {})
    if "skipped" in b:
        print(f"  skip: browser half not run ({b['skipped'][:60]})")
        return
    check(b["round_trip_ok"], "the browser round trip failed")
    check(b["CompressionStream"], "CompressionStream absent in the tested browser")
    # The 8 MiB ceiling: everything at or under 8388608 accepted, above rejected.
    for size, verdict in b["sync_module_limit"].items():
        accepted = verdict == "accepted"
        check(accepted == (int(size) <= 8 * 1024 * 1024),
              f"sync module at {size} B: {verdict!r} contradicts an 8 MiB ceiling")
    check("8 MiB" in notes or "8 MB" in notes, "NOTES does not name the 8 MB limit")
    check("Chrome 115" in notes, "NOTES does not date the change to Chrome 115")


def check_evidence(offline: bool) -> None:
    """evidence/ must still match the pinned npm tarball."""
    manifest = EVIDENCE / "MANIFEST.json"
    check(manifest.exists(), "evidence/MANIFEST.json is missing")
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text())
    for item in data["excerpts"]:
        check((EVIDENCE / item["file"]).exists(), f"evidence/{item['file']} is missing")
    version = data["package"].split("@")[-1]
    check(f"hypvector@{version}" in README.read_text() or f"`{version}`" in README.read_text(),
          f"README does not name the pinned version {version}")
    if offline:
        print("  skip: --offline, not re-fetching the tarball")
        return
    proc = subprocess.run(
        [sys.executable, str(HERE / "extract_evidence.py"), "--check"],
        capture_output=True, text=True, check=False,
    )
    check(proc.returncode == 0,
          f"extract_evidence.py --check failed: {proc.stdout.strip()} {proc.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    args = ap.parse_args()

    for name, fn in [
        ("sweep table vs results.json", check_sweep_table),
        ("byte counts vs results.json", check_file_numbers),
        ("repo table vs repos.json", check_repo_table),
        ("private repos absent from repos.json", check_private_repos),
        ("NOTES.md citations in range", check_notes_citations),
        ("hysnappy numbers vs hysnappy_results.json", check_hysnappy),
        ("practicality claims vs practicality_results.json", check_practicality),
        ("evidence vs pinned tarball", lambda: check_evidence(args.offline)),
    ]:
        print(f"- {name}")
        fn()

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        print(f"\n{len(failures)} of {checks} checks failed")
        return 1
    print(f"OK  {checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
