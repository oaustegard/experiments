"""Theory-empirical bridge pipeline runner.

Orchestrates all stages in order:
  1. te_corpus.py   — assemble asymmetric corpus pools
  2. te_embed.py    — SPECTER2 fetch
  3. te_scan.py     — cross-axis cosine + dedup
  4. te_extract.py  — body fetch + slot extraction
  5. te_rerank.py   — slot embedding + cosine rerank
  6. te_judge.py    — binary cheap-LLM-judge
  7. te_translate.py — Claude subagent translations

Each stage is resumable; re-running this script skips completed stages
(detected by presence of their output files).

Usage:
  python te_pipeline.py [--from-stage N] [--dry-run]
  python te_pipeline.py --from-stage 3    # skip corpus + embed, restart from scan
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DATA    = SCRIPTS.parent / "data"

STAGES = [
    (1, "Corpus assembly",   ["python", str(SCRIPTS / "te_corpus.py")],
        DATA / "empirical_corpus.json"),
    (2, "SPECTER2 fetch",    ["python", str(SCRIPTS / "te_embed.py")],
        DATA / "empirical_meta.json"),
    (3, "Cross-axis scan",   ["python", str(SCRIPTS / "te_scan.py")],
        DATA / "te_candidates.json"),
    (4, "Body fetch + slot extract", ["python", str(SCRIPTS / "te_extract.py")],
        DATA / "te_extractions.json"),
    (5, "Slot embed + rerank",       ["python", str(SCRIPTS / "te_rerank.py")],
        DATA / "te_reranked.json"),
    (6, "Cheap-LLM-judge",  ["python", str(SCRIPTS / "te_judge.py")],
        DATA / "te_judged.json"),
    (7, "Claude translations", ["python", str(SCRIPTS / "te_translate.py")],
        DATA / "te_translations.json"),
]


def run_stage(num: int, name: str, cmd: list[str], sentinel: Path, dry_run: bool) -> bool:
    if sentinel.exists():
        print(f"  Stage {num} ({name}): SKIP (output exists: {sentinel.name})")
        return True
    print(f"\n{'='*60}")
    print(f"Stage {num}: {name}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  DRY RUN: would run: {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: stage {num} ({name}) exited {result.returncode}", file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", type=int, default=1,
                    help="Start from this stage number (1-7)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run without executing")
    args = ap.parse_args()

    print("Theory-Empirical Bridge Pipeline")
    print(f"  from-stage={args.from_stage}  dry-run={args.dry_run}")

    for num, name, cmd, sentinel in STAGES:
        if num < args.from_stage:
            print(f"  Stage {num} ({name}): SKIP (--from-stage={args.from_stage})")
            continue
        ok = run_stage(num, name, cmd, sentinel, args.dry_run)
        if not ok:
            sys.exit(1)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
