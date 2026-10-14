# Verification

The documents were written from the code and the tests. This directory is the protocol for checking them against the installed library, one observable claim at a time.

## What is here

| File | Covers |
| --- | --- |
| [persistence.md](persistence.md) | `foundations/the-encoding-contract.md` and `persistence/npz-files.md` |
| [run_pass.py](run_pass.py) | the automated part of a pass, one function per checklist ID |
| `results.json` | the last pass's results, written by `run_pass.py --json` |

Each checklist row is an item with a stable ID (`NPZ-07`, `CONTRACT-05`), a priority, the condition it needs, the claim linked to its section, setup, steps, the expected result, and a Result column. Priorities: **P1** is an established fact many documents depend on, or a suspected bug; **P2** an ordinary claim; **P3** a number or a size.

## How to run a pass

1. In the remex checkout, confirm the commit with `git rev-parse --short HEAD`. Every document says `Verified against oaustegard/remex commit b3a0ed9`; if the commit differs, a failure may be drift rather than a defect.
2. Install it into a fresh environment: `pip install -e .`. Leave every `REMEX_*` variable unset.
3. Run `python3 verification/run_pass.py --json verification/results.json`. It prints `ID result note` per item and exits 1 if any item is not `pass`.
4. Copy each result into the checklist's Result column, with the note for anything other than a clean pass.
5. For items marked (suspected bug), `pass` confirms the defect is present as the document describes it. File it in [`bug-triage.md`](../bug-triage.md) with a Status line naming the ID.
6. A `fail` means the product did not do what the document says. Decide whether the document or the product is wrong and say which in the Status line; NPZ-07 in the first pass was the document.
7. When every P1 and P2 item for a document has passed or been filed, it may move from `drafted` to `verified` in the [coverage table](../README.md#coverage). The automated pass alone does not justify that move; see below.

## Devices and conditions

- **py**: a plain Python process.
- **proc**: the item starts a child process and kills it with SIGKILL partway through a write. POSIX only. The 150 ms delay assumes a 109 MB file takes longer than that to write; on a much faster disk raise the corpus size, or the child finishes and the item reports a complete file.
- **env**: the item runs two child processes, one with the variable set.

## Results so far

One pass, 2026-10-14, remex 2.0.0 at `b3a0ed9`, Python 3.13, NumPy 2.5.2, Linux x86-64, run by `run_pass.py`. 30 items, 30 pass after one correction. The first run reported NPZ-07 as passing vacuously (the item accepted any exception); tightened to check each message, it showed the pilot was wrong about which error an unreadable file raises, and the pilot was corrected.

Every suspected-bug item confirmed its defect: NPZ-03, NPZ-06, NPZ-08 and NPZ-16, and CONTRACT-05, CONTRACT-06 and CONTRACT-07. They map to B-01 to B-06 in [`bug-triage.md`](../bug-triage.md).

What the pass did not cover: disk-full and concurrent-mutation behavior, Windows (where an open file cannot be truncated by another process and NPZ-08 would behave differently), and anything about speed. No document is marked `verified`, because nobody has read the documents against the results by hand.
