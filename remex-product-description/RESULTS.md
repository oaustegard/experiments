# describing-products on remex

## Answer

Question: does the `describing-products` skill (oaustegard/claude-skills, adapted from Steve Ruiz's `product-description` gist) work on a Python library, a product with no screen and no end user, and does it find anything?

It works, after one adaptation, and it found six defects in remex 2.0.0, two of them high. The skill's four product kinds (pointer editor, web app, CLI, chat) do not include a library. The CLI column came closest. Its unit of interaction, an invocation, became a *call*, and the seven interrupt families mapped onto library events without strain. The pilot (`save`/`load` to `.npz`) and one foundation came to about 340 lines; a 30-item automated verification pass then confirmed every suspected bug and caught one wrong claim in the pilot. The two high findings are that a quantizer with a different `seed` decodes any container without error into vectors unrelated to the originals (mean cosine −0.017 against 0.995), and that a `save` killed partway through destroys the file it was overwriting (109 MB to 0 bytes).

The method's value came mostly from the fixed interrupt table. Four of the six defects were found by filling its cells ("the process dies", "the target changes underneath", "input in an unexpected form"), not by reading the code for bugs. A full build (21 documents) could change this if the interrupt rows prove too uniform for read-only calls like `search`.

## Findings

Findings 1–3 are fixed in remex `8d37933` (Log, 2026-10-15).

1. **remex: a different seed decodes without error to unrelated vectors.** Mean cosine −0.017 against 0.995; `seed` is neither recorded in a container nor checked, even though `_check_rotation`'s docstring gives the reason a mismatch must be refused (`bug-triage.md` B-01; `run_pass.py` CONTRACT-05).
2. **remex: every writer truncates its destination before writing.** A SIGKILL 150 ms into overwriting a 109 MB file left 0 bytes (`bug-triage.md` B-02; `run_pass.py` NPZ-08). `save_pq`, `save_arrow` and `RemexIndex.save` have the same pattern, read from code.
3. **remex: `scale` is not recorded and `d` is recorded but never compared.** Wrong `scale` decodes off by exactly the ratio; wrong `d` fails with three unrelated messages depending on the rotation, and silently under `rotation="none"` in scalar mode (`bug-triage.md` B-03, B-06).
4. **Skill: no product kind for a library or SDK.** The mapping used here (unit = a call; phases call / refused / first effect / while running / returns; variants = container and quantizer state; interrupts = exception, concurrent thread, reused cache, resource failure, process death, target changed, odd input) is now a fifth column in `product-kinds.md` (`product-kinds.md`).
5. **Skill: a vacuous automated check passed a wrong claim.** The first NPZ-07 accepted any exception; once it checked the message, the item showed that the pilot had misnamed the error. The verification template now requires every automated item to assert the specific expected outcome.
6. **Skill: the checkers assumed a finished repo.** `check_links.py` reported 8 links to planned documents as broken, and `check_structure.py` would have counted this file as a feature document. Both now read the coverage table: links to `not started` documents are reported as planned, and `--exclude` takes extra non-feature files.
7. **Skill: "where the repo goes" assumed a new git repo.** This description lives in a directory of an existing repo; the skill now allows that.

## Method

- Source: oaustegard/remex at `b3a0ed9` (release 2.0.0), installed with `pip install -e .`; Python 3.13.16, NumPy 2.5.2, Linux x86-64.
- Followed `describing-products` 0.1.0 phases 0, 1, 2 (pilot and one foundation), 5 and 6, plus the mechanical half of phase 4. Phase 3 (parallel drafting) was not run; the remaining 17 documents are `not started` in the coverage table, and `goal.md` is ready to drive them.
- Every behavioral claim in the two documents was probed with a short script before or after writing. The probes became `verification/run_pass.py`, one function per checklist ID.
- Checkers: `check_links.py` and `check_structure.py` from the skill, run before and after the skill fixes.

## Log

### 2026-10-14: pilot, foundation, verification, triage

Asked: "Try it on remex" (the skill had been imported and reviewed earlier the same session).

Ran: Phase 0 decisions written into the README; scaffold (README, glossary, goal, AGENTS, CLAUDE); pilot `persistence/npz-files.md`; foundation `foundations/the-encoding-contract.md`; checklist `verification/persistence.md` with 30 items; `run_pass.py`; `bug-triage.md` with six entries.

The probes before drafting found B-01, B-04 and B-05 directly. Filling the interrupt table's "the process dies" row prompted the SIGKILL test that found B-02. Writing the foundation's parameter table, with one column per question ("recorded?", "guarded?"), prompted the `d` probe that found B-06. The glossary's first draft said `d` was guarded; that probe corrected it.

The first pass showed 30/30 passing, but NPZ-07's expectation was empty: it passed on any exception. Once tightened, it showed that a non-archive file raises `ValueError: This file contains pickled (object) data`, not the `BadZipFile` the pilot claimed. The pilot was corrected and the pass rerun: 30/30.

Checkers: `check_structure.py` 0 errors on two feature documents; `check_links.py` 8 "broken" links, all to planned documents. Wall clock about 2 hours including the remex probes; no API cost beyond the session.

### 2026-10-15: the six remex defects fixed

Asked: "Fix the remex bugs".

Ran: oaustegard/remex `8d37933`. Containers record `seed` and scalar `scale`; every read path checks them and `d`; all six writers go through a write-temp-then-rename helper; `load` finds `save`'s `.npz` name; damaged and foreign files raise a `ValueError` naming them. `tests/test_integrity.py` adds 28 tests, 24 of which fail on 2.0.0; the full suite is 604 passed, 1 skipped.

Then `verification/run_pass.py`, unchanged, against the fixed package: 22 pass and exactly the eight items that had confirmed defects on 2.0.0 now fail (NPZ-03, NPZ-06, NPZ-08, NPZ-09, NPZ-16, CONTRACT-05, CONTRACT-06, CONTRACT-07); NPZ-08 now reads 109,154,831 bytes before and after the kill. The checklist describes `b3a0ed9` and stays as written; a revision of the two documents against `8d37933` would flip those rows.

