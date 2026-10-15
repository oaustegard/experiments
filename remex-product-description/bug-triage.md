# Bug triage

A consolidated list of the defects and inconsistencies the two drafted documents raised, in their bodies and in their "Open questions and verification" sections. Each entry was read from oaustegard/remex at `b3a0ed9` and confirmed against the installed 2.0.0 package by the automated pass in [`verification/`](verification/README.md); every entry carries a **Status** line. None was filed upstream; all six were fixed directly in oaustegard/remex `8d37933` (2026-10-15), each with a Status line saying so. The list exists so the maintainer can decide, item by item, whether to fix, document as intended, or leave.

## Summary

Eleven items were raised across the two documents; they merge to six entries. Two are high. Both high entries break a promise the library makes in its own README: that the codes are fully determined by encoding parameters the library tracks, and, implicitly, that saving an index is safe. The theme of four of the six is the same: a parameter or a file state that the library could check and does not, so the failure is silent or arrives as an unrelated error.

| ID | Title | Severity | Area | Decision needed | Issue |
| --- | --- | --- | --- | --- | --- |
| B-01 | A different seed decodes without error to unrelated vectors | high | encoding contract | product call | — |
| B-02 | An interrupted save destroys the file it was overwriting | high | persistence | fix | — |
| B-03 | Scalar-mode files do not record `scale` | medium | encoding contract | product call | — |
| B-06 | A dimension mismatch is not guarded | medium | encoding contract | fix | — |
| B-04 | `save` and `load` disagree on the `.npz` suffix | low | persistence | fix | — |
| B-05 | Small error-message and validation slips | low | persistence | fix | — |

## High

### B-01: A different seed decodes without error to unrelated vectors

- **Where the caller meets it:** a corpus is encoded and saved by one service and searched by another, or by the same code after its default `seed` changed. Both quantizers have the same `d`, `bits` and rotation.
- **What happens / what was expected:** decode and every search path run normally and return vectors and rankings unrelated to the corpus: mean cosine to the originals −0.017 with the wrong seed against 0.995 with the right one. The caller expects a refusal like the one a rotation mismatch gets, since the library documents the seed as part of the encoding "exactly as" the rotation is.
- **Reproduce:** `c = Quantizer(64, 4, seed=1).encode(X)`; `Quantizer(64, 4, seed=2).decode(c)` returns without error. Checklist item CONTRACT-05.
- **Why (from the code):** `CompressedVectors` has no seed field (`remex/core.py` lines 189–190, `__slots__`), `_npz_fields` (lines 273–291) does not write one, and `_resolve_centroids` (lines 1869–1886) checks rotation, mode, bits and mean only. `_check_rotation`'s own docstring (lines 1089–1097) gives the reason the seed needs the same treatment. `RemexIndex.save` already records the seed and Arrow does when the caller passes it, so the format question is answered elsewhere in the library.
- **Severity:** `high`. Silently returns a wrong answer that looks like an answer, which is the failure the rotation check was added to prevent.
- **Decision needed:** `product call`. Either record the seed in `CompressedVectors` and every format and refuse a mismatch (an absent record meaning "unknown, do not check", so old files keep loading), or document that the caller owns the seed and give `load` a way to say which seed a file expects.
- **Raised by:** [The encoding contract](foundations/the-encoding-contract.md#open-questions-and-verification), [`.npz` files](persistence/npz-files.md#open-questions-and-verification), [`.npz` files: the simple case](persistence/npz-files.md#the-simple-case)
- **Status:** confirmed 2026-10-14 by `run_pass.py`, CONTRACT-05 and NPZ-09.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.

### B-02: An interrupted save destroys the file it was overwriting

- **Where the caller meets it:** re-saving an index to the path it was loaded from (an incremental rebuild, a nightly job), when the process is killed, the disk fills, or an exception is raised partway through the write.
- **What happens / what was expected:** the previous file is truncated the moment the new write begins. A kill partway leaves a zero-byte or partial file and no copy of the old one. The caller expects either the old file or the new one to survive.
- **Reproduce:** save a 300,000 × 384 container at 8 bits (109 MB); in a child process start saving a different container to the same path and SIGKILL it 150 ms in. The file is 0 bytes and `load` raises `EOFError`. Checklist item NPZ-08.
- **Why (from the code):** `CompressedVectors.save` calls `np.savez_compressed(path, ...)` directly (`remex/core.py` lines 263–271). The same pattern is in `save_pq` (`remex/pq_format.py` lines 110 and 150, `open(path, "wb")`), `RemexIndex.save` (`remex/st.py`, `np.savez_compressed`) and `PackedVectors.save_arrow`.
- **Severity:** `high`. Loses work, and the window is the whole duration of the write, which is seconds for a large index.
- **Decision needed:** `fix`. Write to a temporary file in the destination directory and `os.replace` it over the destination after the write completes, in each of the four writers. Note that `np.savez_compressed` appends `.npz` to the temporary name unless it already ends in `.npz` (see B-04).
- **Raised by:** [`.npz` files: the first effect](persistence/npz-files.md#the-first-effect), [`.npz` files: cancel and interrupt](persistence/npz-files.md#cancel-and-interrupt), [`.npz` files: open questions](persistence/npz-files.md#open-questions-and-verification)
- **Status:** confirmed 2026-10-14 for `.npz` by `run_pass.py`, NPZ-08 (109,154,649 bytes before, 0 after). The other three writers were read from code, not exercised.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.

## Medium

### B-03: Scalar-mode files do not record `scale`

- **Where the caller meets it:** scalar mode used for hash keys or joins, with a `scale` chosen to match conditioned inputs, read back by code that uses the default `scale` of 1.0.
- **What happens / what was expected:** decode returns every coordinate multiplied by the ratio of the two scales (exactly 2.0 in the test), with no error. The caller expects a refusal, as for the mode itself.
- **Reproduce:** encode with `Quantizer(64, 4, normalize=False, scale=2.0, rotation="none")`, decode with `scale=1.0`. Checklist item CONTRACT-06.
- **Why (from the code):** `scale` lives only on the quantizer (`remex/core.py` line 917, `self.scale`); `_npz_fields` does not write it and `_check_mode` (lines 1163–1185) compares only whether norms are present. Exact-match uses of scalar codes compare codes, not decoded values, so they are unaffected; anything that decodes is.
- **Severity:** `medium`. Silently wrong, but only in an opt-in mode and only for callers who decode.
- **Decision needed:** `product call`. Record and check `scale` alongside the mode, or document that scalar-mode callers own it.
- **Raised by:** [The encoding contract](foundations/the-encoding-contract.md#what-a-container-does-not-record), [`.npz` files](persistence/npz-files.md#open-questions-and-verification)
- **Status:** confirmed 2026-10-14 by `run_pass.py`, CONTRACT-06.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.

### B-06: A dimension mismatch is not guarded

- **Where the caller meets it:** reading codes with a quantizer built for the wrong embedding model's dimension.
- **What happens / what was expected:** the error depends on the rotation and the path: `ValueError: Expected d=32, got shape (200, 64)` under `"rht"`, a NumPy `matmul` core-dimension error under `"haar"`, an `IndexError` about broadcasting from `search_adc`. Under `rotation="none"` in scalar mode, `decode` returns a 64-wide result from a 32-dimensional quantizer with no error. The caller expects the same "rebuild the Quantizer with d=..." refusal every other recorded parameter gets.
- **Reproduce:** encode with `d = 64`, decode with `d = 32` under each rotation. Checklist item CONTRACT-07.
- **Why (from the code):** the container records `d` (`remex/core.py` line 199) but `_resolve_centroids` (lines 1869–1886) never compares it; the errors that do appear come from shape checks in the rotation (`remex/rotation.py` lines 438–441) or from NumPy.
- **Severity:** `medium`. Usually fails, but with misleading messages, and silently in one mode.
- **Decision needed:** `fix`. Add a `_check_d` beside `_check_bits`.
- **Raised by:** [The encoding contract](foundations/the-encoding-contract.md#what-a-container-does-not-record)
- **Status:** confirmed 2026-10-14 by `run_pass.py`, CONTRACT-07.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.

## Low

### B-04: `save` and `load` disagree on the `.npz` suffix

- **Where the caller meets it:** `c.save("index")` then `CompressedVectors.load("index")`.
- **What happens / what was expected:** `save` writes `index.npz` (and `INDEX.NPZ.npz` for `"INDEX.NPZ"`); `load("index")` raises `FileNotFoundError`. The caller expects the path they saved to to be the path they load from.
- **Reproduce:** checklist items NPZ-02, NPZ-03, NPZ-04.
- **Why (from the code):** NumPy's `savez_compressed` appends `.npz` to string paths that lack it; `np.load` does not. remex passes the path through unchanged in both (`remex/core.py` lines 271 and 302).
- **Severity:** `low`. Fails loudly on the first try.
- **Decision needed:** `fix`. Either have `save` return the path it wrote, or have `load` try `path + ".npz"` when `path` does not exist.
- **Raised by:** [`.npz` files: the call is made](persistence/npz-files.md#the-call-is-made)
- **Status:** confirmed 2026-10-14 by `run_pass.py`, NPZ-02 to NPZ-04.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.

### B-05: Small error-message and validation slips

- **Where the caller meets them:** loading the wrong file, or a container whose fields were edited.
- **What happens / what was expected:**
  - A `.npz` that is not a remex container raises `KeyError: 'd is not a file in the archive'` (NPZ-06).
  - Bytes that are not an archive at all raise `ValueError: This file contains pickled (object) data. If you trust the file you can load it unsafely...`, which invites the caller to enable pickle loading on a file that is merely damaged (NPZ-07).
  - Assigning an unknown name to `container.rotation` is accepted; `save` writes it and `load` then refuses the file (NPZ-16).
  - `Quantizer(64, 9)` says "bits must be 1-4 or 8", and `Quantizer(64, 5)` gives a different message for the same rule (`remex/core.py` lines 881–888).
  - remex's own `CLAUDE.md` repeats one sentence about `renorm` twice in its project overview.
- **Why (from the code):** `CompressedVectors.load` reads `data["d"]` without checking the archive's entries (`remex/core.py` line 303) and calls `np.load` with NumPy's default error behavior; `rotation` is validated in `__init__` (`validate_rotation`, line 202) but is a plain slot afterwards.
- **Severity:** `low`. Each is recoverable and loud, but the pickle message points the caller at an unsafe workaround.
- **Decision needed:** `fix`. Check for the expected entries and raise "not a remex container file"; catch `ValueError`/`BadZipFile`/`EOFError` from `np.load` and raise "file is damaged or not an .npz archive"; validate `rotation` on assignment or in `save`.
- **Raised by:** [`.npz` files: refused at once](persistence/npz-files.md#refused-at-once), [`.npz` files: edge cases](persistence/npz-files.md#edge-cases)
- **Status:** confirmed 2026-10-14 by `run_pass.py`, NPZ-06, NPZ-07, NPZ-16. The `bits` messages and the `CLAUDE.md` repetition were read, not run.
- **Fixed:** oaustegard/remex `8d37933`, covered by `tests/test_integrity.py`; the 2.0.0 checklist items for it now fail against that commit, as they should.
