# Goal: complete the remex product description

You are working in the `remex-product-description/` directory of oaustegard/experiments. Read `README.md`, `glossary.md`, `foundations/the-encoding-contract.md` and `persistence/npz-files.md` first. The README defines the purpose, the document template, the method, the structure and the coverage table. The other three are the exemplars: match their depth, tone and structure. Your job is to write every document in the README's structure until the coverage table has no `not started` rows, then run a consistency pass.

## Source of truth

oaustegard/remex is checked out at `/home/user/oaustegard/remex` (clone it there if absent: `git clone https://github.com/oaustegard/remex`). Describe remex 2.0.0's public Python API (`remex/__init__.py`) in its default configuration: installed with `pip install -e .`, NumPy and SciPy present, a C compiler available, no `REMEX_*` environment variables. `remex/gpu.py`, the Mojo port and `bench/` are out of scope.

For each document, read in this order before writing:

1. The methods themselves in `remex/core.py` (or `ivf.py`, `st.py`, `pq_format.py`), and the guards they call: `_resolve_centroids`, `_check_rotation`, `_check_bits`, `_check_mean`, `_check_mode`.
2. `remex/packing.py` and `remex/rotation.py` where the document touches packing or rotation.
3. The tests in `tests/`. `test_coverage_gaps.py`, `test_scalar_mode.py`, `test_centered_mode.py`, `test_v2_fixes.py` and `test_matryoshka.py` are close to executable specifications.
4. remex's `CLAUDE.md` "Key design decisions": it states the invariants and the reasons for them.
5. Run anything ambiguous. A five-line script settles most claims; keep the script, it becomes a verification item.

Do not describe code. Describe what the caller passes, observes and can rely on. Technical detail goes only in `> Technical note:` block quotes.

## Writing rules

- Every call document uses the eight sections in the README. Foundations and cross-cutting documents may drop sections that do not apply but must still say what a refusal or an interrupt does wherever a call is involved.
- The Variants table has five rows (Container mode; Centred; Rotation; Bit width and precision; Environment variables) and the Cancel and interrupt table has seven, worded and ordered exactly as in `persistence/npz-files.md`. Do not add, drop or reorder rows in a single document.
- The cross-cutting paragraphs are, in order: The encoding contract; Matryoshka precision; Determinism across machines; Memory and caches; Thread safety; Other formats; Version compatibility.
- Use the glossary's words. If you are drafting alone and need a term it lacks, add it with a full definition. If you are one of several parallel drafters, do not edit `glossary.md`; list the term and a proposed definition in your report.
- State surprising behavior plainly. Anything that returns normally with a result the caller would not expect is *silently wrong* (glossary) and goes in "Open questions" as a suspected bug.
- Link instead of restating. `foundations/the-encoding-contract.md` owns which parameters are recorded and guarded and the refusal messages.
- Every document ends with "## Open questions and verification" and `Verified against oaustegard/remex commit \`b3a0ed9\``.
- One Mermaid `stateDiagram-v2` per call.

## Things already established (do not re-derive, do not contradict)

- Encoding parameters: `d`, `bits`, `seed`, `rotation`, `normalize`, `scale`, `mean`. `renorm` is not one.
- Guarded, in check order, at the first decode or search: rotation, mode, bits, mean. Not guarded: `seed` and `scale` (not recorded) and `d` (recorded, never compared). A refused call builds no cache.
- Defaults: `bits = 4`, `seed = 42`, rotation `"rht"` for even `d` and `"haar"` for odd, `normalize = True`, `renorm = True`. Supported widths are 1, 2, 3, 4 and 8.
- A container with no rotation record is `"haar"`, never the current default.
- The mode is recorded as whether `norms` is present.
- Containers are not checked when created, saved or loaded; only when read by a quantizer.
- Every writer (`save`, `save_pq`, `save_arrow`, `RemexIndex.save`) opens its destination for writing directly; an overwrite truncates the old file at the first byte (B-02).
- `save` to `.npz` appends `.npz` to a string path that lacks it; `load` does not (B-04).
- Loaded containers carry no decode cache.

## Order of work

1. `foundations/containers.md`.
2. `search/` next, all five documents. Read `search`, `search_adc`, `search_twostage`, `search_batch` and `ivf.py` before writing any of them: they must agree on every variant, and `tests/test_adc_gpu.py` and `tests/test_ivf.py` assert that they do. `search/search.md` owns the decode cache; the others link to it.
3. `encoding/`, `persistence/`, `integration/`, `cross-cutting/`, in parallel once search exists to link to.
4. Consistency pass: `python3 <describing-products>/scripts/check_links.py .` and `check_structure.py . --reference persistence/npz-files.md --exclude RESULTS.md`, then by reading.
5. Add each document's checklist to `verification/` and its automation to `run_pass.py`. Mark the coverage table `drafted`, never `verified`.

## Working rules

- Commit after each document or coherent group: `docs: add {path}` / `docs: revise {path}`. This repo commits straight to `main`, authored as Oskar Austegard.
- Do not modify the remex checkout.
- When code and tests cannot settle a behavior, write what they settle, put the rest in "Open questions", and move on.
- Depth bar: `persistence/npz-files.md` is about 170 lines. Completeness matters more than length: every variant and interrupt cell filled.
