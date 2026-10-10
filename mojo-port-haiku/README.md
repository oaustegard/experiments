# mojo-port-haiku

Can Haiku 5.5 port established, well-tested, slow pure-Python libraries to
Mojo 1.1, as compiled extension modules behind the original Python API, and
do the ports pay off in speed? Follow-up to `commit0-swarm`, where Haiku 5.5
rebuilt whole Python libraries from skeletons. Results: `RESULTS.md`.

Targets: `difflib.SequenceMatcher` (stdlib), `snowballstemmer` 3.1.1
(english, german, russian, french), `yake` 0.7.3.

Mojo is not a Python superset (Modular deprioritized that in 2025-03), so a
port keeps the Python package and moves the algorithm into a Mojo
extension module built with `mojo build --emit shared-lib`.

## Layout

- `kit/` — what each agent is given: the task, Mojo 1.1 notes with measured
  boundary costs, and (via `harness/new_run.py`) a starter package that builds.
- `oracle/` — correctness: Snowball's published vocabularies plus generated
  words against the pure-Python stemmers; a differential fuzzer plus CPython's
  `test_difflib` (v3.13.16, run in claude-workspace's jail); yake against
  itself on frozen markdown docs under six configs. Each has a dev split the
  agent sees and a held-out test split used for grading.
- `bench/bench.py` — wall clock from Python, boundary and shim included.
- `harness/grade.py` — rebuilds the agent's sources in a clean copy, runs the
  held-out oracle (and `test_difflib` in the jail), optionally benchmarks.
- `data/` — Snowball `voc.txt` (snowball-data at `SOURCE_COMMIT`), the yake
  corpus (this repo's own RESULTS/README files, frozen 2026-10-10).

## Reproduce

```bash
uv pip install --system --break-system-packages modular==26.6.0 --no-deps
uv pip install --system --break-system-packages mojo==1.1.0 max==26.6.0
pip install --break-system-packages snowballstemmer==3.1.1 yake==0.7.3 PyStemmer
git clone https://github.com/modular/skills /tmp/modskills/skills && git -C /tmp/modskills/skills checkout b9b3a8e
python3 harness/new_run.py r1 snowball difflib yake
# dispatch one Haiku 5.5 agent per runs/r1/<target>/ (prompt in harness/PROMPT.md)
python3 harness/grade.py --target snowball --workdir runs/r1/snowball --bench
```
