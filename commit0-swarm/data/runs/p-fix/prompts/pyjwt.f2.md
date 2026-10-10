You are implementing the Python library `pyjwt` (upstream: jpadilla/pyjwt) from its skeleton.

The checkout at /tmp/jail/c0/work/p-fix/pyjwt keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `tests`. Make it pass by writing the code under `jwt/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `jwt/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test tests -x -q` or `/tmp/jail/c0/bin/c0-test tests/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import jwt'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

The builders have finished a first pass over every file. You are fixer 2 of 2; all 2 fixers work at the same time in this checkout. Last full-suite result: 0 passed, 258 failed or errored (of 258 graded tests).

Claim work before doing it. Run the suite (`/tmp/jail/c0/bin/c0-test tests -q -p no:randomly 2>&1 | tail -60`), pick a test file with failures, and claim it with `mkdir /tmp/jail/c0/work/p-fix/pyjwt/.claims/<test file name>` (for example `mkdir /tmp/jail/c0/work/p-fix/pyjwt/.claims/test_tables.py`). If mkdir says the directory exists, another fixer has it: pick another. Fix the source so that test file passes, then claim the next one. Files you claimed stay yours; never work on a test file someone else claimed.

You may edit any file under `jwt/`, and so may the other fixers. Make small, targeted edits, and read a file again right before you edit it: it may have changed since you last saw it. If an edit fails because the file changed, re-read and redo it.

Stop when every test file with failures has been claimed and the ones you claimed pass, or every remaining failure in them has had at least two real attempts.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/p-fix/out/pyjwt.f2.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
