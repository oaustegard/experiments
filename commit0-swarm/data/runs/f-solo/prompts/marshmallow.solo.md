You are implementing the Python library `marshmallow` (upstream: marshmallow-code/marshmallow) from its skeleton.

The checkout at /tmp/jail/c0/work/f-solo/marshmallow keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `tests`. Make it pass by writing the code under `src/marshmallow/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `src/marshmallow/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test tests -x -q` or `/tmp/jail/c0/bin/c0-test tests/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import marshmallow'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

This is a long job: expect hundreds of tool calls. Do not stop to ask questions, do not stop after the first module, and do not report early. Work until `/tmp/jail/c0/bin/c0-test` on the whole suite passes, or until every remaining failure has had at least two real attempts.

Suggested order: run the whole suite once to see what fails, read the tests for the core module, implement the core first (things everything else imports), then work outward, re-running tests as you go.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/f-solo/out/marshmallow.solo.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
