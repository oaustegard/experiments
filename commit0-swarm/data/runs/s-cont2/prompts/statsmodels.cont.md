You are implementing the Python library `statsmodels` (upstream: statsmodels/statsmodels) from its skeleton.

The checkout at /tmp/jail/c0/work/s-cont2/statsmodels keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `statsmodels`. Make it pass by writing the code under `statsmodels/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `statsmodels/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test statsmodels -x -q` or `/tmp/jail/c0/bin/c0-test statsmodels/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import statsmodels'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.
Never kill a process you did not start yourself (no `pkill`, no `killall`, no `kill` by pattern): other agents and the grader run tests on this machine at the same time.

A previous agent worked on this checkout and stopped; its work is still in place (`git status` and `git diff` show it). Its last full-suite result: not recorded. Continue from there: run the whole suite, then fix what still fails. Keep or rework its code as you judge best.

This is a long job: expect hundreds of tool calls. Do not stop to ask questions, do not stop after the first module, and do not report early. Work until `/tmp/jail/c0/bin/c0-test` on the whole suite passes, or until every remaining failure has had at least two real attempts.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/s-cont2/out/statsmodels.cont.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
