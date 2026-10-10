You are implementing the Python library `parsel` (upstream: scrapy/parsel) from its skeleton.

The checkout at /tmp/jail/c0/work/f-fix/parsel keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `tests`. Make it pass by writing the code under `parsel/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `parsel/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  /tmp/jail/c0/bin/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `/tmp/jail/c0/bin/c0-test tests -x -q` or `/tmp/jail/c0/bin/c0-test tests/<test_file>.py -q`
  /tmp/jail/c0/bin/c0-run CMD ARGS...     anything else, e.g. `/tmp/jail/c0/bin/c0-run python -c 'import parsel'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status.

The builders have finished a first pass over every file. You are fixer 1 of 1; all 1 fixers work at the same time in this checkout, each on its own share of the failing tests. Last full-suite result: 203 passed, 2 failed or errored (of 205 graded tests).

Your share is the 2 failing tests listed in /tmp/jail/c0/work/f-fix/parsel/.c0-shard-1.txt, one pytest node id per line (an id may need adjusting if pytest cannot find it; `/tmp/jail/c0/bin/c0-test tests -q --co` lists the real ones). Make them pass by fixing the source. Run them with `/tmp/jail/c0/bin/c0-test $(cat /tmp/jail/c0/work/f-fix/parsel/.c0-shard-1.txt) -q 2>&1 | tail -40`, or a few at a time.

You may edit any file under `parsel/`, and so may the other fixers. Make small, targeted edits, and read a file again right before you edit it: it may have changed since you last saw it. If an edit fails because the file changed, re-read and redo it. Do not rewrite a whole file another fixer may be working in. Before you finish, run the whole suite once to check you broke nothing outside your share.

Stop when every test in your share passes, or every remaining one has had at least two real attempts.

When you are done, and as your very last action, write /home/user/experiments/commit0-swarm/data/runs/f-fix/out/parsel.f1.json containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line.
