"""Prompts for the four roles: solo builder, solo continuation, swarm builder, swarm fixer."""
from __future__ import annotations

from common import BIN

CONTEXT = """You are implementing the Python library `{name}` (upstream: {upstream}) from its skeleton.

The checkout at {path} keeps the library's structure: modules, classes, signatures and docstrings. Most function and method bodies have been replaced with `pass`, and some private helpers were deleted outright (look for names the code uses but nothing defines). The library's own test suite is in `{test_dir}`. Make it pass by writing the code under `{src_dir}/`.

The tests and docstrings are your specification, together with what you know of the real library. Only files under `{src_dir}/` are graded; the grader discards every other change, including any edit to tests. There is no network access, so you cannot fetch the original source.

Running code: library code runs only through two commands, used from inside the checkout:
  {bin}/c0-test [TARGET ...]   pytest on targets (default: the whole suite), e.g. `{bin}/c0-test {test_dir} -x -q` or `{bin}/c0-test {example_target} -q`
  {bin}/c0-run CMD ARGS...     anything else, e.g. `{bin}/c0-run python -c 'import {import_name}'`
Plain python, pytest or pip calls are refused. Reading files, editing them, and git are unrestricted. The last line of each run is its exit status."""

PERSIST = """This is a long job: expect hundreds of tool calls. Do not stop to ask questions, do not stop after the first module, and do not report early. Work until `{bin}/c0-test` on the whole suite passes, or until every remaining failure has had at least two real attempts."""

# Inserted as a format value, so single braces; {out} is filled at dispatch by fanout.
FINISH = """When you are done, and as your very last action, write {out} containing {"summary": "<two sentences: what you implemented and the last full-suite result you saw>", "passed": <int from that run or -1>, "failed": <int or -1>}. Then reply with one line."""

SOLO = """{context}

{persist}

Suggested order: run the whole suite once to see what fails, read the tests for the core module, implement the core first (things everything else imports), then work outward, re-running tests as you go.

{finish}
"""

SOLO_CONT = """{context}

A previous agent worked on this checkout and stopped; its work is still in place (`git status` and `git diff` show it). Its last full-suite result: {last}. Continue from there: run the whole suite, then fix what still fails. Keep or rework its code as you judge best.

{persist}

{finish}
"""

SWARM = """{context}

You are builder {i} of {k}. All {k} builders work at the same time in this same checkout, each on its own files. Yours:
{files}

Implement every stub in your files. Do not edit any other file: another builder owns it and is changing it right now. Code in other files may still be stubs while you work, so tests that need it can fail for reasons that are not yours; focus on making the tests that exercise your files pass, and when one fails because of another builder's file, move on. Run targeted tests (a test file or `-k` expression) rather than the whole suite most of the time.

{persist_swarm}

{finish}
"""

PERSIST_SWARM = """This is a long job: expect a hundred or more tool calls. Do not stop to ask questions and do not report until every stub in your files is implemented and the tests that exercise them pass, apart from failures caused by other builders' files."""

FIXER = """{context}

The builders have finished a first pass over every file. You are fixer {i} of {k}; all {k} fixers work at the same time in this checkout, each on its own share of the failing tests. Last full-suite result: {last}.

Your share is the {n} failing tests listed in {shard}, one pytest node id per line (an id may need adjusting if pytest cannot find it; `{bin}/c0-test {test_dir} -q --co` lists the real ones). Make them pass by fixing the source. Run them by passing the ids to c0-test directly, a few at a time (`{bin}/c0-test <id> <id> -q 2>&1 | tail -40`; a `$(cat ...)` substitution is refused by the sandbox).

You may edit any file under `{src_dir}/`, and so may the other fixers. Make small, targeted edits, and read a file again right before you edit it: it may have changed since you last saw it. If an edit fails because the file changed, re-read and redo it. Do not rewrite a whole file another fixer may be working in. Before you finish, run the whole suite once to check you broke nothing outside your share.

Stop when every test in your share passes, or every remaining one has had at least two real attempts.

{finish}
"""


def context(task: dict, path) -> str:
    src = task["src_dir"]
    test_dir = task["test"]["test_dir"].rstrip("/") or "."
    import_name = src.split("/")[-1]
    return CONTEXT.format(name=task["name"], upstream=task["original_repo"], path=path,
                          test_dir=test_dir, src_dir=src, bin=BIN, import_name=import_name,
                          example_target=f"{test_dir}/<test_file>.py".lstrip("./"))


def solo(task, path) -> str:
    return SOLO.format(context=context(task, path), persist=PERSIST.format(bin=BIN), finish=FINISH)


def solo_cont(task, path, last: str) -> str:
    return SOLO_CONT.format(context=context(task, path), persist=PERSIST.format(bin=BIN),
                            last=last, finish=FINISH)


def swarm(task, path, i: int, k: int, files: list[str]) -> str:
    return SWARM.format(context=context(task, path), i=i, k=k,
                        files="\n".join(f"  - {f}" for f in files),
                        persist_swarm=PERSIST_SWARM, finish=FINISH)


def fixer(task, path, i: int, k: int, last: str, shard, n: int) -> str:
    return FIXER.format(context=context(task, path), i=i, k=k, last=last, bin=BIN, shard=shard, n=n,
                        test_dir=task["test"]["test_dir"].rstrip("/") or ".",
                        src_dir=task["src_dir"], finish=FINISH)
